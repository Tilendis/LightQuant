"""抓取 50 只标的的财务指标，落库到 fundamentals。

为什么需要这一步：CSMD50 原数据集只有 price(OHLCV) 和 news，**没有任何财务字段**。
基本面视角此前只能拿 PE/PB 分位（那是"贵不贵"），缺"公司好不好"的原材料——
ROE、毛利率、增速、负债率、现金流一个都没有。本脚本把这一层补上。

时点红线：财务数据按报告期发布，但报表不是在报告期结束当天就公开的。
用 2024Q2 的数据判断 2024-06-30 那天就是未来函数。所以这里给每一期算一个
法定披露截止日写进 announce_date，服务层只取 announce_date <= 交易日的最近一期。

用法：
    python scripts/fetch_fundamentals.py             # 增量，已有的跳过
    python scripts/fetch_fundamentals.py --refresh   # 重抓
    python scripts/fetch_fundamentals.py --limit 3   # 调试
"""

import argparse
import os
import sys
import time

import akshare as ak
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "scripts"))
from app import db
from fetch_valuations import load_name_code

# akshare 列名 -> 库字段。取不到就留 NULL，不瞎填。
COLMAP = {
    "净资产收益率(%)": "roe",
    "主营业务收入增长率(%)": "revenue_yoy",
    "净利润增长率(%)": "profit_yoy",
    "资产负债率(%)": "debt_ratio",
    "每股经营性现金流(元)": "ocf_ps",
    "流动比率": "current_ratio",
    "存货周转率(次)": "inventory_turn",
}


def _f(v):
    """转 float，NaN/空/字符串异常一律 None。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:      # NaN
        return None
    return x


def gross_margin_of(row, cols):
    """毛利率两路兜底。

    新浪源的「销售毛利率(%)」只有部分期有值（实测万华化学 107 期里仅 76 期），
    而「主营业务成本率(%)」每期都有。毛利率 = 100 - 成本率，两者等价，
    所以先取前者，缺失再用后者反算，避免整列全空。
    """
    direct = _f(row["销售毛利率(%)"]) if "销售毛利率(%)" in cols else None
    if direct is not None:
        return direct
    cost = _f(row["主营业务成本率(%)"]) if "主营业务成本率(%)" in cols else None
    if cost is not None:
        return 100.0 - cost
    return None

# 报告期 -> (period 标签, 法定披露截止日)
def period_of(date_str):
    """把 '2024-06-30' 映射成 ('2024H1', '2024-08-31')。非季末返回 None。"""
    d = str(date_str)[:10]
    if len(d) != 10:
        return None, None
    y, m, day = d[:4], d[5:7], d[8:10]
    if m == "03" and day == "31":
        return "%sQ1" % y, "%s-04-30" % y
    if m == "06" and day == "30":
        return "%sH1" % y, "%s-08-31" % y
    if m == "09" and day == "30":
        return "%sQ3" % y, "%s-10-31" % y
    if m == "12" and day == "31":
        return "%sFY" % y, "%d-04-30" % (int(y) + 1)
    return None, None


def fetch_one(code):
    """返回 [{period, announce_date, roe, ...}]。"""
    df = ak.stock_financial_analysis_indicator(symbol=code)
    if df is None or df.empty:
        return []
    if "日期" not in df.columns:
        return []

    rows = []
    for _, r in df.iterrows():
        period, announce = period_of(r["日期"])
        if period is None:
            continue
        if period[:4] < "2020" or period[:4] > "2024":
            continue
        rec = {"period": period, "announce_date": announce}
        for src, dst in COLMAP.items():
            rec[dst] = _f(r[src]) if src in df.columns else None
        rec["gross_margin"] = gross_margin_of(r, df.columns)
        # 至少要有 roe 或毛利率，全空的记录没意义
        if rec.get("roe") is None and rec.get("gross_margin") is None:
            continue
        rows.append(rec)
    # 同一 period 只留一行（接口偶有重复）
    uniq = {}
    for rec in rows:
        uniq[rec["period"]] = rec
    return [uniq[k] for k in sorted(uniq)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    conn = db.connect()
    db.init_db(conn=conn)        # 触发 _migrate，补齐 fundamentals 新列
    mapping = load_name_code()

    symbols = [r["symbol"] for r in conn.execute(
        "SELECT symbol FROM stocks ORDER BY symbol").fetchall()]
    if not symbols:
        print("stocks 为空，先跑 fetch_valuations.py")
        return 1
    if args.limit:
        symbols = symbols[:args.limit]
    print("目标 %d 只" % len(symbols))

    failures, done, skipped = [], 0, 0
    for i, name in enumerate(symbols, 1):
        code = mapping.get(name)
        if not code:
            failures.append((name, "no_code"))
            print("[%d/%d] %s 无代码映射" % (i, len(symbols), name))
            continue

        have = conn.execute(
            "SELECT COUNT(*) FROM fundamentals WHERE symbol = ?", (name,)).fetchone()[0]
        if have and not args.refresh:
            skipped += 1
            print("[%d/%d] %s 已有 %d 期，跳过" % (i, len(symbols), name, have))
            continue

        try:
            rows = fetch_one(code)
        except Exception as e:
            failures.append((name, "%s: %s" % (type(e).__name__, str(e)[:70])))
            print("[%d/%d] %s 失败 %s" % (i, len(symbols), name, type(e).__name__))
            time.sleep(1.2)
            continue

        if not rows:
            failures.append((name, "empty"))
            print("[%d/%d] %s 接口无有效期" % (i, len(symbols), name))
            continue

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        conn.executemany(
            """INSERT INTO fundamentals
                   (symbol, period, announce_date, roe, gross_margin, revenue_yoy,
                    profit_yoy, debt_ratio, ocf_ps, current_ratio, inventory_turn,
                    source, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?, 'akshare', ?)
               ON CONFLICT (symbol, period) DO UPDATE SET
                   announce_date = excluded.announce_date,
                   roe = excluded.roe,
                   gross_margin = excluded.gross_margin,
                   revenue_yoy = excluded.revenue_yoy,
                   profit_yoy = excluded.profit_yoy,
                   debt_ratio = excluded.debt_ratio,
                   ocf_ps = excluded.ocf_ps,
                   current_ratio = excluded.current_ratio,
                   inventory_turn = excluded.inventory_turn,
                   source = excluded.source,
                   fetched_at = excluded.fetched_at""",
            [(name, r["period"], r["announce_date"], r["roe"], r["gross_margin"],
              r["revenue_yoy"], r["profit_yoy"], r["debt_ratio"], r["ocf_ps"],
              r["current_ratio"], r["inventory_turn"], now) for r in rows],
        )
        conn.commit()
        done += 1
        print("[%d/%d] %s(%s) +%d 期" % (i, len(symbols), name, code, len(rows)))
        time.sleep(0.35)

    total = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
    cov = conn.execute("SELECT COUNT(DISTINCT symbol) FROM fundamentals").fetchone()[0]
    print("\n== done %d / skipped %d / failed %d ==" % (done, skipped, len(failures)))
    print("fundamentals 共 %d 期，覆盖 %d/%d 只" % (total, cov, len(symbols)))

    # 覆盖检查：2024-08-08 ~ 12-31 每天能用到最近一期是哪期
    for probe in ("2024-08-15", "2024-09-20", "2024-11-15", "2024-12-20"):
        r = conn.execute(
            """SELECT period, COUNT(DISTINCT symbol) n FROM fundamentals
               WHERE announce_date <= ? GROUP BY period ORDER BY period DESC""",
            (probe,)).fetchall()
        print("%s 可见期: %s" % (probe, [(x["period"], x["n"]) for x in r]))
    if failures:
        print("失败清单:")
        for n, r in failures:
            print("  ", n, r)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

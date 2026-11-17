"""一次性抓取 50 只标的的历史估值（PE_TTM / PB），落库到 app/data/lightquant.db。

为什么要做这一步：CSMD50 只有价格和新闻，没有财务与估值。基本面视角没有输入
就只能让大模型凭常识编，答辩一问"你的基本面数据是什么"就塌。

时点红线：只取 2021-01-01 ~ 2024-12-31 的历史值，抓到的本身就是当天可见的数字；
服务层再按 "取当天之前最近一条" 对齐到交易日，绝不做插值或向后填充。

用法：
    python scripts/fetch_valuations.py            # 增量，已有的跳过
    python scripts/fetch_valuations.py --refresh  # 重抓
"""

import argparse
import glob
import json
import os
import sys
import time

import akshare as ak

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from app import db

DATE_MIN = "2021-01-01"
DATE_MAX = "2024-12-31"
CACHE = os.path.join(BASE, "app", "data", "name_code.json")
# 数据集用的是当年简称，全市场列表已被更名覆盖，这里显式补上（603501 现名"豪威集团"）
EXTRA_CODES = {"韦尔股份": "603501"}


def load_name_code():
    """中文简称 -> 6 位交易所代码。缓存到本地，避免每次抓都重新拉全市场列表。

    注意：akshare 1.20 的 stock_info_a_code_name() 内部会调北交所接口，
    该接口时常返回非 JSON 内容导致 JSONDecodeError，所以这里做了降级：
    先试它，失败就退回新浪全市场快照（代码形如 sh600309，需要去掉前缀）。
    """
    if os.path.exists(CACHE):
        with open(CACHE, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        mapping.update(EXTRA_CODES)      # 缓存里可能没有后补的旧简称
        return mapping

    mapping = {}
    try:
        base = ak.stock_info_a_code_name()
        mapping = {str(r["name"]): str(r["code"]).zfill(6) for _, r in base.iterrows()}
        print("name_code: 用 stock_info_a_code_name")
    except Exception as e:
        print("name_code: stock_info_a_code_name 失败(%s)，改用新浪快照" % type(e).__name__)

    if not mapping:
        base = ak.stock_zh_a_spot()
        for _, r in base.iterrows():
            raw = str(r["代码"])                 # sh600309 / sz000001 / bj920000
            code = raw[2:] if raw[:2].isalpha() else raw
            if len(code) == 6 and code.isdigit():
                mapping[str(r["名称"])] = code
        print("name_code: 新浪快照 %d 条" % len(mapping))

    mapping.update(EXTRA_CODES)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1)
    return mapping


def fetch_one(code):
    """返回 {date: {pe_ttm, pb}}，两个指标按日期合并。"""
    out = {}
    for indicator, key in (("市盈率(TTM)", "pe_ttm"), ("市净率", "pb")):
        df = ak.stock_zh_valuation_baidu(symbol=code, indicator=indicator, period="全部")
        for _, row in df.iterrows():
            d = str(row["date"])[:10]
            if not (DATE_MIN <= d <= DATE_MAX):
                continue
            try:
                val = float(row["value"])
            except (TypeError, ValueError):
                continue
            out.setdefault(d, {})[key] = val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只抓前 N 只，调试用")
    args = ap.parse_args()

    conn = db.connect()
    db.init_db(conn=conn)          # 幂等，确保 valuations 表存在
    mapping = load_name_code()

    files = sorted(glob.glob(os.path.join(BASE, "dataset", "CSMD50", "test", "price", "*.csv")))
    names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    if args.limit:
        names = names[:args.limit]

    failures, empty, done, skipped = [], [], 0, 0

    for i, name in enumerate(names, 1):
        code = mapping.get(name)
        if not code:
            failures.append((name, "no_code"))
            print("[%d/%d] %s 无代码映射，跳过" % (i, len(names), name))
            continue

        have = conn.execute(
            "SELECT COUNT(*) FROM valuations WHERE symbol = ?", (name,)
        ).fetchone()[0]
        if have and not args.refresh:
            skipped += 1
            print("[%d/%d] %s 已有 %d 条，跳过" % (i, len(names), name, have))
            continue

        try:
            data = fetch_one(code)
        except Exception as e:
            failures.append((name, "%s: %s" % (type(e).__name__, str(e)[:80])))
            print("[%d/%d] %s 抓取失败 %s" % (i, len(names), name, type(e).__name__))
            time.sleep(1.0)
            continue

        if not data:
            empty.append(name)
            print("[%d/%d] %s 接口无数据" % (i, len(names), name))
            continue

        conn.executemany(
            """INSERT INTO valuations (symbol, date, pe_ttm, pb, source)
               VALUES (?, ?, ?, ?, 'baidu')
               ON CONFLICT (symbol, date) DO UPDATE SET
                   pe_ttm = excluded.pe_ttm, pb = excluded.pb, source = excluded.source""",
            [(name, d, v.get("pe_ttm"), v.get("pb")) for d, v in sorted(data.items())],
        )
        conn.execute(
            "INSERT INTO stocks (symbol, name, code) VALUES (?, ?, ?) "
            "ON CONFLICT (symbol) DO UPDATE SET code = excluded.code",
            (name, name, code),
        )
        conn.commit()
        done += 1
        print("[%d/%d] %s (%s) 写入 %d 条" % (i, len(names), name, code, len(data)))
        time.sleep(0.35)

    total = conn.execute("SELECT COUNT(*) FROM valuations").fetchone()[0]
    covered = conn.execute("SELECT COUNT(DISTINCT symbol) FROM valuations").fetchone()[0]
    print("\n== done: %d, skipped: %d, failed: %d, empty: %d ==" % (
        done, skipped, len(failures), len(empty)))
    print("valuations 总行数 %d，覆盖 %d/%d 只" % (total, covered, len(names)))

    # 2024 下半年（测试期）覆盖率——这是演示真正要用的区间
    seg = conn.execute(
        "SELECT symbol, COUNT(*) c FROM valuations WHERE date >= '2024-07-01' AND date <= ? "
        "GROUP BY symbol ORDER BY c", (DATE_MAX,)).fetchall()
    if seg:
        counts = [r["c"] for r in seg]
        print("测试期(2024H2)覆盖 %d 只，每条数 min=%d 中位=%d max=%d" % (
            len(counts), min(counts), sorted(counts)[len(counts) // 2], max(counts)))
    if failures:
        print("失败清单:")
        for n, r in failures:
            print("  ", n, r)
    conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

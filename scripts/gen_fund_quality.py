"""基本面视角（二）：财务质量截面分 -> judgements(view='fund', prompt_version='fund-quality-v1')

为什么要做这一版：
    fund-rule-v1 只用 pe_ttm / pb 的历史分位（估值），在 val 段权重被标定为 0。
    三组实验（加权 / 选池 / 网格）方向一致：估值分位在日频二分类上没有增量。
    但那只能说"估值分位不行"，不能说"基本面不行"——fundamentals 表里已有
    ROE、营收/利润同比、毛利率、负债率 984 行，此前完全没进过信号。
    本脚本把这些字段合成一个截面质量分，用**与估值版完全相同的池筛选方法**去检验，
    这样能把"方法问题"和"信号问题"分开。

防穿越口径（与 gen_fundamental.py 一致，schema.sql 约束）：
    每个交易日只能用 announce_date <= 当日 的报告；同一天之内不允许使用未披露公司的数据。
    质量分在同 period 内做截面 z-score，且只用当日已披露的公司参与计算。

用法：
    python scripts/gen_fund_quality.py --start 2024-08-08 --end 2024-12-31
    python scripts/gen_fund_quality.py --start 2024-03-15 --end 2024-08-07 ^
        --price_dir dataset/CSMD50_feat/val/price --reset
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

PROMPT_VERSION = "fund-quality-v1"
PRICE_DIR = os.path.join(ROOT, "dataset", "CSMD50", "test", "price")

# 质量分构成：盈利、成长、护城河、杠杆
W_ROE = 1.0
W_REV_YOY = 0.7
W_PROF_YOY = 0.7
W_DEBT = -0.6

# 同比口径的极端值（亏损转盈会让 yoy 冲到几千 %），按分位截断而不是直接丢
YOY_LO, YOY_HI = -100.0, 200.0


def trading_dates(start, end, price_dir=None):
    dates = set()
    for f in glob.glob(os.path.join(price_dir or PRICE_DIR, "*.csv")):
        try:
            df = pd.read_csv(f, usecols=["Date"])
        except Exception:
            continue
        for d in df["Date"].astype(str):
            if start <= d <= end:
                dates.add(d)
    return sorted(dates)


def load_fundamentals(conn):
    rows = conn.execute(
        """SELECT symbol, period, announce_date, roe, revenue_yoy, profit_yoy,
                  gross_margin, debt_ratio
           FROM fundamentals
           WHERE announce_date IS NOT NULL ORDER BY symbol, announce_date"""
    ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return df
    df["announce_date"] = df["announce_date"].astype(str)
    df["revenue_yoy"] = df["revenue_yoy"].clip(YOY_LO, YOY_HI)
    df["profit_yoy"] = df["profit_yoy"].clip(YOY_LO, YOY_HI)
    return df


def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 3:
        return pd.Series(np.nan, index=s.index)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def quality_scores(frame):
    """frame 是同一天已披露公司的快照（可能混多个 period）。

    只有在同一个 period 内部，ROE / 负债率才是可比的（累计口径不同），
    所以按 period 分组做截面 z，缺组（只有 1~2 家）的置 0 而不是用全样本凑。
    """
    out = pd.Series(np.nan, index=frame.index)
    for _, g in frame.groupby("period"):
        if len(g) < 5:
            out.loc[g.index] = 0.0
            continue
        parts, weights = [], []
        for col, w in (("roe", W_ROE), ("revenue_yoy", W_REV_YOY),
                       ("profit_yoy", W_PROF_YOY), ("debt_ratio", W_DEBT)):
            z = zscore(g[col]).fillna(0.0)
            parts.append(z * w)
            weights.append(abs(w))
        raw = sum(parts) / sum(weights)
        out.loc[g.index] = np.tanh(raw / 2.0)     # 压到 (-1, 1)，与估值分同量级
    return out


def score_to_signal(score, th=0.2):
    if score >= th:
        return "BUY"
    if score <= -th:
        return "SELL"
    return "HOLD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-08-08")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--price_dir", default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--reset", action="store_true", help="先删掉本版本旧记录")
    args = ap.parse_args()

    conn = db.connect(args.db)
    db.init_db(conn=conn)

    if args.reset:
        n = conn.execute("DELETE FROM judgements WHERE prompt_version = ?",
                         (PROMPT_VERSION,)).rowcount
        conn.commit()
        print("清理旧记录 %d 条" % n)

    dates = trading_dates(args.start, args.end, args.price_dir)
    print("交易日 %d 个: %s ~ %s" % (len(dates), dates[0] if dates else "-",
                                    dates[-1] if dates else "-"))

    fund = load_fundamentals(conn)
    if fund.empty:
        print("fundamentals 为空，先跑 scripts/fetch_fundamentals.py")
        return
    symbols = sorted(fund["symbol"].unique())
    print("有财报数据的标的: %d，报告期 %d 个" % (len(symbols), fund["period"].nunique()))

    written = 0
    for d in dates:
        # 当日可见集：announce_date <= d；每家公司保留 period 最新的一条
        vis = fund[fund["announce_date"] <= d]
        if vis.empty:
            continue
        vis = (vis.sort_values(["symbol", "announce_date", "period"])
                  .groupby("symbol", as_index=False).last())
        vis = vis.set_index("symbol")
        sc = quality_scores(vis)
        for sym, s in sc.items():
            if not np.isfinite(s):
                continue
            sig = score_to_signal(float(s))
            roe = vis.at[sym, "roe"]
            reason = ("财务质量分 %+.2f（ROE %s、营收同比 %s、利润同比 %s、负债率 %s，" % (
                s,
                "%.1f%%" % roe if pd.notna(roe) else "NA",
                "%.1f%%" % vis.at[sym, "revenue_yoy"] if pd.notna(vis.at[sym, "revenue_yoy"]) else "NA",
                "%.1f%%" % vis.at[sym, "profit_yoy"] if pd.notna(vis.at[sym, "profit_yoy"]) else "NA",
                "%.1f%%" % vis.at[sym, "debt_ratio"] if pd.notna(vis.at[sym, "debt_ratio"]) else "NA",
            ) + "同报告期截面标准化），判定%s" % ("偏优" if sig == "BUY" else ("偏弱" if sig == "SELL" else "中性")))
            db.upsert_judgement(
                conn, sym, d, "fund", sig, float(s), reason=reason,
                provider="local", model="fund_quality_composite",
                temperature=None, prompt_version=PROMPT_VERSION,
                source="offline_rule")
            written += 1

    conn.commit()
    conn.close()
    print("写入 %d 条（%s）" % (written, PROMPT_VERSION))
    print("口径说明：全部本地规则计算，source=offline_rule；不是大模型研判，")
    print("         答辩时不能把这一版说成 LLM 生成的财务分析。")


if __name__ == "__main__":
    main()

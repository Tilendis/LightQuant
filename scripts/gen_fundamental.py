"""基本面视角：估值分位 -> 分数 -> judgements(view='fund')

为什么先做规则版而不是直接调 LLM：
- 估值判断的核心是"当前 PE/PB 在过去三年处于什么位置"，这是可精确计算的事实
- 让 LLM 重算一遍分位没有意义，它只会猜；LLM 适合做的是"解释为什么便宜/贵"
- 规则分可复现、可消融、断网可跑，符合普惠化要求

红线（与 schema.sql 一致）：对齐交易日时必须取"当天之前最近一条"估值，不得插值或回填，
否则 2024-09 的判断会用到 2024-10 才公布的估值。

用法：
    python scripts/gen_fundamental.py --start 2024-08-08 --end 2024-12-31
"""

import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

PROMPT_VERSION = "fund-rule-v1"
PRICE_DIR = os.path.join(ROOT, "dataset", "CSMD50", "test", "price")

def load_valuations(conn, symbol):
    rows = conn.execute(
        "SELECT date, pe_ttm, pb FROM valuations WHERE symbol = ? ORDER BY date",
        (symbol,)).fetchall()
    if not rows:
        return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
    return pd.DataFrame([dict(r) for r in rows])


def trading_dates(start, end, price_dir=None):
    """从价格文件取真实交易日（比自己造日期可靠）。

    price_dir 可指向 val / train 段，用于跨窗口验证。不指定时用 test 段。
    """
    dates = set()
    folder = price_dir or PRICE_DIR
    for f in glob.glob(os.path.join(folder, "*.csv")):
        try:
            df = pd.read_csv(f, usecols=["Date"])
        except Exception:
            continue
        for d in df["Date"].astype(str):
            if start <= d <= end:
                dates.add(d)
    return sorted(dates)


def pct_rank(series, value):
    """value 在 series 中的分位（0-1）。只用 <= value 对应的历史，避免未来信息。"""
    s = series.dropna()
    if len(s) < 8 or value is None or pd.isna(value):
        return None
    return float((s <= value).mean())


def fund_score(pe_pct, pb_pct):
    """低估给正分、高估给负分。分位 0.5 附近趋于 0，避免把"合理估值"当成信号。"""
    parts, weights = [], []
    if pe_pct is not None:
        parts.append((0.5 - pe_pct) * 2.0)
        weights.append(0.6)
    if pb_pct is not None:
        parts.append((0.5 - pb_pct) * 2.0)
        weights.append(0.4)
    if not parts:
        return None
    score = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
    return max(-1.0, min(1.0, score))


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
    ap.add_argument("--price_dir", default=None,
                    help="交易日来源目录，跨窗口验证时指向 val/train 段")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    conn = db.connect(args.db)
    db.init_db(conn=conn)

    dates = trading_dates(args.start, args.end, args.price_dir)
    print("交易日 %d 个: %s ~ %s" % (len(dates), dates[0] if dates else "-", dates[-1] if dates else "-"))

    symbols = [r["symbol"] for r in conn.execute(
        "SELECT DISTINCT symbol FROM valuations ORDER BY symbol").fetchall()]
    print("有估值数据的标的:", len(symbols))

    written, no_data = 0, 0
    for sym in symbols:
        v = load_valuations(conn, sym)
        if v.empty:
            no_data += 1
            continue
        v["date"] = v["date"].astype(str)
        for d in dates:
            hist = v[v["date"] <= d]
            if hist.empty:
                continue
            cur = hist.iloc[-1]          # 当天之前最近一条（ffill，不插值）
            pe_hist = hist["pe_ttm"]
            pb_hist = hist["pb"]
            pe_pct = pct_rank(pe_hist, cur["pe_ttm"])
            pb_pct = pct_rank(pb_hist, cur["pb"])
            sc = fund_score(pe_pct, pb_pct)
            if sc is None:
                continue
            # 研判理由：把分位数翻译成估值判断，模板拼接，不是大模型生成。
            if sc >= 0.2:
                position = "偏低"
            elif sc <= -0.2:
                position = "偏高"
            else:
                position = "中性"
            reason = "PE 处历史 %s 分位、PB 处 %s 分位，综合估值分 %+.2f，估值%s" % (
                "%.0f%%" % (pe_pct * 100) if pe_pct is not None else "NA",
                "%.0f%%" % (pb_pct * 100) if pb_pct is not None else "NA",
                sc, position)
            db.upsert_judgement(
                conn, sym, d, "fund", score_to_signal(sc), sc,
                reason=reason, provider="local", model="valuation_percentile",
                temperature=None, prompt_version=PROMPT_VERSION,
                source="offline_rule")
            written += 1

    conn.close()
    print("写入 %d 条，无估值数据 %d 只" % (written, no_data))
    print("说明：source=offline_rule，因为这是本地规则算的，不是 LLM 给的；")
    print("     答辩时要能指出这一点，不能把规则分说成大模型研判。")


if __name__ == "__main__":
    main()

"""三视角融合：tech / fund / sent -> judgements(view='fusion')

为什么必须做截面去均值（demean）：
实测情绪面 50 条里 32 条 BUY、10 条 SELL，均分 +0.46 对 -0.45，明显偏乐观。
如果直接加权，融合信号会系统性偏多，回测变成"几乎天天满仓"，这和"融合提升判断力"
是两回事。去均值后保留的是"相对同期其他标的的看法"，去掉的是视角自身的乐观/悲观底噪。
技术面不做去均值：它的概率是绝对值，代表该股次日上涨的估计，去掉就没有意义。

融合两种模式：
- weighted: score_f = Σ w_i * score_i（缺数据的视角自动按可用权重归一化）
- vote:     三路 signal 多数决（2/3 才给出方向，否则 HOLD）

用法：
    python scripts/fuse_views.py --start 2024-08-08 --end 2024-12-31
    python scripts/fuse_views.py --method vote --weights 0.5,0.25,0.25
"""

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

PROMPT_VERSION = "fusion-v1"
VIEWS = ("tech", "fund", "sent")


def load_matrix(conn, view, start, end, prompt_versions):
    rows = conn.execute(
        """SELECT symbol, date, score, signal FROM judgements
           WHERE view = ? AND date BETWEEN ? AND ?""",
        (view, start, end)).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    return df.pivot_table(index="date", columns="symbol", values="score", aggfunc="last")


def trading_dates(conn, start, end):
    """融合的日期轴必须是真实交易日，不能用研判日期的并集。

    踩过的坑：情绪面按"新闻发布日"落库，新闻在周末和节假日也发，
    于是 fusion 会多出 46 个不存在的"交易日"（2024-10-01~10-07 国庆假期等）。
    回测只按行情日期取信号，所以指标没受影响，但落库和前端都对不上。
    """
    rows = conn.execute(
        "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
        (start, end)).fetchall()
    return pd.Index([r["date"] for r in rows])


def demean(df):
    """按日横截面去均值：只保留相对同期其他标的的偏离。"""
    if df.empty:
        return df
    return df.sub(df.mean(axis=1), axis=0)


def to_signal(score, th):
    if score >= th:
        return "BUY"
    if score <= -th:
        return "SELL"
    return "HOLD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-08-08")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--method", default="weighted", choices=["weighted", "vote"])
    ap.add_argument("--weights", default="0.5,0.25,0.25",
                    help="tech,fund,sent 的权重")
    ap.add_argument("--no_demean", action="store_true",
                    help="关闭截面去均值（不建议，情绪面乐观偏置会传导）")
    ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--prompt_version", default=None,
                    help="写入版本号。默认 weighted=fusion-v1、vote=fusion-vote-v1，"
                         "两版可共存于同一张表，回测时按版本读取")
    ap.add_argument("--reset", action="store_true",
                    help="先删除本版本已有的 fusion 记录。改过日期轴后必须用，"
                         "否则非交易日的旧记录会残留")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    w = [float(x) for x in args.weights.split(",")]
    if len(w) != 3:
        raise SystemExit("--weights 需要三个数：tech,fund,sent")

    conn = db.connect(args.db)
    db.init_db(conn=conn)
    pv = args.prompt_version or (
        "fusion-vote-v1" if args.method == "vote" else "fusion-v1")
    if args.reset:
        n = conn.execute(
            "DELETE FROM judgements WHERE view='fusion' AND prompt_version=?",
            (pv,)).rowcount
        conn.commit()
        print("已清除 %s 旧记录 %d 条" % (pv, n))

    mats = {}
    for v in VIEWS:
        m = load_matrix(conn, v, args.start, args.end, None)
        mats[v] = m
        print("%-5s 覆盖 %d 日 × %d 只 = %d 条" % (
            v, m.shape[0], m.shape[1], int(m.notna().sum().sum()) if not m.empty else 0))

    if mats["tech"].empty:
        raise SystemExit("tech 视角为空，先跑 scripts/gen_tech.py")

    idx = trading_dates(conn, args.start, args.end)
    if len(idx) == 0:
        raise SystemExit("prices 表在该区间没有交易日，先跑 scripts/load_tables.py")
    print("交易日轴: %d 天（%s ~ %s）" % (len(idx), idx[0], idx[-1]))

    cols = mats["tech"].columns
    for v in VIEWS:
        if not mats[v].empty:
            cols = cols.union(mats[v].columns)

    parts = {}
    for v in VIEWS:
        if mats[v].empty:
            parts[v] = pd.DataFrame(index=idx, columns=cols, dtype=float)
        elif v == "sent":
            # 新闻可能在周末/节假日发布，交易日上没有对应记录。
            # 先按"交易日 ∪ 新闻日"建索引再 ffill，等价于"周末新闻影响下一个交易日"，
            # 这是金融上的常识，也是把稀疏新闻变成连续信号的必要处理。
            # 局限：简单 ffill 假设新闻影响持续到下一次新闻，未做衰减，论文里要写明。
            full_idx = mats[v].index.union(idx).sort_values()
            full = mats[v].reindex(index=full_idx, columns=cols).ffill()
            parts[v] = full.reindex(index=idx)
        else:
            parts[v] = mats[v].reindex(index=idx, columns=cols)

        if v in ("fund", "sent") and not args.no_demean:
            parts[v] = demean(parts[v])

    written, skip_empty = 0, 0
    dist = {"BUY": 0, "SELL": 0, "HOLD": 0}

    for date in idx:
        for sym in cols:
            scores = {v: parts[v].loc[date, sym] for v in VIEWS}
            avail = {v: float(s) for v, s in scores.items() if pd.notna(s)}
            if not avail:
                skip_empty += 1
                continue

            if args.method == "weighted":
                ws = sum(w[VIEWS.index(v)] for v in avail)
                if ws <= 0:
                    continue
                fscore = sum(w[VIEWS.index(v)] * s for v, s in avail.items()) / ws
            else:
                sigs = [to_signal(s, args.threshold) for s in avail.values()]
                buys = sigs.count("BUY")
                sells = sigs.count("SELL")
                need = len(avail) // 2 + 1
                if buys >= need:
                    fscore = float(max(avail.values()))
                elif sells >= need:
                    fscore = float(min(avail.values()))
                else:
                    fscore = 0.0

            fscore = max(-1.0, min(1.0, fscore))
            sig = to_signal(fscore, args.threshold)
            detail = " ".join("%s=%+.2f" % (v, s) for v, s in avail.items())
            db.upsert_judgement(
                conn, sym, date, "fusion", sig, fscore,
                reason=detail[:80], provider="local", model="rule_" + args.method,
                temperature=None, prompt_version=pv, source="live")
            dist[sig] += 1
            written += 1

    conn.close()
    print("\n写入 %d 条 fusion 研判（跳过空行 %d）" % (written, skip_empty))
    print("信号分布:", dist)
    if dist["BUY"] > 0.7 * max(1, written):
        print("!! 注意：BUY 占比过高，检查是否漏了去均值，或某个视角权重过大")


if __name__ == "__main__":
    main()

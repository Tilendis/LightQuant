"""基本面视角 v2：财报事件驱动（fund-event-v1）

背景：原来 fund 视角只有"估值分位"一个判据，而估值分位对次日涨跌无效
（权重被标定成 0.00）。但这不代表"基本面没用"，而是这个模块被做窄了：
公司经营层面的信息（业绩增没增、亏没亏）一直没接进来。CSMD50 数据集不含
财报表，但 news 里有大量带具体数字的财报报道，可以抽成结构化事件。

信号定义：
    对交易日 T，回看 [T-3, T-1] 窗口内的财报事件（T 日收盘后才看得到新闻，
    所以 T 日不能用自己的新闻 → 严格用 T-1 及更早，避免未来函数）。
    score = 窗口中 |score| 最大那条事件的分数，按天线性衰减（半衰期实测 1~3 天）。
    无事件 → score = 0，signal = HOLD。

实测依据（scripts 之外的验证脚本，结果见 docs/财报事件与基本面口径.md）：
    test 段（08-08~12-31）事件后 N=1 方向 ACC 0.5676、N=3 0.5270，N=5 起失效。
    所以衰减窗口只取 3 天，不做长期漂移。

与 fund-rule-v1（估值分位）的关系：**不是替换，是并列**。本版本只写事件信号，
        不掺估值，这样可以单独量出"公司经营信息"这一路的贡献。
        prompt_version 用 fund-event-v1，与旧的 fund-rule-v1 记录互不覆盖。

用法：
    python scripts/gen_fund_event.py --start 2024-03-15 --end 2024-12-31
"""

import argparse
import os
import sys
from bisect import bisect_left
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

PROMPT_VERSION = "fund-event-v1"
WINDOW = 3          # 回看天数（实测半衰期 1~3 天）
TH = 0.2            # 与 tech 的 score_to_signal 阈值保持一致，便于横向比


def score_to_signal(score, th=TH):
    if score >= th:
        return "BUY"
    if score <= -th:
        return "SELL"
    return "HOLD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-03-15")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    conn = db.connect(args.db)
    db.init_db(conn=conn)

    dates = [r["date"] for r in conn.execute(
        "SELECT DISTINCT date FROM prices ORDER BY date")]
    if not dates:
        raise SystemExit("prices 表为空，先跑 load_tables.py")

    events = conn.execute(
        "SELECT symbol,date,event_type,score,np_yoy,rev_yoy,is_loss,evidence "
        "FROM fund_events WHERE subject_ok=1 AND score IS NOT NULL ORDER BY date"
    ).fetchall()

    # symbol -> [(date, event)]
    ev_by_sym = defaultdict(list)
    for e in events:
        ev_by_sym[e["symbol"]].append((e["date"], e))

    symbols = [r["symbol"] for r in conn.execute("SELECT symbol FROM stocks ORDER BY symbol")]

    written = 0
    sig_cnt = {"BUY": 0, "SELL": 0, "HOLD": 0}
    with_ev = 0
    for sym in symbols:
        series = ev_by_sym.get(sym, [])
        if not series:
            continue
        for i, d in enumerate(dates):
            if not (args.start <= d <= args.end):
                continue
            # 窗口：T-3 .. T-1（严格不含 T 当天，避免把当日收盘后才知道的新闻当已知）
            lo = dates[max(0, i - WINDOW)]
            hi = dates[i - 1] if i > 0 else None
            if hi is None or hi < lo:
                continue
            cand = [e for (ed, e) in series if lo <= ed <= hi]
            if not cand:
                continue
            with_ev += 1
            # 取 |score| 最大的一条；再按距离 T 的天数衰减
            best = max(cand, key=lambda e: abs(float(e["score"])))
            dist = i - (bisect_left(dates, best["date"]))
            w = 1.0 - max(0, dist - 1) / float(WINDOW + 1)   # 隔 1 天权重 1.0，隔 3 天约 0.5
            score = max(-1.0, min(1.0, float(best["score"]) * w))
            sig = score_to_signal(score)

            et = best["event_type"]
            np_y = best["np_yoy"]
            rv = best["rev_yoy"]
            facts = []
            if np_y is not None:
                facts.append("净利润同比 %+.1f%%" % float(np_y))
            if rv is not None:
                facts.append("营收同比 %+.1f%%" % float(rv))
            if best["is_loss"]:
                facts.append("当期亏损")
            fact_txt = ("，" + "、".join(facts)) if facts else ""
            reason = ("财报事件（%s，%s）：原文「%s」%s。按实测盈余反应半衰期 %d 天，"
                      "距今 %d 天，权重 %.2f，判为%s"
                      % (et, best["date"], (best["evidence"] or "")[:60], fact_txt,
                         WINDOW, dist, w, {"BUY": "偏多", "SELL": "偏空", "HOLD": "中性"}[sig]))

            sig_cnt[sig] += 1
            if not args.dry_run:
                db.upsert_judgement(
                    conn, sym, d, "fund", sig, round(score, 4),
                    reason=reason, provider="local",
                    model="fund-event-extract", temperature=None,
                    prompt_version=PROMPT_VERSION, source="offline_rule")
            written += 1

    if not args.dry_run:
        conn.commit()

    print("候选 %d 条（有事件的 股票×交易日）" % with_ev)
    print("写入 %d 条 fund 研判（prompt_version=%s，窗口 %d 天，阈值 %.2f）"
          % (written, PROMPT_VERSION, WINDOW, TH))
    print("信号分布: BUY %d / SELL %d / HOLD %d"
          % (sig_cnt["BUY"], sig_cnt["SELL"], sig_cnt["HOLD"]))
    if args.dry_run:
        print("(dry_run：不写库)")
    conn.close()


if __name__ == "__main__":
    main()

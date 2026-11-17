# -*- coding: utf-8 -*-
"""基本面视角可用性实测（val 段，零 API 成本，只读库）。

回答一个问题：fund 视角除了"猜次日涨跌"，还有没有别的方式产生价值。

四组实验：
  1. 截面选池：同日按 score 排序取 top10 / bottom10，看次日收益与多空差
  2. 方向提示：按 signal / score 分档看次日收益与上涨占比
  3. 融合否决票：tech 有方向时，fund 强烈反对则跳过，看收益与命中率变化
  4. 事件分流 + 反向对照：只取有财报事件的样本；以及反向使用是否等价于短期反转

口径：
  收益 = adj_close(T+1) / adj_close(T) - 1
  T 日分数 -> T+1 收益，不含未来函数
  默认窗口 2024-03-15 ~ 2024-08-07（val 段）

用法：
  python scripts/eval_fund_paths.py
  python scripts/eval_fund_paths.py --fund_pv fund-rule-v1
"""
import argparse
import os
import sqlite3
import statistics
import sys
from bisect import bisect_left
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_DB = os.path.join(ROOT, "app", "data", "lightquant.db")
VAL = ("2024-03-15", "2024-08-07")
TECH_PV = "tech-v1"
TH_TECH = 0.05
EVENT_WIN = 90


def mean(xs):
    return statistics.mean(xs) if xs else None


def winrate(xs):
    return sum(1 for x in xs if x > 0) / len(xs) if xs else None


def tstat(xs):
    if len(xs) < 2:
        return None
    se = statistics.stdev(xs) / (len(xs) ** 0.5)
    return mean(xs) / se if se else None


def build_returns(conn):
    px = defaultdict(list)
    for r in conn.execute("SELECT symbol, date, adj_close FROM prices ORDER BY symbol, date"):
        px[r["symbol"]].append((r["date"], r["adj_close"]))
    out = {}
    prev = {}
    for sym, rows in px.items():
        for i in range(len(rows) - 1):
            d0, a0 = rows[i]
            d1, a1 = rows[i + 1]
            if a0 and a1 and a0 > 0:
                out[(sym, d0)] = a1 / a0 - 1.0
            if i >= 1:
                d_1, a_1 = rows[i - 1]
                if a_1 and a0 and a_1 > 0:
                    prev[(sym, d0)] = a0 / a_1 - 1.0
    return out, prev


def load_view(conn, view, pv, start, end):
    score, sig = {}, {}
    for r in conn.execute(
            "SELECT symbol,date,score,signal FROM judgements "
            "WHERE view=? AND prompt_version=? AND date BETWEEN ? AND ?",
            (view, pv, start, end)):
        k = (r["symbol"], r["date"])
        score[k] = r["score"]
        sig[k] = r["signal"]
    return score, sig


def cross_section(score_map, dates, ret, k=10, min_names=30):
    top, bot, diffs = [], [], []
    for d in dates:
        day = [(s, sc, ret[(s, d)]) for (s, dd), sc in score_map.items()
               if dd == d and (s, d) in ret]
        if len(day) < min_names:
            continue
        day.sort(key=lambda x: -x[1])
        top.append(mean([x[2] for x in day[:k]]))
        bot.append(mean([x[2] for x in day[-k:]]))
        diffs.append(top[-1] - bot[-1])
    return top, bot, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--fund_pv", default="fund-agent-v2")
    ap.add_argument("--start", default=VAL[0])
    ap.add_argument("--end", default=VAL[1])
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    ret, prev = build_returns(conn)
    fund, fund_sig = load_view(conn, "fund", args.fund_pv, args.start, args.end)
    tech, _ = load_view(conn, "tech", TECH_PV, args.start, args.end)
    dates = sorted({d for (_, d) in fund})

    print("基本面版本 %s：%d 条 / %d 个交易日（%s ~ %s）"
          % (args.fund_pv, len(fund), len(dates), dates[0] if dates else "-",
             dates[-1] if dates else "-"))
    print("tech 记录 %d，可算次日收益 %d\n" % (len(tech), len(ret)))

    # ---------- 一、截面选池 ----------
    print("=" * 74)
    print("一、截面选池（同日排序 top10 / bottom10，次日收益）")
    print("=" * 74)
    print("%-16s %6s %11s %11s %11s %8s" % ("排序依据", "天数", "top10", "bottom10", "多空差", "t"))
    for name, sm in ((args.fund_pv, fund), (TECH_PV, tech)):
        top, bot, diffs = cross_section(sm, dates, ret)
        if not diffs:
            print("%-16s 样本不足" % name)
            continue
        print("%-16s %6d %10.3f%% %10.3f%% %10.3f%% %8s"
              % (name, len(diffs), mean(top) * 100, mean(bot) * 100, mean(diffs) * 100,
                 ("%.2f" % tstat(diffs)) if tstat(diffs) is not None else "-"))

    # ---------- 二、方向提示 ----------
    print("\n" + "=" * 74)
    print("二、方向提示（按 signal / score 分档，次日收益与上涨占比）")
    print("=" * 74)
    print("%-18s %7s %12s %11s" % ("分组", "n", "次日均收益", "上涨占比"))
    for s in ("BUY", "HOLD", "SELL"):
        xs = [ret[k] for k, v in fund_sig.items() if v == s and k in ret]
        if xs:
            print("%-18s %7d %11.3f%% %10.2f%%" % (s, len(xs), mean(xs) * 100, winrate(xs) * 100))
    print()
    bins = [(-9, -0.2), (-0.2, -0.05), (-0.05, 0.05), (0.05, 0.2), (0.2, 9)]
    labels = ["score < -0.2", "-0.2 ~ -0.05", "-0.05 ~ 0.05", "0.05 ~ 0.2", "score > 0.2"]
    for (lo, hi), lab in zip(bins, labels):
        xs = [ret[k] for k, v in fund.items() if k in ret and lo <= v < hi]
        if xs:
            print("%-18s %7d %11.3f%% %10.2f%%" % (lab, len(xs), mean(xs) * 100, winrate(xs) * 100))

    # ---------- 三、否决票 ----------
    print("\n" + "=" * 74)
    print("三、融合否决票（tech 有方向时，fund 强烈反对则跳过）")
    print("=" * 74)
    common = [k for k in set(tech) & set(fund) & set(ret) if ret[k] is not None]
    print("tech 与 fund 都有分数的样本 %d\n" % len(common))
    print("%-10s %7s %12s %9s %10s %7s %12s"
          % ("否决阈值", "执行n", "执行均收益", "胜率", "方向命中", "跳过n", "跳过均收益"))

    def evaluate(veto_th):
        keep, skip, hit = [], [], 0
        for k in common:
            t = tech[k]
            if abs(t) <= TH_TECH:
                continue
            f = fund.get(k, 0.0)
            opposed = (t > 0 and f <= -veto_th) or (t < 0 and f >= veto_th)
            if opposed:
                skip.append(ret[k])
                continue
            keep.append(ret[k])
            y = 1 if ret[k] > 0 else 0
            if (t > 0 and y == 1) or (t < 0 and y == 0):
                hit += 1
        return keep, skip, (hit / len(keep) if keep else None)

    keep, skip, hit = evaluate(99.0)
    print("%-10s %7d %11.3f%% %8.2f%% %9.2f%% %7d %11s"
          % ("不否决", len(keep), mean(keep) * 100, winrate(keep) * 100, hit * 100, len(skip), "-"))
    for th in (0.10, 0.15, 0.20, 0.30):
        keep, skip, hit = evaluate(th)
        print("%-10s %7d %11.3f%% %8.2f%% %9.2f%% %7d %10.3f%%"
              % ("±%.2f" % th, len(keep), mean(keep) * 100, winrate(keep) * 100,
                 hit * 100, len(skip), mean(skip) * 100))

    # ---------- 四、事件分流 ----------
    print("\n" + "=" * 74)
    print("四、按是否有财报事件覆盖分流")
    print("=" * 74)
    ev = defaultdict(list)
    for r in conn.execute("SELECT symbol, date FROM fund_events WHERE subject_ok=1"):
        ev[r["symbol"]].append(r["date"])
    all_dates = sorted({r["date"] for r in conn.execute("SELECT DISTINCT date FROM prices")})
    idx = {d: i for i, d in enumerate(all_dates)}

    def covered(sym, d):
        ds = ev.get(sym)
        if not ds or d not in idx:
            return False
        i = bisect_left(ds, d)
        if i == 0:
            return False
        last = ds[i - 1]
        return (d in idx and last in idx and idx[d] - idx[last] <= EVENT_WIN)

    has = [k for k in fund if k in ret and covered(*k)]
    no = [k for k in fund if k in ret and not covered(*k)]
    print("有事件覆盖 %d，无事件覆盖 %d\n" % (len(has), len(no)))
    for nm, keys in (("有事件", has), ("无事件", no)):
        for s in ("BUY", "SELL"):
            xs = [ret[k] for k in keys if fund_sig[k] == s]
            if xs:
                print("%-8s %-5s n=%4d  次日均收益 %+.3f%%  上涨占比 %.2f%%"
                      % (nm, s, len(xs), mean(xs) * 100, winrate(xs) * 100))
        print()

    # ---------- 五、反向与反转对照 ----------
    print("=" * 74)
    print("五、反向使用 vs 纯短期反转因子")
    print("=" * 74)
    pos = neg = tot = 0
    for k, s in fund_sig.items():
        if k not in ret or s == "HOLD":
            continue
        y = 1 if ret[k] > 0 else 0
        tot += 1
        if (s == "BUY" and y == 1) or (s == "SELL" and y == 0):
            pos += 1
        if (s == "BUY" and y == 0) or (s == "SELL" and y == 1):
            neg += 1
    se = (0.25 / tot) ** 0.5 if tot else None
    print("正向 ACC %.4f（z=%+.2f）" % (pos / tot, (pos / tot - 0.5) / se))
    print("反向 ACC %.4f（z=%+.2f）" % (neg / tot, (neg / tot - 0.5) / se))

    fw = []
    for d in dates:
        day = [(s, fund[(s, d)], ret[(s, d)]) for (s, dd) in fund if dd == d and (s, d) in ret]
        if len(day) < 30:
            continue
        day.sort(key=lambda x: -x[1])
        fw.append(mean([x[2] for x in day[:10]]) - mean([x[2] for x in day[-10:]]))
    print("\n反向截面多空差 %+.3f%%（t=%+.2f）"
          % (-mean(fw) * 100, -tstat(fw)))

    rev = []
    for d in dates:
        day = [(s, prev[(s, d)], ret[(s, d)]) for (s, dd) in fund
               if dd == d and (s, d) in ret and (s, d) in prev]
        if len(day) < 30:
            continue
        day.sort(key=lambda x: x[1])     # 昨日跌最多的在前 -> 今天买
        rev.append(mean([x[2] for x in day[:10]]) - mean([x[2] for x in day[-10:]]))
    print("纯昨日反转因子    %+.3f%%（t=%+.2f）"
          % (mean(rev) * 100, tstat(rev)))

    conn.close()


if __name__ == "__main__":
    main()

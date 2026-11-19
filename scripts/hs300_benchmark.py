# -*- coding: utf-8 -*-
"""沪深300 成分等权基准对照（CSMD300），用于回答"基准是不是自己挑的"。

背景：主口径回测的基准是 CSMD50 等权（50 只票，close/close[0] 后逐票平均，
见 backtest/engine_v2.py:352、370-371）。CSMD50 对应上证 50 的 50 只成分股，CSMD300 对应沪深 300 的 300 只成分股，
二者来自同名论文（CSMD: Curated Multimodal Dataset for Chinese Stock Analysis, CIKM'25），
实测 CSMD50 的 50 只在 CSMD300 中全部命中。因此可用 CSMD300 全 300 只票在同窗口
构造"沪深300 成分等权"代理——注意它不是官方指数 000300，无权重、无调样、无分红口径。

口径对齐（与主口径逐项一致，不改动任何回测数字）：
- 价格：Close（不复权），与 summary 的 price_basis=close_unadjusted 一致
- 归一：每只票 close / close[窗口首日]，逐票归一后等权平均
- 窗口：默认取主口径 nav.csv 的 93 个交易日，保证与 19.04% 可比
- 缺失：窗口内无成交的票按前值填充；窗口首日即缺失的按首个有效值回填

输出：backtest/bench_compare/hs300_ew.json 与 hs300_ew_nav.csv
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"D:\03_Study\GraduationProject\LightQuant"
PRICE50 = os.path.join(ROOT, "dataset", "CSMD50", "price")
PRICE300 = os.path.join(ROOT, "dataset", "CSMD300", "price")
NAV = os.path.join(ROOT, "backtest", "result_v2", "test_fusion_3view_v2", "nav.csv")
OUT_DIR = os.path.join(ROOT, "backtest", "bench_compare")
BAR_PER_YEAR = 252


def load_dates():
    """窗口日期取主口径 nav.csv，保证与 19.04% 逐日可比。"""
    rows = list(csv.DictReader(open(NAV, encoding="utf-8-sig")))
    return [r["date"] for r in rows]


def read_close(path, dates):
    """读单票 Close，按窗口日期 reindex + 前向填充；首日缺失用首个有效值回填。"""
    df = pd.read_csv(path, usecols=["Date", "Close"], encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    s = df.set_index("Date")["Close"].astype(float)
    s = s.reindex(dates).ffill().bfill()
    return s


def equal_weight_bench(folder, dates, label):
    files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
    series, missing = [], []
    for f in files:
        s = read_close(os.path.join(folder, f), dates)
        if s.isna().any():
            missing.append(f)
            continue
        series.append((s / s.iloc[0]).to_numpy())
    mat = np.vstack(series)
    bench = mat.mean(axis=0)
    stacked = pd.DataFrame({"date": dates, "nav": bench})
    print(f"{label}: {len(series)} 只票参与等权，{len(missing)} 只因窗口内全缺被剔除")
    if missing:
        print("  剔除样例:", missing[:5])
    return stacked, mat


def max_drawdown(nav, dates):
    peak_idx = 0
    mdd = 0.0
    start = end = dates[0]
    for i in range(1, len(nav)):
        if nav[i] > nav[peak_idx]:
            peak_idx = i
        dd = (nav[peak_idx] - nav[i]) / nav[peak_idx]
        if dd > mdd:
            mdd, start, end = dd, dates[peak_idx], dates[i]
    return mdd, start, end


def annualized(nav, dates):
    return (nav[-1] / nav[0]) ** (BAR_PER_YEAR / len(dates)) - 1


def main():
    dates = load_dates()
    os.makedirs(OUT_DIR, exist_ok=True)

    b50, m50 = equal_weight_bench(PRICE50, dates, "CSMD50 等权（上证50 成分，主口径基准）")
    b300, m300 = equal_weight_bench(PRICE300, dates, "CSMD300 等权（沪深300 成分，数据集代理，非官方指数）")

    # 校验：复现出的 CSMD50 等权应等于主口径 nav.csv 的 benchmark_nav
    ref = list(csv.DictReader(open(NAV, encoding="utf-8-sig")))
    ref_bench = np.array([float(r["benchmark_nav"]) for r in ref])
    err = float(np.max(np.abs(b50["nav"].to_numpy() / b50["nav"].iloc[0] - ref_bench)))
    print(f"\n口径校验：复现 CSMD50 等权与 nav.csv 的 benchmark_nav 最大偏差 = {err:.3e}")
    if err > 1e-6:
        raise SystemExit("校验失败：复现基准与主口径不一致，先查口径再出结论")

    strat = np.array([float(r["portfolio_nav"]) for r in ref])

    def block(name, nav, mat):
        cum = nav[-1] / nav[0] - 1
        mdd, s, e = max_drawdown(nav, dates)
        ann = annualized(nav, dates)
        r = np.diff(nav) / nav[:-1]
        vol = r.std(ddof=1) * np.sqrt(BAR_PER_YEAR)
        # 等权组合的日收益标准差 vs 个股平均（衡量分散化程度）
        stock_vol = np.mean([np.diff(row).std(ddof=1) for row in mat]) * np.sqrt(BAR_PER_YEAR)
        print(f"\n{name}")
        print(f"  累计 {cum*100:6.2f}%   年化 {ann*100:6.2f}%   年化波动 {vol*100:5.2f}%   "
              f"最大回撤 {mdd*100:5.2f}%（{s} → {e}）")
        print(f"  （参考）成分股平均年化波动 {stock_vol*100:.2f}%，组合分散化后降至 {vol*100:.2f}%")
        return dict(cum_return=cum, annual_return=ann, annual_vol=vol, mdd=mdd,
                    mdd_start=s, mdd_end=e, n_stocks=mat.shape[0])

    s50 = block("CSMD50 等权（上证50 成分，主口径基准）", b50["nav"].to_numpy(), m50)
    s300 = block("CSMD300 等权（沪深300 成分，数据集代理，非官方指数）", b300["nav"].to_numpy(), m300)
    s_strat = block("主口径策略（fusion-test-3view-v2）", strat, m300)

    # 策略 vs 两个基准
    r_strat = np.diff(strat) / strat[:-1]
    print("\n" + "=" * 68)
    print("策略与两个基准的对照（同 93 个交易日，窗口 %s ~ %s）" % (dates[0], dates[-1]))
    print("=" * 68)
    print(f"{'指标':<22}{'CSMD50 等权':>14}{'CSMD300 等权':>15}")
    bench50 = b50["nav"].to_numpy()
    bench300 = b300["nav"].to_numpy()
    for name, b in (("累计收益", None),):
        pass
    rows = [
        ("基准累计收益", s50["cum_return"], s300["cum_return"]),
        ("基准最大回撤", -s50["mdd"], -s300["mdd"]),
        ("策略累计收益", s_strat["cum_return"], s_strat["cum_return"]),
        ("超额（pp）", (s_strat["cum_return"] - s50["cum_return"]) * 100,
         (s_strat["cum_return"] - s300["cum_return"]) * 100),
    ]
    for label, a, c in rows:
        if label.endswith("（pp）"):
            print(f"{label:<22}{a:>+13.2f}{c:>+14.2f}")
        else:
            print(f"{label:<22}{a*100:>13.2f}%{c*100:>13.2f}%")

    for tag, b in (("CSMD50", bench50), ("CSMD300", bench300)):
        rb = np.diff(b) / b[:-1]
        te = (r_strat - rb).std(ddof=1) * np.sqrt(BAR_PER_YEAR)
        ir = (r_strat - rb).mean() / (r_strat - rb).std(ddof=1) * np.sqrt(BAR_PER_YEAR)
        print(f"相对 {tag}：跟踪误差 {te*100:.2f}%，IR {ir:+.3f}")

    corr = np.corrcoef(np.diff(bench50) / bench50[:-1], np.diff(bench300) / bench300[:-1])[0, 1]
    print(f"两个基准日收益相关系数 {corr:.4f}")

    # 同仓位折算（平均名义仓位 28.6%，来自 trades 明细：15.1 只 / 50 × 95%）
    expo = 0.286
    print("\n同仓位折算（策略平均名义仓位 %.1f%%）" % (expo * 100))
    for tag, s in (("CSMD50", s50), ("CSMD300", s300)):
        adj = s["cum_return"] * expo
        print(f"  vs {tag}：折算基准 {adj*100:.2f}%，策略超额 "
              f"{(s_strat['cum_return'] - adj)*100:+.2f}pp")

    out = {
        "window": {"start": dates[0], "end": dates[-1], "bars": len(dates)},
        "bench_csmd50": s50,
        "bench_csmd300": s300,
        "strategy": s_strat,
        "bench_daily_corr": float(corr),
        "avg_exposure": expo,
        "note": "CSMD50=上证50 成分（50 只），CSMD300=沪深300 成分（300 只），均来自 CSMD 数据集（CIKM'25）；CSMD300 等权为沪深300 成分代理，非官方指数 000300；CSMD50 的 50 只在 CSMD300 中 50/50 命中",
    }
    with open(os.path.join(OUT_DIR, "hs300_ew.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pd.DataFrame({"date": dates, "nav_csmd50": bench50, "nav_csmd300": bench300,
                  "nav_strategy": strat}).to_csv(
        os.path.join(OUT_DIR, "hs300_ew_nav.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已写出 {os.path.join(OUT_DIR, 'hs300_ew.json')} 与 hs300_ew_nav.csv")


if __name__ == "__main__":
    main()

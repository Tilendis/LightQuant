"""截面选股策略的组合回测。

为什么不能复用 run_backtest：那套引擎是"每只股票各开一个 10 万账户、最后等权
平均净值"。截面策略要求"每天只持有 top N、其余空仓"，套进去会让另外 40 只的
空账户净值恒为 1，把组合收益稀释成原来的 1/N——数字看着像回测，其实毫无意义。

所以这里按真实组合重写：一个共享资金池，调仓日卖出全部旧持仓、等权买入新
top N，非调仓日按收盘价估值。成本、指标、自检项与 engine_v2 完全同口径，
保证两条路径的结果可以放在一张表里比。

成交时点：score 来自上一交易日收盘，所以只在下一交易日开盘成交——这是这套
信号唯一可执行的时点。之前的 +0.660% 是按"当日收盘买入"算的，含了拿不到的
隔夜跳空，这里不重复那个错误。
"""

import json
import os

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_v2 import (  # noqa: E402
    annualized_return,
    cumulative_return,
    information_ratio,
    max_drawdown,
    sharpe,
)


def _load_panel(price_folder):
    """读所有 csv，返回 (dates, opens, closes, symbols, adjusted_flag)。"""
    files = sorted(f for f in os.listdir(price_folder) if f.endswith(".csv"))
    if not files:
        raise RuntimeError("no csv in %s" % price_folder)

    opens, closes, dates_ref, adjusted = {}, {}, None, False
    for f in files:
        sym = f[:-4]
        df = pd.read_csv(os.path.join(price_folder, f))
        dates = [str(d) for d in df["Date"]]
        if dates_ref is None:
            dates_ref = dates
        elif dates != dates_ref:
            raise RuntimeError("日期不对齐：%s" % f)

        raw_close = df["Close"].astype(float).values
        raw_open = df["Open"].astype(float).values
        if "Adj Close" in df.columns:
            adj = df["Adj Close"].astype(float).values
            factor = (np.ones_like(raw_close) if np.allclose(adj, raw_close, rtol=0, atol=1e-9)
                      else adj / raw_close)
        else:
            factor = np.ones_like(raw_close)
        if not np.all(np.isfinite(factor)) or np.any(factor <= 0):
            raise RuntimeError("复权因子异常：%s" % f)
        if not np.allclose(factor, 1.0):
            adjusted = True

        # Open 也要跟着复权：只调 Close 会让"开盘价 × 股数"的成交金额与净值脱节
        opens[sym] = dict(zip(dates, raw_open * factor))
        closes[sym] = dict(zip(dates, raw_close * factor))

    return dates_ref, opens, closes, [f[:-4] for f in files], adjusted


def _jsonable(obj):
    """递归把 NaN / Inf / numpy 标量转成 JSON 能吃的值。

    起因是实测出来的：稳健性分段不足时 cum_drop_best_k 会是 NaN，json.dump 默认
    把 NaN 原样写成 `NaN`，这不是合法 JSON——前端 JSON.parse 整页挂掉，一屏白。
    per_stock_mean 早就避开了这个坑（只留四个字段），robustness 漏了。
    NaN 一律写 null，前端 pct(null) 渲染成「—」，语义上也正好是"算不出来"。
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def run_cross_section(price_folder, score_df, cfg, tag, out_dir, start, end):
    """跑截面策略。score_df: DataFrame(index=date, columns=symbol, values=score)。

    cfg 需要：top_n / hold_days / initial_cash / commission_rate / min_commission
             / stamp_duty_rate / slippage_rate / lot_size / rf_annual
    """
    dates_all, opens, closes, symbols, adjusted = _load_panel(price_folder)
    dates = [d for d in dates_all if start <= d <= end]
    if len(dates) < 2:
        raise RuntimeError("区间内交易日不足：%s ~ %s" % (start, end))

    top_n = int(cfg.get("top_n") or 10)
    hold = max(1, int(cfg.get("hold_days") or 5))
    cash = float(cfg["initial_cash"])
    comm = float(cfg["commission_rate"])
    min_comm = float(cfg["min_commission"])
    stamp = float(cfg["stamp_duty_rate"])
    slip = float(cfg["slippage_rate"])
    lot = int(cfg.get("lot_size") or 1)

    score_df = score_df.reindex(dates)
    held = {}
    nav, trades = [], []
    rebalance_log = []

    for i, d in enumerate(dates):
        # 调仓日：i 为 hold 的整数倍。用 i-1 日的分数（收盘后可得），d 日开盘成交。
        if i > 0 and i % hold == 0:
            prev = dates[i - 1]
            row = score_df.loc[prev].dropna() if prev in score_df.index else pd.Series(dtype=float)
            row = row[row.index.isin([s for s in symbols if d in opens[s]])]
            if len(row) >= top_n:
                picks = row.sort_values(ascending=False).head(top_n).index.tolist()

                for sym, qty in list(held.items()):
                    px = opens[sym].get(d)
                    if px is None or qty <= 0:
                        continue
                    exec_px = px * (1 - slip)
                    amount = qty * exec_px
                    fee = max(min_comm, amount * comm) + amount * stamp
                    cash += amount - fee
                    trades.append(dict(date=d, side="SELL", price=round(exec_px, 4),
                                       shares=int(qty), amount=round(amount, 2),
                                       fee=round(fee, 2), reason="rebalance", stock=sym))
                    del held[sym]

                # 等权：现金按实际可买到的只数分配，停牌股不进池就不占预算
                tradable = [s for s in picks if d in opens[s]]
                budget_each = cash / max(1, len(tradable))
                bought = 0
                for sym in tradable:
                    px = opens[sym].get(d)
                    exec_px = px * (1 + slip)
                    qty = int(budget_each / exec_px // lot) * lot
                    while qty > 0:
                        amount = qty * exec_px
                        fee = max(min_comm, amount * comm)
                        if amount + fee <= cash:
                            break
                        qty -= lot
                    if qty <= 0:
                        continue
                    amount = qty * exec_px
                    fee = max(min_comm, amount * comm)
                    cash -= amount + fee
                    held[sym] = held.get(sym, 0) + qty
                    trades.append(dict(date=d, side="BUY", price=round(exec_px, 4),
                                       shares=int(qty), amount=round(amount, 2),
                                       fee=round(fee, 2), reason="rebalance", stock=sym))
                    bought += 1
                rebalance_log.append((d, bought, len(tradable), round(budget_each, 2)))

        equity = cash + sum(q * closes[s].get(d, 0.0) for s, q in held.items())
        nav.append(equity / float(cfg["initial_cash"]))

    nav = np.asarray(nav, dtype=float)

    # 基准：50 只等权买入持有，与 engine_v2 同口径（都从区间首日收盘算起）
    bench_curve = [np.mean([closes[s].get(d, closes[s].get(dates[0], 0.0))
                            for s in symbols]) for d in dates]
    b0 = bench_curve[0] if bench_curve and bench_curve[0] else 1.0
    bench = np.asarray(bench_curve, dtype=float) / b0

    mdd, md_start, md_end = max_drawdown(nav, dates)
    arr = annualized_return(nav)

    # 稳健性：按调仓周期切出每段的收益，去掉最好的 N 段后重算累计。
    # 这条策略在 test 段有一半收益来自单次行情（9/24 政策那波），不把这个数
    # 算出来，报告里的 53% 就是不可追问的。
    period_rets = []
    for j in range(len(rebalance_log)):
        i1 = (j + 1) * hold
        i2 = min(i1 + hold, len(nav) - 1)
        if i1 < len(nav) and i2 > i1 and nav[i1] > 0:
            period_rets.append(float(nav[i2] / nav[i1] - 1.0))
    period_rets.sort(reverse=True)
    keep_all = float(np.prod([1 + r for r in period_rets]) - 1) if period_rets else float("nan")
    trimmed = {}
    for k in (1, 2, 3):
        kept = period_rets[k:]
        trimmed[k] = float(np.prod([1 + r for r in kept]) - 1) if kept else float("nan")

    port = dict(
        stocks=len(symbols), bars=len(dates),
        first_date=dates[0], last_date=dates[-1],
        cum_return=cumulative_return(nav), arr=arr,
        sharpe=sharpe(nav, float(cfg.get("rf_annual") or 0.02)),
        mdd=mdd, mdd_start=md_start, mdd_end=md_end,
        calmar=(arr / mdd) if mdd > 0 else 0.0,
        ir_vs_equal_weight=information_ratio(nav, bench),
        bench_cum_return=cumulative_return(bench),
        bench_mdd=max_drawdown(bench, dates)[0],
        total_trades=len(trades),
        total_fee=float(sum(t["fee"] for t in trades)),
        stop_loss_exits=0, take_profit_exits=0,
        final_cash_negative=bool((nav < 0).any()),
        price_basis=("adj_close" if adjusted else "close_unadjusted"),
        price_adjusted=bool(adjusted), price_factor_jumps=0,
        files_with_factor=(len(symbols) if adjusted else 0),
        strategy_type="cross_section",
        top_n=top_n, hold_days=hold, rebalance_count=len(rebalance_log),
        robustness={
            "period_count": len(period_rets),
            "cum_all": keep_all,
            "cum_drop_best_1": trimmed[1],
            "cum_drop_best_2": trimmed[2],
            "cum_drop_best_3": trimmed[3],
            "best_period": period_rets[0] if period_rets else 0.0,
        },
    )

    # 自检：与 engine_v2 同一套，trade_pairs_balanced 换成截面策略该看的
    # "每次调仓清空旧仓"——截面策略每次买卖笔数天然不等，套原式会恒为假。
    nav_arr = nav
    peak = np.maximum.accumulate(nav_arr)
    dd = (peak - nav_arr) / peak
    trough_idx = int(np.argmax(dd))
    checks = {
        "nav_len_matches_dates": len(nav) == len(dates),
        "no_nan_in_nav": bool(not np.isnan(nav_arr).any()),
        "no_negative_equity": not port["final_cash_negative"],
        "mdd_span_covers_trough": bool(dates.index(md_start) <= trough_idx <= dates.index(md_end)),
        "mdd_equals_recomputed": bool(abs(dd.max() - port["mdd"]) < 1e-12),
        "metrics_finite": bool(np.isfinite([arr, port["sharpe"], mdd,
                                            port["ir_vs_equal_weight"]]).all()),
        "rebalance_executed": len(rebalance_log) > 0,
        "price_basis_declared": port["price_basis"] in ("adj_close", "close_unadjusted"),
    }

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({"date": dates, "portfolio_nav": nav, "benchmark_nav": bench}).to_csv(
        os.path.join(out_dir, "nav.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(trades).to_csv(
        os.path.join(out_dir, "trades.csv"), index=False, encoding="utf-8-sig")

    # per_stock：截面策略没有单只独立账户，这里给"每只被选中时的持仓期方向命中"，
    # 让前端的 per_stock_mean.acc 仍有意义，不是留空。
    per_stock = []
    for sym in symbols:
        mine = [t for t in trades if t["stock"] == sym and t["side"] == "BUY"]
        if not mine:
            continue
        idxs = [dates.index(t["date"]) for t in mine]
        hits, rets = [], []
        for j in idxs:
            k = min(j + hold, len(dates) - 1)
            if k <= j:
                continue
            r = closes[sym].get(dates[k], 0.0) / closes[sym].get(dates[j], 0.0) - 1.0
            rets.append(r)
            hits.append(1 if r > 0 else 0)
        per_stock.append(dict(stock=sym, acc=float(np.mean(hits)) if hits else float("nan"),
                              cum_return=float(np.sum(rets)) if rets else 0.0,
                              sharpe=float("nan"), mdd=float("nan"),
                              trades=len(mine),
                              total_fee=float(sum(t["fee"] for t in mine))))
    pd.DataFrame(per_stock).to_csv(
        os.path.join(out_dir, "per_stock.csv"), index=False, encoding="utf-8-sig")

    summary = dict(
        tag=tag, config=cfg, portfolio=port, checks=checks,
        # 只留四个前端会用到的字段：sharpe / mdd 在截面策略里没有单只时间序列可算，
        # 写 NaN 会让 json.dump 产出非法 JSON，前端 JSON.parse 整页挂掉。
        per_stock_mean={
            "acc": float(np.nanmean([s["acc"] for s in per_stock])) if per_stock else 0.0,
            "cum_return": float(np.nanmean([s["cum_return"] for s in per_stock])) if per_stock else 0.0,
            "trades": float(np.nanmean([s["trades"] for s in per_stock])) if per_stock else 0.0,
            "total_fee": float(np.nanmean([s["total_fee"] for s in per_stock])) if per_stock else 0.0,
        },
        rebalance_log=[dict(date=d, bought=b, pool=p, budget=bu) for d, b, p, bu in rebalance_log],
    )
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(summary), fh, ensure_ascii=False, indent=2)

    return summary

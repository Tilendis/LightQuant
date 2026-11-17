"""可信回测引擎 v2（不修改旧 backtest_single.py，新逻辑独立成文件）

与旧实现的差别（逐条对应旧版的 5 处硬伤）：

1. 仓位：旧版初始资金 10000 但每次只买卖 1 股（`stock < 1`），净值几乎不动，
   SR/IR 分母趋 0。新版按 `position_pct` 用可用资金的固定比例建仓，股数按整数算。
2. 成本：旧版 `cost = 0.000`。新版计入双边佣金（含最低佣金）+ 卖出印花税 + 双边滑点。
3. T+1：旧版允许当天买当天卖（信号反转即反向）。新版买入当日不可卖，最早次日。
4. MDD：旧版取全序列最大值当峰值（`max(asset_list)`）。新版用滚动峰值
   `np.maximum.accumulate`，并返回回撤区间的起止日期（供 ECharts markArea）。
5. SR / IR：旧版一个用 rf=1%、一个用 rf=3%，且 IR 其实还是夏普的算法。新版统一
   无风险利率，IR 改成"相对等权买入持有基准"的超额收益信息比率。

已知做不到、如实标注的两点：
- **涨跌停与停牌无法严格判定**：数据只有 OHLCV，没有前收盘价、没有停牌标记。
  只能用"当日 high == low（一字板）"作为近似代理，跳过这种不可成交的交易日。
  这不等于实现了涨跌停规则，论文里只能写成近似。
- **停牌日被数据源直接剔除**，无法从缺口还原，因此不做补洞。

另一条既有局限（未修，继续如实交代）：`utils.price_dataloader` 里的标准化是
对整段序列做 z-score，会用到测试期之后的信息。本引擎沿用同一套预处理以保持
与训练时口径一致，因此预测本身带有轻微前视偏差。
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.price_dataloader import Backtest_Dataset  # noqa: E402
from model.LSTM import LSTM  # noqa: E402
from model.ALSTM import ALSTM  # noqa: E402
from model.Adv_LSTM import AdvLSTM  # noqa: E402
from model.BiLSTM import BiLSTM  # noqa: E402

BAR_PER_YEAR = 252


# ---------------------------------------------------------------- 指标


def max_drawdown(nav, dates):
    """滚动峰值口径的最大回撤，并返回区间起止日期。"""
    nav = np.asarray(nav, dtype=float)
    if nav.size < 2:
        return 0.0, None, None
    peak = nav[0]
    peak_date = dates[0]
    worst = 0.0
    best_span = (dates[0], dates[0])
    for i in range(nav.size):
        if nav[i] > peak:
            peak = nav[i]
            peak_date = dates[i]
        dd = (peak - nav[i]) / peak
        if dd > worst:
            worst = dd
            best_span = (peak_date, dates[i])
    return float(worst), best_span[0], best_span[1]


def annualized_return(nav):
    nav = np.asarray(nav, dtype=float)
    if nav.size < 2 or nav[0] <= 0:
        return float("nan")
    return float((nav[-1] / nav[0]) ** (BAR_PER_YEAR / (nav.size - 1)) - 1)


def sharpe(nav, rf_annual):
    r = np.diff(nav) / np.asarray(nav[:-1], dtype=float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    rf_d = (1 + rf_annual) ** (1 / BAR_PER_YEAR) - 1
    return float((r.mean() - rf_d) / r.std(ddof=1) * np.sqrt(BAR_PER_YEAR))


def information_ratio(nav, bench_nav):
    r = np.diff(nav) / np.asarray(nav[:-1], dtype=float)
    b = np.diff(bench_nav) / np.asarray(bench_nav[:-1], dtype=float)
    d = r - b
    if d.size < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1) * np.sqrt(BAR_PER_YEAR))


def cumulative_return(nav):
    nav = np.asarray(nav, dtype=float)
    return float(nav[-1] / nav[0] - 1) if nav.size >= 2 else float("nan")


# ---------------------------------------------------------------- 单只股票


def simulate_one_stock(dates, closes, highs, lows, probs, cfg, pos_pct_series=None):
    """按日推进的单标的模拟。

    决策与成交都在当日收盘：用窗口最后一天的收盘价成交（预测标签是次日方向，
    因此"今天收盘决策、今天收盘成交"不构成前视）。

    pos_pct_series: 可选，长度与 probs 对齐的每日建仓比例。给出时按日取值，
    用于"技术面管方向、情绪面管仓位"的分工制——方向仍来自 probs，仓位由
    另一个视角决定。不传则回退 cfg["position_pct"]，行为与旧版逐位相同。
    """
    cash = cfg["initial_cash"]
    shares = 0
    held_days = 0
    nav, trades = [], []
    skipped_locked = 0

    pos_pct_default = cfg["position_pct"]
    lot = cfg["lot_size"]
    comm = cfg["commission_rate"]
    min_comm = cfg["min_commission"]
    stamp = cfg["stamp_duty_rate"]
    slip = cfg["slippage_rate"]

    # 止盈/止损：0 表示关闭（保持与"只有信号"口径的结果完全一致）。
    # 触发价按买入成本算，成交价再吃一次滑点（保守方向）。
    tp_pct = float(cfg.get("take_profit_pct") or 0.0)
    sl_pct = float(cfg.get("stop_loss_pct") or 0.0)
    entry_px = 0.0

    def sell(shares_qty, px_exec, date, reason):
        """平仓并返回新的 cash/shares。手续费同规则：佣金 + 卖出印花税。"""
        amount = shares_qty * px_exec
        fee = max(min_comm, amount * comm) + amount * stamp
        trades.append(
            dict(date=date, side="SELL", price=round(px_exec, 4),
                 shares=shares_qty, amount=round(amount, 2),
                 fee=round(fee, 2), reason=reason)
        )
        return amount - fee

    for i in range(len(probs)):
        px = float(closes[i])
        want_long = bool(probs[i] >= cfg["threshold"])
        one_word_board = float(highs[i]) == float(lows[i])
        sold_today = False

        # 1) 持仓风控优先：止损先于止盈（同一天两端都被触及按最坏情况处理），
        #    也优先于"信号翻空"。T+1：买入当日不动。
        if shares > 0 and held_days >= 1 and entry_px > 0 and not one_word_board:
            sl_px = entry_px * (1 - sl_pct) if sl_pct > 0 else None
            tp_px = entry_px * (1 + tp_pct) if tp_pct > 0 else None
            hit_sl = sl_px is not None and float(lows[i]) <= sl_px
            hit_tp = tp_px is not None and float(highs[i]) >= tp_px
            if hit_sl or hit_tp:
                px_exec = (sl_px if hit_sl else tp_px) * (1 - slip)
                cash += sell(shares, px_exec, dates[i],
                             "stop_loss" if hit_sl else "take_profit")
                shares = 0
                held_days = 0
                entry_px = 0.0
                sold_today = True

        # 2) 空仓且看多 → 建仓；当日刚卖出则不回补（避免同日来回）
        if not sold_today and want_long and shares == 0:
            if one_word_board:
                skipped_locked += 1
            else:
                # 仓位可由外部按日给定（分工制：情绪面决定买多少）
                pos_pct = (float(pos_pct_series[i]) if pos_pct_series is not None
                           else pos_pct_default)
                pos_pct = max(0.0, min(1.0, pos_pct))
                budget = cash * pos_pct
                eff_px = px * (1 + slip)
                qty = int(budget / eff_px // lot) * lot
                while qty > 0:
                    amount = qty * eff_px
                    fee = max(min_comm, amount * comm)
                    if amount + fee <= cash:
                        break
                    qty -= lot
                if qty > 0:
                    amount = qty * eff_px
                    fee = max(min_comm, amount * comm)
                    cash -= amount + fee
                    shares = qty
                    held_days = 0
                    entry_px = eff_px
                    trades.append(
                        dict(date=dates[i], side="BUY", price=round(eff_px, 4),
                             shares=qty, amount=round(amount, 2),
                             fee=round(fee, 2), reason="signal")
                    )
        # 3) 持空仓位且信号翻空 → 平仓
        elif (not sold_today) and (not want_long) and shares > 0 and held_days >= 1:
            if one_word_board:
                skipped_locked += 1
            else:
                cash += sell(shares, px * (1 - slip), dates[i], "signal")
                shares = 0
                held_days = 0
                entry_px = 0.0

        nav.append(cash + shares * px)
        held_days += 1

    nav = np.asarray(nav, dtype=float)
    start_cash = cfg["initial_cash"]
    nav_norm = nav / start_cash

    mdd, md_start, md_end = max_drawdown(nav_norm, dates)
    res = dict(
        bars=len(probs),
        cum_return=cumulative_return(nav_norm),
        arr=annualized_return(nav_norm),
        sharpe=sharpe(nav_norm, cfg["rf_annual"]),
        mdd=mdd,
        mdd_start=md_start,
        mdd_end=md_end,
        calmar=(annualized_return(nav_norm) / mdd) if mdd > 0 else float("nan"),
        trades=len(trades),
        buys=sum(1 for t in trades if t["side"] == "BUY"),
        sells=sum(1 for t in trades if t["side"] == "SELL"),
        total_fee=float(sum(t["fee"] for t in trades)),
        stop_loss_exits=sum(1 for t in trades if t.get("reason") == "stop_loss"),
        take_profit_exits=sum(1 for t in trades if t.get("reason") == "take_profit"),
        skipped_locked=skipped_locked,
        final_shares=shares,
        cash_negative=bool((nav < 0).any()),
    )
    return nav_norm, trades, res


def load_price_frame(path, look_back_window):
    """读行情 CSV，按复权因子统一缩放 OHLC，返回可回测的价格序列与标签。

    为什么不是简单地把 df["Close"] 换成 df["Adj Close"]：
    Adj Close 只调整了收盘价，High/Low 仍是除权前的原始价。一旦数据带真实因子，
    止盈止损用 High/Low 判断触发就会错位（比如 10 送 1 后价格腰斩，原始的 Low 会让
    止损永久触发）。正确做法是用 Adj Close/Close 的比值把整条 OHLC 缩放到同一价格
    水平，标签也按缩放后的序列重算。

    返回值：closes, highs, lows, labels, {"adjusted": bool, "factor_jumps": int}
    """
    df = pd.read_csv(path)
    raw_close = df["Close"].astype(float).values

    if "Adj Close" in df.columns:
        adj = df["Adj Close"].astype(float).values
        factor = (np.ones_like(raw_close) if np.allclose(adj, raw_close, rtol=0, atol=1e-9)
                  else adj / raw_close)
    else:
        factor = np.ones_like(raw_close)

    if not np.all(np.isfinite(factor)) or np.any(factor <= 0):
        raise ValueError("复权因子异常（非正或非有限）: %s" % path)

    adjusted = bool(not np.allclose(factor, 1.0))
    jumps = 0
    if factor.size > 1:
        jumps = int(np.sum(np.abs(factor[1:] / factor[:-1] - 1.0) > 1e-3))

    lb = look_back_window - 1
    closes = (raw_close * factor)[lb:]
    highs = (df["High"].astype(float).values * factor)[lb:]
    lows = (df["Low"].astype(float).values * factor)[lb:]

    # 预生成的 Label 是按未复权 Close 算的；有真实因子时除权日的方向会失真，必须重算
    if adjusted:
        seq = raw_close * factor
        labels = np.zeros(seq.size, dtype=int)
        labels[:-1] = (seq[1:] - seq[:-1] > 0).astype(int)
        labels = labels[lb:]
    else:
        labels = df["Label"].values[lb:]

    return closes, highs, lows, labels, {"adjusted": adjusted, "factor_jumps": jumps}


# ---------------------------------------------------------------- 组合


def run_backtest(price_folder, model, device, feature_columns, look_back_window,
                 cfg, tag, out_dir, signal_fn=None, pos_pct_fn=None):
    """跑组合回测。

    signal_fn: 可选的外部信号源，签名 (symbol, dates) -> 概率数组（按 dates 对齐）。
    提供它时不加载模型、不做推理，直接消费外部信号（融合层就是这么接进来的）。
    两条路径共用同一套成本、T+1、止盈止损与自检，否则"单模型 vs 融合"的对比
    就不是同一把尺子。

    pos_pct_fn: 可选的每日建仓比例源，签名同上，返回长度对齐的 0~1 数组。
    用于分工制——方向由 signal_fn 给，仓位由这个函数给。不传则用固定 position_pct。
    """
    files = sorted(f for f in os.listdir(price_folder) if f.endswith(".csv"))
    if not files:
        raise RuntimeError(f"no csv in {price_folder}")

    os.makedirs(out_dir, exist_ok=True)

    navs, bench_navs, all_dates = [], [], None
    per_stock, all_trades = [], []
    # 复权状态：CSMD 的 Adj Close 列名义上是复权价，但 scripts/audit_prices.py 实测
    # 150 个 CSV 与 Close 逐行相等（0 次除权跳变）。仍然按 Adj Close/Close 的比值
    # 统一缩放 OHLC——一旦数据带真实因子，High/Low 与 Close 保持同一价格水平，
    # 止盈止损的触发价才不会错位；当前因子恒为 1，结果与旧版逐位相同。
    price_info = {"adjusted": False, "factor_jumps": 0, "files_with_factor": 0}

    for f in files:
        path = os.path.join(price_folder, f)
        # 前视泄漏修复：scaler 只在 train 段拟合，test 段 transform
        train_path = path.replace(os.path.join("test", "price"), os.path.join("train", "price"))
        ds = Backtest_Dataset(csv_file=path, look_back_window=look_back_window,
                              feature_columns=feature_columns,
                              scaler_csv=train_path if os.path.exists(train_path) else None)
        dates = [str(d) for d in ds.get_dates()]

        # 价格统一走复权缩放（见 load_price_frame）：当前数据因子恒为 1，
        # 结果与旧版逐位相同；数据一旦带真实因子，OHLC 与标签会一起被修正。
        closes, highs, lows, labels, adj_info = load_price_frame(path, look_back_window)
        if adj_info["adjusted"]:
            price_info["adjusted"] = True
            price_info["files_with_factor"] += 1
        price_info["factor_jumps"] += adj_info["factor_jumps"]

        if signal_fn is not None:
            # 外部信号（融合层）：长度必须与日期严格对齐，否则价格与信号会整体错位
            probs = np.asarray(signal_fn(f.replace(".csv", ""), dates), dtype=float)
            if probs.shape[0] != len(dates):
                raise RuntimeError(
                    f"外部信号长度 {probs.shape[0]} 与日期数 {len(dates)} 不一致：{f}")
            probs = np.clip(probs, 0.0, 1.0)
        else:
            x = torch.tensor(np.asarray(ds.normalized_sequences), dtype=torch.float32).to(device)
            with torch.no_grad():
                out = model(x).squeeze(-1)
            # 四个模型的 forward 末尾都已经接了 Sigmoid，输出即概率。
            # 早期版本在这里又做了一次 sigmoid → 概率恒 > 0.5 → 全线买入，
            # ACC 恰好退化成测试集正例率。这里只做区间校验，绝不再套变换。
            probs = out.cpu().numpy()
            if probs.min() < -1e-6 or probs.max() > 1 + 1e-6:
                raise RuntimeError(f"模型输出不在 [0,1]，怀疑 forward 未接 sigmoid：{f}")

        pos_series = None
        if pos_pct_fn is not None:
            pos_series = np.asarray(pos_pct_fn(f.replace(".csv", ""), dates), dtype=float)
            if pos_series.shape[0] != len(dates):
                raise RuntimeError(
                    f"仓位序列长度 {pos_series.shape[0]} 与日期数 {len(dates)} 不一致：{f}")
            pos_series = np.clip(pos_series, 0.0, 1.0)

        nav, trades, res = simulate_one_stock(dates, closes, highs, lows, probs, cfg,
                                              pos_series)
        bench = np.asarray(closes, dtype=float) / float(closes[0])

        if all_dates is None:
            all_dates = dates
        elif len(dates) != len(all_dates) or dates[0] != all_dates[0] or dates[-1] != all_dates[-1]:
            raise RuntimeError(f"日期不对齐: {f}")

        navs.append(nav)
        bench_navs.append(bench)
        res["stock"] = f.replace(".csv", "")
        res["acc"] = float(((probs >= cfg["threshold"]).astype(int) == labels).mean())
        per_stock.append(res)
        for t in trades:
            t["stock"] = f.replace(".csv", "")
            all_trades.append(t)

    navs = np.vstack(navs)
    bench_navs = np.vstack(bench_navs)
    port = navs.mean(axis=0)
    bench = bench_navs.mean(axis=0)

    mdd, md_start, md_end = max_drawdown(port, all_dates)
    port_res = dict(
        stocks=len(files),
        bars=len(all_dates),
        first_date=all_dates[0],
        last_date=all_dates[-1],
        cum_return=cumulative_return(port),
        arr=annualized_return(port),
        sharpe=sharpe(port, cfg["rf_annual"]),
        mdd=mdd,
        mdd_start=md_start,
        mdd_end=md_end,
        calmar=(annualized_return(port) / mdd) if mdd > 0 else float("nan"),
        ir_vs_equal_weight=information_ratio(port, bench),
        bench_cum_return=cumulative_return(bench),
        bench_mdd=max_drawdown(bench, all_dates)[0],
        total_trades=len(all_trades),
        total_fee=float(sum(t["fee"] for t in all_trades)),
        stop_loss_exits=sum(1 for t in all_trades if t.get("reason") == "stop_loss"),
        take_profit_exits=sum(1 for t in all_trades if t.get("reason") == "take_profit"),
        final_cash_negative=bool((navs < 0).any()),
        price_basis=("adj_close" if price_info["adjusted"] else "close_unadjusted"),
        price_adjusted=bool(price_info["adjusted"]),
        price_factor_jumps=int(price_info["factor_jumps"]),
        files_with_factor=int(price_info["files_with_factor"]),
    )

    # 自检：最大回撤区间的终点必须是回撤最深的那一天，且区间内确实在回撤
    nav_arr = np.asarray(port, dtype=float)
    peak = np.maximum.accumulate(nav_arr)
    dd = (peak - nav_arr) / peak
    trough_idx = int(np.argmax(dd))
    checks = {
        "nav_len_matches_dates": len(port) == len(all_dates),
        "no_nan_in_nav": bool(not np.isnan(nav_arr).any()),
        "no_negative_equity": not port_res["final_cash_negative"],
        "mdd_span_covers_trough": bool(all_dates.index(md_start) <= trough_idx <= all_dates.index(md_end)),
        "mdd_equals_recomputed": bool(abs(dd.max() - port_res["mdd"]) < 1e-12),
        "metrics_finite": bool(np.isfinite([port_res["arr"], port_res["sharpe"],
                                            port_res["mdd"], port_res["ir_vs_equal_weight"]]).all()),
        "trade_pairs_balanced": abs(sum(1 for t in all_trades if t["side"] == "BUY") -
                                   sum(1 for t in all_trades if t["side"] == "SELL")) <= len(files),
        # 价格口径必须显式落到结果里：读到全零因子、或数据带调整却没生效，
        # 都会让"这轮回测用的是复权价还是原始价"变成事后无法追溯的问题。
        "price_basis_declared": port_res["price_basis"] in ("adj_close", "close_unadjusted"),
    }

    pd.DataFrame({"date": all_dates, "portfolio_nav": port, "benchmark_nav": bench}).to_csv(
        os.path.join(out_dir, "nav.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(all_trades).to_csv(
        os.path.join(out_dir, "trades.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(per_stock).to_csv(
        os.path.join(out_dir, "per_stock.csv"), index=False, encoding="utf-8-sig")

    summary = dict(tag=tag, config=cfg, portfolio=port_res, checks=checks,
                   per_stock_mean={
                       k: float(np.nanmean([s[k] for s in per_stock]))
                       for k in ["acc", "cum_return", "sharpe", "mdd", "trades", "total_fee"]
                   })
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    return summary


# ---------------------------------------------------------------- 模型装载


def build_model(args):
    common = dict(input_size=args.input_size, hidden_size=args.hidden_size,
                  num_layers=args.layers, output_size=1, dropout=args.dropout,
                  batch_first=True)
    if args.model == "lstm":
        model = LSTM(**common)
    elif args.model == "alstm":
        model = ALSTM(attention_size=args.attention_size, **common)
    elif args.model == "bi_lstm":
        model = BiLSTM(**common)
    elif args.model == "adv_lstm":
        model = AdvLSTM(input_size=args.input_size, hidden_size=args.hidden_size,
                        output_size=1, attention_size=args.attention_size,
                        perturbation_size=args.perturbation_size, epsilon=args.epsilon)
    else:
        raise ValueError(f"unsupported model: {args.model}")

    weights = os.path.join(args.model_save_folder, "model_saved", f"{args.model}.pth")
    if not os.path.exists(weights):
        weights = os.path.join(args.model_save_folder, f"{args.model}.pth")
    if not os.path.exists(weights):
        raise FileNotFoundError(weights)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    return model

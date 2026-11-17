# -*- coding: utf-8 -*-
"""回测引擎的单元测试。

覆盖三类风险：
1. 指标公式——回撤区间、年化、夏普分母为 0 的退化情况
2. 交易规则——T+1、最低佣金、滑点方向、止盈止损、一字板不可成交
3. 价格口径——Adj Close 与 Close 相等时必须退化为因子 1（保证历史结果可复现），
   有真实因子时必须把整条 OHLC 一起缩放，而不是只换收盘价
"""
import os

import numpy as np
import pandas as pd
import pytest

from engine_v2 import (BAR_PER_YEAR, annualized_return, cumulative_return,
                       information_ratio, load_price_frame, max_drawdown, sharpe,
                       simulate_one_stock)

CFG = dict(initial_cash=100000.0, position_pct=0.95, lot_size=1,
           commission_rate=0.0003, min_commission=5.0, stamp_duty_rate=0.001,
           slippage_rate=0.0005, threshold=0.5, rf_annual=0.02,
           take_profit_pct=0.0, stop_loss_pct=0.0)


def _run(dates, closes, highs=None, lows=None, probs=None, **kw):
    """跑单标的模拟。默认 high/low 各偏离收盘 0.01，避免被误判成一字板。"""
    cfg = dict(CFG)
    cfg.update(kw)
    highs = [c + 0.01 for c in closes] if highs is None else highs
    lows = [c - 0.01 for c in closes] if lows is None else lows
    probs = [1.0] * len(closes) if probs is None else probs
    return simulate_one_stock(list(dates), list(closes), list(highs), list(lows),
                              list(probs), cfg)


# ---------------------------------------------------------------- 指标公式


def test_max_drawdown_returns_span():
    nav = [1.0, 1.2, 0.9, 1.0]
    dates = ["d1", "d2", "d3", "d4"]
    mdd, start, end = max_drawdown(nav, dates)
    assert mdd == pytest.approx(0.25)       # (1.2 - 0.9) / 1.2
    assert start == "d2"
    assert end == "d3"


def test_max_drawdown_monotonic_rise_is_zero():
    mdd, start, end = max_drawdown([1.0, 1.1, 1.2], ["a", "b", "c"])
    assert mdd == 0.0


def test_annualized_return_single_bar():
    assert annualized_return([1.0, 1.1]) == pytest.approx(1.1 ** BAR_PER_YEAR - 1)


def test_annualized_return_degenerate_input():
    assert np.isnan(annualized_return([1.0]))


def test_sharpe_constant_nav_is_nan():
    # 净值完全不动 → 标准差 0 → 不应该是 inf 或 0
    assert np.isnan(sharpe([1.0] * 5, 0.02))


def test_cumulative_return():
    assert cumulative_return([1.0, 1.5, 2.0]) == pytest.approx(1.0)


def test_information_ratio_against_itself_is_nan():
    nav = [1.0, 1.1, 1.2, 1.05]
    assert np.isnan(information_ratio(nav, nav))


# ---------------------------------------------------------------- 交易规则


def test_buy_price_includes_slippage_and_fee():
    nav, trades, res = _run(["d1"], [10.0])
    assert len(trades) == 1 and trades[0]["side"] == "BUY"
    assert trades[0]["price"] == pytest.approx(10.0 * (1 + CFG["slippage_rate"]))
    # 建仓金额约 95000（10 万的 95%），佣金 = 95000 × 0.0003 ≈ 28.5，
    # 高于 5 元最低线，所以这里按比例收费而不是最低值
    assert trades[0]["fee"] == pytest.approx(28.5, rel=1e-3)


def test_t_plus_1_blocks_same_day_exit():
    # 第 1 天建仓，当天不允许卖；第 2 天信号翻空才卖
    nav, trades, res = _run(["d1", "d2"], [10.0, 10.0], probs=[1.0, 0.0])
    assert [t["side"] for t in trades] == ["BUY", "SELL"]
    assert trades[1]["date"] == "d2"


def test_held_position_marks_to_market():
    nav, trades, res = _run(["d1", "d2"], [10.0, 11.0])
    # 只建仓不卖出，净值应随价格上行
    assert trades and trades[0]["side"] == "BUY"
    assert nav[-1] > nav[0]


def test_stop_loss_exits_on_low_breach():
    # 买入价 ≈10.005，-5% 线 ≈9.505，第 2 天最低 8.8 必然击穿
    nav, trades, res = _run(["d1", "d2"], [10.0, 9.0],
                            highs=[10.01, 9.2], lows=[9.99, 8.8], stop_loss_pct=0.05)
    assert res["stop_loss_exits"] == 1
    assert trades[-1]["reason"] == "stop_loss"


def test_take_profit_exits_on_high_breach():
    # 买入价 ≈10.005，+8% 线 ≈10.805，第 2 天最高 11.5 必然触发
    nav, trades, res = _run(["d1", "d2"], [10.0, 11.0],
                            highs=[10.01, 11.5], lows=[9.99, 10.5], take_profit_pct=0.08)
    assert res["take_profit_exits"] == 1
    assert trades[-1]["reason"] == "take_profit"


def test_one_word_board_blocks_entry():
    # high == low 视为一字板（不可成交的代理），不建仓
    nav, trades, res = _run(["d1"], [10.0], highs=[10.0], lows=[10.0])
    assert trades == []
    assert res["skipped_locked"] == 1


def test_hold_signal_leaves_cash_untouched():
    nav, trades, res = _run(["d1", "d2"], [10.0, 10.0], probs=[0.0, 0.0])
    assert trades == []
    assert nav[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------- 价格口径


def _write_price_csv(tmp_path, closes, adj_closes, high=None, low=None, labels=None):
    n = len(closes)
    high = [c + 1.0 for c in closes] if high is None else high
    low = [c - 1.0 for c in closes] if low is None else low
    labels = [i % 2 for i in range(n)] if labels is None else labels
    df = pd.DataFrame({
        "Date": ["2024-01-%02d" % (i + 1) for i in range(n)],
        "Open": closes,
        "High": high,
        "Low": low,
        "Close": closes,
        "Adj Close": adj_closes,
        "Volume": [100] * n,
        "Label": labels,
    })
    p = tmp_path / "测试股.csv"
    df.to_csv(p, index=False)
    return str(p)


def test_identical_adj_close_degrades_to_factor_one(tmp_path):
    """Adj Close 与 Close 相等时必须是因子 1：历史结果才能逐位复现。"""
    closes = [10.5, 11.5, 12.5, 13.5, 14.5]
    p = _write_price_csv(tmp_path, closes, list(closes))
    c, h, lo, labels, info = load_price_frame(p, look_back_window=2)

    assert info["adjusted"] is False
    assert info["factor_jumps"] == 0
    assert c[0] == pytest.approx(closes[1])
    assert h[0] == pytest.approx(closes[1] + 1.0)
    assert lo[0] == pytest.approx(closes[1] - 1.0)
    # 未调整时沿用预生成的 Label（默认 [0,1,0,1,0]，窗口 2 吃掉首行）
    assert list(labels) == [1, 0, 1, 0]


def test_real_factor_scales_close_high_and_low_together(tmp_path):
    """有真实因子时，High/Low 必须与 Close 一起缩放，不能只换收盘价。"""
    closes = [20.0, 22.0, 24.0, 26.0]
    adj = [c / 2.0 for c in closes]          # 模拟 1 拆 2 的除权
    p = _write_price_csv(tmp_path, closes, adj,
                         high=[c + 2.0 for c in closes],
                         low=[c - 2.0 for c in closes])
    c, h, lo, labels, info = load_price_frame(p, look_back_window=1)

    assert info["adjusted"] is True
    assert c[0] == pytest.approx(closes[0] / 2.0)
    assert h[0] == pytest.approx((closes[0] + 2.0) / 2.0)
    assert lo[0] == pytest.approx((closes[0] - 2.0) / 2.0)
    # 缩放后 OHLC 仍在同一价格水平：high > close > low
    assert h[0] > c[0] > lo[0]


def test_factor_jump_is_counted(tmp_path):
    """因子突变应被计数，用于回答"这数据有没有除权"。"""
    closes = [10.0, 10.2, 10.4, 10.6]
    adj = [10.0, 10.2, 5.2, 5.3]             # 第 3 天因子腰斩
    p = _write_price_csv(tmp_path, closes, adj)
    c, h, lo, labels, info = load_price_frame(p, look_back_window=1)
    assert info["adjusted"] is True
    assert info["factor_jumps"] >= 1


def test_non_positive_factor_raises(tmp_path):
    closes = [10.0, 10.2, 10.4]
    adj = [10.0, 0.0, 10.4]                  # 零因子必须报错，不能静默算出 inf
    p = _write_price_csv(tmp_path, closes, adj)
    with pytest.raises(ValueError):
        load_price_frame(p, look_back_window=1)


def test_missing_adj_column_falls_back_to_close(tmp_path):
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "Open": [10.0, 10.5], "High": [11.0, 11.5], "Low": [9.0, 9.5],
        "Close": [10.5, 11.0], "Volume": [100, 120], "Label": [1, 0],
    })
    p = tmp_path / "无复权列.csv"
    df.to_csv(p, index=False)
    c, h, lo, labels, info = load_price_frame(str(p), look_back_window=1)
    assert info["adjusted"] is False
    assert c[0] == pytest.approx(10.5)

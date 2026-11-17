# -*- coding: utf-8 -*-
"""截面引擎（engine_cross_section.py）的单元测试。

这 265 行是答辩亮点策略（53.80% / SR 3.676）的唯一执行路径，此前零测试。
测的都是"错了不会报错、只会给一个好看到的假数字"的地方：

- 分数用当日还是上一日 -> 用当日就是未来函数
- 调仓时旧仓有没有清空 -> 不清空就等于越买越重
- 复权因子只调 Close 不调 Open -> 成交金额与净值脱节，手续费也算错
- 日期不对齐 / 因子异常 -> 必须报错，不能静默跑出结果
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backtest"))

import engine_cross_section as ecs  # noqa: E402

DATES = [
    "2024-11-01", "2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07",
    "2024-11-08", "2024-11-11", "2024-11-12", "2024-11-13", "2024-11-14",
    "2024-11-15", "2024-11-18", "2024-11-19", "2024-11-20", "2024-11-21",
    "2024-11-22", "2024-11-25", "2024-11-26", "2024-11-27", "2024-11-28",
    "2024-11-29",
]
SYMS = ["A", "B", "C", "D", "E"]


def _write_prices(folder, n_sym=5, prices=None, adj_factor=None, misalign=False):
    """写临时价格 csv。prices=None 时用 50 + i*0.5 的稳定序列。"""
    for i, s in enumerate(SYMS[:n_sym]):
        if prices is None:
            closes = [50.0 + i * 0.5 + k * 0.05 for k in range(len(DATES))]
        else:
            closes = prices[s]
        dates = DATES if not misalign else DATES[:-1] + ["2024-11-30"]
        df = pd.DataFrame({
            "Date": dates,
            "Open": [c * 0.998 for c in closes],
            "Close": closes,
        })
        if adj_factor is not None:
            df["Adj Close"] = [c * adj_factor for c in closes]
        df.to_csv(os.path.join(folder, s + ".csv"), index=False,
                  encoding="utf-8-sig")


CFG = dict(
    top_n=2, hold_days=5, initial_cash=100000.0,
    commission_rate=0.0003, min_commission=5.0,
    stamp_duty_rate=0.001, slippage_rate=0.002,
    lot_size=1, rf_annual=0.02,
)


def _score_df(values, syms=None):
    """values: {date: {sym: score}}，没给的日期整行为 NaN。"""
    syms = syms or SYMS
    df = pd.DataFrame(index=DATES, columns=syms, dtype=float)
    for d, row in values.items():
        for s, v in row.items():
            df.loc[d, s] = v
    return df


@pytest.fixture
def tmp_price(tmp_path):
    _write_prices(str(tmp_path))
    return str(tmp_path)


# ---------- _load_panel：读盘契约 ----------

def test_load_panel_empty_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no csv"):
        ecs._load_panel(str(tmp_path))


def test_load_panel_misaligned_dates_raises(tmp_path):
    # 任一只日期序列不同就必须报错。静默对齐会让某天缺价的股票在组合里"消失"
    _write_prices(str(tmp_path), n_sym=2)
    _write_prices(str(tmp_path), n_sym=1, misalign=True)  # 覆盖 A 的日期
    with pytest.raises(RuntimeError, match="日期不对齐"):
        ecs._load_panel(str(tmp_path))


def test_load_panel_applies_factor_to_open_too(tmp_path):
    # 只调 Close 不调 Open 是这里最隐蔽的错：成交价用未复权的开盘价，
    # 而净值用复权后的收盘价估值，两边不是一个价格尺度。
    _write_prices(str(tmp_path), n_sym=2, adj_factor=0.5)
    _, opens, closes, syms, adjusted = ecs._load_panel(str(tmp_path))
    assert adjusted is True
    raw = [50.0 + 0 * 0.5 + k * 0.05 for k in range(len(DATES))]
    assert opens["A"][DATES[0]] == pytest.approx(raw[0] * 0.998 * 0.5)
    assert closes["A"][DATES[0]] == pytest.approx(raw[0] * 0.5)


def test_load_panel_rejects_bad_factor(tmp_path):
    # 因子为负会让"开盘价 × 股数"变成负数，必须拦下
    _write_prices(str(tmp_path), n_sym=1, adj_factor=-1.0)
    with pytest.raises(RuntimeError, match="复权因子异常"):
        ecs._load_panel(str(tmp_path))


# ---------- 时点：分数来自上一交易日 ----------

def test_score_used_from_previous_day(tmp_path):
    # 只在 i=4（2024-11-07）给分，hold=5 时调仓发生在 i=5（11-08），
    # 必须读 11-07 的分。若读当日（11-08）则一条都选不出来。
    _write_prices(str(tmp_path))
    sc = _score_df({"2024-11-07": {"A": 1.0, "B": 0.9, "C": 0.1, "D": 0.0, "E": -1.0}})
    out = tempfile.mkdtemp()
    r = ecs.run_cross_section(str(tmp_path), sc, CFG, "t", out,
                              DATES[0], DATES[-1])
    assert r["portfolio"]["rebalance_count"] >= 1
    buys = [t for t in pd.read_csv(os.path.join(out, "trades.csv")).to_dict("records")
            if t["side"] == "BUY"]
    assert buys, "上一日有分数却没买入 = 时点读错了"
    assert buys[0]["date"] == "2024-11-08"


def test_no_scores_means_no_rebalance(tmp_path):
    # 全 NaN：不应建仓，但净值序列仍要完整（全等于 1）
    _write_prices(str(tmp_path))
    sc = _score_df({})
    out = tempfile.mkdtemp()
    r = ecs.run_cross_section(str(tmp_path), sc, CFG, "t", out,
                              DATES[0], DATES[-1])
    assert r["portfolio"]["rebalance_count"] == 0
    assert r["portfolio"]["total_trades"] == 0
    nav = pd.read_csv(os.path.join(out, "nav.csv"))
    assert len(nav) == len(DATES)
    assert abs(nav["portfolio_nav"].iloc[0] - 1.0) < 1e-12


# ---------- 调仓：清空旧仓、只留 top_n ----------

def test_rebalance_clears_previous_holdings(tmp_path):
    # 第一次买 A/B，第二次换成 D/E，A/B 必须先全部卖出。
    # 不清空会让持仓越滚越多，仓位远超 100%，净值曲线好看但不可实现。
    _write_prices(str(tmp_path))
    sc = _score_df({
        "2024-11-07": {"A": 1.0, "B": 0.9, "C": 0.1, "D": 0.0, "E": -1.0},
        "2024-11-14": {"A": -1.0, "B": -0.9, "C": 0.1, "D": 0.9, "E": 1.0},
    })
    out = tempfile.mkdtemp()
    ecs.run_cross_section(str(tmp_path), sc, CFG, "t", out, DATES[0], DATES[-1])
    tr = pd.read_csv(os.path.join(out, "trades.csv")).to_dict("records")

    first_buy = sorted([t["stock"] for t in tr if t["side"] == "BUY" and t["date"] == "2024-11-08"])
    assert len(first_buy) <= CFG["top_n"]
    # 第二个调仓日（i=10，11-15）必须把上一轮持仓卖掉
    sold = [t["stock"] for t in tr if t["side"] == "SELL" and t["date"] == "2024-11-15"]
    for s in first_buy:
        assert s in sold, "旧持仓没清空：%s" % s


def test_top_n_limits_positions(tmp_path):
    _write_prices(str(tmp_path))
    sc = _score_df({"2024-11-07": {s: 1.0 - 0.1 * i for i, s in enumerate(SYMS)}})
    out = tempfile.mkdtemp()
    ecs.run_cross_section(str(tmp_path), sc, CFG, "t", out, DATES[0], DATES[-1])
    tr = pd.read_csv(os.path.join(out, "trades.csv")).to_dict("records")
    held = {}
    for t in tr:
        held[t["stock"]] = held.get(t["stock"], 0) + (t["shares"] if t["side"] == "BUY" else -t["shares"])
    assert len([s for s, q in held.items() if q > 0]) <= CFG["top_n"]


# ---------- 组合口径与产出 ----------

def test_nav_length_and_no_negative_cash(tmp_path):
    _write_prices(str(tmp_path))
    sc = _score_df({"2024-11-07": {"A": 1.0, "B": 0.9, "C": 0.1, "D": 0.0, "E": -1.0}})
    out = tempfile.mkdtemp()
    r = ecs.run_cross_section(str(tmp_path), sc, CFG, "t", out, DATES[0], DATES[-1])
    assert r["portfolio"]["bars"] == len(DATES)
    assert r["checks"]["nav_len_matches_dates"] is True
    assert r["checks"]["no_negative_equity"] is True
    assert r["portfolio"]["final_cash_negative"] is False
    assert r["portfolio"]["stop_loss_exits"] == 0
    assert r["portfolio"]["take_profit_exits"] == 0


def test_summary_is_valid_json_without_nan(tmp_path):
    # 写 NaN 会让前端 JSON.parse 整页挂掉，这是踩过的坑
    _write_prices(str(tmp_path))
    sc = _score_df({"2024-11-07": {"A": 1.0, "B": 0.9, "C": 0.1, "D": 0.0, "E": -1.0}})
    out = tempfile.mkdtemp()
    ecs.run_cross_section(str(tmp_path), sc, CFG, "t", out, DATES[0], DATES[-1])
    with open(os.path.join(out, "summary.json"), encoding="utf-8") as f:
        txt = f.read()
    assert "NaN" not in txt
    s = json.loads(txt)
    assert s["portfolio"]["strategy_type"] == "cross_section"
    assert s["checks"]["metrics_finite"] is True


def test_robustness_trimming_reduces_return(tmp_path):
    # 去掉最好的一段后，累计收益必须 <= 全段。反了就是稳健性算法写错了
    _write_prices(str(tmp_path))
    sc = _score_df({"2024-11-07": {"A": 1.0, "B": 0.9, "C": 0.1, "D": 0.0, "E": -1.0}})
    out = tempfile.mkdtemp()
    r = ecs.run_cross_section(str(tmp_path), sc, CFG, "t", out, DATES[0], DATES[-1])
    rb = r["portfolio"]["robustness"]
    if rb["period_count"] >= 2:
        assert rb["cum_drop_best_1"] <= rb["cum_all"] + 1e-12
        assert rb["cum_drop_best_2"] <= rb["cum_drop_best_1"] + 1e-12


def test_range_too_short_raises(tmp_path):
    # 区间内只有 1 天时无法建组合，必须报错而不是返回一条平的净值
    _write_prices(str(tmp_path))
    sc = _score_df({})
    with pytest.raises(RuntimeError, match="交易日不足"):
        ecs.run_cross_section(str(tmp_path), sc, CFG, "t", tempfile.mkdtemp(),
                              "2024-11-29", "2024-11-29")

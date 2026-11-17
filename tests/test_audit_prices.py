# -*- coding: utf-8 -*-
"""复权审计脚本的单元测试。

审计结论"数据未复权"是写进文档的证据，它本身必须可复现、可回归：
这个测试用合成数据验证"相等 / 有调整 / 有跳变 / 缺列"四种情形都能被正确判定。
"""
import pandas as pd
import pytest

from audit_prices import scan_file


def _write(tmp_path, closes, adj=None, name="测试股.csv"):
    n = len(closes)
    data = {
        "Date": ["2024-01-%02d" % (i + 1) for i in range(n)],
        "Open": closes,
        "High": [c + 1.0 for c in closes],
        "Low": [c - 1.0 for c in closes],
        "Close": closes,
        "Volume": [100] * n,
    }
    if adj is not None:
        data["Adj Close"] = adj
    p = tmp_path / name
    pd.DataFrame(data).to_csv(p, index=False)
    return str(p)


def test_identical_columns_marked_not_adjusted(tmp_path):
    closes = [10.0, 10.5, 11.0]
    rec = scan_file(_write(tmp_path, closes, list(closes)))
    assert rec["status"] == "ok"
    assert rec["identical_to_close"] is True
    assert rec["adjusted"] is False
    assert rec["jump_count"] == 0


def test_real_adjustment_detected(tmp_path):
    closes = [20.0, 21.0, 22.0]
    rec = scan_file(_write(tmp_path, closes, [c / 2.0 for c in closes]))
    assert rec["adjusted"] is True
    assert rec["factor_min"] == pytest.approx(0.5)
    assert rec["factor_max"] == pytest.approx(0.5)


def test_factor_jump_detected(tmp_path):
    closes = [10.0, 10.5, 11.0, 11.5]
    adj = [10.0, 10.5, 6.0, 6.2]      # 第 3 行因子突变，疑似除权
    rec = scan_file(_write(tmp_path, closes, adj))
    assert rec["adjusted"] is True
    assert rec["jump_count"] >= 1
    assert rec["jumps"][0]["date"] == "2024-01-03"


def test_missing_adj_column_reported(tmp_path):
    rec = scan_file(_write(tmp_path, [10.0, 10.5], adj=None))
    assert rec["status"] == "no_adj_column"
    assert rec["adjusted"] is False


def test_custom_threshold_changes_sensitivity(tmp_path):
    """0.5% 的因子波动：阈值 0.1% 时算跳变，阈值 1% 时不算。"""
    closes = [10.0, 10.1, 10.2]
    adj = [10.0, 10.1 * 1.005, 10.2]      # 第 2 行因子相对前一行变化约 0.5%
    p = _write(tmp_path, closes, adj)
    assert scan_file(p, jump_th=0.001)["jump_count"] >= 1
    assert scan_file(p, jump_th=0.01)["jump_count"] == 0

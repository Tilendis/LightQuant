"""除权防护断言的回归测试。

这组用例的目的不是"测函数能不能跑"，而是把一条事实钉住：
CSMD50 数据源未提供复权修正，train 段存在跨除权跳空，
加载层必须能发现它、并且能被配置成直接失败。
"""
import os
import tempfile

import pandas as pd
import pytest

from utils.price_dataloader import (
    ADJ_JUMP_TH,
    adj_audit_summary,
    audit_price_adjustment,
    reset_adj_audit,
)


def _write_csv(path, closes, adj=None):
    n = len(closes)
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-02", periods=n, freq="B").strftime("%Y-%m-%d"),
        "Open": closes,
        "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes],
        "Close": closes,
        "Adj Close": adj if adj is not None else list(closes),
        "Volume": [1000] * n,
    })
    df.to_csv(path, index=False, encoding="utf-8")
    return path


@pytest.fixture
def tmpdir_path():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_flat_prices_have_no_adj_and_no_jump(tmpdir_path):
    """价格平稳、Adj Close 与 Close 相同：判定为未复权、零跳空。"""
    p = _write_csv(os.path.join(tmpdir_path, "flat.csv"), [10.0, 10.2, 10.1, 10.3])
    reset_adj_audit()
    has_adj, n_jumps, max_ret = audit_price_adjustment(pd.read_csv(p), "flat.csv")
    assert has_adj is False          # Adj Close == Close -> 没有复权修正
    assert n_jumps == 0
    assert max_ret < 0.05


def test_adj_close_differs_means_adjusted(tmpdir_path):
    """Adj Close 与 Close 不同：判定为带复权修正，即便存在大跳空也不该报错。"""
    closes = [10.0, 10.2, 7.6, 7.8]          # 第三点跌 25%，人为跳空
    adj = [10.0, 10.2, 10.2, 10.4]
    p = _write_csv(os.path.join(tmpdir_path, "adj.csv"), closes, adj)
    reset_adj_audit()
    os.environ["LQ_STRICT_ADJ"] = "1"
    try:
        has_adj, n_jumps, _ = audit_price_adjustment(pd.read_csv(p), "adj.csv")
    finally:
        os.environ.pop("LQ_STRICT_ADJ", None)
    assert has_adj is True
    assert n_jumps == 1                       # 按 Close 算跳空仍会计数
    # 但 has_adj=True 时严格模式不拦：数据本身有修正，跳空是原始价噪声


def test_strict_mode_raises_on_large_jump(tmpdir_path):
    """严格模式下，未复权且有超阈值跳空必须抛 AssertionError。"""
    closes = [10.0, 12.5, 12.6, 12.8]         # 第二点跳 +25%
    p = _write_csv(os.path.join(tmpdir_path, "jump.csv"), closes)
    reset_adj_audit()
    os.environ["LQ_STRICT_ADJ"] = "1"
    try:
        with pytest.raises(AssertionError) as ei:
            audit_price_adjustment(pd.read_csv(p), "jump.csv")
    finally:
        os.environ.pop("LQ_STRICT_ADJ", None)
    assert "除权防护断言触发" in str(ei.value)
    assert "jump.csv" in str(ei.value)


def test_default_mode_reports_without_raising(tmpdir_path):
    """默认模式只报告不抛错——否则现有 train 段根本跑不动。"""
    closes = [10.0, 12.5, 12.6, 12.8]
    p = _write_csv(os.path.join(tmpdir_path, "jump2.csv"), closes)
    reset_adj_audit()
    has_adj, n_jumps, max_ret = audit_price_adjustment(pd.read_csv(p), "jump2.csv")
    assert has_adj is False
    assert n_jumps == 1
    assert max_ret > ADJ_JUMP_TH


def test_audit_accumulates_across_files(tmpdir_path):
    """累计计数应跨文件相加，供 create_dataset 汇总打印。"""
    p1 = _write_csv(os.path.join(tmpdir_path, "a.csv"), [10.0, 12.5, 12.6])
    p2 = _write_csv(os.path.join(tmpdir_path, "b.csv"), [20.0, 20.2, 25.4])
    reset_adj_audit()
    audit_price_adjustment(pd.read_csv(p1), "a.csv")
    audit_price_adjustment(pd.read_csv(p2), "b.csv")
    s = adj_audit_summary()
    assert s["files"] == 2
    assert s["files_without_adj"] == 2
    assert s["jump_rows"] == 2
    assert len(s["jump_files"]) == 2
    reset_adj_audit()
    assert adj_audit_summary()["files"] == 0


def test_real_train_file_triggers_strict_assertion():
    """真实数据：train 段已知存在除权跳空，严格模式必须能拦到。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "dataset", "CSMD50_feat", "train", "price", "中远海控.csv")
    if not os.path.exists(p):
        pytest.skip("缺少 CSMD50_feat/train/price/中远海控.csv")
    reset_adj_audit()
    os.environ["LQ_STRICT_ADJ"] = "1"
    try:
        with pytest.raises(AssertionError):
            audit_price_adjustment(pd.read_csv(p), "中远海控.csv")
    finally:
        os.environ.pop("LQ_STRICT_ADJ", None)


def test_real_test_file_passes_strict_mode():
    """真实数据：test 段无除权跳空，严格模式不应误拦——否则会误伤正常样本。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(root, "dataset", "CSMD50_feat", "test", "price", "中远海控.csv")
    if not os.path.exists(p):
        pytest.skip("缺少 CSMD50_feat/test/price/中远海控.csv")
    reset_adj_audit()
    os.environ["LQ_STRICT_ADJ"] = "1"
    try:
        has_adj, n_jumps, _ = audit_price_adjustment(pd.read_csv(p), "test/中远海控.csv")
    finally:
        os.environ.pop("LQ_STRICT_ADJ", None)
    assert n_jumps == 0
    assert has_adj is False

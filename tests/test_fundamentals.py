# -*- coding: utf-8 -*-
"""财务数据层测试：报告期换算、毛利率兜底、时点口径、同比取值。

这三件事都会静默出错，所以必须测：
- period_of 算错 -> announce_date 算错 -> 未来函数
- 毛利率不兜底 -> 整列缺失（实测曾经 100% 为空）
- 同比取"上一期"而不是"同报告期上一年" -> Q1 对 H1，比出来不是增长
"""
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fetch_fundamentals import period_of, gross_margin_of  # noqa: E402
from gen_agents import get_fundamentals  # noqa: E402

DDL = """
CREATE TABLE fundamentals (
    symbol TEXT, period TEXT, announce_date TEXT,
    pe REAL, pb REAL, roe REAL, gross_margin REAL,
    revenue REAL, net_profit REAL, revenue_yoy REAL, profit_yoy REAL,
    debt_ratio REAL, ocf_ps REAL, current_ratio REAL, inventory_turn REAL,
    source TEXT, fetched_at TEXT,
    PRIMARY KEY (symbol, period)
)
"""


def make_db(rows):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(DDL)
    cols = ("symbol", "period", "announce_date", "roe", "gross_margin",
            "revenue_yoy", "profit_yoy", "debt_ratio", "ocf_ps",
            "current_ratio", "inventory_turn")
    for r in rows:
        conn.execute(
            "INSERT INTO fundamentals (%s) VALUES (%s)"
            % (",".join(cols), ",".join("?" * len(cols))),
            [r.get(c) for c in cols])
    conn.commit()
    return conn


# ---------- period_of：报告期 -> 法定披露截止日 ----------

@pytest.mark.parametrize("raw,period,announce", [
    ("2024-03-31", "2024Q1", "2024-04-30"),
    ("2024-06-30", "2024H1", "2024-08-31"),
    ("2024-09-30", "2024Q3", "2024-10-31"),
    ("2024-12-31", "2024FY", "2025-04-30"),
    ("2023-12-31", "2023FY", "2024-04-30"),
])
def test_period_of_quarter_ends(raw, period, announce):
    assert period_of(raw) == (period, announce)


def test_period_of_rejects_non_quarter_end():
    # 非季末不是报告期，必须返回 None，不能硬套到某个季度
    for raw in ("2024-04-15", "2024-01-01", "2024-06-29", "bad"):
        assert period_of(raw) == (None, None)


def test_period_of_fy_crosses_year():
    # 年报的披露截止日落在下一年，这是最容易写错的边界
    assert period_of("2024-12-31")[1].startswith("2025")


# ---------- gross_margin_of：两路兜底 ----------

def test_gross_margin_prefers_direct():
    row = {"销售毛利率(%)": 91.5, "主营业务成本率(%)": 8.5}
    cols = set(row)
    assert gross_margin_of(row, cols) == 91.5


def test_gross_margin_falls_back_to_cost_ratio():
    # 新数据源的销售毛利率经常空，靠成本率反算。不兜底就是整列缺失
    row = {"销售毛利率(%)": None, "主营业务成本率(%)": 17.6}
    cols = set(row)
    assert gross_margin_of(row, cols) == pytest.approx(82.4)


def test_gross_margin_column_absent():
    row = {"主营业务成本率(%)": 8.09}
    assert gross_margin_of(row, set(row)) == pytest.approx(91.91)


def test_gross_margin_no_source_returns_none():
    assert gross_margin_of({}, set()) is None


def test_gross_margin_nan_treated_as_missing():
    row = {"销售毛利率(%)": float("nan"), "主营业务成本率(%)": 20.0}
    assert gross_margin_of(row, set(row)) == pytest.approx(80.0)


# ---------- get_fundamentals：时点口径 ----------

def test_hides_undisclosed_period():
    """2024Q3 的 announce_date 是 10-31。在 10-15 这一天不许看到它。"""
    conn = make_db([
        {"symbol": "600309", "period": "2024Q3", "announce_date": "2024-10-31",
         "roe": 11.9},
        {"symbol": "600309", "period": "2024H1", "announce_date": "2024-08-31",
         "roe": 8.94},
    ])
    f = get_fundamentals(conn, "600309", "2024-10-15")
    assert f is not None
    assert f["period"] == "2024H1", "10-15 只应看到半年报，看不到三季报"


def test_sees_period_on_announce_day():
    """披露当日就该可见，不能漏掉（<= 不是 <）。"""
    conn = make_db([
        {"symbol": "600309", "period": "2024Q3", "announce_date": "2024-10-31",
         "roe": 11.9},
    ])
    f = get_fundamentals(conn, "600309", "2024-10-31")
    assert f is not None and f["period"] == "2024Q3"


def test_returns_none_when_nothing_disclosed():
    conn = make_db([
        {"symbol": "600309", "period": "2024Q3", "announce_date": "2024-10-31",
         "roe": 11.9},
    ])
    assert get_fundamentals(conn, "600309", "2024-03-15") is None
    assert get_fundamentals(conn, "000001", "2024-12-20") is None


def test_picks_latest_disclosed():
    conn = make_db([
        {"symbol": "600309", "period": "2024Q1", "announce_date": "2024-04-30", "roe": 4.5},
        {"symbol": "600309", "period": "2024H1", "announce_date": "2024-08-31", "roe": 8.94},
        {"symbol": "600309", "period": "2024Q3", "announce_date": "2024-10-31", "roe": 11.9},
    ])
    f = get_fundamentals(conn, "600309", "2024-12-20")
    assert f["period"] == "2024Q3"


# ---------- get_fundamentals：同比取值 ----------

def test_trend_uses_same_period_last_year():
    """2024Q3 必须对 2023Q3。对 H1 是口径错，对 Q1 也是口径错（累计长度不同）。"""
    conn = make_db([
        {"symbol": "600309", "period": "2024Q3", "announce_date": "2024-10-31",
         "roe": 11.9, "gross_margin": 15.38, "debt_ratio": 67.19},
        # 同一年里更早的期，绝不能被拿来当同比基准
        {"symbol": "600309", "period": "2024H1", "announce_date": "2024-08-31",
         "roe": 8.94, "gross_margin": 16.41, "debt_ratio": 66.14},
        {"symbol": "600309", "period": "2023Q3", "announce_date": "2023-10-31",
         "roe": 15.08, "gross_margin": 16.76, "debt_ratio": 64.98},
    ])
    f = get_fundamentals(conn, "600309", "2024-12-20")
    assert f["prev_period"] == "2023Q3"
    # ROE 11.90 - 15.08 = -3.18pp
    assert f["trend"]["roe"] == pytest.approx(-3.18, abs=0.01)
    # 毛利率 15.38 - 16.76 = -1.38pp
    assert f["trend"]["gross_margin"] == pytest.approx(-1.38, abs=0.01)


def test_trend_lookup_survives_narrow_window():
    """回看窗口只有 4 期，跨年同比常被截断，必须回库补查。"""
    rows = [{"symbol": "600309", "period": "2024Q3", "announce_date": "2024-10-31",
             "roe": 11.9, "gross_margin": 15.38, "debt_ratio": 67.19}]
    # 塞 4 期更早的期挤满窗口，把 2023Q3 挤出 hist
    for i, (p, ad, roe) in enumerate([
            ("2024FY", "2025-04-30", None),      # 未披露，不会进窗口
            ]):
        pass
    for p, ad, roe in [
        ("2024Q1", "2024-04-30", 4.5),
        ("2024H1", "2024-08-31", 8.94),
    ]:
        rows.append({"symbol": "600309", "period": p, "announce_date": ad,
                     "roe": roe, "gross_margin": 16.4, "debt_ratio": 65.0})
    rows.append({"symbol": "600309", "period": "2023Q3", "announce_date": "2023-10-31",
                 "roe": 15.08, "gross_margin": 16.76, "debt_ratio": 64.98})
    conn = make_db(rows)
    f = get_fundamentals(conn, "600309", "2024-12-20", n_periods=2)
    assert f["prev_period"] == "2023Q3", "窗口缩到 2 期时必须回库找到去年同期"
    assert f["trend"]["roe"] == pytest.approx(-3.18, abs=0.01)


def test_no_prev_period_gives_none_or_empty_trend():
    """没有去年同期数据时，trend 不该编一个值出来。"""
    conn = make_db([
        {"symbol": "600309", "period": "2024Q3", "announce_date": "2024-10-31",
         "roe": 11.9, "gross_margin": 15.38, "debt_ratio": 67.19},
    ])
    f = get_fundamentals(conn, "600309", "2024-12-20")
    assert f["prev_period"] is None
    assert f["trend"] in (None, {})


def test_missing_field_stays_none():
    """取不到的指标留 None，不能填 0（0 会被当成真实数值读进 prompt）。"""
    conn = make_db([
        {"symbol": "600001", "period": "2024Q3", "announce_date": "2024-10-31",
         "roe": 8.0, "gross_margin": None, "debt_ratio": 60.0},
    ])
    f = get_fundamentals(conn, "600001", "2024-12-20")
    assert f["roe"] == pytest.approx(8.0)
    assert f["gross_margin"] is None

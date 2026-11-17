# -*- coding: utf-8 -*-
"""`GET /api/fundamentals/{symbol}` 的接口级测试。

这个接口只做一件事：把"这一期在展示日当天是否已经公开"算对。
算错的后果是未来函数——回测里用了当时还看不到的财报，整个结论作废。
所以时点口径必须逐个边界测，不能只测一个正常日期。
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SYM = "测试标的"

# 两套数据：2024 年四期 + 2023 年同期（供同比取基线）
ROWS = [
    # 2024 年
    (SYM, "2024Q1", "2024-04-30", 4.5, 17.6, 10.0, 4.0, 64.5, 0.98, 0.65, 1.74),
    (SYM, "2024H1", "2024-08-31", 8.94, 16.4, 10.8, -3.0, 66.1, 3.28, 0.63, 3.53),
    (SYM, "2024Q3", "2024-10-31", 11.9, 15.4, 11.35, -11.2, 67.2, 6.15, 0.66, 5.41),
    (SYM, "2024FY", "2025-04-30", 13.8, 16.2, 3.8, -19.2, 64.7, 9.57, 0.56, 6.82),
    # 2023 年同期
    (SYM, "2023Q1", "2023-04-30", 5.0, 17.5, 0.37, -20.5, 63.9, 1.33, 0.64, 1.90),
    (SYM, "2023H1", "2023-08-31", 10.6, 16.4, -1.67, -14.3, 64.7, 3.57, 0.63, 3.92),
    (SYM, "2023Q3", "2023-10-31", 15.1, 16.8, 1.64, -2.7, 65.0, 5.46, 0.64, 5.93),
]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_file = str(tmp_path_factory.mktemp("funddb") / "test.db")
    os.environ["LIGHTQUANT_DB"] = db_file

    from app import db
    db.init_db(db_file)

    from app import auth
    conn = db.connect(db_file)
    conn.execute(
        "INSERT INTO users (id, username, display_name, role, password_hash)"
        " VALUES (?, ?, ?, ?, ?)",
        (1, "investor_demo", "演示投资者", "investor",
         auth.hash_password("demo1234")))
    for r in ROWS:
        conn.execute(
            "INSERT INTO fundamentals (symbol, period, announce_date, roe, "
            "gross_margin, revenue_yoy, profit_yoy, debt_ratio, ocf_ps, "
            "current_ratio, inventory_turn) VALUES (?,?,?,?,?,?,?,?,?,?,?)", r)
    conn.commit()
    conn.close()

    from fastapi.testclient import TestClient
    from app.api import app
    yield TestClient(app)

    os.environ.pop("LIGHTQUANT_DB", None)


def _tok(client):
    r = client.post("/api/auth/login",
                    json={"username": "investor_demo", "password": "demo1234"})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def _get(client, hdr, date=None, limit=None):
    url = "/api/fundamentals/" + SYM
    q = []
    if date is not None:
        q.append("date=" + date)
    if limit is not None:
        q.append("limit=%d" % limit)
    if q:
        url += "?" + "&".join(q)
    return client.get(url, headers=hdr)


# ---------- 时点口径：这一期在当天是否已公开 ----------

@pytest.mark.parametrize("date,expect_first", [
    ("2024-10-15", "2024H1"),   # Q3 还没到 10-31 截止日，不可见
    ("2024-10-30", "2024H1"),   # 截止日前一天，仍不可见
    ("2024-10-31", "2024Q3"),   # 恰好当天：<= 而非 <，必须可见
    ("2024-08-30", "2024Q1"),   # H1 截止日 08-31 前一天
    ("2024-08-31", "2024H1"),
    ("2024-05-01", "2024Q1"),   # 年报 2025-04-30 才披露，2024 年内都看不到 2024FY
    ("2023-05-01", "2023Q1"),
])
def test_visibility_by_date(client, date, expect_first):
    rows = _get(client, _tok(client), date=date).json()
    assert rows, "该日期之前应当有已披露的期"
    assert rows[0]["period"] == expect_first


def test_before_any_disclosure_returns_empty(client):
    # 2023Q1 的截止日是 2023-04-30，早于它的日期什么都没有，不能回退到更晚的期
    rows = _get(client, _tok(client), date="2023-01-01").json()
    assert rows == []


def test_annual_report_invisible_within_year(client):
    # 年报的披露截止日在次年 04-30，2024-12-31 当天不可能看到 2024FY
    rows = _get(client, _tok(client), date="2024-12-31").json()
    assert all(r["period"] != "2024FY" for r in rows)
    assert rows[0]["period"] == "2024Q3"


# ---------- 同比取值：必须对同报告期上一年 ----------

def test_yoy_uses_same_period_last_year(client):
    rows = _get(client, _tok(client), date="2024-12-31").json()
    assert rows[0]["period"] == "2024Q3"
    # 若取"上一期"（2024H1），基期就是 2024H1，这里的断言会失败
    assert rows[0]["yoy_period"] == "2023Q3"
    assert rows[0]["yoy"]["roe"] == pytest.approx(11.9 - 15.1, abs=0.01)


def test_yoy_null_when_prior_year_missing(client):
    # 2023Q1 的上一年是 2022Q1，库里没有 -> yoy 必须是 None，不能拿 2023Q1 自己比自己
    rows = _get(client, _tok(client), date="2023-08-31").json()
    q1 = [r for r in rows if r["period"] == "2023Q1"][0]
    assert q1["yoy"] is None
    assert q1["yoy_period"] is None


def test_yoy_covers_non_percentage_fields(client):
    # 现金流是元、周转率是次，同样要有同比；只算百分比指标会漏掉一半信息
    rows = _get(client, _tok(client), date="2024-12-31").json()
    y = rows[0]["yoy"]
    assert y["ocf_ps"] == pytest.approx(6.15 - 5.46, abs=0.01)
    assert y["inventory_turn"] == pytest.approx(5.41 - 5.93, abs=0.01)


# ---------- 排序、条数、鉴权 ----------

def test_default_order_is_disclosure_desc(client):
    rows = _get(client, _tok(client)).json()
    dates = [r["announce_date"] for r in rows]
    assert dates == sorted(dates, reverse=True)
    assert rows[0]["period"] == "2024FY"   # 全量里披露最晚的是 2025-04-30 的年报


def test_limit_is_applied(client):
    rows = _get(client, _tok(client), date="2024-12-31", limit=2).json()
    assert len(rows) == 2
    assert [r["period"] for r in rows] == ["2024Q3", "2024H1"]


@pytest.mark.parametrize("limit", [0, 41])
def test_limit_out_of_range_rejected(client, limit):
    # limit 有边界约束（1..40），越界必须 422 而不是静默截断
    assert _get(client, _tok(client), limit=limit).status_code == 422


def test_requires_auth(client):
    assert client.get("/api/fundamentals/" + SYM).status_code == 401


def test_unknown_symbol_returns_empty(client):
    r = client.get("/api/fundamentals/不存在的标的", headers=_tok(client))
    assert r.status_code == 200
    assert r.json() == []


def test_fields_are_complete(client):
    rows = _get(client, _tok(client), date="2024-12-31").json()
    need = {"period", "announce_date", "roe", "gross_margin", "revenue_yoy",
            "profit_yoy", "debt_ratio", "ocf_ps", "current_ratio",
            "inventory_turn", "yoy", "yoy_period"}
    assert need <= set(rows[0].keys())

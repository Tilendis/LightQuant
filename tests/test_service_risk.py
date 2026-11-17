# -*- coding: utf-8 -*-
"""风控与跟单的口径测试。

这三件事最容易被答辩追问，也最容易"看起来在跑其实算错"：
  1. 强平线到底是移动止损还是固定止损；
  2. 预警（MARGIN_CALL）与强平（LIQUIDATED）的分界线；
  3. 社区发帖的三条规则到底拦什么、放什么。

load_nav 用 monkeypatch 注入构造曲线，不读 backtest/result_v2，测试与回测产物解耦。
"""

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture()
def conn(tmp_path):
    from app import db
    path = str(tmp_path / "risk.db")
    db.init_db(path)
    c = db.connect(path)
    yield c
    c.close()


def _seed_account(conn, initial=100000.0, pct=0.10, tag="demo_tag"):
    """建一条最小可跑的策略→订阅→账户链。"""
    conn.execute("INSERT INTO users (id, username, role) VALUES (1, 'inv', 'investor')")
    conn.execute("INSERT INTO users (id, username, role) VALUES (2, 'au', 'author')")
    params = json.dumps({"tag": tag})
    conn.execute("INSERT INTO strategies (id, author_id, name, params_json, status)"
                 " VALUES (1, 2, '测试策略', ?, 'PUBLISHED')", (params,))
    conn.execute("INSERT INTO subscriptions (id, investor_id, strategy_id)"
                 " VALUES (1, 1, 1)")
    conn.execute("INSERT INTO follow_accounts"
                 " (id, subscription_id, initial_nav, current_nav, peak_nav,"
                 "  stop_loss_nav, stop_loss_pct)"
                 " VALUES (1, 1, ?, ?, ?, ?, ?)",
                 (initial, initial, initial, initial * (1 - pct), pct))
    conn.commit()


def test_advance_without_account_is_error(conn):
    from app import service_risk as risk
    ok, payload = risk.advance_account(conn, 999)
    assert ok is False
    assert "not found" in payload


def test_missing_curve_is_error(conn, monkeypatch):
    from app import service_risk as risk
    _seed_account(conn)
    monkeypatch.setattr(risk, "load_nav", lambda tag: [])
    ok, payload = risk.advance_account(conn, 1)
    assert ok is False
    assert "no nav curve" in payload


def test_account_scales_with_curve(conn, monkeypatch):
    """账户净值按 initial/base 等比放大，不是直接把策略净值当账户净值。"""
    from app import service_risk as risk
    _seed_account(conn, initial=100000.0)
    curve = [("2024-09-01", 1.0), ("2024-09-02", 1.1)]
    monkeypatch.setattr(risk, "load_nav", lambda tag: curve)

    ok, payload = risk.advance_account(conn, 1, dry_run=True)
    assert ok is True
    assert payload["current_nav"] == pytest.approx(110000.0, abs=0.01)
    assert payload["peak_nav"] == pytest.approx(110000.0, abs=0.01)


def test_stop_line_is_trailing_not_fixed(conn, monkeypatch):
    """强平线 = max(初始×0.9, 峰值×0.9)，峰值上去后线跟着上移。"""
    from app import service_risk as risk
    _seed_account(conn, initial=100000.0, pct=0.10)
    curve = [("2024-09-01", 1.0),
             ("2024-09-02", 1.5),        # 峰值抬到 150000
             ("2024-09-03", 1.32)]       # 132000：低于 135000，但高于 130950
    monkeypatch.setattr(risk, "load_nav", lambda tag: curve)

    ok, payload = risk.advance_account(conn, 1, dry_run=True)
    assert ok is True
    assert payload["peak_nav"] == pytest.approx(150000.0, abs=0.01)
    # 峰值×0.9 = 135000，明显高于初始×0.9 = 90000，证明是移动止损
    assert payload["stop_loss_nav"] == pytest.approx(135000.0, abs=0.01)
    # 132000 落在预警线与强平线之间 -> 预警（不是强平）
    assert payload["liquidate_nav"] == pytest.approx(130950.0, abs=0.01)
    assert payload["event"] == "MARGIN_CALL"
    assert payload["status"] == "ACTIVE"


def test_touch_line_is_margin_call(conn, monkeypatch):
    from app import service_risk as risk
    _seed_account(conn, initial=100000.0, pct=0.10)
    curve = [("2024-09-01", 1.0), ("2024-09-02", 0.89)]   # 89000 < 90000
    monkeypatch.setattr(risk, "load_nav", lambda tag: curve)

    ok, payload = risk.advance_account(conn, 1, dry_run=True)
    assert payload["event"] == "MARGIN_CALL"
    assert payload["status"] == "ACTIVE"
    assert payload["warn_date"] == "2024-09-02"


def test_deeper_drop_is_liquidated(conn, monkeypatch):
    """预警线再跌 3% 才真强平：90000 × 0.97 = 87300。"""
    from app import service_risk as risk
    _seed_account(conn, initial=100000.0, pct=0.10)
    curve = [("2024-09-01", 1.0), ("2024-09-02", 0.86)]   # 86000 < 87300
    monkeypatch.setattr(risk, "load_nav", lambda tag: curve)

    ok, payload = risk.advance_account(conn, 1, dry_run=True)
    assert payload["event"] == "LIQUIDATED"
    assert payload["status"] == "LIQUIDATED"


def test_boundary_between_call_and_liquidate(conn, monkeypatch):
    """刚好落在两条线之间时只能预警，这条边界最容易写反。"""
    from app import service_risk as risk
    _seed_account(conn, initial=100000.0, pct=0.10)
    # 88000：低于 stop 90000，但高于 liq 87300
    curve = [("2024-09-01", 1.0), ("2024-09-02", 0.88)]
    monkeypatch.setattr(risk, "load_nav", lambda tag: curve)

    ok, payload = risk.advance_account(conn, 1, dry_run=True)
    assert payload["event"] == "MARGIN_CALL"
    assert payload["status"] == "ACTIVE"


def test_dry_run_writes_nothing(conn, monkeypatch):
    from app import service_risk as risk
    _seed_account(conn)
    curve = [("2024-09-01", 1.0), ("2024-09-02", 0.85)]
    monkeypatch.setattr(risk, "load_nav", lambda tag: curve)

    before = conn.execute("SELECT current_nav, status FROM follow_accounts"
                          " WHERE id = 1").fetchone()
    risk.advance_account(conn, 1, dry_run=True)
    after = conn.execute("SELECT current_nav, status FROM follow_accounts"
                         " WHERE id = 1").fetchone()
    assert after["current_nav"] == before["current_nav"]
    assert after["status"] == before["status"]
    assert conn.execute("SELECT COUNT(*) FROM risk_events").fetchone()[0] == 0


def test_event_persisted_and_deduped(conn, monkeypatch):
    """同一账户同一触发日只记一条，重复扫描不刷屏。"""
    from app import service_risk as risk
    _seed_account(conn)
    curve = [("2024-09-01", 1.0), ("2024-09-02", 0.88)]
    monkeypatch.setattr(risk, "load_nav", lambda tag: curve)

    risk.advance_account(conn, 1)
    risk.advance_account(conn, 1)
    rows = conn.execute("SELECT kind, detail FROM risk_events").fetchall()
    assert len(rows) == 1
    assert rows[0]["kind"] == "MARGIN_CALL"
    assert "date=2024-09-02" in rows[0]["detail"]


def test_liquidated_account_is_skipped(conn, monkeypatch):
    from app import service_risk as risk
    _seed_account(conn)
    conn.execute("UPDATE follow_accounts SET status = 'LIQUIDATED' WHERE id = 1")
    conn.commit()
    ok, payload = risk.advance_account(conn, 1)
    assert ok is True
    assert payload["skipped"] == "already liquidated"


def test_scan_all_only_touches_active(conn, monkeypatch):
    from app import service_risk as risk
    _seed_account(conn)
    conn.execute("UPDATE follow_accounts SET status = 'LIQUIDATED' WHERE id = 1")
    conn.commit()
    monkeypatch.setattr(risk, "load_nav", lambda tag: [("2024-09-01", 1.0)])
    assert risk.scan_all(conn) == []


# ---------- 发帖规则层 -------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    ("这只票保证收益", "guaranteed_return"),
    ("稳赚不赔，跟就完了", "guaranteed_return"),
    ("明天必涨", "guaranteed_return"),
    ("零风险套利机会", "guaranteed_return"),
    ("想学的加微信", "solicit_group"),
    ("进群领取名单", "solicit_group"),
    ("代操盘，收益五五分", "proxy_trading"),
])
def test_moderation_rules_hit(text, kind):
    from app import service_risk as risk
    hits = [k for k, pat in risk.MODERATION_RULES if pat.search(text)]
    assert kind in hits


@pytest.mark.parametrize("text", [
    "今天大盘量能不错，观察为主",
    "财报出来了，毛利率环比改善",
    "这个位置我不追，等回踩",
    "回测跑完了，夏普 3.5，但样本外要打折",
])
def test_moderation_rules_clean_text_passes(text):
    from app import service_risk as risk
    assert [k for k, pat in risk.MODERATION_RULES if pat.search(text)] == []


def test_liquidate_extra_is_three_percent():
    """凭证式常量：改这个数等于改了强平口径，必须显式确认。"""
    from app import service_risk as risk
    assert risk.LIQUIDATE_EXTRA == 0.03
    assert risk.DEFAULT_STOP_LOSS_PCT == 0.10

# -*- coding: utf-8 -*-
"""认证与角色权限的接口级测试。

覆盖的目标很具体：证明"改请求头就能换身份"这个老漏洞确实被封住了。
全部用临时库（LIGHTQUANT_DB 环境变量），不碰 app/data/lightquant.db。
"""

import base64
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db_file = str(tmp_path_factory.mktemp("authdb") / "test.db")
    os.environ["LIGHTQUANT_DB"] = db_file

    from app import db
    db.init_db(db_file)

    from app import auth
    conn = db.connect(db_file)
    for uid, username, name, role in [
            (1, "investor_demo", "演示投资者", "investor"),
            (2, "author_demo", "演示策略作者", "author"),
            (3, "risk_demo", "演示风控员", "risk"),
            (4, "operator_demo", "演示运营", "operator")]:
        conn.execute(
            "INSERT INTO users (id, username, display_name, role, password_hash)"
            " VALUES (?, ?, ?, ?, ?)",
            (uid, username, name, role, auth.hash_password("demo1234")))
    conn.commit()
    conn.close()

    from fastapi.testclient import TestClient
    from app.api import app
    # 不用 with：跳过 lifespan 里的引擎预热，测试不需要 torch
    yield TestClient(app)

    os.environ.pop("LIGHTQUANT_DB", None)


def _login(client, username="risk_demo", password="demo1234"):
    r = client.post("/api/auth/login",
                    json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["token"]


# ---------- 没有凭证一律 401 -------------------------------------------------

def test_missing_authorization_is_401(client):
    assert client.get("/api/strategies").status_code == 401


def test_demo_users_is_open(client):
    """初始化接口本身是登录的前置步骤，不能要求认证。"""
    assert client.post("/api/auth/demo-users").status_code == 200


def test_old_header_spoofing_is_dead(client):
    """回归测试：老的 X-User-Id 冒充必须失效，否则等于没改。"""
    r = client.get("/api/risk/events", headers={"X-User-Id": "3"})
    assert r.status_code == 401


# ---------- 登录 -------------------------------------------------------------

def test_login_ok_returns_token(client):
    r = client.post("/api/auth/login",
                    json={"username": "risk_demo", "password": "demo1234"})
    assert r.status_code == 200
    body = r.json()
    assert body["token"]
    assert body["user"]["role"] == "risk"
    assert body["expires_in"] > 0


def test_wrong_password_is_401(client):
    r = client.post("/api/auth/login",
                    json={"username": "risk_demo", "password": "wrong"})
    assert r.status_code == 401


def test_unknown_user_same_message_as_wrong_password(client):
    """两条错误返回同一句话，避免泄露账号是否存在。"""
    a = client.post("/api/auth/login",
                    json={"username": "risk_demo", "password": "wrong"})
    b = client.post("/api/auth/login",
                    json={"username": "nobody", "password": "wrong"})
    assert a.status_code == b.status_code == 401
    assert a.json()["detail"] == b.json()["detail"]


def test_login_missing_fields_is_400(client):
    assert client.post("/api/auth/login", json={"username": "risk_demo"}).status_code == 400


# ---------- 带 token 访问 ----------------------------------------------------

def test_token_grants_access(client):
    tok = _login(client)
    r = client.get("/api/strategies",
                   headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 200


def test_role_still_enforced_after_auth(client):
    """认证过了不代表有权限：投资者拿自己的合法 token，读风控仍然 403。"""
    inv = _login(client, "investor_demo")
    r = client.get("/api/risk/events",
                   headers={"Authorization": "Bearer " + inv})
    assert r.status_code == 403


def test_risk_token_can_read_risk_events(client):
    tok = _login(client)
    r = client.get("/api/risk/events",
                   headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 200


# ---------- 伪造 token -------------------------------------------------------

def _b64d(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64e(raw):
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def test_tampered_payload_rejected(client):
    """把 token 里 uid 改成 1，签名沿用原值 —— 最典型的伪造方式。"""
    tok = _login(client, "risk_demo")
    body, sig = tok.rsplit(".", 1)
    payload = json.loads(_b64d(body).decode())
    payload["uid"] = 1
    forged = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    r = client.get("/api/risk/events",
                   headers={"Authorization": "Bearer " + forged + "." + sig})
    assert r.status_code == 401


def test_tampered_signature_rejected(client):
    tok = _login(client, "risk_demo")
    body, sig = tok.rsplit(".", 1)
    bad = sig[:-1] + ("A" if sig[-1] != "A" else "B")
    r = client.get("/api/risk/events",
                   headers={"Authorization": "Bearer " + body + "." + bad})
    assert r.status_code == 401


def test_garbage_token_rejected(client):
    for bad in ["abc", "a.b.c", "Bearer", ""]:
        r = client.get("/api/strategies", headers={"Authorization": "Bearer " + bad})
        assert r.status_code == 401


def test_bad_scheme_rejected(client):
    tok = _login(client)
    r = client.get("/api/strategies", headers={"Authorization": "Token " + tok})
    assert r.status_code == 401


# ---------- 过期 -------------------------------------------------------------

def test_expired_token_rejected(client):
    from app import auth
    tok = auth.make_token(3, "risk", ttl=-1)      # 立即过期
    r = client.get("/api/risk/events",
                   headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 401


def test_token_for_nonexistent_user_rejected(client):
    """签名有效但 uid 不在库里 —— 不能让一个已删除的账号继续通行。"""
    from app import auth
    tok = auth.make_token(999, "risk", ttl=3600)
    r = client.get("/api/risk/events",
                   headers={"Authorization": "Bearer " + tok})
    assert r.status_code == 401

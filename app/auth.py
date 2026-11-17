"""轻量认证：HMAC-SHA256 签名的无状态 token。

设计取舍（答辩要能解释）：
- 不引 JWT / python-jose 等库。本项目的 token 只需要"证明这个 id 是服务端签发的"，
  HMAC 签名足够，且实现只有几十行，可逐行讲清。
- token 结构是 `payload_b64.signature_b64`。payload 是 JSON，含 uid / role / exp。
  改 payload 里任何一个字符，签名都对不上，直接 401。
- 密钥存 app/data/.auth_secret（首次运行自动生成）。不入库、不进前端、不进仓库。
- 密码用 PBKDF2-HMAC-SHA256（10 万轮）+ 随机盐，只存 derivation，不存明文。

局限（必须写明）：密钥文件与库同目录，本地演示够用；生产应放到环境变量或密钥管理服务。
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SECRET_PATH = os.path.join(APP_DIR, "data", ".auth_secret")
TOKEN_TTL = 8 * 3600          # 演示够用，一次启动内不会过期

_cache = {"secret": None}


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def get_secret() -> str:
    """密钥优先取环境变量，其次读本地文件，都没有就生成一份。"""
    if _cache["secret"]:
        return _cache["secret"]
    env = os.environ.get("LIGHTQUANT_SECRET")
    if env:
        _cache["secret"] = env
        return env
    if os.path.exists(SECRET_PATH):
        with open(SECRET_PATH, "r", encoding="utf-8") as f:
            _cache["secret"] = f.read().strip()
        return _cache["secret"]
    s = secrets.token_urlsafe(32)
    os.makedirs(os.path.dirname(SECRET_PATH), exist_ok=True)
    with open(SECRET_PATH, "w", encoding="utf-8") as f:
        f.write(s)
    _cache["secret"] = s
    return s


def hash_password(password: str, salt: str = None) -> str:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                             salt.encode("utf-8"), 100000)
    return "%s$%s" % (salt, dk.hex())


def verify_password(password: str, stored: str) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, _ = stored.split("$", 1)
    return hmac.compare_digest(hash_password(password, salt), stored)


def make_token(uid: int, role: str, ttl: int = TOKEN_TTL) -> str:
    payload = {"uid": int(uid), "role": role, "exp": int(time.time()) + int(ttl)}
    body = _b64e(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    sig = _b64e(hmac.new(get_secret().encode(), body.encode(), hashlib.sha256).digest())
    return body + "." + sig


def parse_token(token: str):
    """验签通过且未过期才返回 payload，否则返回 None。"""
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    expect = _b64e(hmac.new(get_secret().encode(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expect):
        return None
    try:
        payload = json.loads(_b64d(body).decode("utf-8"))
    except Exception:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload

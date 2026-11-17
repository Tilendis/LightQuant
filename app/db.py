"""应用层数据库：连接、建表、以及几条必须"后端也校验"的角色规则。

为什么不用 ORM：表只有十张，字段稳定，写 SQL 比引 SQLAlchemy 更省事，也让
"某条数据到底怎么来的"一眼可见。默认库文件放在 app/data/lightquant.db，
可以用 LIGHTQUANT_DB 环境变量覆盖（演示时换成临时库做干净彩排）。
"""

import os
import sqlite3

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(APP_DIR, "schema.sql")
DEFAULT_DB = os.path.join(APP_DIR, "data", "lightquant.db")

ROLES = ("investor", "author", "risk", "operator")
VIEWS = ("tech", "fund", "sent", "fusion")


def db_path():
    return os.environ.get("LIGHTQUANT_DB") or DEFAULT_DB


def connect(path=None):
    """打开连接：WAL + 外键 + Row 工厂（便于直接当 dict 用）。

    check_same_thread=False 是必需的：FastAPI 会把同步依赖（get_conn、
    get_user）分别丢进线程池，两者可能落在不同线程，用默认的线程绑定
    连接会抛 sqlite3.ProgrammingError 变成 500。每个请求独占一条连接、
    同一时刻只有一个线程在用它，因此放宽这一限制是安全的。
    """
    path = path or db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _migrate(conn):
    """补齐后加过的列。

    CREATE TABLE IF NOT EXISTS 对"已存在的表"什么都不做，所以给老表加字段
    必须单独 ALTER，否则升级到新版本后会因为列缺失直接报错
    （users.password_hash 就踩过这个坑：本机手动 ALTER 过能跑，
    别人重建库反而缺列）。
    """
    existing = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    if "password_hash" not in existing:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")

    # fundamentals 扩字段：原表只有 pe/pb/roe/revenue/net_profit，判断"公司好不好"
    # 还缺毛利率、增速、偿债与现金流，以及时点对齐必需的 announce_date。
    fund_cols = {
        "announce_date": "TEXT",
        "gross_margin": "REAL",
        "revenue_yoy": "REAL",
        "profit_yoy": "REAL",
        "debt_ratio": "REAL",
        "ocf_ps": "REAL",
        "current_ratio": "REAL",
        "inventory_turn": "REAL",
        "source": "TEXT",
    }
    existing = {r[1] for r in conn.execute("PRAGMA table_info(fundamentals)")}
    for name, typ in fund_cols.items():
        if name not in existing:
            conn.execute("ALTER TABLE fundamentals ADD COLUMN %s %s" % (name, typ))
    # 补列之后再建索引，否则老库会因缺列直接报错
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fundamentals_announce "
        "ON fundamentals (symbol, announce_date)")

    # follow_accounts 补 as_of：账户净值推到哪个交易日，原来没记，
    # 于是前端只能写死一个日期去调推进。老库里 current_nav 已经偏离初始值的账户，
    # 说明当初确实推演过，回填成行情末端（而不是留空，留空会被当成"还没推过"）。
    existing = {r[1] for r in conn.execute("PRAGMA table_info(follow_accounts)")}
    if "as_of" not in existing:
        conn.execute("ALTER TABLE follow_accounts ADD COLUMN as_of TEXT")
    conn.execute(
        "UPDATE follow_accounts SET as_of = (SELECT MAX(date) FROM prices) "
        "WHERE as_of IS NULL AND current_nav <> initial_nav "
        "AND (SELECT MAX(date) FROM prices) IS NOT NULL")


def init_db(path=None, conn=None):
    """幂等建表：重复调用不会重复创建，也不会丢数据。"""
    own = conn is None
    conn = conn or connect(path)
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        _migrate(conn)
        conn.commit()
    finally:
        if own:
            conn.close()


def table_names(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return sorted(r["name"] for r in rows)


def require_role(conn, user_id, *allowed):
    """后端角色校验。演示时前端会隐藏入口，但接口这一层必须自己拦。

    返回 (ok, role)：角色不符时 ok=False，调用方据此返回 403。
    """
    row = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        return False, None
    role = row["role"]
    return (role in allowed), role


def upsert_judgement(conn, symbol, date, view, signal, score, reason=None,
                     provider=None, model=None, temperature=None,
                     prompt_version="v1", source="live"):
    """写入或更新一条研判。

    UNIQUE(symbol, date, view, prompt_version) 保证同一口径不会写两条；
    重复计算时覆盖分数与理由，但 provider/model 等溯源字段一并更新。
    """
    if view not in VIEWS:
        raise ValueError(f"unknown view: {view}")
    if signal not in ("BUY", "SELL", "HOLD"):
        raise ValueError(f"unknown signal: {signal}")
    score = float(score)
    if not -1.0 <= score <= 1.0:
        raise ValueError(f"score out of range: {score}")
    conn.execute(
        """INSERT INTO judgements
               (symbol, date, view, signal, score, reason, provider, model,
                temperature, prompt_version, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (symbol, date, view, prompt_version) DO UPDATE SET
               signal = excluded.signal,
               score = excluded.score,
               reason = excluded.reason,
               provider = excluded.provider,
               model = excluded.model,
               temperature = excluded.temperature,
               source = excluded.source""",
        (symbol, date, view, signal, score, reason, provider, model,
         temperature, prompt_version, source),
    )
    conn.commit()

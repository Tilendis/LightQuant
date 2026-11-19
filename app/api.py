"""LightQuant 应用层 API（FastAPI）。

设计约定（答辩要能解释清楚）：
- 身份：走 `Authorization: Bearer <token>`，token 由 app/auth.py 用 HMAC-SHA256 签名，
  校验不过一律 401。角色校验由 need() 负责（403），两层分开：认证回答"你是谁"，
  授权回答"你能干什么"。详见 docs/认证与角色权限.md。
- 数据：每个请求独立 SQLite 连接（WAL），不引 ORM。
- 回测：POST 立刻返回 task_id，后台线程执行，前端轮询 progress。
- 断网：全部读本地库与 result_v2 产物，不依赖任何外部 API。

启动：
    uvicorn app.api:app --host 127.0.0.1 --port 8000
"""

import csv
import json
import math
import os
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import auth, db, service_backtest, service_judge, service_risk  # noqa: E402

RESULT_ROOT = os.path.join(PROJECT_ROOT, "backtest", "result_v2")
ACCURACY_PATH = os.path.join(APP_DIR, "data", "view_accuracy.json")
DEMO_PASSWORD = "demo1234"      # 四个演示账号的统一口令，答辩时可现场演示登录

_ENGINE = {"ready": False, "error": None}


def _warm_engine():
    """后台预热两个引擎，避免首次请求让用户干等。

    run_backtest_fusion 顶层 import torch，纯 CPU 冷启动约 30–40 秒；
    service_judge 的 BiLSTM 首次加载实测 49 秒。两者都在后台线程预热，
    不阻塞 health 与所有读接口。
    """
    try:
        service_backtest._load_fusion()
        _ENGINE["ready"] = True
    except Exception as e:      # 预热失败不影响服务，真跑回测时会再完整报错
        _ENGINE["error"] = str(e)
    try:
        service_judge.warm_model()
    except Exception:
        pass                    # 研判调度失败时会在 /api/judgements/run 里单独报错


@asynccontextmanager
async def lifespan(_app):
    # 建表 + 补齐后加过的列。放在这里而不是只在脚本里，
    # 是为了让"换一个新库启动"（LIGHTQUANT_DB 指向空文件）也能直接跑起来，
    # 而不是等到第一个请求才报 no such table。
    db.init_db()
    # 启动时清理上次进程中断遗留的 PENDING/RUNNING 任务，
    # 否则前端会永远显示 RUNNING 且永远不会被执行。
    _conn = db.connect()
    _conn.execute(
        "UPDATE backtest_tasks SET status='FAILED', error='服务重启导致任务中断' "
        "WHERE status IN ('PENDING','RUNNING')")
    _conn.commit()
    _conn.close()
    threading.Thread(target=_warm_engine, daemon=True, name="warm-engine").start()
    threading.Thread(target=_risk_watchdog_loop, daemon=True, name="risk-watchdog").start()
    yield


# R4 定时巡检：后台每 30 秒只读一遍所有账户水位，
# 进风控页即见"上次自动巡检时间"，不靠手动点扫描。
# dry-run：不推进、不写事件，只更新内存缓存。
_RISK_WATCH = {"last_scan": None, "accounts": [], "interval_sec": 30}


def _risk_watchdog_loop():
    import time as _t
    while True:
        try:
            c = db.connect()
            try:
                _RISK_WATCH["accounts"] = service_risk.inspect_all(c)
                _RISK_WATCH["last_scan"] = _t.strftime("%Y-%m-%d %H:%M:%S")
            finally:
                c.close()
        except Exception:
            pass
        _t.sleep(_RISK_WATCH["interval_sec"])


app = FastAPI(title="LightQuant 量化投研平台", version="0.1.0", lifespan=lifespan)


# ---------- 基础依赖 ---------------------------------------------------------

def get_conn():
    conn = db.connect()
    try:
        yield conn
    finally:
        conn.close()


def get_user(conn=Depends(get_conn), authorization: str = Header(default=None)):
    """从 `Authorization: Bearer <token>` 解析身份。

    早期版本直接信任 X-User-Id 请求头：把 1 改成 3 就拿到风控员权限，没有任何凭证。
    现在身份由服务端签名签发，token 里改一个字符签名就对不上，直接 401。
    角色校验仍由 need() 负责，两层分开：认证回答"你是谁"，授权回答"你能干什么"。
    """
    if not authorization:
        raise HTTPException(401, "缺少 Authorization 头，先调 POST /api/auth/login 取 token")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(401, "Authorization 格式应为 Bearer <token>")
    payload = auth.parse_token(token.strip())
    if not payload:
        raise HTTPException(401, "token 无效或已过期，重新登录")
    row = conn.execute(
        "SELECT id, username, display_name, role FROM users WHERE id = ?",
        (payload["uid"],)).fetchone()
    if row is None:
        raise HTTPException(401, "token 对应用户不存在")
    return dict(row)


def need(user, *roles):
    if user["role"] not in roles:
        raise HTTPException(
            403, "角色 %s 无权执行，需要 %s" % (user["role"], "/".join(roles)))


def rows_to_list(rows):
    return [dict(r) for r in rows]


# ---------- 健康与元信息 -----------------------------------------------------

@app.get("/api/health")
def health(conn=Depends(get_conn)):
    stats = {}
    for t in ("users", "stocks", "prices", "news", "valuations", "judgements",
              "backtest_tasks", "strategies", "subscriptions",
              "follow_accounts", "risk_events", "posts"):
        stats[t] = conn.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0]
    rng = conn.execute(
        "SELECT MIN(date) a, MAX(date) b FROM prices").fetchone()
    return {"ok": True, "counts": stats,
            "price_range": [rng["a"], rng["b"]],
            "engine_ready": _ENGINE["ready"],
            "engine_error": _ENGINE["error"]}


@app.post("/api/auth/demo-users")
def seed_demo_users(conn=Depends(get_conn)):
    """幂等创建四个角色账号，并保证每个账号都有可登录的密码。

    演示密码统一为 demo1234。返回值只给用户名和角色，不回传 password_hash。
    这个接口不需要认证：它本身就是登录之前的前置步骤。
    """
    demo = [("investor_demo", "演示投资者", "investor"),
            ("author_demo", "演示策略作者", "author"),
            ("risk_demo", "演示风控员", "risk"),
            ("operator_demo", "演示运营", "operator")]
    created = []
    for username, name, role in demo:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = ?",
            (username,)).fetchone()
        if row:
            uid = row["id"]
            if not row["password_hash"]:
                conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                             (auth.hash_password(DEMO_PASSWORD), uid))
        else:
            cur = conn.execute(
                "INSERT INTO users (username, display_name, role, password_hash)"
                " VALUES (?, ?, ?, ?)",
                (username, name, role, auth.hash_password(DEMO_PASSWORD)))
            uid = cur.lastrowid
        created.append({"id": uid, "username": username,
                        "display_name": name, "role": role})
    conn.commit()
    return {"users": created, "demo_password": DEMO_PASSWORD}


@app.post("/api/auth/login")
def login(body: dict = Body(...), conn=Depends(get_conn)):
    """用户名 + 密码换 token。token 里带 uid、role、exp，服务端签名。"""
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if not username or not password:
        raise HTTPException(400, "username 与 password 必填")
    row = conn.execute(
        "SELECT id, username, display_name, role, password_hash FROM users"
        " WHERE username = ?", (username,)).fetchone()
    # 用户不存在与密码错误返回同一句话，避免泄露账号是否存在
    if row is None or not auth.verify_password(password, row["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")
    token = auth.make_token(row["id"], row["role"])
    return {
        "token": token,
        "user": {"id": row["id"], "username": row["username"],
                 "display_name": row["display_name"], "role": row["role"]},
        "expires_in": auth.TOKEN_TTL,
    }


@app.get("/api/meta")
def meta():
    """模型与视角清单，前端图例直接用这个。"""
    return {
        "views": [
            {"key": "tech", "name": "技术面", "provider": "local",
             "model": "BiLSTM + 10D features"},
            {"key": "fund", "name": "基本面", "provider": "offline_rule",
             "model": "PE/PB 三年分位规则"},
            {"key": "sent", "name": "情绪面", "provider": "zhipu",
             "model": "glm-5.3-flash"},
            {"key": "fusion", "name": "融合", "provider": "local",
             "model": "加权 / 投票"},
        ],
        "signal_colors": {"BUY": "#d14343", "SELL": "#2b8a3e", "HOLD": "#8a8a8a"},
    }


# ---------- 行情与研判（所有角色可读）---------------------------------------

@app.get("/api/market/stocks")
def list_stocks(conn=Depends(get_conn), user=Depends(get_user)):
    rows = conn.execute(
        "SELECT s.symbol, s.name, s.code, COUNT(p.date) bars "
        "FROM stocks s LEFT JOIN prices p ON p.symbol = s.symbol "
        "GROUP BY s.symbol ORDER BY s.symbol").fetchall()
    return rows_to_list(rows)


# 研判数据的版本选择：按列表顺序逐日取"第一个有记录的版本"。
#
# tech/fund/sent 是两段：智能体重构版（只覆盖 2024-12-16~12-31）优先，
# 该日没有就回退基线版，否则 K 线与研判页会在 12 月前面突然断档。
#
# fusion 有三段，且必须"输入与权重同源"，不能跨段混用：
#   fusion-agent-v1        12-16~12-31  输入是 LLM 三视角，权重绑该链路（tech1.0/sent0.5）
#   fusion-test-3view-v2   08-08~12-31  基线三视角 + 验证集标定权重 0.90/0.00/0.10（修复后 tech）
#   fusion-val-3view       03-15~08-07  同上，标定与回测都在 val 段
# 历史遗留的 fusion-val-v1(tech=1.0) / fusion-v1(0.5/0.25/0.25) / fusion-vote-v1
# 以及被 v2 取代的 fusion-test-3view（12 月段 tech 为修复前数据）
# 只留在回测结果目录里做对照证据，不再进线上链路，避免界面展示与论证口径不一致。
PV_PREF = {
    "tech": ("tech-agent-v1", "tech-v1"),
    # fund 用 v3：v3 的四类材料里独有"公司财务"（ROE/毛利率/增速/负债率/现金流），
    # 是三个 fund 版本中材料最全的一个。v3 只覆盖 12-16~12-31，
    # 更早日期按顺序回退 fund-agent-v1、fund-rule-v1，不会断档。
    "fund": ("fund-agent-v3", "fund-agent-v1", "fund-rule-v1"),
    "sent": ("sent-agent-v1", "sent-v1"),
    "fusion": ("fusion-agent-v1", "fusion-test-3view-v2", "fusion-val-3view"),
}

# 编排层与主融合表在库里同为 view='fusion'，只能靠 prompt_version 区分，
# 所以它不能走 PV_PREF（那会取到主融合表）。智能体版编排优先、缺失回退规则版。
ORCH_PREF = ("fusion-orch-agent-v1", "fusion-orch-v1")


def pick_by_date(symbol, view, pref, lo, hi, conn):
    """pref 是版本优先级列表，逐日取列表中排最前、且该日有记录的版本。

    返回 {date: row}。融合表多个日期段并存，单个二元组表达不了，
    故改为列表；列表长度为 1 时就是"只认这一个版本"。
    """
    pvs = list(pref)
    ph = ",".join(["?"] * len(pvs))
    rows = conn.execute(
        "SELECT symbol,date,view,signal,score,reason,provider,model,"
        "temperature,prompt_version,source FROM judgements "
        "WHERE symbol=? AND view=? AND date BETWEEN ? AND ? "
        "AND prompt_version IN (%s)" % ph,
        tuple([symbol, view, lo, hi] + pvs)).fetchall()
    rank = {pv: i for i, pv in enumerate(pvs)}
    out = {}
    for r in rows:
        d = r["date"]
        cur = out.get(d)
        if cur is None or rank[r["prompt_version"]] < rank[cur["prompt_version"]]:
            out[d] = r
    return out


@app.get("/api/market/kline/{symbol}")
def kline(symbol: str, start: str = Query(default=None),
          end: str = Query(default=None),
          fusion_version: str = Query(default="auto"),
          conn=Depends(get_conn), user=Depends(get_user)):
    """K 线 + 当日四视角信号，前端叠加对比直接用这个。

    signals 里每个视角给 score 与 signal，score>=0 判看多。
    版本按新优先、缺失回退基线，不会因智能体版本覆盖不全而断档。
    """
    sql = ("SELECT date, open, high, low, close, volume, label "
           "FROM prices WHERE symbol = ?")
    args = [symbol]
    if start:
        sql += " AND date >= ?"
        args.append(start)
    if end:
        sql += " AND date <= ?"
        args.append(end)
    sql += " ORDER BY date"
    rows = conn.execute(sql, args).fetchall()
    if not rows:
        raise HTTPException(404, "无该标的行情：%s" % symbol)

    lo, hi = rows[0]["date"], rows[-1]["date"]
    if fusion_version != "auto":
        pref = dict(PV_PREF)
        # 显式指定融合版本时只认这一个，不做任何回退
        pref["fusion"] = (fusion_version,)
    else:
        pref = PV_PREF

    sc = {}
    for v in ("tech", "fund", "sent", "fusion"):
        one = pick_by_date(symbol, v, pref[v], lo, hi, conn)
        sc[v] = {d: float(r["score"]) for d, r in one.items()}

    def sig(v):
        if v is None:
            return None
        return "BUY" if v > 0.2 else ("SELL" if v < -0.2 else "HOLD")

    bars = []
    for r in rows:
        d = r["date"]
        bars.append({
            "date": d,
            "open": r["open"], "high": r["high"], "low": r["low"],
            "close": r["close"], "volume": r["volume"], "label": r["label"],
            "signals": {
                v: {"score": sc[v].get(d), "signal": sig(sc[v].get(d))}
                for v in ("tech", "fund", "sent", "fusion")
            },
        })
    return {"symbol": symbol, "bars": bars}


@app.get("/api/judgements/{symbol}")
def judgements_of(symbol: str, view: str = Query(default="fusion"),
                  prompt_version: str = Query(default=None),
                  limit: int = Query(default=200, ge=1, le=2000),
                  conn=Depends(get_conn), user=Depends(get_user)):
    """prompt_version=auto 时按"智能体优先、缺失回退基线"逐日取版本。

    前端研判页用 auto，这样 2024-12 显示智能体重构版，更早的日期
    自动落回 tech-v1 / fund-rule-v1 / sent-v1，不会留空。
    """
    if prompt_version == "auto":
        if view == "orch":
            # 编排层：view 仍是 fusion，只换版本链；智能体版 12 天优先，更早日期回退规则版。
            merged = pick_by_date(symbol, "fusion", ORCH_PREF,
                                  "0000-00-00", "9999-99-99", conn)
            return rows_to_list([merged[d] for d in sorted(merged.keys())][:limit])
        pref_lst = PV_PREF.get(view, ("%s-agent-v1" % view, "%s-v1" % view))
        merged = pick_by_date(symbol, view, pref_lst,
                              "0000-00-00", "9999-99-99", conn)
        return rows_to_list([merged[d] for d in sorted(merged.keys())][:limit])

    sql = ("SELECT symbol,date,view,signal,score,reason,provider,model,"
           "temperature,prompt_version,source FROM judgements "
           "WHERE symbol = ? AND view = ?")
    args = [symbol, view]
    if prompt_version:
        sql += " AND prompt_version = ?"
        args.append(prompt_version)
    sql += " ORDER BY date LIMIT ?"
    args.append(limit)
    rows = conn.execute(sql, args).fetchall()
    return rows_to_list(rows)


@app.get("/api/fundamentals/{symbol}")
def fundamentals_of(symbol: str, date: str = Query(default=None),
                    limit: int = Query(default=8, ge=1, le=40),
                    conn=Depends(get_conn), user=Depends(get_user)):
    """公司财务指标（只读）。基本面三层里"公司财务"这一层的原始数据。

    时点口径是这个接口的命门：财务数据按报告期发布，但报表不是报告期结束
    当天就公开的。每一期都带 announce_date（法定披露截止日），传 date 时
    只返回该日期之前**已披露**的期，否则就是未来函数。

    返回最近 limit 期（按披露日倒序），每期附 yoy——与**同报告期上一年**
    的对比（2024Q3 对 2023Q3）。不能用"上一期"：Q1→H1 是累计口径，
    比出来的不是增长。

    这个接口只用于描述财务状态，不参与方向预测。实测该视角对次日涨跌
    IC≈0（T+1 −0.0237、T+5 −0.0615），融合权重为 0；强行给权重会稀释
    技术面（val ACC 0.5538 → 0.4792）。
    """
    sql = ("SELECT period,announce_date,roe,gross_margin,revenue_yoy,"
           "profit_yoy,debt_ratio,ocf_ps,current_ratio,inventory_turn "
           "FROM fundamentals WHERE symbol = ?")
    args = [symbol]
    if date:
        sql += " AND announce_date <= ?"
        args.append(date)
    sql += " ORDER BY announce_date DESC LIMIT ?"
    args.append(limit)
    rows = conn.execute(sql, args).fetchall()
    if not rows:
        return []

    # 同报告期上一年：2024Q3 <-> 2023Q3。一次查齐，避免逐期回库。
    per = [r["period"] for r in rows]
    yoy_keys = [str(int(p[:4]) - 1) + p[4:] if p[:4].isdigit() else p for p in per]
    prev = {}
    if yoy_keys:
        ph = ",".join(["?"] * len(yoy_keys))
        for r in conn.execute(
                "SELECT period,roe,gross_margin,debt_ratio,revenue_yoy,profit_yoy,"
                "ocf_ps,current_ratio,inventory_turn "
                "FROM fundamentals WHERE symbol = ? AND period IN (%s)" % ph,
                [symbol] + yoy_keys).fetchall():
            prev[r["period"]] = dict(r)

    out = []
    for r, k in zip(rows, yoy_keys):
        d = dict(r)
        p = prev.get(k)
        yoy = None
        if p is not None:
            yoy = {}
            for f in ("roe", "gross_margin", "debt_ratio", "revenue_yoy",
                      "profit_yoy", "ocf_ps", "current_ratio", "inventory_turn"):
                a, b = d.get(f), p.get(f)
                if a is not None and b is not None:
                    yoy[f] = round(float(a) - float(b), 2)
        d["yoy"] = yoy
        d["yoy_period"] = p["period"] if p is not None else None
        out.append(d)
    return out


@app.get("/api/judgements")
def judgements_summary(conn=Depends(get_conn), user=Depends(get_user)):
    rows = conn.execute(
        "SELECT view, COUNT(*) n, MIN(date) first_date, MAX(date) last_date, "
        "AVG(score) avg_score, "
        "SUM(CASE WHEN signal='BUY' THEN 1 ELSE 0 END) buy, "
        "SUM(CASE WHEN signal='SELL' THEN 1 ELSE 0 END) sell, "
        "SUM(CASE WHEN signal='HOLD' THEN 1 ELSE 0 END) hold "
        "FROM judgements GROUP BY view ORDER BY view").fetchall()
    return rows_to_list(rows)


@app.post("/api/judgements/run")
def run_judgements(payload: dict = Body(...),
                   conn=Depends(get_conn), user=Depends(get_user)):
    """现场调度三个视角跑一次研判，当场返回三份报告与融合结论。

    这是课题「调度多个大模型（或不同分析视角的智能体）分别从技术面、基本面和
    情绪面生成研判报告」中"调度"与"生成"两个动词的落点。

    单只：{\"symbol\":\"万华化学\",\"date\":\"2024-12-31\"}
    批量：{\"symbols\":[\"万华化学\",\"贵州茅台\"],\"dates\":[\"2024-12-30\",\"2024-12-31\"]}
         或 {\"symbols\":[\"万华化学\"],\"date_start\":\"2024-12-29\",\"date_end\":\"2024-12-31\"}
    批量上限 50 个组合，全程共用一个 rerun 版本，演示完一键 cleanup。

    结果写独立版本 rerun-<时间戳>，不覆盖主口径 fusion-test-3view-v2；
    传 write=false 则只算不写，用于彩排。
    """
    need(user, "author", "operator")
    symbols = payload.get("symbols")
    allow_llm = bool(payload.get("allow_llm", True))
    write = bool(payload.get("write", True))
    pv = payload.get("prompt_version") or None

    if isinstance(symbols, list):
        if not symbols:
            raise HTTPException(400, "批量模式 symbols 不能为空列表")
        dates = payload.get("dates")
        if not isinstance(dates, list) or not dates:
            start = (payload.get("date_start") or "").strip()
            end = (payload.get("date_end") or "").strip()
            if not start or not end:
                raise HTTPException(400, "批量模式需要 dates 列表，或 date_start + date_end")
            rows = conn.execute(
                "SELECT DISTINCT date FROM prices WHERE date>=? AND date<=? ORDER BY date",
                (start, end)).fetchall()
            dates = [r["date"] for r in rows]
            if not dates:
                raise HTTPException(400, "%s ~ %s 之间没有交易日" % (start, end))
        try:
            result = service_judge.run_batch(
                conn, symbols, dates, allow_llm=allow_llm, write=write, prompt_version=pv)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return result

    symbol = (payload.get("symbol") or "").strip()
    date = (payload.get("date") or "").strip()
    if not symbol or not date:
        raise HTTPException(400, "需要 symbol 与 date，例如 {\"symbol\":\"万华化学\",\"date\":\"2024-12-31\"}")
    result = service_judge.run(
        conn, symbol, date, allow_llm=allow_llm, write=write, prompt_version=pv)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error") or "研判调度失败")
    return result


@app.delete("/api/judgements/run/{prompt_version}")
def cleanup_judgements(prompt_version: str,
                       conn=Depends(get_conn), user=Depends(get_user)):
    """删除某次 rerun 版本的全部记录（演示完一键清理，不动主口径）。"""
    need(user, "author", "operator")
    try:
        n = service_judge.cleanup(conn, prompt_version)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"prompt_version": prompt_version, "deleted": n}


@app.get("/api/accuracy")
def accuracy(user=Depends(get_user)):
    if not os.path.exists(ACCURACY_PATH):
        raise HTTPException(404, "缺少 view_accuracy.json，先跑 scripts/eval_views.py")
    with open(ACCURACY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- 异步回测（策略作者）----------------------------------------------

@app.post("/api/backtest/tasks")
def create_task(payload: dict = Body(...),
                conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "author")
    try:
        task_id, tag = service_backtest.submit(user["id"], payload)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"task_id": task_id, "tag": tag, "status": "PENDING"}


@app.get("/api/backtest/tasks")
def list_tasks(limit: int = Query(default=20, ge=1, le=200),
               conn=Depends(get_conn), user=Depends(get_user)):
    rows = conn.execute(
        "SELECT id,user_id,status,progress,result_path,error,created_at,"
        "started_at,finished_at,params_json FROM backtest_tasks "
        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["params"] = json.loads(d.pop("params_json"))
        except (ValueError, TypeError):
            d["params"] = {}
        out.append(d)
    return out


@app.get("/api/backtest/tasks/{task_id}")
def get_task(task_id: int, conn=Depends(get_conn), user=Depends(get_user)):
    r = conn.execute(
        "SELECT id,user_id,status,progress,result_path,error,created_at,"
        "started_at,finished_at,params_json FROM backtest_tasks WHERE id = ?",
        (task_id,)).fetchone()
    if r is None:
        raise HTTPException(404, "任务不存在")
    d = dict(r)
    d["params"] = json.loads(d.pop("params_json") or "{}")
    return d


REPORT_WHITELIST_FILE = os.path.join(PROJECT_ROOT, "docs", "backtest_report_whitelist.json")


def _json_safe(obj):
    """把 NaN / Inf 换成 None，再返回给前端。

    为什么必要：回测产物里 per_stock_mean 的 sharpe、mdd 在"该票全程没持仓"
    时会写成 NaN。json.load 能读，但 Starlette 响应序列化用 allow_nan=False，
    遇到 NaN 直接抛 ValueError，浏览器只看到 Internal Server Error，
    表现为"这张结果一点就报错、其它结果却正常"。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _report_whitelist():
    """读取「已有结果」下拉白名单。

    返回 set(tags)；文件不存在、解析失败或 tags 为空时返回 None，
    表示不做过滤（保持旧行为，避免白名单文件损坏导致下拉变空）。
    """
    if not os.path.isfile(REPORT_WHITELIST_FILE):
        return None
    try:
        with open(REPORT_WHITELIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    tags = []
    for g in (data.get("groups") or []):
        tags.extend(str(t) for t in (g.get("tags") or []))
    return set(tags) if tags else None


@app.get("/api/backtest/reports")
def list_reports(include_all: bool = Query(False, alias="all"), user=Depends(get_user)):
    """列出磁盘上的回测结果目录。

    为什么需要它：网页"已有结果"下拉原本只列出通过 API 提交的任务，
    命令行跑的（如验证集标定后的 api_f_val）不会出现在任务表里，
    演示时选不到新结果。这里直接扫 result_v2，把两边合并给前端。

    为什么还要白名单：历次调参与联调会在 result_v2 下留下几十个目录，
    接口若全部返回，演示下拉会变成一屏实验垃圾，评委找不到主口径。
    白名单只收敛"界面展示"，磁盘文件与文档里的引证路径一概不动；
    需要临时看全部时用 ?all=1。
    """
    if not os.path.isdir(RESULT_ROOT):
        return {"reports": [], "whitelist": None}
    wl = _report_whitelist()
    out = []
    hidden_tags = []
    for name in sorted(os.listdir(RESULT_ROOT)):
        sp = os.path.join(RESULT_ROOT, name, "summary.json")
        if not os.path.isfile(sp):
            continue
        if wl is not None and not include_all and name not in wl:
            # 只把"被藏起来的 tag 名"回报给前端：任务表里同名旧任务也要一并隐藏，
            # 新提交的 tag 不在这个名单里，提交完成后照常出现在下拉中。
            hidden_tags.append(name)
            continue
        try:
            with open(sp, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        pf = s.get("portfolio") or {}
        cfg = s.get("config") or {}
        out.append({
            "tag": name,
            "cum_return": pf.get("cum_return"),
            "sharpe": pf.get("sharpe"),
            "trades": pf.get("total_trades"),
            # 口径字段由回测脚本写入；补齐之前生成的结果没有这两项，
            # 前端据此显示"口径未记录"，不再按 tag 名猜测读的是哪张表。
            "views": cfg.get("views"),
            "fusion_version": cfg.get("prompt_version"),
        })
    return _json_safe({"reports": out, "hidden_tags": hidden_tags,
                       "whitelist": (sorted(wl) if wl is not None else None)})


@app.get("/api/backtest/{tag}/report")
def report(tag: str, conn=Depends(get_conn), user=Depends(get_user)):
    """回测报告：指标 + 净值 + 交易明细 + 最大回撤区间（前端画阴影用）。"""
    import re as _re
    if not _re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise HTTPException(400, "非法的 tag 名")
    d = os.path.join(RESULT_ROOT, tag)
    sp = os.path.join(d, "summary.json")
    if not os.path.exists(sp):
        raise HTTPException(404, "无该回测结果：%s" % tag)
    with open(sp, "r", encoding="utf-8") as f:
        summary = json.load(f)

    nav = []
    np_ = os.path.join(d, "nav.csv")
    if os.path.exists(np_):
        with open(np_, "r", encoding="utf-8") as f:
            rd = csv.reader(f)
            next(rd, None)
            for r in rd:
                if len(r) >= 3:
                    try:
                        nav.append({"date": r[0], "nav": float(r[1]),
                                    "bench": float(r[2])})
                    except ValueError:
                        continue

    trades = []
    tp = os.path.join(d, "trades.csv")
    if os.path.exists(tp):
        # utf-8-sig：trades.csv 带 BOM，用 utf-8 读会把首列列名变成
        # "\ufeffdate"，前端按 t.date 取日期全部落空，买卖点一个都画不出来。
        with open(tp, "r", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            for r in rd:
                trades.append(r)

    pf = dict(summary.get("portfolio", {}))
    # 这几个别名是前端契约：summary 里叫 arr / ir_vs_equal_weight / 没有 trades 计数，
    # 直接下发原字段会让页面上的年化、信息比率、成交笔数都显示成 "—"。
    pf["trades"] = len(trades)
    if "annual_return" not in pf:
        pf["annual_return"] = pf.get("arr")
    if "ir" not in pf:
        pf["ir"] = pf.get("ir_vs_equal_weight")
    return _json_safe({
        "tag": tag,
        "config": summary.get("config"),
        "portfolio": pf,
        "checks": summary.get("checks"),
        "per_stock_mean": summary.get("per_stock_mean"),
        "mdd_range": {"start": pf.get("mdd_start"), "end": pf.get("mdd_end")},
        "nav": nav,
        "trades": trades,
    })


# ---------- 策略发布（策略作者）----------------------------------------------

@app.post("/api/strategies")
def create_strategy(payload: dict = Body(...),
                    conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "author")
    task_id = payload.get("task_id")
    name = (payload.get("name") or "").strip()
    if not task_id or not name:
        raise HTTPException(400, "需要 task_id 与 name")

    t = conn.execute(
        "SELECT * FROM backtest_tasks WHERE id = ?", (task_id,)).fetchone()
    if t is None:
        raise HTTPException(404, "任务不存在")
    if t["status"] != "SUCCEEDED":
        raise HTTPException(409, "任务未成功（status=%s），不能发布" % t["status"])
    if t["user_id"] != user["id"]:
        raise HTTPException(403, "只能发布自己提交的任务")

    params = json.loads(t["params_json"] or "{}")
    if payload.get("take_profit_pct") is not None:
        params["take_profit_pct"] = float(payload["take_profit_pct"])
    if payload.get("stop_loss_pct") is not None:
        params["stop_loss_pct"] = float(payload["stop_loss_pct"])

    sp = os.path.join(PROJECT_ROOT, t["result_path"], "summary.json")
    metrics = {}
    if os.path.exists(sp):
        with open(sp, "r", encoding="utf-8") as f:
            metrics = json.load(f).get("portfolio", {})

    publish_now = bool(payload.get("publish", False))
    cur = conn.execute(
        "INSERT INTO strategies (author_id,name,params_json,metrics_json,status,"
        "published_at) VALUES (?,?,?,?,?,?)",
        (user["id"], name, json.dumps(params, ensure_ascii=False),
         json.dumps(metrics, ensure_ascii=False),
         "PUBLISHED" if publish_now else "DRAFT",
         None))
    sid = cur.lastrowid
    if publish_now:
        conn.execute(
            "UPDATE strategies SET published_at = datetime('now','localtime') "
            "WHERE id = ?", (sid,))
    conn.commit()
    return {"strategy_id": sid, "status": "PUBLISHED" if publish_now else "DRAFT",
            "metrics": metrics}


@app.get("/api/strategies")
def list_strategies(conn=Depends(get_conn), user=Depends(get_user)):
    # 已发布的策略对所有角色可见；未发布的草稿只对作者本人可见。
    # 不加这一层过滤，别人未发布的 DRAFT 策略会被所有人看到。
    rows = conn.execute(
        "SELECT id,author_id,name,status,metrics_json,published_at,params_json "
        "FROM strategies WHERE status = 'PUBLISHED' OR author_id = ? "
        "ORDER BY id DESC", (user["id"],)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d.pop("metrics_json") or "{}")
        d["params"] = json.loads(d.pop("params_json") or "{}")
        out.append(d)
    return out


@app.post("/api/strategies/{sid}/publish")
def publish_strategy(sid: int, conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "author")
    r = conn.execute("SELECT * FROM strategies WHERE id = ?", (sid,)).fetchone()
    if r is None:
        raise HTTPException(404, "策略不存在")
    if r["author_id"] != user["id"]:
        raise HTTPException(403, "只能发布自己的策略")
    if r["status"] == "PUBLISHED":
        return {"strategy_id": sid, "status": "PUBLISHED", "note": "已是发布状态"}
    conn.execute(
        "UPDATE strategies SET status='PUBLISHED', "
        "published_at=datetime('now','localtime') WHERE id = ?", (sid,))
    conn.commit()
    return {"strategy_id": sid, "status": "PUBLISHED"}


# ---------- 订阅与模拟跟单（投资者）------------------------------------------

@app.post("/api/subscriptions")
def subscribe(payload: dict = Body(...),
              conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "investor")
    sid = payload.get("strategy_id")
    initial = float(payload.get("initial_nav", 100000.0))
    # 强平线：相对峰值的回撤比例，默认 10%，允许投资者按风险偏好收紧
    try:
        stop_pct = float(payload.get("stop_loss_pct", service_risk.DEFAULT_STOP_LOSS_PCT))
    except (TypeError, ValueError):
        stop_pct = service_risk.DEFAULT_STOP_LOSS_PCT
    if not (0 < stop_pct < 1):
        stop_pct = service_risk.DEFAULT_STOP_LOSS_PCT
    if not sid or initial <= 0:
        raise HTTPException(400, "需要 strategy_id 与正数 initial_nav")

    s = conn.execute("SELECT * FROM strategies WHERE id = ?", (sid,)).fetchone()
    if s is None:
        raise HTTPException(404, "策略不存在")
    if s["status"] != "PUBLISHED":
        raise HTTPException(409, "策略未发布（status=%s）" % s["status"])

    dup = conn.execute(
        "SELECT 1 FROM subscriptions WHERE investor_id=? AND strategy_id=?",
        (user["id"], sid)).fetchone()
    if dup:
        raise HTTPException(409, "已订阅该策略")

    cur = conn.execute(
        "INSERT INTO subscriptions (investor_id, strategy_id) VALUES (?, ?)",
        (user["id"], sid))
    sub_id = cur.lastrowid
    stop_nav = initial * (1 - stop_pct)
    cur2 = conn.execute(
        "INSERT INTO follow_accounts (subscription_id, initial_nav, current_nav, "
        "peak_nav, stop_loss_nav, stop_loss_pct, status, as_of) "
        "VALUES (?,?,?,?,?,?,'ACTIVE',NULL)",
        (sub_id, initial, initial, initial, stop_nav, stop_pct))
    acc_id = cur2.lastrowid
    conn.commit()
    return {"subscription_id": sub_id, "account_id": acc_id,
            "initial_nav": initial, "stop_loss_nav": round(stop_nav, 2),
            "stop_loss_pct": stop_pct,
            "as_of": None,
            "note": "账户已建，净值等于初始值；用「推进一日」按交易日推演"}


@app.get("/api/subscriptions")
def my_subscriptions(conn=Depends(get_conn), user=Depends(get_user)):
    rows = conn.execute(
        "SELECT s.id, s.strategy_id, s.status, s.subscribed_at, "
        "a.id account_id, a.initial_nav, a.current_nav, a.peak_nav, "
        "a.stop_loss_nav, a.status account_status, a.as_of "
        "FROM subscriptions s LEFT JOIN follow_accounts a "
        "ON a.subscription_id = s.id WHERE s.investor_id = ? ORDER BY s.id DESC",
        (user["id"],)).fetchall()
    return rows_to_list(rows)


@app.get("/api/follow-accounts/{aid}")
def get_account(aid: int, conn=Depends(get_conn), user=Depends(get_user)):
    # 全仓最后一处缺失的角色校验：此前任何角色都能读任意跟单账户。
    need(user, "investor", "risk")
    r = conn.execute(
        "SELECT a.*, s.strategy_id, s.investor_id, p.name strategy_name FROM follow_accounts a "
        "JOIN subscriptions s ON s.id = a.subscription_id "
        "JOIN strategies p ON p.id = s.strategy_id WHERE a.id = ?",
        (aid,)).fetchone()
    if r is None:
        raise HTTPException(404, "跟单账户不存在")
    # 投资者只能看自己的账户；风控员需要看全部账户才能监控强平。
    if user["role"] == "investor" and r["investor_id"] != user["id"]:
        raise HTTPException(403, "只能查看自己的跟单账户")
    d = dict(r)
    tag = service_risk.strategy_tag(conn, d["strategy_id"])
    nav = service_risk.load_nav(tag) if tag else []
    d["nav_curve"] = [{"date": x, "nav": round(v * d["initial_nav"], 2)}
                      for x, v in nav]
    d["drawdown_pct"] = round(
        (d["current_nav"] / d["peak_nav"] - 1) * 100, 2) if d["peak_nav"] else 0.0
    return d


@app.post("/api/follow-accounts/{aid}/advance")
def advance(aid: int, payload: dict = Body(default={}),
            conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "investor", "risk")
    # 归属校验：投资者只能推进自己的账户；风控员可以推进全部。
    r = conn.execute(
        "SELECT s.investor_id FROM follow_accounts a "
        "JOIN subscriptions s ON s.id = a.subscription_id WHERE a.id = ?",
        (aid,)).fetchone()
    if r is None:
        raise HTTPException(404, "跟单账户不存在")
    if user["role"] == "investor" and r["investor_id"] != user["id"]:
        raise HTTPException(403, "只能推进自己的跟单账户")
    date = payload.get("date")
    if date == "latest":
        # 显式推到期末：只有明确要求才这么做
        ok, data = service_risk.advance_account(conn, aid, None)
    elif date:
        # 显式指定日期时必须落在交易日上：行情表只有交易日，传周六会被
        # cutoff 静默吞成周五，看起来"点了没反应"。
        days = service_risk.trade_days(conn)
        if days and date not in days:
            raise HTTPException(400, "%s 不是交易日" % date)
        # 防时光倒流：指定日期不能早于当前净值已推到的日期。
        cur = conn.execute(
            "SELECT as_of FROM follow_accounts WHERE id = ?", (aid,)).fetchone()
        if cur and cur["as_of"] and date < cur["as_of"]:
            raise HTTPException(400, "不能回退：净值已推到 %s" % cur["as_of"])
        ok, data = service_risk.advance_account(conn, aid, date)
    else:
        # 默认「推进一步」而不是「推到期末」：订阅跟单是按日走，
        # 一次请求往前走一个交易日，净值变化才看得见。
        ok, data = service_risk.step_account(conn, aid)
    if not ok:
        raise HTTPException(400, data)
    return data


@app.get("/api/trading-days")
def trading_days(conn=Depends(get_conn), user=Depends(get_user)):
    """交易日历。前端的「推进一日」靠它算出下一站——不写死任何日期。"""
    days = service_risk.trade_days(conn)
    return {"count": len(days),
            "first": days[0] if days else None,
            "last": days[-1] if days else None,
            "days": days}


@app.post("/api/follow-accounts/advance-all")
def advance_all(payload: dict = Body(default={}),
                conn=Depends(get_conn), user=Depends(get_user)):
    """把所有在途跟单账户统一推进到同一个交易日。

    不传 date 就是「推一步」：目标日由交易日历决定（各账户 as_of 最大值之后的那一天），
    调用方不需要知道任何日期。投资者只能推自己的（授权来源是 subscriptions.investor_id），
    风控员可以推全部。
    """
    need(user, "investor", "risk")
    ids = None
    if user["role"] == "investor":
        ids = [r["id"] for r in conn.execute(
            "SELECT a.id FROM follow_accounts a JOIN subscriptions s "
            "ON s.id = a.subscription_id WHERE s.investor_id = ?",
            (user["id"],)).fetchall()]
        if not ids:
            raise HTTPException(400, "你还没有跟单账户，先订阅一个策略")
    ok, data = service_risk.advance_all(conn, ids, payload.get("date"))
    if not ok:
        raise HTTPException(400, data)
    return data


# ---------- 风控（风控员）----------------------------------------------------

@app.post("/api/subscriptions/{sid}/cancel")
def cancel_subscription(sid: int, conn=Depends(get_conn),
                        user=Depends(get_user)):
    """投资者主动取消订阅。

    与"强平"是两回事：强平是风控按回撤线执行的，取消订阅是投资者自己退出。
    两者都要让账户停止推演，所以账户统一置为 LIQUIDATED（表里只有 ACTIVE /
    LIQUIDATED 两个状态），但 risk_events 的 detail 必须写清是哪一种——
    否则风控面板上会分不清"被强平"和"自己走了"。
    """
    need(user, "investor")
    s = conn.execute(
        "SELECT * FROM subscriptions WHERE id = ?", (sid,)).fetchone()
    if s is None:
        raise HTTPException(404, "订阅不存在")
    if s["investor_id"] != user["id"]:
        raise HTTPException(403, "只能取消自己的订阅")
    if s["status"] == "CANCELLED":
        return {"subscription_id": sid, "status": "CANCELLED",
                "note": "已是取消状态"}

    conn.execute("UPDATE subscriptions SET status='CANCELLED' WHERE id = ?", (sid,))
    acc = conn.execute(
        "SELECT id, current_nav, peak_nav FROM follow_accounts "
        "WHERE subscription_id = ?", (sid,)).fetchone()
    if acc is not None:
        conn.execute(
            "UPDATE follow_accounts SET status='LIQUIDATED', "
            "updated_at=datetime('now','localtime') WHERE id = ?", (acc["id"],))
        conn.execute(
            "INSERT INTO risk_events (account_id, kind, detail, handled_by) "
            "VALUES (?, 'LIQUIDATED', ?, ?)",
            (acc["id"],
             "投资者主动取消订阅（非强平）current=%.2f peak=%.2f"
             % (acc["current_nav"], acc["peak_nav"]), user["id"]))
    conn.commit()
    return {"subscription_id": sid, "status": "CANCELLED",
            "account_id": acc["id"] if acc is not None else None}


@app.get("/api/follow-accounts/{aid}/positions")
def account_positions(aid: int, date: str = None,
                      conn=Depends(get_conn), user=Depends(get_user)):
    """跟单账户的持仓明细。

    口径：模拟跟单是"用策略净值曲线等比缩放"，所以持仓本身由策略决定。
    这里从策略的 trades.csv 现算，不另存一份——存一份就有两个真相来源。
    date 可选，按交易日复盘到那一天为止的持仓。
    """
    need(user, "investor", "risk")
    r = conn.execute(
        "SELECT a.*, s.strategy_id, s.investor_id FROM follow_accounts a "
        "JOIN subscriptions s ON s.id = a.subscription_id WHERE a.id = ?",
        (aid,)).fetchone()
    if r is None:
        raise HTTPException(404, "跟单账户不存在")
    if user["role"] == "investor" and r["investor_id"] != user["id"]:
        raise HTTPException(403, "只能查看自己的跟单账户")

    tag = service_risk.strategy_tag(conn, r["strategy_id"])
    if not tag:
        raise HTTPException(409, "策略未记录 tag，无法复算持仓")
    snap = service_risk.positions(conn, tag, date)
    if snap is None:
        raise HTTPException(409, "策略没有成交明细（tag=%s）" % tag)

    # 口径必须说清：engine_v2 是"每只股票独立 10 万账户再等权平均"，
    # 所以 trades.csv 里的股数/市值是策略内单票口径，乘账户净值会得到一个
    # 荒唐的数（2066 股 × 10 万）。这里如实返回原始口径，另附两个参照值
    # 让前端自己算比例，而不是编一个假的"换算系数"。
    nav = service_risk.load_nav(tag)
    strat_nav = nav[-1][1] if nav else None

    return {
        "account_id": aid,
        "strategy_tag": tag,
        "as_of": snap["as_of"],
        "account_current_nav": round(r["current_nav"], 2),
        "strategy_portfolio_nav": round(strat_nav, 6) if strat_nav else None,
        "note": ("持仓来自策略成交明细，是策略内单票口径"
                 "（engine_v2 每只票独立 10 万账户）；不是该账户的真实持仓，"
                 "也不做等比缩放——股数缩放没有实际含义"),
        "total_market_value": snap["total_market_value"],
        "rows": snap["rows"],
    }


@app.post("/api/follow-accounts/{aid}/liquidate")
def liquidate_account(aid: int, payload: dict = Body(default={}),
                      conn=Depends(get_conn), user=Depends(get_user)):
    """风控员手工强平。

    自动强平已经写在 service_risk.advance_account 里（回撤破线自动置位），
    这个接口补的是"人对着监控面板按下去"的动作：需要记谁按的、为什么按，
    所以必须写一条带 handled_by 的 LIQUIDATED 事件。
    """
    need(user, "risk")
    acc = conn.execute(
        "SELECT id, status, current_nav, peak_nav, stop_loss_nav "
        "FROM follow_accounts WHERE id = ?", (aid,)).fetchone()
    if acc is None:
        raise HTTPException(404, "跟单账户不存在")
    if acc["status"] == "LIQUIDATED":
        return {"account_id": aid, "status": "LIQUIDATED",
                "note": "已是强平状态"}

    reason = (payload.get("reason") or "").strip() or "手工强平"
    conn.execute(
        "UPDATE follow_accounts SET status='LIQUIDATED', "
        "updated_at=datetime('now','localtime') WHERE id = ?", (aid,))
    conn.execute(
        "INSERT INTO risk_events (account_id, kind, detail, handled_by) "
        "VALUES (?, 'LIQUIDATED', ?, ?)",
        (aid,
         "手工强平：%s　current=%.2f peak=%.2f stop=%.2f"
         % (reason[:80], acc["current_nav"], acc["peak_nav"],
            acc["stop_loss_nav"]), user["id"]))
    conn.commit()
    return {"account_id": aid, "status": "LIQUIDATED", "reason": reason,
            "current_nav": acc["current_nav"],
            "stop_loss_nav": acc["stop_loss_nav"],
            "handled_by": user["id"]}


@app.post("/api/risk/scan")
def risk_scan(payload: dict = Body(default={}),
              conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "risk")
    # date 可选：不传则推进到各自曲线末端，传入则按该交易日复盘
    return {"results": service_risk.scan_all(conn, payload.get("date"))}


@app.get("/api/risk/events")
def risk_events(limit: int = Query(default=50, ge=1, le=500),
                conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "risk", "operator")
    rows = conn.execute(
        "SELECT id,account_id,post_id,kind,detail,handled_by,created_at "
        "FROM risk_events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return rows_to_list(rows)


@app.get("/api/risk/accounts")
def risk_accounts(date: str = None,
                  conn=Depends(get_conn), user=Depends(get_user)):
    """只读账户水位：在途账户离强平线还有多远，不推进、不写事件。

    不传 date 时读后台定时巡检的缓存（每 30 秒一次），进页即见；
    传 date 则按交易日现算复盘。
    """
    need(user, "risk", "operator")
    if not date:
        return {"as_of": None, "accounts": _RISK_WATCH["accounts"],
                "last_scan": _RISK_WATCH["last_scan"],
                "interval_sec": _RISK_WATCH["interval_sec"]}
    return {"as_of": date, "accounts": service_risk.inspect_all(conn, date)}


@app.get("/api/risk/scan-status")
def risk_scan_status(user=Depends(get_user)):
    """后台定时巡检状态：上次扫描时间、间隔、账户数。"""
    need(user, "risk", "operator")
    return {"last_scan": _RISK_WATCH["last_scan"],
            "interval_sec": _RISK_WATCH["interval_sec"],
            "accounts_count": len(_RISK_WATCH["accounts"])}


# ---------- 社区发帖与审核 ---------------------------------------------------

@app.post("/api/posts")
def create_post(payload: dict = Body(...),
                conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "investor", "author")
    content = (payload.get("content") or "").strip()
    if not content:
        raise HTTPException(400, "content 不能为空")
    # strategy_id 可空；填了就必须是一条已发布策略，避免帖子指向草稿或不存在的东西
    sid = payload.get("strategy_id")
    if sid is not None:
        s = conn.execute("SELECT id, status FROM strategies WHERE id = ?",
                         (sid,)).fetchone()
        if s is None:
            raise HTTPException(400, "strategy_id 不存在")
        if s["status"] != "PUBLISHED":
            raise HTTPException(400, "只能关联已发布（PUBLISHED）的策略")
    cur = conn.execute(
        "INSERT INTO posts (user_id, symbol, strategy_id, content) VALUES (?,?,?,?)",
        (user["id"], payload.get("symbol"), sid, content))
    conn.commit()
    return {"post_id": cur.lastrowid, "status": "PENDING", "strategy_id": sid}


@app.post("/api/posts/{pid}/moderate")
def moderate_post(pid: int, conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "risk")
    p = conn.execute("SELECT * FROM posts WHERE id = ?", (pid,)).fetchone()
    if p is None:
        raise HTTPException(404, "帖子不存在")
    decision, rule, detail, llm_verdict = service_risk.moderate(p["content"])
    conn.execute(
        "INSERT INTO post_moderation (post_id, rule_hit, llm_verdict, decision) "
        "VALUES (?,?,?,?)", (pid, rule, llm_verdict, decision))
    conn.execute("UPDATE posts SET status = ? WHERE id = ?", (decision, pid))
    if decision == "BLOCKED":
        conn.execute(
            "INSERT INTO risk_events (post_id, kind, detail, handled_by) "
            "VALUES (?, 'POST_BLOCKED', ?, ?)", (pid, detail, user["id"]))
    conn.commit()
    return {"post_id": pid, "decision": decision, "rule_hit": rule,
            "detail": detail, "llm_verdict": llm_verdict}


@app.get("/api/posts")
def list_posts(status: str = Query(default=None),
               conn=Depends(get_conn), user=Depends(get_user)):
    # 左连策略：帖子可以只是一段行情观点，也可以挂在一个已发布策略上
    sql = ("SELECT p.id,p.user_id,p.symbol,p.strategy_id,p.content,p.status,p.created_at,"
           "       s.name AS strategy_name, s.author_id AS strategy_author "
           "FROM posts p LEFT JOIN strategies s ON s.id = p.strategy_id")
    args = []
    # 普通用户只能看到已审核通过的帖子（以及自己发的任何状态）；
    # risk/operator 才能看到 PENDING 和 BLOCKED 的全部内容。
    if user["role"] not in ("risk", "operator"):
        sql += " WHERE (p.status = 'APPROVED' OR p.user_id = ?)"
        args.append(user["id"])
        if status:
            sql += " AND p.status = ?"
            args.append(status)
    elif status:
        sql += " WHERE p.status = ?"
        args.append(status)
    sql += " ORDER BY p.id DESC LIMIT 100"
    return rows_to_list(conn.execute(sql, args).fetchall())


# ---------- 运营看板 ---------------------------------------------------------

@app.get("/api/dashboard/activity")
def activity(conn=Depends(get_conn), user=Depends(get_user)):
    need(user, "operator", "author")
    """策略活跃度：按日看回测提交、策略发布、订阅、发帖四个动作。"""
    def by_day(table, col):
        rows = conn.execute(
            "SELECT date(%s) d, COUNT(*) n FROM %s "
            "WHERE %s IS NOT NULL AND %s != '' GROUP BY date(%s)"
            % (col, table, col, col, col)).fetchall()
        return {r["d"]: r["n"] for r in rows}

    bt = by_day("backtest_tasks", "created_at")
    pub = by_day("strategies", "published_at")
    sub = by_day("subscriptions", "subscribed_at")
    posts = by_day("posts", "created_at")
    days = sorted(set(list(bt) + list(pub) + list(sub) + list(posts)))
    return {
        "days": days,
        "series": {
            "backtest_tasks": [bt.get(d, 0) for d in days],
            "strategies_published": [pub.get(d, 0) for d in days],
            "subscriptions": [sub.get(d, 0) for d in days],
            "posts": [posts.get(d, 0) for d in days],
        },
        "totals": {
            "backtest_tasks": sum(bt.values()),
            "strategies": conn.execute(
                "SELECT COUNT(*) FROM strategies").fetchone()[0],
            "published": conn.execute(
                "SELECT COUNT(*) FROM strategies WHERE status='PUBLISHED'"
            ).fetchone()[0],
            "subscriptions": conn.execute(
                "SELECT COUNT(*) FROM subscriptions").fetchone()[0],
            "risk_events": conn.execute(
                "SELECT COUNT(*) FROM risk_events").fetchone()[0],
        },
    }


@app.get("/api/dashboard/strategy-activity")
def strategy_activity(prompt_version: str = Query(default="auto"),
                      conn=Depends(get_conn), user=Depends(get_user)):
    """策略运行活跃度：按交易日统计融合层产出信号的标的数与强信号数。

    与 /api/dashboard/activity 分开：那条按"操作日"统计，演示时全挤在一天，
    画出来是一根竖线；这条按"交易日"统计，才是策略真的在跑的曲线。

    必须按 prompt_version 过滤：judgements 表里 fusion 有九个版本并存，
    不过滤会把所有版本叠加，n 变成每天几百（实际 50 只），活跃度翻倍。

    prompt_version=auto 时走 PV_PREF["fusion"] 的同一套优先级逐日选版本，
    这样看板和 K 线叠加的信号是同一份数据，不会出现"图上两条线说的不是一回事"。
    """
    need(user, "operator", "author")
    if prompt_version == "auto":
        pvs = list(PV_PREF["fusion"])
        ph = ",".join(["?"] * len(pvs))
        rows = conn.execute(
            "SELECT date, prompt_version, score, signal FROM judgements "
            "WHERE view = 'fusion' AND prompt_version IN (%s)" % ph,
            tuple(pvs)).fetchall()
        rank = {pv: i for i, pv in enumerate(pvs)}
        # 先定每天用哪个版本，再只统计该版本的行
        best = {}
        for r in rows:
            d = r["date"]
            if d not in best or rank[r["prompt_version"]] < rank[best[d]]:
                best[d] = r["prompt_version"]
        agg = {}
        for r in rows:
            d = r["date"]
            if r["prompt_version"] != best[d]:
                continue
            s = agg.get(d)
            if s is None:
                s = agg[d] = {"date": d, "n": 0, "strong_n": 0,
                              "buy_n": 0, "sell_n": 0,
                              "prompt_version": r["prompt_version"]}
            s["n"] += 1
            if abs(r["score"] or 0) >= 0.2:
                s["strong_n"] += 1
            if r["signal"] == "BUY":
                s["buy_n"] += 1
            elif r["signal"] == "SELL":
                s["sell_n"] += 1
        return [agg[d] for d in sorted(agg)]

    rows = conn.execute(
        "SELECT date, COUNT(*) n, "
        "SUM(CASE WHEN ABS(score) >= 0.2 THEN 1 ELSE 0 END) strong_n, "
        "SUM(CASE WHEN signal = 'BUY' THEN 1 ELSE 0 END) buy_n, "
        "SUM(CASE WHEN signal = 'SELL' THEN 1 ELSE 0 END) sell_n "
        "FROM judgements WHERE view = 'fusion' AND prompt_version = ? "
        "GROUP BY date ORDER BY date", (prompt_version,)).fetchall()
    return rows_to_list(rows)


# ---------- 前端静态资源 -----------------------------------------------------

STATIC_DIR = os.path.join(APP_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    """前端入口：与 API 同源，ECharts 走本地文件，断网可直接演示。"""
    p = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(p):
        raise HTTPException(404, "前端未部署：缺 app/static/index.html")
    return FileResponse(p)

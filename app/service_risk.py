"""风控与跟单：模拟盘净值推演、强平预警、发帖拦截。

口径说明（答辩必须讲清楚，否则会被质疑"这不是实时撮合"）：
- 模拟跟单不是撮合引擎，而是"用策略参数在其回测区间内复算净值"。
  订阅后账户从策略净值曲线的起点开始按比例放大，日频、收盘价、T+1，
  与回测引擎完全同一套成本假设，因此可以逐日复盘、可以断网演示。
- 强平线是移动止损：max(初始净值 × (1-p), 历史峰值 × (1-p))，p 默认 10%。
  跌破强平线 → MARGIN_CALL（预警，账户仍在）；再跌 3% → LIQUIDATED（实际强平）。
"""

import csv
import os
import re

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
RESULT_ROOT = os.path.join(PROJECT_ROOT, "backtest", "result_v2")

DEFAULT_STOP_LOSS_PCT = 0.10
LIQUIDATE_EXTRA = 0.03     # 预警线再跌 3% 才真正强平

MODERATION_RULES = [
    ("guaranteed_return", re.compile(r"保证收益|稳赚|必涨|包赚|躺赚|翻倍|100\s*%|零风险")),
    ("solicit_group", re.compile(r"加微信|加\s*Q\s*Q|进群|私聊|带你|扫码|留联系方式")),
    ("proxy_trading", re.compile(r"代操盘|代管账户|代客理财|代你下单")),
]


def load_nav(tag):
    """读取策略净值曲线，返回 [(date, portfolio_nav), ...]。"""
    path = os.path.join(RESULT_ROOT, tag, "nav.csv")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) >= 2:
                try:
                    rows.append((r[0], float(r[1])))
                except ValueError:
                    continue
    return rows


def strategy_tag(conn, strategy_id):
    import json
    row = conn.execute("SELECT params_json FROM strategies WHERE id = ?",
                       (strategy_id,)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["params_json"]).get("tag")
    except (ValueError, TypeError):
        return None


def trade_days(conn):
    """全库交易日序列。取 prices 的 DISTINCT date，不是某一只票的 K 线——
    单只票停牌的日子仍然算交易日，账户不该因为一只票停牌就卡住。"""
    rows = conn.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()
    return [r[0] for r in rows]


def next_trade_day(days, after=None):
    """返回 after 之后的下一个交易日；after 为 None 表示从头开始。没有下一个就返回 None。"""
    if not days:
        return None
    if after is None:
        return days[0]
    for d in days:
        if d > after:
            return d
    return None


def advance_account(conn, account_id, target_date=None, dry_run=False):
    """把跟单账户净值推进到 target_date（不传则推进到曲线末端）。

    返回 (ok, payload)：ok=False 时 payload 是错误原因。
    """
    acc = conn.execute(
        "SELECT a.*, s.strategy_id, s.status AS sub_status FROM follow_accounts a "
        "JOIN subscriptions s ON s.id = a.subscription_id WHERE a.id = ?",
        (account_id,)).fetchone()
    if acc is None:
        return False, "account not found"
    if acc["status"] == "LIQUIDATED":
        return True, {"skipped": "already liquidated"}
    # 投资者取消订阅后，账户即便还是 ACTIVE 也不该继续推演——订阅关系才是授权来源
    if acc["sub_status"] != "ACTIVE":
        return True, {"skipped": "subscription cancelled"}

    tag = strategy_tag(conn, acc["strategy_id"])
    rows = load_nav(tag) if tag else []
    if not rows:
        return False, "strategy has no nav curve (tag=%s)" % tag

    cutoff = [r for r in rows if (target_date is None or r[0] <= target_date)]
    if not cutoff:
        return False, "target_date 早于策略净值起点"

    base = rows[0][1]
    initial = float(acc["initial_nav"])

    # 强平线按账户可配；老库没有该列时退回默认值
    try:
        p = float(acc["stop_loss_pct"])
    except (KeyError, IndexError, TypeError, ValueError):
        p = DEFAULT_STOP_LOSS_PCT
    if not (0 < p < 1):
        p = DEFAULT_STOP_LOSS_PCT

    # 逐日遍历：既取末端水位，也记录区间内最深回撤与首次触发日期。
    # 只算末端的话，曲线恢复到高位后就看不出"曾经触及过强平线"。
    peak = initial
    cur = initial
    worst_dd = 0.0
    first_call = None
    first_liquid = None

    for d, v in cutoff:
        cur = initial * (v / base)
        if cur > peak:
            peak = cur
        dd = cur / peak - 1.0
        if dd < worst_dd:
            worst_dd = dd
        stop_nav = max(initial * (1 - p), peak * (1 - p))
        liq_nav = stop_nav * (1 - LIQUIDATE_EXTRA)
        if first_liquid is None and cur < liq_nav:
            first_liquid = (d, cur, stop_nav, peak)
        elif first_call is None and cur < stop_nav:
            first_call = (d, cur, stop_nav, peak)

    stop_nav = max(initial * (1 - p), peak * (1 - p))
    liquidate_nav = stop_nav * (1 - LIQUIDATE_EXTRA)

    status = "ACTIVE"
    event = None
    hit = None
    if first_liquid is not None:
        status = "LIQUIDATED"
        event = "LIQUIDATED"
        hit = first_liquid
    elif first_call is not None:
        event = "MARGIN_CALL"
        hit = first_call

    warn_date = None
    if event:
        warn_date = hit[0]

    # dry_run 只算不写：风控面板打开时也要能看到水位，但不该因为"看了"就改账户状态
    if not dry_run:
        conn.execute(
            "UPDATE follow_accounts SET current_nav = ?, peak_nav = ?, "
            "stop_loss_nav = ?, status = ?, as_of = ?, "
            "updated_at = datetime('now','localtime') WHERE id = ?",
            (cur, peak, stop_nav, status, cutoff[-1][0], account_id))
        if event:
            dup = conn.execute(
                "SELECT 1 FROM risk_events WHERE account_id = ? AND kind = ? "
                "AND detail LIKE ?",
                (account_id, event, "date=" + warn_date + "%")).fetchone()
            if dup is None:
                conn.execute(
                    "INSERT INTO risk_events (account_id, kind, detail) VALUES (?, ?, ?)",
                    (account_id, event,
                     "date=%s nav=%.2f stop=%.2f peak=%.2f 回撤=%.2f%%（强平线 %.0f%%）"
                     % (warn_date, hit[1], hit[2], hit[3],
                        (hit[1] / hit[3] - 1) * 100, p * 100))
                )
        conn.commit()
    return True, {
        "account_id": account_id,
        "strategy_tag": tag,
        "as_of": cutoff[-1][0],
        "initial_nav": round(initial, 2),
        "current_nav": round(cur, 2),
        "peak_nav": round(peak, 2),
        "stop_loss_nav": round(stop_nav, 2),
        "liquidate_nav": round(liquidate_nav, 2),
        "stop_loss_pct": p,
        "max_dd_pct": round(worst_dd * 100, 2),
        "status": status,
        "event": event,
        "warn_date": warn_date,
    }


def step_account(conn, account_id):
    """把账户推进到它 as_of 之后的下一个交易日。

    这是「推进一日」的真实语义。`advance_account(target_date=None)` 是「推到期末」，
    两者不能混用——混用的结果就是新订阅账户（as_of 为空）一次被推到 12-31，
    界面上看不出净值逐日走，也正说明原来的"推进"其实是离线算好摆上来的。
    """
    row = conn.execute(
        "SELECT a.as_of, s.strategy_id FROM follow_accounts a "
        "JOIN subscriptions s ON s.id = a.subscription_id WHERE a.id = ?",
        (account_id,)).fetchone()
    if row is None:
        return False, "account not found"
    days = trade_days(conn)
    if not days:
        return False, "prices 表没有交易日"

    if row["as_of"] is None:
        # 还没推过：第一步必须落在**该策略自己的净值起点**，不能用全库第一天。
        # 策略 7 的曲线从 2024-08-08 起，而全库交易日从 03-15 起——用全库第一天
        # 会一路撞上"target_date 早于策略净值起点"直接 400。
        tag = strategy_tag(conn, row["strategy_id"])
        nav = load_nav(tag) if tag else []
        start = nav[0][0] if nav else None
        if not start:
            return False, "strategy has no nav curve (tag=%s)" % tag
        target = None
        for d in days:
            if d >= start:
                target = d
                break
        if target is None:
            return True, {"account_id": account_id, "as_of": None,
                          "skipped": "策略区间已超出行情范围"}
    else:
        target = next_trade_day(days, row["as_of"])
        if target is None:
            return True, {"account_id": account_id, "as_of": row["as_of"],
                          "skipped": "已经是最后一个交易日（%s）" % days[-1]}
    return advance_account(conn, account_id, target)


def advance_all(conn, account_ids=None, target_date=None, only_active=True):
    """批量推进：每个在途账户各自往前走一个交易日。

    为什么不统一推到一个 target：各账户进度可以不同（老账户已到期末，新订阅的
    还没开始）。统一取最新一天，落后账户会被一步拉平；统一取最落后的一天，已到
    期末的账户净值会倒退回早期——两种都会破坏数据。传了 target_date 才统一推。

    返回 (ok, payload)。ok=False 只有两种：没有可推进账户、prices 为空。
    「已经是最后一个交易日」是终态，不是错误，放在每个账户的 skipped 里。
    """
    rows = conn.execute(
        "SELECT id, as_of, status FROM follow_accounts ORDER BY id").fetchall()
    if account_ids is not None:
        wanted = {int(x) for x in account_ids}
        rows = [r for r in rows if r["id"] in wanted]
    if only_active:
        rows = [r for r in rows if r["status"] == "ACTIVE"]
    if not rows:
        return False, "没有可推进的跟单账户"

    days = trade_days(conn)
    if not days:
        return False, "prices 表没有交易日"

    if target_date is not None and target_date not in days:
        return False, "%s 不是交易日" % target_date

    results = []
    for r in rows:
        if target_date is None:
            ok, payload = step_account(conn, r["id"])
        else:
            ok, payload = advance_account(conn, r["id"], target_date)
        results.append({
            "account_id": r["id"],
            "ok": ok,
            "data": payload if ok else None,
            "error": None if ok else payload,
        })

    moved = [x for x in results if x["ok"] and not (x["data"] or {}).get("skipped")]
    skipped = [x for x in results if x["ok"] and (x["data"] or {}).get("skipped")]
    failed = [x for x in results if not x["ok"]]
    return True, {
        "mode": "single-day" if target_date is None else "fixed-date",
        "target": target_date,
        "count": len(results),
        "moved": len(moved),
        "skipped": len(skipped),
        "failed": len(failed),
        "as_of_max": max([(x["data"] or {}).get("as_of") or "" for x in moved] or [""]) or None,
        "results": results,
    }


def load_trades(tag):
    """读取策略成交明细，返回 dict 列表。"""
    path = os.path.join(RESULT_ROOT, tag, "trades.csv")
    if not os.path.exists(path):
        return []
    out = []
    # utf-8-sig：回测产物是带 BOM 的 UTF-8，用普通 utf-8 读会把首列
    # 读成 "\ufeffdate"，DictReader 全行 KeyError，最终静默返回空列表。
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                out.append({
                    "date": r["date"], "side": r["side"],
                    "price": float(r["price"]), "shares": float(r["shares"]),
                    "amount": float(r["amount"]), "fee": float(r["fee"]),
                    "stock": r["stock"],
                })
            except (KeyError, TypeError, ValueError):
                continue
    return out


def positions(conn, tag, as_of=None):
    """按成交明细复算截至 as_of 的持仓快照。

    为什么不在 follow_accounts 里存一份持仓：模拟跟单是"用策略净值曲线等比缩放"，
    持有什么由策略决定。存一份会和 trades.csv 形成两个真相来源，
    改一次回测就两边对不上。这里每次现算，天然与回测产物一致。

    成本用移动加权平均（A 股实务口径），不是先进先出——先 FIFO 会让
    "同一只票反复进出"的成本价与券商对账单对不上。
    """
    trades = load_trades(tag)
    if not trades:
        return None
    if as_of:
        trades = [t for t in trades if t["date"] <= as_of]
    if not trades:
        return None

    book = {}   # stock -> {shares, cost}（cost 是总成本，含买入手续费）
    for t in trades:
        s = book.setdefault(t["stock"], {"shares": 0.0, "cost": 0.0})
        if t["side"] == "BUY":
            s["shares"] += t["shares"]
            s["cost"] += t["amount"] + t["fee"]
        else:
            # 卖出按当前均价结转成本，剩余成本按比例减少；份额可能被清空
            if s["shares"] > 0:
                unit = s["cost"] / s["shares"]
                sold = min(t["shares"], s["shares"])
                s["shares"] -= sold
                s["cost"] -= unit * sold
                if s["shares"] <= 1e-9:
                    s["shares"] = 0.0
                    s["cost"] = 0.0

    last_date = trades[-1]["date"]
    rows = []
    for stock, s in book.items():
        if s["shares"] <= 1e-9:
            continue
        px = conn.execute(
            "SELECT close FROM prices WHERE symbol = ? AND date <= ? "
            "ORDER BY date DESC LIMIT 1", (stock, last_date)).fetchone()
        last_px = float(px["close"]) if px else None
        avg_cost = s["cost"] / s["shares"] if s["shares"] else 0.0
        mv = last_px * s["shares"] if last_px is not None else None
        rows.append({
            "stock": stock,
            "shares": round(s["shares"], 2),
            "avg_cost": round(avg_cost, 4),
            "last_price": round(last_px, 4) if last_px is not None else None,
            "market_value": round(mv, 2) if mv is not None else None,
            "unrealized_pnl": round(mv - s["cost"], 2) if mv is not None else None,
            "weight_pct": None,
        })
    total_mv = sum(r["market_value"] or 0.0 for r in rows)
    for r in rows:
        r["weight_pct"] = (round(r["market_value"] / total_mv * 100, 2)
                           if total_mv and r["market_value"] else None)
    rows.sort(key=lambda r: -(r["market_value"] or 0.0))
    return {"as_of": last_date, "total_market_value": round(total_mv, 2), "rows": rows}


def scan_all(conn, target_date=None):
    """风控员一键扫描：推进所有 ACTIVE 账户并触发预警。

    target_date 为空时推进到各自曲线的末端；指定日期则只看到那天为止的水位，
    便于按交易日复盘。
    """
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM follow_accounts WHERE status = 'ACTIVE'").fetchall()]
    results = []
    for aid in ids:
        ok, payload = advance_account(conn, aid, target_date)
        results.append({"ok": ok, "payload": payload})
    return results


def inspect_all(conn, target_date=None):
    """只读一遍所有账户的水位，不推进、不写事件。

    供风控面板打开时展示"在途账户离强平线还有多远"——监控不该只在出事后才有东西看。
    """
    ids = [r["id"] for r in conn.execute(
        "SELECT id FROM follow_accounts").fetchall()]
    results = []
    for aid in ids:
        ok, payload = advance_account(conn, aid, target_date, dry_run=True)
        results.append({"ok": ok, "payload": payload})
    return results


# --- 发帖审核：规则层 + 可选 LLM 层 ------------------------------------------
# 规则层永远先跑、断网可用；LLM 层只在配置了**另一家** provider 时启用。
# 这一层同时承担两个作用：
#   1) 把原来"预留的 LLM 审核"填实（post_moderation.llm_verdict 不再恒为 NULL）；
#   2) 让命题里的"多个大模型"落到实处——情绪面/技术面用智谱，审核用另一家，
#      避免整个系统只有一个 provider 的口径风险。
MOD_PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/chat/completions",
                 "DEEPSEEK_API_KEY", "deepseek-chat"),
    "dashscope": ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                  "DASHSCOPE_API_KEY", "deepseek-v4.1-flash"),
}

MOD_PROMPT = (
    "你是股票社区的内容审核员。判断下面这条帖子是否属于违规推广："
    "承诺收益、诱导加群或加微信、代客理财、荐股收费、拉人开户。\n"
    "只输出 JSON，不要解释，不要代码块包裹："
    '{"violation": true 或 false, "reason": "一句话依据"}\n'
    "帖子内容：{text}"
)


def _llm_settings():
    """读环境变量，返回可用配置；未配置则返回 None（=这一层不启用）。

    MODERATOR_PROVIDER 决定"走哪条通道"，MODERATOR_MODEL 决定"这条通道上调哪个模型"。
    百炼一条通道挂着 qwen / deepseek / kimi / glm 多家模型，所以两者必须分开配，
    否则换个模型就得改代码。
    """
    name = (os.environ.get("MODERATOR_PROVIDER") or "").strip().lower()
    conf = MOD_PROVIDERS.get(name)
    if not conf:
        return None
    url, key_env, default_model = conf
    key = os.environ.get(key_env)
    if not key:
        return None
    model = (os.environ.get("MODERATOR_MODEL") or "").strip() or default_model
    return {"provider": name, "url": url, "key": key, "model": model}


def _llm_moderate(text):
    """返回 (violation: bool|None, verdict_text: str, note: str)。

    violation=None 表示这一层没跑起来（未配置 / 网络失败 / 解析失败），
    此时必须回落到"仅规则层"，并在 note 里写明原因，不能静默当作通过。
    """
    s = _llm_settings()
    if s is None:
        return None, None, "LLM 层未启用（需设 MODERATOR_PROVIDER 与该家 key）"

    import json
    try:
        import requests
    except ImportError:
        return None, None, "LLM 层未启用（requests 不可用）"

    body = {
        "model": s["model"],
        "temperature": 0.0,
        "max_tokens": 400,
        "messages": [{"role": "user",
                      "content": MOD_PROMPT.replace("{text}", (text or "")[:500])}],
    }
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + s["key"]}
    try:
        r = requests.post(s["url"], headers=headers, json=body, timeout=(10, 30))
    except Exception as e:  # 网络类失败：不重试，回落到规则层并记录原因
        return None, None, "LLM 调用失败（%s）" % type(e).__name__
    if r.status_code != 200:
        return None, None, "LLM 返回 HTTP %d" % r.status_code

    try:
        content = r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None, None, "LLM 响应结构异常"

    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        obj = json.loads(raw)
        violation = bool(obj.get("violation"))
        reason = str(obj.get("reason") or "")[:120]
    except Exception:
        return None, None, "LLM 输出无法解析为 JSON"

    verdict = "%s/%s: %s" % (s["provider"], s["model"],
                             json.dumps({"violation": violation, "reason": reason},
                                        ensure_ascii=False))
    return violation, verdict, ""


def moderate(content):
    """发帖审核：规则层优先（断网可用），规则未命中时再走 LLM 层。

    返回 (decision, rule_hit, detail, llm_verdict)。
    命中任一规则即 BLOCKED，不再调 LLM——规则是硬性红线，不需要模型再确认一次。
    """
    text = content or ""
    for name, pat in MODERATION_RULES:
        m = pat.search(text)
        if m:
            return ("BLOCKED", name,
                    "命中规则 %s，匹配文本：%s" % (name, m.group(0)), None)

    violation, verdict, note = _llm_moderate(text)
    if violation is True:
        return "BLOCKED", "llm_moderation", "LLM 判定为违规推广", verdict
    if violation is False:
        return "APPROVED", None, "规则层与 LLM 层均未命中", verdict
    # 这一层没跑起来：如实说明只走了规则层，不假装"双保险"
    return "APPROVED", None, "仅规则层（%s）" % note, note


"""编排层：三视角冲突检测 → 取证分支 → 综合研判。

落库口径：view='fusion'，prompt_version='fusion-orch-v1'，model='orchestrator-v1'。

三条边界（答辩必须能讲清，也是本脚本存在的理由）：
1. 本脚本产出的是"人读的综合研判与解释"，不是回测信号。
   回测仍读 fusion-val-v1 的冻结分数，两者互不影响。
2. 一致日走规则合成（不调 LLM），冲突日才进取证分支。
   这样绝大多数结果可复现，LLM 只在真正需要判断的地方介入。
3. 取证分支的证据全部来自本地库（新闻标题、近 5 日涨跌、估值分位），
   不引入外部实时数据，断网也能演示。

与 pipeline 的区别（这是"编排器"而非"流水线"的地方）：
- 它会根据三视角的离散程度决定"要不要额外取证"；
- 需要取证时自己去查三张表，而不是把固定输入喂给模型；
- 输出带 branch 字段，能回答"哪些日子走了冲突分支"。

用法：
    python scripts/orchestrate.py --start 2024-12-31 --end 2024-12-31 --limit 50
    python scripts/orchestrate.py --start 2024-08-08 --end 2024-12-31 --workers 6
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

MODEL_TAG = "orchestrator-v1"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
PRIMARY_MODEL = "glm-4-flash"     # 不思考、单次约 2 秒、200 token 内出结果，适合短结构化判断
FALLBACK_MODEL = "glm-4.5-flash"  # 主模型失败时的备用（带思考，较慢）
TEMPERATURE = 0.0

# 视角口径：必须带 prompt_version，否则同一 view 的多版本会混排。
# spec=rule  → 规则/本地版三视角，落 fusion-orch-v1（历史口径，保持不变，可复现）
# spec=agent → 技术面/基本面改用智能体版本（DeepSeek 判），落 fusion-orch-agent-v1
#              情绪面仍用 sent-v1：sent-agent-v1 只有 7 只 × 12 天试点，
#              切过去会让其余 43 只直接失去情绪面，反而更不像“三视角”。
#              情绪面本质是单次 LLM 对新闻打分，不存在“智能体版”的强需求。
VIEW_SPEC_RULE = {"tech": "tech-v1", "fund": "fund-rule-v1", "sent": "sent-v1"}
VIEW_SPEC_AGENT = {"tech": "tech-agent-v1", "fund": "fund-agent-v1", "sent": "sent-v1"}

_CFG = {
    "spec": "rule",
    "prompt_version": "fusion-orch-v1",
    "view_spec": VIEW_SPEC_RULE,
    "sent_max_lag": 5,
}


def configure(spec, sent_max_lag=5, prompt_version=None):
    """按 spec 切换视角与落库版本号，并设置情绪面外推时效上限。

    agent 版写 fusion-orch-agent-v1，与 v1 并存：judgements 的
    UNIQUE(symbol,date,view,prompt_version) 不含 model，版本名不同即两套记录共存。
    prompt_version 显式传入时可再起一套（例如换了裁决规则要留旧证据对比）。
    """
    if spec == "agent":
        _CFG["spec"] = "agent"
        _CFG["prompt_version"] = prompt_version or "fusion-orch-agent-v1"
        _CFG["view_spec"] = VIEW_SPEC_AGENT
    else:
        _CFG["spec"] = "rule"
        _CFG["prompt_version"] = prompt_version or "fusion-orch-v1"
        _CFG["view_spec"] = VIEW_SPEC_RULE
    _CFG["sent_max_lag"] = sent_max_lag

# 这里刻意不设分数阈值：三视角 score 尺度不同（概率偏离 / 估值分位 / 新闻倾向），
# 用极差判冲突会把几乎每天都判成冲突（实测 96%），详见 decide() 的说明。

_STATS = {"in_tokens": 0, "out_tokens": 0, "llm_calls": 0}
_LOCK = threading.Lock()


def api_key():
    return os.environ.get("ZHIPU_KEY", "").strip()


def load_views(conn, symbol, date):
    """读三视角当日研判；情绪面允许向前取最近一次，但受时效上限约束。

    为什么必须设时效：情绪信号有效期只有几天（验收标准写的是 3–5 天）。
    实测无约束时，600 条编排里有 563 条（93.8%）用的是外推旧值，最远推到
    58 天前——等于拿 11 月初的新闻情绪判断 12 月底行情，这不是"三视角研判"。
    超期即视为缺失，让当日降级为 tech+fund 两视角：宁缺，不假。
    """
    views = {}
    max_lag = _CFG["sent_max_lag"]
    for view, pv in _CFG["view_spec"].items():
        if view == "sent":
            row = conn.execute(
                "SELECT symbol,date,signal,score,reason,prompt_version FROM judgements "
                "WHERE view='sent' AND symbol=? AND date<=? AND date>=date(?, ?) "
                "AND prompt_version=? ORDER BY date DESC LIMIT 1",
                (symbol, date, date, "-%d days" % max_lag, pv),
            ).fetchone()
            if row is not None:
                views["sent"] = dict(row)
                views["sent"]["ffill"] = (row["date"] != date)
            continue
        row = conn.execute(
            "SELECT symbol,date,signal,score,reason,prompt_version FROM judgements "
            "WHERE view=? AND symbol=? AND date=? AND prompt_version=?",
            (view, symbol, date, pv),
        ).fetchone()
        if row is not None:
            views[view] = dict(row)
    return views


def decide(views):
    """根据三视角的方向组合决定走哪条分支（这是编排器与流水线的本质差别）。

    为什么不用分数极差：三个视角的 score 尺度天然不同（技术面是概率偏离，
    基本面是估值分位，情绪面是新闻倾向），极差几乎恒大于阈值，会把每天都
    判成冲突。真正的矛盾只有一种——同一标的同时出现看多和看空。
    """
    if "tech" not in views:
        return "no_tech"
    signals = {v["signal"] for v in views.values()}
    if len(signals) == 1:
        return "unanimous"
    if "BUY" in signals and "SELL" in signals:
        return "conflict"
    return "minor"


def gather_evidence(conn, symbol, date, k=3):
    """取证分支：自己去查新闻、行情、估值三张表，而不是把固定输入喂给模型。"""
    ev = {}

    rows = conn.execute(
        "SELECT title, content FROM news WHERE symbol=? AND date=? ORDER BY id LIMIT ?",
        (symbol, date, k),
    ).fetchall()
    ev["news"] = [
        (r["title"] or (r["content"] or "")[:80]) for r in rows
    ]

    rows = conn.execute(
        "SELECT date, close FROM prices WHERE symbol=? AND date<=? "
        "ORDER BY date DESC LIMIT 6",
        (symbol, date),
    ).fetchall()
    if len(rows) >= 2:
        newest, oldest = rows[0]["close"], rows[-1]["close"]
        if oldest:
            ev["ret5"] = round((newest / oldest - 1.0) * 100, 2)
            ev["ret5_dates"] = "%s~%s" % (rows[-1]["date"], rows[0]["date"])

    row = conn.execute(
        "SELECT pe_ttm, pb FROM valuations WHERE symbol=? AND date<=? "
        "ORDER BY date DESC LIMIT 1",
        (symbol, date),
    ).fetchone()
    if row is not None:
        ev["pe_ttm"] = row["pe_ttm"]
        ev["pb"] = row["pb"]
    return ev


class _ReviewNeeded(Exception):
    """模型命中 R5（证据不足）时主动拒绝裁决。

    这是一个哨兵，不是错误：它把"证据不够、不该硬判"和"模型乱猜一个方向"
    区分开。上层收到它就降级到规则分支，并在 branch 标签里留下 review 原因，
    而不是让一个没有依据的方向流进回测。
    """


ARB_PROMPT = """同一标的同日三个视角的研判如下，三者方向可能互相矛盾。你的任务不是重新预测涨跌，而是在给定裁决规则里挑一条执行，并说明用了哪条。

{views}

本地证据：新闻{news}；近5日涨跌{ret5}%；PE{pe}；PB{pb}。

【前置约束 P1（优先于下面所有规则）】
基本面估值（PE、PB 及其分位）不构成任何方向理由：fund 权重为 0，即使它与技术面完全对立，
也不得据此翻向。R1 所说的"证据"只指近5日涨跌与新闻，**不含估值**。
（这条是实测补的：第一版没写死，模型把"估值分位高位"当 R1 的证据，26 条裁决里 23 条判 SELL。）

【裁决规则】按顺序匹配，命中第一条即停止，不要继续往下比：

R1 证据压倒：三视角中出现方向对立，且近5日涨跌或新闻明显只支持其中一方 → 采信被支持的那一方。
注意：PE/PB 估值不能作为本规则的证据；若唯一支撑只来自估值，则不得用 R1，继续往下判。

R2 技术优先：技术面与其他视角方向相反，且技术面 |score| >= 0.05 → 采信技术面。
依据：验证集标定把 tech 权重定到 0.90、fund 归零、sent 仅 0.10，技术面是唯一被验证有效的方向源。

R3 情绪不翻向：情绪面与其他视角相反、且对立只来自情绪面 → 不翻向，方向跟随技术面，只按情绪面强弱调整 final_score 幅度（增量绝对值不超过 0.1）。
依据：情绪面方向准确率 0.5054，接近随机，权重仅 0.10。

R4 基本面不参与：fund 权重为 0，其方向不构成翻向理由；即使它与技术面完全对立，也不因此改变方向。

R5 证据不足 → REVIEW：出现下列任一情形，不要硬给方向，直接输出 final_signal="REVIEW"、final_score=0.0：
(a) 技术面缺失或 score 恰为 0；
(b) 对立双方都没有本地证据支撑；
(c) 新闻为空且 PE、PB 均为 NA。

【硬约束】
- final_signal 只能取 BUY、SELL、HOLD、REVIEW 之一。
- 不得输出三个视角里都没出现过的方向。
- 不得引用上面没给出的数字；why 里出现的数字必须能在输入里找到。
- why 不超过 40 字，且必须写明命中的规则编号（如 "R2"）。

【输出格式】只输出一个 JSON 对象，不要解释、不要 markdown、不要代码块。
输出示例（仅示意格式，数值无意义）：
{{"final_signal": "HOLD", "final_score": 0.0, "conflict": true, "rule": "R5", "why": "R5 新闻为空且估值 NA，证据不足待复核"}}"""


def arbitrate_llm(views, ev, key, retries=2):
    """冲突分支：调模型做仲裁。温度 0，同一输入同一输出。

    该账号在并发下容易被限流（429），所以这里带退避重试；
    重试仍失败则交给上层降级到规则分支，并在分支标签里留下原因。
    """
    payload_views = []
    for name, v in views.items():
        payload_views.append({
            "view": name,
            "signal": v["signal"],
            "score": round(float(v["score"]), 4),
            "reason": (v.get("reason") or "")[:80],
            "ffill": bool(v.get("ffill")),
        })
    prompt = ARB_PROMPT.format(
        views=json.dumps(payload_views, ensure_ascii=False),
        news=json.dumps(ev.get("news", []), ensure_ascii=False),
        ret5=ev.get("ret5", "NA"),
        pe=ev.get("pe_ttm", "NA"),
        pb=ev.get("pb", "NA"),
    )

    model = PRIMARY_MODEL
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                BASE_URL + "/chat/completions",
                headers={"Authorization": "Bearer " + key},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": TEMPERATURE,
                    "max_tokens": 400,
                },
                timeout=60,
            )
            resp.raise_for_status()
            j = resp.json()
            usage = j.get("usage") or {}
            with _LOCK:
                _STATS["in_tokens"] += usage.get("prompt_tokens", 0)
                _STATS["out_tokens"] += usage.get("completion_tokens", 0)
                _STATS["llm_calls"] += 1
            ch = j["choices"][0]
            text = (ch["message"].get("content") or "").strip()
            if not text:
                # 思考型模型把额度全用在 reasoning 时正文会为空，显式报错而不是静默降级
                raise ValueError("empty content (finish_reason=%s)" % ch.get("finish_reason"))
            text = text.replace("```json", "").replace("```", "").strip()
            out = json.loads(text)
            sig = out.get("final_signal")
            if sig == "REVIEW":
                # 模型按 R5 主动拒绝裁决：不落方向，交给上层降级到规则分支。
                # 必须先于 bad_signal 判定，否则会被当成非法输出反复重试。
                raise _ReviewNeeded(out.get("why") or "")
            if sig not in ("BUY", "SELL", "HOLD"):
                raise ValueError("bad final_signal: %r" % sig)
            score = float(out.get("final_score", 0.0))
            score = max(-1.0, min(1.0, score))
            # P1 硬校验：R1 的证据只含近5日涨跌与新闻，不含估值。模型若用 R1 却拿
            # PE/PB/分位当理由，说明它把权重为 0 的基本面当成了翻向依据，按 R5 处理。
            # 只在 prompt 里写约束拦不住（实测 26 条里 23 条就是这么判 SELL 的）。
            rule = (out.get("rule") or "").upper()
            why = out.get("why") or ""
            if rule == "R1" and any(k in why for k in ("PE", "PB", "pe", "pb", "估值", "分位")):
                raise _ReviewNeeded("R1 违规引用估值: " + why[:30])
            # 符号一致性校正：模型偶尔给出与方向相反的分数（如 BUY 配 -0.3），
            # 落库前按方向归正，避免 signal 与 score 互相矛盾。
            if sig == "BUY" and score < 0.05:
                score = 0.1 if score >= 0 else -score
            elif sig == "SELL" and score > -0.05:
                score = -0.1 if score <= 0 else -score
            elif sig == "HOLD":
                # HOLD 保留模型给出的原始分值：它表示"方向不明确但强弱有别"，
                # 压成 0 会让 30 多条记录在展示上失去梯度。
                pass
            return sig, score, out, "zhipu", model
        except (requests.Timeout, requests.HTTPError, ValueError, json.JSONDecodeError) as e:
            last_err = e
            if isinstance(e, requests.HTTPError):
                status = e.response.status_code if e.response is not None else "?"
                # 4xx 里只有限流值得重试，其余直接降级
                if status != 429:
                    raise
            if attempt < retries:
                time.sleep(8 * (attempt + 1))
                continue
            raise
    raise last_err


def arbitrate_rule(views, ev, branch="minor"):
    """无 key 或调用失败时的兜底：按标定权重合成，文案随分支不同。

    这个兜底不是"假装智能"，而是保证脚本在没有凭据时也能跑出可解释结果。
    """
    if "tech" not in views:
        return "HOLD", 0.0, {"why": "缺技术面，拒绝给出方向"}, "local", "orch-rule"

    ws = {"tech": 1.0, "fund": 0.0, "sent": 0.5}   # 与标定权重一致：基本面权重为 0
    num = 0.0
    den = 0.0
    for name, v in views.items():
        w = ws.get(name, 0.0)
        num += w * float(v["score"])
        den += w
    score = num / den if den else 0.0

    if branch == "unanimous":
        why = "三视角方向一致，按标定权重合成"
    elif branch == "conflict":
        why = "三视角多空对立，按标定权重合成（未启用模型仲裁）"
    else:
        why = "存在分歧但未形成多空对立，按标定权重合成"
    ret5 = ev.get("ret5")
    if ret5 is not None:
        why += "（近5日 %+.2f%%）" % ret5
    sig = "BUY" if score > 0.05 else ("SELL" if score < -0.05 else "HOLD")
    return sig, round(score, 4), {"why": why}, "local", "orch-rule"


def orchestrate(conn, symbol, date, key, dry_run=False):
    views = load_views(conn, symbol, date)
    branch = decide(views)

    if branch == "no_tech":
        return None
    if dry_run:
        # 只走分支判定，不调 LLM、不写库：跑量前用它在零成本下估调用次数
        return {"symbol": symbol, "date": date, "branch": branch + "_dry",
                "signal": None, "score": None, "n_views": len(views)}
    if branch in ("unanimous", "minor"):
        sig, score, out, prov, model = arbitrate_rule(views, {}, branch)
        branch_used = branch
    else:
        ev = gather_evidence(conn, symbol, date)
        if key:
            try:
                sig, score, out, prov, model = arbitrate_llm(views, ev, key)
                branch_used = "conflict_llm"
            except _ReviewNeeded as e:
                # 模型按 R5 判定证据不足：不能退回规则加权去硬算方向——那等于把模型
                # 刚说的"证据不足"抹掉，还照样开仓（实测出现过 REVIEW 后算出 BUY +0.15）。
                # 正确动作是保守 HOLD + 0 分：不开仓，把 review 原因留痕待人工复核。
                sig, score = "HOLD", 0.0
                out = {"why": "R5 证据不足，保守 HOLD 待复核"}
                prov, model = "local", "orch-review"
                branch_used = "conflict_review_hold(review:%s)" % (str(e)[:40] or "no_reason")
            except Exception as e:
                sig, score, out, prov, model = arbitrate_rule(views, ev, "conflict")
                branch_used = "conflict_rule(fallback:%s)" % type(e).__name__
        else:
            sig, score, out, prov, model = arbitrate_rule(views, ev, "conflict")
            branch_used = "conflict_rule"

    reason = "[%s] %s" % (branch_used, out.get("why", ""))
    db.upsert_judgement(
        conn, symbol, date, "fusion", sig, score,
        reason=reason, provider=prov, model=model,
        temperature=(TEMPERATURE if prov == "zhipu" else None),
        prompt_version=_CFG["prompt_version"],
        source=("live" if prov == "zhipu" else "offline_rule"),
    )
    return {"symbol": symbol, "date": date, "branch": branch_used,
            "signal": sig, "score": round(score, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--spec", default="rule", choices=["rule", "agent"],
                    help="rule=规则/本地版三视角；agent=技术面与基本面走智能体版本")
    ap.add_argument("--out", default="docs/exp_orchestrator.json",
                    help="结果 json 路径（相对仓库根）")
    ap.add_argument("--dry_run", action="store_true",
                    help="只统计分支分布，不调 LLM、不写库")
    ap.add_argument("--sent_max_lag", type=int, default=5,
                    help="情绪面外推时效上限（天），超期视为缺失；默认 5")
    ap.add_argument("--prompt_version", default="",
                    help="覆盖落库版本名；留空按 spec 走默认（换了裁决规则时可另起一套留证据）")
    args = ap.parse_args()

    configure(args.spec, args.sent_max_lag, args.prompt_version or None)
    # dry_run 时强制清空 key：即便环境里有凭据也绝不发起调用
    key = "" if args.dry_run else api_key()

    conn = db.connect()
    symbols = [r["symbol"] for r in conn.execute(
        "SELECT DISTINCT symbol FROM prices ORDER BY symbol LIMIT ?", (args.limit,)
    ).fetchall()]
    dates = [r["date"] for r in conn.execute(
        "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
        (args.start, args.end),
    ).fetchall()]

    print("spec=%s pv=%s symbols=%d dates=%d llm=%s dry_run=%s sent_max_lag=%d"
          % (_CFG["spec"], _CFG["prompt_version"], len(symbols), len(dates),
             "on" if key else "off", args.dry_run, _CFG["sent_max_lag"]))

    tasks = [(s, d) for d in dates for s in symbols]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(orchestrate, db.connect(), s, d, key, args.dry_run): (s, d)
                for s, d in tasks}
        for f in as_completed(futs):
            try:
                r = f.result()
                if r:
                    results.append(r)
            except Exception as e:
                print("FAIL", futs[f], type(e).__name__, e)

    from collections import Counter
    c = Counter(r["branch"] for r in results)
    print("\n按分支统计：")
    for k, v in c.most_common():
        print("  %-28s %5d  (%5.1f%%)" % (k, v, 100.0 * v / max(1, len(results))))
    print("\n总计 %d 条，LLM 调用 %d 次，输入 %d tokens，输出 %d tokens"
          % (len(results), _STATS["llm_calls"], _STATS["in_tokens"], _STATS["out_tokens"]))

    if args.dry_run:
        # dry_run 只用于估调用次数，不落盘：默认 --out 指向 docs/exp_orchestrator.json，
        # 而那个文件存着已提交的 rule 版实验结果，写一次就把历史证据冲了（踩过）。
        print("\n(dry_run：不写结果文件)")
        return

    pout = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(pout), exist_ok=True)
    with open(pout, "w", encoding="utf-8") as f:
        json.dump({
            "prompt_version": _CFG["prompt_version"],
            "spec": _CFG["spec"],
            "view_spec": _CFG["view_spec"],
            "sent_max_lag_days": _CFG["sent_max_lag"],
            "window": [args.start, args.end],
            "symbols": len(symbols),
            "dates": len(dates),
            "dry_run": bool(args.dry_run),
            "branch_counts": dict(c),
            "llm_stats": _STATS,
            "note": "本层只用于解释与展示，不进入回测信号链；回测仍读 fusion-val-v1",
        }, f, ensure_ascii=False, indent=2)
    print("written", pout)


if __name__ == "__main__":
    main()

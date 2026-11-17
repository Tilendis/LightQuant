"""技术面 LLM 解读层：在本地模型分数之上，让大模型结合行情指标给出技术面研判。

为什么需要这一层
----------------
课题要求"调度多个大模型（或不同分析视角的智能体）分别从技术面、基本面和情绪面
生成研判报告"。原实现里技术面的方向完全由本地 BiLSTM 决定，reason 是模板拼接
（"BiLSTM 估计次日上涨概率 43.3%…"），严格说这一路并没有经过大模型。

本层不替代模型，而是把**模型分 + 近期行情指标**交给 LLM，由它确认或修正方向
并写出可读理由。这样三个视角都真正经过大模型，同时保留"底分来自模型"的可追溯性。

落库口径
--------
view='tech'，prompt_version='tech-llm-v1'，model='glm-4-flash'，source='live'。
与技术面原版（prompt_version='tech-v1'，model='bi_lstm_feat10'）**共存**：
judgements 表的 UNIQUE(symbol,date,view,prompt_version) 保证两个版本互不覆盖，
回测读哪一版由 --prompt_version 决定。

用法
----
    python scripts/gen_tech_llm.py --start 2024-12-31 --end 2024-12-31 --limit 50
    python scripts/gen_tech_llm.py --start 2024-08-14 --end 2024-12-31 --workers 4
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

PROMPT_VERSION = "tech-llm-v1"
PROMPT_VERSION_BLIND = "tech-llm-blind-v1"   # 盲判版：不给模型方向，检验 LLM 的独立判断力
BASE_PROMPT_VERSION = "tech-v1"     # 本地模型版，读它作为输入
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MODEL = "glm-4-flash"               # 不思考、单次约 2 秒，适合短结构化输出
TEMPERATURE = 0.0
MAX_TOKENS = 400

VALID_SIGNALS = ("BUY", "SELL", "HOLD")

_STATS = {"in_tokens": 0, "out_tokens": 0, "llm_calls": 0, "failed": 0, "fallback": 0}
_LOCK = threading.Lock()
_LOCAL = threading.local()


def api_key():
    return os.environ.get("ZHIPU_KEY", "").strip()


def load_base(conn, symbol, date):
    """读本地模型的技术面判断（tech-v1），本层的输入。"""
    row = conn.execute(
        "SELECT symbol,date,signal,score,reason,model FROM judgements "
        "WHERE view='tech' AND symbol=? AND date=? AND prompt_version=?",
        (symbol, date, BASE_PROMPT_VERSION),
    ).fetchone()
    return dict(row) if row else None


def load_features(conn, symbol, date):
    """从 prices 表现算技术指标：涨跌幅、均线偏离、量比、波动率。

    只用 date <= 目标日的数据，避免把未来信息喂进当日的研判。
    """
    rows = conn.execute(
        "SELECT date, close, volume FROM prices WHERE symbol=? AND date<=? "
        "ORDER BY date DESC LIMIT 21",
        (symbol, date),
    ).fetchall()
    if len(rows) < 6:
        return None
    closes = [r["close"] for r in rows]
    vols = [r["volume"] for r in rows]

    def ret(n):
        if len(closes) <= n:
            return None
        return round((closes[0] / closes[n] - 1.0) * 100, 2)

    ma20 = sum(closes) / len(closes) if closes else None
    ma20_dev = round((closes[0] / ma20 - 1.0) * 100, 2) if ma20 else None

    v5 = sum(vols[:5]) / min(5, len(vols))
    v20 = sum(vols) / len(vols)
    vol_ratio = round(v5 / v20, 2) if v20 else None

    daily = []
    for i in range(min(20, len(closes)) - 1):
        if closes[i + 1]:
            daily.append(closes[i] / closes[i + 1] - 1.0)
    ann_vol = None
    if len(daily) >= 2:
        mean = sum(daily) / len(daily)
        var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
        ann_vol = round(math.sqrt(var) * math.sqrt(242) * 100, 1)

    return {
        "ret5": ret(5),
        "ret20": ret(20),
        "ma20_dev": ma20_dev,
        "vol_ratio": vol_ratio,
        "ann_vol": ann_vol,
        "window": "%s~%s" % (rows[-1]["date"], rows[0]["date"]),
        "n": len(rows),
    }


def _fmt(v):
    """把可能缺失的特征写成 NA，而不是把 None 原样塞进 prompt。"""
    return "NA" if v is None else v


PROMPT = """你是 A 股技术面分析师。请结合本地模型的判断与近期行情统计，给出该标的当日的技术面研判。

标的：{symbol}
日期：{date}
本地模型判断：{signal}（分数 {score}，正数偏多、负数偏空）
近 5 个交易日涨跌：{ret5}%
近 20 个交易日涨跌：{ret20}%
收盘价相对 20 日均线偏离：{ma20_dev}%
近 5 日均量相对 20 日均量：{vol_ratio} 倍
近 20 日年化波动率：{ann_vol}%

只输出一个 JSON 对象，不要任何解释文字，不要 markdown 代码块。
字段为 signal、score、reason 三个。
signal 只能是 BUY 或 SELL 或 HOLD 三个单词之一，直接写单词本身。
score 取 -1 到 1 之间的数字，正数表示看多、负数表示看空，方向必须与 signal 一致；判 HOLD 时取接近 0 的值。
reason 不超过 40 个汉字，必须引用上面至少一个具体数值。"""

BLIND_PROMPT = """你是 A 股技术面分析师。请只根据下列行情统计，给出该标的当日的技术面研判。

标的：{symbol}
日期：{date}
近 5 个交易日涨跌：{ret5}%
近 20 个交易日涨跌：{ret20}%
收盘价相对 20 日均线偏离：{ma20_dev}%
近 5 日均量相对 20 日均量：{vol_ratio} 倍
近 20 日年化波动率：{ann_vol}%

只输出一个 JSON 对象，不要任何解释文字，不要 markdown 代码块。
字段为 signal、score、reason 三个。
signal 只能是 BUY 或 SELL 或 HOLD 三个单词之一，直接写单词本身。
score 取 -1 到 1 之间的数字，正数表示看多、负数表示看空，方向必须与 signal 一致；判 HOLD 时取接近 0 的值。
reason 不超过 40 个汉字，必须引用上面至少一个具体数值。"""


def _session():
    """每个线程复用一条连接，避免上千次请求反复做 TLS 握手（实测会成片 SSLError）。"""
    s = getattr(_LOCAL, "s", None)
    if s is None:
        s = requests.Session()
        _LOCAL.s = s
    return s


def _call(prompt, key, retries=3):
    """调模型。区分两类失败：429 走长退避，网络类错误走短退避，其余 4xx 直接抛出。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = _session().post(
                BASE_URL + "/chat/completions",
                headers={"Authorization": "Bearer " + key},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                },
                timeout=(10, 60),
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
                raise ValueError("empty content (finish_reason=%s)" % ch.get("finish_reason"))
            text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if status != 429:
                raise        # 余额不足、参数错误等重试无意义
            last_err = e
            if attempt < retries:
                time.sleep(8 * (attempt + 1))
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.SSLError,
                requests.exceptions.ChunkedEncodingError,
                ValueError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last_err


def normalize(out):
    """校验并归正模型输出：signal 枚举、score 范围、方向与分数一致性。

    模型偶尔会把三个候选值整串写进 signal（"BUY|SELL|HOLD"），
    或给出与方向相反的分数；这两种都要在落库前拦下。
    """
    sig = str(out.get("signal", "")).strip().strip('"').upper()
    if sig not in VALID_SIGNALS:
        # 容错：若文本里只出现唯一一个合法单词，取其
        hits = [s for s in VALID_SIGNALS if s in sig]
        if len(hits) == 1:
            sig = hits[0]
        else:
            raise ValueError("bad signal: %r" % out.get("signal"))

    score = float(out.get("score", 0.0))
    score = max(-1.0, min(1.0, score))
    if sig == "BUY" and score < 0:
        score = -score
    elif sig == "SELL" and score > 0:
        score = -score
    elif sig == "HOLD":
        score = max(-0.2, min(0.2, score))   # HOLD 保留梯度，但不让它冒充方向

    reason = str(out.get("reason", "")).strip().replace("\n", " ")
    if len(reason) > 60:
        reason = reason[:60]
    if not reason:
        raise ValueError("empty reason")
    return sig, score, reason


def analyze(conn, symbol, date, key, blind=False):
    base = load_base(conn, symbol, date)
    if base is None:
        return None, "no_base"
    feat = load_features(conn, symbol, date)
    if feat is None:
        return None, "no_prices"

    pv = PROMPT_VERSION_BLIND if blind else PROMPT_VERSION
    if blind:
        prompt = BLIND_PROMPT.format(
            symbol=symbol, date=date,
            ret5=_fmt(feat["ret5"]), ret20=_fmt(feat["ret20"]),
            ma20_dev=_fmt(feat["ma20_dev"]), vol_ratio=_fmt(feat["vol_ratio"]),
            ann_vol=_fmt(feat["ann_vol"]),
        )
    else:
        prompt = PROMPT.format(
            symbol=symbol, date=date,
            signal=base["signal"], score=round(float(base["score"]), 4),
            ret5=_fmt(feat["ret5"]), ret20=_fmt(feat["ret20"]),
            ma20_dev=_fmt(feat["ma20_dev"]), vol_ratio=_fmt(feat["vol_ratio"]),
            ann_vol=_fmt(feat["ann_vol"]),
        )

    try:
        out = _call(prompt, key)
        sig, score, reason = normalize(out)
        prov, model, source = "zhipu", MODEL, "live"
    except Exception as e:
        # 失败不静默：明确降级为"模型方向 + 模型理由"，并在 reason 里留下降级标记
        with _LOCK:
            _STATS["failed"] += 1
            _STATS["fallback"] += 1
            _STATS["err_" + type(e).__name__] = _STATS.get("err_" + type(e).__name__, 0) + 1
        sig = base["signal"]
        score = float(base["score"])
        reason = "%s（LLM 降级：%s）" % ((base.get("reason") or "")[:40], type(e).__name__)
        prov, model, source = "local", base.get("model") or "bi_lstm_feat10", "offline_rule"

    db.upsert_judgement(
        conn, symbol, date, "tech", sig, score,
        reason=reason, provider=prov, model=model,
        temperature=(TEMPERATURE if prov == "zhipu" else None),
        prompt_version=pv, source=source,
    )
    return {
        "symbol": symbol, "date": date,
        "base_signal": base["signal"], "llm_signal": sig,
        "base_score": round(float(base["score"]), 4), "llm_score": round(score, 4),
        "same": (base["signal"] == sig),
        "provider": prov,
    }, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true", help="只算特征与 prompt，不调模型、不落库")
    ap.add_argument("--blind", action="store_true",
                    help="盲判模式：不把本地模型方向放进 prompt，用于检验 LLM 的独立判断力")
    args = ap.parse_args()

    key = api_key()
    if not key and not args.dry_run:
        raise SystemExit("缺少 ZHIPU_KEY 环境变量；先 set 再跑，或加 --dry_run 只看输入")

    conn = db.connect()
    symbols = [r["symbol"] for r in conn.execute(
        "SELECT DISTINCT symbol FROM prices ORDER BY symbol LIMIT ?", (args.limit,)
    ).fetchall()]
    dates = [r["date"] for r in conn.execute(
        "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
        (args.start, args.end),
    ).fetchall()]
    print("symbols=%d dates=%d llm=%s" % (len(symbols), len(dates), "on" if key else "dry"))

    tasks = [(s, d) for d in dates for s in symbols]

    if args.dry_run:
        # 只看一条样例的输入与特征，不调模型、不落库
        s, d = tasks[0]
        base = load_base(conn, s, d)
        feat = load_features(conn, s, d)
        print("sample symbol=%s date=%s" % (s, d))
        print("base:", base)
        print("features:", feat)
        if base and feat:
            if args.blind:
                print("\n" + BLIND_PROMPT.format(
                    symbol=s, date=d,
                    ret5=_fmt(feat["ret5"]), ret20=_fmt(feat["ret20"]),
                    ma20_dev=_fmt(feat["ma20_dev"]), vol_ratio=_fmt(feat["vol_ratio"]),
                    ann_vol=_fmt(feat["ann_vol"])))
            else:
                print("\n" + PROMPT.format(
                    symbol=s, date=d, signal=base["signal"],
                    score=round(float(base["score"]), 4),
                    ret5=_fmt(feat["ret5"]), ret20=_fmt(feat["ret20"]),
                    ma20_dev=_fmt(feat["ma20_dev"]), vol_ratio=_fmt(feat["vol_ratio"]),
                    ann_vol=_fmt(feat["ann_vol"])))
        print("\n(dry run，未调模型)")
        return

    results = []
    errors = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(analyze, db.connect(), s, d, key, args.blind): (s, d) for s, d in tasks}
        for f in as_completed(futs):
            try:
                r, status = f.result()
                if r:
                    results.append(r)
                errors[status] += 1
            except Exception as e:
                errors["exception:" + type(e).__name__] += 1
                print("FAIL", futs[f], type(e).__name__, e)

    if results and not args.dry_run:
        n = len(results)
        same = sum(1 for r in results if r["same"])
        print("\n落库 %d 条，方向与本地模型一致 %d 条（%.1f%%）"
              % (n, same, 100.0 * same / n))
        print("LLM 信号分布：", dict(Counter(r["llm_signal"] for r in results)))
        print("本地基线分布：", dict(Counter(r["base_signal"] for r in results)))
        print("LLM 调用 %d 次，输入 %d tokens，输出 %d tokens，降级 %d 条"
              % (_STATS["llm_calls"], _STATS["in_tokens"], _STATS["out_tokens"],
                 _STATS["fallback"]))
        p = os.path.join(ROOT, "docs",
                         "exp_tech_llm_blind.json" if args.blind else "exp_tech_llm.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "prompt_version": PROMPT_VERSION_BLIND if args.blind else PROMPT_VERSION,
                "base_prompt_version": BASE_PROMPT_VERSION,
                "model": MODEL,
                "blind": bool(args.blind),
                "window": [args.start, args.end],
                "symbols": len(symbols),
                "dates": len(dates),
                "rows": n,
                "agree_with_base": same,
                "agree_rate": round(same / n, 4),
                "llm_signal_counts": dict(Counter(r["llm_signal"] for r in results)),
                "base_signal_counts": dict(Counter(r["base_signal"] for r in results)),
                "llm_stats": _STATS,
                "note": ("技术面 LLM 盲判：prompt 不含模型方向，用于检验 LLM 的独立判断力"
                         if args.blind else
                         "技术面 LLM 解读层：底分来自本地 BiLSTM，LLM 确认或修正方向并写出理由"),
            }, f, ensure_ascii=False, indent=2)
        print("written", p)

    print("\n状态统计：", dict(errors))


if __name__ == "__main__":
    main()

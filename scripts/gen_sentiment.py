"""情绪面批量研判：新闻摘要 -> 结构化分数 -> judgements(view='sent')

设计要点（都是这个项目里踩过的坑）：
- 幂等：UNIQUE(symbol, date, view, prompt_version)，重复跑不会写两条
- 并发只用于调 API，写库统一回主线程；SQLite 不做多线程写
- 主模型 glm-5.3-flash（当前走实名赠送的 500 万 token 体验包）
- 调用失败降级 glm-4.7-flash（免费但白天限流严重）
- 每条记录都带 provider/model/temperature/prompt_version，答辩可溯源
- 另外统计"新闻摘要里含涨跌词的比例"，用于判断情绪面是否真独立于价格序列

用法：
    python scripts/gen_sentiment.py --start 2024-08-08 --end 2024-12-31 --limit 50
    python scripts/gen_sentiment.py --start 2024-03-15 --end 2024-12-31 --workers 12
"""

import argparse
import glob
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

NEWS_ROOT = os.path.join(ROOT, "dataset", "CSMD50", "news")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
PROMPT_VERSION = "sent-v1"
PRIMARY_MODEL = "glm-5.3-flash"
FALLBACK_MODEL = "glm-4.7-flash"
TEMPERATURE = 0.0

# 用于判断情绪面是否只是价格的滞后映射：摘要里出现这些词，说明文本已含当日行情
PRICE_WORDS = ["上涨", "下跌", "涨停", "跌停", "涨幅", "跌幅", "收涨", "收跌",
               "大涨", "大跌", "新高", "新低", "反弹", "回调", "走强", "走弱"]

PROMPT_TMPL = """你是A股财经新闻情绪分析器。下面是该股票在 {date} 的新闻摘要。判断其对下一交易日股价的情绪倾向。

只输出一个 JSON 对象，不要输出解释、不要 markdown、不要代码块。字段：
{{"score": 浮点数(-1到1，正为利好，负为利空，无信息填0), "confidence": 浮点数(0到1), "reason": 不超过30字, "evidence": [1-2条最关键的依据片段]}}

新闻摘要：
{news}"""

_STATS = {"out_tokens": 0, "in_tokens": 0}
_LOCK = threading.Lock()


def api_key():
    key = os.environ.get("ZHIPU_KEY", "").strip()
    if not key:
        raise SystemExit("缺少环境变量 ZHIPU_KEY（不要写进代码或文档，只在命令行里设）")
    return key


def read_news(path):
    """读一份当日新闻 CSV，返回摘要列表。列名固定是 code_name/ticker/created_at/text。"""
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "text" not in df.columns:
        return []
    out = []
    for t in df["text"].dropna():
        s = str(t).strip()
        if s:
            out.append(s[:220])
    return out


def call_once(model, prompt, key, timeout=45):
    body = {"model": model, "temperature": TEMPERATURE,
            "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(
        BASE_URL + "/chat/completions",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        json=body, timeout=timeout,
    )
    if r.status_code != 200:
        return None, "HTTP%d %s" % (r.status_code, r.text[:80].replace("\n", " "))
    data = r.json()
    txt = data["choices"][0]["message"]["content"]
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return None, "no_json"
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        return None, "json_err %s" % e
    u = data.get("usage") or {}
    with _LOCK:
        _STATS["in_tokens"] += int(u.get("prompt_tokens") or 0)
        _STATS["out_tokens"] += int(u.get("completion_tokens") or 0)
    return obj, None


def judge(symbol, date, texts, key):
    prompt = PROMPT_TMPL.format(date=date, news="\n".join("- " + t for t in texts))
    last = "unknown"
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        for attempt in range(2):
            try:
                obj, err = call_once(model, prompt, key)
            except Exception as e:
                obj, err = None, "EXC %s" % type(e).__name__
            if obj is not None:
                try:
                    score = float(obj.get("score", 0.0))
                except Exception:
                    score = 0.0
                score = max(-1.0, min(1.0, score))
                conf = obj.get("confidence")
                reason = str(obj.get("reason", ""))[:60]
                return {"symbol": symbol, "date": date, "score": score,
                        "reason": reason, "model": model, "confidence": conf}, None
            last = err or "unknown"
            if err and err.startswith("HTTP4") and "1305" not in err:
                break
            time.sleep(1.5)
    return None, last


def score_to_signal(score, th=0.2):
    if score >= th:
        return "BUY"
    if score <= -th:
        return "SELL"
    return "HOLD"


def collect_tasks(start, end, only_missing, existing, conn):
    tasks = []
    for sym in sorted(os.listdir(NEWS_ROOT)):
        d = os.path.join(NEWS_ROOT, sym)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            date = os.path.basename(f)[:-4]
            if not (start <= date <= end):
                continue
            if (sym, date) in existing:
                continue
            texts = read_news(f)
            if not texts:
                continue
            tasks.append((sym, date, texts, f))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-08-08")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--limit", type=int, default=0, help="0 表示不限")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="忽略已有记录重算")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    key = api_key()
    conn = db.connect(args.db)
    db.init_db(conn=conn)

    existing = set()
    if not args.force:
        rows = conn.execute(
            "SELECT symbol, date FROM judgements WHERE view='sent' AND prompt_version=?",
            (PROMPT_VERSION,)).fetchall()
        existing = {(r["symbol"], r["date"]) for r in rows}
    print("已有 sent 记录:", len(existing))

    tasks = collect_tasks(args.start, args.end, not args.force, existing, conn)
    total = len(tasks)
    if args.limit:
        tasks = tasks[:args.limit]
    print("待处理:", len(tasks), "/ 区间内总任务:", total, "并发:", args.workers)

    # 先做一次文本层面的独立性检查：摘要里是不是已经写了当日涨跌
    price_hit = sum(1 for _, _, t, _ in tasks if any(w in " ".join(t) for w in PRICE_WORDS))
    print("含涨跌词的新闻占比: %.1f%% (%d/%d)" % (
        100.0 * price_hit / max(1, len(tasks)), price_hit, max(1, len(tasks))))

    if not tasks:
        print("没有待处理任务，退出")
        return

    t0 = time.time()
    done, failed, rows_written = 0, 0, 0
    fail_reasons = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut = {pool.submit(judge, s, d, t, key): (s, d) for s, d, t, _ in tasks}
        for f in as_completed(fut):
            sym, date = fut[f]
            res, err = f.result()
            if res is None:
                failed += 1
                fail_reasons[err] = fail_reasons.get(err, 0) + 1
            else:
                db.upsert_judgement(
                    conn, res["symbol"], res["date"], "sent",
                    score_to_signal(res["score"]), res["score"],
                    reason=res["reason"], provider="zhipu",
                    model=res["model"], temperature=TEMPERATURE,
                    prompt_version=PROMPT_VERSION, source="live")
                rows_written += 1
            done += 1
            if done % 20 == 0 or done == len(tasks):
                el = time.time() - t0
                print("  进度 %d/%d  成功%d 失败%d  %.0fs  速率%.1f条/分" % (
                    done, len(tasks), rows_written, failed, el,
                    done / el * 60 if el > 0 else 0))

    conn.close()
    el = time.time() - t0
    print("\n=== 汇总 ===")
    print("成功写入 %d  失败 %d  耗时 %.1f 分钟" % (rows_written, failed, el / 60))
    print("token 消耗: 输入 %d 输出 %d 合计 %d" % (
        _STATS["in_tokens"], _STATS["out_tokens"],
        _STATS["in_tokens"] + _STATS["out_tokens"]))
    if fail_reasons:
        print("失败原因:", dict(sorted(fail_reasons.items(), key=lambda x: -x[1])[:5]))
    if failed:
        print("提示：失败项下次重跑会自动补齐（幂等），不需要 --force")


if __name__ == "__main__":
    main()

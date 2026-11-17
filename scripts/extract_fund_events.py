"""从 news 里抽财报事件，落 fund_events 表。

为什么做这一步：基本面视角现在只有一个"估值分位"替身，缺的正是"公司经营"这一维。
CSMD50 数据集不含财报表，但 news 里大量财报报道带具体数字（净利润同比、
营收同比、亏损），这类事实是能抽成结构化信号的。这也是课题那句
"非结构化文本分析如何转化为量化投资信号"最标准的落点。

为什么用 LLM 而不是正则：
- 正则只覆盖 54%（459 条里 249 条），且对"净利润为 X 亿元、同比下降 Y%"这类
  多数字句式方向判定不稳；
- 更要命的是**主体归属**：新闻里常有"中信证券保荐的安达科技上市当年亏损"
  这种提及他人，正则会把别家公司的亏损算到中信证券头上（实测踩到）。
  所以 prompt 强制要求先判断 subject_ok。

幂等：fund_events.UNIQUE(news_id)，重跑是覆盖不是追加，不会重复计数。
不用 --reset 也能安全重跑。

用法：
    python scripts/extract_fund_events.py                    # 全量
    python scripts/extract_fund_events.py --limit 20         # 先小样本试
    python scripts/extract_fund_events.py --start 2024-08-08 --end 2024-12-31
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

KEYWORDS = ("业绩预告", "业绩快报", "净利润", "净利", "营业收入", "营收",
            "年报", "季报", "分红", "亏损")

PROMPT = """你在做金融文本的信息抽取。给定一只股票和一条财经新闻，抽出其中的财务事实。

严格要求：
1. 先判断这条新闻的主体是不是这只股票本身。若新闻主要讲的是别家公司
   （例如"某券商保荐的 A 公司亏损"讲的是 A 公司不是这家券商），
   则 subject_ok=0，并停止抽取数字。
2. 只抽新闻原文里明确出现的数字，绝不推算、绝不补充常识。
3. np_yoy / rev_yoy 是**同比增速的百分数**，下降为负（例如"同比下降 39.45%" → -39.45）。
   原文没写同比增速，就给 null，不要用绝对金额倒推。
4. is_loss 只在原文明确说亏损（含预亏）时为 1。
5. direction 是这条新闻对公司经营的方向含义：偏多 1、中性 0、偏空 -1。
   依据只看经营事实本身（业绩增/减、亏损与否），不要照抄新闻里的"可能影响股价"。
6. evidence 摘原文片段，不超过 80 字。

只输出 JSON，不要任何解释文字：
{"subject_ok": 1, "event_type": "三季报", "np_yoy": -39.45, "rev_yoy": -17.74,
 "is_loss": 0, "direction": -1, "confidence": 0.8, "evidence": "前三季度净利润同比下降39.45%"}
"""

EVENT_TYPES = ("业绩预告", "业绩快报", "半年报", "三季报", "年报", "季报", "分红", "其他")

_LOCK = threading.Lock()
_STATS = {"calls": 0, "in": 0, "out": 0, "fail": 0}


def api_key():
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()


def pick_news(conn, start, end, limit):
    where = " OR ".join(["content LIKE ?"] * len(KEYWORDS))
    sql = ("SELECT id,symbol,date,content FROM news WHERE (" + where + ") "
           "AND date BETWEEN ? AND ? ORDER BY date, symbol")
    params = tuple("%" + k + "%" for k in KEYWORDS) + (start, end)
    rows = conn.execute(sql, params).fetchall()
    return rows[:limit] if limit else rows


def parse_json(text):
    """模型偶尔带 ```json 围栏或前后废话，尽量取到第一个完整 JSON 对象。"""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def call_one(session, symbol, content, key):
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "股票：%s\n新闻：%s" % (symbol, content[:600])},
        ],
        "temperature": 0.0,
        "max_tokens": 900,      # deepseek 是思考型，太小会被截断（踩过）
    }
    r = session.post("https://api.deepseek.com/chat/completions",
                     headers={"Authorization": "Bearer " + key,
                              "Content-Type": "application/json"},
                     json=body, timeout=60)
    r.raise_for_status()
    d = r.json()
    content_out = d["choices"][0]["message"]["content"]
    usage = d.get("usage") or {}
    with _LOCK:
        _STATS["calls"] += 1
        _STATS["in"] += usage.get("prompt_tokens", 0) or 0
        _STATS["out"] += usage.get("completion_tokens", 0) or 0
    return parse_json(content_out)


def to_score(subject_ok, np_yoy, rev_yoy, is_loss, direction):
    """把事件折成一个 [-1,1] 的分。

    口径说明：优先用净利润同比的幅度，因为它是"经营结果"最直接的度量；
    营收只做辅证（营收涨但利润跌，说明成本端出了问题，方向以利润为准）。
    """
    if not subject_ok:
        return 0.0
    if is_loss:
        return -0.6
    if np_yoy is None and rev_yoy is None:
        return 0.0
    base = np_yoy if np_yoy is not None else (rev_yoy * 0.5 if rev_yoy is not None else 0.0)
    # ±30% 以上视为强信号，线性映射到 ±1
    s = max(-1.0, min(1.0, float(base) / 30.0))
    return round(s, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-03-15")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    key = api_key()
    if not key:
        raise SystemExit("缺少 DEEPSEEK_API_KEY 环境变量（从 ~/.lightquant/secrets.env 读）")

    conn = db.connect(args.db)
    db.init_db(conn=conn)

    news = pick_news(conn, args.start, args.end, args.limit)
    print("待抽取 %d 条财报类新闻（%s ~ %s）" % (len(news), args.start, args.end))

    symbol_names = {r["symbol"]: r["symbol"] for r in
                    conn.execute("SELECT symbol FROM stocks")}

    def work(row, sess):
        try:
            out = call_one(sess, symbol_names.get(row["symbol"], row["symbol"]),
                           row["content"] or "", key)
        except Exception as e:
            with _LOCK:
                _STATS["fail"] += 1
            return row, None, "%s: %s" % (type(e).__name__, str(e)[:60])
        return row, out, None

    results = []
    local = threading.local()

    def worker(row):
        if not hasattr(local, "s"):
            local.s = requests.Session()
        return work(row, local.s)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(worker, r): r for r in news}
        for f in as_completed(futs):
            row, out, err = f.result()
            if err:
                print("FAIL %s %s %s" % (row["symbol"], row["date"], err))
                continue
            if not out:
                print("PARSE_FAIL %s %s" % (row["symbol"], row["date"]))
                continue
            results.append((row, out))

    n_written = 0
    for row, out in results:
        try:
            subject_ok = 1 if int(out.get("subject_ok", 1)) else 0
        except (TypeError, ValueError):
            subject_ok = 1

        def num(v):
            try:
                return None if v is None else float(v)
            except (TypeError, ValueError):
                return None

        np_yoy = num(out.get("np_yoy"))
        rev_yoy = num(out.get("rev_yoy"))
        is_loss = 1 if int(out.get("is_loss", 0) or 0) else 0
        et = (out.get("event_type") or "其他").strip()
        if et not in EVENT_TYPES:
            et = "其他"
        try:
            conf = float(out.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        direction = 1 if np_yoy is not None and np_yoy > 5 else (
            -1 if (np_yoy is not None and np_yoy < -5) or is_loss else 0)
        if not subject_ok:
            direction = 0
        score = to_score(subject_ok, np_yoy, rev_yoy, is_loss, direction)

        conn.execute(
            """INSERT INTO fund_events
               (symbol, date, news_id, event_type, subject_ok, np_yoy, rev_yoy,
                is_loss, direction, score, confidence, evidence)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT (news_id) DO UPDATE SET
                 event_type=excluded.event_type, subject_ok=excluded.subject_ok,
                 np_yoy=excluded.np_yoy, rev_yoy=excluded.rev_yoy,
                 is_loss=excluded.is_loss, direction=excluded.direction,
                 score=excluded.score, confidence=excluded.confidence,
                 evidence=excluded.evidence""",
            (row["symbol"], row["date"], row["id"], et, subject_ok,
             np_yoy, rev_yoy, is_loss, direction, score, conf,
             (out.get("evidence") or "")[:120]))
        n_written += 1
    conn.commit()

    from collections import Counter
    c1 = Counter(r["event_type"] for r in
                 conn.execute("SELECT event_type FROM fund_events"))
    ok = conn.execute("SELECT COUNT(*) c FROM fund_events WHERE subject_ok=1").fetchone()["c"]
    tot = conn.execute("SELECT COUNT(*) c FROM fund_events").fetchone()["c"]
    d = Counter(r["direction"] for r in
                conn.execute("SELECT direction FROM fund_events"))

    print("\n写入/更新 %d 条；表内共 %d 条" % (n_written, tot))
    print("  主体一致 %d / 提及他人 %d" % (ok, tot - ok))
    print("  方向分布: 偏多 %d / 中性 %d / 偏空 %d"
          % (d.get(1, 0), d.get(0, 0), d.get(-1, 0)))
    print("  事件类型:", dict(c1))
    print("LLM 调用 %d 次，输入 %d tokens，输出 %d tokens，失败 %d"
          % (_STATS["calls"], _STATS["in"], _STATS["out"], _STATS["fail"]))
    conn.close()


if __name__ == "__main__":
    main()

"""实时研判调度：一次请求内调度三个视角，当场生成研判报告并算融合。

为什么要有这个模块
------------------
课题原文是「调度多个大模型（或不同分析视角的智能体）分别从技术面、基本面和
情绪面生成研判报告，并通过投票或加权机制输出综合预测」。而在此之前，系统里
只有"研判结果表"、没有"调度动作"——三视角结果全部由 `scripts/` 下的离线脚本
批量跑完落库，服务层只读表。评审追问"你的调度器是哪个模块""现场换只票跑一次"
时，指不出一段代码。

本模块把三条链路的既有实现搬进服务层，一次调用完成：调度 → 生成 → 融合 → 落库。
不另造算法，三个视角的判定逻辑与脚本逐行一致。

三个必须说清的边界
------------------
1. **写独立 prompt_version**（`rerun-<时间戳>`），不覆盖主口径 `fusion-test-3view`。
   演示随便跑，跑完整批可删（`db_cleanup(conn, pv)`）。
2. **截面去均值需要横截面**。`fuse_views` 对 fund / sent 做按日截面去均值，是为了
   去掉"情绪面整体偏乐观"这类底噪（实测 50 条里 32 条 BUY、均分 +0.46）。只调
   一只票时没有横截面，故取当日库里已有的同版本分数算均值；当日不足 5 只时
   退化为不去均值，并在返回里标明 `demean: skipped`，不假装做过。
3. **技术面的标准化用该股全序列均值方差**（继承自原仓库 `standardize_dataframe`），
   这是已知的前视泄漏，此处不修正，只在返回里如实标注。

用法
----
    from app import service_judge
    result = service_judge.run(conn, "万华化学", "2024-12-31", allow_llm=True)
"""

import json
import os
import re
import sys
import threading
import time

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import db  # noqa: E402

# 与 scripts/gen_tech.py 完全一致：同样的十维特征、同样的权重目录、同样的窗口
COLS = ["Open", "High", "Low", "Close", "Adj Close",
        "Return", "RSI6", "MACD_Hist", "BIAS5", "Amplitude"]
MODEL_FOLDER = os.path.join(ROOT, "result", "CSMD50_w5_feat_bi")
PRICE_DIR = os.path.join(ROOT, "dataset", "CSMD50_feat", "test", "price")
NEWS_ROOT = os.path.join(ROOT, "dataset", "CSMD50", "news")
LOOK_BACK = 5

# 主口径融合权重（验证集三方网格标定，docs/calib_weights_3view.json）
WEIGHTS = {"tech": 0.90, "fund": 0.00, "sent": 0.10}
THRESHOLD = 0.2
DEMEAN_MIN_N = 5          # 当日横截面少于这个数就不做去均值

# 去均值取的是哪一版：必须跟主口径一致，否则"今天的均值"和"今天的分数"不是一回事
PV_REF = {"fund": "fund-rule-v1", "sent": "sent-v1"}

# 情绪面：与 scripts/gen_sentiment.py 同一条契约（智谱、temperature 0、JSON 输出）
SENT_BASE = "https://open.bigmodel.cn/api/paas/v4"
SENT_PRIMARY = "glm-5.3-flash"
SENT_FALLBACK = "glm-4.7-flash"
SENT_TMPL = """你是A股财经新闻情绪分析器。下面是该股票在 {date} 的新闻摘要。判断其对下一交易日股价的情绪倾向。

只输出一个 JSON 对象，不要输出解释、不要 markdown、不要代码块。字段：
{{"score": 浮点数(-1到1，正为利好，负为利空，无信息填0), "confidence": 浮点数(0到1), "reason": 不超过30字, "evidence": [1-2条最关键的依据片段]}}

新闻摘要：
{news}"""

# 无 key 时的词典兜底。明确标成规则分，不冒充大模型研判。
POS_WORDS = ["利好", "增长", "中标", "签约", "回购", "分红", "提价", "超预期",
             "净利润增加", "扭亏", "获批", "扩产", "增持"]
NEG_WORDS = ["利空", "亏损", "下滑", "减持", "处罚", "违规", "诉讼", "低于预期",
             "净利润下降", "计提", "停牌", "退市", "质押"]

_MODEL = {"obj": None, "error": None}
_LOCK = threading.Lock()


# ---------------------------------------------------------------- 技术面
def warm_model():
    """提前加载 BiLSTM。服务启动时在后台线程调一次，避免首次点击让用户干等。"""
    return _model()[0]


def _model():
    with _LOCK:
        if _MODEL["obj"] is not None:
            return _MODEL["obj"], None
        if _MODEL["error"]:
            return None, _MODEL["error"]
        try:
            from types import SimpleNamespace

            from backtest.engine_v2 import build_model
            ns = SimpleNamespace(
                model="bi_lstm", model_save_folder=MODEL_FOLDER,
                input_size=len(COLS), hidden_size=128, layers=2,
                attention_size=128, dropout=0.1, epsilon=0.1,
                perturbation_size=0.1,
            )
            _MODEL["obj"] = build_model(ns)
        except Exception as e:
            _MODEL["error"] = "%s: %s" % (type(e).__name__, e)
            return None, _MODEL["error"]
    return _MODEL["obj"], None


def judge_tech(symbol, date):
    """技术面：本地 BiLSTM 十维特征推理。score = 2p - 1，与另外两路同尺度。"""
    t0 = time.time()
    model, err = _model()
    if model is None:
        return _part("tech", "error", error="模型加载失败：%s" % err)
    path = os.path.join(PRICE_DIR, symbol + ".csv")
    if not os.path.exists(path):
        return _part("tech", "error", error="缺少价格文件 %s" % path)
    try:
        import torch

        from utils.price_dataloader import Backtest_Dataset
        # 前视泄漏修复：scaler 在 train 段拟合
        train_path = path.replace(os.path.join("test", "price"),
                                  os.path.join("train", "price"))
        ds = Backtest_Dataset(csv_file=path, look_back_window=LOOK_BACK,
                              feature_columns=COLS,
                              scaler_csv=train_path if os.path.exists(train_path) else None)
        dates = [str(d) for d in ds.get_dates()]
        if date not in dates:
            return _part("tech", "skip", reason="%s 不在该标的价格序列内（非交易日或超出 test 段）" % date)
        i = dates.index(date)
        x = torch.tensor(np.asarray([ds.normalized_sequences[i]]), dtype=torch.float32)
        with torch.no_grad():
            p = float(model(x).squeeze(-1).cpu().numpy()[0])
    except Exception as e:
        return _part("tech", "error", error="%s: %s" % (type(e).__name__, e))

    # forward 末尾已有 Sigmoid，越界说明权重/结构不对，直接报错而不是静默写错数据
    if not (-1e-6 <= p <= 1 + 1e-6):
        return _part("tech", "error", error="模型输出不在 [0,1]：%.4f" % p)

    score = max(-1.0, min(1.0, 2.0 * p - 1.0))
    try:
        last = ds.raw_sequences[i][-1]
        rsi, macd, bias = float(last[6]), float(last[7]), float(last[8])
        feat_txt = "，近 5 日 RSI6 %.1f、MACD柱 %+.4f、BIAS5 %+.2f" % (rsi, macd, bias)
    except Exception:
        feat_txt = ""
    verdict = "偏多" if p >= 0.55 else ("偏空" if p <= 0.45 else "中性")
    reason = "BiLSTM 估计次日上涨概率 %.1f%%（阈值 50%%），判为%s%s" % (p * 100, verdict, feat_txt)

    return _part("tech", "ok", score=score, signal=_signal(score),
                 reason=reason, provider="local", model="bi_lstm_feat10",
                 source="live", prob=round(p, 4),
                 ms=int((time.time() - t0) * 1000),
                 note="标准化用该股全序列均值方差（继承自原仓库，已知前视泄漏）")


# ---------------------------------------------------------------- 基本面
def _pct_rank(series, value):
    """value 在历史中的分位，只用 <= value 的历史，避免未来信息。"""
    s = pd.Series(series).dropna()
    if len(s) < 8 or value is None or pd.isna(value):
        return None
    return float((s <= value).mean())


def _fund_score(pe_pct, pb_pct):
    parts, weights = [], []
    if pe_pct is not None:
        parts.append((0.5 - pe_pct) * 2.0)
        weights.append(0.6)
    if pb_pct is not None:
        parts.append((0.5 - pb_pct) * 2.0)
        weights.append(0.4)
    if not parts:
        return None
    score = sum(p * w for p, w in zip(parts, weights)) / sum(weights)
    return max(-1.0, min(1.0, score))


def judge_fund(conn, symbol, date):
    """基本面（规则版）：PE/PB 在自身历史的分位 → 低估给正分、高估给负分。"""
    t0 = time.time()
    rows = conn.execute(
        "SELECT date, pe_ttm, pb FROM valuations WHERE symbol=? AND date<=? ORDER BY date",
        (symbol, date)).fetchall()
    if not rows:
        return _part("fund", "skip", reason="%s 在 %s 及之前没有估值记录" % (symbol, date))
    v = pd.DataFrame([dict(r) for r in rows])
    cur = v.iloc[-1]
    pe_pct = _pct_rank(v["pe_ttm"], cur["pe_ttm"])
    pb_pct = _pct_rank(v["pb"], cur["pb"])
    sc = _fund_score(pe_pct, pb_pct)
    if sc is None:
        return _part("fund", "skip", reason="历史估值样本不足（少于 8 期），分位不可靠")

    position = "偏低" if sc >= 0.2 else ("偏高" if sc <= -0.2 else "中性")
    reason = "PE 处历史 %s 分位、PB 处历史 %s 分位，综合估值分 %+.2f，估值%s" % (
        "%.0f%%" % (pe_pct * 100) if pe_pct is not None else "NA",
        "%.0f%%" % (pb_pct * 100) if pb_pct is not None else "NA",
        sc, position)
    return _part("fund", "ok", score=sc, signal=_signal(sc), reason=reason,
                 provider="local", model="valuation_percentile",
                 source="offline_rule",
                 ms=int((time.time() - t0) * 1000),
                 note="规则计算，不是大模型研判")


# ---------------------------------------------------------------- 情绪面
def _read_news(symbol, date):
    path = os.path.join(NEWS_ROOT, symbol, date + ".csv")
    if not os.path.exists(path):
        return [], path
    try:
        df = pd.read_csv(path)
    except Exception:
        return [], path
    if "text" not in df.columns:
        return [], path
    out = []
    for t in df["text"].dropna():
        s = str(t).strip()
        if s:
            out.append(s[:220])
    return out, path


def _call_llm(prompt, key):
    import requests

    body = {"model": SENT_PRIMARY, "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(SENT_BASE + "/chat/completions",
                      headers={"Authorization": "Bearer " + key,
                               "Content-Type": "application/json"},
                      json=body, timeout=45)
    if r.status_code != 200:
        raise RuntimeError("HTTP%d %s" % (r.status_code, r.text[:80].replace("\n", " ")))
    txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        raise RuntimeError("no_json")
    return json.loads(m.group(0))


def _lexicon_score(texts):
    """无 key 时的词典兜底。返回 (score, 命中词)。"""
    hits_pos = [w for w in POS_WORDS if any(w in t for t in texts)]
    hits_neg = [w for w in NEG_WORDS if any(w in t for t in texts)]
    if not hits_pos and not hits_neg:
        return 0.0, []
    raw = (len(hits_pos) - len(hits_neg)) / float(len(hits_pos) + len(hits_neg))
    return max(-1.0, min(1.0, raw * 0.6)), hits_pos + hits_neg


def judge_sent(symbol, date, allow_llm=True):
    """情绪面：当日新闻 → LLM 结构化打分；无 key 或调用失败时降级为词典规则。

    没有新闻时**弃权**，不落 HOLD。理由与 scripts/gen_agents.py 一致：
    落一条 HOLD 在投票里就是一张实打实的反对票，会让"没信息"变成"看空"。
    """
    t0 = time.time()
    texts, path = _read_news(symbol, date)
    if not texts:
        return _part("sent", "skip", reason="%s 当日无新闻（%s 不存在），该视角弃权"
                     % (date, os.path.relpath(path, ROOT)))

    key = (os.environ.get("ZHIPU_KEY") or "").strip()
    last = "unknown"
    if allow_llm and key:
        prompt = SENT_TMPL.format(date=date, news="\n".join("- " + t for t in texts))
        for model in (SENT_PRIMARY, SENT_FALLBACK):
            try:
                out = _call_llm(prompt, key)
                score = max(-1.0, min(1.0, float(out.get("score", 0.0))))
                return _part("sent", "ok", score=score, signal=_signal(score),
                             reason=str(out.get("reason", ""))[:60],
                             provider="zhipu", model=model, source="live",
                             confidence=out.get("confidence"),
                             ms=int((time.time() - t0) * 1000),
                             note="%d 条新闻" % len(texts))
            except Exception as e:
                last = "%s: %s" % (type(e).__name__, str(e)[:40])
    else:
        last = "未配 ZHIPU_KEY" if not key else "本次调用显式关闭 LLM"

    score, hits = _lexicon_score(texts)
    return _part("sent", "ok", score=score, signal=_signal(score),
                 reason="未走大模型（%s），按情绪词典命中 %s，判为%s"
                        % (last, "/".join(hits) if hits else "无涨跌词",
                           "利好" if score > 0 else ("利空" if score < 0 else "中性")),
                 provider="local", model="sent-lexicon-v1", source="offline_rule",
                 ms=int((time.time() - t0) * 1000),
                 note="规则分，不是大模型研判；%d 条新闻" % len(texts))


# ---------------------------------------------------------------- 融合
def _cross_mean(conn, view, date):
    """当日横截面均值：(mean, n)。与 fuse_views 的 demean 同一语义。"""
    rows = conn.execute(
        "SELECT score FROM judgements WHERE view=? AND date=? AND prompt_version=?",
        (view, date, PV_REF[view])).fetchall()
    scores = [float(r["score"]) for r in rows]
    if len(scores) < DEMEAN_MIN_N:
        return None, len(scores)
    return float(np.mean(scores)), len(scores)


def fuse(conn, date, parts):
    """按主口径权重加权；fund/sent 先去截面均值，再按可用权重归一化。"""
    adjusted, demean_info = {}, {}
    for k in ("tech", "fund", "sent"):
        p = parts.get(k) or {}
        if p.get("status") != "ok":
            continue
        sc = float(p["score"])
        if k in ("fund", "sent"):
            mean, n = _cross_mean(conn, k, date)
            if mean is None:
                demean_info[k] = {"applied": False, "n": n,
                                  "reason": "当日横截面 %d 只 < %d，未去均值" % (n, DEMEAN_MIN_N)}
            else:
                demean_info[k] = {"applied": True, "n": n, "mean": round(mean, 4)}
                sc = max(-1.0, min(1.0, sc - mean))
        else:
            demean_info[k] = {"applied": False, "reason": "技术面概率是绝对值，按定义不去均值"}
        adjusted[k] = sc

    if not adjusted:
        return None, demean_info
    ws = sum(WEIGHTS[k] for k in adjusted)
    if ws <= 0:
        # 只有 fund 有值时权重和为 0，融合无意义
        return None, demean_info
    fscore = sum(WEIGHTS[k] * v for k, v in adjusted.items()) / ws
    fscore = max(-1.0, min(1.0, fscore))
    # 理由里要把"权重为 0 的视角"标出来：它算出了分，但没参与这次融合，
    # 否则前端看到 fund=+0.51 却得到 HOLD 会以为算错了
    bits = []
    for k, v in adjusted.items():
        star = "*" if k in ("fund", "sent") and demean_info.get(k, {}).get("applied") else ""
        bits.append("%s=%+.2f%s%s" % (k, v, star, "(w=0 不参与)" if WEIGHTS[k] <= 0 else ""))
    return {"score": round(fscore, 4), "signal": _signal(fscore),
            "reason": " ".join(bits)[:80],
            "weights": {k: WEIGHTS[k] for k in adjusted},
            "normalized_by": round(ws, 4)}, demean_info


# ---------------------------------------------------------------- 入口
def _signal(score, th=THRESHOLD):
    if score >= th:
        return "BUY"
    if score <= -th:
        return "SELL"
    return "HOLD"


def _part(view, status, score=None, signal=None, reason=None, provider=None,
          model=None, source=None, error=None, ms=None, note=None, **extra):
    d = {"view": view, "status": status}
    if score is not None:
        d["score"] = round(float(score), 4)
    for k, v in (("signal", signal), ("reason", reason), ("provider", provider),
                 ("model", model), ("source", source), ("error", error),
                 ("ms", ms), ("note", note)):
        if v is not None:
            d[k] = v
    d.update(extra)
    return d


def new_version(prefix="rerun"):
    """生成独立版本号。演示随便跑，不碰主口径 fusion-test-3view。"""
    return "%s-%s" % (prefix, time.strftime("%Y%m%d%H%M%S"))


def run(conn, symbol, date, allow_llm=True, write=True, prompt_version=None):
    """调度三个视角 → 融合 → 落库。返回给前端可直接渲染的结果。

    write=False 时只算不写（调试用，不污染库）。
    """
    t_all = time.time()
    row = conn.execute("SELECT symbol FROM stocks WHERE symbol=?", (symbol,)).fetchone()
    if row is None:
        return {"ok": False, "error": "未知标的：%s（不在 stocks 表里）" % symbol}
    trade = conn.execute(
        "SELECT date FROM prices WHERE symbol=? AND date=?", (symbol, date)).fetchone()
    if trade is None:
        return {"ok": False, "error": "%s 不是 %s 的交易日" % (date, symbol)}

    pv = prompt_version or new_version()

    parts = {}
    parts["tech"] = judge_tech(symbol, date)
    parts["fund"] = judge_fund(conn, symbol, date)
    parts["sent"] = judge_sent(symbol, date, allow_llm=allow_llm)

    fusion, demean_info = fuse(conn, date, parts)

    written = []
    if write:
        for k in ("tech", "fund", "sent"):
            p = parts[k]
            if p.get("status") != "ok":
                continue
            db.upsert_judgement(
                conn, symbol, date, k, p["signal"], p["score"],
                reason=(p.get("reason") or "")[:120],
                provider=p.get("provider"), model=p.get("model"),
                temperature=(0.0 if p.get("source") == "live" else None),
                prompt_version=pv, source=p.get("source") or "live")
            written.append(k)
        if fusion is not None:
            db.upsert_judgement(
                conn, symbol, date, "fusion", fusion["signal"], fusion["score"],
                reason=fusion["reason"], provider="local", model="rule_weighted",
                temperature=None, prompt_version=pv, source="live")
            written.append("fusion")
        conn.commit()

    skipped = [k for k in ("tech", "fund", "sent") if parts[k].get("status") != "ok"]
    return {
        "ok": True,
        "symbol": symbol,
        "date": date,
        "prompt_version": pv,
        "written": written,
        "skipped": skipped,
        "views": parts,
        "fusion": fusion,
        "demean": demean_info,
        "elapsed_ms": int((time.time() - t_all) * 1000),
        "note": ("三视角调度完成；结果写在独立版本 %s，不影响主口径 fusion-test-3view。"
                 % pv) if write else "只计算未写库",
    }


def run_batch(conn, symbols, dates, allow_llm=True, write=True, prompt_version=None):
    """批量调度：多只股 × 多个交易日，全程共用同一个 prompt_version。

    一次批量只产一个 rerun 版本，方便演示完一键 cleanup；
    上限 50 个组合，防止误操作把 LLM 调用量炸掉。
    """
    symbols = [s.strip() for s in symbols if s and s.strip()]
    dates = [d.strip() for d in dates if d and d.strip()]
    if not symbols or not dates:
        raise ValueError("symbols 和 dates 都不能为空")
    n = len(symbols) * len(dates)
    if n > 50:
        raise ValueError("批量上限 50 个组合（当前 %d）；防止误操作炸掉 LLM 调用量" % n)

    pv = prompt_version or new_version()
    items = []
    ok_count = fail_count = skip_views = 0
    for symbol in symbols:
        for date in dates:
            r = run(conn, symbol, date, allow_llm=allow_llm, write=write, prompt_version=pv)
            items.append(r)
            if r.get("ok"):
                ok_count += 1
                skip_views += len(r.get("skipped", []))
            else:
                fail_count += 1
    return {
        "ok": fail_count == 0,
        "prompt_version": pv,
        "total": n,
        "ok_count": ok_count,
        "fail_count": fail_count,
        "skipped_views": skip_views,
        "items": items,
    }


def cleanup(conn, prompt_version):
    """删除某个 rerun 版本的全部记录（tech/fund/sent/fusion 一并清）。"""
    if not prompt_version or not prompt_version.startswith("rerun-"):
        raise ValueError("只允许删除 rerun-* 版本，避免误删主口径")
    n = conn.execute(
        "DELETE FROM judgements WHERE prompt_version=?", (prompt_version,)).rowcount
    conn.commit()
    return n

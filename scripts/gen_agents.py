"""三分析师智能体：技术面 / 基本面 / 情绪面各自独立研判。

和旧版的区别（这是本脚本存在的理由）
------------------------------------
- 旧 `tech-v1`：BiLSTM 判方向，LLM 只负责写理由（解读层）。
- 旧 `fund-rule-v1`：规则算分位，LLM 完全不参与。
- 旧 `sent-v1`：只有这一路真的让 LLM 判断。

课题要求"调度多个大模型（或不同分析视角的智能体）分别从技术面、基本面和情绪面
生成研判报告"。只把表述改一改不算满足，所以这里把三个视角都改成同一种结构：

    角色设定（Profile） + 工具返回值（Action） + 自己判断（Planning） + 近3日历史（Memory）

BiLSTM 分数、PE/PB 分位、新闻原文全部降级成"工具返回值"，LLM 拿着它们自己判断，
可以同意也可以反对。这样三个视角的判断内核才真的是 LLM，而不是"规则判完让 LLM 编理由"。

信息隔离是这套设计的地基：技术面看不到新闻与估值，基本面看不到 K 线与新闻，
情绪面看不到价格与估值。否则三路会退化成同一个模型看同一份材料抄三遍。

用法
----
    # 只看一条样例的 prompt，不调模型、不落库
    python scripts/gen_agents.py --start 2024-12-02 --end 2024-12-31 --limit 50 --dry_run

    # 100 样本预实验：2 个交易日 × 50 只 × 3 视角 = 300 次调用
    python scripts/gen_agents.py --start 2024-12-30 --end 2024-12-31 --limit 50 --workers 4
"""

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

PROVIDER = "deepseek"
MODEL = "deepseek-flash"          # V4.1-Flash，1M 上下文，按量计费
BASE_URL = "https://api.deepseek.com"
TEMPERATURE = 0.0
# 这个模型是思考型：思考内容与正文共用 max_tokens。
# 实测 max_tokens=320 时思考就吃掉 304，正文只剩 33 字符导致 JSON 截断，
# 降级率 97%；提到 1200 并压 effort=low 后思考约 330、正文完整。
MAX_TOKENS = 1200
EFFORT = "low"

# 三个视角各自的版本名。judgements 的 UNIQUE 不含 model，
# 换模型必须换版本名，否则会把旧口径静默覆盖掉。
PV = {
    "tech": "tech-agent-v1",
    "fund": "fund-agent-v1",
    "sent": "sent-agent-v1",
}
BASE_PV = {"tech": "tech-v1", "fund": "fund-rule-v1"}

VALID_SIGNALS = ("BUY", "SELL", "HOLD")

_STATS = {"in_tokens": 0, "out_tokens": 0, "calls": 0,
          "failed": 0, "fallback": 0, "by_view": {}}
_LOCK = threading.Lock()
_LOCAL = threading.local()


# ---------------------------------------------------------------- 密钥
def api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    path = os.path.join(os.environ.get("USERPROFILE", ""), ".lightquant", "secrets.env")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


def _v(x):
    """把 None 写成 NA，而不是把 None 原样塞进 prompt。"""
    return "NA" if x is None else x


# ---------------------------------------------------------------- 工具层
def calc_indicators(conn, symbol, date):
    """工具 1：技术指标。只用 date <= 目标日的数据，杜绝未来信息。"""
    rows = conn.execute(
        "SELECT date,open,high,low,close,volume FROM prices "
        "WHERE symbol=? AND date<=? ORDER BY date DESC LIMIT 60",
        (symbol, date),
    ).fetchall()
    if len(rows) < 21:
        return None
    rs = list(reversed(rows))
    closes = [float(r["close"]) for r in rs]
    highs = [float(r["high"]) for r in rs]
    lows = [float(r["low"]) for r in rs]
    vols = [float(r["volume"]) for r in rs]

    ma5 = sum(closes[-5:]) / 5.0
    ma10 = sum(closes[-10:]) / 10.0
    ma20 = sum(closes[-20:]) / 20.0

    gains = losses = 0.0
    for i in range(-6, 0):
        ch = closes[i] - closes[i - 1]
        if ch > 0:
            gains += ch
        else:
            losses += -ch
    ag, al = gains / 6.0, losses / 6.0
    rsi6 = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)

    def ema_series(vals, n):
        k = 2.0 / (n + 1.0)
        out = [vals[0]]
        for v in vals[1:]:
            out.append(v * k + out[-1] * (1.0 - k))
        return out

    e12 = ema_series(closes, 12)
    e26 = ema_series(closes, 26)
    dif = [a - b for a, b in zip(e12, e26)]
    dea_series = ema_series(dif[-9:], 9) if len(dif) >= 9 else dif
    dea = dea_series[-1]
    hist = dif[-1] - dea

    v5 = sum(vols[-5:]) / 5.0
    v20 = sum(vols[-20:]) / 20.0
    vol_ratio = (v5 / v20) if v20 else None
    amplitude = (highs[-1] - lows[-1]) / closes[-1] * 100.0

    def ret(n):
        return (closes[-1] / closes[-1 - n] - 1.0) * 100.0

    daily = [closes[i] / closes[i - 1] - 1.0 for i in range(len(closes) - 20, len(closes))]
    mean = sum(daily) / len(daily)
    var = sum((x - mean) ** 2 for x in daily) / (len(daily) - 1)
    ann_vol = math.sqrt(var) * math.sqrt(242.0) * 100.0

    return {
        "asof": rs[-1]["date"],
        "close": round(closes[-1], 2),
        "ma5": round(ma5, 2), "ma10": round(ma10, 2), "ma20": round(ma20, 2),
        "vs_ma5": round((closes[-1] / ma5 - 1.0) * 100.0, 2),
        "vs_ma20": round((closes[-1] / ma20 - 1.0) * 100.0, 2),
        "rsi6": round(rsi6, 1),
        "macd_dif": round(dif[-1], 3),
        "macd_dea": round(dea, 3),
        "macd_hist": round(hist, 3),
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "amplitude": round(amplitude, 2),
        "ret5": round(ret(5), 2),
        "ret20": round(ret(20), 2),
        "ann_vol": round(ann_vol, 1),
    }


def local_model_score(conn, symbol, date):
    """工具 2：本地 BiLSTM 的方向与分数。只是证据，答案由 LLM 自己定。"""
    row = conn.execute(
        "SELECT signal,score,reason,model FROM judgements "
        "WHERE view='tech' AND symbol=? AND date=? AND prompt_version=?",
        (symbol, date, BASE_PV["tech"]),
    ).fetchone()
    if row is None:
        return None
    return {
        "signal": row["signal"],
        "score": round(float(row["score"]), 4),
        "reason": (row["reason"] or "")[:50],
        "model": row["model"],
    }


def get_valuation(conn, symbol, date):
    """工具 3：估值与历史分位。取"当天之前最近一条"，不插值不回溯。"""
    rows = conn.execute(
        "SELECT date,pe_ttm,pb,ps_ttm FROM valuations "
        "WHERE symbol=? AND date<=? ORDER BY date",
        (symbol, date),
    ).fetchall()
    if not rows:
        return None
    cur = rows[-1]
    pe_series = [float(r["pe_ttm"]) for r in rows if r["pe_ttm"] is not None]
    pb_series = [float(r["pb"]) for r in rows if r["pb"] is not None]

    def pct(series, value):
        if not series or value is None or len(series) < 8:
            return None
        return sum(1 for x in series if x <= value) / float(len(series))

    pe_v = float(cur["pe_ttm"]) if cur["pe_ttm"] is not None else None
    pb_v = float(cur["pb"]) if cur["pb"] is not None else None
    ps_v = float(cur["ps_ttm"]) if cur["ps_ttm"] is not None else None
    return {
        "asof": cur["date"],
        "pe_ttm": round(pe_v, 2) if pe_v is not None else None,
        "pb": round(pb_v, 2) if pb_v is not None else None,
        "ps_ttm": round(ps_v, 2) if ps_v is not None else None,
        "pe_pct": round(pct(pe_series, pe_v), 4) if pe_v is not None else None,
        "pb_pct": round(pct(pb_series, pb_v), 4) if pb_v is not None else None,
        "n_history": len(rows),
    }


def get_industry(conn, symbol, date):
    """工具 5（fund-v2）：行业归属 + 同大类股票数。

    只有行业归属，不提供"同行业平均涨跌"——那条信号已实测无效
    （Spearman −0.0002，见 docs/基本面口径与财报事件验证.md 第三节），
    给了反而会诱导模型编方向。
    """
    row = conn.execute(
        "SELECT industry FROM stocks WHERE symbol=?", (symbol,)).fetchone()
    if row is None or not row["industry"]:
        return None
    ind = row["industry"]
    try:
        from fill_industry import sector_of, SECTOR_MAP
    except Exception:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from fill_industry import sector_of, SECTOR_MAP
    sec = sector_of(ind)

    # 按大类统计同类数：申万一级粒度下有一半行业只有 1 只（基础化工就是万华化学自己），
    # 报"同类 1 只"等于没信息。大类才有对比意义（能源材料 6 只）。
    peers = []
    for p in conn.execute("SELECT symbol, industry FROM stocks WHERE industry IS NOT NULL"):
        if sector_of(p["industry"]) == sec and p["symbol"] != symbol:
            peers.append(p["symbol"])
    return {
        "industry": ind,
        "sector": sec,
        "peer_count": len(peers) + 1,
        "peers": peers[:6],
    }


def get_fund_events(conn, symbol, date, days=90, limit=6):
    """工具 6（fund-v2）：近 N 日的财报事件（业绩预告/财报/亏损/分红）。

    窗口 90 天：财报是季度稀疏事件，30 天窗口实测 88% 的样本「无事件」，
    等于没材料。90 天能把单只股票的命中率提到可用水平，同时仍远短于一个季度，
    不至于把两季前的旧业绩当当期。

    时间口径：严格 **date 之前**（`e.date < date`），不含当天。
    原因：财报新闻多为盘后发布，若允许用当天，等于用收盘后才有的信息
    在当天收盘价上成交——这条信号在验证阶段就是因这个未来函数虚增了收益
    （见 docs 第五节），这里必须收紧。
    """
    cutoff = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT date,event_type,np_yoy,rev_yoy,is_loss,direction,score,"
        "       subject_ok,confidence,evidence FROM fund_events "
        "WHERE symbol=? AND date>=? AND date<? AND subject_ok=1 "
        "ORDER BY date DESC LIMIT ?",
        (symbol, cutoff, date, limit),
    ).fetchall()
    if not rows:
        return None
    return [{
        "date": r["date"],
        "type": r["event_type"],
        "np_yoy": round(float(r["np_yoy"]), 2) if r["np_yoy"] is not None else None,
        "rev_yoy": round(float(r["rev_yoy"]), 2) if r["rev_yoy"] is not None else None,
        "is_loss": int(r["is_loss"]),
        "score": round(float(r["score"]), 3) if r["score"] is not None else None,
        "evidence": (r["evidence"] or "")[:80],
    } for r in rows]


def get_fundamentals(conn, symbol, date, n_periods=4):
    """工具 7（fund-v3）：公司财务指标，含同比趋势。

    时点口径是这条工具的命门：财务数据按**报告期**发布，不是报告期结束当天
    就公开。数据库里每期都存了 announce_date（法定披露截止日），这里只取
    announce_date <= 交易日 的最近一期，回看更早的期做同比趋势。

    实测在 test 段（2024-08-08~12-31）内可见期只有 2024Q1 / 2024H1 / 2024Q3：
    11 月之前用不到三季报，这是真实世界的披露滞后，不是数据缺陷。

    返回 None 表示这只股票这一天还没有任何已披露的财务数据。
    """
    rows = conn.execute(
        "SELECT period,announce_date,roe,gross_margin,revenue_yoy,profit_yoy,"
        "       debt_ratio,ocf_ps,current_ratio,inventory_turn FROM fundamentals "
        "WHERE symbol=? AND announce_date<=? ORDER BY announce_date DESC LIMIT ?",
        (symbol, date, n_periods),
    ).fetchall()
    if not rows:
        return None
    cur = rows[0]
    hist = rows[1:]

    def num(v):
        return round(float(v), 2) if v is not None else None

    # 同比趋势：拿同报告期的上一年做对比（2024Q1 vs 2023Q1）。
    # 不能用"上一期"，因为 Q1→H1 是累计口径，比出来的不是增长。
    same_period_prev = None
    want = str(int(cur["period"][:4]) - 1) + cur["period"][4:]
    for r in hist:
        if r["period"] == want:
            same_period_prev = r
            break
    if same_period_prev is None:
        # 回看窗口不够时，直接查库（窗口只有 4 期，跨年同比常被截断）
        r = conn.execute(
            "SELECT period,roe,gross_margin,debt_ratio,revenue_yoy,profit_yoy "
            "FROM fundamentals WHERE symbol=? AND period=?", (symbol, want)).fetchone()
        if r is not None:
            same_period_prev = r

    trend = None
    if same_period_prev is not None:
        trend = {}
        for f in ("roe", "gross_margin", "debt_ratio"):
            a, b = cur[f], same_period_prev[f]
            if a is not None and b is not None:
                trend[f] = round(float(a) - float(b), 2)

    return {
        "period": cur["period"],
        "announce_date": cur["announce_date"],
        "roe": num(cur["roe"]),
        "gross_margin": num(cur["gross_margin"]),
        "revenue_yoy": num(cur["revenue_yoy"]),
        "profit_yoy": num(cur["profit_yoy"]),
        "debt_ratio": num(cur["debt_ratio"]),
        "ocf_ps": num(cur["ocf_ps"]),
        "current_ratio": num(cur["current_ratio"]),
        "inventory_turn": num(cur["inventory_turn"]),
        "prev_period": same_period_prev["period"] if same_period_prev is not None else None,
        "trend": trend,
    }


def get_news(conn, symbol, date, days=3, limit=6):
    """工具 4：近 N 日新闻。没有新闻返回空列表，由 LLM 自己判 HOLD。"""
    cutoff = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT date,title,substr(coalesce(content,''),1,130) body FROM news "
        "WHERE symbol=? AND date>=? AND date<=? ORDER BY date DESC LIMIT ?",
        (symbol, cutoff, date, limit),
    ).fetchall()
    return [{"date": r["date"], "title": (r["title"] or "")[:60],
             "body": (r["body"] or "").replace("\n", " ")} for r in rows]


def recent_memory(conn, symbol, date, view, n=3):
    """Memory：近 N 个交易日同一视角、同一版本的历史判断。"""
    rows = conn.execute(
        "SELECT date,signal,score FROM judgements "
        "WHERE view=? AND symbol=? AND date<? AND prompt_version=? "
        "ORDER BY date DESC LIMIT ?",
        (view, symbol, date, PV[view], n),
    ).fetchall()
    return [{"date": r["date"], "signal": r["signal"], "score": round(float(r["score"]), 3)}
            for r in rows]


def fmt_memory(rows):
    if not rows:
        return "（该视角此前没有历史判断）"
    return "\n".join("- %s：%s（%+0.2f）" % (r["date"], r["signal"], r["score"]) for r in rows)


# ---------------------------------------------------------------- 三份角色设定
TECH_PROMPT = """你是 A 股技术面分析师，只负责技术面，不看基本面、不看新闻。

【研判纪律】
1. 结论必须由你自己判断。下面的本地模型分数只是证据之一，不是标准答案；
   你可以同意，也可以反对，反对时必须在 reason 里说明依据。
2. 只能使用下面给出的数字，不得引入没有提供的信息。
3. 证据互相矛盾或方向不明确时，可以判 HOLD，但要在 reason 写明矛盾在哪。

【标的】{symbol}　【交易日】{date}

【工具 calc_indicators 返回值】
收盘价 {close}
MA5 {ma5} / MA10 {ma10} / MA20 {ma20}
收盘价相对 MA5 {vs_ma5}%，相对 MA20 {vs_ma20}%
RSI6 {rsi6}
MACD：DIF {macd_dif} / DEA {macd_dea} / 柱 {macd_hist}
量比（5日均量÷20日均量）{vol_ratio}
当日振幅 {amplitude}%
近5日涨跌 {ret5}% / 近20日涨跌 {ret20}%
20日年化波动率 {ann_vol}%

【工具 local_model_score 返回值】
本地 BiLSTM 判断：{base_signal}，分数 {base_score}，原理由：{base_reason}

【近3日同视角历史判断】
{memory}

【输出要求】
只输出一个 JSON 对象，不要解释文字、不要 markdown、不要代码块。
{{"signal": "...", "score": 0.0, "reason": "...", "tools_used": ["calc_indicators", "local_model_score"]}}
signal 只能取 BUY、SELL、HOLD 三个单词之一，直接写单词本身。
score 取 -1 到 1 之间，正数看多、负数看空，方向必须与 signal 一致；HOLD 时取接近 0 的值。
reason 不超过 40 个汉字，必须提到上面至少一个具体数值。"""

FUND_PROMPT = """你是 A 股基本面分析师，只负责估值，不看 K 线、不看新闻。

【研判纪律】
1. 结论必须由你自己判断。下面的 PE/PB 历史分位是事实，但"分位低就一定该买"
   不是必然结论——估值处在极端位置时，要说明它可能反映的是什么。
2. 只能使用下面给出的数字。
3. 分位数据不足、或两个指标方向矛盾时，可以判 HOLD，并在 reason 写明。

【标的】{symbol}　【交易日】{date}

【工具 get_valuation 返回值】
PE_TTM {pe_ttm}，在历史序列中的分位 {pe_pct}
PB {pb}，历史分位 {pb_pct}
PS_TTM {ps_ttm}
该标的可得估值序列长度 {n_history} 条（分位即以此为样本）

【近3日同视角历史判断】
{memory}

【输出要求】
只输出一个 JSON 对象，不要解释文字、不要 markdown、不要代码块。
{{"signal": "...", "score": 0.0, "reason": "...", "tools_used": ["get_valuation"]}}
signal 只能取 BUY、SELL、HOLD 三个单词之一，直接写单词本身。
score 取 -1 到 1 之间，正数表示估值支持看多、负数表示支持看空，方向必须与 signal 一致；HOLD 时取接近 0 的值。
reason 不超过 40 个汉字，必须提到至少一个分位数或具体估值。"""

FUND_PROMPT_V2 = """你是 A 股基本面分析师。你从三个角度分析：公司经营事实（财报）、估值水平、行业归属。

【研判纪律】
1. 材料有主次，按下面顺序使用：
   - **公司经营事实（财报事件）** 是最硬的材料。净利润同比大幅下滑、当期亏损 → 偏空；
     净利润同比大幅增长 → 偏多。但你的判断置信度有限（这类信号在可交易口径下的
     历史准确率约 0.49~0.54，接近随机），所以**即使方向明确，score 也不得超过 ±0.30**。
   - **估值分位** 只能说明当前价格在历史区间的位置，本身不构成方向。但极端位置可以作为
     弱旁证：PE 分位高于 0.90 时略偏空（估值透支），低于 0.10 时略偏多（估值极低）。
     这类旁证的 score 不得超过 ±0.15。
   - **行业归属** 只说明它跟谁同类，你没有同行业收益对比数据，**不参与方向判断**，
     只在 reason 里用来交代标的属性。
2. 两类材料方向一致时，取较强的那一个，但叠加后仍不超过 ±0.30。
3. 两类材料互相矛盾（例如净利大幅下滑但估值极低，或净利高增但估值极高）→ 判 HOLD，
   在 reason 里写明矛盾在哪。不要强行选一边。
4. 完全没有财报事件、且估值分位在 0.10~0.90 的中性区间 → 判 HOLD，即"没有材料就没有判断"。
5. 只能使用下面给出的数字，不得引入没有提供的信息，不得编造同行业对比。
6. 某项材料写"（无）"就是真的没有，不要猜。

【标的】{symbol}　【交易日】{date}

【工具 get_valuation 返回值】
PE_TTM {pe_ttm}，历史分位 {pe_pct}
PB {pb}，历史分位 {pb_pct}
PS_TTM {ps_ttm}
估值序列长度 {n_history} 条

【工具 get_industry 返回值】
行业 {industry}（大类 {sector}），同类股票 {peer_count} 只

【工具 get_fund_events 返回值（近90日财报事件，不含当日）】
{events}

【近3日同视角历史判断】
{memory}

【输出要求】
只输出一个 JSON 对象，不要解释文字、不要 markdown、不要代码块。
{{"signal": "...", "score": 0.0, "reason": "...", "tools_used": ["get_valuation", "get_industry", "get_fund_events"]}}
signal 只能取 BUY、SELL、HOLD 三个单词之一，直接写单词本身。
score 取 -1 到 1 之间，正数看多、负数看空，方向必须与 signal 一致；HOLD 时取接近 0 的值。
reason 不超过 40 个汉字，必须提到上面至少一个具体数值或事实。"""


FUND_PROMPT_V3 = """你是 A 股基本面分析师。基本面按定义分三层：宏观、行业、公司财务。
你没有宏观数据，所以只从**公司财务**和**行业归属**两层判断，外加估值做旁证。

【研判纪律】
1. 材料共四类，权重从高到低：
   - **公司财务指标（ROE、毛利率、营收/净利增速、负债率、每股经营现金流、流动比率、
     存货周转）** 是最硬的"公司好不好"证据。判读方式：
     高 ROE（明显高于 10%）+ 高毛利率 + 营收净利双增 + 负债率不高 → 偏多；
     净利增速转负、毛利率下滑、负债率高企、经营现金流为负 → 偏空。
     但要注意：财务反映的是**公司质地**，不是**明天涨不涨**。财报是季度数据，
     一年只更新四次，对次日涨跌的解释力天然有限。所以**即使方向明确，
     score 也不得超过 ±0.35**。
   - **财报事件** 是最新一期的即时业绩信号（业绩预告/财报/亏损）。
     与财务指标方向一致时算加强，矛盾时以更新的那一方为准。score 不得超过 ±0.30。
   - **估值分位** 只说明当前价格在历史区间的位置，本身不构成方向。极端位置可作弱旁证：
     PE 分位高于 0.90 略偏空（估值透支），低于 0.10 略偏多（估值极低）。score 不得超过 ±0.15。
   - **行业归属** 只说明它跟谁同类。你没有同行业收益对比数据，**不参与方向判断**，
     只在 reason 里交代标的属性。
2. 多类材料同向时取加权后的合力，叠加后总 score 仍不超过 ±0.40。
3. 材料互相矛盾（例如财务优秀但估值极高，或财务恶化但估值极低）→ 判 HOLD，
   在 reason 写明矛盾。不要强行选一边。
4. 完全没有财务数据、没有财报事件、估值也落在 0.10~0.90 中性区间 → 判 HOLD，
   即"没有材料就没有判断"。
5. 只能使用下面给出的数字，不得引入没有提供的信息，不得编造同行业对比、
   不得引用你看过的其他年份财报。
6. 某项材料写"（无）"就是真的没有，不要猜。

【标的】{symbol}　【交易日】{date}

【工具 get_fundamentals 返回值（已按披露日约束，只含当日已知的期）】
报告期 {period}（法定披露截止 {announce_date}）
ROE {roe}%，毛利率 {gross_margin}%，营收同比 {revenue_yoy}%，净利同比 {profit_yoy}%
资产负债率 {debt_ratio}%，每股经营现金流 {ocf_ps} 元，流动比率 {current_ratio}，存货周转 {inventory_turn} 次
同比趋势（对比 {prev_period}）{trend}

【工具 get_fund_events 返回值（近90日财报事件，不含当日）】
{events}

【工具 get_valuation 返回值】
PE_TTM {pe_ttm}，历史分位 {pe_pct}
PB {pb}，历史分位 {pb_pct}
PS_TTM {ps_ttm}
估值序列长度 {n_history} 条

【工具 get_industry 返回值】
行业 {industry}（大类 {sector}），同类股票 {peer_count} 只

【近3日同视角历史判断】
{memory}

【输出要求】
只输出一个 JSON 对象，不要解释文字、不要 markdown、不要代码块。
{{"signal": "...", "score": 0.0, "reason": "...", "tools_used": ["get_fundamentals", "get_fund_events", "get_valuation", "get_industry"]}}
signal 只能取 BUY、SELL、HOLD 三个单词之一，直接写单词本身。
score 取 -1 到 1 之间，正数看多、负数看空，方向必须与 signal 一致；HOLD 时取接近 0 的值。
reason 不超过 40 个汉字，必须提到上面至少一个具体数值或事实。"""


def fmt_fundamentals(f):
    """把 get_fundamentals 的返回格式化成 prompt 里的几行文本。"""
    if not f:
        return "（无）该标的当日还没有任何已披露的财务数据"
    trend = f.get("trend")
    if trend:
        parts = []
        for k, label in (("roe", "ROE"), ("gross_margin", "毛利率"), ("debt_ratio", "负债率")):
            if k in trend:
                parts.append("%s%+.2fpp" % (label, trend[k]))
        trend_txt = "；".join(parts) if parts else "（无可比项）"
    else:
        trend_txt = "（无可比项，缺上年同期数据）"
    return trend_txt


def fmt_events(rows):
    if not rows:
        return "（无）近90日内没有财报事件记录"
    lines = []
    for e in rows:
        bits = ["[%s] %s" % (e["date"], e["type"])]
        if e["np_yoy"] is not None:
            bits.append("净利润同比%+.1f%%" % e["np_yoy"])
        if e["rev_yoy"] is not None:
            bits.append("营收同比%+.1f%%" % e["rev_yoy"])
        if e["is_loss"]:
            bits.append("当期亏损")
        if e["evidence"]:
            bits.append("原文：“%s”" % e["evidence"])
        lines.append("- " + "，".join(bits))
    return "\n".join(lines)


SENT_PROMPT = """你是 A 股情绪面分析师，只负责新闻情绪，不看 K 线、不看估值。

【研判纪律】
1. 只依据下面列出的新闻，不得编造没有出现的消息，也不得靠行业常识补方向。
2. 判断的是一两天内的短期情绪，不是这家公司的长期质地。
3. 区分公司自身消息与行业/宏观消息，前者对个股方向的权重更高；
   若新闻整体中性或正负相互抵消，可以判 HOLD 并写明。

【标的】{symbol}　【交易日】{date}

【工具 get_news 返回值（近3日）】
{news}

【近3日同视角历史判断】
{memory}

【输出要求】
只输出一个 JSON 对象，不要解释文字、不要 markdown、不要代码块。
{{"signal": "...", "score": 0.0, "reason": "...", "tools_used": ["get_news"]}}
signal 只能取 BUY、SELL、HOLD 三个单词之一，直接写单词本身。
score 取 -1 到 1 之间，正数表示利好、负数表示利空，方向必须与 signal 一致；HOLD 时取接近 0 的值。
reason 不超过 40 个汉字，必须提到具体新闻内容或写明无新闻。"""


# ---------------------------------------------------------------- LLM 调用
def _session():
    """线程内复用连接：不这样做，几百次请求会成片触发 SSLError。"""
    s = getattr(_LOCAL, "s", None)
    if s is None:
        s = requests.Session()
        _LOCAL.s = s
    return s


def _call(prompt, key, retries=3):
    """调模型。429 走长退避，网络类错误走短退避，其余 4xx 直接抛（重试无意义）。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = _session().post(
                BASE_URL + "/chat/completions",
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "effort": EFFORT,
                },
                timeout=(10, 60),
            )
            resp.raise_for_status()
            j = resp.json()
            usage = j.get("usage") or {}
            with _LOCK:
                _STATS["in_tokens"] += int(usage.get("prompt_tokens") or 0)
                _STATS["out_tokens"] += int(usage.get("completion_tokens") or 0)
                _STATS["calls"] += 1
            ch = j["choices"][0]
            text = (ch["message"].get("content") or "").strip()
            if not text:
                raise ValueError("empty content (finish_reason=%s)" % ch.get("finish_reason"))
            text = text.replace("```json", "").replace("```", "").strip()
            # 模型偶尔在 JSON 前后加一句话，取第一个 { 到最后一个 }
            if not text.startswith("{"):
                i, k = text.find("{"), text.rfind("}")
                if i >= 0 and k > i:
                    text = text[i:k + 1]
            return json.loads(text)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if status != 429:
                raise
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
    """校验并归正：signal 枚举、score 范围、方向与分数一致性。"""
    sig = str(out.get("signal", "")).strip().strip('"').upper()
    if sig not in VALID_SIGNALS:
        hits = [s for s in VALID_SIGNALS if s in sig]
        if len(hits) == 1:
            sig = hits[0]
        else:
            raise ValueError("bad signal: %r" % out.get("signal"))

    try:
        score = float(out.get("score", 0.0))
    except Exception:
        score = 0.0
    score = max(-1.0, min(1.0, score))
    if sig == "BUY" and score < 0:
        score = -score
    elif sig == "SELL" and score > 0:
        score = -score
    elif sig == "HOLD":
        score = max(-0.2, min(0.2, score))

    reason = str(out.get("reason", "")).strip().replace("\n", " ")
    if len(reason) > 60:
        reason = reason[:60]
    if not reason:
        raise ValueError("empty reason")
    return sig, score, reason


# ---------------------------------------------------------------- 单样本
def build_prompt(conn, symbol, date, view):
    """按视角取各自工具数据并拼 prompt。返回 (prompt, fallback) 或 (None, 原因)。"""
    mem = fmt_memory(recent_memory(conn, symbol, date, view))

    if view == "tech":
        ind = calc_indicators(conn, symbol, date)
        if ind is None:
            return None, "no_prices"
        base = local_model_score(conn, symbol, date)
        if base is None:
            return None, "no_base"
        prompt = TECH_PROMPT.format(
            symbol=symbol, date=date,
            close=ind["close"], ma5=ind["ma5"], ma10=ind["ma10"], ma20=ind["ma20"],
            vs_ma5=ind["vs_ma5"], vs_ma20=ind["vs_ma20"], rsi6=ind["rsi6"],
            macd_dif=ind["macd_dif"], macd_dea=ind["macd_dea"], macd_hist=ind["macd_hist"],
            vol_ratio=_v(ind["vol_ratio"]), amplitude=ind["amplitude"],
            ret5=ind["ret5"], ret20=ind["ret20"], ann_vol=ind["ann_vol"],
            base_signal=base["signal"], base_score=base["score"],
            base_reason=_v(base["reason"]), memory=mem,
        )
        return prompt, base

    if view == "fund":
        val = get_valuation(conn, symbol, date)
        if val is None:
            return None, "no_valuation"

        if PV["fund"] == "fund-agent-v3":
            ind = get_industry(conn, symbol, date)
            evs = get_fund_events(conn, symbol, date)
            fund = get_fundamentals(conn, symbol, date)
            # 四类材料全空 → 这视角没有信息，弃权不落库（与 sent 无新闻同理）
            if fund is None and not evs and val["pe_pct"] is None:
                return None, "no_fund_material"
            if fund is None:
                roe = gross_margin = revenue_yoy = profit_yoy = None
                debt_ratio = ocf_ps = current_ratio = inventory_turn = None
                period_txt = announce_txt = "（无已披露财务数据）"
            else:
                roe, gross_margin = fund["roe"], fund["gross_margin"]
                revenue_yoy, profit_yoy = fund["revenue_yoy"], fund["profit_yoy"]
                debt_ratio, ocf_ps = fund["debt_ratio"], fund["ocf_ps"]
                current_ratio, inventory_turn = fund["current_ratio"], fund["inventory_turn"]
                period_txt, announce_txt = fund["period"], fund["announce_date"]
            prompt = FUND_PROMPT_V3.format(
                symbol=symbol, date=date,
                period=period_txt, announce_date=announce_txt,
                roe=_v(roe), gross_margin=_v(gross_margin),
                revenue_yoy=_v(revenue_yoy), profit_yoy=_v(profit_yoy),
                debt_ratio=_v(debt_ratio), ocf_ps=_v(ocf_ps),
                current_ratio=_v(current_ratio), inventory_turn=_v(inventory_turn),
                prev_period=(fund["prev_period"] if fund and fund["prev_period"] else "（无）"),
                trend=fmt_fundamentals(fund),
                events=fmt_events(evs),
                pe_ttm=_v(val["pe_ttm"]), pb=_v(val["pb"]), ps_ttm=_v(val["ps_ttm"]),
                pe_pct=_v(val["pe_pct"]), pb_pct=_v(val["pb_pct"]),
                n_history=val["n_history"],
                industry=(ind["industry"] if ind else "（无行业记录）"),
                sector=(ind["sector"] if ind else "（未知）"),
                peer_count=(ind["peer_count"] if ind else 0),
                memory=mem,
            )
        elif PV["fund"] == "fund-agent-v2":
            ind = get_industry(conn, symbol, date)
            evs = get_fund_events(conn, symbol, date)
            prompt = FUND_PROMPT_V2.format(
                symbol=symbol, date=date,
                pe_ttm=_v(val["pe_ttm"]), pb=_v(val["pb"]), ps_ttm=_v(val["ps_ttm"]),
                pe_pct=_v(val["pe_pct"]), pb_pct=_v(val["pb_pct"]),
                n_history=val["n_history"],
                industry=(ind["industry"] if ind else "（无行业记录）"),
                sector=(ind["sector"] if ind else "（未知）"),
                peer_count=(ind["peer_count"] if ind else 0),
                events=fmt_events(evs), memory=mem,
            )
        else:
            prompt = FUND_PROMPT.format(
                symbol=symbol, date=date,
                pe_ttm=_v(val["pe_ttm"]), pb=_v(val["pb"]), ps_ttm=_v(val["ps_ttm"]),
                pe_pct=_v(val["pe_pct"]), pb_pct=_v(val["pb_pct"]),
                n_history=val["n_history"], memory=mem,
            )
        fb = conn.execute(
            "SELECT signal,score FROM judgements WHERE view='fund' AND symbol=? "
            "AND date=? AND prompt_version=?", (symbol, date, BASE_PV["fund"]),
        ).fetchone()
        return prompt, ({"signal": fb["signal"], "score": float(fb["score"])} if fb else None)

    # sent
    news = get_news(conn, symbol, date)
    if not news:
        # 没有新闻 = 这个视角没有信息，不等于"判断中性"。
        # 落一条 HOLD 进投票会变成一张实打实的反对票：
        #   tech HOLD + sent HOLD 就凑成两票 HOLD，fund 判什么都不算了。
        # 所以这里直接弃权、不落库，让融合时按"缺这个视角"处理。
        return None, "no_news_coverage"
    news_text = "\n".join(
        "- [%s] %s | %s" % (n["date"], n["title"], n["body"]) for n in news
    )
    prompt = SENT_PROMPT.format(symbol=symbol, date=date, news=news_text, memory=mem)
    return prompt, None


def run_one(conn, symbol, date, view, key):
    prompt, fallback = build_prompt(conn, symbol, date, view)
    if prompt is None:
        return None, fallback

    try:
        out = _call(prompt, key)
        sig, score, reason = normalize(out)
        provider, model, source = PROVIDER, MODEL, "live"
        tools = out.get("tools_used") or []
        if tools:
            reason = "%s｜工具:%s" % (reason, ",".join(str(t) for t in tools))[:70]
    except Exception as e:
        with _LOCK:
            _STATS["failed"] += 1
            _STATS["fallback"] += 1
            k = "err_" + type(e).__name__
            _STATS[k] = _STATS.get(k, 0) + 1
        if fallback is None:
            # 情绪面没有可降级的下游判断，只能给 HOLD；这一条必须计入降级率
            sig, score = "HOLD", 0.0
            reason = "无新闻且LLM调用失败（降级HOLD）"
            provider, model, source = "local", "sent-fallback", "offline_rule"
        else:
            sig, score = fallback["signal"], float(fallback["score"])
            reason = "（LLM降级：%s）沿用旧版判断" % type(e).__name__
            provider, model, source = "local", BASE_PV.get(view, "local"), "offline_rule"

    db.upsert_judgement(
        conn, symbol, date, view, sig, score,
        reason=reason, provider=provider, model=model,
        temperature=(TEMPERATURE if source == "live" else None),
        prompt_version=PV[view], source=source,
    )
    return {"symbol": symbol, "date": date, "view": view,
            "signal": sig, "score": round(score, 4),
            "live": source == "live"}, "ok"


# ---------------------------------------------------------------- 评估
# 加权融合的权重。不是拍的：docs/calib_weights.json 在验证集（2024-03-15 ~ 08-07）
# 上标定出 w_tech=1.0 / w_fund=0.0（fund_only 准确率 47.92%、mean_next_ret 为负）；
# sent 没有独立标定，沿用 scripts/orchestrate.py 的 arbitrate_rule() 里的 0.5。
# 2024-12 样本外检验：等权 1:1:1 是 0.4577，本配置 0.6098（h=1），
# 对照表见 docs/三分析师预实验诊断.md 第 9.10 节。
WEIGHTS = {"tech": 1.0, "fund": 0.0, "sent": 0.5}
SCORE_TH = 0.05


def _acc(pairs):
    """pairs 为 [(signal, label)]；HOLD 与无 label 的不计入分母。"""
    hit = tot = 0
    for sig, lab in pairs:
        if lab is None or sig == "HOLD":
            continue
        tot += 1
        if (sig == "BUY" and lab == 1) or (sig == "SELL" and lab == 0):
            hit += 1
    return {"n": tot, "hit": hit, "acc": round(hit / tot, 4) if tot else None}


def _dist(pairs):
    return {s: sum(1 for x, _ in pairs if x == s) for s in VALID_SIGNALS}


def evaluate(conn, start, end):
    """单视角 + 投票 + 分数加权，三种口径一起算。

    口径（三条都会写进论文，不能含糊）：
      1. label = 次日涨跌（1 涨 / 0 跌）。只取有 t+1 行情记录的日期——
         数据最后一天（2024-12-31）没有次日行情却被填成 0，必须排除，
         否则当天全判 SELL 都算对，准确率虚高。
      2. HOLD 不计入分母，与 /api/accuracy 一致。
      3. 加权只对有记录的视角求和：缺视角=弃权，分母同步缩小。
         这样"没有新闻"不会像多数决那样变成一张反对票。
    """
    labels = {}
    for r in conn.execute(
        "SELECT p.symbol,p.date,p.label FROM prices p "
        "WHERE p.date BETWEEN ? AND ? "
        "  AND EXISTS (SELECT 1 FROM prices q "
        "              WHERE q.symbol=p.symbol AND q.date>p.date)",
        (start, end),
    ).fetchall():
        labels[(r["symbol"], r["date"])] = r["label"]

    rows = conn.execute(
        "SELECT symbol,date,view,signal,score FROM judgements "
        "WHERE date BETWEEN ? AND ? AND prompt_version IN (?,?,?)",
        (start, end, PV["tech"], PV["fund"], PV["sent"]),
    ).fetchall()

    by_view = {v: [] for v in PV}
    by_day = {v: {} for v in PV}
    per_sample = {}
    for r in rows:
        key = (r["symbol"], r["date"])
        lab = labels.get(key)
        by_view[r["view"]].append((r["signal"], lab))
        by_day[r["view"]].setdefault(r["date"], []).append((r["signal"], lab))
        per_sample.setdefault(key, {})[r["view"]] = (r["signal"], float(r["score"]))

    out = {}
    for v in PV:
        e = _acc(by_view[v])
        e["dist"] = _dist(by_view[v])
        out[v] = e

    # 多数决：三视角齐全才算一票，缺视角时该样本不参与
    vote_pairs, vote_dist = [], {s: 0 for s in VALID_SIGNALS}
    wide_pairs, wide_dist = [], {s: 0 for s in VALID_SIGNALS}
    for (sym, d), views in per_sample.items():
        lab = labels.get((sym, d))
        if len(views) == len(PV):
            sigs = [views[v][0] for v in PV]
            counts = {s: sigs.count(s) for s in VALID_SIGNALS}
            top = max(counts.values())
            win = [s for s in VALID_SIGNALS if counts[s] == top]
            final = win[0] if len(win) == 1 else "HOLD"
            vote_dist[final] += 1
            vote_pairs.append((final, lab))

        num = den = 0.0
        for v, (sig, sc) in views.items():
            w = WEIGHTS.get(v, 1.0)
            num += w * sc
            den += w
        if den <= 0:
            continue
        fs = num / den
        final = "BUY" if fs > SCORE_TH else ("SELL" if fs < -SCORE_TH else "HOLD")
        wide_dist[final] += 1
        wide_pairs.append((final, lab))

    out["vote"] = dict(_acc(vote_pairs), dist=vote_dist)
    out["weighted"] = dict(_acc(wide_pairs), dist=wide_dist)
    out["weighted_weights"] = WEIGHTS
    out["score_th"] = SCORE_TH

    # 按日分层：单日涨跌分布很极端的日期不能被合并后掩盖
    per_day = {}
    for d in sorted({k[1] for k in per_sample}):
        row = {}
        for v in PV:
            if d in by_day[v]:
                row[v] = _acc(by_day[v][d])
        pairs = []
        for (sym, dd), views in per_sample.items():
            if dd != d:
                continue
            num = den = 0.0
            for v, (sig, sc) in views.items():
                w = WEIGHTS.get(v, 1.0)
                num += w * sc
                den += w
            if den <= 0:
                continue
            fs = num / den
            f = "BUY" if fs > SCORE_TH else ("SELL" if fs < -SCORE_TH else "HOLD")
            pairs.append((f, labels.get((sym, dd))))
        row["weighted"] = _acc(pairs)
        per_day[d] = row
    out["per_day"] = per_day

    return out


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--limit", type=int, default=50, help="取前 N 只标的")
    ap.add_argument("--views", default="tech,fund,sent")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry_run", action="store_true",
                    help="只打印一条样例的完整 prompt，不调模型、不落库")
    ap.add_argument("--only_missing", action="store_true",
                    help="跳过已经写过的 (symbol,date,view)")
    ap.add_argument("--fund_pv", default="fund-agent-v1",
                    choices=["fund-agent-v1", "fund-agent-v2", "fund-agent-v3"],
                    help="基本面版本：v1 只看估值；v2 加行业归属与财报事件；"
                         "v3 再加公司财务指标（ROE/毛利率/增速/负债率/现金流等）")
    args = ap.parse_args()

    PV["fund"] = args.fund_pv

    views = [v.strip() for v in args.views.split(",") if v.strip() in PV]
    key = api_key()
    if not key and not args.dry_run:
        raise SystemExit("缺少 DeepSeek key：secrets.env 里没有 DEEPSEEK_API_KEY")

    conn = db.connect()
    db.init_db(conn=conn)

    symbols = [r["symbol"] for r in conn.execute(
        "SELECT DISTINCT symbol FROM prices ORDER BY symbol LIMIT ?", (args.limit,)
    ).fetchall()]
    dates = [r["date"] for r in conn.execute(
        "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
        (args.start, args.end),
    ).fetchall()]
    print("标的 %d，交易日 %d，视角 %s，模型 %s" % (len(symbols), len(dates), views, MODEL))
    if not symbols or not dates:
        raise SystemExit("区间内没有数据")

    tasks = [(s, d, v) for d in dates for s in symbols for v in views]

    if args.only_missing:
        done = set()
        for v in views:
            for r in conn.execute(
                "SELECT symbol,date FROM judgements WHERE view=? AND prompt_version=?",
                (v, PV[v]),
            ).fetchall():
                done.add((r["symbol"], r["date"], v))
        tasks = [t for t in tasks if t not in done]

    if args.dry_run:
        s, d, v = tasks[0]
        prompt, fb = build_prompt(conn, s, d, v)
        print("=" * 70)
        print("样例：%s %s %s，prompt 长度 %s 字符" % (s, d, v, len(prompt or "")))
        print("=" * 70)
        print(prompt if prompt else "构建失败：%s" % fb)
        print("=" * 70)
        # 顺便估算 token：中文约 1 字 ≈ 0.7 token
        if prompt:
            print("估算单次输入 token ≈ %d" % int(len(prompt) * 0.75))
        return

    print("待处理 %d 次调用" % len(tasks))
    t0 = time.time()
    done = ok = bad = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fut = {pool.submit(run_one, db.connect(), s, d, v, key): (s, d, v)
               for s, d, v in tasks}
        for f in as_completed(fut):
            s, d, v = fut[f]
            try:
                r, status = f.result()
            except Exception as e:
                r, status = None, "exception:" + type(e).__name__
                print("FAIL", s, d, v, type(e).__name__, e)
            if r is None:
                bad += 1
                print("SKIP", s, d, v, status)
            else:
                ok += 1
            done += 1
            if done % 30 == 0 or done == len(tasks):
                el = time.time() - t0
                print("  进度 %d/%d 成功%d 跳过%d %.0fs" % (done, len(tasks), ok, bad, el))

    el = time.time() - t0
    print("\n=== 汇总 ===")
    print("成功 %d，跳过 %d，耗时 %.1f 分钟" % (ok, bad, el / 60.0))
    print("LLM 调用 %d 次，输入 %d tokens，输出 %d tokens，降级 %d 条（降级率 %.1f%%）"
          % (_STATS["calls"], _STATS["in_tokens"], _STATS["out_tokens"],
             _STATS["fallback"], 100.0 * _STATS["fallback"] / max(1, ok)))
    extra = {k: v for k, v in _STATS.items() if k.startswith("err_")}
    if extra:
        print("失败分类：", extra)

    ev = evaluate(conn, args.start, args.end)
    print("\n=== 预实验指标（HOLD 不计分母；已排除无次日行情的数据边界日）===")
    for v in ("tech", "fund", "sent", "vote", "weighted"):
        if v in ev:
            e = ev[v]
            print("%-9s n=%-4s acc=%-8s dist=%s" % (v, e["n"], e["acc"], e["dist"]))
    print("加权权重 %s，判定阈值 |score| > %.2f" % (ev.get("weighted_weights"), ev.get("score_th", 0)))

    single_best = max((ev[v]["acc"] or 0.0) for v in ("tech", "fund", "sent") if v in ev)
    wide_acc = (ev.get("weighted") or {}).get("acc") or 0.0
    print("\n单视角最高 %.4f，加权 %.4f，差值 %+.2f pp"
          % (single_best, wide_acc, (wide_acc - single_best) * 100))

    print("\n按日分层（样本 <20 的按噪音看待）：")
    for d, row in ev.get("per_day", {}).items():
        parts = ["%s n=%s acc=%s" % (v, row[v]["n"], row[v]["acc"])
                 for v in ("tech", "fund", "sent", "weighted") if v in row]
        print("  %s  %s" % (d, " | ".join(parts)))

    payload = {
        "model": MODEL,
        "provider": PROVIDER,
        "prompt_versions": PV,
        "window": [args.start, args.end],
        "symbols": len(symbols),
        "dates": len(dates),
        "views": views,
        "calls": _STATS["calls"],
        "tokens": {"in": _STATS["in_tokens"], "out": _STATS["out_tokens"]},
        "fallback": _STATS["fallback"],
        "fallback_rate": round(_STATS["fallback"] / max(1, ok), 4),
        "metrics": ev,
        "note": ("三分析师：判断权在 LLM，BiLSTM/估值分位/新闻为工具返回值。"
                 "情绪面无新闻时不落库（弃权，不投反对票）；"
                 "加权融合只对有记录的视角求和；"
                 "评估排除无次日行情的数据边界日。"),
    }
    p = os.path.join(ROOT, "docs", "exp_agents_pilot.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("written", p)

    # 这个固定路径只留"最近一次运行"，换视角或换窗口会互相覆盖
    # （曾经把三视角的记录盖成单视角 fund）。另存一份带视角与窗口的存档，
    # 历史实验记录不再丢失。
    arch_dir = os.path.join(ROOT, "docs", "exp_agents")
    os.makedirs(arch_dir, exist_ok=True)
    arch = os.path.join(arch_dir, "%s_%s_%s.json" % (
        "-".join(views), args.start, args.end))
    with open(arch, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("archived", arch)


if __name__ == "__main__":
    main()

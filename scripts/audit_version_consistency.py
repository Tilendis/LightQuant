"""全链路口径一致性审计：找"上游改了、下游没跟"的地方。

背景：本轮已连续发现两处同源问题——
  1) 技术面 z-score 原在 test 段整段拟合（前视泄漏），重算后 tech-v1 被覆盖；
  2) 覆盖后融合表没有跟着重跑，12 月 1100 格的 tech 分量仍是修复前数据。
这类问题的共同特征是：库里有多个版本并存、下游产物按"生成时刻"固化，
而没有任何机制保证下游跟着上游刷新。本脚本把这类检查固化成可重复执行的审计。

五项检查：
  A 版本清单：每个 (view, prompt_version) 的行数、日期范围、覆盖，列出所有并存版本
  B 融合表复现：用 fuse_views 同一套矩阵/去均值/权重归一化逻辑重算，与库内分数逐格比对
  C 回测目录自洽：tag/版本/prompt_version 是否对得上，summary 与 nav.csv、trades.csv 是否一致
  D 看板指标对账：按 eval_views 的判涨规则重算四视角 ACC，与 view_accuracy.json 比对
  E 文档旧数字扫描：列出仍在正文使用、但未被"作废/禁止/修复前"声明豁免的旧口径数字

只读，不写库、不改文件。退出码：0=全部一致，1=存在不一致项。

用法：
    python scripts/audit_version_consistency.py
    python scripts/audit_version_consistency.py --json docs/audit_consistency_2026-11-19.json
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from app import db  # noqa: E402

import fuse_views as F  # noqa: E402

# 主口径融合版本及其权重（与 docs/HANDOFF-交接说明.md 第 3 节一致）
MAIN_FUSION = "fusion-test-3view-v2"
DEFAULT_WEIGHTS = [0.90, 0.00, 0.10]
VERSIONS = {"tech": "tech-v1", "fund": "fund-rule-v1", "sent": "sent-v1"}

# 旧口径数字：出现在正文即需人工确认是否已声明作废
LEGACY_PATTERNS = [
    "9.36", "0.5236", "0.5237", "2.673", "733 笔", "0.5440", "0.5453",
    "58471", "15.97", "16.88", "15.10", "15.48", "15.66",
]
# 行内出现这些词视为"已声明作废/仅作对照"，不再报为待处理
EXEMPT_WORDS = [
    "作废", "禁止", "不得", "不要", "前视", "修复前", "旧口径", "旧主口径", "旧权重",
    "已作废", "仅作", "参考", "变更", "纠正", "更正", "重跑", "污染", "历史", "重截",
    "不可引用", "（旧）", "(旧)",
]

issues = []   # (level, section, message)


def note(level, section, msg):
    issues.append((level, section, msg))


def load_prices_dates(conn, start, end):
    rows = conn.execute(
        "SELECT DISTINCT date FROM prices WHERE date BETWEEN ? AND ? ORDER BY date",
        (start, end)).fetchall()
    return [r["date"] for r in rows]


# ---------------------------------------------------------------- A 版本清单
def check_inventory(conn):
    print("\n[A] 研判版本清单")
    rows = conn.execute(
        """SELECT view, prompt_version, COUNT(*) n, MIN(date) d0, MAX(date) d1,
                  COUNT(DISTINCT date) nd, COUNT(DISTINCT symbol) ns
           FROM judgements GROUP BY view, prompt_version ORDER BY view, prompt_version"""
    ).fetchall()
    print("  %-9s %-26s %7s %-11s %-11s %5s %5s" %
          ("view", "prompt_version", "rows", "from", "to", "days", "stocks"))
    for r in rows:
        mark = ""
        if r["view"] == "fusion" and r["prompt_version"] == MAIN_FUSION:
            mark = "  <- 主口径"
        print("  %-9s %-26s %7d %-11s %-11s %5d %5d%s" % (
            r["view"], r["prompt_version"], r["n"], r["d0"], r["d1"],
            r["nd"], r["ns"], mark))

    # 同一 view 下并存多版本是"上游重算、下游未跟"的高危结构
    dup = {}
    for r in rows:
        dup.setdefault(r["view"], []).append(r["prompt_version"])
    for v, vers in dup.items():
        if len(vers) > 1:
            note("info", "A", "%s 下并存 %d 个版本：%s（引用时必须显式指定 prompt_version）"
                 % (v, len(vers), ", ".join(vers)))
    return rows


# ------------------------------------------------------------ B 融合表复现
def reproduce_fusion(conn, start, end, weights, versions, method="weighted",
                     threshold=0.2):
    """按 fuse_views.main 的口径复算分数矩阵，返回 (fscore_df, meta)。

    method=weighted 走加权+可用权重归一化；method=vote 走多数决（need=len//2+1），
    两者都必须与 fuse_views.py 逐行同口径，否则审计会给出假结论。
    """
    mats = {}
    for v in F.VIEWS:
        mats[v] = F.load_matrix(conn, v, start, end, versions[v])

    idx = pd.Index(load_prices_dates(conn, start, end))
    if len(idx) == 0:
        raise RuntimeError("prices 表在该区间无交易日")

    cols = mats["tech"].columns
    for v in F.VIEWS:
        if not mats[v].empty:
            cols = cols.union(mats[v].columns)

    parts = {}
    for v in F.VIEWS:
        if mats[v].empty:
            parts[v] = pd.DataFrame(index=idx, columns=cols, dtype=float)
        elif v == "sent":
            full_idx = mats[v].index.union(idx).sort_values()
            full = mats[v].reindex(index=full_idx, columns=cols).ffill()
            parts[v] = full.reindex(index=idx)
        else:
            parts[v] = mats[v].reindex(index=idx, columns=cols)
        if v in ("fund", "sent"):
            parts[v] = F.demean(parts[v])

    out = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for date in idx:
        for sym in cols:
            avail = {}
            for v in F.VIEWS:
                s = parts[v].loc[date, sym]
                if pd.notna(s):
                    avail[v] = float(s)
            if not avail:
                continue
            if method == "weighted":
                ws = sum(weights[F.VIEWS.index(v)] for v in avail)
                if ws <= 0:
                    continue
                fscore = sum(weights[F.VIEWS.index(v)] * s for v, s in avail.items()) / ws
            else:
                sigs = [F.to_signal(s, threshold) for s in avail.values()]
                buys, sells = sigs.count("BUY"), sigs.count("SELL")
                need = len(avail) // 2 + 1
                if buys >= need:
                    fscore = float(max(avail.values()))
                elif sells >= need:
                    fscore = float(min(avail.values()))
                else:
                    fscore = 0.0
            out.loc[date, sym] = max(-1.0, min(1.0, fscore))
    return out, dict(days=len(idx), stocks=len(cols),
                     filled={v: int(mats[v].notna().sum().sum()) for v in F.VIEWS})


def check_fusion(conn, pv, weights, versions, method="weighted", threshold=0.2):
    print("\n[B] 融合表复现：%s（%s，权重 %s，阈值 %.2f）" % (
        pv, method, "/".join("%.2f" % w for w in weights), threshold))
    q = conn.execute(
        """SELECT MIN(date) d0, MAX(date) d1, COUNT(*) n
           FROM judgements WHERE view='fusion' AND prompt_version=?""", (pv,)).fetchone()
    if not q or not q["d0"]:
        note("fail", "B", "库中找不到融合版本 %s" % pv)
        print("  跳过：库中无 %s" % pv)
        return None

    start, end = q["d0"], q["d1"]
    try:
        recomputed, meta = reproduce_fusion(conn, start, end, weights, versions,
                                            method=method, threshold=threshold)
    except RuntimeError as e:
        note("block", "B", str(e))
        print("  跳过：%s" % e)
        return None

    rows = conn.execute(
        """SELECT symbol, date, score FROM judgements
           WHERE view='fusion' AND prompt_version=? AND date BETWEEN ? AND ?""",
        (pv, start, end)).fetchall()
    stored = {(r["symbol"], r["date"]): float(r["score"]) for r in rows}

    # 非交易日残留：库内日期不在 prices 交易日里
    trade_days = set(load_prices_dates(conn, start, end))
    stray = sorted(set(k[1] for k in stored) - trade_days)

    diffs, missing, extra = [], [], []
    for date in recomputed.index:
        for sym in recomputed.columns:
            val = recomputed.loc[date, sym]
            key = (sym, date)
            if pd.isna(val):
                continue
            if key not in stored:
                missing.append(key)
            elif abs(stored[key] - val) > 1e-6:
                diffs.append((key, stored[key], float(val)))
    for key in stored:
        if key[1] in recomputed.index and key[0] in recomputed.columns:
            if pd.isna(recomputed.loc[key[1], key[0]]):
                extra.append(key)

    print("  区间 %s ~ %s，复算 %d 格（覆盖 %d 日 × %d 只），库内 %d 条" % (
        start, end, int(recomputed.notna().sum().sum()),
        meta["days"], meta["stocks"], len(stored)))
    print("  分量有效格：tech %d / fund %d / sent %d（sent 已经 ffill 展开）" % (
        meta["filled"]["tech"], meta["filled"]["fund"], meta["filled"]["sent"]))
    print("  不一致 %d 格 / 库缺 %d 格 / 复算缺 %d 格 / 非交易日残留 %d 日" % (
        len(diffs), len(missing), len(extra), len(stray)))

    if diffs:
        diffs.sort(key=lambda x: -abs(x[1] - x[2]))
        by_month = {}
        for (sym, date), a, b in diffs:
            by_month.setdefault(date[:7], 0)
            by_month[date[:7]] += 1
        print("  不一致月度分布：%s" % ", ".join(
            "%s:%d" % (m, c) for m, c in sorted(by_month.items())))
        print("  最大偏差 %d 例：" % min(5, len(diffs)))
        for (sym, date), a, b in diffs[:5]:
            print("    %s %s 库=%.6f 复算=%.6f Δ=%.6f" % (date, sym, a, b, a - b))
        note("fail", "B", "%s 有 %d 格与当前上游数据复算不一致（最多的月份：%s）" % (
            pv, len(diffs),
            max(by_month.items(), key=lambda kv: kv[1])[0]))
    if missing:
        note("fail", "B", "%s 复算有值但库内缺 %d 格（上游新增了数据，融合没重跑）" % (pv, len(missing)))
    if stray:
        note("fail", "B", "%s 含 %d 个非交易日日期（写库时日期轴取错）" % (pv, len(stray)))
    if extra:
        note("warn", "B", "%s 有 %d 格库内有值但当前上游为空（该格已失效）" % (pv, len(extra)))
    if not (diffs or missing or stray or extra):
        note("ok", "B", "%s 与当前上游逐格一致" % pv)
        print("  结论：逐格一致，可复现")
    return dict(start=start, end=end, diffs=len(diffs), missing=len(missing),
                extra=len(extra), stray=len(stray), cells=int(recomputed.notna().sum().sum()))


# ----------------------------------------------------------- C 回测目录自洽
def check_backtests(conn):
    print("\n[C] 回测目录自洽")
    dirs = sorted(glob.glob(os.path.join(ROOT, "backtest", "result_v2", "*", "summary.json")))
    known = set()
    for r in conn.execute("SELECT DISTINCT prompt_version FROM judgements").fetchall():
        known.add(r["prompt_version"])

    bad = 0
    rows_out = []
    for path in dirs:
        d = os.path.dirname(path)
        name = os.path.basename(d)
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception as e:
            note("fail", "C", "%s summary.json 解析失败：%s" % (name, e))
            bad += 1
            continue
        p = s.get("portfolio", {})
        cfg = s.get("config", {})
        pv_raw = cfg.get("prompt_version")
        # 多视角回测会把各视角版本写成 {"tech": ..., "fund": ...}，不是单一融合版本
        pv_is_map = isinstance(pv_raw, dict)
        pv = None if pv_raw is None else (
            json.dumps(pv_raw, ensure_ascii=False, sort_keys=True) if pv_is_map else str(pv_raw))
        tag = s.get("tag")

        prob = []
        if tag != name:
            prob.append("tag(%s)≠目录名" % tag)
        if pv_is_map:
            missing_ver = [v for v in pv_raw.values() if isinstance(v, str) and v not in known]
            if missing_ver:
                prob.append("视角版本在库里不存在：%s" % ", ".join(missing_ver))
        elif pv and pv not in known:
            prob.append("prompt_version(%s)在库里不存在" % pv)
        if cfg.get("views") and "fusion" in cfg.get("views", []) and pv and not pv_is_map \
                and not pv.startswith("fusion"):
            prob.append("fusion 视图却用了非融合版本 %s" % pv)

        nav_path = os.path.join(d, "nav.csv")
        if os.path.exists(nav_path) and "cum_return" in p:
            nav = pd.read_csv(nav_path, encoding="utf-8-sig")
            # 引擎口径是 nav[-1]/nav[0]-1（首日已含买入成本与手续费，不等于 1.0）
            if len(nav) >= 2:
                last = float(nav["portfolio_nav"].iloc[-1]) / float(nav["portfolio_nav"].iloc[0]) - 1.0
                if abs(last - float(p["cum_return"])) > 1e-9:
                    prob.append("nav 首末比 %.6f ≠ cum_return %.6f" % (last, p["cum_return"]))
        else:
            if "cum_return" not in p:
                prob.append("summary 缺 cum_return")

        tr_path = os.path.join(d, "trades.csv")
        if os.path.exists(tr_path) and "total_trades" in p:
            n = len(pd.read_csv(tr_path, encoding="utf-8-sig"))
            if n != int(p["total_trades"]):
                prob.append("trades 行数 %d ≠ total_trades %d" % (n, int(p["total_trades"])))

        rows_out.append(dict(dir=name, tag=tag, version=pv,
                             cum=p.get("cum_return"), trades=p.get("total_trades"),
                             problems=prob))
        if prob:
            bad += 1
            note("fail", "C", "%s：%s" % (name, "；".join(prob)))
            print("  !! %-28s %s" % (name, "；".join(prob)))
        elif len([r for r in rows_out if not r["problems"]]) <= 15:
            print("  ok  %-28s pv=%-24s cum=%-9s trades=%s" % (
                name, pv, "%.4f" % (p.get("cum_return") or 0), p.get("total_trades")))

    print("  共 %d 个目录，%d 个有问题" % (len(rows_out), bad))
    return rows_out


# --------------------------------------------------------- D 看板指标对账
def check_accuracy(conn):
    print("\n[D] 看板指标对账：view_accuracy.json")
    path = os.path.join(ROOT, "app", "data", "view_accuracy.json")
    if not os.path.exists(path):
        note("block", "D", "缺 view_accuracy.json")
        return None
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    meta = doc.get("meta", {})
    start, end = meta.get("start"), meta.get("end")
    summary = doc.get("summary", {})

    labels = pd.read_sql_query(
        "SELECT symbol, date, label FROM prices WHERE date BETWEEN ? AND ? AND label IS NOT NULL",
        conn, params=(start, end))

    out = {}
    for view, rec in summary.items():
        pv = rec.get("prompt_version")
        j = pd.read_sql_query(
            """SELECT symbol, date, score FROM judgements
               WHERE view=? AND prompt_version=? AND date BETWEEN ? AND ?""",
            conn, params=(view, pv, start, end))
        if j.empty:
            note("fail", "D", "%s/%s 在库中无数据" % (view, pv))
            continue
        m = j.merge(labels, on=["symbol", "date"], how="inner")
        if m.empty:
            note("fail", "D", "%s/%s 无法与标签对齐" % (view, pv))
            continue
        pred = (m["score"] >= 0).astype(int)
        hit = (pred == m["label"]).astype(int)
        acc = round(float(hit.mean()), 4)
        n = int(len(m))
        strong = m["score"].abs() >= 0.2
        n_s = int(strong.sum())
        acc_s = round(float(hit[strong].mean()), 4) if n_s else None
        json_acc = rec.get("acc_all")
        json_n = rec.get("n_all")
        ok = (abs(acc - json_acc) < 5e-4) and (n == json_n) \
             and (rec.get("n_strong") == n_s) \
             and (rec.get("acc_strong") == acc_s)
        out[view] = dict(acc=acc, n=n, acc_strong=acc_s, n_strong=n_s, match=bool(ok))
        if ok:
            print("  ok  %-7s ACC=%.4f n=%d 强=%.4f/%d（与 JSON 一致）" % (
                view, acc, n, acc_s or 0, n_s))
        else:
            note("fail", "D", "%s：重算 ACC=%.4f(n=%d, 强=%s/%d) vs JSON %.4f(n=%s, 强=%s/%s)" % (
                view, acc, n, acc_s, n_s, json_acc, json_n,
                rec.get("acc_strong"), rec.get("n_strong")))
            print("  !!  %-7s ACC=%.4f n=%d 强=%s/%d vs JSON %.4f n=%s 强=%s/%s" % (
                view, acc, n, acc_s, n_s, json_acc, json_n,
                rec.get("acc_strong"), rec.get("n_strong")))
    return out


# --------------------------------------------------------- E 文档旧数字扫描
BANNER_WORDS = ("口径状态", "口径提示", "已作废", "禁止引用", "本文数字")


def check_docs(conn):
    print("\n[E] 文档旧口径数字扫描")
    files = sorted(glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True))
    pattern = re.compile("|".join(
        r"(?<![0-9.])%s(?![0-9])" % re.escape(x) for x in LEGACY_PATTERNS))
    md_strip = re.compile(r"[*`_>|#\s]")
    flagged, bannered, exempt = [], [], 0
    for path in files:
        rel = os.path.relpath(path, ROOT).replace("\\", "/")
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            continue
        # 文件头若有"口径状态/已作废/禁止引用"横幅，正文旧数字视为已声明，只做文件级提示
        has_banner = any(w in "".join(lines[:60]) for w in BANNER_WORDS)
        hits = 0
        for i, line in enumerate(lines, 1):
            plain = md_strip.sub("", line)
            if not pattern.search(plain):
                continue
            if any(w in plain for w in EXEMPT_WORDS):
                exempt += 1
                continue
            hits += 1
            if has_banner:
                bannered.append((rel, i, plain.strip()[:110]))
            else:
                flagged.append((rel, i, plain.strip()[:140]))

    if flagged:
        print("  未声明作废、仍在正文使用旧数字：")
        for rel, i, text in flagged:
            print("  !!  %s:%d  %s" % (rel, i, text))
    if bannered:
        by_file = {}
        for rel, i, _ in bannered:
            by_file.setdefault(rel, []).append(i)
        print("  已有文件级作废横幅、正文仍保留旧数字（可接受）：")
        for rel, ls in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
            print("      %s   %d 处（行 %s…）" % (rel, len(ls), ls[0]))
    if flagged:
        note("warn", "E", "%d 处旧数字在无横幅文档中正文出现，需补作废声明或改数" % len(flagged))
    else:
        note("ok", "E", "无横幅文档中未发现未声明的旧数字（豁免 %d 处、横幅文档 %d 处）"
             % (exempt, len(bannered)))
        print("  结论：无横幅文档中未发现未声明的旧数字（豁免 %d 处、横幅文档 %d 处）"
              % (exempt, len(bannered)))
    return dict(flagged=len(flagged), bannered=len(bannered), exempt=exempt)


def check_app_refs():
    """[F] 应用层是否还在字符串字面量里引用已被取代的融合版本。

    注释、文档串、PV_SHORT 这类"展示映射"不算引用；真正生效的只有引号包裹的
    字面量（默认值、白名单、下拉框 option、派生 return）。
    """
    print("\n[F] 应用层版本引用（字符串字面量）")
    files = ["app/api.py", "app/service_backtest.py", "app/service_judge.py",
             "app/static/index.html", "app/static/app.js", "scripts/selftest.py"]
    lit = re.compile(r"""(['"])fusion-test-3view\1""")
    map_key = re.compile(r"""(['"])fusion-test-3view\1\s*:""")   # 展示映射键，允许
    bad = 0
    for rel in files:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            note("block", "F", "缺文件 %s" % rel)
            continue
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        hits = []
        for i, line in enumerate(lines, 1):
            if not lit.search(line) or map_key.search(line):
                continue
            hits.append((i, line.strip()[:110]))
        if hits:
            bad += len(hits)
            for i, text in hits:
                note("fail", "F", "%s:%d 仍指向旧版本：%s" % (rel, i, text))
                print("  !!  %s:%d  %s" % (rel, i, text))
        else:
            print("  ok  %-32s 无旧版本字面量" % rel)
    if not bad:
        note("ok", "F", "应用层已全部指向 fusion-test-3view-v2")
    return dict(offending=bad)


def check_strategies(conn, main_version, main_dir):
    """[G] strategies 表指标必须能追到回测目录。

    核对优先级：params.tag 指向的目录 > 主口径目录。有产物就逐项对账，
    没产物才标成"仅历史对照"——否则改完指标忘了重跑目录，这里会静默放过。
    """
    print("\n[G] 策略表指标对账")
    rows = pd.read_sql_query("SELECT id, name, params_json, metrics_json FROM strategies", conn)
    out = []
    for _, r in rows.iterrows():
        pj = json.loads(r["params_json"] or "{}")
        mj = json.loads(r["metrics_json"] or "{}")
        pv, tag = pj.get("fusion_version"), pj.get("tag")
        item = dict(id=int(r["id"]), name=r["name"], version=pv, tag=tag,
                    cum=mj.get("cum_return"), sharpe=mj.get("sharpe"),
                    mdd=mj.get("mdd"), trades=mj.get("trades"))
        out.append(item)

        if pv == main_version:
            dir_name, label = main_dir, "主口径"
        elif tag:
            dir_name, label = tag, "产物 tag"
        else:
            dir_name = label = None

        sp = (os.path.join(ROOT, "backtest", "result_v2", dir_name, "summary.json")
              if dir_name else None)
        if not sp or not os.path.exists(sp):
            if pv == main_version:
                note("fail", "G", "策略 #%s 指向主口径但缺 %s" % (r["id"], main_dir))
            else:
                print("  ~   #%s %-34s tag=%s（无产物可对账，仅历史对照）" % (
                    r["id"], r["name"], tag or "—"))
            continue

        with open(sp, "r", encoding="utf-8-sig") as f:
            p = json.load(f)["portfolio"]
        checks = [("cum_return", p.get("cum_return")),
                  ("sharpe", p.get("sharpe")),
                  ("mdd", p.get("mdd")),
                  ("trades", p.get("total_trades"))]
        diffs = [(k, mj.get(k), v) for k, v in checks
                 if v is not None and (mj.get(k) is None or abs(float(mj.get(k)) - float(v)) > 1e-4)]
        if diffs:
            note("fail", "G", "策略 #%s 指标与 %s 不一致：%s" % (r["id"], dir_name, diffs))
            print("  !!  #%s %s 与 %s 不一致：%s" % (r["id"], r["name"], dir_name, diffs))
        else:
            print("  ok  #%s %-34s %s一致(%s) cum=%.4f sharpe=%.3f trades=%s" % (
                r["id"], r["name"], label, dir_name, p["cum_return"],
                p["sharpe"], p["total_trades"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--fusion_version", default=MAIN_FUSION)
    ap.add_argument("--weights", default=",".join("%.2f" % w for w in DEFAULT_WEIGHTS),
                    help="tech,fund,sent 权重，默认主口径 0.90,0.00,0.10")
    ap.add_argument("--tech_version", default=VERSIONS["tech"])
    ap.add_argument("--fund_version", default=VERSIONS["fund"])
    ap.add_argument("--sent_version", default=VERSIONS["sent"])
    ap.add_argument("--json", default=None, help="把审计结果另存为 JSON")
    ap.add_argument("--method", default="weighted", choices=["weighted", "vote"],
                    help="复现用的融合方式，须与目标版本的生成方式一致")
    ap.add_argument("--threshold", type=float, default=0.2,
                    help="vote 模式下的方向阈值")
    ap.add_argument("--main_dir", default="test_fusion_3view_v2",
                    help="主口径回测产物目录名（用于策略表指标对账）")
    args = ap.parse_args()

    weights = [float(x) for x in args.weights.split(",")]
    versions = {"tech": args.tech_version, "fund": args.fund_version,
                "sent": args.sent_version}

    print("=" * 78)
    print("全链路口径一致性审计")
    print("DB: %s" % (args.db or os.path.join(ROOT, "app", "data", "lightquant.db")))
    print("主口径融合版本: %s   权重: %s" % (args.fusion_version, weights))
    print("=" * 78)

    conn = db.connect(args.db)
    payload = {}
    try:
        check_inventory(conn)
        payload["fusion"] = check_fusion(conn, args.fusion_version, weights, versions,
                                         method=args.method, threshold=args.threshold)
        payload["backtests"] = check_backtests(conn)
        payload["accuracy"] = check_accuracy(conn)
        payload["app_refs"] = check_app_refs()
        payload["strategies"] = check_strategies(
            conn, args.fusion_version, args.main_dir)
        payload["docs"] = check_docs(conn)
    finally:
        conn.close()

    print("\n" + "=" * 78)
    fails = [x for x in issues if x[0] == "fail"]
    blocks = [x for x in issues if x[0] == "block"]
    warns = [x for x in issues if x[0] == "warn"]
    print("汇总：fail %d / block %d / warn %d / ok %d" % (
        len(fails), len(blocks), len(warns),
        len([x for x in issues if x[0] == "ok"])))
    for level, sec, msg in blocks + fails + warns:
        print("  [%s][%s] %s" % (level.upper(), sec, msg))
    print("=" * 78)

    if args.json:
        out = dict(
            generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            main_fusion=args.fusion_version, weights=weights,
            summary=dict(fail=len(fails), block=len(blocks), warn=len(warns)),
            issues=[dict(level=l, section=s, message=m) for l, s, m in issues],
            detail=payload,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
        print("结果已写入 %s" % args.json)

    return 1 if (fails or blocks) else 0


if __name__ == "__main__":
    sys.exit(main())

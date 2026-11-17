# -*- coding: utf-8 -*-
"""情绪面第 4 种实现：768 维 news_embedding 的完整检验（零 LLM、零成本）。

跑三件事，输出一份可直接引用的 JSON：
  1. 覆盖：全零向量代表"当天没新闻"，必须剔除；报告真实可用样本量
  2. 时序方向：逻辑回归预测 T+1 涨跌，与"永远猜跌"基线比
  3. 截面排序：每天按概率排序取 top/bottom，看多空差

用法：
    .\\.venv\\Scripts\\python.exe scripts\\probe_sent_embedding.py

产物：docs/sent_embedding_probe.json

口径说明（与主口径严格对齐）：
- label：T+1 相对 T 的涨跌，用 Close 算，同 prices.label
- 训练/验证/测试切分：按日期切，不看随机数
  train < 2024-03-15 / val 2024-03-15~2024-08-07 / test >= 2024-08-08
- 基线必须同时报随机、永远猜涨、永远猜跌——"永远猜跌"是真实基线，
  因为样本上涨率不到 50%
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = os.path.join(ROOT, "dataset", "CSMD50")
EMB = os.path.join(DS, "news_embedding")
PRICE = os.path.join(DS, "price")
OUT = os.path.join(ROOT, "docs", "sent_embedding_probe.json")

VAL0, VAL1, TEST0 = "2024-03-15", "2024-08-07", "2024-08-08"
# C 必须小：768 维 / 1 万样本，不强正则就是纯过拟合
REG_C = 0.05


def _load_symbol(sym):
    pdf = pd.read_csv(os.path.join(PRICE, sym + ".csv"))
    pdf["Date"] = pdf["Date"].astype(str).str[:10]
    close = pdf["Close"].astype(float).values
    dates = pdf["Date"].tolist()
    lab, fwd = {}, {}
    for i in range(len(dates) - 1):
        if close[i] == 0:
            continue
        fwd[dates[i]] = close[i + 1] / close[i] - 1.0
        lab[dates[i]] = 1 if close[i + 1] > close[i] else 0
    return lab, fwd


def collect():
    """返回 (X, y, dates, syms, fwd)。只保留有真实新闻向量的样本。"""
    syms = sorted(x for x in os.listdir(EMB) if os.path.isdir(os.path.join(EMB, x)))
    vecs, labels, dts, ss, fwds = [], [], [], [], []
    for s in syms:
        lab, fwd = _load_symbol(s)
        ed = os.path.join(EMB, s)
        for f in os.listdir(ed):
            if not f.endswith(".npy"):
                continue
            d = f[:-4]
            if d not in lab:
                continue
            v = np.load(os.path.join(ed, f)).astype(np.float32).reshape(-1)
            # 全零 = 当天没有新闻，不是"中性新闻"。当样本用会让模型学到"没新闻->猜跌"
            if (v == 0).all() or not np.isfinite(v).all():
                continue
            vecs.append(v)
            labels.append(lab[d])
            dts.append(d)
            ss.append(s)
            fwds.append(fwd[d])
    return (np.vstack(vecs), np.array(labels, dtype=int), np.array(dts),
            np.array(ss), np.array(fwds, dtype=float))


def main():
    X, y, dts, _syms, fwd = collect()
    n = len(y)
    print("有效样本: %d" % n)
    if n < 100:
        print("样本不足，终止")
        return

    tr = dts < VAL0
    va = (dts >= VAL0) & (dts <= VAL1)
    te = dts >= TEST0
    print("train %d / val %d / test %d" % (tr.sum(), va.sum(), te.sum()))

    report = {"data": dict(
        total=n, train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()),
        dim=int(X.shape[1]),
        note="全零向量代表当天无新闻，已剔除；不是中性新闻",
    )}

    # ---------- 1. 基线 ----------
    baselines = []
    for name, m in (("train", tr), ("val", va), ("test", te)):
        if m.sum() == 0:
            continue
        up = float(y[m].mean())
        baselines.append(dict(split=name, n=int(m.sum()),
                              up_rate=round(up, 4),
                              always_up=round(up, 4),
                              always_down=round(1 - up, 4),
                              random=0.5))
    report["baselines"] = baselines

    # ---------- 2. 时序方向 ----------
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    clf = LogisticRegression(C=REG_C, max_iter=2000).fit(Xs[tr], y[tr])
    report["train_acc"] = round(float((clf.predict(Xs[tr]) == y[tr]).mean()), 4)

    seq = []
    for name, m in (("train", tr), ("val", va), ("test", te)):
        if m.sum() == 0:
            continue
        pred = clf.predict(Xs[m])
        prob = clf.predict_proba(Xs[m])[:, 1]
        ic = float(np.corrcoef(prob, y[m])[0, 1])
        seq.append(dict(split=name, n=int(m.sum()),
                        acc=round(float((pred == y[m]).mean()), 4),
                        ic=round(ic, 4),
                        long_ratio=round(float(pred.mean()), 4)))
    report["sequential"] = seq

    # 分档：看是否单调。单调才有排序能力，即使绝对值校准差
    buckets_all = []
    for name, m in (("val", va), ("test", te)):
        if m.sum() < 50:
            continue
        prob = clf.predict_proba(Xs[m])[:, 1]
        q = np.quantile(prob, [0, .2, .4, .6, .8, 1.0])
        bs = []
        for k in range(5):
            sel = (prob >= q[k]) & (prob <= q[k + 1])
            if sel.sum() == 0:
                continue
            bs.append(dict(q=round(float(q[k]), 4), n=int(sel.sum()),
                           up_rate=round(float(y[m][sel].mean()), 4)))
        buckets_all.append(dict(split=name, buckets=bs))
    report["buckets"] = buckets_all

    # ---------- 3. 截面排序 ----------
    df = pd.DataFrame(dict(date=dts, p=clf.predict_proba(Xs)[:, 1], fwd=fwd))
    cross = []
    for name, lo, hi in (("val", VAL0, VAL1), ("test", TEST0, "2024-12-31")):
        sub = df[(df["date"] >= lo) & (df["date"] <= hi)]
        spreads, tops, bots, cand = [], [], [], []
        for _d, g in sub.groupby("date"):
            g = g.dropna(subset=["p", "fwd"])
            if len(g) < 3:
                continue
            cand.append(len(g))
            k = min(max(1, len(g) // 3), 5)
            tops.append(float(g.nlargest(k, "p")["fwd"].mean()))
            bots.append(float(g.nsmallest(k, "p")["fwd"].mean()))
            spreads.append(tops[-1] - bots[-1])
        if len(spreads) < 3:
            cross.append(dict(split=name, n_days=len(spreads)))
            continue
        sp = np.array(spreads)
        t = float(sp.mean() / (sp.std(ddof=1) / np.sqrt(len(sp))))
        cross.append(dict(
            split=name, n_days=len(sp),
            cand_per_day=round(float(np.mean(cand)), 1),
            top_mean_pct=round(float(np.mean(tops)) * 100, 3),
            bot_mean_pct=round(float(np.mean(bots)) * 100, 3),
            spread_pp=round(float(sp.mean()) * 100, 3),
            spread_t=round(t, 2),
            winrate=round(float((sp > 0).mean()), 3),
        ))
    report["cross_section"] = cross

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("写入:", OUT)

    print("\n=== 时序 ===")
    for r in seq:
        print("  %-5s n=%-5d acc=%.4f ic=%+.4f" % (r["split"], r["n"], r["acc"], r["ic"]))
    print("=== 截面 ===")
    for r in cross:
        if "spread_pp" in r:
            print("  %-5s 天数=%-4d 多空差=%.3fpp t=%.2f 胜率=%.1f%%" % (
                r["split"], r["n_days"], r["spread_pp"], r["spread_t"], r["winrate"] * 100))
        else:
            print("  %-5s 天数不足" % r["split"])


if __name__ == "__main__":
    sys.exit(main())

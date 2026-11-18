# -*- coding: utf-8 -*-
"""P0-1 校准改进实验：val段拟合isotonic回归，test段验证。
红线：只在val段调参，test段只做验证；不替换主口径数字。
输出：校准前后 ACC / Brier / ECE 对比 + 分箱可靠性表。
"""
import sqlite3, sys
import numpy as np
from sklearn.isotonic import IsotonicRegression
sys.stdout.reconfigure(encoding='utf-8')

DB = r"D:\03_Study\GraduationProject\LightQuant\app\data\lightquant.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

VAL_END = "2024-08-08"

def load(view, start, end):
    rows = conn.execute("""
        SELECT j.score, p.label
        FROM judgements j
        JOIN prices p ON p.symbol=j.symbol AND p.date=j.date
        WHERE j.view=? AND j.date>=? AND j.date<?
    """, (view, start, end)).fetchall()
    scores = np.array([r["score"] for r in rows], dtype=float)
    labels = np.array([r["label"] for r in rows], dtype=float)
    return scores, labels

def metrics(prob, labels):
    n = len(labels)
    pred = (prob > 0.5).astype(int)
    acc = float((pred == labels).mean())
    brier = float(((prob - labels) ** 2).mean())
    # ECE: 分5箱加权平均 |平均概率 - 实际上涨率|
    bins = np.linspace(0, 1, 6)
    ece = 0.0
    for i in range(5):
        m = (prob >= bins[i]) & (prob < bins[i+1])
        if m.sum() == 0:
            continue
        ece += m.sum() / n * abs(prob[m].mean() - labels[m].mean())
    return n, acc, brier, ece

def bin_table(prob, labels):
    bins = np.linspace(0, 1, 6)
    out = []
    for i in range(5):
        m = (prob >= bins[i]) & (prob < bins[i+1])
        if m.sum() == 0:
            continue
        out.append((f"[{bins[i]:.1f},{bins[i+1]:.1f})", int(m.sum()),
                    float(prob[m].mean()), float(labels[m].mean()),
                    float(abs(prob[m].mean() - labels[m].mean()))))
    return out

for view in ["tech", "fund", "sent", "fusion"]:
    v_s, v_y = load(view, "2024-03-15", VAL_END)
    t_s, t_y = load(view, VAL_END, "2024-12-31")
    if len(v_s) < 50 or len(t_s) < 50:
        print(f"\n{view}: 样本不足，跳过")
        continue

    # 原始概率: (score+1)/2
    t_prob_raw = (t_s + 1) / 2
    n0, acc0, br0, ece0 = metrics(t_prob_raw, t_y)

    # val段拟合 isotonic: score -> 概率
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    v_prob_raw = (v_s + 1) / 2
    ir.fit(v_prob_raw, v_y)
    t_prob_cal = ir.predict(t_prob_raw)
    n1, acc1, br1, ece1 = metrics(t_prob_cal, t_y)

    print("=" * 72)
    print(f"视角: {view}  (val={len(v_s)}, test={n0})")
    print("-" * 72)
    print(f"{'口径':<12} {'ACC':>8} {'Brier':>8} {'ECE':>8}")
    print(f"{'校准前':<12} {acc0:>8.4f} {br0:>8.4f} {ece0:>8.4f}")
    print(f"{'isotonic校准后':<12} {acc1:>8.4f} {br1:>8.4f} {ece1:>8.4f}")
    print(f"{'Brier变化':<12} {br1-br0:>+8.4f}   ECE变化 {ece1-ece0:>+8.4f}")

    print("\n校准后分箱可靠性:")
    print(f"{'概率区间':<16} {'样本':>6} {'平均概率':>10} {'上涨率':>10} {'误差':>8}")
    for b, n, mp, ur, err in bin_table(t_prob_cal, t_y):
        print(f"{b:<16} {n:>6} {mp:>10.4f} {ur:>10.4f} {err:>8.4f}")

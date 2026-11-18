# -*- coding: utf-8 -*-
"""P0-2 自适应权重实验：固定权重 vs 滚动命中率动态权重。
红线：滚动窗口只用过去20个交易日历史，不看未来；不替换主口径。
样本：tech/fund对齐（sent缺测时权重重归一化）。
"""
import sqlite3, sys
import numpy as np
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

DB = r"D:\03_Study\GraduationProject\LightQuant\app\data\lightquant.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# 固定主口径权重
FIXED = {"tech": 0.9, "fund": 0.0, "sent": 0.1}
WINDOW = 20  # 滚动命中率窗口（交易日）
START = "2024-08-08"

# 拉取 test 段各视角 (date, symbol, score, label)
data = defaultdict(dict)  # data[(date,symbol)] = {view: score}
labels = {}
rows = conn.execute("""
    SELECT j.view, j.symbol, j.date, j.score, p.label
    FROM judgements j
    JOIN prices p ON p.symbol=j.symbol AND p.date=j.date
    WHERE j.date >= ?
""", (START,)).fetchall()
for r in rows:
    key = (r["date"], r["symbol"])
    data[key][r["view"]] = r["score"]
    labels[key] = r["label"]

# 只保留 tech/fund 都有的样本（sent 可缺）
keys = sorted([k for k, v in data.items() if "tech" in v and "fund" in v])
dates = sorted({k[0] for k in keys})
date_idx = {d: i for i, d in enumerate(dates)}

def hit(pred_score, label):
    return 1 if ((pred_score + 1) / 2 > 0.5) == bool(label) else 0

# 逐日滚动命中率
history = {v: [] for v in ("tech", "fund", "sent")}  # 历史 (hit, )
fixed_prob, dyn_prob, y_true = [], [], []

for d, sym in keys:
    views = data[(d, sym)]
    label = labels[(d, sym)]

    # 动态权重：近WINDOW日命中率，低于0.5记0，再归一化
    raw_w = {}
    for v in ("tech", "fund", "sent"):
        if v in views and len(history[v]) >= 5:
            hr = np.mean(history[v][-WINDOW:])
            raw_w[v] = max(0.0, hr - 0.45)
        else:
            raw_w[v] = 0.0
    # 缺测视角不参与；若全0退回固定权重
    avail = [v for v in ("tech", "fund", "sent") if v in views]
    avail_w = {v: raw_w[v] for v in avail}
    if sum(avail_w.values()) <= 1e-6:
        avail_w = {v: FIXED.get(v, 0.1) for v in avail}
    tot = sum(avail_w.values())
    dyn_w = {v: avail_w[v] / tot for v in avail}

    # 合成 score（-1~1 域）
    f_score = sum(FIXED[v] * views[v] for v in ("tech", "fund", "sent") if v in views)
    d_score = sum(dyn_w[v] * views[v] for v in avail)

    fixed_prob.append((f_score + 1) / 2)
    dyn_prob.append((d_score + 1) / 2)
    y_true.append(label)

    # 更新历史（用当日命中）
    for v in avail:
        history[v].append(hit(views[v], label))

fixed_prob = np.array(fixed_prob)
dyn_prob = np.array(dyn_prob)
y_true = np.array(y_true, dtype=float)

def metrics(prob, y):
    pred = (prob > 0.5).astype(int)
    acc = float((pred == y).mean())
    brier = float(((prob - y) ** 2).mean())
    return acc, brier

acc_f, br_f = metrics(fixed_prob, y_true)
acc_d, br_d = metrics(dyn_prob, y_true)

print("=" * 64)
print(f"动态权重实验（test段，n={len(y_true)}，窗口={WINDOW}交易日）")
print("=" * 64)
print(f"{'口径':<16} {'ACC':>8} {'Brier':>8}")
print("-" * 36)
print(f"{'固定权重0.9/0/0.1':<16} {acc_f:>8.4f} {br_f:>8.4f}")
print(f"{'滚动命中率动态权重':<16} {acc_d:>8.4f} {br_d:>8.4f}")
print("-" * 36)
print(f"{'Δ(动态-固定)':<16} {acc_d-acc_f:>+8.4f} {br_d-br_f:>+8.4f}")

# 各视角单独基线
print("\n各视角单独ACC/Brier:")
for v in ("tech", "fund", "sent"):
    vp, vy = [], []
    for d, sym in keys:
        if v in data[(d, sym)]:
            vp.append((data[(d, sym)][v] + 1) / 2)
            vy.append(labels[(d, sym)])
    if vp:
        a, b = metrics(np.array(vp), np.array(vy, dtype=float))
        print(f"  {v:<6} n={len(vp):>5}  ACC={a:.4f}  Brier={b:.4f}")

# -*- coding: utf-8 -*-
"""消融实验 + 可靠性校准图数据。
输出论文用的表格数据，不生成图片（图片后续用ECharts画）。
"""
import sqlite3, sys, math
sys.stdout.reconfigure(encoding='utf-8')

DB = r"D:\03_Study\GraduationProject\LightQuant\app\data\lightquant.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# 取test段所有view的研判 + 对应prices.label
# label: 1=上涨, 0=下跌
def eval_view(view, start="2024-08-08", end="2024-12-31"):
    rows = conn.execute("""
        SELECT j.symbol, j.date, j.score, p.label
        FROM judgements j
        JOIN prices p ON p.symbol = j.symbol AND p.date = j.date
        WHERE j.view = ? AND j.date >= ? AND j.date <= ?
    """, (view, start, end)).fetchall()
    if not rows:
        return None
    n = len(rows)
    correct = 0
    brier_sum = 0.0
    for r in rows:
        score = r["score"]
        label = r["label"]
        # score映射到概率: (score+1)/2
        prob = (score + 1) / 2
        # 预测方向: prob>0.5预测上涨
        pred = 1 if prob > 0.5 else 0
        if pred == label:
            correct += 1
        # Brier: (prob - label)^2
        brier_sum += (prob - label) ** 2
    acc = correct / n
    brier = brier_sum / n
    return {"n": n, "acc": acc, "brier": brier}

print("=" * 70)
print("消融实验：test段(2024-08-08 ~ 2024-12-31)各视角表现")
print("=" * 70)
print(f"{'视角':<10} {'样本数':>8} {'ACC':>8} {'Brier':>8}")
print("-" * 40)
results = {}
for view in ["tech", "fund", "sent", "fusion"]:
    r = eval_view(view)
    if r:
        results[view] = r
        print(f"{view:<10} {r['n']:>8} {r['acc']:>8.4f} {r['brier']:>8.4f}")

# 分样本：8-9月 / 10-11月 / 12月
print("\n" + "=" * 70)
print("分样本ACC对比")
print("=" * 70)
periods = [
    ("8-9月上涨段", "2024-08-08", "2024-09-30"),
    ("10-11月震荡段", "2024-10-01", "2024-11-30"),
    ("12月", "2024-12-01", "2024-12-31"),
]
print(f"{'区间':<16}", end="")
for v in ["tech", "fund", "sent", "fusion"]:
    print(f"{v:>10}", end="")
print()
print("-" * 56)
for name, s, e in periods:
    print(f"{name:<16}", end="")
    for v in ["tech", "fund", "sent", "fusion"]:
        r = eval_view(v, s, e)
        if r:
            print(f"{r['acc']:>10.4f}", end="")
        else:
            print(f"{'N/A':>10}", end="")
    print()

# 可靠性校准：fusion score分5箱
print("\n" + "=" * 70)
print("可靠性校准：fusion视角score分箱 vs 实际上涨率")
print("=" * 70)
rows = conn.execute("""
    SELECT j.score, p.label
    FROM judgements j
    JOIN prices p ON p.symbol = j.symbol AND p.date = j.date
    WHERE j.view = 'fusion' AND j.date >= '2024-08-08'
""").fetchall()
# 分5箱: [-1,-0.6), [-0.6,-0.2), [-0.2,0.2), [0.2,0.6), [0.6,1]
bins = [(-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]
print(f"{'score区间':<18} {'样本数':>8} {'平均score':>10} {'实际上涨率':>10} {'校准误差':>10}")
print("-" * 60)
for lo, hi in bins:
    subset = [r for r in rows if lo <= r["score"] < hi]
    if not subset:
        continue
    avg_score = sum(r["score"] for r in subset) / len(subset)
    avg_prob = (avg_score + 1) / 2
    up_rate = sum(r["label"] for r in subset) / len(subset)
    err = abs(avg_prob - up_rate)
    print(f"[{lo:+.1f}, {hi:+.1f}){'':<8} {len(subset):>8} {avg_score:>10.4f} {up_rate:>10.4f} {err:>10.4f}")

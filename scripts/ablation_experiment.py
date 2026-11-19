# -*- coding: utf-8 -*-
"""消融实验（单版本配对口径）。

口径修正：早期版本把同一视角下多个 prompt_version 的研判叠加计入，
同一 (symbol, date) 被重复统计，样本数超过价格网格格数，对比不成立。
本脚本每视角只取主口径版本，在四视角共同覆盖的 (symbol, date) 交集上配对计算。

主口径版本：
  tech   = tech-v1            （本地 BiLSTM）
  fund   = fund-rule-v1       （估值分位规则版）
  sent   = sent-v1            （智谱新闻文本）
  fusion = fusion-test-3view-v2（三视角融合主口径）

不改动主口径回测结果，仅输出论文用对照表。
"""
import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')

DB = r"D:\03_Study\GraduationProject\LightQuant\app\data\lightquant.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

VER = {
    "tech": "tech-v1",
    "fund": "fund-rule-v1",
    "sent": "sent-v1",
    "fusion": "fusion-test-3view-v2",
}
START, END = "2024-08-08", "2024-12-31"


def load(view, version):
    """返回 {(symbol, date): (score, label)}，并校验无重复单元格。"""
    rows = conn.execute("""
        SELECT j.symbol, j.date, j.score, p.label
        FROM judgements j
        JOIN prices p ON p.symbol = j.symbol AND p.date = j.date
        WHERE j.view = ? AND j.prompt_version = ?
          AND j.date >= ? AND j.date <= ?
    """, (view, version, START, END)).fetchall()
    out = {}
    for r in rows:
        key = (r["symbol"], r["date"])
        if key in out:
            raise SystemExit(f"重复单元格: {view} {key}")
        out[key] = (r["score"], r["label"])
    return out


def metrics(cells, data):
    n = len(cells)
    correct = 0
    brier = 0.0
    for key in cells:
        score, label = data[key]
        prob = (score + 1) / 2
        pred = 1 if prob > 0.5 else 0
        if pred == label:
            correct += 1
        brier += (prob - label) ** 2
    return {"n": n, "acc": correct / n, "brier": brier / n}


data = {v: load(v, pv) for v, pv in VER.items()}

# 四视角共同交集
inter = set(data["tech"]) & set(data["fund"]) & set(data["sent"]) & set(data["fusion"])

print("=" * 72)
print("一、配对消融表（四视角共同交集，%d 格，主口径版本）" % len(inter))
print("=" * 72)
print(f"{'视角':<8} {'版本':<20} {'样本数':>6} {'ACC':>8} {'Brier':>8}")
print("-" * 56)
paired = {}
for v, pv in VER.items():
    m = metrics(inter, data[v])
    paired[v] = m
    print(f"{v:<8} {pv:<20} {m['n']:>6} {m['acc']:>8.4f} {m['brier']:>8.4f}")

best = max(("tech", "fund", "sent"), key=lambda x: paired[x]["acc"])
delta = paired["fusion"]["acc"] - paired[best]["acc"]
print(f"\n融合相对最强单视角({best}) ACC 差: {delta*100:+.2f}pp")

print("\n" + "=" * 72)
print("二、各视角单版本全量覆盖（未配对，仅作覆盖度参考）")
print("=" * 72)
print(f"{'视角':<8} {'版本':<20} {'覆盖格数':>8} {'ACC':>8} {'Brier':>8}")
print("-" * 56)
for v, pv in VER.items():
    m = metrics(set(data[v]), data[v])
    print(f"{v:<8} {pv:<20} {m['n']:>8} {m['acc']:>8.4f} {m['brier']:>8.4f}")

# 分样本（交集口径）
print("\n" + "=" * 72)
print("三、分样本 ACC（交集口径，提示性观察；子样本 40~50 交易日 × 50 标的，非统计显著）")
print("=" * 72)
periods = [
    ("8-9月上涨段", "2024-08-08", "2024-09-30"),
    ("10-11月震荡段", "2024-10-01", "2024-11-30"),
    ("12月", "2024-12-01", "2024-12-31"),
]
print(f"{'区间':<16}" + "".join(f"{v:>10}" for v in VER))
print("-" * 56)
for name, s, e in periods:
    sub = {k for k in inter if s <= k[1] <= e}
    line = f"{name:<16}"
    for v in VER:
        m = metrics(sub, data[v])
        line += f"{m['acc']:>10.4f}"
    print(line)

# 可靠性校准：fusion 主版本分箱
print("\n" + "=" * 72)
print("四、fusion-test-3view-v2 可靠性分箱（交集口径）")
print("=" * 72)
bins = [(-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)]
print(f"{'score区间':<18} {'样本数':>8} {'平均score':>10} {'实际上涨率':>10} {'校准误差':>10}")
print("-" * 60)
for lo, hi in bins:
    subset = [data["fusion"][k] for k in inter if lo <= data["fusion"][k][0] < hi]
    if not subset:
        continue
    avg_score = sum(s for s, _ in subset) / len(subset)
    up_rate = sum(l for _, l in subset) / len(subset)
    err = abs((avg_score + 1) / 2 - up_rate)
    print(f"[{lo:+.1f}, {hi:+.1f}){'':<8} {len(subset):>8} {avg_score:>10.4f} {up_rate:>10.4f} {err:>10.4f}")

# -*- coding: utf-8 -*-
"""分歧度分层分析：多视角融合到底在什么条件下有用？

动机：主口径融合（weights 0.90 / 0.00 / 0.10）相对最强单视角只有 +1.5pp，
"融合无增量"这个结论一刀切。本脚本把样本按三视角（tech/fund/sent）的
方向一致性分层，检验融合的边际价值是否集中在"视角分歧"的样本上。
若集中在少数分歧样本，则论文可写成"条件性发现"而不是单纯否定。

口径与 ablation_experiment.py 完全一致：
- 每视角只取主口径版本，四视角共同覆盖的 (symbol, date) 交集上配对
- score 取值 [-1, 1]，prob = (score + 1) / 2，prob > 0.5 判为上涨
- 真值 label 来自 prices 表，不做二次派生
- test 段 2024-08-08 ~ 2024-12-31

输出三张表：
  表 A 按三视角同向票数分层（3/2/1/0），分别给融合与各单视角 ACC
  表 B 融合相对 tech 的改判率与改判正确率（融合只在少数样本偏离 tech）
  表 C 融合权重是否影响分层结论（0.9/0/0.1 与等权、0.5/0.5 对比 tech+sent）
"""
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")

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
            raise SystemExit("重复单元格: %s %s" % (view, key))
        out[key] = (r["score"], r["label"])
    return out


def up(score):
    return 1 if (score + 1) / 2 > 0.5 else 0


def acc(cells, data):
    if not cells:
        return None
    hit = sum(1 for k in cells if up(data[k][0]) == data[k][1])
    return hit / len(cells)


def ci(p, n):
    """二项比例的 95% 正态近似区间半宽（pp）。"""
    if not n:
        return 0.0
    return 1.96 * (p * (1 - p) / n) ** 0.5 * 100


data = {v: load(v, pv) for v, pv in VER.items()}
inter = set(data["tech"]) & set(data["fund"]) & set(data["sent"]) & set(data["fusion"])

print("=" * 78)
print("分歧度分层分析  交集 %d 格  版本 %s" % (len(inter), ", ".join(f"{k}={v}" for k, v in VER.items())))
print("=" * 78)

# 表 A：按三视角看多票数分层
print("\n表 A  按三视角看多票数分层（票数 = tech+fund+sent 中判上涨的个数）")
print(f"{'票数':>4} {'样本':>6} {'占比':>7} {'实际上涨率':>10} "
      f"{'tech':>8} {'fund':>8} {'sent':>8} {'fusion':>8} {'融-tech':>9}")
print("-" * 78)
rows_a = {}
for k in (3, 2, 1, 0):
    cells = {x for x in inter if up(data["tech"][x][0]) + up(data["fund"][x][0]) + up(data["sent"][x][0]) == k}
    if not cells:
        continue
    up_rate = sum(data["fusion"][x][1] for x in cells) / len(cells)
    a_tech = acc(cells, data["tech"])
    a_fund = acc(cells, data["fund"])
    a_sent = acc(cells, data["sent"])
    a_fus = acc(cells, data["fusion"])
    rows_a[k] = (cells, a_tech, a_fus)
    print(f"{k:>4} {len(cells):>6} {len(cells)/len(inter)*100:>6.1f}% {up_rate:>10.3f} "
          f"{a_tech:>8.3f} {a_fund:>8.3f} {a_sent:>8.3f} {a_fus:>8.3f} "
          f"{(a_fus - a_tech)*100:>+8.2f}pp")

# 表 B：融合相对 tech 的改判
print("\n表 B  融合相对 tech 的方向改判（weights 0.90/0.00/0.10，融合≈tech + 少量sent）")
flips = {x for x in inter if up(data["fusion"][x][0]) != up(data["tech"][x][0])}
same = inter - flips
print(f"改判样本 {len(flips)} / {len(inter)} = {len(flips)/len(inter)*100:.1f}%；未改判 {len(same)}")
if flips:
    hit_flip = sum(1 for x in flips if up(data["fusion"][x][0]) == data["fusion"][x][1])
    hit_same = sum(1 for x in same if up(data["fusion"][x][0]) == data["fusion"][x][1])
    p_flip = hit_flip / len(flips)
    p_same = hit_same / len(same)
    print(f"改判样本正确率 {p_flip:.3f}（±{ci(p_flip, len(flips)):.1f}pp, n={len(flips)}）  "
          f"未改判样本正确率 {p_same:.3f}（±{ci(p_same, len(same)):.1f}pp）")
    print(f"改判净收益 = {(p_flip - p_same)*100:+.2f}pp（正=改判平均改对了）")

# 表 C：一致 vs 分歧两分层，各视角与融合对比 + 融合权重敏感性
print("\n表 C  按「三视角是否全一致」分层")
unanimous = {x for x in inter
             if up(data["tech"][x][0]) == up(data["fund"][x][0]) == up(data["sent"][x][0])}
split = inter - unanimous
print(f"{'分层':<10} {'样本':>6} {'占比':>7} {'实际上涨率':>10} "
      f"{'tech':>8} {'fund':>8} {'sent':>8} {'fusion':>8} {'融-tech':>9}")
print("-" * 78)
for name, cells in (("全一致", unanimous), ("有分歧", split)):
    if not cells:
        continue
    up_rate = sum(data["fusion"][x][1] for x in cells) / len(cells)
    a = {v: acc(cells, data[v]) for v in VER}
    print(f"{name:<10} {len(cells):>6} {len(cells)/len(inter)*100:>6.1f}% {up_rate:>10.3f} "
          f"{a['tech']:>8.3f} {a['fund']:>8.3f} {a['sent']:>8.3f} {a['fusion']:>8.3f} "
          f"{(a['fusion'] - a['tech'])*100:>+8.2f}pp")

# 表 D：如果只在高分歧样本交易，收益/样本量代价
print("\n表 D  分层作为「选择性交易」过滤器的代价（融合口径）")
for name, cells in (("全样本", inter), ("仅全一致", unanimous), ("仅有分歧", split)):
    if not cells:
        continue
    a = acc(cells, data["fusion"])
    print(f"{name:<10} 可交易格 {len(cells):>5} ({len(cells)/len(inter)*100:>5.1f}%)  "
          f"融合 ACC {a:.3f}（±{ci(a, len(cells)):.1f}pp）")

# 表 E：多数投票（三视角等权）作为对照
print("\n表 E  三视角多数投票 vs 融合 vs tech（交集口径）")
vote = {x: (1 if up(data["tech"][x][0]) + up(data["fund"][x][0]) + up(data["sent"][x][0]) >= 2 else 0)
        for x in inter}
hit_vote = sum(1 for x in inter if vote[x] == data["fusion"][x][1])
hit_tech = sum(1 for x in inter if up(data["tech"][x][0]) == data["tech"][x][1])
hit_fus = sum(1 for x in inter if up(data["fusion"][x][0]) == data["fusion"][x][1])
n = len(inter)
print(f"多数投票 ACC {hit_vote/n:.3f}（±{ci(hit_vote/n, n):.1f}pp）")
print(f"tech     ACC {hit_tech/n:.3f}（±{ci(hit_tech/n, n):.1f}pp）")
print(f"fusion   ACC {hit_fus/n:.3f}（±{ci(hit_fus/n, n):.1f}pp）")

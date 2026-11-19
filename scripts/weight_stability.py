# -*- coding: utf-8 -*-
"""P2-7 权重稳定性检验：val 段切前后两半，分别网格搜融合权重。
离线分析，不写库、不动主口径。目标指标：样本外 ACC。
"""
import sqlite3, sys, itertools
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

DB = r"D:\03_Study\GraduationProject\LightQuant\app\data\lightquant.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

VAL_START, VAL_END = "2024-03-15", "2024-08-08"

# 锁定 val 段主口径版本，避免多版本重复计入覆盖同一单元格
VAL_VER = {"tech": "tech-v1", "fund": "fund-rule-v1", "sent": "sent-v1"}

def load_panel(start, end):
    rows = conn.execute("""
        SELECT j.view, j.symbol, j.date, j.score, j.prompt_version, p.label
        FROM judgements j
        JOIN prices p ON p.symbol=j.symbol AND p.date=j.date
        WHERE j.date >= ? AND j.date < ?
    """, (start, end)).fetchall()
    panel = {}
    for r in rows:
        if VAL_VER.get(r["view"]) != r["prompt_version"]:
            continue
        key = (r["date"], r["symbol"])
        panel.setdefault(key, {})[r["view"]] = r["score"]
        panel[key]["label"] = r["label"]
    return panel

def grid_search(panel):
    """网格搜 wt,wf,ws（步长0.1，和=1），返回最优 ACC 与权重。"""
    keys = list(panel.keys())
    best = {"acc": -1, "w": None}
    # 候选权重组合
    combos = []
    for wt in range(0, 11):
        for wf in range(0, 11 - wt):
            ws = 10 - wt - wf
            combos.append((wt/10, wf/10, ws/10))
    for wt, wf, ws in combos:
        correct = 0
        n = 0
        for k in keys:
            d = panel[k]
            parts, ws_parts = [], []
            for v, w in (("tech", wt), ("fund", wf), ("sent", ws)):
                if v in d and not np.isnan(d[v]) and w > 0:
                    parts.append(w * d[v]); ws_parts.append(w)
            if not ws_parts:
                continue
            fscore = sum(parts) / sum(ws_parts)
            pred = 1 if (fscore + 1) / 2 > 0.5 else 0
            correct += int(pred == d["label"])
            n += 1
        acc = correct / n if n else 0
        if acc > best["acc"]:
            best = {"acc": acc, "w": (wt, wf, ws), "n": n}
    return best

# val 段日期中位数切半
all_dates = sorted({k[0] for k in load_panel(VAL_START, VAL_END).keys()})
mid = all_dates[len(all_dates) // 2]
print(f"val 段交易日 {len(all_dates)} 天，分界日 {mid}")
print(f"前半 {VAL_START} ~ {mid}，后半 {mid} ~ {VAL_END}")

for tag, s, e in [("前半", VAL_START, mid), ("后半", mid, VAL_END), ("全val", VAL_START, VAL_END)]:
    panel = load_panel(s, e)
    b = grid_search(panel)
    print(f"\n{tag} (n={b['n']}) 最优权重 tech/fund/sent = "
          f"{b['w'][0]:.1f}/{b['w'][1]:.1f}/{b['w'][2]:.1f}  ACC={b['acc']:.4f}")

# 主口径固定权重在全val上的ACC
panel = load_panel(VAL_START, VAL_END)
fixed = (0.9, 0.0, 0.1)
correct = n = 0
for k, d in panel.items():
    parts, wp = [], []
    for v, w in zip(("tech", "fund", "sent"), fixed):
        if v in d and not np.isnan(d[v]) and w > 0:
            parts.append(w * d[v]); wp.append(w)
    if not wp: continue
    fs = sum(parts) / sum(wp)
    correct += int((1 if (fs+1)/2 > 0.5 else 0) == d["label"]); n += 1
print(f"\n主口径固定权重 0.9/0/0.1 在全val: ACC={correct/n:.4f} (n={n})")

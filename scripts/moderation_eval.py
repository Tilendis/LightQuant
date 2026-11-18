# -*- coding: utf-8 -*-
"""P1-5 发帖审核规则层评测。
离线复用 service_risk.MODERATION_RULES 正则，对标注集算 P/R/F1。
LLM 层：本地未配置 MODERATOR_PROVIDER/key，本轮不调用，结果留待有 key 后复测；
语义违规类（semantic_violation）设计上即由 LLM 层兜底，规则层不应拦截。
"""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\03_Study\GraduationProject\LightQuant"
sys.path.insert(0, os.path.join(ROOT, "app"))
from service_risk import MODERATION_RULES  # noqa: E402

with open(os.path.join(ROOT, "scripts", "moderation_samples.json"), encoding="utf-8") as f:
    data = json.load(f)
samples = data["samples"]

def rule_block(text):
    for name, pat in MODERATION_RULES:
        if pat.search(text):
            return True, name
    return False, None

# 混淆矩阵：正类=BLOCKED
tp = fp = fn = tn = 0
by_cat = {}
for s in samples:
    pred_block, rule = rule_block(s["text"])
    pred = "BLOCKED" if pred_block else "APPROVED"
    truth = s["label"]
    c = by_cat.setdefault(s["category"], {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0})
    c["n"] += 1
    if truth == "BLOCKED" and pred == "BLOCKED":
        tp += 1; c["tp"] += 1
    elif truth == "APPROVED" and pred == "BLOCKED":
        fp += 1; c["fp"] += 1
    elif truth == "BLOCKED" and pred == "APPROVED":
        fn += 1; c["fn"] += 1
    else:
        tn += 1; c["tn"] += 1

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f1

p, r, f1 = prf(tp, fp, fn)
print("=" * 70)
print("规则层评测（正类=BLOCKED，共 %d 条）" % len(samples))
print("=" * 70)
print(f"整体: 精确率 P={p:.3f}  召回率 R={r:.3f}  F1={f1:.3f}")
print(f"混淆矩阵: TP={tp} FP={fp} FN={fn} TN={tn}")
print("-" * 70)
print(f"{'类别':<22} {'样本':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
for cat, c in by_cat.items():
    print(f"{cat:<22} {c['n']:>4} {c['tp']:>3} {c['fp']:>3} {c['fn']:>3} {c['tn']:>3}")

print("\n逐类明细：")
for s in samples:
    pred_block, rule = rule_block(s["text"])
    pred = "BLOCKED" if pred_block else "APPROVED"
    flag = "OK" if pred == s["label"] else "MISS"
    print(f"[{flag}] {s['category']:<20} 标注={s['label']:<9} 规则={pred:<9} {rule or ''} | {s['text'][:30]}")

print("\n说明：")
print("- keyword_violation 类由规则层硬性红线拦截；")
print("- semantic_violation 类无规则词，规则层放行，属预期漏判，设计上由 LLM 层兜底；")
print("- LLM 层本轮未配置 key，未参与评测，待配置后复测规则+LLM 串联的 P/R/F1。")

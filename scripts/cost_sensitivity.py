# -*- coding: utf-8 -*-
"""P1-4 成本敏感性表：固定主口径交易序列，四档费率重算费用/净值/Sharpe。
口径声明：买卖点、股数、成交价沿用主口径 result_v2/test_fusion_3view，
费率倍数只作用于佣金（双边、最低5元）与印花税（卖出端）；滑点已计入成交价，不重复扣。
近似：费用在发生日从现金扣除，按当日权益比例下调净值序列。
"""
import csv, json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\03_Study\GraduationProject\LightQuant"
DIR = os.path.join(ROOT, "backtest", "result_v2", "test_fusion_3view")
INITIAL = 100000.0
RF = 0.02
ANN = 252

# 主口径费率
COMM = 0.0003
STAMP = 0.001
MIN_COMM = 5.0

with open(os.path.join(DIR, "summary.json"), encoding="utf-8") as f:
    summary = json.load(f)

# 口径对齐：summary 的 total_fee 是 N 个独立账户的合计费用，
# nav.csv 的 portfolio_nav 是这 N 个账户合成后的归一化净值（1.0 = 总本金）。
# 两者相加减必须用总本金，用单账户 10 万会把费用放大约 50 倍。
N_ACCOUNTS = int(summary["portfolio"]["stocks"])
TOTAL_CAPITAL = INITIAL * N_ACCOUNTS

trades = []
with open(os.path.join(DIR, "trades.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        row["shares"] = float(row["shares"])
        row["amount"] = float(row["amount"])
        trades.append(row)

nav_dates, nav_vals = [], []
with open(os.path.join(DIR, "nav.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        nav_dates.append(row["date"])
        nav_vals.append(float(row["portfolio_nav"]))
nav_vals = np.array(nav_vals)
date_to_nav = {d: i for i, d in enumerate(nav_dates)}

def fee_for(amount, side, comm_mult, stamp_mult):
    comm = max(MIN_COMM, amount * COMM * comm_mult)
    stamp = amount * STAMP * stamp_mult if side == "SELL" else 0.0
    return comm + stamp

def run(mult_comm, mult_stamp):
    # 按日汇总费用
    fee_by_day = {}
    total_fee = 0.0
    for t in trades:
        fee = fee_for(t["amount"], t["side"], mult_comm, mult_stamp)
        fee_by_day[t["date"]] = fee_by_day.get(t["date"], 0.0) + fee
        total_fee += fee
    # 主口径 1x 费用（用于还原零成本基准）
    fee1_by_day = {}
    fee1_total = 0.0
    for t in trades:
        f1 = fee_for(t["amount"], t["side"], 1.0, 1.0)
        fee1_by_day[t["date"]] = fee1_by_day.get(t["date"], 0.0) + f1
        fee1_total += f1

    # 费用为现金一次性流出：权益 = 持仓市值 + 现金，费用只减少现金绝对额，
    # 不按比例缩减后续持仓市值（避免费用被复利放大）。
    # 1) 还原零成本净值：把 1x 费用按发生日线性加回
    nav0 = nav_vals.copy()
    cum_fee1 = 0.0
    for i, d in enumerate(nav_dates):
        cum_fee1 += fee1_by_day.get(d, 0.0)
        nav0[i] = nav_vals[i] + cum_fee1 / TOTAL_CAPITAL
    # 2) 扣当前档费用（线性累计绝对额）
    nav_new = nav0.copy()
    cum_fee = 0.0
    for i, d in enumerate(nav_dates):
        cum_fee += fee_by_day.get(d, 0.0)
        nav_new[i] = nav0[i] - cum_fee / TOTAL_CAPITAL

    final_nav = nav_new[-1]
    cum = final_nav - 1.0
    rets = np.diff(nav_new) / nav_new[:-1]
    excess = rets.mean() * ANN - RF
    sharpe = excess / (rets.std() * np.sqrt(ANN)) if rets.std() > 0 else float("nan")
    mdd = float((np.maximum.accumulate(nav_new) - nav_new).max())
    return {"cum": cum, "sharpe": sharpe, "mdd": mdd,
            "trades": len(trades), "fee": total_fee}

scenarios = [
    ("0 费率（零成本）", 0.0, 0.0),
    ("1x（现状：佣金万三+印花税千一）", 1.0, 1.0),
    ("2x（佣金印花税翻倍）", 2.0, 2.0),
    ("3x（佣金印花税三倍）", 3.0, 3.0),
]

print("=" * 86)
print("成本敏感性表（固定交易序列，733 笔；滑点已计入成交价，不重复扣除）")
print("=" * 86)
print(f"{'情景':<32} {'笔数':>6} {'总费用(元)':>12} {'累计收益':>10} {'Sharpe':>9} {'最大回撤':>9}")
print("-" * 86)
for name, mc, ms in scenarios:
    r = run(mc, ms)
    print(f"{name:<32} {r['trades']:>6} {r['fee']:>12.2f} {r['cum']*100:>9.2f}% {r['sharpe']:>9.3f} {r['mdd']*100:>8.2f}%")

print("\n主口径 summary 对照: cum=%.2f%% Sharpe=%.3f 总费用=%.2f" % (
    summary["portfolio"]["cum_return"]*100,
    summary["portfolio"]["sharpe"],
    summary["portfolio"]["total_fee"]))

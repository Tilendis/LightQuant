# -*- coding: utf-8 -*-
"""P2-6 撮合时点口径差异对照。
注意：两套引擎信号来源、仓位口径、换手频率均不同，本表只并列口径与指标，
不做"哪种策略收益更高"的结论对比。
"""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\03_Study\GraduationProject\LightQuant"
TIMING = os.path.join(ROOT, "backtest", "result_v2", "test_fusion_3view_v2", "summary.json")
CROSS = os.path.join(ROOT, "backtest", "result_v2", "api_cs_new", "summary.json")

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

t, c = load(TIMING), load(CROSS)

rows = [
    ("对照维度", "择时引擎 engine_v2", "截面引擎 engine_cross_section"),
    ("撮合时点假设", "T 日收盘算分，T 日收盘集合竞价成交", "T-1 收盘算分，T 日开盘成交（剔除隔夜跳空）"),
    ("价格口径", "未复权收盘价 close", "未复权开盘价 open（信号用 T-1 收盘）"),
    ("仓位口径", "单票独立资金、等权持有", "组合口径，Top-N 等权、固定持有期再平衡"),
    ("信号来源", "三视角融合分数（fusion-test-3view-v2）", "技术面单视角截面排序"),
    ("调仓频率", "逐日信号触发，T+1 不可当日卖出", "每 5 个交易日再平衡一次"),
    ("样本区间", t["portfolio"]["first_date"] + " ~ " + t["portfolio"]["last_date"],
              c["portfolio"]["first_date"] + " ~ " + c["portfolio"]["last_date"]),
    ("累计收益", "%.2f%%" % (t["portfolio"]["cum_return"]*100),
                "%.2f%%" % (c["portfolio"]["cum_return"]*100)),
    ("Sharpe", "%.3f" % t["portfolio"]["sharpe"], "%.3f" % c["portfolio"]["sharpe"]),
    ("最大回撤", "%.2f%%" % (t["portfolio"]["mdd"]*100), "%.2f%%" % (c["portfolio"]["mdd"]*100)),
    ("成交笔数", str(t["portfolio"]["total_trades"]), str(c["portfolio"]["total_trades"])),
    ("总费用(元)", "%.2f" % t["portfolio"]["total_fee"], "%.2f" % c["portfolio"]["total_fee"]),
    ("佣金/印花税/滑点", "万三 / 千一(卖出) / 万五", "万三 / 千一(卖出) / 千二"),
]

for r in rows:
    print(f"{r[0]:<16} | {r[1]:<34} | {r[2]}")

print("\n口径说明：两表信号、仓位、换手不同，累计收益不可直接比较；")
print("对照目的是说明系统同时实现了两种撮合时点假设，择时引擎的收盘成交为乐观口径，")
print("截面引擎的 T-1→T 开盘成交为更保守的可执行口径。")

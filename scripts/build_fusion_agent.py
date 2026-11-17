# -*- coding: utf-8 -*-
"""把三个智能体视角按标定权重合成 fusion-agent-v1，写进 judgements。

权重 tech 1.0 / fund 0.0 / sent 0.5：fund 为 0 是 docs/calib_weights.json 在验证集上
标定出来的（fund_only 准确率 47.92%、mean_next_ret 为负），不是这里拍的。
缺失视角时只对有记录的视角求和，分母同步缩小。
"""
import os
import sqlite3

DB = r"D:\03_Study\GraduationProject\LightQuant\app\data\lightquant.db"
PV = {"tech": "tech-agent-v1", "fund": "fund-agent-v1", "sent": "sent-agent-v1"}
W = {"tech": 1.0, "fund": 0.0, "sent": 0.5}
TH = 0.05
OUT_PV = "fusion-agent-v1"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT symbol,date,view,signal,score FROM judgements "
    "WHERE prompt_version IN (?,?,?)", tuple(PV.values())).fetchall()

by_key = {}
for r in rows:
    by_key.setdefault((r["symbol"], r["date"]), {})[r["view"]] = (
        r["signal"], float(r["score"]))

built = []
for (sym, d), views in by_key.items():
    num = den = 0.0
    parts = []
    notes = []
    for name in ("tech", "fund", "sent"):
        if name not in views:
            continue
        sig, sc = views[name]
        w = W[name]
        num += w * sc
        den += w
        if w > 0:
            parts.append("%s %+.4f×%.1f[%s]" % (name, sc, w, sig))
            # 融合层只读连续分数、不读单视角自评的 signal。当某个视角自己写 HOLD、
            # 但分数已经越过阈值时，最终方向会和它自评的不一致，必须在 reason 里点明，
            # 否则界面上一行写 HOLD、下一行写 SELL，看的人只会觉得系统自相矛盾。
            if sig == "HOLD" and abs(sc) > TH:
                notes.append("%s 自评 HOLD 但分数 %+.4f 越过 ±%.2f，由融合层按分数定方向"
                             % (name, sc, TH))
        else:
            parts.append("%s %+.4f（权重0，不参与）" % (name, sc))
    if den <= 0:
        continue
    fs = num / den
    signal = "BUY" if fs > TH else ("SELL" if fs < -TH else "HOLD")
    reason = "加权 " + " + ".join(parts) + " = %+.4f → %s" % (fs, signal)
    if notes:
        reason += "；" + "；".join(notes)

    built.append((sym, d, "fusion", signal, round(fs, 4), reason,
                  "local", "weighted-agent-v1", 0.0, OUT_PV, "offline_rule"))

conn.executemany(
    "INSERT OR REPLACE INTO judgements "
    "(symbol,date,view,signal,score,reason,provider,model,temperature,"
    "prompt_version,source,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
    built)
conn.commit()

n = conn.execute(
    "SELECT COUNT(*) FROM judgements WHERE prompt_version=?", (OUT_PV,)).fetchone()[0]
dist = conn.execute(
    "SELECT signal, COUNT(*) FROM judgements WHERE prompt_version=? GROUP BY signal",
    (OUT_PV,)).fetchall()
print("写入 %s：%d 条" % (OUT_PV, len(built)))
print("库内现有 %d 条" % n)
print("分布: " + ", ".join("%s=%d" % (r[0], r[1]) for r in dist))
print("日期范围: " + str(conn.execute(
    "SELECT MIN(date), MAX(date) FROM judgements WHERE prompt_version=?",
    (OUT_PV,)).fetchone()[:]))
conn.close()

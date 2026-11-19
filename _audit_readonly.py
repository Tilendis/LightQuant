import sqlite3, os, re
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "data", "lightquant.db")
c = sqlite3.connect(DB); c.row_factory = sqlite3.Row
LO, HI = "2024-12-31", "2024-12-31"

def get_view(view, pv, sym, date, sent_ffill=False):
    if view == "sent" and sent_ffill:
        r = c.execute("""SELECT signal,score FROM judgements WHERE view='sent' AND
            symbol=? AND date<=? AND date>=date(?,'-5 days') AND prompt_version=?
            ORDER BY date DESC LIMIT 1""", (sym, date, date, pv)).fetchone()
    else:
        r = c.execute("""SELECT signal,score FROM judgements WHERE view=? AND
            symbol=? AND date=? AND prompt_version=?""", (view, sym, date, pv)).fetchone()
    return dict(r) if r else None

def decide(views):
    if "tech" not in views: return "no_tech"
    sigs = {v["signal"] for v in views.values()}
    if len(sigs) == 1: return "unanimous"
    if "BUY" in sigs and "SELL" in sigs: return "conflict"
    return "minor"

rows = c.execute("""SELECT symbol,date,signal,reason FROM judgements
    WHERE view='fusion' AND prompt_version='fusion-orch-v1'""").fetchall()
print(f"orch-v1 共 {len(rows)} 格")
mismatch_branch = mismatch_sig = 0
for r in rows:
    sym, date = r["symbol"], r["date"]
    views = {}
    t = get_view("tech", "tech-v1", sym, date);  views["tech"] = t
    f = get_view("fund", "fund-rule-v1", sym, date); views["fund"] = f
    s = get_view("sent", "sent-v1", sym, date, True);  views["sent"] = s
    views = {k: v for k, v in views.items() if v}
    branch = decide(views)
    m = re.search(r"\[(\w+)\]", r["reason"] or "")
    reason_tag = m.group(1) if m else "?"
    tech_sig = views["tech"]["signal"] if "tech" in views else "NA"
    if branch != reason_tag: mismatch_branch += 1
    # R2 技术优先：规则版方向通常跟随 tech
    if tech_sig != r["signal"] and r["signal"] not in ("HOLD", "REVIEW"): mismatch_sig += 1
    if branch != reason_tag:
        print(f"  分支不符 {date} {sym}: 复现={branch} 记录={reason_tag} 记录sig={r['signal']} tech={tech_sig}")

print(f"\n分支标签不一致: {mismatch_branch}/{len(rows)}")
print(f"方向与 tech 不一致(非HOLD/REVIEW): {mismatch_sig}/{len(rows)}")

# tech-llm-v1 归属与覆盖
print("\n=== tech-llm-v1 12 月段 signal 分布 ===")
for r in c.execute("""SELECT signal, COUNT(*) n FROM judgements WHERE view='tech'
    AND prompt_version='tech-llm-v1' AND date BETWEEN '2024-12-02' AND '2024-12-31'
    GROUP BY signal"""):
    print(f"  {r['signal']}: {r['n']}")
c.close()

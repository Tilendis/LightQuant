"""多视角准确率对比：给后台折线图准备数据。

口径必须提前定死，否则答辩会被追问"你这个准确率怎么算的"：
- 方向判定：score >= 0 记为"看涨"，label=1 表示次日实际上涨
- 全体口径：所有有研判的 (股票, 日期) 都计入
- 强信号口径：只看 |score| >= 0.2 的样本，反映"模型敢下判断时"的命中率

注意：基本面是月频慢变量、情绪面覆盖稀疏，两者的准确率天然不会高。
这个图的价值不是证明"LLM 很准"，而是把三路视角的差异摆出来，
并如实标注"测试期在训练语料覆盖区间内，存在污染风险"。

用法：
    python scripts/eval_views.py --start 2024-08-08 --end 2024-12-31
"""

import argparse
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-08-08")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default=os.path.join(ROOT, "app", "data", "view_accuracy.json"))
    ap.add_argument("--fusion_version", default="fusion-test-3view-v2",
                    help="融合表版本。默认 test 段三方标定 0.90/0.00/0.10；"
                         "对照旧口径可传 fusion-v1 / fusion-val-v1 / fusion-vote-v1")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    conn = db.connect(args.db)

    labels = pd.read_sql_query(
        "SELECT symbol, date, label FROM prices WHERE date BETWEEN ? AND ? AND label IS NOT NULL",
        conn, params=(args.start, args.end))
    print("有标签样本:", len(labels))

    # 必须按版本过滤：fusion 在表里有多版共存，不过滤会把多版样本叠在一起，
    # 同一 (股票,日期) 被算两次，准确率失去意义。
    specs = [("tech", "tech-v1"), ("fund", "fund-rule-v1"),
             ("sent", "sent-v1"), ("fusion", args.fusion_version)]
    daily, summary = {}, {}
    for v, pv in specs:
        j = pd.read_sql_query(
            """SELECT symbol, date, score, signal FROM judgements
               WHERE view=? AND prompt_version=? AND date BETWEEN ? AND ?""",
            conn, params=(v, pv, args.start, args.end))
        if j.empty:
            print("%-6s 无数据" % v)
            continue
        m = j.merge(labels, on=["symbol", "date"], how="inner")
        if m.empty:
            print("%-6s 无对齐样本" % v)
            continue
        m["pred"] = (m["score"] >= 0).astype(int)
        m["hit"] = (m["pred"] == m["label"]).astype(int)
        m["strong"] = m["score"].abs() >= 0.2

        acc_all = float(m["hit"].mean())
        n_strong = int(m["strong"].sum())
        acc_strong = float(m.loc[m["strong"], "hit"].mean()) if n_strong else None

        by_day = m.groupby("date")["hit"].mean().round(4)
        daily[v] = {d: float(x) for d, x in by_day.items()}
        summary[v] = dict(
            acc_all=round(acc_all, 4), n_all=int(len(m)),
            acc_strong=None if acc_strong is None else round(acc_strong, 4),
            n_strong=n_strong,
            cover_symbols=int(m["symbol"].nunique()),
            cover_dates=int(m["date"].nunique()),
            mean_score=round(float(m["score"].mean()), 3),
            prompt_version=pv,
        )
        print("%-6s ACC=%.4f (n=%d)  强信号ACC=%s (n=%d)  覆盖%d只/%d日  均分%+.3f" % (
            v, acc_all, len(m),
            "%.4f" % acc_strong if acc_strong is not None else "NA", n_strong,
            summary[v]["cover_symbols"], summary[v]["cover_dates"], summary[v]["mean_score"]))

    conn.close()
    payload = dict(
        meta=dict(start=args.start, end=args.end,
                  note="测试期在训练语料覆盖区间内，存在语料污染风险；"
                       "该指标用于横向比较各视角，不作为有效性证据。"
                       "融合权重由验证集(2024-03-15~08-07)三方网格搜索确定，"
                       "最终取 tech 0.90 / fund 0.00 / sent 0.10："
                       "基本面权重从 0 增到 1 时验证集准确率单调下降 5.4pp，"
                       "情绪面在给定技术面主导下提供 0.2pp 的边际改善"
                       "（标准误 0.71pp，不构成显著证据）。"
                       "详见 docs/calib_weights_3view.json",
                  fusion_version=args.fusion_version),
        summary=summary, daily=daily,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("\n写入", args.out)


if __name__ == "__main__":
    main()

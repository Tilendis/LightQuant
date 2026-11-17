"""在验证集上标定融合权重（tech : fund），只产出证据，不写数据库。

背景
----
现有 fusion 权重 0.5 : 0.25 : 0.25 是在测试期（2024-08-08 ~ 12-31）上拍定的，
等价于用测试集调参 —— 论文里这条过不去。本脚本改用验证集
（2024-03-15 ~ 2024-08-07，切分来自 utils/util.py: train<=03-14 / val<=08-07）
重新标定 tech 与 fund 的相对权重。

原始版本只标定 tech:fund，理由是"验证期无新闻语料"。该判断已被推翻：
`dataset/CSMD50_feat/val` 下确实没有 news —— 但新闻根本不在这个目录，它统一放在
`dataset/CSMD50/news`（2021-01-01 ~ 2024-12-31，50 只标的全覆盖）。
本轮已把 val 段新闻灌库并用 `gen_sentiment.py` 生成 `sent-v1` 1274 条，
因此三方（tech / fund / sent）都可以在验证集上评估。

注意：`--out` 默认仍指向 `docs/calib_weights.json`，但那是一版只含 tech:fund 的历史证据。
做三方标定时请显式指定 `--out docs/calib_weights_3view.json`，不要覆盖旧证据。

复现的算法与 scripts/fuse_views.py 一致：
    - tech 不做截面去均值（概率是绝对值）
    - fund 做按日截面去均值（去掉估值视角自身的乐观/悲观底噪）
    - sent 先按"交易日 ∪ 新闻日"建索引再 ffill，然后同样按日截面去均值
    - fscore = Σ w_i * score_i / Σ w_i，只对可用视角归一化
回测口径 --score_th 默认 0.0，即 fscore > 0 才买，所以方向判定用 0 作阈值。

用法：
    python scripts/calibrate_weights.py
    python scripts/calibrate_weights.py --step 0.02 --out docs/calib_weights.json
"""

import argparse
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402
from backtest.engine_v2 import build_model  # noqa: E402
from utils.price_dataloader import Backtest_Dataset  # noqa: E402

DEFAULT_COLS = ["Open", "High", "Low", "Close", "Adj Close",
                "Return", "RSI6", "MACD_Hist", "BIAS5", "Amplitude"]
VAL_PRICE = os.path.join(ROOT, "dataset", "CSMD50_feat", "val", "price")
MODEL_FOLDER = os.path.join(ROOT, "result", "CSMD50_w5_feat_bi")


def _load_fund_helpers():
    """复用 gen_fundamental.py 的分位与打分逻辑，避免两套口径漂移。"""
    p = os.path.join(ROOT, "scripts", "gen_fundamental.py")
    spec = importlib.util.spec_from_file_location("gen_fundamental", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tech_scores(model, price_folder):
    """逐股推理得 tech 分数；date 与 prob 一一对应（前 look_back 行被窗口吃掉）。"""
    rows = []
    for fn in sorted(f for f in os.listdir(price_folder) if f.endswith(".csv")):
        sym = fn[:-4]
        path = os.path.join(price_folder, fn)
        ds = Backtest_Dataset(csv_file=path, look_back_window=5,
                              feature_columns=DEFAULT_COLS)
        dates = [str(d) for d in ds.get_dates()]
        x = torch.tensor(np.asarray(ds.normalized_sequences), dtype=torch.float32)
        with torch.no_grad():
            probs = model(x).squeeze(-1).cpu().numpy()
        if probs.min() < -1e-6 or probs.max() > 1 + 1e-6:
            raise RuntimeError("模型输出不在 [0,1]：%s" % path)
        for d, p in zip(dates, probs):
            rows.append((sym, d, max(-1.0, min(1.0, 2.0 * float(p) - 1.0))))
    return pd.DataFrame(rows, columns=["symbol", "date", "tech"])


def val_truth(price_folder):
    """验证期真实标签与次日收益（Label 是 Close.shift(-1) 的方向）。"""
    rows = []
    for fn in sorted(f for f in os.listdir(price_folder) if f.endswith(".csv")):
        sym = fn[:-4]
        df = pd.read_csv(os.path.join(price_folder, fn))
        df["Date"] = df["Date"].astype(str)
        df["next_ret"] = df["Close"].shift(-1) / df["Close"] - 1.0
        for _, r in df.iterrows():
            rows.append((sym, r["Date"], int(r["Label"]) if pd.notna(r["Label"]) else None,
                         float(r["next_ret"]) if pd.notna(r["next_ret"]) else np.nan))
    return pd.DataFrame(rows, columns=["symbol", "date", "label", "next_ret"])


def fund_scores(conn, gf, dates):
    """估值分位规则分，按日截面去均值（与 fuse_views.py 的 fund 处理一致）。"""
    symbols = [r["symbol"] for r in conn.execute(
        "SELECT DISTINCT symbol FROM valuations ORDER BY symbol").fetchall()]
    rows = []
    for sym in symbols:
        v = gf.load_valuations(conn, sym)
        if v.empty:
            continue
        v["date"] = v["date"].astype(str)
        for d in dates:
            hist = v[v["date"] <= d]
            if hist.empty:
                continue
            cur = hist.iloc[-1]
            pe_pct = gf.pct_rank(hist["pe_ttm"], cur["pe_ttm"])
            pb_pct = gf.pct_rank(hist["pb"], cur["pb"])
            sc = gf.fund_score(pe_pct, pb_pct)
            if sc is None:
                continue
            rows.append((sym, d, sc))
    df = pd.DataFrame(rows, columns=["symbol", "date", "fund_raw"])
    if df.empty:
        return df
    df["fund"] = df.groupby("date")["fund_raw"].transform(lambda s: s - s.mean())
    return df[["symbol", "date", "fund"]]


def sent_scores(conn, dates, symbols):
    """情绪面分数：先按"交易日 ∪ 新闻日"ffill，再按日截面去均值（与 fuse_views.py 一致）。"""
    rows = conn.execute(
        "SELECT symbol, date, score FROM judgements "
        "WHERE view='sent' AND prompt_version='sent-v1' AND date BETWEEN ? AND ?",
        (dates[0], dates[-1])).fetchall()
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "sent"])
    df = pd.DataFrame(
        [(r["symbol"], str(r["date"]), float(r["score"])) for r in rows],
        columns=["symbol", "date", "sent_raw"])
    m = df.pivot_table(index="date", columns="symbol", values="sent_raw", aggfunc="last")
    full_idx = m.index.union(pd.Index(dates)).sort_values()
    m = m.reindex(index=full_idx).ffill().reindex(index=dates, columns=symbols)
    m = m.sub(m.mean(axis=1), axis=0)
    out = m.stack().reset_index()
    out.columns = ["date", "symbol", "sent"]
    return out


def build_panel(conn, gf):
    dates = sorted(pd.read_csv(
        os.path.join(VAL_PRICE, sorted(os.listdir(VAL_PRICE))[0]))["Date"].astype(str).tolist())
    truth = val_truth(VAL_PRICE)
    fund = fund_scores(conn, gf, dates)

    ns = SimpleNamespace(model="bi_lstm", model_save_folder=MODEL_FOLDER,
                         input_size=len(DEFAULT_COLS), hidden_size=128, layers=2,
                         attention_size=128, dropout=0.1, epsilon=0.1,
                         perturbation_size=0.1)
    tech = tech_scores(build_model(ns), VAL_PRICE)
    sent = sent_scores(conn, dates, sorted(tech["symbol"].unique()))

    panel = truth.merge(tech, on=["symbol", "date"], how="left")
    panel = panel.merge(fund, on=["symbol", "date"], how="left")
    panel = panel.merge(sent, on=["symbol", "date"], how="left")
    return panel, len(dates)


def evaluate(panel, w_tech, w_fund, w_sent=0.0):
    """按给定权重算 fscore，返回方向准确率与组合收益。"""
    d = panel.dropna(subset=["label"]).copy()
    t, f = d["tech"], d["fund"]
    s = d["sent"] if "sent" in d.columns else pd.Series(np.nan, index=d.index)

    tw = np.where(t.notna(), w_tech, 0.0)
    fw = np.where(f.notna(), w_fund, 0.0)
    sw = np.where(s.notna(), w_sent, 0.0)
    denom = tw + fw + sw
    num = (np.where(t.notna(), tw * t.fillna(0.0), 0.0)
           + np.where(f.notna(), fw * f.fillna(0.0), 0.0)
           + np.where(s.notna(), sw * s.fillna(0.0), 0.0))

    valid = denom > 0
    fs = np.where(valid, num / np.where(valid, denom, 1.0), 0.0)
    d = d[valid].copy()
    d["fscore"] = fs[valid]
    d["pred"] = (d["fscore"] > 0).astype(int)
    d["hit"] = (d["pred"] == d["label"]).astype(int)

    per_stock = d.groupby("symbol")["hit"].mean()
    sel = d[d["pred"] == 1]
    sel_ret = sel.dropna(subset=["next_ret"])
    daily_ret = sel_ret.groupby("date")["next_ret"].mean()

    return {
        "n": int(len(d)),
        "n_stocks": int(per_stock.size),
        "acc_all": float(d["hit"].mean()),
        "acc_per_stock_mean": float(per_stock.mean()),
        "buy_ratio": float(d["pred"].mean()),
        "precision_selected": float(sel["hit"].mean()) if len(sel) else None,
        "mean_next_ret_selected": float(sel_ret["next_ret"].mean()) if len(sel_ret) else None,
        "daily_ret_mean": float(daily_ret.mean()) if len(daily_ret) else None,
        "daily_ret_positive_ratio": float((daily_ret > 0).mean()) if len(daily_ret) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "calib_weights.json"))
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    conn = db.connect(args.db)
    gf = _load_fund_helpers()
    panel, n_dates = build_panel(conn, gf)
    conn.close()
    has_sent = "sent" in panel.columns and panel["sent"].notna().any()

    cov = {
        "val_dates": n_dates,
        "rows": int(len(panel)),
        "tech_cover": int(panel["tech"].notna().sum()),
        "fund_cover": int(panel["fund"].notna().sum()),
        "sent_cover": int(panel["sent"].notna().sum()) if has_sent else 0,
        "both_cover": int((panel["tech"].notna() & panel["fund"].notna()).sum()),
    }
    print("验证期覆盖:", cov)

    # 单视角基线
    base = {
        "tech_only": evaluate(panel, 1.0, 0.0, 0.0),
        "fund_only": evaluate(panel, 0.0, 1.0, 0.0),
        "sent_only": evaluate(panel, 0.0, 0.0, 1.0) if has_sent else None,
    }

    steps = int(round(1.0 / args.step))

    # 网格一：tech:fund（原口径，用于和历史结论对照）
    grid_tf = []
    for i in range(steps + 1):
        wt = round(i * args.step, 6)
        wf = round(1.0 - wt, 6)
        res = evaluate(panel, wt, wf, 0.0)
        res["w_tech"] = wt
        res["w_fund"] = wf
        grid_tf.append(res)
    best_tf = max(grid_tf, key=lambda r: r["acc_per_stock_mean"])

    # 网格二：tech:sent（fund 固定 0），回答"情绪面到底该不该给权重"
    grid_ts = []
    if has_sent:
        for i in range(steps + 1):
            wt = round(i * args.step, 6)
            ws = round(1.0 - wt, 6)
            res = evaluate(panel, wt, 0.0, ws)
            res["w_tech"] = wt
            res["w_sent"] = ws
            grid_ts.append(res)
    best_ts = max(grid_ts, key=lambda r: r["acc_per_stock_mean"]) if grid_ts else None

    # 线上权重 0.5:0.25:0.25 在验证集的表现（三方都参与）
    legacy = evaluate(panel, 0.5, 0.25, 0.25)

    payload = {
        "meta": {
            "interval": "validation only (2024-03-15 ~ 2024-08-07)",
            "note": ("三方标定。sent 覆盖 %d 行（共 %d 个标的-日），"
                     "未覆盖的格子按可用视角归一化。"
                     % (cov["sent_cover"], cov["rows"])),
            "label": "Label = Close(t+1) > Close(t)，逐股先算再平均",
            "threshold": "fscore > 0 判买（对应回测 --score_th 0.0）",
            "fund": "估值分位规则分，按日截面去均值；tech 不去均值",
            "sent": "sent-v1，先按交易日∪新闻日 ffill，再按日截面去均值",
            "grid_tech_fund": "tech:fund 两维，sent 权重固定 0（历史口径）",
            "grid_tech_sent": "tech:sent 两维，fund 权重固定 0",
        },
        "coverage": cov,
        "baselines": base,
        "legacy_0_5_0_25_0_25_on_val": legacy,
        "best": best_ts or best_tf,
        "best_tech_fund": best_tf,
        "best_tech_sent": best_ts,
        "grid_tech_fund": grid_tf,
        "grid_tech_sent": grid_ts,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n单视角：tech acc=%.4f  fund acc=%.4f  sent acc=%s" % (
        base["tech_only"]["acc_per_stock_mean"],
        base["fund_only"]["acc_per_stock_mean"],
        ("%.4f" % base["sent_only"]["acc_per_stock_mean"]) if base["sent_only"] else "NA"))
    print("线上 0.5/0.25/0.25：acc=%.4f  buy_ratio=%.3f" % (
        legacy["acc_per_stock_mean"], legacy["buy_ratio"]))
    print("tech:fund 最优 w_tech=%.2f w_fund=%.2f acc=%.4f" % (
        best_tf["w_tech"], best_tf["w_fund"], best_tf["acc_per_stock_mean"]))
    if best_ts:
        print("tech:sent 最优 w_tech=%.2f w_sent=%.2f acc=%.4f buy_ratio=%.3f" % (
            best_ts["w_tech"], best_ts["w_sent"], best_ts["acc_per_stock_mean"],
            best_ts["buy_ratio"]))
    print("\n已写出:", args.out)


if __name__ == "__main__":
    main()

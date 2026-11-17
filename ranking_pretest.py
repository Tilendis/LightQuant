"""
排序 pre-test：不重训模型，直接用已训练好的 ALSTM 概率做同日截面排序，
看"排序质量"（Rank IC / Top-5 命中率）有没有信号。

用途：决定 Phase 1（截面排序重构）值不值得投入一周。
不产生收益结论，只评估排序质量。

用法：
  python ranking_pretest.py --split test --look_back_window 5 \
      --ckpt ./result/CSMD50_w5/model_saved/alstm.pth \
      --out ./result/CSMD50_w5/ranking_pretest.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from model.ALSTM import ALSTM
from utils.price_dataloader import Normal_Dataset, Backtest_Dataset


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="CSMD50")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--look_back_window", default=5, type=int)
    p.add_argument("--ckpt", default="./result/CSMD50_w5/model_saved/alstm.pth")
    p.add_argument("--topk", default=5, type=int)
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--out", default="./result/CSMD50_w5/ranking_pretest.json")
    return p.parse_args()


def build_model(ckpt, input_size=5, hidden=128, layers=2, attention=128, dropout=0.1):
    model = ALSTM(input_size=input_size, hidden_size=hidden, num_layers=layers,
                  output_size=1, dropout=dropout, batch_first=True, attention_size=attention)
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    args = parse()
    folder = os.path.join("./dataset", args.dataset, args.split, "price")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
    files.sort()
    print(f"[info] {len(files)} 个股票文件, split={args.split}, look_back={args.look_back_window}")

    ds = Normal_Dataset(files, args.look_back_window)
    features = ds.features.float()
    print(f"[info] dataset shape = {tuple(features.shape)}")

    # ---- 构造 (股票, 日期, 次日收益) 索引，顺序必须与 Normal_Dataset 完全一致 ----
    meta_stock, meta_date, meta_ret = [], [], []
    for f in files:
        bd = Backtest_Dataset(f, look_back_window=args.look_back_window,
                              feature_columns=['Open', 'High', 'Low', 'Close', 'Adj Close'],
                              label_column='Label')
        n = len(bd)
        df = pd.read_csv(f)
        close = df["Close"].values
        lb = args.look_back_window
        # 第 i 个窗口覆盖 [i, i+lb-1]，预测的是 i+lb-1 -> i+lb 的涨跌
        rets = []
        for i in range(n):
            j = i + lb - 1
            rets.append(close[j + 1] / close[j] - 1 if j + 1 < len(close) else np.nan)
        stock = os.path.splitext(os.path.basename(f))[0]
        meta_stock.extend([stock] * n)
        meta_date.extend(list(bd.dates))
        meta_ret.extend(rets)

    # ---- 索引一致性校验：样本数 + 特征值 ----
    if len(meta_ret) != features.shape[0]:
        raise RuntimeError(f"索引长度 {len(meta_ret)} != 数据集长度 {features.shape[0]}，顺序可能不一致")
    lb = args.look_back_window
    chk = Backtest_Dataset(files[0], look_back_window=lb,
                           feature_columns=['Open', 'High', 'Low', 'Close', 'Adj Close'],
                           label_column='Label')
    ref = np.stack(chk.normalized_sequences)
    diff = np.abs(ref - features[:len(ref)].numpy()).max()
    print(f"[check] 首文件特征逐值比对 max|diff| = {diff:.6g}")
    if diff > 1e-4:
        raise RuntimeError("Normal_Dataset 与 Backtest_Dataset 的预处理不一致，索引不可信")

    # ---- 前向推理 ----
    model = build_model(args.ckpt)
    probs = []
    with torch.no_grad():
        for i in range(0, features.shape[0], args.batch_size):
            x = features[i:i + args.batch_size]
            probs.append(model(x).squeeze().numpy())
    probs = np.concatenate(probs)

    frame = pd.DataFrame({"stock": meta_stock, "date": meta_date, "prob": probs, "ret": meta_ret})
    frame = frame.dropna(subset=["ret"]).reset_index(drop=True)
    print(f"[info] 可用于排序的样本 {len(frame)}，覆盖 {frame['date'].nunique()} 个交易日")

    # ---- 逐日截面指标 ----
    ic_list, prec_list, up_list, ret5_list, retall_list, n_list = [], [], [], [], [], []
    for d, g in frame.groupby("date"):
        if len(g) < 20:
            continue
        ret = g["ret"].values
        prob = g["prob"].values
        rho, _ = spearmanr(prob, ret)
        if np.isnan(rho):
            continue
        ic_list.append(float(rho))

        k = min(args.topk, len(g))
        top_pred = np.argsort(-prob)[:k]
        top_real = np.argsort(-ret)[:k]
        prec_list.append(len(set(top_pred.tolist()) & set(top_real.tolist())) / k)
        up_list.append(float((ret[top_pred] > 0).mean()))
        ret5_list.append(float(ret[top_pred].mean()))
        retall_list.append(float(ret.mean()))
        n_list.append(len(g))

    ic = np.array(ic_list)
    summary = {
        "config": {
            "split": args.split,
            "look_back_window": args.look_back_window,
            "ckpt": args.ckpt.replace("\\", "/"),
            "topk": args.topk,
            "n_samples": int(len(frame)),
            "n_days": int(len(ic)),
            "avg_stocks_per_day": float(np.mean(n_list)),
        },
        "ranking_quality": {
            "rank_ic_mean": float(ic.mean()),
            "rank_ic_std": float(ic.std(ddof=1)),
            "rank_icir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else None,
            "ic_positive_rate": float((ic > 0).mean()),
            f"top{args.topk}_precision_vs_realized": float(np.mean(prec_list)),
            f"top{args.topk}_random_precision": float(args.topk / np.mean(n_list)),
            f"top{args.topk}_up_rate": float(np.mean(up_list)),
            "all_stocks_up_rate": float((frame["ret"] > 0).mean()),
            f"top{args.topk}_mean_next_ret": float(np.mean(ret5_list)),
            "all_stocks_mean_next_ret": float(np.mean(retall_list)),
            f"top{args.topk}_excess_vs_all": float(np.mean(np.array(ret5_list) - np.array(retall_list))),
        },
        "note": "本脚本只评估排序质量，不含交易成本与仓位管理，不能用于收益结论。",
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

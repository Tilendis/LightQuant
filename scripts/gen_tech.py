"""技术面视角：加载本地模型逐股推理 -> 落 judgements(view='tech')

用哪个模型：BiLSTM + 十维特征（测试期 ACC 54.82% / MCC 0.0940，是所有已复现里最强的）。
分数定义：score = 2 * p - 1，把概率映射到 [-1, 1]，与 fund / sent 同尺度，
这样融合加权时三路可比，不会出现"技术面天然占主导（因为概率在 0-1）"的偏差。

注意：模型 forward 末尾已有 Sigmoid，这里绝不能再套一次。加区间校验，越界直接报错。

用法：
    python scripts/gen_tech.py --start 2024-08-08 --end 2024-12-31
"""

import argparse
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

PROMPT_VERSION = "tech-v1"
DEFAULT_COLS = ["Open", "High", "Low", "Close", "Adj Close",
                "Return", "RSI6", "MACD_Hist", "BIAS5", "Amplitude"]
DEFAULT_FOLDER = os.path.join(ROOT, "result", "CSMD50_w5_feat_bi")
DEFAULT_PRICE = os.path.join(ROOT, "dataset", "CSMD50_feat", "test", "price")


def build(args):
    ns = SimpleNamespace(
        model="bi_lstm", model_save_folder=args.model_folder,
        input_size=len(DEFAULT_COLS), hidden_size=128, layers=2,
        attention_size=128, dropout=0.1, epsilon=0.1, perturbation_size=0.1,
    )
    return build_model(ns)


def score_to_signal(score, th=0.2):
    if score >= th:
        return "BUY"
    if score <= -th:
        return "SELL"
    return "HOLD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-08-08")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--model_folder", default=DEFAULT_FOLDER)
    ap.add_argument("--price_folder", default=DEFAULT_PRICE)
    ap.add_argument("--look_back_window", type=int, default=5)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    model = build(args)
    conn = db.connect(args.db)
    db.init_db(conn=conn)

    files = sorted(f for f in os.listdir(args.price_folder) if f.endswith(".csv"))
    print("标的数:", len(files))

    written, prob_min, prob_max = 0, 2.0, -1.0
    for f in files:
        symbol = f.replace(".csv", "")
        path = os.path.join(args.price_folder, f)
        # 前视泄漏修复：scaler 在 train 段拟合
        train_path = path.replace(os.path.join("test", "price"),
                                  os.path.join("train", "price"))
        ds = Backtest_Dataset(csv_file=path, look_back_window=args.look_back_window,
                              feature_columns=DEFAULT_COLS,
                              scaler_csv=train_path if os.path.exists(train_path) else None)
        dates = [str(d) for d in ds.get_dates()]
        x = torch.tensor(np.asarray(ds.normalized_sequences), dtype=torch.float32)
        with torch.no_grad():
            probs = model(x).squeeze(-1).cpu().numpy()

        # forward 已含 Sigmoid；越界说明模型/权重不对，直接失败而不是静默写错数据
        if probs.min() < -1e-6 or probs.max() > 1 + 1e-6:
            raise RuntimeError("模型输出不在 [0,1]：%s" % path)

        prob_min = min(prob_min, float(probs.min()))
        prob_max = max(prob_max, float(probs.max()))

        for i, (d, p) in enumerate(zip(dates, probs)):
            if not (args.start <= d <= args.end):
                continue
            score = max(-1.0, min(1.0, 2.0 * float(p) - 1.0))
            # 研判理由用模板把"概率 + 当日技术指标"组织成人能读的句子。
            # 注意：这是模板拼接，不是大模型生成，model 字段保持 bi_lstm_feat10 不变。
            try:
                last = ds.raw_sequences[i][-1]
                rsi, macd, bias = float(last[6]), float(last[7]), float(last[8])
                feat_txt = "，近 5 日 RSI6 %.1f、MACD柱 %+.4f、BIAS5 %+.2f" % (rsi, macd, bias)
            except Exception:
                feat_txt = ""
            if p >= 0.55:
                verdict = "偏多"
            elif p <= 0.45:
                verdict = "偏空"
            else:
                verdict = "中性"
            reason = "BiLSTM 估计次日上涨概率 %.1f%%（阈值 50%%），判为%s%s" % (
                p * 100, verdict, feat_txt)
            db.upsert_judgement(
                conn, symbol, d, "tech", score_to_signal(score), score,
                reason=reason, provider="local",
                model="bi_lstm_feat10", temperature=None,
                prompt_version=PROMPT_VERSION, source="live")
            written += 1

    conn.close()
    print("写入 %d 条 tech 研判" % written)
    print("概率区间 [%.4f, %.4f]" % (prob_min, prob_max))
    if prob_max < 0.5:
        print("!! 警告：所有概率都在 0.5 以下，模型可能未正确加载")


if __name__ == "__main__":
    main()

"""可信回测 v2 的运行入口。

用法示例：
  python backtest/run_backtest_v2.py --model alstm \
      --model_save_folder result/CSMD50_w5_base --input_size 5 \
      --tag base_alstm_w5

  # 一次跑四组口径（5 维 / 10 维 × ALSTM / BiLSTM）
  python backtest/run_backtest_v2.py --matrix

参数默认值取 A 股个人投资者的常见档位（可覆盖）：佣金万三、最低 5 元、
卖出印花税千一、双边滑点万五、仓位 95%、T+1。
"""

import os
import sys
import argparse
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

import torch  # noqa: E402
from engine_v2 import build_model, run_backtest  # noqa: E402

DEFAULT_5 = ["Open", "High", "Low", "Close", "Adj Close"]
DEFAULT_10 = DEFAULT_5 + ["Return", "RSI6", "MACD_Hist", "BIAS5", "Amplitude"]

MATRIX = [
    dict(tag="base_alstm_w5", model="alstm", folder="result/CSMD50_w5_base",
         cols=DEFAULT_5, size=5, price="dataset/CSMD50/test/price"),
    dict(tag="base_bilstm_w5", model="bi_lstm", folder="result/CSMD50_w5_nols_bi_lstm",
         cols=DEFAULT_5, size=5, price="dataset/CSMD50/test/price"),
    dict(tag="feat_alstm_w5", model="alstm", folder="result/CSMD50_w5_feat",
         cols=DEFAULT_10, size=10, price="dataset/CSMD50_feat/test/price"),
    dict(tag="feat_bilstm_w5", model="bi_lstm", folder="result/CSMD50_w5_feat_bi",
         cols=DEFAULT_10, size=10, price="dataset/CSMD50_feat/test/price"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="alstm")
    p.add_argument("--model_save_folder", default="result/CSMD50_w5_base")
    p.add_argument("--test_price_folder", default="dataset/CSMD50/test/price")
    p.add_argument("--input_size", type=int, default=5)
    p.add_argument("--feature_columns", default="")
    p.add_argument("--look_back_window", type=int, default=5)
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--attention_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--perturbation_size", type=float, default=0.1)

    # 交易设定
    p.add_argument("--initial_cash", type=float, default=100000.0)
    p.add_argument("--position_pct", type=float, default=0.95)
    p.add_argument("--lot_size", type=int, default=1)
    p.add_argument("--commission_rate", type=float, default=0.0003)
    p.add_argument("--min_commission", type=float, default=5.0)
    p.add_argument("--stamp_duty_rate", type=float, default=0.001)
    p.add_argument("--slippage_rate", type=float, default=0.0005)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--rf_annual", type=float, default=0.02)
    p.add_argument("--take_profit_pct", type=float, default=0.0,
                   help="止盈阈值（相对买入价），0 表示关闭")
    p.add_argument("--stop_loss_pct", type=float, default=0.0,
                   help="止损阈值（相对买入价），0 表示关闭")

    p.add_argument("--tag", default="")
    p.add_argument("--out_root", default=os.path.join("backtest", "result_v2"))
    p.add_argument("--matrix", action="store_true")
    return p.parse_args()


def resolve_cols(args):
    if args.feature_columns:
        cols = [c.strip() for c in args.feature_columns.split(",") if c.strip()]
        if len(cols) != args.input_size:
            raise ValueError(f"feature_columns 有 {len(cols)} 列，但 input_size={args.input_size}")
        return cols
    return DEFAULT_5 if args.input_size == 5 else DEFAULT_10


def make_cfg(args):
    return {
        "initial_cash": args.initial_cash,
        "position_pct": args.position_pct,
        "lot_size": args.lot_size,
        "commission_rate": args.commission_rate,
        "min_commission": args.min_commission,
        "stamp_duty_rate": args.stamp_duty_rate,
        "slippage_rate": args.slippage_rate,
        "threshold": args.threshold,
        "rf_annual": args.rf_annual,
        "take_profit_pct": args.take_profit_pct,
        "stop_loss_pct": args.stop_loss_pct,
    }


def run_one(args, tag, folder, cols, size, model_name, price_folder=None):
    args.model_save_folder = os.path.join(PROJECT_ROOT, folder)
    args.model = model_name
    args.input_size = size
    args.feature_columns = ",".join(cols)
    args.tag = tag
    if price_folder:
        args.test_price_folder = price_folder

    model = build_model(args)
    out_dir = os.path.join(PROJECT_ROOT, args.out_root, tag)
    summary = run_backtest(
        price_folder=os.path.join(PROJECT_ROOT, args.test_price_folder),
        model=model,
        device=torch.device("cpu"),
        feature_columns=cols,
        look_back_window=args.look_back_window,
        cfg=make_cfg(args),
        tag=tag,
        out_dir=out_dir,
    )
    return summary


def main():
    args = parse_args()

    if not args.matrix:
        summary = run_one(args, args.tag or f"{args.model}_w{args.look_back_window}",
                          args.model_save_folder, resolve_cols(args), args.input_size, args.model)
        print(json.dumps(summary["portfolio"], ensure_ascii=False, indent=2))
        print("checks:", json.dumps(summary["checks"], ensure_ascii=False))
        return

    results = []
    for item in MATRIX:
        summary = run_one(args, item["tag"], item["folder"], item["cols"], item["size"],
                          item["model"], item["price"])
        results.append(summary)
        pf = summary["portfolio"]
        print(f"[{item['tag']}] ACC(mean)={summary['per_stock_mean']['acc']:.4f} "
              f"CumRet={pf['cum_return']:.2%} ARR={pf['arr']:.2%} SR={pf['sharpe']:.3f} "
              f"MDD={pf['mdd']:.2%} IR={pf['ir_vs_equal_weight']:.3f} trades={pf['total_trades']}")
        if not all(summary["checks"].values()):
            print("  !! 自检未全通过:", summary["checks"])

    print(json.dumps(
        {"runs": [{"tag": r["tag"], "portfolio": r["portfolio"], "checks": r["checks"]} for r in results]},
        ensure_ascii=False, indent=2)[:800])


if __name__ == "__main__":
    main()

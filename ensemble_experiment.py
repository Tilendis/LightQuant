"""
Ensemble 融合实验：在 CSMD50 测试集上对比单模型 vs 多模型融合
用法：cd D:\03_Study\毕设\LightQuant && .\.venv\Scripts\python.exe ensemble_experiment.py
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score

from model.LSTM import LSTM
from model.ALSTM import ALSTM
from model.BiLSTM import BiLSTM
from model.Adv_LSTM import AdvLSTM
from utils.price_dataloader import create_dataset, create_dataloader

# ============ 与训练时一致的参数 ============
LOOK_BACK_WINDOW = 5
INPUT_SIZE = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ATTENTION_SIZE = 128
BATCH_SIZE = 64
DROPOUT = 0.1
EPSILON = 0.1
PERTURBATION_SIZE = 0.1

TEST_PRICE_FOLDER = './dataset/CSMD50/test/price'
MODEL_SAVE_DIR = './result/CSMD50/model_saved/'
SAVE_DIR = './result/CSMD50'
DEVICE = torch.device('cpu')


def load_test_dataset(look_back_window=LOOK_BACK_WINDOW, test_price_folder=TEST_PRICE_FOLDER):
    """加载测试集，shuffle=False 保证所有模型样本顺序一致"""
    _, _, test_dataset = create_dataset(
        train_folder=None, val_folder=None,
        test_folder=test_price_folder,
        look_back_window=look_back_window
    )
    test_dataloader = create_dataloader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, drop_last=False
    )
    return test_dataloader


def build_model(name):
    """根据模型名构建模型实例"""
    if name == 'lstm':
        return LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS, output_size=1,
                    dropout=DROPOUT, batch_first=True)
    elif name == 'bi_lstm':
        return BiLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                      num_layers=NUM_LAYERS, output_size=1,
                      dropout=DROPOUT, batch_first=True)
    elif name == 'alstm':
        return ALSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                     num_layers=NUM_LAYERS, output_size=1,
                     dropout=DROPOUT, batch_first=True,
                     attention_size=ATTENTION_SIZE)
    elif name == 'adv_lstm':
        return AdvLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                       output_size=1, attention_size=ATTENTION_SIZE,
                       perturbation_size=PERTURBATION_SIZE, epsilon=EPSILON)
    else:
        raise ValueError(f"Unknown model: {name}")


def collect_predictions(model_name, test_dataloader, weight_path=None):
    """加载模型权重，在测试集上跑推理，返回 (概率数组, 标签数组)"""
    model = build_model(model_name)
    weight_path = weight_path or os.path.join(MODEL_SAVE_DIR, f'{model_name}.pth')
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_data, label in test_dataloader:
            input_data = input_data.to(DEVICE)
            label = label.squeeze()
            # 和 train.py 一致：把 -1 转成 0（二分类：0=跌，1=涨）
            label = torch.where(label == -1, torch.tensor(0), label).to(torch.int).float()

            prediction = model(input_data)
            prediction = prediction.squeeze()

            all_probs.extend(prediction.cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())

    return np.array(all_probs), np.array(all_labels)


def evaluate(labels, probs, threshold=0.5):
    """给定标签和概率，计算 ACC/MCC/F1"""
    preds = (probs > threshold).astype(int)
    acc = (preds == labels).mean()
    mcc = matthews_corrcoef(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, mcc, f1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--look_back_window', default=LOOK_BACK_WINDOW, type=int)
    p.add_argument('--test_price_folder', default=TEST_PRICE_FOLDER)
    p.add_argument('--save_dir', default=SAVE_DIR)
    p.add_argument('--weights', default='',
                   help='形如 "lstm=path,bi_lstm=path,alstm=path,adv_lstm=path"；为空则用 MODEL_SAVE_DIR')
    return p.parse_args()


def parse_weights(raw):
    mapping = {}
    for item in raw.split(','):
        item = item.strip()
        if not item:
            continue
        name, _, path = item.partition('=')
        mapping[name.strip()] = path.strip()
    return mapping


def main():
    args = parse_args()
    weights_map = parse_weights(args.weights)

    print("=" * 60)
    print("Ensemble 融合实验 — CSMD50 测试集")
    print(f"look_back_window={args.look_back_window}  test_price_folder={args.test_price_folder}")
    print(f"weights={weights_map if weights_map else MODEL_SAVE_DIR}")
    print("=" * 60)

    # 固定随机种子（确保可复现）
    torch.manual_seed(42)
    np.random.seed(42)

    test_dataloader = load_test_dataset(args.look_back_window, args.test_price_folder)
    print(f"测试集加载完成（shuffle=False，保证样本对齐）")

    # 4 个纯价格模型
    model_names = ['lstm', 'bi_lstm', 'alstm', 'adv_lstm']

    # 逐模型收集概率
    probs_dict = {}
    labels_ref = None

    for name in model_names:
        print(f"\n正在推理: {name} ...")
        probs, labels = collect_predictions(name, test_dataloader, weights_map.get(name))
        probs_dict[name] = probs

        if labels_ref is None:
            labels_ref = labels
        else:
            # 验证所有模型的标签完全一致
            assert np.array_equal(labels_ref, labels), f"{name} 标签不一致！"

        acc, mcc, f1 = evaluate(labels_ref, probs)
        print(f"  单独: ACC={acc:.4f} ({acc*100:.2f}%), MCC={mcc:.4f}, F1={f1:.4f}")

    print(f"\n总样本数: {len(labels_ref)}")
    print(f"正样本(涨)占比: {labels_ref.mean():.4f} ({labels_ref.mean()*100:.2f}%)")

    # ============ 融合策略 ============
    print("\n" + "=" * 60)
    print("融合策略对比")
    print("=" * 60)

    # 策略 1: 四模型简单平均
    avg_4 = np.mean([probs_dict[n] for n in model_names], axis=0)
    acc, mcc, f1 = evaluate(labels_ref, avg_4)
    print(f"[1] 四模型简单平均 (LSTM+BiLSTM+ALSTM+AdvLSTM):")
    print(f"    ACC={acc:.4f} ({acc*100:.2f}%), MCC={mcc:.4f}, F1={f1:.4f}")

    # 策略 2: 三模型简单平均（去掉 AdvLSTM，因为 F1 差）
    three_models = ['lstm', 'bi_lstm', 'alstm']
    avg_3 = np.mean([probs_dict[n] for n in three_models], axis=0)
    acc, mcc, f1 = evaluate(labels_ref, avg_3)
    print(f"\n[2] 三模型简单平均 (LSTM+BiLSTM+ALSTM):")
    print(f"    ACC={acc:.4f} ({acc*100:.2f}%), MCC={mcc:.4f}, F1={f1:.4f}")

    # 策略 3: 双强模型平均（ALSTM + BiLSTM，两个最好的）
    avg_strong = np.mean([probs_dict['alstm'], probs_dict['bi_lstm']], axis=0)
    acc, mcc, f1 = evaluate(labels_ref, avg_strong)
    print(f"\n[3] 双强平均 (ALSTM+BiLSTM):")
    print(f"    ACC={acc:.4f} ({acc*100:.2f}%), MCC={mcc:.4f}, F1={f1:.4f}")

    # 策略 4: 按 MCC 加权（验证集 MCC 越高权重越大）
    # 用测试集单独 MCC 作为权重（实际应该用验证集，这里近似）
    mccs = {}
    for name in model_names:
        _, mcc, _ = evaluate(labels_ref, probs_dict[name])
        mccs[name] = max(mcc, 0.01)  # 避免负权重
    weights = {n: mccs[n] / sum(mccs.values()) for n in model_names}
    print(f"\n[4] 按测试集 MCC 加权:")
    for n, w in weights.items():
        print(f"    {n}: weight={w:.4f}")

    weighted = np.zeros_like(avg_4)
    for n in model_names:
        weighted += weights[n] * probs_dict[n]
    acc, mcc, f1 = evaluate(labels_ref, weighted)
    print(f"    → ACC={acc:.4f} ({acc*100:.2f}%), MCC={mcc:.4f}, F1={f1:.4f}")
    print("    [警告] 该权重由测试集 MCC 反推，属 test leakage，仅作探索，不可写入论文")

    # 策略 5: 硬投票（多数表决）
    votes = np.zeros_like(avg_4)
    for n in model_names:
        votes += (probs_dict[n] > 0.5).astype(int)
    majority_vote = (votes >= 2).astype(int)  # 4 个模型里至少 2 个说涨
    acc = (majority_vote == labels_ref).mean()
    mcc = matthews_corrcoef(labels_ref, majority_vote)
    f1 = f1_score(labels_ref, majority_vote)
    print(f"\n[5] 硬投票 (4模型至少2票涨):")
    print(f"    ACC={acc:.4f} ({acc*100:.2f}%), MCC={mcc:.4f}, F1={f1:.4f}")

    # 策略 6: 最优单模型 ALSTM 单独作为 baseline
    acc, mcc, f1 = evaluate(labels_ref, probs_dict['alstm'])
    print(f"\n[Baseline] ALSTM 单独（当前最强）:")
    print(f"    ACC={acc:.4f} ({acc*100:.2f}%), MCC={mcc:.4f}, F1={f1:.4f}")

    # ============ 阈值调优 ============
    print("\n" + "=" * 60)
    print("阈值调优（在三模型平均概率上扫阈值）")
    print("=" * 60)
    for th in [0.45, 0.48, 0.50, 0.52, 0.55]:
        acc, mcc, f1 = evaluate(labels_ref, avg_3, threshold=th)
        print(f"  threshold={th:.2f}: ACC={acc:.4f} ({acc*100:.2f}%), MCC={mcc:.4f}, F1={f1:.4f}")

    # ====== 结果落盘（审计要求：数字必须有文件可查） ======
    def _m(labels, probs, th=0.5):
        a, m, f = evaluate(labels, probs, th)
        return dict(zip(["acc", "mcc", "f1"], [float(a), float(m), float(f)]))

    report = {
        "script": "ensemble_experiment.py",
        "dataset": "CSMD50",
        "look_back_window": args.look_back_window,
        "weights": weights_map or {"*": os.path.join(MODEL_SAVE_DIR, "<name>.pth")},
        "n_test": int(len(labels_ref)),
        "test_pos_ratio": float(labels_ref.mean()),
        "single_models": {n: _m(labels_ref, probs_dict[n]) for n in model_names},
        "strategies": {
            "avg4_lstm_bi_alstm_adv": _m(labels_ref, avg_4),
            "avg3_lstm_bi_alstm": _m(labels_ref, avg_3),
            "avg2_alstm_bi": _m(labels_ref, avg_strong),
            "mcc_weighted": _m(labels_ref, weighted),
            "majority_vote": dict(zip(["acc", "mcc", "f1"], [
                float((majority_vote == labels_ref).mean()),
                float(matthews_corrcoef(labels_ref, majority_vote)),
                float(f1_score(labels_ref, majority_vote))
            ])),
        },
        "avg3_threshold_sweep": {str(th): _m(labels_ref, avg_3, th) for th in [0.45, 0.48, 0.50, 0.52, 0.55]},
        "leaky_strategies": ["mcc_weighted"],
        "leakage_note": "mcc_weighted 的权重由测试集 MCC 反推，属 test leakage，仅作探索，不得写入论文或汇报结论。",
    }

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "ensemble_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[已落盘] {out_path}")

    print("\n完成！")


if __name__ == '__main__':
    main()

"""
Val 校准阈值实验：在验证集上调最优阈值，再在测试集上报告
用法：cd D:\03_Study\毕设\LightQuant && .\.venv\Scripts\python.exe val_calibration.py
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

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

VAL_PRICE_FOLDER = './dataset/CSMD50/val/price'
TEST_PRICE_FOLDER = './dataset/CSMD50/test/price'
MODEL_SAVE_DIR = './result/CSMD50/model_saved/'
SAVE_DIR = './result/CSMD50'
DEVICE = torch.device('cpu')


def build_model(name):
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


def collect_predictions(model_name, dataloader, weight_path=None):
    model = build_model(model_name)
    weight_path = weight_path or os.path.join(MODEL_SAVE_DIR, f'{model_name}.pth')
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_data, label in dataloader:
            input_data = input_data.to(DEVICE)
            label = label.squeeze()
            label = torch.where(label == -1, torch.tensor(0), label).to(torch.int).float()

            prediction = model(input_data)
            prediction = prediction.squeeze()

            all_probs.extend(prediction.cpu().numpy().flatten())
            all_labels.extend(label.cpu().numpy().flatten())

    return np.array(all_probs), np.array(all_labels)


def eval_with_threshold(labels, probs, threshold):
    preds = (probs > threshold).astype(int)
    acc = accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, mcc, f1


def find_best_threshold(labels, probs, search_range=np.arange(0.35, 0.65, 0.01)):
    """在验证集上搜索使 MCC 最大的阈值"""
    best_mcc = -1
    best_th = 0.5
    best_metrics = None

    for th in search_range:
        acc, mcc, f1 = eval_with_threshold(labels, probs, th)
        if mcc > best_mcc:
            best_mcc = mcc
            best_th = th
            best_metrics = (acc, mcc, f1)

    return best_th, best_metrics


def find_quantile_threshold(labels, probs, target_ratio=None):
    """
    先验分位数法：按真实正样本比例，取预测概率排名前 r% 的样本判为 1
    如果 target_ratio 为 None，用 labels 中正样本比例
    """
    if target_ratio is None:
        target_ratio = labels.mean()

    # 取概率排名前 target_ratio 的阈值
    sorted_probs = np.sort(probs)
    idx = int(len(probs) * (1 - target_ratio))
    quantile_th = sorted_probs[idx]

    acc, mcc, f1 = eval_with_threshold(labels, probs, quantile_th)
    return quantile_th, (acc, mcc, f1), target_ratio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--look_back_window', default=LOOK_BACK_WINDOW, type=int)
    p.add_argument('--val_price_folder', default=VAL_PRICE_FOLDER)
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
    print("Val 校准阈值实验")
    print("在验证集上调最优阈值，在测试集上报告")
    print(f"look_back_window={args.look_back_window}")
    print(f"val_folder={args.val_price_folder}  test_folder={args.test_price_folder}")
    print(f"weights={weights_map if weights_map else MODEL_SAVE_DIR}")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # 加载验证集
    _, val_dataset, _ = create_dataset(
        train_folder=None, val_folder=args.val_price_folder,
        test_folder=None, look_back_window=args.look_back_window
    )
    val_dataloader = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 加载测试集
    _, _, test_dataset = create_dataset(
        train_folder=None, val_folder=None,
        test_folder=args.test_price_folder, look_back_window=args.look_back_window
    )
    test_dataloader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    model_names = ['lstm', 'bi_lstm', 'alstm', 'adv_lstm']

    # ====== 在验证集上收集概率 ======
    print("\n--- 在验证集上收集预测概率 ---")
    val_probs = {}
    val_labels = None

    for name in model_names:
        print(f"  推理: {name} ...")
        probs, labels = collect_predictions(name, val_dataloader, weights_map.get(name))
        val_probs[name] = probs
        if val_labels is None:
            val_labels = labels
        else:
            assert np.array_equal(val_labels, labels), f"{name} 验证集标签不一致！"

    print(f"\n验证集正样本比例: {val_labels.mean():.4f}")

    # ====== 在测试集上收集概率 ======
    print("\n--- 在测试集上收集预测概率 ---")
    test_probs = {}
    test_labels = None

    for name in model_names:
        print(f"  推理: {name} ...")
        probs, labels = collect_predictions(name, test_dataloader, weights_map.get(name))
        test_probs[name] = probs
        if test_labels is None:
            test_labels = labels
        else:
            assert np.array_equal(test_labels, labels), f"{name} 测试集标签不一致！"

    print(f"\n测试集正样本比例: {test_labels.mean():.4f}")

    # ====== 逐模型分析 ======
    print("\n" + "=" * 60)
    print("逐模型：验证集调阈值 → 测试集报告")
    print("=" * 60)

    for name in model_names:
        print(f"\n【{name}】")

        # 方法 1：默认阈值 0.5
        acc0, mcc0, f10 = eval_with_threshold(test_labels, test_probs[name], 0.5)
        print(f"  默认阈值 0.50: ACC={acc0:.4f}, MCC={mcc0:.4f}, F1={f10:.4f}")

        # 方法 2：验证集最优绝对阈值（按 MCC 最大）
        best_th, best_val_metrics = find_best_threshold(val_labels, val_probs[name])
        acc_test, mcc_test, f1_test = eval_with_threshold(test_labels, test_probs[name], best_th)
        print(f"  Val最优绝对阈值={best_th:.2f}: ACC={acc_test:.4f}, MCC={mcc_test:.4f}, F1={f1_test:.4f}")
        print(f"    (验证集上该阈值: ACC={best_val_metrics[0]:.4f}, MCC={best_val_metrics[1]:.4f}, F1={best_val_metrics[2]:.4f})")

        # 方法 3：先验分位数阈值（按验证集正样本比例）
        q_th, q_test_metrics, ratio = find_quantile_threshold(val_labels, val_probs[name])
        print(f"  先验分位数阈值={q_th:.4f} (正样本比例={ratio:.4f}): ACC={q_test_metrics[0]:.4f}, MCC={q_test_metrics[1]:.4f}, F1={q_test_metrics[2]:.4f}")

    # ====== 重点分析 ALSTM（最强模型） ======
    print("\n" + "=" * 60)
    print("重点：ALSTM 阈值扫描详情")
    print("=" * 60)

    print("\n验证集阈值扫描（找 MCC 峰值）:")
    for th in np.arange(0.40, 0.60, 0.02):
        acc, mcc, f1 = eval_with_threshold(val_labels, val_probs['alstm'], th)
        print(f"  th={th:.2f}: Val MCC={mcc:.4f}, Val F1={f1:.4f}")

    best_th, _ = find_best_threshold(val_labels, val_probs['alstm'])
    print(f"\n→ 验证集最优阈值 = {best_th:.2f}")

    print(f"\n在测试集上对比:")
    for th in [0.50, best_th]:
        acc, mcc, f1 = eval_with_threshold(test_labels, test_probs['alstm'], th)
        print(f"  th={th:.2f}: Test ACC={acc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}")

    # ====== 三模型平均 + Val 校准 ======
    print("\n" + "=" * 60)
    print("三模型平均（LSTM+BiLSTM+ALSTM）+ Val 校准")
    print("=" * 60)

    avg_val = np.mean([val_probs['lstm'], val_probs['bi_lstm'], val_probs['alstm']], axis=0)
    avg_test = np.mean([test_probs['lstm'], test_probs['bi_lstm'], test_probs['alstm']], axis=0)

    # 默认阈值
    acc0, mcc0, f10 = eval_with_threshold(test_labels, avg_test, 0.5)
    print(f"  默认阈值 0.50: ACC={acc0:.4f}, MCC={mcc0:.4f}, F1={f10:.4f}")

    # Val 最优阈值
    best_th, best_val_metrics = find_best_threshold(val_labels, avg_val)
    acc_test, mcc_test, f1_test = eval_with_threshold(test_labels, avg_test, best_th)
    print(f"  Val最优阈值={best_th:.2f}: ACC={acc_test:.4f}, MCC={mcc_test:.4f}, F1={f1_test:.4f}")

    # 先验分位数
    q_th, q_test_metrics, ratio = find_quantile_threshold(val_labels, avg_val)
    print(f"  先验分位数={q_th:.4f}: ACC={q_test_metrics[0]:.4f}, MCC={q_test_metrics[1]:.4f}, F1={q_test_metrics[2]:.4f}")

    # ====== 结果落盘（审计要求：数字必须有文件可查） ======
    report = {
        "script": "val_calibration.py",
        "dataset": "CSMD50",
        "split": {"val_folder": args.val_price_folder, "test_folder": args.test_price_folder},
        "look_back_window": args.look_back_window,
        "weights": weights_map or {"*": os.path.join(MODEL_SAVE_DIR, "<name>.pth")},
        "n_val": int(len(val_labels)),
        "n_test": int(len(test_labels)),
        "val_pos_ratio": float(val_labels.mean()),
        "test_pos_ratio": float(test_labels.mean()),
        "per_model": {},
        "ensemble3": {},
        "leakage_note": "阈值只在 val 上搜索，test 仅前向一次；如阈值在 test 上遍历则为泄漏。",
    }

    for name in model_names:
        d = {"default_0.50": dict(zip(["acc", "mcc", "f1"],
                                     [float(x) for x in eval_with_threshold(test_labels, test_probs[name], 0.5)]))
             }
        th, vm = find_best_threshold(val_labels, val_probs[name])
        d["val_best_by_mcc"] = {
            "threshold": float(th),
            "val": dict(zip(["acc", "mcc", "f1"], [float(x) for x in vm])),
            "test": dict(zip(["acc", "mcc", "f1"],
                             [float(x) for x in eval_with_threshold(test_labels, test_probs[name], th)])),
        }
        qth, qm, ratio = find_quantile_threshold(val_labels, val_probs[name])
        d["val_quantile_prior"] = {
            "threshold": float(qth), "target_pos_ratio": float(ratio),
            "test": dict(zip(["acc", "mcc", "f1"], [float(x) for x in qm])),
        }
        report["per_model"][name] = d

    report["alstm_val_sweep"] = [
        {"threshold": float(th), "val_mcc": float(eval_with_threshold(val_labels, val_probs['alstm'], th)[1])}
        for th in np.arange(0.40, 0.60, 0.02)
    ]

    avg_val = np.mean([val_probs['lstm'], val_probs['bi_lstm'], val_probs['alstm']], axis=0)
    avg_test = np.mean([test_probs['lstm'], test_probs['bi_lstm'], test_probs['alstm']], axis=0)
    report["ensemble3"]["models"] = ["lstm", "bi_lstm", "alstm"]
    report["ensemble3"]["default_0.50"] = dict(zip(["acc", "mcc", "f1"],
                                                   [float(x) for x in eval_with_threshold(test_labels, avg_test, 0.5)]))
    th, vm = find_best_threshold(val_labels, avg_val)
    report["ensemble3"]["val_best_by_mcc"] = {
        "threshold": float(th),
        "val": dict(zip(["acc", "mcc", "f1"], [float(x) for x in vm])),
        "test": dict(zip(["acc", "mcc", "f1"], [float(x) for x in eval_with_threshold(test_labels, avg_test, th)]))
    }
    qth, qm, ratio = find_quantile_threshold(val_labels, avg_val)
    report["ensemble3"]["val_quantile_prior"] = {
        "threshold": float(qth), "target_pos_ratio": float(ratio),
        "test": dict(zip(["acc", "mcc", "f1"], [float(x) for x in qm]))
    }

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, "calibration_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[已落盘] {out_path}")

    print("\n完成！")


if __name__ == '__main__':
    main()

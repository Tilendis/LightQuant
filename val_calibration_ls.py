"""
Label Smoothing 版 ALSTM 单模型 Val 校准
用法：cd D:\03_Study\毕设\LightQuant && .\.venv\Scripts\python.exe val_calibration_ls.py
"""
import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

from model.ALSTM import ALSTM
from utils.price_dataloader import create_dataset, create_dataloader

# ============ 与训练时一致的参数 ============
LOOK_BACK_WINDOW = 5
INPUT_SIZE = 5
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ATTENTION_SIZE = 128
BATCH_SIZE = 64
DROPOUT = 0.1

VAL_PRICE_FOLDER = './dataset/CSMD50/val/price'
TEST_PRICE_FOLDER = './dataset/CSMD50/test/price'
MODEL_PATH = './result/CSMD50_ls/model_saved/alstm.pth'
SAVE_DIR = './result/CSMD50_ls'
DEVICE = torch.device('cpu')


def build_model():
    return ALSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_size=1,
                 dropout=DROPOUT, batch_first=True,
                 attention_size=ATTENTION_SIZE)


def collect_predictions(dataloader):
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
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


def main():
    print("=" * 60)
    print("Label Smoothing 版 ALSTM 单模型 Val 校准")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # 加载验证集
    _, val_dataset, _ = create_dataset(
        train_folder=None, val_folder=VAL_PRICE_FOLDER,
        test_folder=None, look_back_window=LOOK_BACK_WINDOW
    )
    val_dataloader = create_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 加载测试集
    _, _, test_dataset = create_dataset(
        train_folder=None, val_folder=None,
        test_folder=TEST_PRICE_FOLDER, look_back_window=LOOK_BACK_WINDOW
    )
    test_dataloader = create_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # 收集概率
    print("\n--- 在验证集上收集预测概率 ---")
    val_probs, val_labels = collect_predictions(val_dataloader)
    print(f"验证集正样本比例: {val_labels.mean():.4f}")
    print(f"预测概率均值: {val_probs.mean():.4f}, 标准差: {val_probs.std():.4f}")

    print("\n--- 在测试集上收集预测概率 ---")
    test_probs, test_labels = collect_predictions(test_dataloader)
    print(f"测试集正样本比例: {test_labels.mean():.4f}")

    # ====== 阈值扫描 ======
    print("\n" + "=" * 60)
    print("Val 集阈值扫描（找最优点）")
    print("=" * 60)

    search_range = np.arange(0.35, 0.65, 0.01)

    best_mcc = -1
    best_mcc_th = 0.5
    best_mcc_metrics = None

    best_combo = -1
    best_combo_th = 0.5
    best_combo_metrics = None

    print(f"{'阈值':>6} {'Val_ACC':>8} {'Val_MCC':>8} {'Val_F1':>8} {'ACC+0.5*MCC':>12}")
    print("-" * 60)

    for th in search_range:
        acc, mcc, f1 = eval_with_threshold(val_labels, val_probs, th)
        combo = acc + 0.5 * mcc

        if mcc > best_mcc:
            best_mcc = mcc
            best_mcc_th = th
            best_mcc_metrics = (acc, mcc, f1)

        if combo > best_combo:
            best_combo = combo
            best_combo_th = th
            best_combo_metrics = (acc, mcc, f1)

        if abs(th - 0.40) < 0.005 or abs(th - 0.45) < 0.005 or abs(th - 0.50) < 0.005 or abs(th - best_mcc_th) < 0.005 or abs(th - best_combo_th) < 0.005:
            print(f"{th:6.2f} {acc:8.4f} {mcc:8.4f} {f1:8.4f} {combo:12.4f}")

    # ====== 两个点位在测试集上的结果 ======
    print("\n" + "=" * 60)
    print("测试集结果")
    print("=" * 60)

    # 默认阈值 0.5
    acc0, mcc0, f10 = eval_with_threshold(test_labels, test_probs, 0.5)
    print(f"\n【基准】默认阈值 0.50:")
    print(f"  ACC={acc0:.4f}, MCC={mcc0:.4f}, F1={f10:.4f}")

    # 点位 A：最大化 ACC + 0.5 * MCC（保高 ACC）
    acc_a, mcc_a, f1_a = eval_with_threshold(test_labels, test_probs, best_combo_th)
    print(f"\n【点位 A】Val最优阈值={best_combo_th:.2f} (最大化 ACC + 0.5*MCC):")
    print(f"  Test ACC={acc_a:.4f}, MCC={mcc_a:.4f}, F1={f1_a:.4f}")
    print(f"  (Val 上: ACC={best_combo_metrics[0]:.4f}, MCC={best_combo_metrics[1]:.4f}, F1={best_combo_metrics[2]:.4f})")

    # 点位 B：最大化 MCC（冲极限）
    acc_b, mcc_b, f1_b = eval_with_threshold(test_labels, test_probs, best_mcc_th)
    print(f"\n【点位 B】Val最优阈值={best_mcc_th:.2f} (最大化 MCC):")
    print(f"  Test ACC={acc_b:.4f}, MCC={mcc_b:.4f}, F1={f1_b:.4f}")
    print(f"  (Val 上: ACC={best_mcc_metrics[0]:.4f}, MCC={best_mcc_metrics[1]:.4f}, F1={best_mcc_metrics[2]:.4f})")

    # ====== 汇总对比 ======
    print("\n" + "=" * 60)
    print("汇总对比")
    print("=" * 60)
    print(f"{'方案':<30} {'ACC':>8} {'MCC':>8} {'F1':>8}")
    print("-" * 60)
    print(f"{'基线 (0.5阈值)':<30} {acc0:8.4f} {mcc0:8.4f} {f10:8.4f}")
    print(f"{'点位A (ACC+0.5*MCC最优)':<30} {acc_a:8.4f} {mcc_a:8.4f} {f1_a:8.4f}")
    print(f"{'点位B (MCC最优)':<30} {acc_b:8.4f} {mcc_b:8.4f} {f1_b:8.4f}")

    # ====== 结果落盘（审计要求：数字必须有文件可查） ======
    report = {
        "script": "val_calibration_ls.py",
        "model": "alstm + label_smoothing(eps=0.1)",
        "model_path": MODEL_PATH,
        "dataset": "CSMD50",
        "n_val": int(len(val_labels)),
        "n_test": int(len(test_labels)),
        "val_pos_ratio": float(val_labels.mean()),
        "test_pos_ratio": float(test_labels.mean()),
        "val_prob_mean": float(val_probs.mean()),
        "val_prob_std": float(val_probs.std()),
        "test_prob_mean": float(test_probs.mean()),
        "test_prob_std": float(test_probs.std()),
        "baseline_0.50": dict(zip(["acc", "mcc", "f1"], [float(x) for x in (acc0, mcc0, f10)]))
        ,
        "point_A_max_acc_plus_half_mcc": {
            "rule": "argmax(Val_ACC + 0.5*Val_MCC)",
            "threshold": float(best_combo_th),
            "val": dict(zip(["acc", "mcc", "f1"], [float(x) for x in best_combo_metrics])),
            "test": dict(zip(["acc", "mcc", "f1"], [float(x) for x in (acc_a, mcc_a, f1_a)]))
        },
        "point_B_max_mcc": {
            "rule": "argmax(Val_MCC)",
            "threshold": float(best_mcc_th),
            "val": dict(zip(["acc", "mcc", "f1"], [float(x) for x in best_mcc_metrics])),
            "test": dict(zip(["acc", "mcc", "f1"], [float(x) for x in (acc_b, mcc_b, f1_b)]))
        },
        "val_threshold_sweep": [
            {"threshold": float(th),
             "val_acc": float(eval_with_threshold(val_labels, val_probs, th)[0]),
             "val_mcc": float(eval_with_threshold(val_labels, val_probs, th)[1]),
             "val_f1": float(eval_with_threshold(val_labels, val_probs, th)[2])}
            for th in search_range
        ],
        "leakage_note": "阈值只在 val 上搜索，test 仅前向一次。",
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, "calibration_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[已落盘] {out_path}")

    print("\n完成！")


if __name__ == '__main__':
    main()

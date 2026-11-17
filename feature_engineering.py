"""
特征工程：在 CSMD50 完整序列上计算技术指标，再按原日期切分 train/val/test
输出到 dataset/CSMD50_feat/ 目录
用法：cd D:\03_Study\毕设\LightQuant && .\.venv\Scripts\python.exe feature_engineering.py
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_SRC = 'dataset/CSMD50'
DATASET_DST = 'dataset/CSMD50_feat'

# 切分行数（不含表头）
N_TRAIN = 774
N_VAL = 98
N_TEST = 97


def compute_rsi(close, window=6):
    """RSI(6)"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd_hist(close, fast=12, slow=26, signal=9):
    """MACD 柱 = (DIF - DEA) * 2"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return macd_hist


def compute_bias(close, window=5):
    """BIAS(5) = (Close - MA5) / MA5"""
    ma = close.rolling(window=window).mean()
    bias = (close - ma) / ma
    return bias


def process_one_stock(stock_name):
    """对单支股票：拼接 train/val/test → 算指标 → 切分 → 保存"""
    train_path = f'{DATASET_SRC}/train/price/{stock_name}'
    val_path = f'{DATASET_SRC}/val/price/{stock_name}'
    test_path = f'{DATASET_SRC}/test/price/{stock_name}'

    # 读三个 CSV
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # 按日期拼接成完整 969 天序列
    df_full = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # ===== 在完整序列上计算技术指标 =====
    df_full['Return'] = df_full['Close'].pct_change()
    df_full['RSI6'] = compute_rsi(df_full['Close'], window=6)
    df_full['MACD_Hist'] = compute_macd_hist(df_full['Close'])
    df_full['BIAS5'] = compute_bias(df_full['Close'], window=5)
    df_full['Amplitude'] = (df_full['High'] - df_full['Low']) / df_full['Close']

    # 开头 NaN 处理：前向填充，再填 0
    feature_cols = ['Return', 'RSI6', 'MACD_Hist', 'BIAS5', 'Amplitude']
    df_full[feature_cols] = df_full[feature_cols].bfill().fillna(0)

    # 按各自实际行数切分回去（不同股票数据量不同）
    n_train = len(df_train)
    n_val = len(df_val)
    n_test = len(df_test)

    df_out_train = df_full.iloc[:n_train].copy()
    df_out_val = df_full.iloc[n_train:n_train + n_val].copy()
    df_out_test = df_full.iloc[n_train + n_val:].copy()

    # 校验行数与原始一致
    assert len(df_out_train) == n_train, f"train 行数不对: {len(df_out_train)} != {n_train}"
    assert len(df_out_val) == n_val, f"val 行数不对: {len(df_out_val)} != {n_val}"
    assert len(df_out_test) == n_test, f"test 行数不对: {len(df_out_test)} != {n_test}"

    # 校验 Label 与原始一致（防止切错位）
    assert (df_out_train['Label'].values == df_train['Label'].values).all(), "train Label 不一致！"
    assert (df_out_val['Label'].values == df_val['Label'].values).all(), "val Label 不一致！"
    assert (df_out_test['Label'].values == df_test['Label'].values).all(), "test Label 不一致！"

    # 保存
    for split, df_split in [('train', df_out_train), ('val', df_out_val), ('test', df_out_test)]:
        out_dir = Path(f'{DATASET_DST}/{split}/price')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / stock_name
        df_split.to_csv(out_path, index=False)

    return len(df_out_train), len(df_out_val), len(df_out_test)


def main():
    print("=" * 60)
    print("特征工程：CSMD50 → CSMD50_feat")
    print("新增 5 个特征: Return, RSI6, MACD_Hist, BIAS5, Amplitude")
    print("=" * 60)

    # 获取所有股票文件名
    train_dir = Path(f'{DATASET_SRC}/train/price')
    stock_files = sorted([f.name for f in train_dir.iterdir() if f.suffix == '.csv'])

    print(f"共 {len(stock_files)} 支股票")

    for i, stock_name in enumerate(stock_files):
        n_tr, n_va, n_te = process_one_stock(stock_name)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(stock_files)}] {stock_name}: train={n_tr}, val={n_va}, test={n_te}")

    # 验证输出
    sample = pd.read_csv(f'{DATASET_DST}/train/price/{stock_files[0]}')
    print(f"\n完成！样本列名: {list(sample.columns)}")
    print(f"特征列: {[c for c in sample.columns if c not in ['Date', 'Label']]}")
    print(f"前 3 行新特征:")
    print(sample[['Return', 'RSI6', 'MACD_Hist', 'BIAS5', 'Amplitude']].head(3).to_string())

    # 检查 NaN
    total_nan = sample[['Return', 'RSI6', 'MACD_Hist', 'BIAS5', 'Amplitude']].isna().sum().sum()
    print(f"\n样本 NaN 总数: {total_nan}")


if __name__ == '__main__':
    main()

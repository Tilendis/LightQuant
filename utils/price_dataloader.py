import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================ 除权防护 ============================
# 为什么需要这道检查：CSMD50 数据源不提供复权因子。scripts/audit_prices.py 实测
# 150/150 个文件的 Adj Close 与 Close 逐行完全相同，即价格未经复权修正；
# docs/price_jump_scan.json 实测 train 段 2021-01-04~2024-03-14 存在 19 次除权，
# 38 行超阈值跳空全部落在 train 段（val/test 段为 0）。
# 若滑动窗口或标签跨过除权日，特征与标签会同时失真，而这种失真在准确率上
# 看不出来——它不会让指标变差，只会让指标失去意义。因此每次加载都做一次
# 静态检查，不修改数据、只如实报告；接入新数据源时设 LQ_STRICT_ADJ=1 可直接把关。
ADJ_JUMP_TH = 0.21   # 任何板块的涨跌停都到不了 21%，超过即视为除权/异常，不按板块分流

_ADJ_AUDIT = {
    "files": 0,
    "files_without_adj": 0,
    "jump_rows": 0,
    "jump_files": [],
    "max_abs_ret": 0.0,
}


def reset_adj_audit():
    """清空加载期除权审计的累计计数（每次 create_dataset 开始时调用）。"""
    _ADJ_AUDIT["files"] = 0
    _ADJ_AUDIT["files_without_adj"] = 0
    _ADJ_AUDIT["jump_rows"] = 0
    _ADJ_AUDIT["jump_files"] = []
    _ADJ_AUDIT["max_abs_ret"] = 0.0


def audit_price_adjustment(df, name=""):
    """检查单只股票的价格表是否带复权修正、是否存在超阈值跳空，只报告不改数据。

    has_adj 的判据是 Adj Close 与 Close 是否逐行完全相同：相同即"没有复权修正"。
    n_jumps 用 Close 自身的 pct_change 计算，不依赖 Return 列是否存在。
    返回 (has_adj, n_jumps, max_abs_ret)。
    """
    has_adj = True
    if "Adj Close" in df.columns and "Close" in df.columns:
        has_adj = not bool((df["Adj Close"] == df["Close"]).all())

    ret = df["Close"].pct_change()
    n_jumps = int((ret.abs() > ADJ_JUMP_TH).sum())
    max_abs_ret = float(ret.abs().max()) if len(ret) else 0.0

    _ADJ_AUDIT["files"] += 1
    if not has_adj:
        _ADJ_AUDIT["files_without_adj"] += 1
    if n_jumps:
        _ADJ_AUDIT["jump_rows"] += n_jumps
        if len(_ADJ_AUDIT["jump_files"]) < 10:
            _ADJ_AUDIT["jump_files"].append((name, n_jumps))
    if max_abs_ret > _ADJ_AUDIT["max_abs_ret"]:
        _ADJ_AUDIT["max_abs_ret"] = max_abs_ret

    # 只有"未提供复权修正 + 存在超阈值跳空"才拦：有修正的文件跳空已被抹平，
    # 再报错就是误伤；而未修正的文件跨除权日训练，特征与标签会同时失真。
    if os.environ.get("LQ_STRICT_ADJ") == "1" and n_jumps and not has_adj:
        raise AssertionError(
            "除权防护断言触发：%s 存在 %d 处单日涨跌幅超过 %.0f%%，"
            "且数据未提供复权修正。跨除权日的窗口会让特征与标签同时失真。"
            % (name, n_jumps, ADJ_JUMP_TH * 100)
        )
    return has_adj, n_jumps, max_abs_ret


def adj_audit_summary():
    """返回加载期除权审计的累计结果（不清空）。"""
    s = dict(_ADJ_AUDIT)
    s["jump_files"] = list(s["jump_files"])
    return s


def _print_adj_summary():
    s = adj_audit_summary()
    if not s["files"]:
        return
    print("[除权防护] 加载 %d 个文件；未提供复权修正 %d 个；"
          "单日跳空 >%.0f%% 共 %d 行；最大单日 |涨跌幅| %.2f%%"
          % (s["files"], s["files_without_adj"], ADJ_JUMP_TH * 100,
             s["jump_rows"], s["max_abs_ret"] * 100))
    if s["jump_files"]:
        print("[除权防护] 跳空文件样例: %s" % (s["jump_files"][:5],))


def load_csv_file(csv_file, feature_columns, label_columns):
    """读取单只股票的 csv，返回 (标准化后的特征张量, 原始标签张量)。

    注意：标准化发生在读取阶段，且使用该文件**全序列**的均值/方差，
    会用到训练期之后的信息。这是原仓库的既有写法，本次未改动，
    在报告中作为"数据泄漏风险"如实列为局限，不宣称已修正。
    """
    dataframe = pd.read_csv(csv_file)
    audit_price_adjustment(dataframe, os.path.basename(csv_file))
    # print(csv_file)
    standardized_dataframe = standardize_dataframe(dataframe, feature_columns)

    feature = torch.tensor(standardized_dataframe[feature_columns].values, dtype=torch.float32)
    label = torch.tensor(standardized_dataframe[label_columns].values, dtype=torch.float32)
    return feature, label


def split_sequence(sequence, length):
    """把 (T, dim) 的序列切成滑动窗口，返回 (T-length+1, length, dim)。

    逐文件调用，因此不同股票的时间序列不会被拼进同一个窗口。
    """
    sequences = []
    for i in range(len(sequence) - length + 1):
        sequences.append(sequence[i: i + length])
    sequences = torch.stack(sequences)
    return sequences


def standardize_dataframe(dataframe, feature_columns):
    """按列做 z-score（减均值、除标准差），就地修改并返回同一个 DataFrame。

    逐文件独立标准化，不做跨股票的统一样本统计。
    """

    for col in feature_columns:
        mean = dataframe[col].mean()
        std = dataframe[col].std()
        dataframe[col] = (dataframe[col] - mean) / std
    return dataframe


DEFAULT_FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close']


def resolve_feature_columns(args):
    """从命令行解析 --feature_columns；为空时返回 None（沿用默认 5 列价格）。

    与 --input_size 不一致时直接报错：这两者必须同步，否则模型第一层维度对不上，
    或者悄悄只用了几列（正是"窗口参数没生效"那类静默失败的同款问题）。
    """
    raw = getattr(args, "feature_columns", "") or ""
    columns = [c.strip() for c in raw.split(",") if c.strip()]
    if not columns:
        return None
    input_size = getattr(args, "input_size", len(columns))
    if len(columns) != input_size:
        raise ValueError(
            f"--feature_columns 给出了 {len(columns)} 列 {columns}，"
            f"但 --input_size = {input_size}；两者必须一致"
        )
    return columns


class Normal_Dataset(Dataset):
    """纯价格/特征模型用的数据集：每个样本 = 一个窗口，标签 = 窗口最后一天。

    样本数 = 每只股票 (行数 - 窗口 + 1) 之和；CSMD50 在 5 天窗口下
    train 37712 / val 4700 / test 4650。
    """

    def __init__(self, csv_files=[], look_back_window=1, feature_columns=None):
        super().__init__()
        # 修复：原实现未把 look_back_window 传给 load_csv_files，
        # 导致 look_back 恒为默认值 1，"5 天窗口"实际只有 1 天。
        self.feature_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)
        self.load_csv_files(csv_files, look_back=look_back_window, features_columns=self.feature_columns)
        self.mean = None
        self.std = None

    def load_csv_files(
            self,
            csv_files=[],
            look_back=1,
            features_columns=DEFAULT_FEATURE_COLUMNS,
            label_columns=['Label'],
    ):
        self.features, self.labels = [], []
        for csv_file in csv_files:
            feature, label = load_csv_file(csv_file, features_columns, label_columns)
            features = split_sequence(feature, look_back)
            # 修复：原来对 label 也做 split_sequence，得到 (N, look_back, 1)；
            # 但模型每个窗口只输出 1 个概率，标签应取窗口最后一天的 Label，
            # 即"用过去 look_back 天预测紧接着的下一天"。
            # 原写法在 look_back>1 时形状为 (B, look_back)，与预测 (B,) 无法计算 BCE。
            labels = label[look_back - 1: look_back - 1 + features.shape[0]]
            self.features.append(features)
            self.labels.append(labels)

        self.features = torch.concat(self.features, dim=0)
        self.labels = torch.concat(self.labels, dim=0)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label

class Backtest_Dataset(Dataset):
    def __init__(self, csv_file, look_back_window=7, feature_columns=None,
                 label_column='Label', scaler_csv=None):
        super().__init__()
        self.look_back_window = look_back_window
        self.feature_columns = feature_columns or ['Open', 'High', 'Low', 'Close', 'Adj Close']
        self.label_column = label_column

        df = pd.read_csv(csv_file)
        self.dates = df['Date'].values[look_back_window - 1:]


        raw_close = df['Close'].values
        self.raw_close_prices = raw_close

        # 特征
        raw_features = df[self.feature_columns].values
        labels = df[label_column].values

        # 前视泄漏修复：scaler_csv 传入训练段 csv 时，mean/std 只在 train 段
        # 拟合，test 段拿来 transform。不传则退回旧行为（整段 test 自己 fit），
        # 但那会让 8 月的特征减掉 12 月参与的均值。
        if scaler_csv:
            train_df = pd.read_csv(scaler_csv)
            standardized_df = df.copy()
            for col in self.feature_columns:
                m = float(train_df[col].mean())
                s = float(train_df[col].std())
                standardized_df[col] = (df[col] - m) / s
        else:
            standardized_df = standardize_dataframe(df.copy(), self.feature_columns)
        normalized_features = standardized_df[self.feature_columns].values


        self.normalized_sequences = []
        self.raw_sequences = []
        self.labels = []

        for i in range(len(raw_features) - look_back_window + 1):
            norm_seq = normalized_features[i:i + look_back_window]
            raw_seq = raw_features[i:i + look_back_window]
            label = labels[i + look_back_window - 1]

            self.normalized_sequences.append(norm_seq)
            self.raw_sequences.append(raw_seq)
            self.labels.append(label)

    def __len__(self):
        return len(self.normalized_sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.normalized_sequences[idx], dtype=torch.float32),
            torch.tensor(self.raw_sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

    def get_dates(self):
        return self.dates[:len(self.normalized_sequences)]

def create_dataset(train_folder=None, val_folder=None, test_folder=None, backtest_file=None, look_back_window=1, feature_columns=None):
    """统一入口：按目录建 train/val/test 三个数据集；给 backtest_file 时改建回测数据集。

    feature_columns 为空则退回默认 5 列价格；传入时需与 --input_size 一致。
    每次调用都会重算一遍加载期除权审计并在结束时打印一行汇总，让"这批价格有没有
    复权、跨没跨除权日"成为可核验的事实，而不是靠记忆。
    """
    reset_adj_audit()

    def load_dataset_from_folder(folder, look_back_window):

        if folder is None:
            return None
        csv_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".csv")]
        return Normal_Dataset(csv_files, look_back_window, feature_columns)

    def load_dataset_from_csv(csv_file, look_back_window):
        if csv_file is None:
            return None
        return Backtest_Dataset(csv_file, look_back_window)

    if backtest_file is not None:
        dataset = load_dataset_from_csv(backtest_file, look_back_window)
        _print_adj_summary()
        return dataset
    else:
        train_dataset = load_dataset_from_folder(train_folder, look_back_window)
        val_dataset = load_dataset_from_folder(val_folder, look_back_window)
        test_dataset = load_dataset_from_folder(test_folder, look_back_window)
        _print_adj_summary()
        return train_dataset, val_dataset, test_dataset

class DTML_Dataset(Dataset):
    def __init__(self, data_folder, look_back_window=7, n_stocks=5):
        self.data_folder = data_folder
        self.seq_len = look_back_window
        self.n_stocks = n_stocks


        self.csv_files = [
            os.path.join(data_folder, f)
            for f in os.listdir(data_folder)
            if f.endswith('.csv')
        ]
        if not self.csv_files:
            raise ValueError(f"No CSV files found in {data_folder}")


        self.processed_data = []
        for csv_file in self.csv_files:
            self._process_single_stock(csv_file)


        np.random.shuffle(self.processed_data)

    def _process_single_stock(self, csv_path):

        df = pd.read_csv(csv_path)


        df['Date'] = pd.to_datetime(df['Date'])
        df['weekday'] = df['Date'].dt.weekday
        df['month'] = df['Date'].dt.month


        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        scaler = StandardScaler()
        df[price_cols] = scaler.fit_transform(df[price_cols])


        features = df[price_cols + ['weekday', 'month']].values
        closes = df['Close'].values
        labels = [int(closes[i] > closes[i - 1]) for i in range(1, len(closes))]
        labels = [0] + labels

        # 分割序列
        for i in range(len(df) - self.seq_len):
            seq_feat = features[i: i + self.seq_len]
            label = labels[i + self.seq_len]
            self.processed_data.append((seq_feat, label))

    def __len__(self):
        return len(self.processed_data) // self.n_stocks

    def __getitem__(self, idx):

        sampled_indices = np.random.choice(
            len(self.processed_data),
            self.n_stocks,
            replace=False
        )


        features, labels = [], []
        for i in sampled_indices:
            feat, lbl = self.processed_data[i]
            features.append(feat)
            labels.append(lbl)


        features = np.stack(features, axis=1)
        return (
            torch.FloatTensor(features),
            torch.FloatTensor(labels)
        )


class SCINet_Dataset(Dataset):
    def __init__(self, data_folder, seq_len, pred_len):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len


        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"NO such file or directory: {data_folder}")


        csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder)
                     if f.endswith('.csv')]


        if not csv_files:
            raise ValueError(f"no such file or directory: {data_folder}")

        self.features, self.labels = self._process_files(csv_files)


        if len(self.features) == 0 or len(self.labels) == 0:
            raise RuntimeError("dataset is empty")

    def _process_files(self, csv_files):
        all_features, all_labels = [], []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df = self._add_time_features(df)


                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close']
                if not all(col in df.columns for col in required_columns):
                    continue


                features, labels = self._generate_sequences(df)
                if features.size(0) == 0 or labels.size(0) == 0:
                    continue

                all_features.append(features)
                all_labels.append(labels)
            except Exception as e:
                continue

        if not all_features or not all_labels:
            raise RuntimeError("failed")

        return torch.cat(all_features), torch.cat(all_labels)

    def _add_time_features(self, df):
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df['day'] = df['Date'].dt.day
            df['weekday'] = df['Date'].dt.weekday
            df['month'] = df['Date'].dt.month
            return df
        except KeyError:
            raise KeyError("No 'Date' column")

    def _generate_sequences(self, df):

        min_required_length = self.seq_len + self.pred_len
        if len(df) < min_required_length:

            return torch.empty(0), torch.empty(0)


        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        scaler = StandardScaler()
        df[price_cols] = scaler.fit_transform(df[price_cols])


        feature_cols = price_cols + ['day', 'weekday', 'month']
        features = torch.tensor(df[feature_cols].values, dtype=torch.float32)


        close_prices = df['Close'].values
        labels = []
        for i in range(len(close_prices) - self.pred_len):
            future_closes = close_prices[i + 1: i + self.pred_len + 1]
            lbls = [1 if future_closes[j] > future_closes[j - 1] else 0
                    for j in range(1, len(future_closes))]
            labels.append(lbls)
        labels = torch.tensor(labels, dtype=torch.float32)


        seq_features, seq_labels = [], []
        for i in range(len(features) - self.seq_len - self.pred_len + 1):
            seq_feat = features[i: i + self.seq_len]
            seq_lbl = labels[i + self.seq_len - 1]
            seq_features.append(seq_feat)
            seq_labels.append(seq_lbl)

        return torch.stack(seq_features), torch.stack(seq_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Backtest_SCINet_Dataset(Dataset):
    def __init__(self, csv_file, seq_len=90, pred_len=7, feature_columns=None, label_column='Label', scaler_csv=None):

        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_columns = feature_columns or ['Open', 'High', 'Low', 'Close', 'Adj Close', 'day', 'weekday', 'month']
        self.label_column = label_column

        df = pd.read_csv(csv_file)

        df['Date'] = pd.to_datetime(df['Date'])
        df['day'] = df['Date'].dt.day
        df['weekday'] = df['Date'].dt.weekday
        df['month'] = df['Date'].dt.month

        self.raw_close_prices = df['Close'].values


        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        # 前视泄漏修复：scaler_csv 传入 train 段 csv 时，mean/std 只在 train 段
        # 拟合，test 段拿来 transform。不传则退回旧行为（整段 test 自己 fit）。
        if scaler_csv:
            train_df = pd.read_csv(scaler_csv)
            for col in price_cols:
                m = float(train_df[col].mean())
                s = float(train_df[col].std())
                df[col] = (df[col] - m) / s
        else:
            scaler = StandardScaler()
            df[price_cols] = scaler.fit_transform(df[price_cols])


        features = df[self.feature_columns].values
        labels = []
        for i in range(len(df) - pred_len):
            future_closes = df['Close'].values[i+1:i+pred_len+1]
            direction = [1 if future_closes[j] > future_closes[j-1] else 0 for j in range(1, len(future_closes))]
            labels.append(direction)
        labels = np.array(labels)


        self.normalized_sequences = []
        self.labels = []

        for i in range(seq_len, len(features) - pred_len + 1):
            start_idx = max(i - seq_len, 0)
            end_idx = i
            seq_feat = features[start_idx:end_idx]
            seq_label = labels[i - 1] if i <= len(labels) else [0]

            self.normalized_sequences.append(seq_feat)
            self.labels.append(seq_label[0] if len(seq_label) > 0 else 0)


        self.normalized_sequences = torch.tensor(self.normalized_sequences, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.normalized_sequences)

    def __getitem__(self, idx):
        return (
            self.normalized_sequences[idx],
            torch.tensor(self.raw_close_prices[idx + self.seq_len], dtype=torch.float32),
            self.labels[idx]
        )

    def get_dates(self):

        df = pd.read_csv(self.csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        return df['Date'].values[self.seq_len:]
def create_dataloader(dataset, batch_size=32, shuffle=True, drop_last=False):

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

import argparse
import torch
import random
import numpy as np
import yaml
import os
import sys
import subprocess
import datetime
from eval_single import eval_single
from eval_multi import eval_multi

# [CPU adaptation] swanlab is optional now. When disabled, monkeypatch its
# functions to no-ops so the training scripts run without an API key or network.
import swanlab as _swanlab

def str2bool(v):
    return str(v).lower() in ('true', '1', 't', 'y', 'yes', 'on')


def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _write_manifest(args):
    """自动创建输出目录，并把"这次实验是怎么跑的"落成 run_manifest.txt。

    为什么需要：实验记录规范要求任何数字都能指回具体命令、commit 和时间。
    以前这些靠手写，漏写就无法分辨两条结果到底差在哪个参数上。

    写入内容：运行时间、git commit（取不到记 unknown）、工作目录、
    完整命令行，以及模型/数据集/窗口/输入维/特征列/标签平滑/种子这七个
    决定口径的键值。
    """
    for folder in (args.model_save_folder, args.figure_save_folder,
                   args.test_result_save_folder, args.history_output):
        if folder:
            os.makedirs(folder, exist_ok=True)

    # 约定：model_save_folder 形如 result/<exp_id>/model_saved/，
    # 其上一级就是实验目录，manifest 放在实验目录根下。
    exp_dir = os.path.dirname(os.path.normpath(args.model_save_folder)) or '.'
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = 'unknown'

    lines = [
        'time: ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'commit: ' + commit,
        'cwd: ' + os.getcwd(),
        'argv: ' + ' '.join(sys.argv),
        'key: model=%s dataset=%s window=%s input_size=%s feature_columns=%s '
        'label_smoothing=%s seed=%s' % (args.model, args.dataset, args.look_back_window,
                                        args.input_size, args.feature_columns,
                                        args.label_smoothing, args.seed),
    ]
    with open(os.path.join(exp_dir, 'run_manifest.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    # [fix] 支持 --config：先从命令行取出配置文件，把其中的键值作为 argparse 默认值，
    # 再由命令行覆盖。这样一条命令即可完整复跑一个实验，避免长参数串被 shell 拆散
    # （此前 BiLSTM+10 维那次就是因为 "Open,High,Low,Close,Adj Close,..." 被按逗号
    # 切开，argparse 报 unrecognized arguments，进程直接退出）。
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--config', default='')
    _known, _ = _pre.parse_known_args()
    _cfg = {}
    if _known.config:
        with open(_known.config, 'r', encoding='utf-8') as _f:
            _cfg = yaml.safe_load(_f) or {}

    parser = argparse.ArgumentParser()


    parser.add_argument('--dataset', default="CSMD50", type=str)
    # [fix] 配置入口：yaml 里的键名与下面的参数名一一对应，命令行同名参数优先覆盖
    parser.add_argument('--config', default=_known.config, help='path to a yaml experiment config')
    # [fix] 原先种子写死 set_seed(37)，无法做多种子复验；改为参数，默认沿用 37 保持可比
    parser.add_argument('--seed', default=37, type=int, help='random seed; keep 37 for comparability')
    parser.add_argument('--train_price_folder', default='./dataset/{}/train/price'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--val_price_folder', default='./dataset/{}/val/price'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--test_price_folder', default='./dataset/{}/test/price'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--train_news_folder', default='./dataset/{}/train/news_embedding'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--val_news_folder', default='./dataset/{}/val/news_embedding'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--test_news_folder', default='./dataset/{}/test/news_embedding'.format(parser.get_default('dataset')),
                        type=str)
    parser.add_argument('--use_news', default=True, type=str2bool, help='use news or not')
    parser.add_argument('--model', default="HAN", type=str) # lstm, alstm, bi_lstm, adv_lstm, dtml, scinet, StockNet, HAN, PEN
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--input_size', default=5, type=int)
    # [fix] 纯价格模型过去把特征列写死成 5 列价格，做"加技术指标 / 换 Volume"消融时
    # 只能临时改代码，无法复现。这里改为显式传入，必须与 --input_size 一致。
    parser.add_argument('--feature_columns', default='',
                        help='comma separated feature columns in csv, e.g. "Open,High,Low,Close,Adj Close,Return,RSI6,MACD_Hist,BIAS5,Amplitude"')
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--attention_size', default=128, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--look_back_window', default=7, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--useGPU', default=True, type=str2bool)
    parser.add_argument('--GPU_ID', default=1, type=int)
    parser.add_argument('--batch_first', default=True, type=str2bool)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--label_smoothing', default=True, type=str2bool, help='enable label smoothing; set False for a clean baseline')
    parser.add_argument('--label_smoothing_epsilon', default=0.1, type=float, help='label smoothing epsilon')
    parser.add_argument('--model_save_folder', default='./result/{}/model_saved/'.format(parser.get_default('dataset')))
    parser.add_argument('--figure_save_folder', default='./result/{}/figure/'.format(parser.get_default('dataset')))
    parser.add_argument('--test_result_save_folder',default='./result/{}/test_result/'.format(parser.get_default('dataset')))
    parser.add_argument('--swanlab_api', default='')
    parser.add_argument( '--swanlab_project', default='framework')
    parser.add_argument( '--swanlab_workspace', default='')
    parser.add_argument('--use_swanlab', default=False, type=str2bool, help='[CPU adaptation] enable swanlab logging or not')
    parser.add_argument('--epsilon', default=0.1, type=float, help='epsilon for adversarial lstm')
    parser.add_argument('--perturbation_size', default=0.1, type=float, help='perturbation size for adversarial lstm')
    parser.add_argument('--n_stocks', default=5, type=int, help='stocks number for DTML')
    parser.add_argument('--SCINet_Layers', default=3, type=int, help='SCINet layers number')
    parser.add_argument('--seq_len', default=32, type=int, help='input sequence length for SCINet')
    parser.add_argument('--pred_len', default=5, type=int, help='prediction length for SCINet')
    parser.add_argument('--history_output', default='./result/{}/history_output/'.format(parser.get_default('dataset')), help='storage history outputs for StockNet')
    parser.add_argument('--pretrained_model', type=str, default='yiyanghkust/finbert-pretrain', help='pretrained financial model for PEN')
    parser.add_argument('--train_start_date', type=str, default='2021-01-01', help='train start date for HAN dataset')
    parser.add_argument('--train_end_date', type=str, default='2024-03-14', help='train end date for HAN dataset')
    parser.add_argument('--dev_start_date', type=str, default='2024-03-15', help='dev start date for HAN dataset')
    parser.add_argument('--dev_end_date', type=str, default='2024-08-07', help='dev end date for HAN dataset')
    # [fix] HAN_Dataset_Prepare 引用 val_start_date/val_end_date，原 run.py 未定义（会 AttributeError）；沿用 dev 值
    parser.add_argument('--val_start_date', type=str, default='2024-03-15', help='val start date for HAN dataset')
    parser.add_argument('--val_end_date', type=str, default='2024-08-07', help='val end date for HAN dataset')
    parser.add_argument('--test_start_date', type=str, default='2024-08-08', help='test start date for HAN dataset')
    parser.add_argument('--test_end_date', type=str, default='2024-12-31', help='test end date for HAN dataset')
    parser.add_argument('--max_num_tweets_len', type=int, default=20, help='max number of tweets for HAN')
    parser.add_argument('--max_num_tokens_len', type=int, default=30, help='max number of t okens for HAN')
    parser.add_argument('--days', type=int, default=5, help='days for HAN')
    parser.add_argument('--bert_dim', type=int, default= 768, help='bert dim for HAN')
    # [CPU adaptation] missing args used by multi_exp (HAN x/y paths) and PEN
    parser.add_argument('--train_x_path', default='./dataset/{}/train_x.csv'.format(parser.get_default('dataset')), type=str)
    parser.add_argument('--train_y_path', default='./dataset/{}/train_y.csv'.format(parser.get_default('dataset')), type=str)
    parser.add_argument('--val_x_path', default='./dataset/{}/val_x.csv'.format(parser.get_default('dataset')), type=str)
    parser.add_argument('--val_y_path', default='./dataset/{}/val_y.csv'.format(parser.get_default('dataset')), type=str)
    parser.add_argument('--test_x_path', default='./dataset/{}/test_x.csv'.format(parser.get_default('dataset')), type=str)
    parser.add_argument('--test_y_path', default='./dataset/{}/test_y.csv'.format(parser.get_default('dataset')), type=str)
    parser.add_argument('--max_num_tweets', type=int, default=20, help='max number of tweets for PEN')
    parser.add_argument('--max_num_tokens', type=int, default=30, help='max number of tokens for PEN')



    # [fix] 配置文件里的键直接作为默认值；命令行同名字段仍然优先
    if _cfg:
        parser.set_defaults(**_cfg)

    args = parser.parse_args()

    # [fix] 种子从参数读，默认 37 与历史结果保持一致；做多种子复验时传 --seed 即可
    set_seed(args.seed)

    # [fix] 上面这些默认路径在 add_argument 时就用 parser.get_default('dataset')
    # 拼死了，所以传 --dataset CSMD50_feat 时 price/news 路径仍指向 dataset/CSMD50，
    # 结果 10 维特征那次直接 KeyError: 'Return'（脚本跑 8 分钟才炸）。
    # 这里在解析后按实际 dataset 重写一次，覆盖 dataset/ 与 result/ 两类默认路径，
    # 否则 --config 里只写 dataset 时，权重和结果仍会写进默认目录。
    _default_ds = 'CSMD50'
    if args.dataset != _default_ds:
        for _k, _v in list(vars(args).items()):
            if not isinstance(_v, str):
                continue
            _norm = _v.replace('\\', '/')
            if ('dataset/%s/' % _default_ds) in _norm or ('result/%s/' % _default_ds) in _norm:
                setattr(args, _k,
                        _norm.replace('dataset/%s/' % _default_ds, 'dataset/%s/' % args.dataset)
                             .replace('result/%s/' % _default_ds, 'result/%s/' % args.dataset))

    # [fix] 把最终生效的参数打进日志：此前日志不回显 argparse，
    # 事后无法复核 --look_back_window / --label_smoothing 是否真的传进去了，
    # 因此误跑过一整轮（三个模型跑成了 LS 版）。这行是那次事故的补丁。
    print("[args] " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())), flush=True)

    # [feat] 实验目录自动创建 + run_manifest 自动落盘。
    # 原脚本要求手工先建 model_saved/figure/test_result，漏建就静默丢结果；
    # manifest 原来也是手写，事后无法证明"这条结果是用哪条命令跑出来的"。
    _write_manifest(args)

    if args.use_swanlab:
        _swanlab.login(api_key=args.swanlab_api, save=True)
        # Initialize a new SwanLab run to track this script
        _swanlab.init(
            # configurate your own infor
            project=args.swanlab_project,
            workspace=args.swanlab_workspace,
            experiment_name=args.model,
            config={
                "learning_rate": args.lr,
                "model": args.model,
                "dataset": args.dataset,
                "epochs": args.epochs
            },
        )
    else:
        # [CPU adaptation] disable swanlab (no API key / network required)
        _swanlab.login = lambda *a, **k: None
        _swanlab.init = lambda *a, **k: None
        _swanlab.log = lambda *a, **k: None

    if args.use_news == False:
        eval_single(args)
    else:
        eval_multi(args)

"""按多个随机种子重复同一个配置，用来估计单次运行的标准差。

为什么需要：所有"改进 +Xpp"的判断都必须先知道噪声有多大。单次运行之间
的差别可能来自种子（初始化、dropout、数据顺序），不先测这个标准差，
就无法说清 +1.18pp 是改进还是运气。

用法：
    python scripts/run_seeds.py --config configs/base_alstm.yaml --seeds 43,101,2024
    python scripts/run_seeds.py --configs "configs/base_alstm.yaml configs/exp_feat_alstm.yaml" --seeds 43,101,2024

每个种子跑在自己的目录里：<原实验名>_s<种子>/，跑完自动读取 test_result
并打印均值与标准差。子进程用当前 python 解释器，日志写进各自目录的
run_stdout.txt，崩溃时不会污染其他种子。
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

RESULT_FIELDS = {
    'Test Accuracy': 'acc',
    'Test MCC': 'mcc',
    'Test F1 Score': 'f1',
}


def parse_result(path):
    out = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            for label, key in RESULT_FIELDS.items():
                if line.startswith(label):
                    out[key] = float(line.split(':', 1)[-1].strip())
    return out


def mean_std(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5


def main():
    ap = argparse.ArgumentParser(description='同一配置多种子重复，估计标准差')
    # 可重复传入，如 --config a.yaml --config b.yaml；不要用带空格的单个参数，
    # 会被 Start-Process / shell 按空格拆开（本次已踩坑）
    ap.add_argument('--config', action='append', default=[])
    ap.add_argument('--seeds', required=True, help='逗号分隔，如 43,101,2024')
    ap.add_argument('--dry', action='store_true', help='只打印将要执行的命令')
    # 断点续跑：已有可解析的 test_result 就跳过，中断后重跑同一条命令即可继续
    ap.add_argument('--skip_existing', action='store_true',
                    help='已有有效 test_result 的种子直接跳过')
    args = ap.parse_args()

    configs = [c for c in args.config if c]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    if not configs or not seeds:
        print('需要至少一个配置和一个种子')
        return 1

    collected = {}
    # 每跑完一个就落一份增量汇总，中断也不会丢掉已算出的均值/标准差
    summary_path = os.path.join('result', 'seeds_summary.json')

    def flush_summary():
        data = {}
        for exp, rows in collected.items():
            data[exp] = {
                'seeds': [s for s, _ in rows],
                'runs': [m for _, m in rows],
                'stats': {k: dict(zip(('mean', 'std'),
                                      mean_std([m[k] for _, m in rows if k in m])))
                          for k in ('acc', 'mcc', 'f1')},
            }
        os.makedirs('result', exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    for cfg in configs:
        exp = os.path.basename(cfg).replace('.yaml', '')
        for seed in seeds:
            out_root = 'result/%s_s%d' % (exp, seed)
            existing = glob.glob(os.path.join(out_root, 'test_result', '*.txt'))
            if args.skip_existing and existing:
                m = parse_result(existing[0])
                if 'acc' in m:
                    collected.setdefault(exp, []).append((seed, m))
                    print('  seed=%d 已有结果，跳过 (acc=%.4f)' % (seed, m['acc']), flush=True)
                    continue
            cmd = [sys.executable, 'run.py', '--config', cfg,
                   '--seed', str(seed),
                   '--model_save_folder', out_root + '/model_saved/',
                   '--figure_save_folder', out_root + '/figure/',
                   '--test_result_save_folder', out_root + '/test_result/']
            print('[run] ' + ' '.join(cmd), flush=True)
            if args.dry:
                continue
            os.makedirs(out_root, exist_ok=True)
            with open(os.path.join(out_root, 'run_stdout.txt'), 'w',
                      encoding='utf-8') as log:
                code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            if code != 0:
                print('  失败，退出码 %d，见 %s/run_stdout.txt' % (code, out_root))
                continue
            files = glob.glob(os.path.join(out_root, 'test_result', '*.txt'))
            if not files:
                print('  没有 test_result，视为无效')
                continue
            m = parse_result(files[0])
            collected.setdefault(exp, []).append((seed, m))
            flush_summary()
            print('  seed=%d acc=%.4f mcc=%.4f f1=%.4f' % (
                seed, m.get('acc', float('nan')), m.get('mcc', float('nan')),
                m.get('f1', float('nan'))), flush=True)

    print('\n== 汇总 ==')
    for exp, rows in collected.items():
        for key in ('acc', 'mcc', 'f1'):
            vals = [m[key] for _, m in rows if key in m]
            mean, std = mean_std(vals)
            if mean is not None:
                print('%s %s: mean=%.4f std=%.4f n=%d' % (exp, key, mean, std, len(vals)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

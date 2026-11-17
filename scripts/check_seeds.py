"""查看多种子实验的进度：哪些跑完了、哪些还在跑、跑到了第几个 epoch。

用法：
    python scripts/check_seeds.py
    python scripts/check_seeds.py --configs base_alstm exp_feat_alstm

判据：有 test_result 且能解析出 acc 才算完成；只有 run_stdout.txt 说明中途断过。
"""

import argparse
import glob
import json
import os
import re


RESULT_FIELDS = {'Test Accuracy': 'acc', 'Test MCC': 'mcc', 'Test F1 Score': 'f1'}
# 脚本可能从任意工作目录被调用，所以按脚本位置定位项目根，不依赖 cwd
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_result(path):
    out = {}
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            for label, key in RESULT_FIELDS.items():
                if line.startswith(label):
                    out[key] = float(line.split(':', 1)[-1].strip())
    return out


def last_epoch(log_path):
    """从日志里取最后一个 'Epoch N/300'，顺便取最后一行的进度百分比。"""
    try:
        raw = open(log_path, 'r', encoding='utf-8', errors='replace').read()
    except OSError:
        return None
    hits = re.findall(r'Epoch (\d+)/(\d+)', raw)
    return (int(hits[-1][0]), int(hits[-1][1])) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='*', default=['base_alstm', 'exp_feat_alstm'])
    args = ap.parse_args()

    rows = []
    for exp in args.configs:
        for d in sorted(glob.glob(os.path.join(BASE, 'result', '%s_s*' % exp))):
            seed = int(os.path.basename(d).rsplit('_s', 1)[1])
            tr = glob.glob(os.path.join(d, 'test_result', '*.txt'))
            m = parse_result(tr[0]) if tr else {}
            ep = last_epoch(os.path.join(d, 'run_stdout.txt'))
            rows.append(dict(exp=exp, seed=seed,
                             acc=m.get('acc'), mcc=m.get('mcc'), f1=m.get('f1'),
                             epoch=ep, done=bool(m.get('acc'))))

    print('%-14s %5s %9s %7s %7s %10s %s' % ('exp', 'seed', 'acc', 'mcc', 'f1', 'epoch', 'state'))
    for r in rows:
        ep = '%d/%d' % r['epoch'] if r['epoch'] else '-'
        print('%-14s %5d %9s %7s %7s %10s %s' % (
            r['exp'], r['seed'],
            '%.4f' % r['acc'] if r['acc'] else '-',
            '%.4f' % r['mcc'] if r['mcc'] else '-',
            '%.4f' % r['f1'] if r['f1'] else '-',
            ep, 'done' if r['done'] else 'running/pending'))

    summ = os.path.join(BASE, 'result', 'seeds_summary.json')
    if os.path.exists(summ):
        data = json.load(open(summ, encoding='utf-8'))
        print('\n== 增量汇总（每跑完一组就会更新）==')
        for exp, v in data.items():
            for key in ('acc', 'mcc', 'f1'):
                st = v['stats'].get(key) or {}
                if st.get('mean') is not None:
                    print('%s %s: mean=%.4f std=%.4f n=%d' % (
                        exp, key, st['mean'], st['std'], len(v['runs'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

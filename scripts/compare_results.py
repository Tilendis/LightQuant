"""把 result/ 下各实验目录汇总成一张对照表。

解决的问题：此前多个实验的结果分散在 result/<实验名>/test_result/*.txt 里，
横向比较时只能人工抄表，既慢又容易抄错口径（例如把 1 天窗口的数字和 5 天
窗口的数字放在同一行）。这个脚本只做"读取 + 并表"，不重算任何指标。

用法：
    python scripts/compare_results.py                     # 打印 markdown 表
    python scripts/compare_results.py --csv out.csv       # 另存 csv
    python scripts/compare_results.py --root result

判定规则（对应《实验记录规范》）：
    1. 没有 test_result/*.txt 的目录，标记为"无结果"，不进主表数值区；
    2. run_log.txt 为空或缺失，标记"参数未回显"，该行数值不可信；
    3. input_size 与 feature_columns 的列数不一致，标记"维度冲突"；
    4. run_err.txt 非空，标记"有报错"。

这些标记只是提示，脚本不会替你判断哪个实验能用——口径要人来定。
"""

import argparse
import csv
import json
import os
import re

# test_result/*.txt 由 eval_single/test.py 写出，固定四行
FIELD_MAP = {
    'Test Loss': 'loss',
    'Test Accuracy': 'acc',
    'Test MCC': 'mcc',
    'Test F1 Score': 'f1',
}

# 日志首行的 [args] 回显中，值可能含空格（如 "Adj Close"），
# 因此按 " key=" 的边界切分，而不是按空格切分。
ARG_PATTERN = re.compile(r'(\w+)=(.*?)(?= \w+=|\Z)')


def read_lines(path, limit=1):
    """按 BOM 自适应编码读取若干行。

    历史日志有两种：早期用 PowerShell 重定向写出的 UTF-16LE（含 BOM），
    后来用 Start-Process 直接重定向 python 输出的是 UTF-8。不按 BOM 判断
    就会读出满屏 \\x00，导致"参数未回显"被误判。
    """
    with open(path, 'rb') as f:
        head = f.read(4)
    if head.startswith(b'\xff\xfe'):
        enc = 'utf-16'
    elif head.startswith(b'\xef\xbb\xbf'):
        enc = 'utf-8-sig'
    else:
        enc = 'utf-8'
    out = []
    with open(path, 'r', encoding=enc, errors='replace') as f:
        for _ in range(limit):
            line = f.readline()
            if not line:
                break
            out.append(line)
    return out


def read_args_echo(path):
    """读取 run_log.txt 首行的 [args] 回显，返回 dict；没有则返回 None。"""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    lines = read_lines(path, limit=1)
    if not lines:
        return None
    first = lines[0]
    if '[args]' not in first:
        return None
    body = first.split('[args]', 1)[1].strip()
    return {k: v for k, v in ARG_PATTERN.findall(body)}


def read_test_result(folder):
    """读取 test_result/ 下所有 txt，返回 {model: {acc, mcc, f1, loss}}。"""
    out = {}
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not name.endswith('.txt'):
            continue
        model = name[:-4]
        metrics = {}
        with open(os.path.join(folder, name), 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                for label, key in FIELD_MAP.items():
                    if line.startswith(label):
                        raw = line.split(':', 1)[-1].strip()
                        try:
                            metrics[key] = float(raw)
                        except ValueError:
                            metrics[key] = None
        out[model] = metrics
    return out


def collect(exp_dir):
    """把一个实验目录读成一条记录（含若干模型行）。"""
    log_path = os.path.join(exp_dir, 'run_log.txt')
    err_path = os.path.join(exp_dir, 'run_err.txt')
    args = read_args_echo(log_path)
    results = read_test_result(os.path.join(exp_dir, 'test_result'))

    flags = []
    if not results:
        flags.append('无结果')
    if args is None:
        flags.append('参数未回显')
    # 只认真正的异常：tqdm 进度条也会写 stderr，不能一律判为报错
    if os.path.exists(err_path):
        with open(err_path, 'r', encoding='utf-8', errors='replace') as f:
            err_text = f.read()
        if 'Traceback' in err_text or 'Error:' in err_text:
            flags.append('有报错')
    if args:
        cols = [c for c in args.get('feature_columns', '').split(',') if c.strip()]
        if cols and args.get('input_size') and len(cols) != int(args['input_size']):
            flags.append('维度冲突(%d列/%s维)' % (len(cols), args['input_size']))

    rows = []
    for model, m in results.items():
        rows.append({
            'exp': os.path.basename(exp_dir),
            'model': model,
            'acc': m.get('acc'),
            'mcc': m.get('mcc'),
            'f1': m.get('f1'),
            'window': args.get('look_back_window', '?') if args else '?',
            'input_size': args.get('input_size', '?') if args else '?',
            'label_smoothing': args.get('label_smoothing', '?') if args else '?',
            'seed': args.get('seed', '?') if args else '?',
            'flags': '; '.join(flags),
        })
    if not rows:
        rows.append({
            'exp': os.path.basename(exp_dir), 'model': '-', 'acc': None,
            'mcc': None, 'f1': None, 'window': '?', 'input_size': '?',
            'label_smoothing': '?', 'seed': '?', 'flags': '; '.join(flags),
        })
    return rows


def pct(v):
    """ACC/MCC/F1 统一显示成百分数，保留两位小数；缺失显示 '-'。"""
    return '-' if v is None else '%.2f%%' % (v * 100)


def raw(v):
    return '-' if v is None else '%.4f' % v


def main():
    ap = argparse.ArgumentParser(description='汇总 result/ 下的实验结果')
    ap.add_argument('--root', default='result')
    ap.add_argument('--csv', default='')
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print('找不到目录：%s（请在仓库根目录运行）' % args.root)
        return 1

    rows = []
    for name in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, name)
        if os.path.isdir(d):
            rows.extend(collect(d))

    header = ['实验目录', '模型', 'ACC', 'MCC', 'F1', '窗口', '输入维', 'LS', 'seed', '备注']
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] * len(header)) + '|')
    for r in rows:
        print('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['exp'], r['model'], pct(r['acc']), raw(r['mcc']), pct(r['f1']),
            r['window'], r['input_size'], r['label_smoothing'], r['seed'], r['flags']))

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([r['exp'], r['model'], r['acc'], r['mcc'], r['f1'],
                            r['window'], r['input_size'], r['label_smoothing'],
                            r['seed'], r['flags']])
        print('\n已写出 %s（%d 行）' % (args.csv, len(rows)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

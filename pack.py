# -*- coding: utf-8 -*-
"""把项目纯源码打包成单文件，给外部 AI 做 Code Review 用。
排除：依赖包、数据集、模型权重、回测结果、数据库、编译产物。
"""
import os, sys

ROOT = r"D:\03_Study\GraduationProject\LightQuant"
OUT = os.path.join(ROOT, "project_snapshot.txt")

INCLUDE_EXTS = {'.py', '.js', '.html', '.sql', '.yaml', '.yml', '.json', '.css'}
EXCLUDE_DIRS = {
    'node_modules', '.git', '.venv', 'venv', '__pycache__',
    'dist', 'build', '.idea', '.vscode',
    'dataset', 'data', 'logs',
    'result_v2',          # 回测结果不是代码
    'screenshots',        # 截图不打
    '.pytest_cache',
    'vendor',             # 第三方前端库（echarts.min.js 1MB）
    'model_cache',        # BERT 模型缓存
    'factor_lib',         # Qlib 因子库，不是自己写的
}
EXCLUDE_FILES = {
    'package-lock.json', 'yarn.lock', 'pnpm-lock.yaml',
    'pack.py', 'project_snapshot.txt',
    'lightquant.db', 'lightquant.db-journal',
    'exp_orch_agent_nolag_bak.json',
    'price_audit.json', 'price_jump_scan.json', 'price_source_check.json',
}
# 只打这些目录下的 .json（排除回测结果里的 summary.json）
JSON_WHITELIST_DIRS = {'docs', 'app'}

def should_include(rel_path, fname):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in INCLUDE_EXTS:
        return False
    if fname in EXCLUDE_FILES:
        return False
    if ext == '.json':
        # 只保留 docs/ 和 app/ 下的 json，排除 backtest/result_v2 等
        parts = rel_path.replace('\\', '/').split('/')
        if parts[0] not in JSON_WHITELIST_DIRS:
            return False
    return True

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    tree_lines = ["# 项目目录结构", "```text"]
    body_lines = []
    file_count = 0

    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel_root = os.path.relpath(root, ROOT)
        if rel_root == '.':
            rel_root = ''
        level = rel_root.count(os.sep) if rel_root else 0
        indent = '  ' * level
        dirname = os.path.basename(root) if rel_root else '.'
        tree_lines.append(f"{indent}{dirname}/")
        for f in sorted(files):
            rel_path = os.path.join(rel_root, f) if rel_root else f
            if not should_include(rel_path, f):
                continue
            tree_lines.append(f"{indent}  {f}")
            full = os.path.join(root, f)
            try:
                with open(full, 'r', encoding='utf-8') as fp:
                    content = fp.read()
                body_lines.append(f"\n{'='*60}\nFILE: {rel_path}\n{'='*60}\n{content}")
                file_count += 1
            except Exception as e:
                body_lines.append(f"\nFILE: {rel_path} [读取跳过: {e}]")

    tree_lines.append("```\n")
    out_text = '\n'.join(tree_lines) + '\n' + '\n'.join(body_lines)
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(out_text)

    size_kb = os.path.getsize(OUT) / 1024
    print(f"打包完成: {OUT}")
    print(f"文件数: {file_count}")
    print(f"大小: {size_kb:.1f} KB")

if __name__ == '__main__':
    main()

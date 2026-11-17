# -*- coding: utf-8 -*-
"""pytest 公共配置：让 engine_v2 / audit_prices 能被直接 import。"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "backtest"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

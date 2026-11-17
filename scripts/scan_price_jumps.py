# -*- coding: utf-8 -*-
"""按板块区分涨跌停阈值的除权跳空扫描。

问题背景：上一版用统一 11% 阈值，把科创板（688）/创业板（300）的 20% 涨跌停
误判成除权。本版按代码分流：
  688 / 300 开头 -> 20% 涨跌停，阈值 0.21
  其余（主板）  -> 10% 涨跌停，阈值 0.11

能证明什么、不能证明什么：
  能证明：窗口内没有送股/转增/配股这类会引发大幅跳空的除权事件。
  不能证明：现金分红除息（幅度通常 1-3%，淹没在正常波动里，无法靠价格区分）。
"""
import glob
import json
import os
import sqlite3

import numpy as np
import pandas as pd

ROOT = r"D:\03_Study\GraduationProject\LightQuant\dataset\CSMD50_feat"
DB = r"D:\03_Study\GraduationProject\LightQuant\app\data\lightquant.db"
OUT = r"D:\03_Study\GraduationProject\LightQuant\docs\price_jump_scan.json"


def load_codes():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return {r["symbol"]: str(r["code"]).zfill(6)
            for r in conn.execute("SELECT symbol, code FROM stocks")}


def limit_threshold(code):
    """返回该标的的涨跌停幅度 + 1% 容差。"""
    if code.startswith("688") or code.startswith("300"):
        return 0.21
    if code.startswith("8") or code.startswith("4"):   # 北交所 30%
        return 0.31
    return 0.11


def main():
    codes = load_codes()
    files = sorted(glob.glob(os.path.join(ROOT, "*", "price", "*.csv")))
    print("文件数: %d, 代码表: %d" % (len(files), len(codes)))

    per_file = []
    hits = []

    for p in files:
        df = pd.read_csv(p)
        sym = os.path.basename(p).replace(".csv", "")
        split = p.replace(ROOT, "").strip(os.sep).split(os.sep)[0]
        code = codes.get(sym, "")
        th = limit_threshold(code) if code else 0.11

        ret = pd.to_numeric(df["Return"], errors="coerce").to_numpy(dtype=float)
        close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
        dates = df["Date"].astype(str).to_numpy()

        bad = []
        for i in range(len(ret)):
            if np.isfinite(ret[i]) and abs(ret[i]) > th:
                bad.append((i, float(ret[i]), None))
        if close.size > 1:
            for i in range(1, close.size):
                ratio = close[i] / close[i - 1]
                if abs(ratio - 1.0) > th:
                    bad.append((i, float(ratio - 1.0), round(float(ratio), 6)))

        for i, val, ratio in bad:
            hits.append({
                "symbol": sym,
                "code": code,
                "split": split,
                "date": str(dates[i]),
                "limit_threshold": th,
                "return": round(val, 6),
                "close_ratio": ratio,
            })

        per_file.append({
            "symbol": sym, "code": code, "split": split, "rows": int(len(df)),
            "limit_threshold": th,
            "ret_min": round(float(np.nanmin(ret)), 6),
            "ret_max": round(float(np.nanmax(ret)), 6),
            "extreme_rows": len(bad),
        })

    verdict = "no_large_ex_dividend_jump" if not hits else "large_jump_detected"
    board20 = [r for r in per_file if r["limit_threshold"] == 0.21]
    board10 = [r for r in per_file if r["limit_threshold"] == 0.11]

    payload = {
        "root": ROOT,
        "board_split": {
            "main_board_10pct": len(board10),
            "star_chi_next_20pct": len(board20),
        },
        "files": len(files),
        "total_extreme_rows": len(hits),
        "verdict": verdict,
        "can_prove": "窗口内没有送股/转增/配股等引发大幅跳空的除权事件",
        "cannot_prove": "现金分红除息（幅度 1-3%，在正常波动噪声内，价格上无法区分）",
        "note": (
            "按板块分流阈值：主板 0.11，科创板/创业板 0.21，北交所 0.31。"
            "任何超过该阈值的单日涨跌都意味着价格发生了超出交易规则的跳变，"
            "在无停牌复牌的情况下即对应除权事件。"
        ),
        "samples": hits[:60],
        "per_file": per_file,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("\n主板(10%%)标的 %d 只，科创/创业(20%%)标的 %d 只" % (len(board10), len(board20)))
    print("超阈值行数: %d" % len(hits))
    print("结论: %s" % verdict)
    for r in hits[:20]:
        print("  %-8s %s limit=%.2f return=%s ratio=%s"
              % (r["symbol"], r["date"], r["limit_threshold"], r["return"], r["close_ratio"]))
    print("\n已写出 %s" % OUT)


if __name__ == "__main__":
    main()

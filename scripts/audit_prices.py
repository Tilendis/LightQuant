# -*- coding: utf-8 -*-
"""价格复权审计：扫描行情 CSV，检查 Adj Close 是否真的做过除权调整。

用途：
  1. 判断"回测用未复权价"是代码缺陷还是数据源限制
  2. 找出因子跳变日（疑似除权日），供人工核对
  3. 产出一份可入库、可引用的审计结果

用法：
    python scripts/audit_prices.py
    python scripts/audit_prices.py --root dataset/CSMD50_feat --out docs/price_audit.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# 因子相邻两日相对变化超过这个比例，就记为一次疑似除权
FACTOR_JUMP_TH = 0.001


def scan_file(path, jump_th=FACTOR_JUMP_TH):
    df = pd.read_csv(path)
    cols = [c.strip().strip('"') for c in df.columns]
    df.columns = cols
    sym = os.path.splitext(os.path.basename(path))[0]

    rec = {"symbol": sym, "rows": int(len(df))}
    if "Close" not in cols:
        rec["status"] = "no_close_column"
        return rec

    close = df["Close"].astype(float).values
    rec["close_min"] = float(np.nanmin(close))
    rec["close_max"] = float(np.nanmax(close))

    if "Adj Close" not in cols:
        rec["status"] = "no_adj_column"
        rec["adjusted"] = False
        return rec

    adj = df["Adj Close"].astype(float).values
    factor = adj / close

    n_bad = int(np.sum(~np.isfinite(factor)) + np.sum(factor <= 0))
    # 用 np.allclose 而不是逐点相等：浮点列从 CSV 读回来可能有末位误差
    identical = bool(np.allclose(adj, close, rtol=0, atol=1e-9))

    jumps = []
    for i in range(1, len(factor)):
        if factor[i - 1] > 0 and abs(factor[i] / factor[i - 1] - 1.0) > jump_th:
            date = str(df.iloc[i]["Date"]) if "Date" in cols else str(i)
            jumps.append({"date": date, "factor": float(factor[i])})

    rec.update({
        "status": "ok",
        "adjusted": not identical,
        "identical_to_close": identical,
        "factor_min": float(np.nanmin(factor)),
        "factor_max": float(np.nanmax(factor)),
        "bad_factor_points": n_bad,
        "jump_count": len(jumps),
        "jumps": jumps[:20],
    })
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/CSMD50_feat")
    ap.add_argument("--out", default="docs/price_audit.json")
    ap.add_argument("--jump_th", type=float, default=FACTOR_JUMP_TH)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "*", "price", "*.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(args.root, "price", "*.csv")))
    if not files:
        raise SystemExit("未找到行情 CSV：%s" % args.root)

    recs = [scan_file(f, args.jump_th) for f in files]

    ok = [r for r in recs if r.get("status") == "ok"]
    adjusted = [r for r in ok if r.get("adjusted")]
    identical = [r for r in ok if r.get("identical_to_close")]
    total_jumps = sum(r.get("jump_count", 0) for r in ok)

    payload = {
        "root": os.path.abspath(args.root),
        "jump_threshold": args.jump_th,
        "files": len(files),
        "ok_files": len(ok),
        "adjusted_files": len(adjusted),
        "identical_files": len(identical),
        "total_factor_jumps": total_jumps,
        "verdict": "data_not_adjusted" if not adjusted else "data_has_adjustment",
        "per_file": recs,
        "note": ("verdict=data_not_adjusted 表示行情 CSV 的 Adj Close 与 Close 逐行相等，"
                 "即数据源未提供复权因子；此时引擎按未复权价回测是数据限制，不是代码缺陷。"
                 " 引擎已按 Adj Close/Close 的比值统一缩放 OHLC，数据一旦带真实因子会自动生效。"),
    }

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("扫描 %d 个文件" % len(files))
    print("  Adj Close 与 Close 完全相等 : %d" % len(identical))
    print("  Adj Close 有真实调整       : %d" % len(adjusted))
    print("  疑似除权跳变总数           : %d" % total_jumps)
    print("  结论                       : %s" % payload["verdict"])
    if adjusted:
        print("\n有调整的文件：")
        for r in adjusted[:10]:
            print("  %-10s factor %.4f~%.4f  jumps=%d"
                  % (r["symbol"], r["factor_min"], r["factor_max"], r["jump_count"]))
    if total_jumps:
        print("\n跳变样例：")
        for r in ok:
            for j in r.get("jumps", [])[:3]:
                print("  %-10s %s factor=%.6f" % (r["symbol"], j["date"], j["factor"]))

    print("\n已写出 %s" % out)


if __name__ == "__main__":
    main()

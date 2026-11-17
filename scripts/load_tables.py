"""把行情与新闻灌进应用层库：prices / news。

为什么要灌：四角色业务流、K 线叠加图、策略活跃度看板都直接查这两张表。
不灌的话前端只能读 CSV，演示时等于绕开了系统本身。

口径：
- 默认只灌测试期（2024-08-08 ~ 2024-12-31），与回测、研判的时间范围一致
- 价格目录默认 `dataset/CSMD50/test/price`；要灌验证期 / 训练期，用 `--price_dir` 显式指定
  （例如 `dataset/CSMD50_feat/val/price`），否则日期过滤会把所有行滤掉、prices 写 0 行
- 新闻没有真实 title 字段，用正文前 60 字代替，并在 source 标注来源
- symbol 用中文简称，与 dataset 目录名、judgements 表完全一致

用法：
    python scripts/load_tables.py
    python scripts/load_tables.py --start 2024-03-15 --end 2024-08-07 ^
        --price_dir dataset/CSMD50_feat/val/price
"""

import argparse
import glob
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

PRICE_DIR = os.path.join(ROOT, "dataset", "CSMD50", "test", "price")
NEWS_ROOT = os.path.join(ROOT, "dataset", "CSMD50", "news")


def load_prices(conn, start, end, price_dir=None):
    price_dir = price_dir or PRICE_DIR
    n = 0
    for f in sorted(glob.glob(os.path.join(price_dir, "*.csv"))):
        symbol = os.path.basename(f)[:-4]
        df = pd.read_csv(f)
        df["Date"] = df["Date"].astype(str)
        df = df[(df["Date"] >= start) & (df["Date"] <= end)]
        rows = []
        for _, r in df.iterrows():
            rows.append((
                symbol, r["Date"], float(r["Open"]), float(r["High"]),
                float(r["Low"]), float(r["Close"]), float(r.get("Adj Close", r["Close"])),
                float(r.get("Volume", 0) or 0),
                int(r["Label"]) if pd.notna(r.get("Label")) else None,
            ))
        conn.executemany(
            """INSERT INTO prices (symbol, date, open, high, low, close, adj_close, volume, label)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT (symbol, date) DO UPDATE SET
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, adj_close=excluded.adj_close,
                 volume=excluded.volume, label=excluded.label""", rows)
        n += len(rows)
        conn.commit()
    return n


def load_news(conn, start, end):
    n, files = 0, 0
    for sym in sorted(os.listdir(NEWS_ROOT)):
        d = os.path.join(NEWS_ROOT, sym)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
            date = os.path.basename(f)[:-4]
            if not (start <= date <= end):
                continue
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "text" not in df.columns:
                continue
            files += 1
            for t in df["text"].dropna():
                s = str(t).strip()
                if not s:
                    continue
                conn.execute(
                    """INSERT INTO news (symbol, date, title, content, source)
                       VALUES (?,?,?,?,?)
                       ON CONFLICT (symbol, date, title) DO NOTHING""",
                    (sym, date, s[:60], s[:400], "CSMD50"))
                n += 1
            conn.commit()
    return n, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-08-08")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--price_dir", default=None,
                    help="价格 CSV 目录；默认 dataset/CSMD50/test/price")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    conn = db.connect(args.db)
    db.init_db(conn=conn)

    p = load_prices(conn, args.start, args.end, args.price_dir)
    print("prices 写入 %d 行" % p)
    n, files = load_news(conn, args.start, args.end)
    print("news 写入 %d 条（来自 %d 个文件）" % (n, files))

    for t in ("stocks", "prices", "news", "judgements"):
        c = conn.execute("SELECT COUNT(*) c FROM %s" % t).fetchone()["c"]
        print("  %-12s %d" % (t, c))
    conn.close()


if __name__ == "__main__":
    main()

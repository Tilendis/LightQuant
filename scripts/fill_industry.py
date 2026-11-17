"""填充 stocks.industry：50 只股票的申万一级行业 + 大类分组。

为什么要这一步：`stocks` 表早就有 industry 字段，但 50 只全是 NULL，全仓库
零引用——建了字段从来没人写也没人读。基本面分析最基本的抓手（行业归属）
在系统里等于不存在。

口径：
- `industry` 存申万一级行业，保留精度，前端可以直接展示
- `SECTOR_MAP` 再做一层"大类"归并。原因是 50 只分到 20 个一级行业后，
  有一半行业只有 1 只股票，「同行业平均」恒等于自己，相对强弱恒为 0，
  这条信号会直接失效。归并成 9 个大类后，最大一类 11 只、最小 3 只，
  样本量才够算均值。
- 幂等：重复执行只覆盖 industry/name/code，不动其它字段

用法：
    python scripts/fill_industry.py
    python scripts/fill_industry.py --dry_run     # 只打印，不写库
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app import db  # noqa: E402

# 申万一级行业（按 2021 版分类）
INDUSTRY = {
    "万华化学": "基础化工",
    "三一重工": "机械设备",
    "上汽集团": "汽车",
    "中信证券": "非银金融",
    "中国中免": "商贸零售",
    "中国中铁": "建筑装饰",
    "中国人寿": "非银金融",
    "中国太保": "非银金融",
    "中国平安": "非银金融",
    "中国建筑": "建筑装饰",
    "中国核电": "公用事业",
    "中国电信": "通信",
    "中国电建": "建筑装饰",
    "中国石化": "石油石化",
    "中国石油": "石油石化",
    "中国神华": "煤炭",
    "中国移动": "通信",
    "中国联通": "通信",
    "中国船舶": "国防军工",
    "中国银行": "银行",
    "中微公司": "电子",
    "中芯国际": "电子",
    "中远海控": "交通运输",
    "交通银行": "银行",
    "伊利股份": "食品饮料",
    "保利发展": "房地产",
    "兆易创新": "电子",
    "兴业银行": "银行",
    "农业银行": "银行",
    "国电南瑞": "电力设备",
    "山西汾酒": "食品饮料",
    "工商银行": "银行",
    "恒瑞医药": "医药生物",
    "招商银行": "银行",
    "海光信息": "电子",
    "海天味业": "食品饮料",
    "海尔智家": "家用电器",
    "片仔癀": "医药生物",
    "特变电工": "电力设备",
    "紫金矿业": "有色金属",
    "药明康德": "医药生物",
    "贵州茅台": "食品饮料",
    "通威股份": "电力设备",
    "邮储银行": "银行",
    "金山办公": "计算机",
    "长城汽车": "汽车",
    "长江电力": "公用事业",
    "陕西煤业": "煤炭",
    "隆基绿能": "电力设备",
    "韦尔股份": "电子",
}

# 大类归并：把小样本的一级行业合成 9 个有统计意义的大类
SECTOR_MAP = {
    "银行": "金融",
    "非银金融": "金融",
    "电子": "科技",
    "计算机": "科技",
    "通信": "科技",
    "食品饮料": "消费",
    "商贸零售": "消费",
    "家用电器": "消费",
    "医药生物": "医药",
    "电力设备": "电力公用",
    "公用事业": "电力公用",
    "石油石化": "能源材料",
    "煤炭": "能源材料",
    "有色金属": "能源材料",
    "基础化工": "能源材料",
    "建筑装饰": "基建地产",
    "房地产": "基建地产",
    "机械设备": "制造",
    "汽车": "制造",
    "国防军工": "制造",
    "交通运输": "制造",
}


def sector_of(industry):
    """一级行业 → 大类。未登记的一级行业原样返回，不静默归错。"""
    return SECTOR_MAP.get(industry, industry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    conn = db.connect(args.db)
    db.init_db(conn=conn)

    rows = conn.execute("SELECT symbol, name, code, industry FROM stocks ORDER BY symbol").fetchall()
    on_db = [r["symbol"] for r in rows]
    missing = [s for s in on_db if s not in INDUSTRY]
    extra = [s for s in INDUSTRY if s not in on_db]

    print("库内股票 %d 只，本表覆盖 %d 只" % (len(on_db), len(on_db) - len(missing)))
    if missing:
        print("!! 库内有但本表没写行业：", missing)
    if extra:
        print("!! 本表有但库里没这只：", extra)

    if args.dry_run:
        from collections import Counter
        c = Counter(INDUSTRY.values())
        sc = Counter(sector_of(v) for v in INDUSTRY.values())
        print("\n申万一级分布：")
        for k, v in c.most_common():
            print("  %-10s %2d" % (k, v))
        print("\n大类分布：")
        for k, v in sc.most_common():
            print("  %-10s %2d" % (k, v))
        print("\n(dry_run：不写库)")
        conn.close()
        return

    for sym, ind in INDUSTRY.items():
        conn.execute(
            "UPDATE stocks SET industry = ? WHERE symbol = ?", (ind, sym))
    conn.commit()

    # 复核：确认没有 NULL 残留
    left = conn.execute(
        "SELECT COUNT(*) c FROM stocks WHERE industry IS NULL OR industry = ''"
    ).fetchone()["c"]
    print("\n已写入 %d 条，industry 仍为 NULL 的 %d 条" % (len(INDUSTRY), left))

    from collections import Counter
    sc = Counter(sector_of(r["industry"]) for r in
                 conn.execute("SELECT industry FROM stocks WHERE industry IS NOT NULL"))
    print("\n大类分布：")
    for k, v in sc.most_common():
        print("  %-10s %2d" % (k, v))
    conn.close()


if __name__ == "__main__":
    main()

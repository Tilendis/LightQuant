# -*- coding: utf-8 -*-
"""从 SQLite 直接导出 15 张表的结构与行数，生成 docs/数据库表清单.md。

为什么不手写：手抄 schema 一定会漏列、写错行数，而且改表之后文档就静默过期。
这里每次跑都从 PRAGMA 与 COUNT 现读。
"""
import io
import os
import sqlite3

ROOT = r"D:\03_Study\GraduationProject\LightQuant"
DB = os.path.join(ROOT, "app", "data", "lightquant.db")
OUT = os.path.join(ROOT, "docs", "数据库表清单.md")

# 表 → (中文说明, 对应题目要求)
META = {
    "users": ("用户与四类角色", "R7 四类角色：investor / author / risk / operator 一处定义，后端逐接口校验"),
    "stocks": ("标的池", "50 只 A 股，主键是中文简称（与 dataset 文件名一致，避免二次映射）"),
    "prices": ("日线行情", "R1 技术面输入：OHLCV + adj_close + 次日方向 label"),
    "news": ("新闻原文", "R1 情绪面输入：交易日 × 标的的新闻标题与正文"),
    "valuations": ("逐日估值序列", "R1 基本面输入（估值维度）：PE_TTM / PB / PS_TTM"),
    "fundamentals": ("按报告期的财务数据", "R1 基本面输入（公司经营维度）：ROE / 毛利率 / 增速 / 负债率 / 现金流"),
    "fund_events": ("财报事件", "R1 基本面输入（事件维度）：从新闻抽出的净利润同比等结构化财务事实"),
    "judgements": ("研判缓存（三视角 + 融合）", "R1 核心产出：每行带 provider / model / temperature / prompt_version，可逐条复核"),
    "backtest_tasks": ("异步回测任务", "R2 异步回测：PENDING → RUNNING → SUCCEEDED / FAILED，进度 0~1"),
    "strategies": ("策略（草稿 → 发布）", "R2 发布：回测结果指标来自 summary.json 的 portfolio 段"),
    "subscriptions": ("订阅关系", "R3 模拟跟单：投资者订阅已发布策略"),
    "follow_accounts": ("模拟跟单账户", "R3 + R4：peak_nav 算回撤，stop_loss_nav 是风控员看的强平线"),
    "risk_events": ("风控事件日志", "R4 三类事件统一一张表：MARGIN_CALL / LIQUIDATED / POST_BLOCKED"),
    "posts": ("社区发帖", "R4 违规拦截的对象；strategy_id 可空，支持「帖子绑定策略」"),
    "post_moderation": ("发帖审核结果", "R4 双层审核：rule_hit 记正则命中，llm_verdict 记模型判定原文"),
}


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    tables = [r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name")]

    lines = []
    lines.append("# 数据库表清单")
    lines.append("")
    lines.append("> 由 `scripts/dump_schema.py` 从 `app/data/lightquant.db` 直接导出，"
                 "不手抄——改表之后重跑一次即可，文档不会静默过期。")
    lines.append("> 建表语句的唯一来源是 `app/schema.sql`；后加的列由 `app/db.py` 的 `_migrate()` 补齐。")
    lines.append("")
    lines.append("共 **%d** 张表。设计原则写在 `schema.sql` 开头："
                 "每张表都对应题目里一个可演示的业务动作，不留“以后可能用得上”的空表。"
                 % len(tables))
    lines.append("")
    lines.append("## 一览")
    lines.append("")
    lines.append("| # | 表 | 中文说明 | 行数 | 对应题目要求 |")
    lines.append("|---|---|---|---|---|")
    total = 0
    for i, t in enumerate(tables, 1):
        n = conn.execute("SELECT COUNT(*) FROM %s" % t).fetchone()[0]
        total += n
        desc, req = META.get(t, ("（未登记）", ""))
        lines.append("| %d | `%s` | %s | %s | %s |"
                     % (i, t, desc, format(n, ","), req))
    lines.append("")
    lines.append("合计 **%s** 行。" % format(total, ","))
    lines.append("")

    lines.append("## 字段明细")
    lines.append("")
    for t in tables:
        desc, req = META.get(t, ("", ""))
        lines.append("### `%s` — %s" % (t, desc))
        lines.append("")
        if req:
            lines.append("对应：%s" % req)
            lines.append("")
        cols = conn.execute("PRAGMA table_info(%s)" % t).fetchall()
        lines.append("| 字段 | 类型 | 非空 | 主键 | 注释 |")
        lines.append("|---|---|---|---|---|")
        for c in cols:
            comment = COL_COMMENTS.get((t, c["name"]), "")
            lines.append("| `%s` | %s | %s | %s | %s |"
                         % (c["name"], c["type"] or "-",
                            "是" if c["notnull"] else "",
                            "是" if c["pk"] else "",
                            comment))
        idx = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' "
            "AND tbl_name=? AND sql IS NOT NULL", (t,)).fetchall()
        if idx:
            lines.append("")
            lines.append("索引：" + "、".join("`%s`" % r["name"] for r in idx))
        lines.append("")

    lines.append("## 三条读表红线")
    lines.append("")
    lines.append("1. **`valuations` 对齐到交易日必须 ffill，不得插值或回填。**"
                 "否则 2024-09 的判断会用到 2024-10 之后才知道的估值，属于未来函数。")
    lines.append("2. **`fundamentals` 用 `announce_date` 而不是 `period` 对齐。**"
                 "2024Q3 的报告要到 10-31 才可见，用 period 对齐等于提前知道两个月。")
    lines.append("3. **`judgements` 的 `source` 只有 live / cache / offline_rule 三种。**"
                 "前端据此标「离线预计算」，不得把缓存结果说成实时调用。")

    text = "\n".join(lines)
    io.open(OUT, "w", encoding="utf-8").write(text)
    print("写入 %s（%d 张表 / %s 行）" % (OUT, len(tables), format(total, ",")))
    conn.close()


COL_COMMENTS = {
    ("fundamentals", "symbol"): "中文简称，与 stocks.symbol 一致",
    ("fundamentals", "announce_date"): "法定披露截止日，时点对齐用这个而不是 period",
    ("judgements", "prompt_version"): "版本标签，融合层与编排层靠它区分",
    ("judgements", "source"): "live / cache / offline_rule，前端据此标离线预计算",
    ("follow_accounts", "stop_loss_nav"): "强平线，风控员界面读这个",
    ("follow_accounts", "peak_nav"): "滚动峰值，回撤 = current_nav / peak_nav - 1",
    ("prices", "label"): "次日方向（1 涨 0 跌），与训练口径一致",
    ("prices", "adj_close"): "复权收盘价；回测的成交价还要对 open 复权",
    ("fund_events", "subject_ok"): "1=主体是这只股票，0=新闻里只是提及他人",
    ("post_moderation", "rule_hit"): "规则层命中的正则名，空表示规则未命中",
    ("post_moderation", "llm_verdict"): "模型判定原文，含 provider/model",
}

if __name__ == "__main__":
    main()

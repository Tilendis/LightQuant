"""融合信号回测入口：让 judgements 里的研判真正参与决策。

这是补掉"跑题"的关键一步。原来 run_backtest_v2 是加载模型权重逐股推理、
拿模型自身概率当信号，融合层完全没参与，造成"研判是研判、回测是回测"两张皮。
这里改成：从 judgements 表读信号 → 走 engine_v2 同一套成本/T+1/自检。

支持的消融组合（论文需要，否则无法证明融合有增益）：
    --views tech                 技术面单视角
    --views tech,fund            + 基本面
    --views tech,sent            + 情绪面
    --views tech,fund,sent       三视角融合（可加 --method vote 换投票）

分数到概率的换算：p = (score + 1) / 2，所以 score 阈值 0.2 对应概率阈值 0.6。
这样融合的 BUY/SELL 判定与 judgements 里写的 signal 完全一致，不会出现
"研判说 HOLD、回测却买入"的两套阈值打架。

用法：
    python backtest/run_backtest_fusion.py --views tech,fund,sent --tag fusion_v1
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

from engine_v2 import run_backtest  # noqa: E402
from app import db  # noqa: E402

DEFAULT_10 = ["Open", "High", "Low", "Close", "Adj Close",
              "Return", "RSI6", "MACD_Hist", "BIAS5", "Amplitude"]


def load_scores(conn, views, start, end, prompt_version=None):
    """按视角读研判分数。fusion 有 weighted / vote 两版，靠 prompt_version 区分。

    prompt_version 可以是单个字符串（所有视角共用一个版本），也可以是
    {view: version} 字典。必须支持后者：tech 现在有 tech-v1 / tech-agent-v1 /
    tech-llm-v1 三个版本，若按 view 全量读再 pivot_table(aggfunc="last")，
    同一格会取到不同版本的记录，读出来的曲线无法解释。
    """
    mats = {}
    for v in views:
        pv = (prompt_version.get(v) if isinstance(prompt_version, dict)
              else prompt_version)
        sql = ("SELECT symbol, date, score FROM judgements "
               "WHERE view=? AND date BETWEEN ? AND ?")
        args = [v, start, end]
        if pv:
            sql += " AND prompt_version=?"
            args.append(pv)
        rows = conn.execute(sql, args).fetchall()
        if not rows:
            mats[v] = pd.DataFrame()
            print("  %-5s 空" % v)
            continue
        df = pd.DataFrame([dict(r) for r in rows])
        mats[v] = df.pivot_table(index="date", columns="symbol", values="score", aggfunc="last")
        print("  %-5s %d 日 × %d 只%s" % (v, mats[v].shape[0], mats[v].shape[1],
                                          ("（%s）" % pv) if pv else "（未指定版本，混合所有版本）"))
    return mats


def load_pool_map(conn, prompt_version, start, end, frac):
    """基本面选池：用估值分位给全市场排序，每天只留低估的前 frac。

    为什么基本面该管选池而不是管择时：估值分位是中期因子，低估值要几个月
    才均值回归，而本回测是日频、数天持仓。让它预测次日涨跌必然失效（验证集
    标定出来的权重就是 0）。但"哪些标的便宜"在截面上是有区分度的，用它缩小
    候选池、再由技术面在池内择时，基本面才真正参与决策而不是被权重归零。

    口径：直接复用 fund-rule-v1 已算好的 score（正=低估），按日期做截面排序，
        score 降序取前 ceil(n*frac)。该 score 本身只用 date<=d 的历史分位，
        不含未来信息，所以这里的排序也不前视。
    """
    rows = conn.execute(
        "SELECT symbol, date, score FROM judgements "
        "WHERE prompt_version=? AND date BETWEEN ? AND ?", (prompt_version, start, end)
    ).fetchall()
    if not rows:
        return None
    df = pd.DataFrame([dict(r) for r in rows])
    df = df.dropna(subset=["score"])
    pool = {}
    for d, g in df.groupby("date"):
        g = g.sort_values("score", ascending=False)
        k = max(1, int(np.ceil(len(g) * frac)))
        pool[str(d)] = set(g["symbol"].head(k).tolist())
    sizes = [len(v) for v in pool.values()]
    print("  选池 %d 日，池内标的 min %d / max %d / 均 %.1f（frac=%.2f）"
          % (len(pool), min(sizes), max(sizes), float(np.mean(sizes)), frac))
    return pool


def make_signal_fn(mats, weights, method, score_th, pool_map=None):
    """返回 (symbol, dates) -> 概率数组。缺失视角按可用权重归一化，不硬填 0。

    pool_map: {date: set(symbol)}。给出时，不在当日池内的标的强制输出 0.5(HOLD)，
    即只允许在池子里交易。日期不在 pool_map 时放行（首日可能无估值记录，
    一刀切拒绝会让整段没有成交，反而引入新偏差）。
    """
    views = list(mats.keys())

    def fn(symbol, dates):
        out = np.empty(len(dates), dtype=float)
        for i, d in enumerate(dates):
            if pool_map is not None:
                members = pool_map.get(d)
                if members is not None and symbol not in members:
                    out[i] = 0.5
                    continue
            vals = {}
            for v in views:
                m = mats[v]
                if m.empty or symbol not in m.columns or d not in m.index:
                    continue
                x = m.at[d, symbol]
                if pd.notna(x):
                    vals[v] = float(x)
            if not vals:
                out[i] = 0.5          # 无任何视角 → HOLD
                continue
            if method == "vote":
                s = np.mean([1.0 if v >= score_th else (-1.0 if v <= -score_th else 0.0)
                             for v in vals.values()])
            else:
                ws = sum(weights[views.index(v)] for v in vals)
                s = sum(weights[views.index(v)] * x for v, x in vals.items()) / ws if ws > 0 else 0.0
            s = max(-1.0, min(1.0, s))
            out[i] = (s + 1.0) / 2.0
        return out

    return fn


def make_pos_fn(sent_mat, th, full, mid, hedge, neutral, mode="level"):
    """分工制的仓位端：情绪面只决定"买多少"，不决定"买不买"。

    为什么这样切：情绪面覆盖只有 21.57%，绝大多数标的×日没有新闻。如果让它
    参与方向判定，无新闻的日子就等于弃权、等于把决策权交回技术面，融合在
    效果上退化成技术面的复制品（实测同向率 86.5%）。而"仓位"是个天然可以
    部分参与的维度——有新闻就调仓，没新闻就保持基准仓位，不影响方向。

    mode="level"（默认）：四档离散映射，与研判层的 BUY/HOLD/SELL 一一对应
        score >= +th  → full   （利好，满仓）
        -th < score < +th → mid（中性，略减）
        score <= -th  → hedge  （利空，明显减仓）
    mode="linear"：仓位随情绪分连续变化 pos = 0.5 + 0.45*score，裁到 [hedge, full]。
        离散档位会把 -0.99 和 -0.21 的利空同样处理，连续映射保留强弱差别，
        用来检验"信号是弱还是档位太粗"。
    无新闻一律 → neutral（弃权，不干预）。
    """
    def fn(symbol, dates):
        out = np.full(len(dates), float(neutral), dtype=float)
        if sent_mat is None or sent_mat.empty:
            return out
        if symbol not in sent_mat.columns:
            return out
        for i, d in enumerate(dates):
            if d not in sent_mat.index:
                continue
            x = sent_mat.at[d, symbol]
            if pd.isna(x):
                continue
            s = float(x)
            if mode == "linear":
                out[i] = float(np.clip(0.5 + 0.45 * s, hedge, full))
            elif s >= th:
                out[i] = full
            elif s <= -th:
                out[i] = hedge
            else:
                out[i] = mid
        return out

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="tech,fund,sent")
    ap.add_argument("--weights", default=None,
                    help="tech,fund,sent 的权重，按 --views 顺序；不填则等权")
    ap.add_argument("--method", default="weighted", choices=["weighted", "vote"])
    # 分数阈值 0 对应概率阈值 0.5，与单模型基线的判定完全一致。
    # 这样"技术面单视角"能和模型直接推理的结果逐位对齐，融合增益才是可比的。
    # 若要用研判层的 BUY 口径（score>=0.2）做敏感性分析，显式传 --score_th 0.2。
    ap.add_argument("--score_th", type=float, default=0.0)
    ap.add_argument("--prompt_version", default=None,
                    help="读 fusion 表时指定版本：fusion-v1 / fusion-vote-v1")
    # 基本面选池：把基本面从"择时"改成"缩小候选池"，解决权重被标定为 0 的问题
    ap.add_argument("--pool_from", default=None,
                    help="用哪个版本的估值分做选池，如 fund-rule-v1；不填则不选池")
    ap.add_argument("--pool_frac", type=float, default=0.5,
                    help="每天保留的低估值比例，0.5 = 留前一半")
    # 分工制：技术面管方向，情绪面管仓位
    ap.add_argument("--pos_from", default=None,
                    help="用哪个视角决定仓位，如 sent（需配合 --pos_version）")
    ap.add_argument("--pos_version", default="sent-v1")
    ap.add_argument("--pos_th", type=float, default=0.2,
                    help="情绪分阈值，与研判层 BUY/SELL 阈值一致")
    ap.add_argument("--pos_full", type=float, default=0.95, help="利好时的仓位上限")
    ap.add_argument("--pos_mid", type=float, default=0.75, help="中性时的仓位")
    ap.add_argument("--pos_hedge", type=float, default=0.45, help="利空时的减仓位")
    ap.add_argument("--pos_neutral", type=float, default=0.95,
                    help="无新闻（弃权）时的仓位，默认不干预")
    ap.add_argument("--pos_mode", default="level", choices=["level", "linear"],
                    help="level=四档离散，linear=随情绪分连续调节")
    ap.add_argument("--start", default="2024-08-08")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--price_folder", default="dataset/CSMD50_feat/test/price")
    ap.add_argument("--look_back_window", type=int, default=5)
    ap.add_argument("--tag", default="fusion_v1")
    ap.add_argument("--out_root", default=os.path.join("backtest", "result_v2"))
    # 成本参数与 v2 完全一致，保证对比是同一把尺子
    ap.add_argument("--initial_cash", type=float, default=100000.0)
    ap.add_argument("--position_pct", type=float, default=0.95)
    ap.add_argument("--lot_size", type=int, default=1)
    ap.add_argument("--commission_rate", type=float, default=0.0003)
    ap.add_argument("--min_commission", type=float, default=5.0)
    ap.add_argument("--stamp_duty_rate", type=float, default=0.001)
    ap.add_argument("--slippage_rate", type=float, default=0.0005)
    ap.add_argument("--rf_annual", type=float, default=0.02)
    ap.add_argument("--take_profit_pct", type=float, default=0.0)
    ap.add_argument("--stop_loss_pct", type=float, default=0.0)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    if "fusion" in views:
        # fusion 表里存的已经是"去均值 + 加权/投票"后的综合预测。
        # 题目原话是"通过投票或加权机制输出综合预测；策略作者基于预测结果回测"，
        # 所以主路径必须读这张表，而不是在回测里再算一遍（那样两边会分叉）。
        if len(views) != 1:
            raise SystemExit(
                "--views fusion 不能与其他视角混用：fusion 已是融合结果，权重固定 1.0")
        print("读 fusion 表（口径 = 样例外截面去均值 + 情绪面 ffill，权重 1.0）")
    if args.weights is None:
        weights = [1.0] * len(views)
        print("权重: 等权 %s" % ("/".join(["%.2f" % (1.0 / len(views))] * len(views))))
    else:
        weights = [float(x) for x in args.weights.split(",")]
        if len(weights) != len(views):
            raise SystemExit("--weights 个数必须与 --views 一致（%d vs %d）" % (
                len(weights), len(views)))

    # 版本必须逐个视角指定：不指定会读到同名 view 下的所有 prompt_version
    # （tech 有 tech-v1 / tech-agent-v1 / tech-llm-v1），结果不可解释。
    VER_DEFAULT = {"tech": "tech-v1", "fund": "fund-rule-v1", "sent": "sent-v1"}
    if args.prompt_version:
        pv_arg = args.prompt_version
        if len(views) > 1:
            # 多视角时显式给的版本只对 fusion 有意义，其余走默认
            pv_arg = {v: (args.prompt_version if v == "fusion" else VER_DEFAULT.get(v))
                      for v in views}
    else:
        pv_arg = {v: VER_DEFAULT.get(v) for v in views}
        if any(x is None for x in pv_arg.values()):
            raise SystemExit(
                "视角 %s 没有默认版本，必须显式传 --prompt_version"
                % [v for v, x in pv_arg.items() if x is None])

    conn = db.connect(args.db)
    print("读取研判:")
    mats = load_scores(conn, views, args.start, args.end, pv_arg)
    pool_map = None
    if args.pool_from:
        print("读取选池（%s，保留前 %.0f%% 低估标的）:" % (args.pool_from, args.pool_frac * 100))
        pool_map = load_pool_map(conn, args.pool_from, args.start, args.end, args.pool_frac)
        if pool_map is None:
            raise SystemExit("--pool_from %s 无数据，先跑 gen_fundamental" % args.pool_from)
    pos_mat = None
    if args.pos_from:
        print("读取仓位源（%s / %s）:" % (args.pos_from, args.pos_version))
        pm = load_scores(conn, [args.pos_from], args.start, args.end, args.pos_version)
        pos_mat = pm.get(args.pos_from)
    conn.close()
    if all(m.empty for m in mats.values()):
        raise SystemExit("所选视角全为空，先跑 gen_tech / gen_fundamental / gen_sentiment")

    cfg = dict(
        initial_cash=args.initial_cash, position_pct=args.position_pct,
        lot_size=args.lot_size, commission_rate=args.commission_rate,
        min_commission=args.min_commission, stamp_duty_rate=args.stamp_duty_rate,
        slippage_rate=args.slippage_rate,
        threshold=(args.score_th + 1.0) / 2.0,   # 分数阈值映射到概率阈值
        rf_annual=args.rf_annual,
        take_profit_pct=args.take_profit_pct, stop_loss_pct=args.stop_loss_pct,
    )
    # 口径写进结果，报告页才能如实说"读的是哪张表、哪一版"。
    # 之前只存成本参数，接口的 fusion_version 恒为 None，前端只好按 tag 名猜，
    # 选到单视角结果（api_tech）时会误标成"读融合表"，属于不实陈述。
    cfg["views"] = views
    cfg["method"] = args.method
    cfg["score_th"] = args.score_th
    # 单视图（--views fusion）时 pv_arg 是字符串，多视图时才是 dict，两种都要兼容
    if isinstance(pv_arg, dict):
        cfg["prompt_version"] = pv_arg.get("fusion") if "fusion" in views else pv_arg
    else:
        cfg["prompt_version"] = pv_arg
    # 选池口径也要落进结果，否则"加池 vs 不加池"两组结果在报告页无法区分
    cfg["pool_from"] = args.pool_from
    cfg["pool_frac"] = args.pool_frac if args.pool_from else None
    # 分工制的口径：仓位由谁给、阈值、四档取值
    cfg["pos_from"] = args.pos_from
    cfg["pos_version"] = args.pos_version if args.pos_from else None
    cfg["pos_th"] = args.pos_th if args.pos_from else None
    cfg["pos_mode"] = args.pos_mode if args.pos_from else None
    cfg["pos_levels"] = ([args.pos_full, args.pos_mid, args.pos_hedge, args.pos_neutral]
                         if args.pos_from else None)

    signal_fn = make_signal_fn(mats, weights, args.method, args.score_th, pool_map)

    pos_pct_fn = None
    if args.pos_from:
        if pos_mat is None or pos_mat.empty:
            raise SystemExit("--pos_from %s（%s）无数据" % (args.pos_from, args.pos_version))
        pos_pct_fn = make_pos_fn(pos_mat, args.pos_th, args.pos_full,
                                 args.pos_mid, args.pos_hedge, args.pos_neutral,
                                 mode=args.pos_mode)
        covered = int(pos_mat.notna().sum().sum())
        print("  仓位源覆盖 %d 个标的×日（占 %d 格的 %.2f%%）"
              % (covered, pos_mat.shape[0] * pos_mat.shape[1],
                 100.0 * covered / max(1, pos_mat.shape[0] * pos_mat.shape[1])))

    out_dir = os.path.join(PROJECT_ROOT, args.out_root, args.tag)
    summary = run_backtest(
        price_folder=os.path.join(PROJECT_ROOT, args.price_folder),
        model=None, device=torch.device("cpu"),
        feature_columns=DEFAULT_10, look_back_window=args.look_back_window,
        cfg=cfg, tag=args.tag, out_dir=out_dir, signal_fn=signal_fn,
        pos_pct_fn=pos_pct_fn,
    )

    pf = summary["portfolio"]
    print("\n[%s] views=%s method=%s" % (args.tag, args.views, args.method))
    print("ACC=%.4f CumRet=%.2f%% ARR=%.2f%% SR=%.3f MDD=%.2f%% IR=%.3f trades=%d" % (
        summary["per_stock_mean"]["acc"], pf["cum_return"] * 100, pf["arr"] * 100,
        pf["sharpe"], pf["mdd"] * 100, pf["ir_vs_equal_weight"], pf["total_trades"]))
    print("基准(等权买入持有) CumRet=%.2f%% MDD=%.2f%%" % (
        pf["bench_cum_return"] * 100, pf["bench_mdd"] * 100))
    if not all(summary["checks"].values()):
        print("!! 自检未全通过:", summary["checks"])
    else:
        # 项数从 checks 里读，不写死：复权改造新增 price_basis_declared 后，
        # 写死的"7 项"会和 summary.json 的 8 个键对不上，属于自相矛盾的口径。
        print("自检 %d 项全部通过" % len(summary["checks"]))


if __name__ == "__main__":
    main()

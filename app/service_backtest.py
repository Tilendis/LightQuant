"""异步回测任务执行器。

设计取舍：
- 用 ThreadPoolExecutor 而不是 Celery。理由：单机演示、纯 CPU、断网可跑，
  引 Celery 要带 Redis/RabbitMQ，与"普惠化、依赖最小化"冲突。
- 任务状态写回 backtest_tasks，进度按"视角数"推进，让前端能轮询。
- 回测逻辑不重新实现，直接复用 backtest/run_backtest_fusion.py 里的
  load_scores / make_signal_fn / engine_v2.run_backtest，保证与论文口径同源。
- 每个线程独立连接 SQLite（WAL 下可并发读），写操作都是短事务。
"""

import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
BACKTEST_DIR = os.path.join(PROJECT_ROOT, "backtest")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import db  # noqa: E402

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bt")
_lock = threading.Lock()
_fusion_mod = None


def _load_fusion():
    """按文件路径加载 run_backtest_fusion，避免要求 backtest 成为包。"""
    global _fusion_mod
    if _fusion_mod is not None:
        return _fusion_mod
    path = os.path.join(BACKTEST_DIR, "run_backtest_fusion.py")
    spec = importlib.util.spec_from_file_location("lq_run_backtest_fusion", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _fusion_mod = mod
    return mod


# 允许显式指定的融合版本。只放覆盖完整 test 段（08-08 起）或完整 val 段（03-15 起）的表。
# 12 月那批 agent 版（fusion-agent-v1 / fusion-orch-* 等）只有 600 行、12-16 起，
# 从 08-08 跨段回测会取不全，却照样"跑成功"并给出半截结果，所以不放进来。
FUSION_VERSIONS = ("fusion-test-3view", "fusion-vote-v1", "fusion-v1",
                   "fusion-val-3view", "fusion-val-v1")


DEFAULT_PARAMS = {
    # 默认走融合表：题目原话是"通过投票或加权机制输出综合预测，
    # 策略作者基于预测结果回测"，所以主路径读 fusion，而不是回测里再算一次。
    # 传 tech/fund/sent 可以跑单视角消融。
    "views": ["fusion"],
    "weights": [1.0],
    "method": "weighted",
    "fusion_version": None,   # 由 method 与起点推导：vote→fusion-vote-v1；weighted→按 start 选 3view 段
    "score_th": 0.0,
    "start": "2024-08-08",
    "end": "2024-12-31",
    "take_profit_pct": 0.0,
    "stop_loss_pct": 0.0,
    "position_pct": 0.95,
    "initial_cash": 100000.0,
    # 策略类型：timing=单标的择时（原路径，整仓跟随一个信号）；
    # cross_section=截面选股（每日/每 K 日从池里挑 top N 等权建组合）。
    # 两条路径共用同一份研判分数，差别只在"怎么把分数变成持仓"。
    "strategy_type": "timing",
    "top_n": 10,
    "hold_days": 5,
    # 成本假设必须可配、必须一路透传到引擎。此前择时路径把这四项写死在
    # _run() 里（0.0003 / 5 / 0.001 / 0.0005），而 DEFAULT_PARAMS 的
    # slippage_rate 却是 0.0020（给截面策略设的）——同一个系统里两套默认值，
    # 页面上改滑点对择时策略完全不生效，用户也看不出来。
    "commission_rate": 0.0003,
    "min_commission": 5.0,
    "stamp_duty_rate": 0.001,
    "slippage_rate": 0.0005,   # 择时口径：整仓跟随信号，换手低，万五
}

# 截面策略日频调仓，换手高得多，滑点默认取 0.2%（A 股现实值）。
# 与择时分开设，而不是让一条路径的默认值绑架另一条。
CROSS_SECTION_SLIPPAGE = 0.0020


def normalize_params(raw):
    """合并默认值并做最小校验；返回 (params, error)。"""
    p = dict(DEFAULT_PARAMS)
    p.update(raw or {})

    st = p.get("strategy_type") or "timing"
    if st not in ("timing", "cross_section"):
        return None, "strategy_type 只能是 timing 或 cross_section"
    p["strategy_type"] = st

    if st == "cross_section":
        # 截面策略只读技术面分数：目前只有 tech 在截面上有稳定排序能力，
        # fund 的截面多空差是负的（-0.176%，t=-1.62），sent 未做截面验证。
        # 这里固定单视角，不开放 views 参数，避免跑出无法解释的组合。
        p["views"] = ["tech"]
        p["weights"] = [1.0]
        p["fusion_version"] = None
        try:
            p["top_n"] = int(p.get("top_n") or 10)
            p["hold_days"] = int(p.get("hold_days") or 5)
        except (TypeError, ValueError):
            return None, "top_n / hold_days 必须是整数"
        if not 1 <= p["top_n"] <= 50:
            return None, "top_n 应在 1~50 之间，收到 %s" % p["top_n"]
        if not 1 <= p["hold_days"] <= 60:
            return None, "hold_days 应在 1~60 之间，收到 %s" % p["hold_days"]
        # val 与 test 是两个独立价格目录，日期不连续。跨段跑会静默只用其中一段，
        # 结果区间和用户填的不一致——直接拒掉，而不是给一个错区间。
        if str(p.get("start")) < "2024-08-08" and str(p.get("end")) >= "2024-08-08":
            return None, "截面策略暂不支持跨 val/test 两段（价格目录不连续），请把区间收在一段内"
        # 滑点默认按截面口径（0.2%）；用户显式传了就尊重用户的值。
        if not (isinstance(raw, dict) and raw.get("slippage_rate") is not None):
            p["slippage_rate"] = CROSS_SECTION_SLIPPAGE

    # 成本假设两条路径共用同一套校验：都必须是非负小数。
    # 只在校分截面分支校验的话，择时路径传 8 会一路穿到引擎，变成 800% 的滑点。
    for key in ("slippage_rate", "commission_rate", "stamp_duty_rate"):
        if key not in p or p.get(key) is None:
            continue
        try:
            v = float(p[key])
        except (TypeError, ValueError):
            return None, "%s 必须是数字" % key
        if v < 0 or v >= 1:
            return None, "%s 应为小数（0.002 表示 0.2%%），收到 %s" % (key, v)
        p[key] = v
    try:
        mc = float(p.get("min_commission") or 0.0)
    except (TypeError, ValueError):
        return None, "min_commission 必须是数字"
    if mc < 0:
        return None, "min_commission 不能为负，收到 %s" % mc
    p["min_commission"] = mc

    views = p.get("views") or []
    if not isinstance(views, list) or not views:
        return None, "views 必须是非空数组"
    for v in views:
        if v not in ("tech", "fund", "sent", "fusion"):
            return None, "unknown view: %s" % v
    if "fusion" in views and len(views) > 1:
        return None, "fusion 不能与其他视角混用：它已经是融合结果"
    # weights 未显式给出时按 views 数量等权。
    # 不能直接沿用 DEFAULT_PARAMS 里的 3 个权重：单视角（ablation）调用会冲突。
    raw_weights = (raw or {}).get("weights") if isinstance(raw, dict) else None
    if raw_weights is None:
        p["weights"] = [1.0] * len(views)
    else:
        weights = raw_weights
        if not isinstance(weights, list) or len(weights) != len(views):
            return None, "weights 个数必须与 views 一致（%d vs %d）" % (
                len(weights) if isinstance(weights, list) else -1, len(views))
        for w in weights:
            if float(w) < 0:
                return None, "weights 不能为负"
        if sum(float(w) for w in weights) <= 0:
            return None, "weights 之和必须大于 0"
        p["weights"] = [float(w) for w in weights]
    # 止盈止损必须是小数：0.08 表示 8%。界面上已经除以 100，这里再拦一次——
    # 直接把百分数传给引擎会让 tp 变成 800%、sl 变成负价，两者都永不触发，
    # 而回测仍能正常跑完，属于"看不出错"的静默失败。
    for key in ("take_profit_pct", "stop_loss_pct"):
        try:
            v = float(p.get(key) or 0.0)
        except (TypeError, ValueError):
            return None, "%s 必须是数字" % key
        if v < 0 or v >= 1:
            return None, "%s 应为小数（0.08 表示 8%%），收到 %s" % (key, v)
        p[key] = v
    if p.get("method") not in ("weighted", "vote"):
        return None, "method 只能是 weighted 或 vote"
    # fusion 表里有加权 / 投票两版，必须显式指定读哪一版，否则会混读。
    # 注意不能直接沿用 DEFAULT_PARAMS 里的值：那会让 method=vote 也读到 weighted 版。
    #
    # 加权版的默认值按回测起点选段，不能统一写 fusion-val-v1：
    #   03-15~08-07 → fusion-val-3view（标定与回测同段，属样本内）
    #   08-08~12-31 → fusion-test-3view（真样本外）
    # 两者都是 0.90/0.00/0.10。fusion-val-v1（tech=1.0）是旧标定，只作对照保留。
    # 跨段回测必须显式传 fusion_version，否则只会取到起点所在那一段。
    if "fusion" in views:
        raw_pv = (raw or {}).get("fusion_version") if isinstance(raw, dict) else None
        if raw_pv:
            # 不校验就"跑成功"是最坏的一种失败：版本名打错时读到空表，
            # 引擎照样产出一条净值水平线，指标看着有数、其实没有信号。
            if raw_pv not in FUSION_VERSIONS:
                return None, ("fusion_version 只能是 %s，收到 %s"
                              % (" / ".join(FUSION_VERSIONS), raw_pv))
            # method=vote 只对应一张表（fusion-vote-v1）。允许把投票方法配到加权表，
            # 会产出一条"名字叫投票、实际读加权"的曲线——指标有数，口径是错的。
            if p["method"] == "vote" and raw_pv != "fusion-vote-v1":
                return None, ("method=vote 只能读 fusion-vote-v1，收到 %s" % raw_pv)
            p["fusion_version"] = raw_pv
        elif p["method"] == "vote":
            p["fusion_version"] = "fusion-vote-v1"
        else:
            p["fusion_version"] = ("fusion-val-3view"
                                   if str(p.get("start") or "") < "2024-08-08"
                                   else "fusion-test-3view")
    else:
        p["fusion_version"] = None
    return p, None


def submit(user_id, raw_params):
    """建任务 → 立刻返回 task_id → 后台线程执行。"""
    params, err = normalize_params(raw_params)
    if err:
        raise ValueError(err)
    tag = params.get("tag") or _auto_tag(user_id)
    params["tag"] = tag

    conn = db.connect()
    try:
        cur = conn.execute(
            "INSERT INTO backtest_tasks (user_id, params_json, status, progress) "
            "VALUES (?, ?, 'PENDING', 0.0)",
            (user_id, json.dumps(params, ensure_ascii=False)))
        task_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    _EXECUTOR.submit(_run, task_id, params, tag)
    return task_id, tag


_seq = 0


def _next_seq():
    """进程内自增序号，仅作备用；不要用它生成 tag（见 _auto_tag 的说明）。"""
    global _seq
    with _lock:
        _seq += 1
        return _seq


def _auto_tag(user_id):
    """给未指定 tag 的任务生成目录名：时间戳 + 冲突时加后缀。

    不能用进程内自增序号：服务重启后序号归零，第一个任务又会拿到 api_2_1，
    直接把已有结果目录覆盖掉。回测照样跑通、页面照样出数，旧结果却静默消失——
    这是本次实际发生过的数据损坏，不是假设。

    只靠 os.path.exists 判断不够：目录真正的创建发生在后台线程里，同一秒提交的
    两条任务会两次都判成"不存在"，然后共用同一个目录、后跑的覆盖先跑的（本次
    自测就撞了）。所以这里必须加锁并当场把目录占下来，让第二次立刻看到冲突。
    """
    base = "api_%d_%s" % (user_id or 0, time.strftime("%Y%m%d%H%M%S"))
    root = os.path.join(PROJECT_ROOT, "backtest", "result_v2")
    with _lock:
        k = 1
        while True:
            name = base if k == 1 else "%s_%d" % (base, k)
            out = os.path.join(root, name)
            try:
                os.makedirs(out, exist_ok=False)
                return name
            except FileExistsError:
                k += 1
                if k > 99:
                    raise


def _update(task_id, **fields):
    conn = db.connect()
    try:
        sets = ", ".join("%s = ?" % k for k in fields)
        conn.execute("UPDATE backtest_tasks SET %s WHERE id = ?" % sets,
                     list(fields.values()) + [task_id])
        conn.commit()
    finally:
        conn.close()


def _run(task_id, params, tag):
    _update(task_id, status="RUNNING",
            started_at=_now(), progress=0.05)
    try:
        if params.get("strategy_type") == "cross_section":
            _run_cross_section(task_id, params, tag)
            return
        fusion = _load_fusion()
        conn = db.connect()
        try:
            mats = fusion.load_scores(conn, params["views"],
                                      params["start"], params["end"],
                                      params.get("fusion_version"))
        finally:
            conn.close()
        _update(task_id, progress=0.30)

        weights = params.get("weights") or [1.0] * len(params["views"])
        # fusion 表里已经是融合结果，回测直接透传，不能再投一次票
        method = "weighted" if "fusion" in params["views"] else params["method"]
        signal_fn = fusion.make_signal_fn(
            mats, weights, method, float(params["score_th"]))
        _update(task_id, progress=0.55)

        import torch  # 局部导入：只有跑回测时才付 torch 的启动成本
        cfg = dict(
            initial_cash=float(params["initial_cash"]),
            position_pct=float(params["position_pct"]),
            lot_size=1,
            # 四项成本从 params 读，不再写死；页面改成本假设对择时策略也生效
            commission_rate=float(params["commission_rate"]),
            min_commission=float(params["min_commission"]),
            stamp_duty_rate=float(params["stamp_duty_rate"]),
            slippage_rate=float(params["slippage_rate"]),
            threshold=(float(params["score_th"]) + 1.0) / 2.0,
            rf_annual=0.02,
            take_profit_pct=float(params["take_profit_pct"]),
            stop_loss_pct=float(params["stop_loss_pct"]),
        )
        out_dir = os.path.join(PROJECT_ROOT, "backtest", "result_v2", tag)
        # 口径必须写进结果：命令行脚本会写这几个字段，接口这条路之前漏了，
        # 导致页面"读表口径"只能显示"单视角"——把读的是融合表这件事说反了。
        cfg["views"] = params["views"]
        cfg["method"] = method
        cfg["score_th"] = float(params["score_th"])
        cfg["prompt_version"] = params.get("fusion_version") if "fusion" in params["views"] else None
        _update(task_id, progress=0.90)
        summary = fusion.run_backtest(
            price_folder=os.path.join(PROJECT_ROOT, "dataset", "CSMD50_feat", "test", "price"),
            model=None, device=torch.device("cpu"),
            feature_columns=fusion.DEFAULT_10, look_back_window=5,
            cfg=cfg, tag=tag, out_dir=out_dir, signal_fn=signal_fn)

        _update(task_id, progress=1.0, status="SUCCEEDED",
                result_path=os.path.relpath(out_dir, PROJECT_ROOT),
                finished_at=_now())
    except Exception:
        _update(task_id, status="FAILED", error=traceback.format_exc()[-2000:],
                finished_at=_now())


def _now():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_cross_section():
    """按文件路径加载截面引擎，理由同 _load_fusion：backtest 不是包。"""
    global _cs_mod
    if _cs_mod is not None:
        return _cs_mod
    path = os.path.join(BACKTEST_DIR, "engine_cross_section.py")
    spec = importlib.util.spec_from_file_location("lq_engine_cross_section", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _cs_mod = mod
    return mod


_cs_mod = None


def _run_cross_section(task_id, params, tag):
    """截面选股：读技术面分数 → 每日/每 K 日在池内挑 top N → 等权建组合。

    与 _run 的区别只在"怎么把分数变成持仓"，成本、指标、自检同一套。
    输出目录名加 cs_ 前缀，避免和择时策略的结果撞名。
    """
    fusion = _load_fusion()
    cs = _load_cross_section()
    start, end = params["start"], params["end"]

    # 价格目录按起点选段：val 与 test 是两个独立目录，日期不连续，
    # 跨段必须显式拼接，这里先按段跑，不做跨段合并（避免引入新的对齐问题）。
    seg = "val" if str(start) < "2024-08-08" else "test"
    price_folder = os.path.join(PROJECT_ROOT, "dataset", "CSMD50_feat", seg, "price")

    conn = db.connect()
    try:
        mats = fusion.load_scores(conn, ["tech"], start, end, "tech-v1")
    finally:
        conn.close()
    _update(task_id, progress=0.50)

    score_df = mats.get("tech")
    if score_df is None or score_df.empty:
        raise RuntimeError("区间 %s ~ %s 内没有 tech-v1 研判分数" % (start, end))

    cfg = dict(
        top_n=params["top_n"], hold_days=params["hold_days"],
        initial_cash=float(params["initial_cash"]),
        commission_rate=float(params.get("commission_rate", 0.0003)),
        min_commission=float(params.get("min_commission", 5.0)),
        stamp_duty_rate=float(params.get("stamp_duty_rate", 0.001)),
        slippage_rate=float(params["slippage_rate"]),
        lot_size=1, rf_annual=0.02,
        strategy_type="cross_section",
        views=["tech"], method="cross_section", score_th=0.0,
        prompt_version="tech-v1",
    )
    _update(task_id, progress=0.80)

    out_dir = os.path.join(PROJECT_ROOT, "backtest", "result_v2", tag)
    cs.run_cross_section(price_folder, score_df, cfg, tag, out_dir, start, end)

    _update(task_id, progress=1.0, status="SUCCEEDED",
            result_path=os.path.relpath(out_dir, PROJECT_ROOT),
            finished_at=_now())


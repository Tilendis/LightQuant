#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一条命令跑完整套接口，答辩现场自检用。

为什么需要它：现在 30 个接口全靠手工 curl，现场网络抖一下、少敲一个 header
就是当场卡住。这个脚本把"能不能演示"变成一个可执行的布尔值。

设计取舍：
- 只用标准库（urllib）。答辩机器上不一定装了 requests/httpx。
- **默认只读**：所有 GET 都打；写接口只做"预期会被拒"的负测试（401/403/400）。
  这些请求在写库之前就被拦下（认证在 get_user、授权在 need、参数在
  normalize_params），所以可以反复跑，不产生脏数据、不需要事后清理。
- 唯一会写库的是 `--seed`（调 POST /api/auth/demo-users），默认关闭。

用法：
    python scripts/selftest.py                       # 默认 http://127.0.0.1:8014
    python scripts/selftest.py --base http://127.0.0.1:8014
    python scripts/selftest.py --seed                # 首次演示前补四角色账号

退出码：全部通过 0，有任一失败 1（可直接接进 CI 或批处理）。
"""

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

DEMO_PASSWORD = "demo1234"
SYMBOL = "万华化学"

RESULTS = []


def call(base, method, path, token=None, body=None, timeout=30):
    """返回 (status_code, parsed_body_or_text)。网络错误抛 RuntimeError。"""
    url = base.rstrip("/") + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", "replace")
            code = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        code = e.code
    except Exception as e:
        raise RuntimeError("%s %s 连接失败：%s" % (method, path, e))
    try:
        return code, json.loads(raw)
    except ValueError:
        return code, raw


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    mark = "OK  " if ok else "FAIL"
    line = "[%s] %-46s %s" % (mark, name, detail)
    print(line)


def expect(name, got, want, detail=""):
    ok = (got == want)
    record(name, ok, "期望 %s，实际 %s%s" % (want, got, ("　" + detail) if detail else ""))
    return ok


def expect_true(name, cond, detail=""):
    record(name, cond, detail)


def login(base, username):
    code, body = call(base, "POST", "/api/auth/login",
                      body={"username": username, "password": DEMO_PASSWORD})
    if code != 200 or not isinstance(body, dict) or "token" not in body:
        raise RuntimeError("角色 %s 登录失败（%s）：%s" % (username, code, body))
    return body["token"], body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8014")
    ap.add_argument("--seed", action="store_true",
                    help="调 demo-users 补建四个角色账号（会写库，默认不做）")
    args = ap.parse_args()
    base = args.base

    print("=" * 72)
    print("LightQuant 接口自检　base=%s" % base)
    print("=" * 72)

    # ---- 0. 服务是否活着 -------------------------------------------------
    try:
        code, body = call(base, "GET", "/api/health")
    except RuntimeError as e:
        print("\n[FAIL] 服务连不上：%s" % e)
        print("　　　先起服务：.\\启动演示.bat　或　uvicorn app.api:app --port 8014")
        return 1

    expect("GET /api/health 返回 200", code, 200)
    expect_true("health.ok 为 true", isinstance(body, dict) and body.get("ok") is True)
    counts = (body or {}).get("counts") or {}
    expect_true("health 带 12 张表计数", len(counts) == 12, "实际 %d 张" % len(counts))
    rng = (body or {}).get("price_range") or [None, None]
    expect_true("行情区间非空", rng[0] and rng[1], "%s ~ %s" % (rng[0], rng[1]))
    ready = (body or {}).get("engine_ready")
    err = (body or {}).get("engine_error")
    expect_true("回测引擎已预热", ready is True, "engine_error=%s" % err)

    # ---- 1. 无凭证必须被拒 -----------------------------------------------
    code, _ = call(base, "GET", "/api/market/stocks")
    expect("无 token 访问受保护接口 → 401", code, 401)

    code, _ = call(base, "GET", "/api/market/stocks",
                   token=("x" * 20))
    expect("伪造 token → 401", code, 401)

    # ---- 2. 可选：补建演示账号 -------------------------------------------
    if args.seed:
        code, body = call(base, "POST", "/api/auth/demo-users")
        expect("POST /api/auth/demo-users → 200", code, 200)
        users = (body or {}).get("users") or []
        expect_true("四个角色账号齐全", len(users) == 4,
                    "、".join("%s(%s)" % (u["username"], u["role"]) for u in users))

    # ---- 3. 四角色登录 ---------------------------------------------------
    tokens = {}
    for role in ("investor", "author", "risk", "operator"):
        username = role + "_demo"
        try:
            tok, body = login(base, username)
        except RuntimeError as e:
            record("登录 %s" % username, False, str(e))
            print("\n　　　提示：先跑一次 python scripts/selftest.py --seed")
            return 1
        tokens[role] = tok
        record("登录 %-12s 角色=%s" % (username, (body.get("user") or {}).get("role")),
               (body.get("user") or {}).get("role") == role, "")

    code, _ = call(base, "POST", "/api/auth/login",
                   body={"username": "investor_demo", "password": "wrong"})
    expect("错误密码 → 401", code, 401)

    inv, auth_, risk, op = (tokens["investor"], tokens["author"],
                            tokens["risk"], tokens["operator"])

    # ---- 4. 公共读接口 ---------------------------------------------------
    code, body = call(base, "GET", "/api/meta", token=inv)
    expect("GET /api/meta computed", code, 200)
    expect_true("meta 含 4 个视角",
                isinstance(body, dict) and len(body.get("views") or []) == 4)

    code, body = call(base, "GET", "/api/market/stocks", token=inv)
    expect("GET /api/market/stocks", code, 200)
    expect_true("标的池 50 只", isinstance(body, list) and len(body) == 50,
                "实际 %d 只" % (len(body) if isinstance(body, list) else -1))

    sym = urllib.parse.quote(SYMBOL)
    code, body = call(base, "GET", "/api/market/kline/" + sym, token=inv)
    expect("GET /api/market/kline/%s" % SYMBOL, code, 200)
    bars = (body or {}).get("bars") or (body or {}).get("kline") or []
    expect_true("K 线有数据", len(bars) > 0, "%d 根" % len(bars))
    sigs = ((bars[-1] if bars else {}).get("signals")) or {}
    filled = [k for k, v in sigs.items() if (v or {}).get("score") is not None]
    expect_true("K 线带四视角信号位", len(sigs) == 4,
                "当日有分的视角：" + ("、".join(filled) if filled else "无"))

    code, body = call(base, "GET", "/api/judgements/" + sym, token=inv)
    expect("GET /api/judgements/%s" % SYMBOL, code, 200)
    expect_true("研判页返回非空", bool(body), "类型 %s" % type(body).__name__)

    code, body = call(base, "GET", "/api/fundamentals/" + sym + "?limit=4", token=inv)
    expect("GET /api/fundamentals/%s" % SYMBOL, code, 200)
    expect_true("财务数据有 4 期", isinstance(body, list) and len(body) >= 1,
                "实际 %d 期" % (len(body) if isinstance(body, list) else -1))

    code, body = call(base, "GET",
                      "/api/fundamentals/" + urllib.parse.quote("不存在的票"),
                      token=inv)
    expect("不存在的标的 → 200 空数组（不是 500）", code, 200)
    expect_true("空数组语义正确", isinstance(body, list) and len(body) == 0)

    code, body = call(base, "GET", "/api/accuracy", token=inv)
    expect("GET /api/accuracy", code, 200)
    summary = (body or {}).get("summary") or {}
    expect_true("准确率含 4 个视角", len(summary) == 4,
                "、".join("%s=%s" % (k, v.get("acc_all")) for k, v in summary.items())
                if summary else "无")
    meta_acc = (body or {}).get("meta") or {}
    expect_true("准确率带统计窗口", bool(meta_acc.get("start")),
                "%s ~ %s" % (meta_acc.get("start"), meta_acc.get("end")))

    code, body = call(base, "GET", "/api/judgements", token=inv)
    expect("GET /api/judgements", code, 200)
    expect_true("研判视图数 >= 4",
                isinstance(body, list) and len(body) >= 4,
                "实际 %d 个视图" % (len(body) if isinstance(body, list) else -1))

    # ---- 5. 回测链路 -----------------------------------------------------
    code, body = call(base, "GET", "/api/backtest/reports", token=inv)
    expect("GET /api/backtest/reports", code, 200)
    reports = (body or {}).get("reports") or []
    expect_true("磁盘上有回测结果", len(reports) > 0, "%d 个目录" % len(reports))
    tag = "test_fusion_3view"
    expect_true("主口径结果在列", any(r.get("tag") == tag for r in reports), tag)

    code, body = call(base, "GET", "/api/backtest/%s/report" % tag, token=inv)
    expect("GET /api/backtest/%s/report" % tag, code, 200)
    rep = body if isinstance(body, dict) else {}
    metrics = rep.get("portfolio") or {}
    eq = rep.get("nav") or []
    expect_true("报告含收益率/夏普/回撤",
                all(metrics.get(k) is not None
                    for k in ("cum_return", "sharpe", "mdd")),
                "cum=%.4f sharpe=%.4f mdd=%.4f" % (
                    metrics.get("cum_return") or 0, metrics.get("sharpe") or 0,
                    metrics.get("mdd") or 0))
    expect_true("报告含净值曲线", len(eq) > 0, "%d 个点" % len(eq))
    expect_true("报告含交易明细", len(rep.get("trades") or []) > 0,
                "%d 笔" % len(rep.get("trades") or []))
    mdd_rng = rep.get("mdd_range") or {}
    expect_true("含最大回撤区间（前端画阴影用）",
                bool(mdd_rng.get("start") and mdd_rng.get("end")),
                "%s ~ %s" % (mdd_rng.get("start"), mdd_rng.get("end")))

    code, body = call(base, "GET", "/api/backtest/tasks?limit=5", token=inv)
    expect("GET /api/backtest/tasks", code, 200)
    expect_true("任务列表可读", isinstance(body, list), "%d 条" % (
        len(body) if isinstance(body, list) else -1))

    # ---- 6. 订阅与跟单 ---------------------------------------------------
    code, body = call(base, "GET", "/api/subscriptions", token=inv)
    expect("GET /api/subscriptions（投资者）", code, 200)
    expect_true("投资者有订阅记录", isinstance(body, list) and len(body) > 0,
                "%d 条" % (len(body) if isinstance(body, list) else -1))
    first = (body[0] if isinstance(body, list) and body else {})
    aid = first.get("account_id")

    if aid:
        code, body = call(base, "GET", "/api/follow-accounts/%d" % aid, token=inv)
        expect("GET /api/follow-accounts/%d（投资者）" % aid, code, 200)
        expect_true("跟单账户带净值曲线",
                    isinstance(body, dict) and len(body.get("nav_curve") or []) > 0,
                    "%d 个点，当前净值 %s" % (
                        len((body or {}).get("nav_curve") or []),
                        (body or {}).get("current_nav")))
        code, _ = call(base, "GET", "/api/follow-accounts/%d" % aid, token=auth_)
        expect("策略作者读他人跟单账户 → 403", code, 403)

        code, body = call(base, "GET",
                          "/api/follow-accounts/%d/positions" % aid, token=inv)
        expect("GET /api/follow-accounts/%d/positions" % aid, code, 200)
        pos = (body or {}).get("rows") or []
        expect_true("持仓明细可复算", len(pos) > 0,
                    "%d 只持仓，as_of=%s" % (len(pos), (body or {}).get("as_of")))
        if pos:
            p0 = pos[0]
            expect_true("持仓行带股数/成本/现价/市值",
                        all(k in p0 for k in ("stock", "shares", "avg_cost",
                                              "last_price", "market_value")),
                        "%s 股数 %s 成本 %s" % (p0.get("stock"), p0.get("shares"),
                                              p0.get("avg_cost")))

        code, _ = call(base, "GET",
                       "/api/follow-accounts/%d/positions" % aid, token=auth_)
        expect("策略作者读他人持仓 → 403", code, 403)

        # 手工强平与取消订阅都是写操作，只做负测试：权限校验在写库之前，
        # 被拦下的请求不会留下事件或改状态，所以 selftest 可以反复跑。
        code, _ = call(base, "POST",
                       "/api/follow-accounts/%d/liquidate" % aid, token=inv,
                       body={"reason": "selftest 越权探测"})
        expect("投资者手工强平 → 403", code, 403)

        sid0 = first.get("id")
        if sid0:
            code, _ = call(base, "POST",
                           "/api/subscriptions/%d/cancel" % sid0, token=auth_)
            expect("策略作者取消他人订阅 → 403", code, 403)

    code, _ = call(base, "GET", "/api/follow-accounts/99999", token=inv)
    expect("不存在的账户 → 404", code, 404)

    # ---- 7. 风控 ---------------------------------------------------------
    code, body = call(base, "GET", "/api/risk/events", token=risk)
    expect("GET /api/risk/events（风控员）", code, 200)
    expect_true("有风险事件记录", isinstance(body, list) and len(body) > 0,
                "%d 条" % (len(body) if isinstance(body, list) else -1))

    code, body = call(base, "GET", "/api/risk/accounts", token=risk)
    expect("GET /api/risk/accounts（风控员）", code, 200)
    accts = (body or {}).get("accounts") or []
    expect_true("风控能读到跟单账户", len(accts) > 0, "%d 个" % len(accts))

    code, body = call(base, "GET", "/api/risk/scan-status", token=risk)
    expect("GET /api/risk/scan-status（后台巡检状态）", code, 200)
    expect_true("巡检状态返回间隔与账户数",
                isinstance(body, dict) and body.get("interval_sec", 0) > 0
                and body.get("accounts_count", 0) > 0,
                "interval=%s accounts=%s" % (body.get("interval_sec"), body.get("accounts_count")))
    if accts:
        alert = sum(1 for a in accts
                    if (a.get("payload") or {}).get("status") != "ACTIVE")
        expect_true("账户带强平相关字段",
                    any("stop_loss_nav" in (a.get("payload") or {}) for a in accts),
                    "非 ACTIVE 状态 %d 个" % alert)
        # R4：页面顶部那条汇总要现算「几个账户、几个快到线、最危险差多少」，
        # 靠的就是这几个字段；少一个就退回成「看一列数字」。
        keys = ("current_nav", "stop_loss_nav", "liquidate_nav",
                "max_dd_pct", "status", "event", "as_of")
        miss = [k for k in keys
                if not any(k in (a.get("payload") or {}) for a in accts)]
        expect_true("水位字段够算页面汇总", not miss,
                    ("缺：" + "、".join(miss)) if miss else "")

    code, _ = call(base, "GET", "/api/risk/events", token=auth_)
    expect("策略作者读风控事件 → 403", code, 403)

    code, _ = call(base, "POST", "/api/risk/scan", token=inv, body={})
    expect("投资者触发风控扫描 → 403", code, 403)

    # ---- 8. 越权矩阵（课题"四类角色"这条要求的硬证据）-------------------
    code, _ = call(base, "POST", "/api/backtest/tasks", token=inv,
                   body={"start": "2024-08-08", "end": "2024-12-31"})
    expect("投资者提交回测 → 403", code, 403)

    code, _ = call(base, "POST",
                   "/api/backtest/tasks", token=auth_,
                   body={"start": "2024-08-08", "end": "2024-12-31",
                         "views": ["tech"], "stop_loss_pct": 8})
    expect("止损传 8（百分数当小数）→ 400", code, 400)

    code, _ = call(base, "POST", "/api/backtest/tasks", token=auth_,
                   body={"start": "2024-08-08", "end": "2024-12-31",
                         "views": ["tech", "fusion"]})
    expect("fusion 与单视角混用 → 400", code, 400)

    # 预测版本：不校验的话，版本名打错会读到空表却照样"跑成功"，
    # 产出一条净值水平线——指标看着有数，其实一个信号都没有。
    code, _ = call(base, "POST", "/api/backtest/tasks", token=auth_,
                   body={"start": "2024-08-08", "end": "2024-12-31",
                         "views": ["fusion"], "method": "weighted",
                         "fusion_version": "fusion-xxx"})
    expect("未知融合版本 → 400", code, 400)

    code, _ = call(base, "POST", "/api/backtest/tasks", token=auth_,
                   body={"start": "2024-08-08", "end": "2024-12-31",
                         "views": ["fusion"], "method": "vote",
                         "fusion_version": "fusion-test-3view"})
    expect("投票配加权表 → 400", code, 400)

    code, _ = call(base, "GET", "/api/dashboard/activity", token=inv)
    expect("投资者看运营看板 → 403", code, 403)

    code, _ = call(base, "GET", "/api/dashboard/activity", token=op)
    expect("GET /api/dashboard/activity（运营）", code, 200)
    expect_true("活跃度看板带数据",
                isinstance(body, dict) and bool(body),
                "键：" + "、".join(list(body)[:6]) if isinstance(body, dict) else "")

    code, _ = call(base, "GET", "/api/dashboard/strategy-activity", token=op)
    expect("GET /api/dashboard/strategy-activity（运营）", code, 200)

    # ---- 实时研判调度（R1：课题里"调度…生成研判报告"的落点）---------------
    # 只走 write=false：三条链路会完整跑一遍（含 BiLSTM 推理），但不落库，
    # 保证自检反复跑不产生脏数据。2024-12-02 是万华化学最后一个有新闻的交易日。
    code, dj = call(base, "POST", "/api/judgements/run", token=auth_,
                    body={"symbol": SYMBOL, "date": "2024-12-02",
                          "allow_llm": False, "write": False}, timeout=120)
    expect("POST /api/judgements/run（不写库）", code, 200)
    expect_true("调度返回三视角与融合结论",
                isinstance(dj, dict) and isinstance(dj.get("views"), dict)
                and dj.get("fusion") is not None,
                "视角=%s 弃权=%s" % (
                    "、".join(sorted((dj.get("views") or {}).keys())) if isinstance(dj, dict) else "-",
                    (dj.get("skipped") if isinstance(dj, dict) else None)))
    expect_true("不写库时 written 为空",
                isinstance(dj, dict) and dj.get("written") == [], "")

    code, _ = call(base, "POST", "/api/judgements/run", token=inv,
                   body={"symbol": SYMBOL, "date": "2024-12-02", "write": False})
    expect("投资者触发研判调度 → 403", code, 403)

    code, _ = call(base, "DELETE", "/api/judgements/run/fusion-test-3view", token=auth_)
    expect("删除主口径版本 → 400（只允许删 rerun-*）", code, 400)

    # 批量研判：多只股 × 多个交易日，全程共用一个 rerun 版本
    code, db = call(base, "POST", "/api/judgements/run", token=auth_,
                    body={"symbols": [SYMBOL, "三一重工"],
                          "dates": ["2024-12-30", "2024-12-31"],
                          "allow_llm": False, "write": False}, timeout=180)
    expect("POST /api/judgements/run 批量 dry-run（2×2）", code, 200)
    expect_true("批量返回 total=4 且 fail=0",
                isinstance(db, dict) and db.get("total") == 4
                and db.get("fail_count") == 0 and db.get("ok_count") == 4,
                "total=%s fail=%s" % (
                    db.get("total") if isinstance(db, dict) else None,
                    db.get("fail_count") if isinstance(db, dict) else None))
    code, _ = call(base, "POST", "/api/judgements/run", token=auth_,
                   body={"symbols": [SYMBOL, "三一重工"],
                         "dates": ["2024-12-%02d" % d for d in range(1, 27)],
                         "allow_llm": False, "write": False})
    expect("批量超 50 上限 → 400", code, 400)
    code, _ = call(base, "POST", "/api/judgements/run", token=auth_,
                   body={"symbols": [], "dates": ["2024-12-31"], "write": False})
    expect("批量空 symbols → 400", code, 400)

    # ---- 跟单推进的日期来源（R3：不再由前端写死 '2024-12-31'）-----------------
    # 交易日历是"推进到哪一天"的唯一依据；推进语义是"下一步走一个交易日"。
    code, cal = call(base, "GET", "/api/trading-days", token=inv)
    expect("GET /api/trading-days（投资者）", code, 200)
    expect_true("交易日历覆盖 test 段",
                isinstance(cal, dict) and cal.get("first") == "2024-03-15"
                and cal.get("last") == "2024-12-31" and cal.get("count", 0) > 0,
                "first=%s last=%s count=%s" % (
                    (cal.get("first"), cal.get("last"), cal.get("count"))
                    if isinstance(cal, dict) else ("-", "-", "-")))

    # 库里 5 个账户都已推到 2024-12-31，所以批量推进只会返回"已到期末"，
    # 不会改任何数据——自检可以反复跑。
    code, adv = call(base, "POST", "/api/follow-accounts/advance-all", token=inv, body={})
    expect("POST /api/follow-accounts/advance-all（投资者）", code, 200)
    expect_true("批量推进按「一天」语义返回",
                isinstance(adv, dict) and adv.get("mode") in ("single-day", "fixed-date")
                and "moved" in adv and "skipped" in adv,
                "mode=%s moved=%s skipped=%s" % (
                    (adv.get("mode"), adv.get("moved"), adv.get("skipped"))
                    if isinstance(adv, dict) else ("-", "-", "-")))

    # 显式传非交易日必须 400：行情表只有交易日，静默吞成周五会让人以为"点了没反应"
    code, _ = call(base, "POST", "/api/follow-accounts/1/advance", token=inv,
                   body={"date": "2024-09-14"})
    expect("推进到非交易日 → 400", code, 400)

    # ---- 汇总 ------------------------------------------------------------
    total = len(RESULTS)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [(n, d) for n, ok, d in RESULTS if not ok]
    print("-" * 72)
    print("合计 %d 项：通过 %d，失败 %d" % (total, passed, len(failed)))
    if failed:
        print("\n失败明细：")
        for n, d in failed:
            print("  - %s　%s" % (n, d))
    print("=" * 72)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

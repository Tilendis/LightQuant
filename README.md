# LightQuant — 股票趋势多模型研判与策略回测系统

> 扬州大学 软件工程 2026 届毕业设计（课题 2）
> 一个面向个人投资者的量化投研平台：把「技术面 / 基本面 / 情绪面」三个分析视角
> 跑成可复算的研判，再用异步回测验证成策略，最后落到订阅跟单与风控。

**一句话说清它解决什么**：单一模型给出一个信号，你没法判断它是真看法还是偏差；
这里让三个不同来源的视角各自出分、各自留痕，再按事先定好的规则合成一个结果，
并且每一步都能回到原始数据上重算。

---

## 1. 这是什么 / 不是什么

**是**：
- 一个能跑起来的 Web 系统，37 个 HTTP 接口、5 个前端面板、四类角色权限隔离
- 研判可被系统调度：`POST /api/judgements/run` 一次调度技术面/基本面/情绪面三视角并当场出报告（前端「重新研判」按钮，实测 73–95 ms）
- 业务链每个环节在页面上都有入口：跟单账户能看持仓明细、能主动取消订阅；风控员能对单个账户手工强平（必须填原因）
- `scripts/selftest.py` 一条命令跑完 77 项接口检查（默认只读、可反复跑、不产生脏数据）
- `tests/` 下 8 个 pytest 文件、121 个用例，`python -m pytest` 一条命令跑完（全绿，覆盖指标公式、截面引擎、财务时点口径、鉴权与风控边界）
- 风控页进页面即出一条监控汇总（在途几个账户、几个不足 3% / 1%、最危险那个差多少），强平判定随账户推进自动执行并落 `risk_events`，不需要点「扫描」
- 三份交付文档：`docs/架构图-应用层.html`（附 PNG）、`docs/数据库表清单.md`（15 张表现读生成）、`docs/三分钟演示脚本.md`
- 三个研判视角各自独立实现，不是同一个模型换个提示词
- 回测、策略发布、订阅跟单、风控强平、社区审核连成一条完整业务链
- 所有结论都带口径说明，跑输的地方也照实写

**不是**：
- 不是实盘交易系统。跟单是「按策略净值曲线等比缩放复算」，不是逐笔撮合
- 不是全市场选股工具。演示数据是 CSMD50 的 50 只标的，2024-03-15 ～ 2024-12-31
- 不是「多模型一定能提升准确率」。实测里融合策略的收益**跑输买入持有基准**，
  原因写在 [第 5 节](#5-融合口径为什么权重是-0900010)

---

## 2. 30 秒跑起来

仓库自带 SQLite 演示库（`app/data/lightquant.db`，约 24 MB，含行情、研判、策略、
账户、风控事件），**不需要数据集也能完整演示**。

```powershell
# Windows：双击也行
.\启动演示.bat
```

或手动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8014
```

打开 <http://127.0.0.1:8014/> 。

首次启动会自动生成 `app/data/.auth_secret`（token 签名密钥，已在 `.gitignore` 里）。

也可以指向另一个库：

```powershell
$env:LIGHTQUANT_DB = "D:\tmp\clean.db"      # 指向新文件
$env:LIGHTQUANT_SECRET = "<32字节随机串>"   # 换签名密钥
```

**注意**：指向新文件时服务会自动建好 15 张空表（不会报错），但**没有演示数据**——
账号、行情、研判、策略都要自己灌。想要干净空库做彩排，这样起是对的；
想要直接演示，用仓库自带的 `app/data/lightquant.db`。

要跑新的回测任务，才需要 `dataset/`（见[附录 A](#附录-a本机复现说明)）。

---

## 3. 系统里有什么

### 四类角色，权限在服务端拦

| 角色 | 能干什么 | 演示账号 |
|---|---|---|
| 投资者 investor | 看行情与研判、订阅策略、看自己的跟单账户 | `investor_demo` |
| 策略作者 author | 配置参数、提交回测、发布策略 | `author_demo` |
| 风控员 risk | 扫描强平预警、处置风险事件、审核社区发帖 | `risk_demo` |
| 平台运营 operator | 看准确率对比、策略活跃度看板 | `operator_demo` |

认证用 HMAC-SHA256 签名的无状态 token（`app/auth.py`，92 行，不引 JWT 库），
授权靠每个接口的 `require_role`。前端隐藏菜单只是体验，**后端自己拦**：
投资者拿合法 token 读风控接口仍是 403。

细节与边界见 [`docs/认证与角色权限.md`](docs/认证与角色权限.md)。

### 业务流

```
行情 + 新闻  →  三视角研判（各出分）  →  融合（投票 / 加权）
                                        ↓
                          作者配止盈止损 → 异步回测 → 报告
                                        ↓
                                     发布策略
                                        ↓
                          投资者订阅 → 跟单账户 → 风控扫描
                                        ↓
                              MARGIN_CALL / LIQUIDATED
```

社区发帖另有一条线：规则层先拦（承诺收益、诱导加群、代客理财），
可再叠加一家**不同 provider** 的 LLM 做第二道判断。

### 前端 5 个面板

| 面板 | 内容 |
|---|---|
| 行情研判 | K 线 + 三视角信号叠加（不同颜色标注看多/看空）、研判理由、准确率 |
| 策略回测 | 提交任务、买卖点标注、最大回撤阴影、多任务对比 |
| 策略与订阅 | 已发布策略、参数、指标、订阅入口、跟单账户净值 |
| 风控与社区 | 强平预警、风险事件、发帖、审核结果 |
| 运营看板 | 多模型准确率折线、策略活跃度统计 |

截图在 [`docs/screenshots/`](docs/screenshots/)（16 张）。

---

## 4. 三个研判视角分别是什么

这里必须说清楚，因为「三个智能体」容易被含糊带过。**三个视角的技术形态不同**：

### 技术面 tech — 本地模型 + LLM 解读层

- 主体是本地训练的 **BiLSTM**（`bi_lstm_feat10`），输入 10 维：
  5 维价格序列 + `Return / RSI6 / MACD_Hist / BIAS5 / Amplitude`
- 输出 `score = 2p - 1`，落库为 `tech-v1`
- 另有智谱 `glm-4-flash` 的**解读层**（`tech-llm-v1` / `tech-llm-blind-v1`），
  它读 K 线与指标写理由，但**方向仍以本地模型为准**——避免「LLM 看着答案附和」

> 所以技术面严格说是「判别模型 + 语言模型解释」，不是纯 LLM 智能体。

### 基本面 fund — 真 LLM 智能体

- `fund-agent-v2`，provider = **DeepSeek**（`deepseek-flash`）
- 材料：估值分位（PE/PB 历史分位）+ 行业 + 财报事件（`fund_events`，458 条）
- 让它自己判断方向并给理由，落库 `fund-agent-v2`
- 另有规则兜底 `fund-rule-v1`（纯估值分位）与事件抽取 `fund-event-v1`

### 情绪面 sent — 真 LLM

- `sent-v1`，provider = **智谱**（`glm-5.3-flash`）
- 输入是当日该股的新闻文本，输出 -1 ～ +1 的情绪分
- 新闻覆盖只有 2024-03-15 ～ 12-31，且**不是每天每只股都有新闻**

> 三家 provider 各有分工：**本地 torch 模型 / DeepSeek / 智谱**，
> 这是有意的——避免整个系统只有一个模型供应商的口径风险。

### 覆盖率的真实情况

| 视角 | val 段覆盖 | test 段覆盖 |
|---|---|---|
| tech | 4450 格（89 天） | 4650 格（93 天） |
| fund | ~4900 格（含 agent 4872） | ~4850 格（含 agent 603） |
| sent | 1263 格**原始** → ffill 后 4570 | 1046 格**原始** → ffill 后 4617 |

情绪面是**稀疏**的：原始覆盖只有 23% / 19%，靠 `ffill` 补到 93% / 96%。
代价是**旧新闻会一直生效到下一次新闻**——test 段约 22% 的格子，
其情绪分来自 10 天以前的新闻。这是已知短板，见[第 9 节](#9-已知边界)。

---

## 5. 融合口径：为什么权重是 0.90 / 0.00 / 0.10

融合有两种实现，都留着：

- **投票** `fusion-vote-v1`：三视角多数决。实测失效——`HOLD` 占比 73.4%，
  多数决在「大家都弃权」的结构下必然输出 HOLD
- **加权** `fusion-test-3view-v2`：`score = 0.90×tech + 0.00×fund + 0.10×sent`

为什么 fund 权重是 0，这是实测结论不是偏好：

1. **基本面这一路只有「估值贵不贵」，没有「公司好不好」。**
   `valuations` 表 10001 行里可用的只有 `pe_ttm` / `pb`，`ps_ttm` 全 NULL，
   `fundamentals` 表 0 行。估值分位对**次日涨跌**没有预测力

2. **权重从 0 调到 1，val 段 ACC 单调下降**：
   `0.5538 → 0.4792`（单视角 fund 只有 0.4792）

3. **三条基本面路径全负**（截面选池、方向提示、否决票，见
   [`docs/基本面口径与财报事件验证.md`](docs/基本面口径与财报事件验证.md)）：
   多空差 −0.176%（t = −1.62），且**反向使用它的收益只比纯反转因子高 0.003pp**
   —— 那点「反向有效」是短期反转的复制品，不是基本面信息

> 换句话说：**不是「基本面没用」，是这一路当前拿到的数据只够判断「贵不贵」，
> 不够判断「好不好」。** 系统的价值在于能测出这一点并据此调权，而不是把三个
> 视角硬捏成「融合一定更好」。
>
> 财报事件这条路（`fund-event-v1`，458 条事件）已建但未接入主口径，
> 可交易性验证未通过（T+1 开盘买入后 val 段负收益）。

### 主口径的实测数字

`fusion-test-3view-v2`（test 段 2024-08-08 ～ 12-31）：
产物目录是 `backtest/result_v2/test_fusion_3view_v2/`，
`fusion-test-3view-v2` 是它的 `prompt_version`（也就是这一口径的版本名）。

| 累计收益 | 年化 | 夏普 | 最大回撤 | 交易笔数 | 信息比率 |
|---|---|---|---|---|---|
| **7.90%** | **23.15%** | **2.236** | **2.78%** | 616 | −1.419 |

**同期买入持有基准 +19.04%。**

这句话必须自己先说：**策略层跑输基准 11.14 个百分点，信息比率为负。**
胜在回撤小（2.78% vs 9.74%），但方向上没有跑赢市场。

> 两次口径修复说明：（1）技术面特征的 z-score 标准化原来在 test 段整段拟合
> （8 月的特征减掉了含 12 月分布的均值），现已改为只在 train 段拟合、
> test 段 transform，修复后 tech ACC 从 0.5482 降到 0.5060。
> （2）`fusion-test-3view` 在 2024-12-02~12-15 的 500 格仍读修复前的 tech，
> 已用当前 tech 重算为 `fusion-test-3view-v2`，主口径累计收益从 15.97% 依次
> 降到 9.36%（仅修泄漏）、再降到 7.90%（含 12 月段重算）——数字变差但口径可辩护。
> 旧 `fusion-test-3view` 目录保留作证据，**禁止引用**。
把 21.07% 当主结果是不对的（那是另一个口径的历史值）。

---

## 6. 回测与策略

- **异步**：提交任务立即返回 `task_id`，后台线程跑，前端轮询进度
- **两种策略类型**：
  - 时序（`engine_v2`）：每只股票独立 10 万账户，等权平均净值
  - 截面（`engine_cross_section`）：共享资金池，每日按分数排序取 TOP N 轮动
- **报告字段**：累计收益、年化、夏普、最大回撤、交易笔数、手续费、信息比率、准确率
- **止盈止损**：界面填百分数（如 3 表示 3%），引擎吃小数，`normalize_params` 统一转换
- **成本**：佣金 + 滑点，T+1，一字板不成交，停牌不成交

当前库里有 19 个回测任务、6 条已发布策略、5 个跟单账户。

截面策略（#8）是另一条线，与主口径**同源但不同引擎**，数字不能直接横比：

| 段 | 调仓次数 | 累计收益 | 年化 | 夏普 | 最大回撤 | 基准 |
|---|---|---|---|---|---|---|
| val | 18 | 12.13% | 34.64% | 1.740 | 6.76% | −13.65% |
| test | 19 | 53.80% | 209.59% | 3.676 | 5.29% | +14.63% |

**test 段的 53.80% 不能当「选股能力」讲**：剔掉单只股票后只剩 16.54%，
与基准 14.63% 只差 1.91pp。也就是说这段超额主要来自 2024-09-24 那一次普涨行情。
详见 [`docs/截面选股策略与可执行口径.md`](docs/截面选股策略与可执行口径.md)。

---

## 7. 目录结构

```
app/                     Web 应用
  api.py                 37 个接口（含登录）
  auth.py                token 签发与校验、PBKDF2 密码
  db.py                  连接、建表、幂等迁移
  schema.sql             15 张表
  service_backtest.py    异步回测调度（时序 / 截面两条分支）
  service_judge.py       三视角实时研判调度（tech/fund/sent → 融合）
  service_risk.py        跟单净值推演、强平、发帖审核
  static/                前端（原生 JS + ECharts，无框架）
backtest/                回测引擎
  engine_v2.py           时序引擎
  engine_cross_section.py 截面引擎（共享资金池）
  result_v2/             各 tag 的 nav.csv / trades.csv / summary.json
model/ utils/ scripts/   模型定义、数据处理、生成与评估脚本
docs/                    13 篇口径文档 + 16 张截图
tests/                   70 个用例
dataset/                 CSMD 数据（不入库，见附录 A）
```

---

## 8. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

121 个用例，分八组（`pytest.ini` 已固定 `testpaths = tests`，`python -m pytest` 一条命令即可）：

| 文件 | 数量 | 覆盖 |
|---|---|---|
| `test_engine_v2.py` | 19 | 指标公式、T+1、佣金滑点、止盈止损、一字板、复权缩放 |
| `test_engine_cross_section.py` | 12 | 截面引擎：分数取上一日（取当日即未来函数）、调仓清旧仓、复权因子、NAV 不含负现金、JSON 不含 NaN、稳健性剔除 |
| `test_price_dataloader.py` | 7 | 价格加载、除权跳变、strict 模式 |
| `test_audit_prices.py` | 5 | 复权审计脚本 |
| `test_fundamentals.py` | 20 | 财务层：`period_of` 推导、毛利率兜底、时点口径（未公告不可见）、同比取同报告期上一年 |
| `test_api_auth.py` | 16 | 401/403 两条线、伪造 token、过期、老请求头冒充回归 |
| `test_api_fundamentals.py` | 19 | `GET /api/fundamentals/{symbol}`：展示日可见性逐个边界、年报年内不可见、同比为空、排序与 limit、未知标的、鉴权 |
| `test_service_risk.py` | 23 | 移动止损、预警/强平边界、dry_run 不写库、发帖规则 |

接口测试用临时库（`LIGHTQUANT_DB` 环境变量），不碰演示库。

---

## 9. 已知边界

这些不是「没想到」，是**有意留白**，答辩时应该主动说：

1. **情绪面 ffill 无时间衰减** —— test 段 22% 的格子情绪分来自 10 天前的新闻。
   加衰减会改动主口径的输入，需要重跑并做影子对比，目前未做
2. **技术面加指标要重训** —— `bi_lstm_feat10` 输入固定 10 维，
   加布林带/KDJ/ATR 就必须重训，进而 tech / 融合 / 回测全链重跑
3. **基本面只有估值** —— `fundamentals` 表 0 行，财务数据未接入；
   这条路的权重因此是 0
4. **新闻只覆盖 2024-03-15 ～ 12-31** —— 训练段（2021-01 ～ 2024-03-14）
   的新闻未导入，`news` 表 6726 行全部落在测试期内
5. **跟单不是撮合** —— 按净值曲线等比缩放，`service_risk.advance_account`
   里逐日复算，不是逐笔成交
6. **token 无法主动吊销** —— 无状态签名，签出去只能等 8 小时过期
7. **无 HTTPS、无登录失败限流** —— 本地演示环境，未做传输层防护
8. **截面策略与主口径基准起点不同** —— 08-08 vs 08-14，基准 14.63% vs 19.04%

---

## 附录 A：本机复现说明

以下是原 CSMD 复现说明，保留作为运行纪律记录。

### 环境

- Python 3.11.9；`torch 2.4.1+cpu`、`pyyaml 6.0.3`（完整依赖见 `requirements.txt`）
- 纯 CPU 可跑，无需 GPU；`use_swanlab: false` 时不校验 API key，也不联网
- 虚拟环境位于 `.venv/`，命令统一用 `.venv\Scripts\python.exe`

### 数据放置

```
dataset/CSMD50/{train,val,test}/price/<股票代码>.csv
    表头：Date,Open,High,Low,Close,Adj Close,Volume,Label
dataset/CSMD50_feat/{train,val,test}/price/<股票代码>.csv
    表头：上述 8 列 + Return,RSI6,MACD_Hist,BIAS5,Amplitude
```

时间切分：train 2021-01-04～2024-03-14、val 2024-03-15～2024-08-07、
test 2024-08-08～2024-12-31。

`CSMD50_feat` 的技术指标先在完整序列上计算再切分，避免边界 NaN。

`dataset/` 不进仓库（412 MB），需从 CSMD 原始来源获取。

### 最小复现（四条命令）

```bash
# 1. 基线：ALSTM / 5 天窗口 / 5 维价格 / 关标签平滑
python run.py --config configs/base_alstm.yaml

# 2. 改进 A1：同一模型加 5 个技术指标（输入 10 维）
python run.py --config configs/exp_feat_alstm.yaml

# 3. 改进 A2：第 5 维换成成交量（替换，不追加）
python run.py --config configs/exp_vol_alstm.yaml

# 4. 改进 E2：把十维输入迁到最强基线 BiLSTM
python run.py --config configs/exp_feat_bi.yaml
```

配置里的键与 argparse 参数名一一对应，命令行同名参数优先覆盖：

```bash
python run.py --config configs/base_alstm.yaml --seed 101
```

每次运行会自动做三件事：建好 `model_saved/ figure/ test_result/`；
在 `result/<实验名>/run_manifest.txt` 写入时间、git commit、完整命令行和
模型/数据集/窗口/输入维/特征列/标签平滑/种子七项口径；日志首行打印 `[args]` 全参回显。

### 结果并表

```bash
python scripts/compare_results.py                  # 打印对照表
python scripts/compare_results.py --csv result/summary.csv
```

脚本只读取不重算，并会标出「无结果 / 参数未回显 / 维度冲突 / 有报错」四类记录问题。

### 已知问题（都已踩过）

1. `--look_back_window` 默认是 **7**，不是论文的 5；用配置时已显式写 5
2. `--label_smoothing` 默认是 **True**；做干净基线必须显式写 false
3. `--dataset` 只在传参后自动改写 `dataset/` 与 `result/` 默认路径；
   **已在配置里写死的输出路径不会被改写**，换实验名要一起改
4. 带空格的列名（`Adj Close`）不要用命令行字符串传，会被 shell 按逗号拆散
5. 启动约 30 秒（torch / swanlab 导入），前 30 秒日志为空是正常的
6. `--feature_columns` 的列数必须等于 `--input_size`，不一致会在入口直接报错
7. 早停是每 5 轮验证一次、`early_stop_patience=10`（等价容忍 50 轮），
   实测早停点在 55～115 轮

### 口径提醒

只有 **5 天窗口** 的结果可作为对外数字。早期 1 天窗口（4850 样本）的结果已作废，
仅在 `result/CSMD50*/` 留档，不参与任何横向比较。

**价格口径**：数据集不提供复权因子——`scripts/audit_prices.py` 实测 150 个 CSV 的
`Adj Close` 与 `Close` 逐行相等、0 次除权跳变，因此回测按未复权价成交。
引擎已按 `Adj Close/Close` 比值统一缩放 OHLC，换到带真实因子的数据源会自动生效。
细节见 [`docs/复权口径与回测价格处理.md`](docs/复权口径与回测价格处理.md)。

---

## 附录 B：上游数据集与框架（CSMD / LightQuant 原始说明）

> 以下内容来自上游仓库，我方未修改。注意其 "LightQuant" 指的是**原始作者的
> 轻量量化框架**，与本毕设系统同名但不同物——本系统是在其代码基础上
> 扩展出的 Web 平台。

### CSMD: Chinese Stock Multimodal Dataset

A multimodal dataset for the Chinese stock market encompassing stock prices and
financial news texts, publicly available for research.

- 整合股价与财经新闻文本，经严格清洗
- 用 LLM 从新闻抽取因子，提升可解释性
- 新闻源为证券时报等权威媒体
- 数据经去噪处理

CSMD V1.0 含 CSMD50 与 CSMD300 两个规模。

### 上游目录说明

```
./dataset              数据集
./evaluate             LLM 可读性评估
./model                主模型
./price_experiment     单模态（价格）实验
./price_news_experiment 多模态（价格+新闻）实验
./result               实验结果
./utils                 工具
./backtest              回测
./llm_factor            新闻因子抽取
```

### 上游 Changelog

- **[2025.06.16]** CSMD 开源部分数据集与完整数据处理流程
- **[2025.08.05]** 论文被 CIKM 2025 接收
- **[2025.09.08]** 开源完整 CSMD50 与 CSMD300

---

## 许可以及引用

本仓库的**应用层代码**（`app/`、`backtest/engine_cross_section.py`、`tests/`、
`docs/`）为毕设原创。上游数据集与框架代码的授权遵循其原始声明，
使用数据集请引用 CSMD / CIKM 2025 原论文。

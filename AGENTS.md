# AGENTS.md — LightQuant 工程规则（给 VSCode 里的编码助手）

> 本文件是这个仓库的**硬约束**。与任何 prompt 冲突时，以本文件为准。
> 上级目录还有 `CLAUDE.md`（项目规范）与 `题目2分析.md`（选题论证），需要背景时读它们，但不要在 `LightQuant/` 之外新建文件。

## 一、项目在做什么

毕设题目2：《股票趋势多模型研判与策略回测系统设计与实现》。
在本仓库既有实验代码之上，**新增**一个纯 CPU 的应用层：FastAPI + SQLite + 单页 HTML/ECharts，实现"三路研判 → 加权仲裁 → 异步回测 → 策略发布 → 订阅跟单 → 风控 → 运营看板"。

三路研判的实际构成（不要写成"多个大模型"）：

- 技术面 = 本地已训练好的 BiLSTM（不调用大模型）
- 基本面 = 云端模型 A
- 情绪面 = 云端模型 B（必须与 A 不同 provider）

## 二、绝对红线

1. **不许编数字**。任何指标必须能从 `result/*/test_result/*.txt`、`run_manifest.txt` 或数据库记录指回去。拿不到就报错或留空，不要填一个"看起来合理"的值。
2. **不许覆盖 `result/` 与 `backtest/result/` 下已有目录**。新实验、新回测一律写新目录，命名带用途后缀。
3. **`../finbench/` 只读**，不许改、不许当代码来源（它只是官方基准原码）。
4. **纯 CPU**。不许引入 CUDA/vLLM/本地大模型权重；技术面只做单样本前向推理。
5. **技术栈固定**：Python 3.11（`.venv`，torch 2.4.1+cpu 保持不动）、FastAPI、SQLite(WAL)、`ThreadPoolExecutor`、原生 HTML/JS + 本地 ECharts。**禁 MySQL / Redis / Celery / Docker / Node 构建链 / 在线 CDN**。
6. **不许 `git add -A` 或 `git commit -A`**。`dataset/` 巨大，误提交极难清理。要提交就逐个文件点。
7. **不许伪造进度**：禁止 `sleep()` 造"正在计算"的假进度条；异步任务用真实状态机 `PENDING → RUNNING → COMPLETED / FAILED`。

## 三、可引用的数字（只有这些）

同口径（5 天窗口、关闭 Label Smoothing、CSMD50、test 4650 样本）：

| 模型 | ACC | MCC | F1 | 证据 |
| --- | --- | --- | --- | --- |
| BiLSTM（单种子 seed37，旧"最强基线"叙事） | 54.60% | 0.0884 | 0.4687 | `result/CSMD50_w5_nols_bi_lstm/test_result/bi_lstm.txt` |
| ↳ BiLSTM 三种子均值（37/43/101） | **53.71% ± 0.94** | 0.0700 ± 0.0195 | 0.3885 ± 0.0878 | `result/CSMD50_w5_nols_bi_lstm{,_s43,_s101}` |
| LSTM（单种子 seed37） | 53.74% | 0.0703 | 0.4414 | `result/CSMD50_w5_nols_lstm/test_result/lstm.txt` |
| ↳ LSTM 三种子均值（37/43/101） | **53.42% ± 0.28** | 0.0634 ± 0.0060 | 0.4082 ± 0.0321 | 同上 + `_s{43,101}` |
| ALSTM（单种子 seed37） | 53.31% | 0.0610 | 0.4093 | `result/CSMD50_w5_base/test_result/alstm.txt` |
| ↳ ALSTM 三种子均值（43/101/2024） | **53.30% ± 0.54** | 0.0610 ± 0.0116 | 0.3749 ± 0.0521 | `result/base_alstm_s{43,101,2024}`（`result/seeds_summary.json`） |
| AdvLSTM（单种子 seed37） | 53.01% | 0.0559 | 0.3161 | `result/CSMD50_w5_nols_adv_lstm/test_result/adv_lstm.txt` |
| ↳ AdvLSTM 三种子均值（37/43/101） | **52.60% ± 0.62** | 0.0453 ± 0.0153 | 0.3095 ± 0.0500 | 同上 + `_s{43,101}` |
| ALSTM + 十维技术指标 | 54.84% ± 0.13（3 种子；单种子 54.49%） | 0.0950 ± 0.0023 | 0.4165 ± 0.0317 | `result/exp_feat_alstm_s{43,101,2024}` |
| BiLSTM + 十维技术指标 | 54.82% ± 0.17（3 种子；单种子 54.82%） | 0.0950 ± 0.0049 | 0.4089 ± 0.0162 | `result/CSMD50_w5_feat_bi/test_result/bi_lstm.txt` + `result/exp_feat_bi_s{43,101}` |
| 四模型概率平均 `avg4` | 54.04%（成员**全为 seed37 权重**；相对同批成员均值 53.67% 为 +0.38pp，相对三种子成员均值 53.26% 为 +0.78pp，比 BiLSTM 单种子 54.60% 低 0.56pp） | 0.0768 | 0.4275 | `result/CSMD50_w5_ensemble/ensemble_result.json` + `docs/exp_multiseed_four_models_w5.json` |

**不可引用**：`backtest/result/` 下任何绩效数字（仓位退化，SR −22.2 那批）、集成里"按 test MCC 加权 54.09%"（test leakage）、任何 `G-方案1.md` 里的旧结论。新口径只认 `backtest/result_v2/`。

## 四、代码与运行约定

- 训练/实验**只能通过 `--config configs/*.yaml` 传参**。带空格的列名（`Adj Close`）写在 yaml 里，**绝不要**拼在命令行字符串里传——会被逗号拆散。
- `--input_size` 必须等于 `feature_columns` 的列数，否则 `utils/price_dataloader.py` 会直接 `ValueError`（这是有意的强校验）。
- 默认值不是论文口径：`--label_smoothing` 默认 True、`--look_back_window` 默认 7。新建配置一律显式写死。
- 训练启动约 30 秒内日志为空属正常，不要据此判断进程已死；判断依据是 `run_log.txt` 是否开始增长 / 进程是否在。
- 推理封装**禁止**对整段序列再做 z-score（原 `load_csv_file` 是整段标准化，含未来信息，这是已知局限，不要扩大它的影响面）。
- 写 Python 源码用 UTF-8 **无 BOM**；控制台中文可能乱码，校验靠回读文件。
- 每条对外接口都要能回答"这个数字哪来的"。看板上的 LLM 两行必须带脚注：**历史段不报胜率**（测试期是 2024 下半年，多数大模型很可能已见过，是背书不是预测）。
- 回测只认 `backtest/engine_v2.py` 的结果；`backtest/result/` 下的旧绩效数字（SR −22.2 那批）是仓位退化产物，不得上屏、不得引用。

## 五、当前任务与验收方式

按 `../CLAUDE.md` 第八节的三步走，**一次只做一张卡，做完贴出验收命令的真实输出**。

第一步（只读，不写代码）：盘点 `run.py` / `configs/` / `scripts/` / `backtest/` / `single_exp/` / `evaluate/` / `utils/` 的真实调用链，产出"新增文件清单 + 改动文件清单（含影响面）"。
验收：清单里每个路径都真实存在；能说清"训练→预测→回测"三段的输入输出文件。

第二步（后端）：先回测引擎（五项修复）→ 再 SQLite 与 `seed_db.py` → 再研判层与缓存。
验收：`python -m app.selftest` 打印每个接口的 status code 与关键字段；回测自检要给出"交易次数 = 买卖点数""MDD 起止日期落在净值最低区间"。

第三步（前端）：三张 ECharts 图 + 四角色视图切换（后端校验 `role`）。
验收：断网（拔网线或改 hosts）也能画图出结果，响应标 `source:"offline_cache"`。

## 六、动手前先确认

- 改任何现有脚本前，先 `git status` 确认工作区干净或已知；改完能 `git diff` 说清动了什么。
- 新增依赖前先 `pip install --dry-run`，确认不会动 `torch 2.4.1+cpu` 与 `pandas`。当前 venv 已装 `fastapi` / `uvicorn`，**尚未装 `akshare`**（装之前必须先 dry-run，akshare 依赖链长，容易顶版本）。

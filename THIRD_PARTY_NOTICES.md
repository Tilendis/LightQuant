# 第三方组件与数据来源声明（THIRD PARTY NOTICES）

本项目（面向本科毕业设计的量化投研平台）**基于开源框架 LightQuant 二次开发**，
并使用 CSMD 公开数据集。以下第三方作品的版权、许可与使用条件，均以其原始仓库／
论文为准；本文件仅作汇总与声明，不改变任何原始条款。

---

## 1. LightQuant 框架代码

| 项 | 内容 |
|---|---|
| 来源 | https://github.com/ECNU-CILAB/LightQuant |
| 作者 | ECNU CILAB（华东师范大学计算机与智能教育实验室） |
| 许可 | MIT License |
| 附加声明 | 上游 README 的 Disclaimer 明确写明：**仅限学术与非商业用途，禁止商业使用**（"This project is for academic and non-commercial use only. Commercial use is strictly prohibited."）|
| 本项目使用的范围 | 数据加载与预处理流程（`dataset_construction/`）、基础模型（`model/`）、工具函数（`utils/`）、实验骨架（`single_exp/`、`multi_exp/`、`evaluate/`、`factor_lib/`、`llm_factor/`） |
| 本项目对其的改动 | `backtest/engine_v2.py`、`backtest/engine_cross_section.py` 为自研回测引擎；应用层 `app/` 全部自研；上游未提供多角色 Web 平台、三视角研判留痕、Point-in-Time 防穿越与口径溯源机制 |
| 义务 | 依 MIT 条款，保留上游版权与许可声明；本项目 `LICENSE` 已并列保留 `Copyright (c) ECNU CILAB` |

---

## 2. CSMD 数据集（CSMD50 / CSMD300）

| 项 | 内容 |
|---|---|
| 来源 | CSMD: Curated Multimodal Dataset for Chinese Stock Analysis（CIKM '25，pp. 6471–6475；华东师范大学 CILAB） |
| 构成 | CSMD50 = **上证 50** 的 50 只成分股；CSMD300 = **沪深 300** 的 300 只成分股；时间区间 2021–2024；股价（Baostock）+ 新闻文本（证券时报网 STCN） |
| 许可 | **CC BY-NC-SA 4.0**（署名 — 非商业性使用 — 相同方式共享） |
| 义务 | ① 署名（引用原论文）；② 不得用于商业目的；③ 若基于数据集产生衍生作品并**对外分发**，须以相同许可（CC BY-NC-SA 4.0）分发 |
| 本项目的使用性质 | 本科毕业设计的学术研究用途，非商业；仓库当前为**私有仓库**，未构成公开分发 |
| 重要提醒 | 数据集中成分股名单为**数据集构造时的静态快照**，而官方指数每半年调样；本项目无法还原历史成分，相关结论已在论文限制章节声明（幸存者偏差） |

**建议引用：**

> Liu Y., Li Z., Yang R., Mo F., Chen C. *CSMD: Curated Multimodal Dataset for
> Chinese Stock Analysis*. In Proceedings of the 34th ACM International Conference
> on Information and Knowledge Management (CIKM '25), pp. 6471–6475, 2025.
> DOI: 10.1145/3746252.3761636

---

## 3. 开源依赖（运行时）

除上述两项外，本项目还使用以下公开开源库，均按其各自许可使用，未做修改分发：

`fastapi`、`uvicorn`、`pydantic`、`pandas`、`numpy`、`torch`、`scikit-learn`、
`echarts`（前端图表库）、`pytest`。

---

## 4. 归属与增量边界（本项目的贡献声明）

为避免答辩与查重误解，明确以下三点：

1. **不属于本项目**：LightQuant 框架本身、CSMD 数据集本身。
2. **属于本项目**：四角色权限隔离的 Web 平台（`app/`）、两套回测引擎与成本模型
   （`backtest/`）、三视角研判与留痕、Point-in-Time 防穿越修复、全链路口径溯源、
   离线实验矩阵（`scripts/`，含消融 / 校准 / 权重稳定性 / 成本敏感性 / 撮合口径 /
   审核评测 / 分歧度分层 / 沪深300 成分基准）、121 项 pytest 用例（`tests/`）。
3. **量化口径**：应用层与实验层自研代码约 14,000 行，上游结构代码约 4,200 行
   （约 77% : 23%），统计口径见 `docs/数据来源与依赖声明.md`。

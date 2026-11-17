-- LightQuant 应用层数据库结构（SQLite）
-- 设计原则：每张表都对应题目里一个可演示的业务动作，不留"以后可能用得上"的空表。
-- 四角色（investor / author / risk / operator）在一处定义，后端与前端共用同一套枚举。

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- 1. 用户与角色 ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    display_name  TEXT,
    -- 演示用：切换角色时前端只换视图，后端仍按 role 校验，不做"前端隐藏即安全"
    role          TEXT NOT NULL CHECK (role IN ('investor', 'author', 'risk', 'operator')),
    -- PBKDF2-HMAC-SHA256（10 万轮）+ 随机盐，存 "salt$derivation"，从不存明文。
    -- 允许 NULL：已有库升级上来时先无密码，由 /api/auth/demo-users 补写。
    password_hash TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 2. 标的与行情（只入测试期，够演示、够回测复算；全量进 CSV 仍走文件）--------
CREATE TABLE IF NOT EXISTS stocks (
    symbol   TEXT PRIMARY KEY,           -- 用中文简称，与 dataset 里的文件名一致，避免二次映射
    name     TEXT,
    code     TEXT,                       -- 交易所代码，抓数据时用
    industry TEXT
);

CREATE TABLE IF NOT EXISTS prices (
    symbol    TEXT NOT NULL,
    date      TEXT NOT NULL,
    open      REAL, high REAL, low REAL, close REAL, adj_close REAL, volume REAL,
    label     INTEGER,                    -- 次日方向，与训练口径一致
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_prices_date ON prices (date);

-- 3. 新闻（离线原文，情绪面 Agent 的输入；覆盖稀疏，取不到就记缺失）-----------
CREATE TABLE IF NOT EXISTS news (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,
    title   TEXT,
    content TEXT,
    source  TEXT,
    UNIQUE (symbol, date, title)
);
CREATE INDEX IF NOT EXISTS idx_news_date ON news (date, symbol);

-- 4. 基本面（按报告期；CSMD50 原始数据里没有，必须一次性抓取入库）------------
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol         TEXT NOT NULL,
    period         TEXT NOT NULL,         -- 报告期，如 2024Q2
    announce_date  TEXT,                  -- 可用日（法定披露截止日），时点对齐用这个
    pe             REAL,
    pb             REAL,
    roe            REAL,                  -- 净资产收益率 %
    gross_margin   REAL,                  -- 销售毛利率 %
    revenue        REAL,                  -- 主营业务收入（元）
    net_profit     REAL,                  -- 净利润（元）
    revenue_yoy    REAL,                  -- 营收同比 %
    profit_yoy     REAL,                  -- 净利润同比 %
    debt_ratio     REAL,                  -- 资产负债率 %
    ocf_ps         REAL,                  -- 每股经营现金流（元）
    current_ratio  REAL,                  -- 流动比率
    inventory_turn REAL,                  -- 存货周转率（次）
    source         TEXT,
    fetched_at     TEXT,
    PRIMARY KEY (symbol, period)
);
-- 注意：(symbol, announce_date) 的索引不在这里建。老库升级时表已存在，
-- CREATE TABLE IF NOT EXISTS 不会补列，此时建索引会因缺 announce_date 报错。
-- 索引统一放到 db._migrate 补列之后创建。

-- 5. 估值（按交易日的 PE/PB 历史值）-----------------------------------------
-- 与 fundamentals 分开：fundamentals 是按报告期的财务口径（Q2/Q3 报表），
-- 这里是从公开估值接口取的历史序列。两者不能混用：报告期数据有公布滞后，
-- 估值序列已经是"当天市场就知道的数字"。
-- 使用红线：对齐到交易日时必须"取当天之前最近一条"（ffill），不得插值或回填，
-- 否则 2024-09 的判断会用到 2024-10 之后才知道的估值。
CREATE TABLE IF NOT EXISTS valuations (
    symbol  TEXT NOT NULL,
    date    TEXT NOT NULL,
    pe_ttm  REAL,
    pb      REAL,
    ps_ttm  REAL,
    source  TEXT,                        -- baidu / em / self_calc
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_valuations_date ON valuations (date);

-- 5b. 财报事件（从 news 抽出的结构化财务事实，基本面视角的「公司经营」依据）---
-- 为什么要这张表：fundamentals 表一直 0 行，基本面视角只剩下"估值分位"一个替身。
-- 但 news 里有大量带具体数字的财报报道（业绩预告、半年报、三季报），
-- 这类"净利润同比 +40%"的事实，恰恰是 PE 分位覆盖不到的"公司经营"维度。
-- subject_ok 是关键字段：新闻里常出现"中信证券保荐的某公司亏损"这种提及他人，
-- 不判主体归属就会把别家公司的事算到这只股票头上（正则抽取已经踩过）。
CREATE TABLE IF NOT EXISTS fund_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    date        TEXT NOT NULL,             -- 新闻发布日期
    news_id     INTEGER,                   -- 可回溯到 news.id
    event_type  TEXT,                      -- 业绩预告/业绩快报/半年报/三季报/年报/分红/其他
    subject_ok  INTEGER NOT NULL DEFAULT 1, -- 1=主体就是该股本身，0=提及他人
    np_yoy      REAL,                      -- 净利润同比增速（%），负值为下降
    rev_yoy     REAL,                      -- 营收同比增速（%）
    is_loss     INTEGER NOT NULL DEFAULT 0,
    direction   INTEGER,                   -- 1 偏多 / 0 中性 / -1 偏空（给人看）
    score       REAL,                      -- [-1,1] 连续分，与 tech/sent 同尺度（给融合用）
    confidence  REAL,
    evidence    TEXT,                      -- 原文依据片段（<=120 字）
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE (news_id)
);
CREATE INDEX IF NOT EXISTS idx_fund_events ON fund_events (symbol, date);

-- 6. 研判缓存（三视角 + 融合；没有 API Key / 断网时这里就是数据源）------------
-- 红线：必须记下 provider / model / temperature / prompt_version，
-- 否则答辩无法回答"这条结论是哪次、哪个模型、什么参数生成的"。
CREATE TABLE IF NOT EXISTS judgements (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol         TEXT NOT NULL,
    date           TEXT NOT NULL,
    view           TEXT NOT NULL CHECK (view IN ('tech', 'fund', 'sent', 'fusion')),
    signal         TEXT NOT NULL CHECK (signal IN ('BUY', 'SELL', 'HOLD')),
    score          REAL NOT NULL CHECK (score >= -1.0 AND score <= 1.0),
    reason         TEXT,
    provider       TEXT,                  -- local / deepseek / zhipu / qwen
    model          TEXT,
    temperature    REAL,
    prompt_version TEXT,
    source         TEXT NOT NULL DEFAULT 'live'
                   CHECK (source IN ('live', 'cache', 'offline_rule')),
    created_at     TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE (symbol, date, view, prompt_version)
);
CREATE INDEX IF NOT EXISTS idx_judgements_lookup ON judgements (symbol, date);

-- 6. 异步回测任务（PENDING -> RUNNING -> SUCCEEDED / FAILED）-----------------
CREATE TABLE IF NOT EXISTS backtest_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users (id),
    params_json TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'PENDING'
                CHECK (status IN ('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED')),
    progress    REAL NOT NULL DEFAULT 0.0 CHECK (progress >= 0.0 AND progress <= 1.0),
    result_path TEXT,                    -- 指向 result_v2 下的目录，不把指标复制一份进来
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON backtest_tasks (status, created_at);

-- 7. 策略（作者提交的回测结果 → 发布 → 投资 者订阅）------------------------
CREATE TABLE IF NOT EXISTS strategies (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    author_id    INTEGER REFERENCES users (id),
    name         TEXT NOT NULL,
    params_json  TEXT NOT NULL,
    metrics_json TEXT,                   -- 来自 summary.json 的 portfolio 段
    status       TEXT NOT NULL DEFAULT 'DRAFT'
                 CHECK (status IN ('DRAFT', 'PUBLISHED', 'ARCHIVED')),
    published_at TEXT
);

-- 8. 订阅与模拟跟单 ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS subscriptions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    investor_id   INTEGER NOT NULL REFERENCES users (id),
    strategy_id   INTEGER NOT NULL REFERENCES strategies (id),
    status        TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'CANCELLED')),
    subscribed_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    UNIQUE (investor_id, strategy_id)
);

-- 跟单账户：peak_nav 用来算回撤，stop_loss_nav 是风控员看的强平线
CREATE TABLE IF NOT EXISTS follow_accounts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL UNIQUE REFERENCES subscriptions (id),
    initial_nav    REAL NOT NULL,
    current_nav    REAL NOT NULL,
    peak_nav       REAL NOT NULL,
    stop_loss_nav  REAL NOT NULL,
    stop_loss_pct  REAL NOT NULL DEFAULT 0.10,   -- 该账户的强平线（相对峰值的回撤比例）
    status         TEXT NOT NULL DEFAULT 'ACTIVE' CHECK (status IN ('ACTIVE', 'LIQUIDATED')),
    -- 账户净值已经推演到哪个交易日。NULL 表示订阅后还没推过（当前净值等于初始值）。
    -- 有了这一列，"推进一日"才是把账户从 as_of 走到下一个交易日，而不是由前端写死一个日期。
    as_of          TEXT,
    updated_at     TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

-- 9. 风控事件（强平预警 / 已强平 / 发帖拦截，统一一张日志表）-----------------
CREATE TABLE IF NOT EXISTS risk_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id INTEGER REFERENCES follow_accounts (id),
    post_id    INTEGER,                  -- 发帖拦截时指向 posts.id
    kind       TEXT NOT NULL CHECK (kind IN ('MARGIN_CALL', 'LIQUIDATED', 'POST_BLOCKED')),
    detail     TEXT,
    handled_by INTEGER REFERENCES users (id),
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);
CREATE INDEX IF NOT EXISTS idx_risk_kind ON risk_events (kind, created_at);

-- 10. 社区发帖与审核 --------------------------------------------------------
-- strategy_id 可空：帖子可以只讲行情，也可以绑定一个已发布策略
-- （命题「社区牛人策略订阅跟投」需要帖子与策略之间的这条边）
CREATE TABLE IF NOT EXISTS posts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users (id),
    symbol      TEXT,
    strategy_id INTEGER REFERENCES strategies (id),
    content     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'PENDING'
                CHECK (status IN ('PENDING', 'APPROVED', 'BLOCKED')),
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS post_moderation (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id      INTEGER NOT NULL REFERENCES posts (id),
    rule_hit     TEXT,                   -- 命中的正则规则名，如 "guaranteed_return"
    llm_verdict  TEXT,                   -- LLM 的判定原文（可为空，断网时只有规则层）
    decision     TEXT NOT NULL CHECK (decision IN ('APPROVED', 'BLOCKED')),
    created_at   TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

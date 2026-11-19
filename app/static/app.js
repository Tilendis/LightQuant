/* LightQuant 前端。全部数据走本地 API，ECharts 本地加载，断网可用。 */
(function () {
  'use strict';

  var VIEWS = ['tech', 'fund', 'sent', 'fusion'];
  var VNAME = { tech: '技术面', fund: '基本面', sent: '情绪面', fusion: '融合' };
  var VCOLOR = { tech: '#2563eb', fund: '#0891b2', sent: '#ea580c', fusion: '#4d7c0f' };

  var state = {
    user: null,
    users: [],
    token: null,
    meta: null,
    tab: 'market',
    charts: {},
    lastAccountId: null,
    // 当前在持仓面板里展示的是哪个账户；换账户或取消订阅时要能被清掉，
    // 否则会把上一个账户的持仓留在屏幕上
    posAccountId: null,
    // 研判页当前展示的交易日；「重新研判」不填日期时用它
    judgeDate: '',
    // 交易日历与系统当前交易日：推进不再由前端写死日期，而是从日历算下一站
    tradeDays: [],
    cursor: null
  };

  /* ---------------- 工具 ---------------- */

  function api(path, opt) {
    opt = opt || {};
    var headers = { 'Content-Type': 'application/json' };
    // 身份走 Authorization: Bearer <token>，不再用可伪造的 X-User-Id 头。
    if (state.token) headers['Authorization'] = 'Bearer ' + state.token;
    return fetch(path, {
      method: opt.method || 'GET',
      headers: headers,
      body: opt.body ? JSON.stringify(opt.body) : undefined
    }).then(function (r) {
      return r.text().then(function (t) {
        var data = {};
        try { data = t ? JSON.parse(t) : {}; } catch (e) { data = { detail: t }; }
        if (!r.ok) {
          var d = data && data.detail;
          var msg = typeof d === 'string' ? d : (d ? JSON.stringify(d) : ('HTTP ' + r.status));
          var err = new Error(msg);
          err.status = r.status;
          throw err;
        }
        return data;
      });
    });
  }

  function el(id) { return document.getElementById(id); }

  function hint(id, text, cls) {
    var n = el(id);
    if (!n) return;
    n.textContent = text || '';
    n.className = 'hint' + (cls ? ' ' + cls : '');
  }

  function money(v) {
    if (v === null || v === undefined) return '—';
    return Number(v).toLocaleString('zh-CN', { maximumFractionDigits: 2 });
  }

  function pct(v, digits) {
    if (v === null || v === undefined || v === '') return '—';
    return (Number(v) * 100).toFixed(digits === undefined ? 2 : digits) + '%';
  }

  function num(v, digits) {
    if (v === null || v === undefined || v === '') return '—';
    return Number(v).toFixed(digits === undefined ? 3 : digits);
  }

  function metricCard(k, v, sub, cls) {
    return '<div class="metric"><div class="k">' + k + '</div>' +
      '<div class="v ' + (cls || '') + '">' + v + '</div>' +
      (sub ? '<div class="s">' + sub + '</div>' : '') + '</div>';
  }

  function ensureChart(id, height) {
    var wrap = el(id);
    if (!wrap) return null;
    wrap.style.height = height + 'px';
    var inner = wrap.querySelector('.ec-inner');
    if (!inner) {
      // 上一次可能因为权限不足在这里写过说明文字，画图前必须清掉，
      // 否则提示和图会叠在同一个容器里。
      if (wrap.innerHTML.replace(/\s/g, '')) wrap.innerHTML = '';
      inner = document.createElement('div');
      inner.className = 'ec-inner';
      inner.style.width = '100%';
      inner.style.height = '100%';
      wrap.appendChild(inner);
    }
    var c = state.charts[id];
    if (!c || c.isDisposed()) {
      c = echarts.init(inner);
      state.charts[id] = c;
    }
    return c;
  }

  function resizeAll() {
    Object.keys(state.charts).forEach(function (k) {
      var c = state.charts[k];
      if (c && !c.isDisposed()) c.resize();
    });
  }

  /* ---------------- 角色与标签页 ---------------- */

  var TABS = [
    { key: 'market', name: '行情研判' },
    { key: 'backtest', name: '策略回测' },
    { key: 'strategy', name: '策略与订阅' },
    { key: 'risk', name: '风控与社区' },
    { key: 'ops', name: '运营看板' }
  ];

  var ROLE_HOME = { investor: 'market', author: 'backtest', risk: 'risk', operator: 'ops' };
  var ROLE_NAME = { investor: '投资者', author: '策略作者', risk: '风控员', operator: '运营' };

  function buildRoles() {
    var box = el('roles');
    box.innerHTML = '';
    state.users.forEach(function (u) {
      var b = document.createElement('button');
      b.className = 'role-btn' + (state.user && u.id === state.user.id ? ' on' : '');
      b.textContent = ROLE_NAME[u.role] || u.role;
      b.onclick = function () { switchUser(u); };
      box.appendChild(b);
    });
  }

  function buildTabs() {
    var box = el('tabs');
    box.innerHTML = '';
    TABS.forEach(function (t) {
      var b = document.createElement('button');
      b.className = 'tab' + (state.tab === t.key ? ' on' : '');
      b.textContent = t.name;
      b.onclick = function () { switchTab(t.key); };
      box.appendChild(b);
    });
  }

  function login(username, password) {
    return fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: username, password: password })
    }).then(function (r) {
      return r.text().then(function (t) {
        var data = {};
        try { data = t ? JSON.parse(t) : {}; } catch (e) { data = {}; }
        if (!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
        return data;
      });
    });
  }

  function switchUser(u) {
    hint('status', '登录中…', '');
    return login(u.username, state.demoPassword).then(function (res) {
      state.token = res.token;
      state.user = res.user;
      // 换角色必须清掉上一个角色看过的持仓：否则投资者切到风控员后，
      // 屏幕上还留着"别人的"持仓表，读起来像是风控员查的
      hidePositions();
      buildRoles();
      switchTab(ROLE_HOME[u.role] || 'market');
      hint('status', ROLE_NAME[u.role] + '（id=' + u.id + '，已签发 token）', 'ok');
    }).catch(function (e) {
      hint('status', '登录失败：' + e.message, 'bad');
    });
  }

  function switchTab(key) {
    state.tab = key;
    Array.prototype.forEach.call(document.querySelectorAll('.panel'), function (p) {
      p.classList.remove('on');
    });
    var p = el('panel-' + key);
    if (p) p.classList.add('on');
    buildTabs();
    setTimeout(resizeAll, 30);

    if (key === 'market') loadMarket();
    if (key === 'backtest') loadBacktestPanel();
    if (key === 'strategy') loadStrategy();
    if (key === 'risk') loadRisk();
    if (key === 'ops') loadOps();
  }

  /* ---------------- 图 1：K 线 + 四视角信号 ---------------- */

  function loadMarket() {
    var sel = el('symSel');
    if (sel.options.length === 0) return;
    var sym = sel.value;
    if (!sym) return;
    api('/api/market/kline/' + encodeURIComponent(sym)).then(function (d) {
      renderKline(d);
      hint('marketHint', sym + '　' + d.bars.length + ' 个交易日', 'ok');
      loadJudge(sym);
    }).catch(function (e) {
      hint('marketHint', '读取失败：' + e.message, 'bad');
    });
  }

  function renderKline(d) {
    var c = ensureChart('chart1', 560);
    if (!c) return;
    var bars = d.bars;
    var dates = [], ohlc = [], vols = [];
    bars.forEach(function (b) {
      dates.push(b.date);
      ohlc.push([b.open, b.close, b.low, b.high]);
      vols.push(b.volume);
    });

    var series = [{
      name: 'K线', type: 'candlestick', data: ohlc,
      itemStyle: { color: '#d14343', color0: '#2b8a3e', borderColor: '#d14343', borderColor0: '#2b8a3e' }
    }];

    VIEWS.forEach(function (v, vi) {
      var pts = [];
      bars.forEach(function (b, bi) {
        var s = b.signals && b.signals[v];
        if (!s || s.score === null || s.score === undefined) return;
        if (s.signal === 'HOLD' || !s.signal) return;
        var off = 0.013 * (vi + 1);
        var y = s.signal === 'BUY' ? b.low * (1 - off) : b.high * (1 + off);
        pts.push({
          value: [bi, Number(y.toFixed(2))],
          symbol: 'triangle',
          symbolRotate: s.signal === 'BUY' ? 0 : 180,
          symbolSize: 8,
          itemStyle: { color: VCOLOR[v], opacity: 0.8 }
        });
      });
      series.push({
        name: VNAME[v], type: 'scatter', data: pts,
        itemStyle: { color: VCOLOR[v] },
        z: 10, symbolSize: 8
      });
    });

    c.setOption({
      backgroundColor: '#fff',
      animation: false,
      title: {
        text: d.symbol + ' 四视角研判叠加',
        subtext: '▲ 看多 / ▼ 看空，颜色区分视角（' + dates[0] + ' ~ ' + dates[dates.length - 1] + '）',
        left: 8, top: 4,
        textStyle: { fontSize: 14, fontWeight: 600 },
        subtextStyle: { fontSize: 11, color: '#6b7280' }
      },
      legend: { top: 8, right: 12, itemWidth: 12, itemHeight: 8, textStyle: { fontSize: 11 } },
      tooltip: {
        trigger: 'axis', triggerOn: 'mousemove|click',
        axisPointer: { type: 'cross' }, confine: true,
        textStyle: { fontSize: 11 }
      },
      grid: { left: 60, right: 24, top: 64, bottom: 52, containLabel: false },
      xAxis: {
        type: 'category', data: dates, boundaryGap: true,
        axisLabel: { fontSize: 10, interval: Math.max(1, Math.floor(dates.length / 10)), rotate: 30 },
        axisLine: { lineStyle: { color: '#dfe3e8' } }
      },
      yAxis: {
        // 不写死 min/max：写死会按全序列定标，dataZoom 只看一段时
        // K 线被压在图的下半部，四个视角的信号点也会挤成一条。
        scale: true,
        axisLabel: { fontSize: 10 },
        splitLine: { lineStyle: { color: '#f0f2f5' } }
      },
      dataZoom: [{ type: 'inside', start: 0, end: 100 }],
      series: series
    }, true);
  }

  /* ---------------- 研判报告：四视角文本理由 ---------------- */

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  /* 公司财务状态块：基本面三层里"公司财务"这一层的原始指标。
     它是风险描述，不是交易信号——实测该层对次日涨跌 IC≈0（T+1 −0.0237、
     T+5 −0.0615），融合权重为 0。所以这里只展示"现在是什么状态"，
     不输出 BUY/SELL，也不进加权。 */
  function finCell(label, val, unit, yoyKey, yoy, goodWhenUp) {
    var v = (val === null || val === undefined)
      ? '—' : Number(val).toFixed(2) + (unit || '');
    var dv = yoy ? yoy[yoyKey] : null;
    var dy;
    if (dv === null || dv === undefined) {
      dy = '<span class="fin-delta">—</span>';
    } else {
      // 变化的方向含义随指标而变：ROE 升是改善，资产负债率升是恶化。
      // 统一用红绿表示"涨跌"会把"负债率上升"画成红色，等于说它变好了。
      // 单位也要跟着走：只有百分比指标能叫 pp，现金流是元、周转率是次、流动比率无量纲。
      var dUnit = (unit === '%') ? 'pp' : (unit || '');
      var better = goodWhenUp ? dv >= 0 : dv <= 0;
      dy = '<span class="fin-delta ' + (better ? 'good' : 'bad') + '">'
        + (dv >= 0 ? '↑ +' : '↓ ') + Number(dv).toFixed(2) + dUnit + '</span>';
    }
    return '<div class="fin-cell"><div class="fin-k">' + label + '</div>'
      + '<div class="fin-v">' + v + '</div>'
      + '<div class="fin-y">' + dy + '</div></div>';
  }

  function finBlock(rows) {
    var head = '<div class="fin-title">公司财务'
      + '<span class="muted">　基本面三层中的「公司财务」层：描述当前财务状态，'
      + '不参与方向预测（该层对次日涨跌 IC≈0，融合权重 0）</span></div>';
    if (!rows || !rows.length) {
      return '<div class="fin-block">' + head
        + '<div class="muted" style="font-size:11px">该标的暂无财务记录</div></div>';
    }
    var r = rows[0];
    var y = r.yoy || {};
    var note = '截至 <b>' + esc(r.period) + '</b>（' + esc(r.announce_date) + ' 披露）'
      + (r.yoy_period
        ? '　第三行 ＝ 与 ' + esc(r.yoy_period) + ' 的差（百分比指标记 pp，现金流记元、周转率记次）；'
          + '营收同比与净利同比那一行是<b>增速</b>变化，不是营收涨了这么多'
        : '　同比基线缺失');
    return '<div class="fin-block">' + head
      + '<div class="fin-meta muted">' + note + '</div>'
      + '<div class="fin-grid">'
      + finCell('ROE', r.roe, '%', 'roe', y, true)
      + finCell('毛利率', r.gross_margin, '%', 'gross_margin', y, true)
      + finCell('营收同比', r.revenue_yoy, '%', 'revenue_yoy', y, true)
      + finCell('净利同比', r.profit_yoy, '%', 'profit_yoy', y, true)
      + finCell('资产负债率', r.debt_ratio, '%', 'debt_ratio', y, false)
      + finCell('每股经营现金流', r.ocf_ps, ' 元', 'ocf_ps', y, true)
      + finCell('流动比率', r.current_ratio, '', 'current_ratio', y, true)
      + finCell('存货周转率', r.inventory_turn, ' 次', 'inventory_turn', y, true)
      + '</div></div>';
  }

  function loadJudge(sym) {
    var box = el('judgeBox');
    if (!box) return;
    var buf = {}, left = VIEWS.length + 3;
    // fusion 在库里有九个版本并存，不指定版本会混排并截断日期轴。
    // auto 的语义是"按后端 PV_PREF 的优先级逐日取版本"，对应后端那套三段链：
    //   2024-12-16~12-31  fusion-agent-v1（三智能体链路）
    //   2024-08-08~12-31  fusion-test-3view（测试段标定 0.90/0.00/0.10）
    //   2024-03-15~08-07  fusion-val-3view（验证段标定 同上）
    // 技术面/基本面/情绪面是两段：智能体版 12 天优先，更早落回 tech-v1 / fund-rule-v1 / sent-v1。
    var VER = { tech: 'auto', fund: 'auto', sent: 'auto', fusion: 'auto' };
    // 编排层与主融合表在库里同为 view=fusion，只能靠版本区分，所以走 view=orch：
    // auto 语义同其它视角——智能体版编排（fusion-orch-agent-v1）优先，缺失回退规则版。
    var ORCH_VIEW = 'orch';
    // 技术面的大模型解读版（tech-llm-v1）与模型版并存，单独取一行做对照展示
    var TECHLLM = 'tech-llm-v1';
    VIEWS.forEach(function (v) {
      api('/api/judgements/' + encodeURIComponent(sym) + '?view=' + v
        + '&prompt_version=' + encodeURIComponent(VER[v]) + '&limit=2000')
        .then(function (rows) { buf[v] = rows || []; },
              function () { buf[v] = []; })
        .then(function () {
          left -= 1;
          if (left === 0) renderJudge(sym, buf);
        });
    });
    api('/api/judgements/' + encodeURIComponent(sym) + '?view=' + ORCH_VIEW
      + '&prompt_version=auto&limit=2000')
      .then(function (rows) { buf.orch = rows || []; },
            function () { buf.orch = []; })
      .then(function () {
        left -= 1;
        if (left === 0) renderJudge(sym, buf);
      });
    api('/api/judgements/' + encodeURIComponent(sym) + '?view=tech'
      + '&prompt_version=' + TECHLLM + '&limit=2000')
      .then(function (rows) { buf.techllm = rows || []; },
            function () { buf.techllm = []; })
      .then(function () {
        left -= 1;
        if (left === 0) renderJudge(sym, buf);
      });
    // 基本面·财报事件版（fund-agent-v2）：LLM 读财报事件 + 行业归属 + 估值分位写出的
    // 完整研判报告。覆盖两段：val 段 2024-03-15~08-07（98 个交易日）与
    // 12 月窗 12-16~12-31（12 个交易日），中间 08-08~12-15 无记录。
    // 它与 fund 那个估值分位版是两个口径，所以单独取一路，不并进 VIEWS。
    api('/api/judgements/' + encodeURIComponent(sym) + '?view=fund'
      + '&prompt_version=fund-agent-v2&limit=2000')
      .then(function (rows) { buf.fundv2 = rows || []; },
            function () { buf.fundv2 = []; })
      .then(function () {
        left -= 1;
        if (left === 0) renderJudge(sym, buf);
      });
  }

  // 取融合视角最新交易日作为展示日，同日对齐其余三个视角
  function renderJudge(sym, buf) {
    var box = el('judgeBox');
    if (!box) return;
    var fus = buf.fusion || [];
    if (fus.length === 0) {
      box.innerHTML = '<span class="muted">该标的暂无研判记录</span>';
      return;
    }
    var d = fus[fus.length - 1].date;
    state.judgeDate = d;   // 「重新研判」的日期留空时用这一天
    // 公司财务块要等融合最新日才能定"用哪一期"，所以在这里补一路请求：
    // 传 date 让后端按 announce_date <= date 判已披露，时点口径只有一处实现。
    // finAsked 保证一趟渲染只发一次，避免 renderJudge 被多次回调触发后重复请求。
    if (!buf.finAsked) {
      buf.finAsked = 1;
      api('/api/fundamentals/' + encodeURIComponent(sym)
        + '?date=' + encodeURIComponent(d) + '&limit=4')
        .then(function (rows) { buf.fin = rows || []; },
              function () { buf.fin = []; })
        .then(function () { renderJudge(sym, buf); });
    }
    function findOne(v) {
      var rows = buf[v] || [];
      for (var i = 0; i < rows.length; i++) {
        if (rows[i].date === d) return rows[i];
      }
      return null;
    }
    // 技术面的大模型解读版：同一交易日、同一技术面结论，由 glm-4-flash 结合行情指标复述理由
    function techLlmRow() {
      var rows = buf.techllm || [];
      var r = null;
      for (var i = 0; i < rows.length; i++) {
        if (rows[i].date === d) r = rows[i];
      }
      if (!r) {
        return '<tr><td class="jv">技术面·大模型解读</td>'
          + '<td class="muted">—</td><td class="muted">当日无记录</td><td class="muted">—</td></tr>';
      }
      var sc = Number(r.score);
      var cls = r.signal === 'BUY' ? 'up' : (r.signal === 'SELL' ? 'down' : 'flat');
      if (r.source && r.source !== 'live') offCount += 1;
      return '<tr><td class="jv">技术面·大模型解读</td>'
        + '<td class="sig ' + cls + '">' + esc(r.signal) + ' '
        + (sc >= 0 ? '+' : '') + sc.toFixed(2) + '</td>'
        + '<td class="rs">' + esc(r.reason || '') + '</td>'
        + '<td class="src">' + esc(r.model || '') + ' · 底分来自本地模型'
        + (r.source && r.source !== 'live'
          ? ' · <span class="offline-tag">离线预计算</span>' : '') + '</td></tr>';
    }
    var offCount = 0;   // 本日展示里有多少行来自离线预计算（source != 'live'）
    // 表头必须跟着当日实际读到的版本走：12-16 之后是三智能体研判，
    // 再套"技术面=本地模型、基本面=规则"的旧话术就等于把新数据说成旧链路。
    var frow = fus[fus.length - 1];
    var pv = String(frow.prompt_version || '');
    var isAgent = pv.indexOf('-agent-') >= 0;
    var headNote = isAgent
      ? '　技术面／基本面＝大模型智能体独立研判（各自调用本地指标、估值分位作为工具，彼此不可见）；情绪面＝新闻文本打分，只覆盖有新闻的交易日（test段覆盖率 19.0%，全段 21.2%），无新闻日弃权、不参与加权；融合＝该段输入是三智能体结论，权重绑该链路（tech 1.0／fund 0／sent 0.5），与新标定 0.9/0/0.1 不是同一套，勿混读'
      : '　技术面＝本地 BiLSTM／基本面＝本地规则／情绪面＝大模型文本研判；融合＝验证集标定权重 0.9/0/0.1 合成（基本面权重归零是验证集判定的结果，不是缺数据）';
    var head = '<div class="judge-head">' + esc(sym) + '　' + esc(d) + ' 研判报告'
      + '<span class="muted">' + headNote + '</span>'
      + '</div>';
    var body = '';
    VIEWS.forEach(function (v) {
      var r = findOne(v);
      if (!r) {
        // 情绪面按新闻日落库，无新闻的交易日靠 ffill 外推到融合层，这里必须说明，
        // 否则会出现"情绪面当日无记录、融合里却有 sent 分"的看不明
        var miss = v === 'sent'
          ? (isAgent
            ? '无新闻覆盖，情绪面弃权（不落库、不参与加权）'
            : '当日无新闻，融合沿用最近一次情绪分（ffill 外推）')
          : '当日无记录';
        body += '<tr><td class="jv">' + esc(VNAME[v]) + '</td>'
          + '<td class="muted">—</td><td class="muted">' + esc(miss) + '</td><td class="muted">—</td></tr>';
        return;
      }
      var sc = Number(r.score);
      var cls = r.signal === 'BUY' ? 'up' : (r.signal === 'SELL' ? 'down' : 'flat');
      var src = esc(r.model || '') + (r.provider && r.provider !== 'local'
        ? ' · ' + esc(r.provider) : ' · 本地');
      // 融合表有九个版本并存（旧加权/旧投票/三段标定/智能体版），
      // 只写"融合"两个字看不出这一行读的是旧权重还是标定后的权重，必须把版本写出来。
      // 三视角同理：基本面在库里有 rule/v1/v2/v3 四个版本，材料完全不同，
      // 不写版本就分不清这一行读的是只有估值分位的 rule 还是带财务的 v3。
      if (r.prompt_version && (v === 'fusion'
          || String(r.prompt_version).indexOf('-agent-') >= 0)) {
        src += ' · ' + esc(r.prompt_version);
      }
      // source 记录这条结论"怎么来的"：live=本次实时调用，cache / offline_rule=离线预计算。
      // 断网演示与在线演示必须在界面上能区分，否则说不清这个分数是哪来的。
      // 例外只有基线版的基本面（fund-rule-v1 全部 offline_rule），它是本地规则，
      // 标成"离线预计算"会让人误以为是降级；智能体版的基本面是真调了模型，照常标。
      var isBaseFund = v === 'fund'
        && String(r.prompt_version || '').indexOf('fund-rule') >= 0;
      if (!isBaseFund && r.source && r.source !== 'live') {
        offCount += 1;
        src += ' · <span class="offline-tag">离线预计算</span>';
      }
      body += '<tr><td class="jv">' + esc(VNAME[v]) + '</td>'
        + '<td class="sig ' + cls + '">' + esc(r.signal) + ' '
        + (sc >= 0 ? '+' : '') + sc.toFixed(2) + '</td>'
        + '<td class="rs">' + esc(r.reason || '') + '</td>'
        + '<td class="src">' + src + '</td></tr>';
      // 解读层只对基线技术面有意义（本地模型给底分、LLM 复述理由）；
      // 智能体版技术面本身就是 LLM 判断，再叠一层解读是重复。
      if (v === 'tech' && !isAgent) body += techLlmRow();
    });
    // 基本面·财报事件版：展示日当天若在覆盖段内就取当天，否则取之前最近一条并标日期。
    // 这一版经过验证集实测不参与方向加权（2437 样本 ACC 0.4735、z=-2.61），
    // 所以它在界面上是"研判报告"而不是"可交易信号"，行内必须写明这一点。
    var fv2 = null;
    var fv2rows = buf.fundv2 || [];
    for (var q = 0; q < fv2rows.length; q++) {
      if (fv2rows[q].date <= d) fv2 = fv2rows[q];
    }
    if (fv2) {
      var v2sc = Number(fv2.score);
      var v2cls = fv2.signal === 'BUY' ? 'up' : (fv2.signal === 'SELL' ? 'down' : 'flat');
      var v2day = fv2.date === d ? '' : '（' + esc(fv2.date) + '）';
      if (fv2.source && fv2.source !== 'live') offCount += 1;
      body += '<tr><td class="jv">基本面·财报事件版' + v2day + '</td>'
        + '<td class="sig ' + v2cls + '">' + esc(fv2.signal) + ' '
        + (v2sc >= 0 ? '+' : '') + v2sc.toFixed(2) + '</td>'
        + '<td class="rs">' + esc(fv2.reason || '') + '</td>'
        + '<td class="src">' + esc(fv2.model || '')
        + ' · 覆盖 03-15~08-07 与 12-16~12-31 · 不参与方向加权'
        + (fv2.source && fv2.source !== 'live'
          ? ' · <span class="offline-tag">离线预计算</span>' : '') + '</td></tr>';
    }
    // 编排层：先判三视角是否形成多空对立，对立才进取证分支；一致或轻微分歧直接规则合成。
    // 它落在 fusion 视图的另一个版本里，只在有记录的日期显示，不参与回测信号。
    var orc = null;
    var orows = buf.orch || [];
    for (var k = 0; k < orows.length; k++) {
      if (orows[k].date === d) { orc = orows[k]; break; }
    }
    // 编排层已跟着智能体版本补跑 12 天（fusion-orch-agent-v1，2024-12-16~12-31），
    // 输入就是 agent 版技术面/基本面 + 情绪面，因此在智能体日期也应当显示；
    // auto 版本选择保证 12 月用 agent 版编排、更早日期回退规则版。
    if (orc) {
      var osc = Number(orc.score);
      var ocls = orc.signal === 'BUY' ? 'up' : (orc.signal === 'SELL' ? 'down' : 'flat');
      if (orc.source && orc.source !== 'live') offCount += 1;
      body += '<tr><td class="jv">编排层</td>'
        + '<td class="sig ' + ocls + '">' + esc(orc.signal) + ' '
        + (osc >= 0 ? '+' : '') + osc.toFixed(2) + '</td>'
        + '<td class="rs">' + esc(orc.reason || '') + '</td>'
        + '<td class="src">' + esc(orc.model || '') + ' · ' + esc(orc.provider || '')
        + (orc.source && orc.source !== 'live'
          ? ' · <span class="offline-tag">离线预计算</span>' : '') + '</td></tr>';
    }
    var offNote = offCount > 0
      ? '<span class="offline-tag">本次展示有 ' + offCount + ' 项来自离线预计算</span>' : '';
    box.innerHTML = head + offNote
      + '<table class="grid judge-tb"><thead><tr><th>视角</th><th>信号/分数</th>'
      + '<th>研判理由</th><th>来源</th></tr></thead><tbody>' + body + '</tbody></table>'
      + finBlock(buf.fin);
  }

  /* ---------------- 现场调度三视角（课题 R1 的"调度"落点） ----------------
     为什么单独一块、不并进上面的主表：主表读的是 auto 版本（主口径），
     重跑写的是独立版本 rerun-<时间戳>。两者分开才谈得上"可核对、可清理"，
     也才不会演示一次就把主口径数据改掉。 */
  function renderJudgeRun(res) {
    var box = el('judgeRunBox');
    if (!box) return;
    var rows = '';
    ['tech', 'fund', 'sent'].forEach(function (v) {
      var p = res.views[v] || {};
      if (p.status !== 'ok') {
        rows += '<tr><td class="jv">' + esc(VNAME[v]) + '</td>'
          + '<td class="muted">—</td>'
          + '<td class="rs muted">' + esc(p.reason || p.error || '未产出') + '</td>'
          + '<td class="src muted">' + esc(p.status) + '</td></tr>';
        return;
      }
      var cls = p.signal === 'BUY' ? 'up' : (p.signal === 'SELL' ? 'down' : 'flat');
      rows += '<tr><td class="jv">' + esc(VNAME[v]) + '</td>'
        + '<td class="sig ' + cls + '">' + esc(p.signal) + ' '
        + (Number(p.score) >= 0 ? '+' : '') + Number(p.score).toFixed(2) + '</td>'
        + '<td class="rs">' + esc(p.reason || '') + '</td>'
        + '<td class="src">' + esc(p.model || '') + ' · ' + esc(p.source || '')
        + ' · ' + Number(p.ms || 0) + 'ms</td></tr>';
    });
    var f = res.fusion;
    if (f) {
      var fcls = f.signal === 'BUY' ? 'up' : (f.signal === 'SELL' ? 'down' : 'flat');
      rows += '<tr><td class="jv">融合</td>'
        + '<td class="sig ' + fcls + '">' + esc(f.signal) + ' '
        + (Number(f.score) >= 0 ? '+' : '') + Number(f.score).toFixed(2) + '</td>'
        + '<td class="rs">' + esc(f.reason || '') + '</td>'
        + '<td class="src">rule_weighted · 权重 ' + esc(JSON.stringify(f.weights || {})) + '</td></tr>';
    }
    var det = res.demean || {};
    var detTxt = Object.keys(det).map(function (k) {
      var x = det[k] || {};
      return VNAME[k] + (x.applied ? '已去均值（当日 ' + x.n + ' 只，均值 ' + x.mean + '）' : '未去均值：' + (x.reason || ''));
    }).join('；');
    box.innerHTML = '<div class="judge-head">现场调度　' + esc(res.symbol) + '　' + esc(res.date)
      + '<span class="muted">　版本 ' + esc(res.prompt_version) + '　耗时 '
      + Number(res.elapsed_ms || 0) + 'ms</span></div>'
      + '<table class="grid judge-tb"><thead><tr><th>视角</th><th>信号/分数</th>'
      + '<th>研判理由</th><th>来源</th></tr></thead><tbody>' + rows + '</tbody></table>'
      + '<div class="note">' + esc(detTxt) + '</div>'
      + (res.skipped && res.skipped.length
        ? '<div class="note">弃权视角：' + esc(res.skipped.map(function (v) { return VNAME[v]; }).join('、'))
        + '（没有材料不等于判中性，弃权不参与加权）</div>' : '');
  }

  function runJudge() {
    var sym = el('symSel') ? el('symSel').value : '';
    var d = (el('judgeDate') && el('judgeDate').value) || state.judgeDate || '';
    if (!sym) return;
    if (!d) {
      hint('judgeRunHint', '还没拿到交易日：先在标的框选一只，或手填日期', 'bad');
      return;
    }
    var btn = el('judgeRunBtn');
    btn.disabled = true;
    btn.textContent = '调度中…';
    hint('judgeRunHint', '正在调度：技术面（本地 BiLSTM）／基本面（估值分位规则）／情绪面（大模型，无 key 则降级词典）', '');
    api('/api/judgements/run', {
      method: 'POST',
      body: { symbol: sym, date: d, allow_llm: true, write: true }
    }).then(function (res) {
      renderJudgeRun(res);
      hint('judgeRunHint', '已写入 ' + res.prompt_version + '（' + res.written.length
        + ' 行，' + res.elapsed_ms + 'ms）；上面主表读的是主口径，不受影响', 'ok');
    }, function (e) {
      hint('judgeRunHint', '调度失败：' + e.message, 'bad');
      var b = el('judgeRunBox');
      if (b) b.innerHTML = '';
    }).then(function () {
      btn.disabled = false;
      btn.textContent = '重新研判';
    });
  }

  function runJudgeBatch() {
    var syms = el('judgeBatchSyms').value.split(/[,，\s]+/).filter(function (s) { return s; });
    var start = el('judgeBatchStart').value;
    var end = el('judgeBatchEnd').value;
    var box = el('judgeBatchBox');
    if (!syms.length || !start || !end) {
      hint('judgeBatchHint', '请填标的列表与起止日期', 'bad');
      return;
    }
    var btn = el('judgeBatchBtn');
    btn.disabled = true;
    btn.textContent = '批量跑中…';
    box.innerHTML = '<span class="muted">调度中：' + syms.length + ' 只 × 区间 ' + start + ' ~ ' + end + '（dry-run，不写库不调 LLM）</span>';
    api('/api/judgements/run', {
      method: 'POST',
      body: {
        symbols: syms,
        date_start: start,
        date_end: end,
        allow_llm: false,
        write: false
      }
    }).then(function (res) {
      var rows = (res.items || []).map(function (it) {
        var f = it.fusion || {};
        var sk = (it.skipped || []).join('/') || '—';
        return '<tr><td>' + it.symbol + '</td><td>' + it.date + '</td><td>' + (f.signal || '—') + '</td><td>' + (f.score == null ? '—' : f.score.toFixed(3)) + '</td><td>' + sk + '</td></tr>';
      }).join('');
      box.innerHTML =
        '<div>共用版本 <code>' + res.prompt_version + '</code>：' +
        '共 ' + res.total + ' 项，成功 ' + res.ok_count + '，失败 ' + res.fail_count +
        '，跳过视角 ' + res.skipped_views + ' 项</div>' +
        '<table class="grid"><thead><tr><th>标的</th><th>日期</th><th>融合</th><th>融合分</th><th>跳过视角</th></tr></thead><tbody>' +
        rows + '</tbody></table>';
      hint('judgeBatchHint', '批量完成：' + res.ok_count + '/' + res.total, '');
    }).catch(function (e) {
      box.innerHTML = '';
      hint('judgeBatchHint', '批量失败：' + e.message, 'bad');
    }).then(function () {
      btn.disabled = false;
      btn.textContent = '批量跑（dry-run）';
    });
  }

  /* ---------------- 图 2：净值 + 买卖点 + 回撤阴影 ---------------- */

  function loadBacktestPanel() {
    /* 先把磁盘结果拿到手：白名单之外的历史调参/联调目录由服务端列为 hidden_tags，
       任务表里同名的旧任务也要一并藏起来，否则下拉又被历史 tag 灌满。
       新提交的回测 tag 不在 hidden_tags 中，提交后照样能看到。 */
    Promise.all([
      api('/api/backtest/reports'),
      api('/api/backtest/tasks?limit=50')
    ]).then(function (res) {
      var rep = res[0] || {};
      var tasks = res[1] || [];
      var sel = el('tagSel');
      var tsel = el('taskSel');
      sel.innerHTML = '';
      tsel.innerHTML = '';
      var hidden = {};
      ((rep.hidden_tags) || []).forEach(function (t) { hidden[t] = 1; });
      // 同一个 tag 可能被提交过多次（如 api_warm 跑了两遍），下拉里只留一个
      var seenTag = {};
      tasks.forEach(function (t) {
        var tag = (t.params && t.params.tag) || '';
        if (tag && !seenTag[tag] && !hidden[tag]) {
          seenTag[tag] = 1;
          var o = document.createElement('option');
          o.value = tag;
          o.textContent = tag + '（' + t.status + '）';
          sel.appendChild(o);
        }
        if (t.status === 'SUCCEEDED') {
          var o2 = document.createElement('option');
          o2.value = t.id;
          o2.textContent = '#' + t.id + ' ' + tag;
          tsel.appendChild(o2);
        }
      });

      /* 合并磁盘结果：命令行跑的回测不写任务表，只靠 tasks 填充会让演示时选不到新结果。 */
      var have = {};
      for (var k = 0; k < sel.options.length; k++) { have[sel.options[k].value] = 1; }
      ((rep.reports) || []).forEach(function (r) {
        if (!r.tag || have[r.tag]) return;
        var o = document.createElement('option');
        o.value = r.tag;
        // 区分"融合表"和"单视角表"：两者读数来源不同，混着标会让口径失真
        var suffix = '磁盘结果';
        if (r.views && r.views.length) {
          suffix = (r.views.indexOf('fusion') >= 0
            ? '融合 · ' + (r.fusion_version || '版本未记录')
            : '单视角 · ' + r.views.join('+'));
        }
        o.textContent = r.tag + '（' + suffix + '）';
        sel.insertBefore(o, sel.firstChild);
        have[r.tag] = 1;
      });

      // 默认落在主口径，而不是随手一个历史 tag：张口就错的口径比列表乱更致命
      if (sel.options.length) {
        var def = null;
        for (var i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === 'test_fusion_3view_v2') { def = sel.options[i].value; break; }
        }
        sel.value = def || sel.options[0].value;
      }
      if (sel.value && !state.charts['chart2']) loadReport(sel.value);
    }).catch(function (e) {
      hint('btHint', '任务列表读取失败：' + e.message, 'bad');
    });
  }

  function loadReport(tag) {
    api('/api/backtest/' + encodeURIComponent(tag) + '/report').then(function (d) {
      renderReport(d);
    }).catch(function (e) {
      hint('btHint', '报告读取失败：' + e.message, 'bad');
    });
  }

  // 口径卡片：答辩问"加权是怎么做的、权重在哪"时，界面必须直接答得出来。
  // 融合表的权重在生成阶段就已固化，回测端不再重复加权，所以这里写明"固化"，
  // 而不是让读者以为回测时还会按提交参数再投一次票。
  function fusionNote(cfg) {
    if ((cfg.strategy_type || '') === 'cross_section') {
      // 截面策略只有一个视角参与排序，说成"多视角直连"会让人以为三视角都在投票
      return '截面选股 · 仅技术面分数参与排序（其余视角无截面排序能力）';
    }
    var views = (cfg.views || []);
    if (views.indexOf('fusion') >= 0) {
      // 对 fusion 表，回测端 method 恒为 weighted（不再重复投票/加权），
      // 真正决定"投票还是加权"的是读了哪一张融合表；所以看 prompt_version，
      // 不能看 method——否则读 fusion-vote-v1 会被标成"加权融合"。
      var pv = String(cfg.prompt_version || '');
      var m = pv.indexOf('vote') >= 0 ? '投票融合' : '加权融合';
      return m + ' · 权重在融合阶段固化（回测不重复加权）';
    }
    return '多视角直连 · 权重按提交参数或等权';
  }

  function renderReport(d) {
    var pf = d.portfolio || {};
    var cfg = d.config || {};
    var cards = el('btCards');
    cards.innerHTML =
      metricCard('累计收益', pct(pf.cum_return), tagText(d), pf.cum_return >= 0 ? 'up' : 'down') +
      metricCard('年化收益', pct(pf.annual_return), '按 242 交易日折算') +
      metricCard('夏普比率', num(pf.sharpe), 'rf = 2%') +
      metricCard('最大回撤', pct(pf.mdd), (d.mdd_range.start || '') + ' ~ ' + (d.mdd_range.end || ''), 'down') +
      metricCard('信息比率', num(pf.ir), '相对等权基准') +
      metricCard('成交笔数', pf.trades || '—', '双边计') +
      metricCard('读表口径', cfg.prompt_version || '单视角', fusionNote(cfg));

    var c = ensureChart('chart2', 460);
    if (!c) return;
    var nav = d.nav || [];
    var dates = [], navV = [], benchV = [];
    var navByDate = {};
    nav.forEach(function (x) {
      dates.push(x.date);
      navV.push(x.nav);
      benchV.push(x.bench);
      navByDate[x.date] = x.nav;
    });

    // 买卖点聚到当日净值上：同一天同方向只画一个点，避免重叠成一团
    var buys = {}, sells = {};
    (d.trades || []).forEach(function (t) {
      var y = navByDate[t.date];
      if (y === undefined) return;
      if (t.side === 'BUY') buys[t.date] = y;
      else if (t.side === 'SELL') sells[t.date] = y;
    });
    var mkPts = function (obj, up) {
      return Object.keys(obj).map(function (k) {
        return {
          value: [k, obj[k]],
          symbol: 'triangle', symbolRotate: up ? 0 : 180, symbolSize: 7,
          // 同一天常常同时有买入和卖出（调仓日），两点净值相同会互相遮住，
          // 故用像素偏移把它们分开：买入上移、卖出下移。
          symbolOffset: [0, up ? -9 : 9],
          itemStyle: { color: up ? '#d14343' : '#2b8a3e', opacity: 0.85 }
        };
      });
    };

    var area = [];
    if (d.mdd_range && d.mdd_range.start && d.mdd_range.end) {
      area = [[{ xAxis: d.mdd_range.start }, { xAxis: d.mdd_range.end }]];
    }

    c.setOption({
      backgroundColor: '#fff',
      animation: false,
      title: { text: '净值曲线与买卖点', left: 8, top: 4,
        textStyle: { fontSize: 14, fontWeight: 600 },
        subtext: '阴影为最大回撤区间　▲ 买入　▼ 卖出', subtextStyle: { fontSize: 11, color: '#6b7280' } },
      legend: { top: 8, right: 12, itemWidth: 12, itemHeight: 8, textStyle: { fontSize: 11 } },
      tooltip: { trigger: 'axis', triggerOn: 'mousemove|click', confine: true, textStyle: { fontSize: 11 } },
      grid: { left: 64, right: 24, top: 62, bottom: 48, containLabel: false },
      xAxis: {
        type: 'category', data: dates, boundaryGap: false,
        axisLabel: { fontSize: 10, interval: Math.max(1, Math.floor(dates.length / 10)), rotate: 30 },
        axisLine: { lineStyle: { color: '#dfe3e8' } }
      },
      yAxis: {
        scale: true, axisLabel: { fontSize: 10 },
        splitLine: { lineStyle: { color: '#f0f2f5' } }
      },
      dataZoom: [{ type: 'inside', start: 55, end: 100 }],
      series: [
        { name: '策略净值', type: 'line', data: navV, symbol: 'none',
          lineStyle: { width: 2, color: '#1f6feb' },
          markArea: { silent: true, itemStyle: { color: 'rgba(209,67,67,0.14)' }, data: area } },
        { name: '等权基准', type: 'line', data: benchV, symbol: 'none',
          lineStyle: { width: 1, type: 'dashed', color: '#9aa4b2' } },
        { name: '买入', type: 'scatter', data: mkPts(buys, true), z: 10,
          itemStyle: { color: '#d14343' } },
        { name: '卖出', type: 'scatter', data: mkPts(sells, false), z: 10,
          itemStyle: { color: '#2b8a3e' } }
      ]
    }, true);

    var chk = d.checks || {};
    var chkPass = Object.keys(chk).filter(function (k) { return chk[k] === true; }).length;
    var cfg = d.config || {};
    // 如实说明读的是哪张表。曾按 tag 名猜版本，选到单视角结果（api_tech）
    // 会被标成"读融合表"，与事实不符，故改为只认结果里记录的 views。
    var srcTxt;
    if (cfg.views && cfg.views.length) {
      if (cfg.views.indexOf('fusion') >= 0) {
        srcTxt = '读融合表（' + (cfg.prompt_version || '版本未记录')
          + '，' + (cfg.method || 'weighted') + '）';
      } else {
        srcTxt = '读单视角研判表（' + cfg.views.join('+') + '）';
      }
    } else {
      srcTxt = '口径版本未记录（该结果生成于补齐元信息之前）';
    }
    // 止盈止损必须出现在口径行：三条早期策略都是 0（未启用），
    // 不写明的话，读者无法从报告页区分"没配"和"配了但没触发"。
    var tp = Number(cfg.take_profit_pct || 0);
    var sl = Number(cfg.stop_loss_pct || 0);
    var exitTxt = (tp > 0 || sl > 0)
      ? '止盈 ' + pct(tp, 1) + '、止损 ' + pct(sl, 1) + '（触发 ' + (pf.take_profit_exits || 0) + ' / ' + (pf.stop_loss_exits || 0) + ' 次）。'
      : '止盈止损未启用（0）。';

    var accTxt = '准确率（逐股算完再平均）：' + num((d.per_stock_mean || {}).acc, 4) + '。';
    if ((cfg.strategy_type || pf.strategy_type) === 'cross_section') {
      // 截面策略的"准确率"是选中后持有期内上涨的比例，与择时策略的次日方向
      // 准确率不是同一个量，标签必须跟着换，否则两个数会被当成可比的。
      accTxt = '持有期命中率（被选中后持有期内上涨的占比）：' + num((d.per_stock_mean || {}).acc, 4) + '。';
      var slip = Number(cfg.slippage_rate || 0);
      var topN = pf.top_n || cfg.top_n || 10;
      var holdDays = pf.hold_days || cfg.hold_days || 5;
      var rbN = pf.rebalance_count || 0;
      var rob = pf.robustness || {};
      var robTxt = (rob.period_count
        ? '稳健性：去掉收益最好的 1/2/3 个调仓周期后，累计收益 ' +
          pct(rob.cum_drop_best_1, 2) + ' / ' + pct(rob.cum_drop_best_2, 2) + ' / ' + pct(rob.cum_drop_best_3, 2) +
          '（全部 ' + pct(rob.cum_all, 2) + '，最好单周期 ' + pct(rob.best_period, 2) + '）。'
        : '');
      el('btNote').innerHTML = '口径：截面选股，' + (pf.stocks || '—') + ' 只候选池中每 ' + holdDays +
        ' 个交易日按技术面分数挑前 ' + topN + ' 只等权（共 ' + rbN + ' 次调仓），' +
        '滑点 ' + pct(slip, 2) + '、佣金万三（单笔最低 5 元）、印花税千一。' +
        '分数取自 T-1 日收盘、T 日开盘成交，不含隔夜跳空。' +
        '截面策略不使用止盈止损。自检 ' + chkPass + '/' + Object.keys(chk).length + ' 项通过。' +
        accTxt + robTxt;
      return;
    }
    el('btNote').innerHTML = '口径：' + srcTxt + '，' + (pf.stocks || '—') + ' 只标的、5 日窗口、等权持有（仓位 ' + pct(cfg.position_pct, 0) +
      '）、佣金万三、印花税千一、滑点万五。' + exitTxt +
      '自检 ' + chkPass + '/' + Object.keys(chk).length + ' 项通过。' +
      accTxt;
  }

  function tagText(d) {
    return d.tag || '';
  }

  /* ---------------- 回测提交与轮询 ---------------- */

  function submitBacktest() {
    var btn = el('btSubmit');
    btn.disabled = true;
    hint('btHint', '提交中…');
    var type = el('btType').value;
    var body = {
      strategy_type: type,
      start: el('btStart').value,
      end: el('btEnd').value
    };
    if (type === 'cross_section') {
      // 界面填的是百分数（0.2 表示 0.2%），引擎吃小数，必须整除 100
      body.top_n = Number(el('csTopN').value || 10);
      body.hold_days = Number(el('csHold').value || 5);
      body.slippage_rate = Number(el('csSlip').value || 0.2) / 100;
    } else {
      // 课题要求「基于预测结果配置止盈止损参数」——预测结果本身要能选。
      // 写死 ['fusion'] 的话，界面上只有"加权/投票"两个选项，等于预设了三视角
      // 一定融合，单视角做不了对照回测。
      var src = el('btViews').value;
      body.views = (src === 'fusion') ? ['fusion'] : src.split('+');
      // 只有融合结论才谈得上"怎么融合"；单视角没有第二方要合并，不传 method
      if (src === 'fusion') {
        body.method = el('btMethod').value;
        // 预测版本：auto 表示不传，由后端按起点与融合方式派生（现状行为，向后兼容）。
        // 投票融合只有一版融合表（fusion-vote-v1），版本下拉在界面上被隐藏，
        // 这里只在加权时才传，避免把一个和后端派生结果矛盾的版本名送进去。
        var pv = el('btFusionVer').value;
        if (body.method === 'weighted' && pv && pv !== 'auto') { body.fusion_version = pv; }
      }
      // 界面填的是百分数（填 8 表示 8%），引擎用的是小数（0.08）。
      // 这里必须整除 100：否则 8 会被当成 800% 止盈、5 会被当成负价止损，两者都永不触发。
      body.take_profit_pct = Number(el('btTp').value || 0) / 100;
      body.stop_loss_pct = Number(el('btSl').value || 0) / 100;
    }
    api('/api/backtest/tasks', { method: 'POST', body: body }).then(function (r) {
      hint('btHint', '任务 #' + r.task_id + ' 已提交，等待进度…', 'ok');
      pollTask(r.task_id, r.tag, 0);
    }).catch(function (e) {
      btn.disabled = false;
      hint('btHint', e.status === 403 ? '403　' + e.message + '（当前角色无权提交）' : '提交失败：' + e.message, 'bad');
    });
  }

  function pollTask(id, tag, n) {
    if (n > 120) { hint('btHint', '轮询超时', 'bad'); el('btSubmit').disabled = false; return; }
    api('/api/backtest/tasks/' + id).then(function (t) {
      hint('btHint', '任务 #' + id + '　' + t.status + '　进度 ' + Math.round((t.progress || 0) * 100) + '%',
        t.status === 'FAILED' ? 'bad' : 'ok');
      if (t.status === 'SUCCEEDED') {
        el('btSubmit').disabled = false;
        hint('btHint', '任务 #' + id + ' 完成，读取报告…', 'ok');
        loadReport(tag);
        loadBacktestPanel();
        return;
      }
      if (t.status === 'FAILED') {
        el('btSubmit').disabled = false;
        hint('btHint', '回测失败：' + (t.error || '').slice(0, 200), 'bad');
        return;
      }
      setTimeout(function () { pollTask(id, tag, n + 1); }, 1500);
    }).catch(function (e) {
      el('btSubmit').disabled = false;
      hint('btHint', '查询失败：' + e.message, 'bad');
    });
  }

  /* ---------------- 策略与订阅 ---------------- */

  // 策略表的口径列：库里五个策略来自不同权重版本，指标不可横比。
  // 不把版本写出来，"加权 15.97% vs 加权 16.88%"看起来像同一方法跑出两个数。
  var PV_SHORT = {
    'fusion-test-3view': '标定 0.9/0/0.1（样本外）',
    'fusion-val-3view': '标定 0.9/0/0.1（样本内）',
    'fusion-agent-v1': '三智能体链路 tech1.0/sent0.5',
    'fusion-vote-v1': '投票（多数决）',
    'fusion-v1': '旧权重 0.5/0.25/0.25',
    'fusion-val-v1': '旧标定 tech1.0',
    'fusion-val-legacy': '旧线上 0.5/0.25/0.25',
    'tech-v1': '单视角技术面',
    'fund-rule-v1': '单视角基本面'
  };

  var VIEW_SHORT = { tech: '单视角·技术面', fund: '单视角·基本面', sent: '单视角·情绪面', fusion: '融合' };

  function pvShortOf(p) {
    if (!p) return '—';
    var pv = p.fusion_version || p.prompt_version || '';
    if (pv && PV_SHORT[pv]) return PV_SHORT[pv];
    if (pv) return pv;
    var vs = (p.views || []).map(function (v) { return VIEW_SHORT[v] || v; }).join(' + ');
    return vs || '—';
  }

  function loadStrategy() {
    api('/api/strategies').then(function (rows) {
      var tb = el('stratTable').querySelector('tbody');
      tb.innerHTML = '';
      if (!rows.length) {
        tb.innerHTML = '<tr><td colspan="8" style="color:#6b7280">暂无策略</td></tr>';
      }
      rows.forEach(function (s) {
        var m = s.metrics || {};
        var tr = document.createElement('tr');
        tr.innerHTML =
          '<td class="num">' + s.id + '</td>' +
          '<td>' + (s.name || '') + '</td>' +
          '<td><span class="tag ' + s.status + '">' + s.status + '</span></td>' +
          '<td class="muted">' + pvShortOf(s.params) + '</td>' +
          '<td class="num">' + pct(m.cum_return) + '</td>' +
          '<td class="num">' + num(m.sharpe) + '</td>' +
          '<td class="num">' + pct(m.mdd) + '</td>' +
          '<td>' + (s.status === 'PUBLISHED' ? '—' : '<button data-sid="' + s.id + '" class="pub">发布</button>') + '</td>';
        tb.appendChild(tr);
      });
      Array.prototype.forEach.call(tb.querySelectorAll('.pub'), function (b) {
        b.onclick = function () { publishStrategy(Number(b.dataset.sid)); };
      });
    }).catch(function (e) {
      hint('stratHint', '策略列表读取失败：' + e.message, 'bad');
    });
    loadSubscriptions();
  }

  function publishStrategy(sid) {
    api('/api/strategies/' + sid + '/publish', { method: 'POST' }).then(function (r) {
      hint('stratHint', '策略 #' + sid + ' 已发布', 'ok');
      loadStrategy();
    }).catch(function (e) {
      hint('stratHint', e.status === 403 ? '403　' + e.message : '发布失败：' + e.message, 'bad');
    });
  }

  function createStrategy() {
    var taskId = Number(el('taskSel').value || 0);
    if (!taskId) { hint('stratHint', '先选一个已完成的任务', 'bad'); return; }
    api('/api/strategies', { method: 'POST', body: { task_id: taskId, name: el('stratName').value, publish: true } })
      .then(function (r) {
        hint('stratHint', '策略 #' + r.strategy_id + ' 已创建并发布', 'ok');
        loadStrategy();
      }).catch(function (e) {
        hint('stratHint', e.status === 403 ? '403　' + e.message + '（需要 author）' : '创建失败：' + e.message, 'bad');
      });
  }

  function loadSubscriptions() {
    // 交易日历要登录后才能拉（接口有鉴权），所以放在这里懒加载，
    // 而不是 init 一开始就请求——那样会在登录前 401，日历永远是空的。
    if (!state.tradeDays.length) loadTradingDays();
    api('/api/subscriptions').then(function (rows) {
      var tb = el('subTable').querySelector('tbody');
      tb.innerHTML = '';
      if (!rows.length) {
        tb.innerHTML = '<tr><td colspan="10" style="color:#6b7280">尚未订阅任何策略</td></tr>';
        renderCursor();
        return;
      }
      rows.forEach(function (r) {
        state.lastAccountId = r.account_id;
        var pnl = r.current_nav / r.initial_nav - 1;
        // 取消订阅只在 ACTIVE 时给按钮：已经 LIQUIDATED 的账户再点一次没有意义
        // （后端重复取消是幂等的，但不该在界面上当成"可用操作"）。
        var ops = '<button class="mini" data-pos="' + r.account_id + '">持仓明细</button>';
        if (r.account_status === 'ACTIVE') {
          ops += ' <button class="mini" data-adv="' + r.account_id +
            '">推进一日</button>';
          ops += ' <button class="mini" data-cancel="' + r.id +
            '" data-aid="' + r.account_id + '">取消订阅</button>';
        }
        var tr = document.createElement('tr');
        tr.innerHTML = '<td class="num">#' + r.account_id + '</td>' +
          '<td>策略 #' + r.strategy_id + '</td>' +
          '<td class="num">' + money(r.initial_nav) + '</td>' +
          '<td class="num">' + money(r.current_nav) + '</td>' +
          '<td class="num ' + (pnl >= 0 ? 'up' : 'down') + '">' + pct(pnl) + '</td>' +
          '<td class="num">' + money(r.peak_nav) + '</td>' +
          '<td class="num">' + money(r.stop_loss_nav) + '</td>' +
          '<td class="num">' + (r.as_of || '<span style="color:#6b7280">未开始</span>') + '</td>' +
          '<td><span class="tag ' + (r.account_status || 'ACTIVE') + '">' +
            (r.account_status || 'ACTIVE') + '</span></td>' +
          '<td>' + ops + '</td>';
        tb.appendChild(tr);
      });
      // 系统当前交易日 = 各账户 as_of 的最大值。这是"系统时间"，不是界面装饰。
      var marked = rows.map(function (x) { return x.as_of; }).filter(Boolean);
      state.cursor = marked.length ? marked.sort().pop() : null;
      // 缓存一份给「推进一日」用：单账户推进要知道它自己的 as_of
      state.subs = rows;
      renderCursor();
    }).catch(function (e) {
      el('subTable').querySelector('tbody').innerHTML =
        '<tr><td colspan="10" style="color:#6b7280">当前角色无权查看订阅</td></tr>';
      hint('subHint', '订阅读取失败：' + e.message, 'bad');
    });
  }

  function renderCursor() {
    var h = el('cursorHint');
    if (!h) return;
    var days = state.tradeDays || [];
    if (!days.length) {
      h.textContent = '系统当前交易日：交易日历未加载';
      return;
    }
    h.textContent = '系统当前交易日：' + (state.cursor || '尚未开始') +
      '（行情区间 ' + days[0] + ' ~ ' + days[days.length - 1] + '，共 ' + days.length + ' 个交易日）';
  }

  function loadTradingDays() {
    return api('/api/trading-days').then(function (d) {
      state.tradeDays = (d && d.days) || [];
      renderCursor();
    }).catch(function () { /* 日历读不到不影响别的面板 */ });
  }

  /* 点击委托放在容器上：表格是 innerHTML 重绘的，逐行绑事件会在重绘后丢。
     这也是为什么按钮只带 data-* 属性、不带内联 onclick。 */
  function bindSubActions() {
    var tb = el('subTable');
    if (!tb) return;
    tb.addEventListener('click', function (ev) {
      var t = ev.target;
      if (!t || t.tagName !== 'BUTTON') return;
      var pos = t.getAttribute('data-pos');
      var adv = t.getAttribute('data-adv');
      var sid = t.getAttribute('data-cancel');
      if (pos) { loadPositions(Number(pos)); return; }
      if (adv) { advanceOne(Number(adv)); return; }
      if (sid) {
        var aid = Number(t.getAttribute('data-aid') || 0);
        var ok = confirm('确认取消订阅 #' + sid + '？\n\n' +
          '取消后该跟单账户停止推演并置为 LIQUIDATED，' +
          '会记一条「投资者主动取消订阅」事件（不是强平）。');
        if (!ok) return;
        api('/api/subscriptions/' + sid + '/cancel', { method: 'POST' })
          .then(function (r) {
            hint('subHint', '已取消订阅 #' + sid + '，账户 #' +
              (r.account_id || aid) + ' 状态 ' + (r.status || 'LIQUIDATED'), 'ok');
            if (state.posAccountId === aid) hidePositions();
            loadSubscriptions();
          }).catch(function (e) {
            hint('subHint', e.status === 403 ? '403　' + e.message
              : '取消失败：' + e.message, 'bad');
          });
      }
    });
  }

  /* 持仓明细：数字来自策略成交明细现算，不是账户真持仓。
     接口自己带了 note 说明口径，这里原样显示，不在前端另编一套说法。 */
  /* prefix 决定渲染到哪一页：策略页用 pos、风控页用 rpos。两页各有各的面板——
     风控员在本页点"持仓"，结果写到策略页的表格里，他在本页看不见，就是"点了没反应"。 */
  function loadPositions(aid, prefix) {
    prefix = prefix || 'pos';
    var d = (el('riskDate') && el('riskDate').value || '').trim();
    var url = '/api/follow-accounts/' + aid + '/positions' +
      (d ? '?date=' + encodeURIComponent(d) : '');
    api(url).then(function (p) {
      state.posAccountId = aid;
      state.posPrefix = prefix;
      el(prefix + 'Wrap').style.display = '';
      el(prefix + 'Title').textContent = '持仓明细 · 账户 #' + aid +
        '（策略 ' + (p.strategy_tag || '-') + '，截至 ' + (p.as_of || '-') + '）';
      var rows = p.rows || [];
      el(prefix + 'Note').innerHTML = esc(p.note || '') +
        '<br>账户当前净值 ' + money(p.account_current_nav) +
        '　策略组合净值 ' + (p.strategy_portfolio_nav === null ||
          p.strategy_portfolio_nav === undefined
          ? '—' : '×' + Number(p.strategy_portfolio_nav).toFixed(4) +
            '（归一化，起点 1.0）') +
        '　持仓市值合计 ' + money(p.total_market_value) +
        '<br><span class="muted">持仓市值合计是策略内单票口径的加总，' +
        '与右侧账户净值不是同一口径，两者不能相减</span>';
      var tb = el(prefix + 'Table').querySelector('tbody');
      tb.innerHTML = '';
      if (!rows.length) {
        tb.innerHTML = '<tr><td colspan="7" style="color:#6b7280">' +
          '该策略在此时点没有持仓（已全部平仓）</td></tr>';
        return;
      }
      rows.forEach(function (x) {
        var pnl = x.unrealized_pnl;
        var tr = document.createElement('tr');
        tr.innerHTML = '<td>' + esc(x.stock) + '</td>' +
          '<td class="num">' + money(x.shares) + '</td>' +
          '<td class="num">' + num(x.avg_cost, 4) + '</td>' +
          '<td class="num">' + num(x.last_price, 4) + '</td>' +
          '<td class="num">' + money(x.market_value) + '</td>' +
          '<td class="num ' + ((pnl || 0) >= 0 ? 'up' : 'down') + '">' +
          money(pnl) + '</td>' +
          '<td class="num">' + num(x.weight_pct, 2) + '%</td>';
        tb.appendChild(tr);
      });
    }).catch(function (e) {
      var msg = e.status === 409 ? '该策略没有成交明细（' + e.message + '）'
        : (e.status === 403 ? '403　' + e.message : '持仓读取失败：' + e.message);
      hint(prefix === 'rpos' ? 'riskHint' : 'subHint', msg, 'bad');
    });
  }

  function hidePositions() {
    ['pos', 'rpos'].forEach(function (pfx) {
      var w = el(pfx + 'Wrap');
      if (w) w.style.display = 'none';
    });
    state.posAccountId = null;
    state.posPrefix = null;
  }

  function subscribe() {
    var sid = Number(el('subSid').value || 0);
    var nav = Number(el('subNav').value || 0);
    var stop = Number(el('subStop').value || 10) / 100;
    api('/api/subscriptions', { method: 'POST', body: { strategy_id: sid, initial_nav: nav, stop_loss_pct: stop } })
      .then(function (r) {
        state.lastAccountId = r.account_id;
        hint('subHint', '已订阅，账户 #' + r.account_id + '，强平线 ' + money(r.stop_loss_nav) +
          '（回撤 ' + (r.stop_loss_pct * 100).toFixed(0) + '%）', 'ok');
        loadSubscriptions();
      }).catch(function (e) {
        hint('subHint', e.status === 403 ? '403　' + e.message + '（需要 investor）' : '订阅失败：' + e.message, 'bad');
      });
  }

  /* 推进一日。**不传日期**：后端按该账户 as_of 推到下一个交易日。
     原来这里写死 '2024-12-31'，看起来像"净值是离线算好摆上来的"，
     而不是系统在按日走——这正是 R3 要修的那处。 */
  function advanceOne(aid) {
    if (!aid) { hint('subHint', '先订阅一个策略', 'bad'); return; }
    api('/api/follow-accounts/' + aid + '/advance', { method: 'POST', body: {} })
      .then(function (r) {
        if (r.skipped) { hint('subHint', '账户 #' + aid + '：' + r.skipped, 'bad'); return; }
        hint('subHint', '账户 #' + aid + ' 推进到 ' + (r.as_of || '-') +
          '：净值 ' + money(r.current_nav) + '　状态 ' + r.status, 'ok');
        loadSubscriptions();
      }).catch(function (e) {
        hint('subHint', '推进失败：' + e.message, 'bad');
      });
  }

  function advance() { advanceOne(state.lastAccountId); }

  function advanceOneTo(aid, date) {
    if (!aid) { hint('subHint', '先订阅一个策略', 'bad'); return; }
    api('/api/follow-accounts/' + aid + '/advance', { method: 'POST', body: { date: date } })
      .then(function (r) {
        if (r.skipped) { hint('subHint', '账户 #' + aid + '：' + r.skipped, 'bad'); return; }
        hint('subHint', '账户 #' + aid + ' 推进到 ' + (r.as_of || '-') +
          '：净值 ' + money(r.current_nav) + '　状态 ' + r.status, 'ok');
        loadSubscriptions();
      }).catch(function (e) {
        hint('subHint', '推进失败：' + e.message, 'bad');
      });
  }
  function advanceTo() {
    var d = el('advDate').value;
    if (!d) { hint('subHint', '先选目标日期', 'bad'); return; }
    advanceOneTo(state.lastAccountId, d);
  }
  function advanceToEnd() { advanceOneTo(state.lastAccountId, 'latest'); }

  /* 推进全部：每个在途账户各自往前走一个交易日。
     不是"统一推到某一天"——老账户已到期末、新账户还没开始，统一推会拉平或倒退。 */
  function advanceAll() {
    api('/api/follow-accounts/advance-all', { method: 'POST', body: {} })
      .then(function (r) {
        var msg = '推进一天：' + (r.moved || 0) + ' 个账户前进';
        if (r.skipped) msg += '，' + r.skipped + ' 个已到期末';
        if (r.failed) msg += '，' + r.failed + ' 个失败';
        if (r.as_of_max) msg += '（最新 ' + r.as_of_max + '）';
        hint('subHint', msg, r.failed ? 'bad' : 'ok');
        loadSubscriptions();
      }).catch(function (e) {
        hint('subHint', e.status === 403 ? '403　' + e.message
          : '推进失败：' + e.message, 'bad');
      });
  }

  /* ---------------- 风控与社区 ---------------- */

  // 风控「监控」的第一层：进页面就按只读水位算一遍，把"几个账户、几个快到线、
  // 最危险的那个差多少"放在表格上面。分档只看离强平线还有多远：
  //   <1% 红、<3% 黄、其余灰。判决本身由后端在推进时写进 risk_events。
  function renderRiskSummary(list) {
    var box = el('riskSum');
    if (!box) return;
    var ok = (list || []).filter(function (x) { return x && x.ok; })
      .map(function (x) { return x.payload || {}; });
    if (!ok.length) { box.style.display = 'none'; return; }
    var active = ok.filter(function (p) { return (p.status || 'ACTIVE') === 'ACTIVE'; });
    var liq = ok.filter(function (p) { return p.status === 'LIQUIDATED'; });
    var near = active.map(function (p) {
      return {
        id: p.account_id,
        gap: (p.current_nav / p.stop_loss_nav - 1) * 100,
        at: p.as_of
      };
    }).sort(function (a, b) { return a.gap - b.gap; });
    var worst = near.length ? near[0] : null;
    var n1 = near.filter(function (x) { return x.gap < 1; }).length;
    var n3 = near.filter(function (x) { return x.gap < 3; }).length;
    var color = (worst && worst.gap < 1) ? '#b91c1c'
      : ((worst && worst.gap < 3) ? '#b45309' : '#6b7280');
    var txt = '在途 ' + active.length + ' 个账户' +
      (liq.length ? '，已强平 ' + liq.length + ' 个' : '');
    if (worst) {
      txt += '；距强平线最近的是 #' + worst.id + '，' +
        (worst.gap >= 0 ? '+' : '') + worst.gap.toFixed(2) + '%';
    }
    if (n3) {
      txt += '；其中 ' + n3 + ' 个不足 3%' + (n1 ? '、' + n1 + ' 个不足 1%' : '');
    }
    // 这行水位是 dry_run 按曲线末端（或复盘日期）算的，不是账户真实 as_of，
    // 所以只写口径，不写日期，免得和订阅页「净值截至」对不上。
    txt += '（按' + (el('riskDate').value || '曲线末端') +
      '水位只读计算）。强平判定随账户推进自动执行并写风控事件，不必点「扫描」。';
    box.style.display = '';
    box.innerHTML = '<span style="color:' + color + '">' + txt + '</span>';
  }

  function loadRisk() {
    // 账户水位：只读，不推进。风控要看的是"在途账户离强平线还有多远"，
    // 只在出事之后才有记录可看就不叫监控了。
    var d = (el('riskDate').value || '').trim();
    api('/api/risk/accounts' + (d ? '?date=' + encodeURIComponent(d) : '')).then(function (d2) {
      var tb = el('acctTable').querySelector('tbody');
      tb.innerHTML = '';
      var list = (d2 && d2.accounts) || [];
      renderRiskSummary(list);
      if (!d && d2 && d2.last_scan) {
        el('riskWatch').textContent = '自动巡检：' + d2.last_scan +
          '（每 ' + (d2.interval_sec || 30) + ' 秒只读巡检一次，不写事件）';
      }
      if (!list.length) {
        tb.innerHTML = '<tr><td colspan="9" style="color:#6b7280">暂无跟单账户</td></tr>';
        return;
      }
      list.forEach(function (x) {
        var p = (x && x.payload) || {};
        var tr = document.createElement('tr');
        if (!x.ok) {
          tr.innerHTML = '<td colspan="9" style="color:#b91c1c">账户读取失败：' + esc(String(p)) + '</td>';
          tb.appendChild(tr);
          return;
        }
        var gap = ((p.current_nav / p.stop_loss_nav - 1) * 100);
        var st = p.status || 'ACTIVE';
        // 状态是"现在"，事件是"曾经"。两者分开写，否则会出现"距强平线 +1.68% 却标预警"的矛盾
        var stCell = '<span class="tag ' + st + '">' + st + '</span>';
        if (p.event) {
          stCell += '<br><span class="muted">曾触发 ' + p.event +
            (p.warn_date ? '（' + p.warn_date + '）' : '') + '</span>';
        }
        // 操作列：只有风控员给"手工强平"。运营也能看账户水位，但强平接口
        // 只对 risk 开放，给他放个按不动的按钮等于误导。
        var ops = '<button class="mini" data-rpos="' + p.account_id + '">持仓</button>';
        if (state.user && state.user.role === 'risk' && st === 'ACTIVE') {
          ops += ' <button class="mini" data-liq="' + p.account_id +
            '" data-tag="' + esc(p.strategy_tag || '') + '">手工强平</button>';
        }
        tr.innerHTML = '<td class="num">#' + p.account_id + '</td>' +
          '<td>' + esc(p.strategy_tag || '-') + '</td>' +
          '<td class="num">' + money(p.current_nav) + '</td>' +
          '<td class="num">' + money(p.peak_nav) + '</td>' +
          '<td class="num">' + money(p.stop_loss_nav) + '（' + (p.stop_loss_pct * 100).toFixed(0) + '%）</td>' +
          '<td class="num">' + (gap >= 0 ? '+' : '') + gap.toFixed(2) + '%</td>' +
          '<td class="num">' + (p.max_dd_pct || 0).toFixed(2) + '%</td>' +
          '<td>' + stCell + '</td>' +
          '<td>' + ops + '</td>';
        tb.appendChild(tr);
      });
    }).catch(function (e) {
      // 无权或读失败时必须清表：留着上一次的渲染结果等于把风控数据展示给了没权限的角色
      el('acctTable').querySelector('tbody').innerHTML =
        '<tr><td colspan="9" style="color:#6b7280">当前角色无权查看账户水位</td></tr>';
      hint('riskHint', e.status === 403 ? '403　' + e.message : '账户读取失败：' + e.message, 'bad');
    });

    api('/api/risk/events?limit=50').then(function (rows) {
      var tb = el('riskTable').querySelector('tbody');
      tb.innerHTML = '';
      if (!rows.length) tb.innerHTML = '<tr><td colspan="4" style="color:#6b7280">暂无风控事件</td></tr>';
      rows.forEach(function (r) {
        var tr = document.createElement('tr');
        tr.innerHTML = '<td class="num">' + r.id + '</td>' +
          '<td><span class="tag ' + r.kind + '">' + r.kind + '</span></td>' +
          '<td>' + (r.detail || '') + '</td>' +
          '<td class="num">' + (r.created_at || '') + '</td>';
        tb.appendChild(tr);
      });
    }).catch(function (e) {
      el('riskTable').querySelector('tbody').innerHTML =
        '<tr><td colspan="4" style="color:#6b7280">当前角色无权查看风控事件</td></tr>';
      hint('riskHint', e.status === 403 ? '403　' + e.message : '事件读取失败：' + e.message, 'bad');
    });
    loadPosts();
  }

  function scanRisk() {
    var d = (el('riskDate').value || '').trim();
    api('/api/risk/scan', { method: 'POST', body: { date: d || null } }).then(function (r) {
      var list = r.results || [];
      var n = list.length;
      var warned = list.filter(function (x) {
        return x.ok && x.payload && x.payload.event;
      }).length;
      hint('riskHint', '扫描完成，' + n + ' 个账户，其中 ' + warned + ' 个触发风控事件',
        warned ? 'bad' : 'ok');
      loadRisk();
    }).catch(function (e) {
      hint('riskHint', e.status === 403 ? '403　' + e.message + '（需要 risk）' : '扫描失败：' + e.message, 'bad');
    });
  }

  /* 账户表的两个按钮同样走委托：loadRisk 每次重绘 tbody，逐行绑事件会丢。 */
  function bindRiskActions() {
    var tb = el('acctTable');
    if (!tb) return;
    tb.addEventListener('click', function (ev) {
      var t = ev.target;
      if (!t || t.tagName !== 'BUTTON') return;
      var rpos = t.getAttribute('data-rpos');
      var liq = t.getAttribute('data-liq');
      if (rpos) { loadPositions(Number(rpos), 'rpos'); return; }
      if (liq) { liquidate(Number(liq), t.getAttribute('data-tag') || ''); }
    });
  }

  /* 手工强平。自动强平写在 service_risk.advance_account 里（回撤破线自动置位），
     这个动作补的是"人对着监控面板按下去"那条路径。两者都要记事件，但手工强平
     必须带操作人（handled_by）和原因——事后复盘时"谁按的、依据什么"比"按了"重要。
     所以这里真的问一句原因，不给默认文案蒙过去。 */
  function liquidate(aid, tag) {
    var reason = prompt('手工强平账户 #' + aid + '（策略 ' + (tag || '-') + '）\n\n' +
      '账户将置为 LIQUIDATED，并记一条带操作人与原因的事件。\n' +
      '请填强平原因（留空即取消）：', '人工判断：回撤逼近强平线，提前处置');
    if (reason === null) return;
    reason = (reason || '').trim();
    if (!reason) { hint('riskHint', '未填原因，已取消', 'bad'); return; }
    api('/api/follow-accounts/' + aid + '/liquidate',
        { method: 'POST', body: { reason: reason } })
      .then(function (r) {
        hint('riskHint', '已强平账户 #' + aid + '，状态 ' + (r.status || 'LIQUIDATED'), 'ok');
        if (state.posAccountId === aid) hidePositions();
        loadRisk();
      }).catch(function (e) {
        hint('riskHint', e.status === 403 ? '403　' + e.message + '（需要 risk）'
          : '强平失败：' + e.message, 'bad');
      });
  }

  function fillPostStrategies() {
    api('/api/strategies').then(function (rows) {
      var sel = el('postStrat');
      var keep = sel.value;
      sel.innerHTML = '<option value="">不关联</option>';
      (rows || []).forEach(function (s) {
        if (s.status !== 'PUBLISHED') return;
        var op = document.createElement('option');
        op.value = s.id;
        op.text = '#' + s.id + ' ' + s.name;
        sel.appendChild(op);
      });
      if (keep) sel.value = keep;
    }).catch(function () { /* 策略列表拉不到时不阻塞发帖 */ });
  }

  function loadPosts() {
    fillPostStrategies();
    api('/api/posts').then(function (rows) {
      var tb = el('postTable').querySelector('tbody');
      tb.innerHTML = '';
      if (!rows.length) tb.innerHTML = '<tr><td colspan="7" style="color:#6b7280">暂无帖子</td></tr>';
      rows.forEach(function (p) {
        var tr = document.createElement('tr');
        var strat = p.strategy_id
          ? ('#' + p.strategy_id + ' ' + (p.strategy_name || ''))
          : '<span class="muted">—</span>';
        var act = '';
        // 只有投资者能订阅，其他角色看帖不该出现这个按钮
        if (p.strategy_id && p.status === 'APPROVED' &&
            state.user && state.user.role === 'investor') {
          act = '<button class="mini" data-follow="' + p.strategy_id + '">订阅该策略</button>';
        }
        tr.innerHTML = '<td class="num">' + p.id + '</td>' +
          '<td class="num">' + p.user_id + '</td>' +
          '<td>' + esc((p.content || '').slice(0, 60)) + '</td>' +
          '<td>' + strat + '</td>' +
          '<td><span class="tag ' + p.status + '">' + p.status + '</span></td>' +
          '<td class="num">' + (p.created_at || '') + '</td>' +
          '<td>' + act + '</td>';
        tb.appendChild(tr);
      });
      Array.prototype.forEach.call(tb.querySelectorAll('[data-follow]'), function (b) {
        b.onclick = function () { followFromPost(Number(b.getAttribute('data-follow'))); };
      });
    }).catch(function (e) {
      hint('postHint', '帖子读取失败：' + e.message, 'bad');
    });
  }

  /* 看帖 → 订阅：帖子挂的是一条已发布策略，这里按默认本金与 10% 强平线跟单 */
  function followFromPost(sid) {
    api('/api/subscriptions', {
      method: 'POST',
      body: { strategy_id: sid, initial_nav: 100000, stop_loss_pct: 0.10 }
    }).then(function (r) {
      hint('postHint', '已按帖子订阅策略 #' + sid + '，账户 #' + r.account_id +
        '，强平线 ' + money(r.stop_loss_nav), 'ok');
      loadSubscriptions();
    }).catch(function (e) {
      var msg = e.status === 409 ? '已订阅该策略，无需重复跟单'
        : (e.status === 403 ? '403　' + e.message : '订阅失败：' + e.message);
      hint('postHint', msg, 'bad');
    });
  }

  function createPost() {
    var sid = el('postStrat').value;
    api('/api/posts', {
      method: 'POST',
      body: { content: el('postText').value, strategy_id: sid ? Number(sid) : null }
    }).then(function (r) {
      hint('postHint', '已发帖 #' + r.post_id + '，状态 ' + r.status +
        (r.strategy_id ? '，关联策略 #' + r.strategy_id : ''), 'ok');
      loadPosts();
    }).catch(function (e) {
      hint('postHint', e.status === 403 ? '403　' + e.message : '发帖失败：' + e.message, 'bad');
    });
  }

  function moderateLatest() {
    api('/api/posts').then(function (rows) {
      var target = null;
      for (var i = 0; i < rows.length; i++) {
        if (rows[i].status === 'PENDING') { target = rows[i]; break; }
      }
      if (!target) { hint('postHint', '没有待审核帖子', 'bad'); return; }
      return api('/api/posts/' + target.id + '/moderate', { method: 'POST' }).then(function (r) {
        hint('postHint', '帖 #' + r.post_id + ' → ' + r.decision +
          (r.rule_hit ? '（命中规则 ' + r.rule_hit + '）' : ''), r.decision === 'BLOCKED' ? 'bad' : 'ok');
        loadRisk();
      });
    }).catch(function (e) {
      hint('postHint', e.status === 403 ? '403　' + e.message + '（需要 risk）' : '审核失败：' + e.message, 'bad');
    });
  }

  /* ---------------- 图 3 / 图 4：运营看板 ---------------- */

  function loadOps() {
    api('/api/health').then(function (h) {
      var c = h.counts || {};
      el('opsCards').innerHTML =
        metricCard('研判记录', money(c.judgements), '四视角合计', '') +
        metricCard('回测任务', money(c.backtest_tasks), '含失败') +
        metricCard('策略总数', money(c.strategies), '含草稿与已发布', '') +
        metricCard('订阅 / 跟单账户', money(c.subscriptions), '账户 ' + money(c.follow_accounts)) +
        metricCard('风控事件', money(c.risk_events), '强平与拦截') +
        metricCard('帖子', money(c.posts), '社区');
    }).catch(function () {});

    api('/api/accuracy').then(renderAccuracy).catch(function (e) {
      hint('opsCards', '准确率读取失败：' + e.message, 'bad');
    });

    api('/api/dashboard/strategy-activity').then(renderActivity).catch(function (e) {
      // 该接口需要 operator/author，其他角色会 403；不能静默留白，
      // 否则演示时看到的是两张空卡片，不知道是没数据还是没权限。
      var wrap = el('chart4');
      if (!wrap) return;
      if (e.status === 403) {
        wrap.style.height = 'auto';
        wrap.innerHTML = '<div class="note">策略活跃度需要「运营」或「策略作者」角色查看；' +
          '当前角色是' + (ROLE_NAME[state.user.role] || state.user.role) +
          '，点顶栏对应角色即可。</div>';
      } else {
        hint('opsCards', '活跃度读取失败：' + e.message, 'bad');
      }
    });
  }

  function renderAccuracy(acc) {
    var c = ensureChart('chart3', 420);
    if (!c) return;
    var daily = acc.daily || {};
    var base = Object.keys(daily.fusion || {});
    var series = VIEWS.map(function (v) {
      var m = daily[v] || {};
      return {
        name: VNAME[v], type: 'line', symbol: 'none',
        data: base.map(function (d) { return (m[d] === undefined) ? null : m[d]; }),
        lineStyle: { width: 1.6, color: VCOLOR[v], opacity: 0.9 }
      };
    });
    var sum = (acc.summary) || {};
    // 看板是全 test 段口径（97 天），研判页是 12 天的 auto 版本，两者不是同一套数据。
    // 不把版本标在图里，就会出现「研判页写着 fund-agent-v3，看板却是 49.4%」的对不上：
    // 49.4% 是 fund-rule-v1（纯估值分位）的数，跟带财务的 v3 不是一回事。
    var pvOf = function (v) { return (sum[v] && sum[v].prompt_version) || '未记录'; };
    var fusPv = (acc.meta && acc.meta.fusion_version)
      || (sum.fusion && sum.fusion.prompt_version) || '未记录';
    var versLine = '\n版本：' + pvOf('tech') + ' / ' + pvOf('fund') + ' / '
      + pvOf('sent') + ' / ' + fusPv
      + '　基本面为估值分位规则版，非研判页展示的 fund-agent-v3';

    c.setOption({
      backgroundColor: '#fff',
      animation: false,
      title: { text: '四视角逐日方向准确率', left: 8, top: 4,
        textStyle: { fontSize: 14, fontWeight: 600 },
        subtext: '全体样本：技术 ' + num(sum.tech && sum.tech.acc_all, 4) +
          '　基本面 ' + num(sum.fund && sum.fund.acc_all, 4) +
          '　情绪 ' + num(sum.sent && sum.sent.acc_all, 4) +
          '　融合 ' + num(sum.fusion && sum.fusion.acc_all, 4) +
          '　（0.5 线即随机水平，测试期存在语料污染风险）' + versLine,
        subtextStyle: { fontSize: 11, color: '#6b7280', lineHeight: 16 } },
      legend: { top: 8, right: 12, itemWidth: 12, itemHeight: 8, textStyle: { fontSize: 11 } },
      tooltip: { trigger: 'axis', triggerOn: 'mousemove|click', confine: true, textStyle: { fontSize: 11 } },
      grid: { left: 52, right: 24, top: 84, bottom: 40, containLabel: false },
      xAxis: { type: 'category', data: base, boundaryGap: false,
        axisLabel: { fontSize: 10, interval: Math.max(1, Math.floor(base.length / 10)), rotate: 30 },
        axisLine: { lineStyle: { color: '#dfe3e8' } } },
      yAxis: { min: 0, max: 1, axisLabel: { fontSize: 10, formatter: '{value}' },
        splitLine: { lineStyle: { color: '#f0f2f5' } } },
      dataZoom: [{ type: 'inside', start: 40, end: 100 }],
      series: series
    }, true);
  }

  function renderActivity(rows) {
    var c = ensureChart('chart4', 360);
    if (!c) return;
    var dates = rows.map(function (r) { return r.date; });
    c.setOption({
      backgroundColor: '#fff',
      animation: false,
      title: { text: '策略活跃度：按交易日的信号产出', left: 8, top: 4,
        textStyle: { fontSize: 14, fontWeight: 600 },
        subtext: '柱为强信号标的数（|score| ≥ 0.2），线为看多 / 看空数',
        subtextStyle: { fontSize: 11, color: '#6b7280' } },
      legend: { top: 8, right: 12, itemWidth: 12, itemHeight: 8, textStyle: { fontSize: 11 } },
      tooltip: { trigger: 'axis', triggerOn: 'mousemove|click', confine: true, textStyle: { fontSize: 11 } },
      grid: { left: 52, right: 24, top: 66, bottom: 40, containLabel: false },
      xAxis: { type: 'category', data: dates, boundaryGap: true,
        axisLabel: { fontSize: 10, interval: Math.max(1, Math.floor(dates.length / 10)), rotate: 30 },
        axisLine: { lineStyle: { color: '#dfe3e8' } } },
      yAxis: { axisLabel: { fontSize: 10 }, splitLine: { lineStyle: { color: '#f0f2f5' } } },
      dataZoom: [{ type: 'inside', start: 40, end: 100 }],
      series: [
        { name: '强信号数', type: 'bar', data: rows.map(function (r) { return r.strong_n; }),
          barMaxWidth: 10, itemStyle: { color: '#9aa4b2', opacity: 0.55 } },
        { name: '看多', type: 'line', symbol: 'none', data: rows.map(function (r) { return r.buy_n; }),
          lineStyle: { width: 1.8, color: '#d14343' } },
        { name: '看空', type: 'line', symbol: 'none', data: rows.map(function (r) { return r.sell_n; }),
          lineStyle: { width: 1.8, color: '#2b8a3e' } }
      ]
    }, true);
  }

  /* ---------------- 初始化 ---------------- */

  function init() {
    fetch('/api/auth/demo-users', { method: 'POST' })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        // 后端返回 {"users": [...], "demo_password": "..."}，这里必须解包；
        // 直接把响应体当数组会让 length 变 undefined，整个初始化链在第一步就断掉，页面全空。
        var users = (res && res.users) ? res.users : [];
        state.users = users;
        state.demoPassword = (res && res.demo_password) || 'demo1234';
        if (!state.users.length) throw new Error('演示账号初始化失败');
        var investor = state.users.filter(function (u) { return u.role === 'investor'; })[0] || state.users[0];
        // 初始化必须真正登录一次拿 token，否则后面所有接口都会 401
        return login(investor.username, state.demoPassword);
      })
      .then(function (res) {
        state.token = res.token;
        state.user = res.user;
        buildRoles();
        hint('status', '已登录（token 已签发）', 'ok');
        return api('/api/meta');
      })
      .then(function (meta) {
        state.meta = meta;
        return api('/api/health');
      })
      .then(function (h) {
        hint('status', '引擎' + (h.engine_ready ? '就绪' : '预热中') + '　行情 ' + h.price_range.join(' ~ '),
          h.engine_ready ? 'ok' : '');
        return api('/api/market/stocks');
      })
      .then(function (stocks) {
        var sel = el('symSel');
        sel.innerHTML = '';
        stocks.forEach(function (s) {
          var o = document.createElement('option');
          o.value = s.symbol;
          o.textContent = s.symbol + '（' + s.bars + ' 根）';
          sel.appendChild(o);
        });
        var pref = '万华化学';
        for (var i = 0; i < sel.options.length; i++) {
          if (sel.options[i].value === pref) { sel.value = pref; break; }
        }
        switchTab(ROLE_HOME[state.user.role] || 'market');
      })
      .catch(function (e) {
        hint('status', '初始化失败：' + e.message, 'bad');
      });

    el('symSel').onchange = loadMarket;
    el('judgeRunBtn').onclick = runJudge;
    el('judgeBatchToggle').onclick = function () {
      var w = el('judgeBatchWrap');
      w.style.display = (w.style.display === 'none') ? '' : 'none';
    };
    el('judgeBatchBtn').onclick = runJudgeBatch;
    el('btSubmit').onclick = submitBacktest;
    el('tagLoad').onclick = function () { loadReport(el('tagSel').value); };
    el('stratRefresh').onclick = loadStrategy;
    el('stratPublish').onclick = createStrategy;
    el('subBtn').onclick = subscribe;
    el('advBtn').onclick = advance;
    el('advToBtn').onclick = advanceTo;
    el('advEndBtn').onclick = advanceToEnd;
    el('advAllBtn').onclick = advanceAll;
    el('riskScan').onclick = scanRisk;
    el('riskRefresh').onclick = loadRisk;
    el('postBtn').onclick = createPost;
    el('postMod').onclick = moderateLatest;
    // 两个表格/卡片区都是 innerHTML 重绘，按钮只能靠委托，所以绑定一次即可
    bindSubActions();
    bindRiskActions();

    // 截面策略只看持仓只数/调仓间隔/滑点，择时看预测来源/融合方式/止盈/止损。
    // 不切换显隐会把不适用的参数留在界面上，用户填了却没生效。
    el('btType').onchange = function () {
      var isCS = el('btType').value === 'cross_section';
      Array.prototype.forEach.call(document.querySelectorAll('.csOnly'), function (n) {
        n.style.display = isCS ? '' : 'none';
      });
      el('csNote').style.display = isCS ? '' : 'none';
      ['btMethod', 'btFusionVer', 'btTp', 'btSl', 'btViews'].forEach(function (id) {
        var input = el(id);
        if (input) { input.disabled = isCS; }
      });
      if (isCS) {
        el('btMethodWrap').style.display = 'none';
        el('btFusionVerWrap').style.display = 'none';
      } else {
        syncBtVer();
      }
      el('btStart').value = isCS ? '2024-08-08' : el('btStart').value;
      el('btEnd').value = isCS ? '2024-12-31' : el('btEnd').value;
      hint('btHint', isCS
        ? '截面策略：只用技术面分数，区间必须收在同一段（val 03-15~08-07 / test 08-08~12-31）'
        : '');
    };

    // 「预测版本」只对"加权融合"有意义：投票融合只有一版融合表。
    // 两个下拉联动抽成一个函数，避免 onchange 里各写一份显隐逻辑而走岔。
    // 后端派生规则（service_backtest.normalize_params）必须在这里有一份同样的，
    // 否则「按起点自动派生」这五个字等于没解释——现场被问"那到底读哪一版"答不上来。
    function btVerDerive() {
      if (el('btMethod').value === 'vote') { return 'fusion-vote-v1'; }
      var s = el('btStart').value || '';
      return (s && s < '2024-08-08') ? 'fusion-val-3view' : 'fusion-test-3view';
    }

    function syncBtVer() {
      var isFusion = (el('btViews').value === 'fusion');
      var isWeighted = (el('btMethod').value === 'weighted');
      el('btMethodWrap').style.display = isFusion ? '' : 'none';
      el('btFusionVerWrap').style.display = (isFusion && isWeighted) ? '' : 'none';
      if (!isFusion || !isWeighted) { return; }   // 其余两种状态的提示由各自 onchange 写
      hint('btHint', el('btFusionVer').value === 'auto'
        ? '预测版本按起点自动派生：当前区间读 ' + btVerDerive()
          + '；改起始日期或显式选版本都会重新定'
        : '');
    }
    el('btMethod').onchange = function () {
      syncBtVer();
      if (el('btViews').value === 'fusion' && el('btMethod').value === 'vote') {
        hint('btHint', '投票融合只有一版融合表（fusion-vote-v1），预测版本不生效');
      }
    };
    el('btFusionVer').onchange = function () { syncBtVer(); };
    el('btStart').onchange = function () {
      if (el('btViews').value === 'fusion' && el('btMethod').value === 'weighted') {
        syncBtVer();
      }
    };

    // 「融合方式」只对融合结论有意义：选了单视角却还留着加权/投票下拉，
    // 会让人以为单视角也能投票。
    el('btViews').onchange = function () {
      var src = el('btViews').value;
      var parts = (src === 'fusion') ? 1 : src.split('+').length;
      syncBtVer();
      if (src === 'fusion') {
        if (el('btMethod').value === 'vote') {
          hint('btHint', '投票融合只有一版融合表（fusion-vote-v1），预测版本不生效');
        }
      } else if (parts === 1) {
        hint('btHint', '单视角回测：直接读该视角的研判分，不经过融合、没有加权或投票');
      } else {
        hint('btHint', '多视角等权叠加：各视角权重相同、直接用加权和取信号，不读三视角融合表');
      }
    };
    el('btViews').onchange();

    window.addEventListener('resize', function () {
      clearTimeout(window.__rz);
      window.__rz = setTimeout(resizeAll, 120);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

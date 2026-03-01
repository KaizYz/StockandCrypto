// ========================================
// StockandCrypto - Equity Deep Panel (8501 style)
// ========================================
(function () {
    'use strict';

    const CACHE_TTL_MS = 30 * 1000;
    const state = {
        projectionChart: null,
        trackingCache: new Map(),
        executionCache: { ts: 0, data: null },
    };

    function safeNumber(value) {
        const n = Number(value);
        return Number.isFinite(n) ? n : null;
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function pickFirstNumber(...vals) {
        for (const v of vals) {
            const n = safeNumber(v);
            if (n !== null) return n;
        }
        return null;
    }

    function setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = String(text);
    }

    function setHTML(id, html) {
        const el = document.getElementById(id);
        if (el) el.innerHTML = html;
    }

    function actionCode(raw) {
        const txt = String(raw || '').trim().toLowerCase();
        if (!txt) return 'WAIT';
        if (txt.includes('long') || txt.includes('buy') || txt.includes('open') || txt === 'keep/open') return 'LONG';
        if (txt.includes('short') || txt.includes('sell') || txt.includes('reduce') || txt === 'monitor/reduce') return 'SHORT';
        if (txt.includes('flat') || txt.includes('wait') || txt === 'none') return 'WAIT';
        return 'WAIT';
    }

    function actionText(code) {
        if (code === 'LONG') return 'LONG / 做多';
        if (code === 'SHORT') return 'SHORT / 做空';
        return 'WAIT / 观望';
    }

    function actionClass(code) {
        if (code === 'LONG') return 'long';
        if (code === 'SHORT') return 'short';
        return 'flat';
    }

    function trendText(raw) {
        const txt = String(raw || '').toLowerCase();
        if (txt.includes('bull')) return '趋势偏多';
        if (txt.includes('bear')) return '趋势偏空';
        return '趋势混合';
    }

    function riskText(raw) {
        const txt = String(raw || '').toLowerCase();
        if (txt === 'low') return '低风险';
        if (txt === 'medium') return '中风险';
        if (txt === 'high') return '高风险';
        if (txt === 'extreme') return '极高风险';
        return '中风险';
    }

    function normalizeMarket(market) {
        const m = String(market || '').toLowerCase();
        if (m.includes('cn')) return 'cn_equity';
        return 'us_equity';
    }

    function currencySymbol(currency, market) {
        const code = String(currency || '').toUpperCase();
        if (code === 'CNY' || normalizeMarket(market) === 'cn_equity') return '¥';
        return '$';
    }

    function formatPrice(value, currency, market) {
        const n = safeNumber(value);
        if (n === null) return '--';
        const symbol = currencySymbol(currency, market);
        return `${symbol}${n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }

    function formatPct(value, digits = 2, signed = true) {
        const n = safeNumber(value);
        if (n === null) return '--';
        const v = n * 100;
        const sign = signed && v > 0 ? '+' : '';
        return `${sign}${v.toFixed(digits)}%`;
    }

    function formatRatio(value, digits = 2) {
        const n = safeNumber(value);
        if (n === null) return '--';
        return n.toFixed(digits);
    }

    function inferConfidence(signal, tracking, pUp) {
        let conf = pickFirstNumber(signal?.confidence_score, signal?.confidence, tracking?.confidence_score);
        if (conf !== null && conf <= 1) conf *= 100;
        if (conf !== null) return clamp(conf, 1, 99);
        if (pUp !== null) return clamp(50 + Math.abs(pUp - 0.5) * 100, 1, 99);
        return 50;
    }

    function inferDirectionProb(signal, tracking, action) {
        let pUp = pickFirstNumber(signal?.p_up);
        let pDown = pickFirstNumber(signal?.p_down);
        const conf = inferConfidence(signal, tracking, pUp);
        if (pUp === null && pDown === null) {
            const bias = clamp((conf - 50) / 100, 0, 0.45);
            if (action === 'LONG') pUp = 0.5 + bias;
            else if (action === 'SHORT') pUp = 0.5 - bias;
            else pUp = 0.5;
        }
        if (pUp === null && pDown !== null) pUp = 1 - pDown;
        if (pDown === null && pUp !== null) pDown = 1 - pUp;
        pUp = clamp(safeNumber(pUp) ?? 0.5, 0.01, 0.99);
        pDown = clamp(safeNumber(pDown) ?? (1 - pUp), 0.01, 0.99);
        return { pUp, pDown };
    }

    function inferRuleStatus(tracking, action) {
        const ruleStatus = String(tracking?.rule_status || '').toLowerCase();
        if (ruleStatus === 'executable') return '可执行';
        if (ruleStatus === 'watch') return '观察';
        if (ruleStatus === 'paused') return '暂停';
        if (action === 'WAIT') return '观察';
        return '可执行';
    }

    function inferPlan(signal, tracking, market) {
        const action = actionCode(signal?.action || signal?.policy_action || tracking?.action || tracking?.recommended_action);
        const currentPrice = pickFirstNumber(signal?.current_price, tracking?.current_price);
        const q50Raw = pickFirstNumber(signal?.q50_change_pct, tracking?.predicted_change_pct) ?? 0;
        const q10Raw = pickFirstNumber(signal?.q10_change_pct);
        const q90Raw = pickFirstNumber(signal?.q90_change_pct);
        const volScore = pickFirstNumber(signal?.volatility_score, tracking?.edge_risk, Math.abs(q90Raw ?? 0) + Math.abs(q10Raw ?? 0));

        let side = action;
        if (side === 'WAIT') side = q50Raw >= 0 ? 'LONG' : 'SHORT';

        let q50 = q50Raw;
        if (side === 'LONG' && q50 < 0) q50 = Math.abs(q50);
        if (side === 'SHORT' && q50 > 0) q50 = -Math.abs(q50);
        if (q50 === 0) q50 = side === 'SHORT' ? -0.005 : 0.005;

        let q10 = q10Raw;
        let q90 = q90Raw;
        if (q10 === null || q90 === null) {
            const span = Math.max(Math.abs(volScore ?? 0.04), 0.02);
            q10 = q50 - span * 0.8;
            q90 = q50 + span * 0.8;
        }

        const entry = currentPrice;
        if (entry === null) {
            return null;
        }

        const slChange = side === 'LONG'
            ? Math.min(q10, -0.008)
            : Math.max(q90, 0.008);
        const tpChange = q50;
        const tp2Change = side === 'LONG'
            ? Math.max(q90, tpChange * 1.3)
            : Math.min(q10, tpChange * 1.3);

        const stopLoss = entry * (1 + slChange);
        const takeProfit = entry * (1 + tpChange);
        const takeProfit2 = entry * (1 + tp2Change);
        const predictedPrice = pickFirstNumber(signal?.target_price, signal?.target_price_q50, tracking?.predicted_price, entry * (1 + q50));

        const riskPct = side === 'LONG'
            ? Math.abs((entry - stopLoss) / entry)
            : Math.abs((stopLoss - entry) / entry);
        const rewardPct = side === 'LONG'
            ? Math.abs((takeProfit - entry) / entry)
            : Math.abs((entry - takeProfit) / entry);
        const rr = rewardPct > 0 && riskPct > 0 ? rewardPct / riskPct : null;

        const costBps = 20;
        const edge = pickFirstNumber(tracking?.edge_score, q50 - costBps / 10000);

        const { pUp, pDown } = inferDirectionProb(signal, tracking, action);
        const pDir = side === 'SHORT' ? pDown : pUp;
        const rrFactor = clamp((safeNumber(rr) ?? 1) / 2, 0, 1);
        const edgeFactor = clamp((((safeNumber(edge) ?? 0) * 10000) + 10) / 40, 0, 1);
        const pTp1 = clamp(0.55 * pDir + 0.25 * rrFactor + 0.2 * edgeFactor, 0.05, 0.95);
        const pTp2 = clamp(pTp1 * (0.55 + 0.25 * rrFactor), 0.02, 0.9);
        const expectedValue = pTp1 * rewardPct - (1 - pTp1) * riskPct;

        const confidence = inferConfidence(signal, tracking, pUp);
        const signalStrengthRaw = String(signal?.signal_strength || '').toLowerCase();
        const signalStrength = signalStrengthRaw === 'strong'
            ? '强信号'
            : signalStrengthRaw === 'medium'
                ? '中信号'
                : '弱信号';

        const score = pickFirstNumber(tracking?.total_score, confidence);
        const riskLevel = String(signal?.risk_level || tracking?.risk_level || 'medium').toLowerCase();
        const ruleStatus = inferRuleStatus(tracking, action);
        const symbol = String(signal?.symbol || tracking?.symbol || '').toUpperCase();
        const reasonTextRaw = String(tracking?.reason || signal?.policy_reason || '').trim();
        const reasonText = reasonTextRaw.replace(/_/g, ' ');
        const method = 'q10/q50/q90';
        const expectedDate = signal?.forecast_generated_at || new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString();

        return {
            action,
            side,
            symbol,
            confidence,
            signalStrength,
            score,
            currentPrice: entry,
            predictedPrice,
            q10,
            q50,
            q90,
            pUp,
            pDown,
            entry,
            stopLoss,
            takeProfit,
            takeProfit2,
            rr,
            edge,
            riskPct,
            rewardPct,
            pTp1,
            pTp2,
            expectedValue,
            ruleStatus,
            riskLevel,
            trend: signal?.trend_label || 'neutral',
            reasonText,
            expectedDate,
            method,
            market,
            tracking,
            volatilityScore: volScore,
        };
    }

    function buildChecks(plan) {
        const pUp = safeNumber(plan?.pUp) ?? 0.5;
        const pDown = safeNumber(plan?.pDown) ?? 0.5;
        const edge = safeNumber(plan?.edge) ?? 0;
        const conf = safeNumber(plan?.confidence) ?? 50;
        const risk = String(plan?.riskLevel || 'medium').toLowerCase();
        const ruleStatus = String(plan?.ruleStatus || '');

        const longChecks = [
            ['p_up >= 55%', pUp >= 0.55],
            ['edge_after_cost > 0', edge > 0],
            ['confidence >= 60', conf >= 60],
            ['风险不过高', !['high', 'extreme'].includes(risk)],
            ['规则状态可执行', ruleStatus === '可执行'],
        ];
        const shortChecks = [
            ['p_down >= 55%', pDown >= 0.55],
            ['edge_after_cost > 0', edge > 0],
            ['confidence >= 60', conf >= 60],
            ['风险不过高', !['high', 'extreme'].includes(risk)],
            ['规则状态可执行', ruleStatus === '可执行'],
        ];

        let selectedChecks = longChecks;
        if (plan?.side === 'SHORT') selectedChecks = shortChecks;
        if (plan?.action === 'WAIT') {
            selectedChecks = [
                ['方向信号强度足够', Math.max(pUp, pDown) >= 0.6],
                ['净Edge为正', edge > 0],
                ['置信度达标', conf >= 60],
                ['规则通过后再执行', ruleStatus === '可执行'],
            ];
        }
        return { longChecks, shortChecks, selectedChecks };
    }

    function renderCheckList(id, checks) {
        const html = (checks || [])
            .map(([label, ok]) => `<li>${ok ? '✅' : '❌'} ${label}</li>`)
            .join('');
        setHTML(id, html || '<li>--</li>');
    }

    function renderDecisionCard(plan, currency) {
        const actionEl = document.getElementById('eqDecisionAction');
        if (actionEl) {
            actionEl.textContent = actionText(plan.action);
            actionEl.className = `decision-action ${actionClass(plan.action)}`;
        }

        const statusText = plan.ruleStatus === '可执行'
            ? '已到价 + 规则通过，可执行。'
            : '规则未通过 · 等待触发';
        setText('eqDecisionExec', statusText);
        setText('eqDecisionStrength', `${plan.signalStrength} / ${formatRatio(plan.score, 1)}`);
        setText('eqDecisionRisk', `新闻风险：${riskText(plan.riskLevel)}`);
        setText('eqDecisionEntry', formatPrice(plan.entry, currency, plan.market));
        setText('eqDecisionSLTP', `${formatPrice(plan.stopLoss, currency, plan.market)} / ${formatPrice(plan.takeProfit, currency, plan.market)}`);
        setText('eqDecisionRR', `${formatRatio(plan.rr)} / ${formatPct(plan.pTp1, 1, false)}`);

        const checks = buildChecks(plan);
        const supports = checks.selectedChecks.filter((x) => x[1]).map((x) => x[0]);
        const blocker = checks.selectedChecks.find((x) => !x[1])?.[0] || '暂无阻断';
        const reasonLine = `✅ ${supports[0] || '--'} | ✅ ${supports[1] || '--'} | ❌ ${blocker}`;
        setText('eqDecisionReasons', reasonLine);

        setText('eqTradeMetricQ50', formatPct(plan.q50));
        setText('eqTradeMetricRisk', formatPct(plan.riskPct, 2, false));
        setText('eqTradeMetricRR', formatRatio(plan.rr));
        setText('eqTradeMetricEdge', formatPct(plan.edge));
        setText('eqTradeMetricTP1', formatPct(plan.pTp1, 1, false));
        setText('eqTradeMetricTP2', formatPct(plan.pTp2, 1, false));
        setText('eqTradeMetricEV', formatPct(plan.expectedValue));

        renderCheckList('eqRuleLong', checks.longChecks);
        renderCheckList('eqRuleShort', checks.shortChecks);
        renderCheckList('eqRuleCurrent', checks.selectedChecks);
    }

    function renderSnapshot(plan, signal, currency) {
        setText('eqSnapshotCurrent', formatPrice(plan.currentPrice, currency, plan.market));
        setText('eqSnapshotPredicted', formatPrice(plan.predictedPrice, currency, plan.market));
        setText('eqSnapshotChange', formatPct(plan.q50));
        setText('eqSnapshotAction', actionText(plan.action).split(' / ')[1] || '观望');

        const expectedDate = new Date(plan.expectedDate);
        const expectedText = Number.isNaN(expectedDate.getTime())
            ? String(plan.expectedDate || '--')
            : expectedDate.toLocaleString('zh-CN');
        setText('eqExpectedDate', expectedText);

        const priceSource = signal?.price_source || signal?.provider || '实时行情';
        const method = plan.method || 'q10/q50/q90';
        setText('eqSnapshotMeta', `价格源: ${priceSource} | 预测方法: ${method}`);
    }

    function renderSignalPlan(plan, currency) {
        setText('eqSignalCurrent', actionText(plan.action).split(' / ')[1] || '观望');
        setText('eqSignalTrend', trendText(plan.trend));
        setText('eqSignalSL', plan.action === 'WAIT' ? '不适用（观望）' : formatPrice(plan.stopLoss, currency, plan.market));
        setText('eqSignalTP', plan.action === 'WAIT' ? '不适用（观望）' : formatPrice(plan.takeProfit, currency, plan.market));
        setText('eqSignalRR', formatRatio(plan.rr));

        const reason = plan.reasonText
            ? `开单理由: ${plan.reasonText}`
            : '开单理由: 模型按 p_up / edge / confidence / risk 组合决策。';
        setText('eqSignalReason', reason);
    }

    function renderFactors(plan) {
        const tracking = plan.tracking || {};
        const liq = safeNumber(tracking.liquidity_score);
        const edge = safeNumber(tracking.edge_score ?? plan.edge);
        const growth = safeNumber(tracking.predicted_change_pct ?? plan.q50);
        const momentum = safeNumber(plan.pUp - 0.5) * 2;
        const reversal = safeNumber(plan.q10) !== null && safeNumber(plan.q90) !== null
            ? -((plan.q90 + plan.q10) / 2)
            : null;
        const vol = safeNumber(plan.volatilityScore);
        const lowVol = vol !== null ? clamp(1 - vol * 4, -1, 1) : null;

        const sizeFactor = liq !== null ? liq / 30 : null;
        const valueFactor = edge;

        setText('eqFactorSize', sizeFactor !== null ? formatRatio(sizeFactor, 3) : '--');
        setText('eqFactorValue', valueFactor !== null ? formatRatio(valueFactor, 4) : '--');
        setText('eqFactorGrowth', growth !== null ? formatPct(growth) : '--');
        setText('eqFactorMomentum', momentum !== null ? formatPct(momentum) : '--');
        setText('eqFactorReversal', reversal !== null ? formatPct(reversal) : '--');
        setText('eqFactorLowVol', lowVol !== null ? formatPct(lowVol) : '--');
    }

    function renderNewsList(plan) {
        const tracking = plan.tracking || {};
        const alerts = String(tracking.alerts || '').trim();
        const reason = String(plan.reasonText || '').trim() || '模型信号来源于多因子与概率阈值。';
        const quality = safeNumber(tracking.data_quality_score);
        const miss = safeNumber(tracking.history_missing_rate);

        const rows = [
            `模型信号：${reason}`,
            `规则状态：${plan.ruleStatus} | 风险等级：${riskText(plan.riskLevel)}`,
            `方向概率：P(up) ${formatPct(plan.pUp, 1, false)} | P(down) ${formatPct(plan.pDown, 1, false)}`,
        ];
        if (quality !== null || miss !== null) {
            rows.push(`数据质量：${quality !== null ? quality.toFixed(1) : '--'} | 缺失率：${miss !== null ? (miss * 100).toFixed(2) + '%' : '--'}`);
        }
        if (alerts) {
            rows.push(`预警标签：${alerts}`);
        }

        const html = rows.map((txt) => `<li>${txt}</li>`).join('');
        setHTML('eqNewsList', html);
    }

    function createProjectionChart(plan, currency) {
        const canvas = document.getElementById('eqProjectionChart');
        if (!canvas || typeof Chart === 'undefined') return;

        const nowLabel = '现在';
        const expLabel = '预期日期';
        const labels = [nowLabel, expLabel];
        const current = plan.currentPrice;
        const q10Price = current * (1 + plan.q10);
        const q50Price = current * (1 + plan.q50);
        const q90Price = current * (1 + plan.q90);

        const datasets = [
            {
                label: 'q90',
                data: [current, q90Price],
                borderColor: '#60a5fa',
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                tension: 0.25,
                pointRadius: 3,
            },
            {
                label: 'q10',
                data: [current, q10Price],
                borderColor: '#1d4ed8',
                backgroundColor: 'rgba(96, 165, 250, 0.20)',
                fill: '-1',
                borderWidth: 1.5,
                tension: 0.25,
                pointRadius: 3,
            },
            {
                label: 'q50',
                data: [current, q50Price],
                borderColor: '#22d3ee',
                backgroundColor: 'transparent',
                borderWidth: 3,
                tension: 0.25,
                pointRadius: 4,
            },
            {
                label: 'Entry',
                data: [plan.entry, plan.entry],
                borderColor: '#94a3b8',
                borderDash: [4, 3],
                borderWidth: 1,
                pointRadius: 0,
            },
            {
                label: 'SL',
                data: [plan.stopLoss, plan.stopLoss],
                borderColor: '#ef4444',
                borderDash: [6, 3],
                borderWidth: 1,
                pointRadius: 0,
            },
            {
                label: 'TP',
                data: [plan.takeProfit, plan.takeProfit],
                borderColor: '#22c55e',
                borderDash: [6, 3],
                borderWidth: 1,
                pointRadius: 0,
            },
        ];

        if (state.projectionChart) {
            state.projectionChart.data.labels = labels;
            state.projectionChart.data.datasets = datasets;
            state.projectionChart.update('none');
            return;
        }

        state.projectionChart = new Chart(canvas.getContext('2d'), {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        labels: { color: '#a7b3c9', boxWidth: 12, usePointStyle: true },
                    },
                    tooltip: {
                        backgroundColor: 'rgba(6, 11, 22, 0.95)',
                        borderColor: 'rgba(34, 211, 238, 0.2)',
                        borderWidth: 1,
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${formatPrice(ctx.parsed.y, currency, plan.market)}`,
                        },
                    },
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.06)' },
                        ticks: { color: '#94a3b8' },
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.06)' },
                        ticks: {
                            color: '#94a3b8',
                            callback: (val) => formatPrice(val, currency, plan.market),
                        },
                    },
                },
            },
        });
    }

    async function getTrackingItem(market, symbol) {
        const key = `${normalizeMarket(market)}:${String(symbol || '').toUpperCase()}`;
        const cached = state.trackingCache.get(key);
        if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
            return cached.data;
        }

        let item = null;
        try {
            if (window.api?.tracking?.getDetail) {
                const detail = await window.api.tracking.getDetail(key);
                if (detail?.item && typeof detail.item === 'object') item = detail.item;
            }
        } catch (_) {
            // fallback below
        }

        if (!item) {
            try {
                if (window.api?.tracking?.getOverview) {
                    const overview = await window.api.tracking.getOverview({ market: normalizeMarket(market) });
                    const list = Array.isArray(overview?.items) ? overview.items : [];
                    const sym = String(symbol || '').toUpperCase();
                    item = list.find((x) => String(x?.symbol || '').toUpperCase() === sym) || null;
                }
            } catch (_) {
                item = null;
            }
        }

        state.trackingCache.set(key, { ts: Date.now(), data: item });
        return item;
    }

    async function getExecutionOverview() {
        if (state.executionCache.data && Date.now() - state.executionCache.ts < CACHE_TTL_MS) {
            return state.executionCache.data;
        }
        try {
            const data = await window.api?.execution?.getOverview?.();
            state.executionCache = { ts: Date.now(), data: data || null };
            return data || null;
        } catch (_) {
            state.executionCache = { ts: Date.now(), data: null };
            return null;
        }
    }

    function findRecentTrades(execution, market, symbol, currency) {
        if (!execution || typeof execution !== 'object') return [];
        const m = normalizeMarket(market);
        const sym = String(symbol || '').toUpperCase();
        const rows = [];
        const all = []
            .concat(Array.isArray(execution.closed_positions) ? execution.closed_positions : [])
            .concat(Array.isArray(execution.open_positions) ? execution.open_positions : [])
            .concat(Array.isArray(execution.positions) ? execution.positions : []);

        for (const row of all) {
            if (String(row?.market || '').toLowerCase() !== m) continue;
            if (String(row?.symbol || '').toUpperCase() !== sym) continue;
            rows.push({
                time: row.entry_time_utc || '--',
                side: String(row.side || '--').toUpperCase(),
                signal: String(row.status || '--'),
                entry: formatPrice(row.entry_price, currency, market),
                sl: formatPrice(row.stop_loss, currency, market),
                tp: formatPrice(row.take_profit, currency, market),
                pnl: formatPct(row.net_pnl_pct),
            });
        }
        return rows.slice(0, 8);
    }

    function renderBacktest(plan, execution, currency) {
        const conf = safeNumber(plan.confidence) ?? 50;
        const vol = Math.abs(safeNumber(plan.volatilityScore) ?? 0.05);
        const proxyReturn = safeNumber(plan.q50);
        const proxySharpe = safeNumber(plan.edge) !== null ? (safeNumber(plan.edge) * 100 + 0.8) : conf / 80;
        const proxyMdd = -Math.max(vol, 0.01);
        const proxyWin = clamp(conf / 100, 0.05, 0.95);

        setText('eqBacktestReturn', formatPct(proxyReturn));
        setText('eqBacktestSharpe', formatRatio(proxySharpe, 2));
        setText('eqBacktestMdd', formatPct(proxyMdd));
        setText('eqBacktestWinRate', formatPct(proxyWin, 1, false));

        const strategyRows = [
            {
                name: 'Policy (snapshot)',
                ret: formatPct(proxyReturn),
                sharpe: formatRatio(proxySharpe, 2),
                mdd: formatPct(proxyMdd),
                win: formatPct(proxyWin, 1, false),
            },
        ];
        setHTML(
            'eqBacktestBody',
            strategyRows
                .map((r) => `<tr><td>${r.name}</td><td>${r.ret}</td><td>${r.sharpe}</td><td>${r.mdd}</td><td>${r.win}</td></tr>`)
                .join(''),
        );

        const trades = findRecentTrades(execution, plan.market, plan.symbol, currency);
        const tradeHtml = trades.length
            ? trades
                .map((r) => `<tr><td>${r.time}</td><td>${r.side}</td><td>${r.signal}</td><td>${r.entry}</td><td>${r.sl}</td><td>${r.tp}</td><td>${r.pnl}</td></tr>`)
                .join('')
            : '<tr><td colspan="7" class="muted-cell">暂无同标的开单记录</td></tr>';
        setHTML('eqRecentTradesBody', tradeHtml);
    }

    async function render(options) {
        const market = normalizeMarket(options?.market);
        const signal = options?.signal || {};
        const symbol = String(options?.symbol || signal?.symbol || '').toUpperCase();
        const currency = options?.currency || (market === 'cn_equity' ? 'CNY' : 'USD');
        if (!symbol) return;

        const tracking = await getTrackingItem(market, symbol);
        const plan = inferPlan(signal, tracking, market);
        if (!plan) return;

        renderDecisionCard(plan, currency);
        renderSnapshot(plan, signal, currency);
        renderSignalPlan(plan, currency);
        renderFactors(plan);
        renderNewsList(plan);
        createProjectionChart(plan, currency);

        const execution = await getExecutionOverview();
        renderBacktest(plan, execution, currency);
    }

    window.equityDeepPanel = { render };
})();

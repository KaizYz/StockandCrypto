// ========================================
// Session Crypto Page (Web parity with dashboard/app.py)
// ========================================

document.addEventListener("DOMContentLoaded", () => {
    initSessionCryptoPage().catch((err) => {
        console.error(err);
        showToast(`加载失败: ${err.message || "未知错误"}`, "error");
    });
});

const scState = {
    payload: null,
    metric: "p_up",
    simChart: null,
    controlsReady: false,
    syncingControls: false,
    refreshTimer: null,
};

function scEl(id) {
    return document.getElementById(id);
}

function scNum(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function scAsProb(value) {
    const n = scNum(value);
    if (n === null) return null;
    if (Math.abs(n) > 1) return n / 100;
    return n;
}

function scAsPct(value) {
    const n = scNum(value);
    if (n === null) return null;
    return Math.abs(n) <= 1 ? n * 100 : n;
}

function scFmtPct(value, digits = 2) {
    const p = scAsPct(value);
    if (p === null) return "-";
    return `${p.toFixed(digits)}%`;
}

function scFmtSignedPct(value, digits = 2) {
    const p = scAsPct(value);
    if (p === null) return "-";
    const sign = p > 0 ? "+" : "";
    return `${sign}${p.toFixed(digits)}%`;
}

function scFmtPrice(value, digits = 2) {
    const n = scNum(value);
    if (n === null) return "-";
    return `$${n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
}

function scFmtNum(value, digits = 2) {
    const n = scNum(value);
    if (n === null) return "-";
    return n.toFixed(digits);
}

function scTrendText(key) {
    const txt = String(key || "").toLowerCase();
    if (txt.includes("bull")) return "看涨";
    if (txt.includes("bear")) return "看跌";
    return "震荡";
}

function scRiskText(key) {
    const txt = String(key || "").toLowerCase();
    if (txt === "low") return "低";
    if (txt === "medium") return "中";
    if (txt === "high" || txt === "extreme") return "高";
    return txt || "-";
}

function scActionClass(action) {
    const key = String(action || "").toUpperCase();
    if (key === "LONG") return "long";
    if (key === "SHORT") return "short";
    return "flat";
}

function scSelectOptions(selectEl, options, selected, labelFn = null) {
    if (!selectEl) return;
    const current = String(selected ?? "");
    selectEl.innerHTML = "";
    (options || []).forEach((opt) => {
        const value = typeof opt === "object" ? String(opt.value ?? opt.key ?? "") : String(opt);
        const label = labelFn
            ? labelFn(opt)
            : (typeof opt === "object" ? String(opt.label ?? opt.name_zh ?? value) : value);
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        if (value === current) option.selected = true;
        selectEl.appendChild(option);
    });
}

function scBuildParams() {
    return {
        symbol: scEl("scSymbol")?.value,
        exchange: scEl("scExchange")?.value,
        market_type: scEl("scMarketType")?.value,
        mode: scEl("scMode")?.value,
        horizon_hours: scEl("scHorizon")?.value,
        lookforward_days: scEl("scLookforward")?.value,
        risk_profile: scEl("scRisk")?.value,
        rank_key: scEl("scRank")?.value,
        top_n: scEl("scTopN")?.value,
        cost_bps: scEl("scCost")?.value,
    };
}

async function initSessionCryptoPage() {
    bindSessionCryptoControls();
    bindSessionCryptoHeatmapTabs();
    await loadSessionCryptoData(true);
    if (scState.refreshTimer) clearInterval(scState.refreshTimer);
    scState.refreshTimer = setInterval(() => loadSessionCryptoData(false), 5 * 60 * 1000);
}

function bindSessionCryptoControls() {
    const ids = [
        "scSymbol",
        "scExchange",
        "scMarketType",
        "scMode",
        "scHorizon",
        "scLookforward",
        "scRisk",
        "scRank",
        "scTopN",
        "scCost",
    ];
    ids.forEach((id) => {
        const node = scEl(id);
        if (!node) return;
        node.addEventListener("change", () => {
            if (scState.syncingControls) return;
            loadSessionCryptoData(true).catch((err) => {
                console.error(err);
                showToast(`刷新失败: ${err.message || "未知错误"}`, "error");
            });
        });
    });

    scEl("scRefreshBtn")?.addEventListener("click", () => {
        loadSessionCryptoData(true).catch((err) => {
            console.error(err);
            showToast(`刷新失败: ${err.message || "未知错误"}`, "error");
        });
    });

    scEl("scCompare")?.addEventListener("change", () => {
        if (!scState.payload) return;
        renderSessionCryptoCompare(scState.payload.compare || {});
    });
}

function bindSessionCryptoHeatmapTabs() {
    const tabs = document.querySelectorAll("#scHeatmapTabs .heatmap-tab");
    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const metric = String(tab.dataset.metric || "p_up");
            scState.metric = metric;
            tabs.forEach((x) => x.classList.remove("active"));
            tab.classList.add("active");
            renderSessionCryptoHeatmap(metric);
        });
    });
}

async function loadSessionCryptoData(showLoading) {
    if (showLoading) scEl("scMetaLine").textContent = "加载中...";
    const payload = await api.session.getCrypto(scBuildParams());
    if (!payload || payload.ok === false) {
        throw new Error(payload?.error || "session_crypto_api_failed");
    }
    scState.payload = payload;
    syncSessionCryptoControls(payload.controls || {});
    renderSessionCryptoPayload(payload);
    if (showLoading) showToast("数据已更新", "success", 1200);
}

function syncSessionCryptoControls(controls) {
    if (!controls || typeof controls !== "object") return;
    const selected = controls.selected || {};
    scState.syncingControls = true;

    scSelectOptions(scEl("scSymbol"), controls.symbol_options || [], selected.symbol);
    scSelectOptions(scEl("scExchange"), controls.exchange_options || [], selected.exchange);
    scSelectOptions(scEl("scMarketType"), controls.market_type_options || [], selected.market_type);
    scSelectOptions(scEl("scMode"), controls.mode_options || [], selected.mode);
    scSelectOptions(scEl("scHorizon"), controls.horizon_options || [], selected.horizon_hours);
    scSelectOptions(scEl("scRisk"), controls.risk_profile_options || [], selected.risk_profile);

    const rankOptions = Object.entries(controls.rank_options || {}).map(([value, label]) => ({ value, label }));
    scSelectOptions(scEl("scRank"), rankOptions, selected.rank_key, (opt) => String(opt.label || opt.value));

    if (scEl("scLookforward")) scEl("scLookforward").value = String(selected.lookforward_days ?? 14);
    if (scEl("scTopN")) scEl("scTopN").value = String(selected.top_n ?? 5);
    if (scEl("scCost")) scEl("scCost").value = String(selected.cost_bps ?? 8);

    scState.syncingControls = false;
    scState.controlsReady = true;
}

function renderSessionCryptoPayload(payload) {
    renderSessionCryptoMeta(payload.meta || {});
    renderSessionCryptoDecision(payload.decision || {});
    renderSessionCryptoSessions(payload.sessions || []);
    renderSessionCryptoCompare(payload.compare || {});
    renderSessionCryptoHeatmap(scState.metric);
    renderSessionCryptoDaily(payload.daily || []);
    renderSessionCryptoTop(payload.top || {});
    renderSessionCryptoNotes(payload.notes || []);
    renderSessionCryptoSim(payload.sim_path || []);
}

function renderSessionCryptoMeta(meta) {
    const line = [
        `最新价格: ${scFmtPrice(meta.current_price)}`,
        `更新时间: ${String(meta.data_updated_at_bj || "-")}`,
        `模式/周期: ${String(meta.mode_actual || "-")} / ${String(meta.horizon_hours || "-")}h`,
    ].join(" | ");
    scEl("scMetaLine").textContent = line;

    const info = [
        `symbol: ${String(meta.symbol || "-")}`,
        `exchange: ${String(meta.exchange_actual || meta.exchange || "-")}`,
        `market_type: ${String(meta.market_type || "-")}`,
        `forecast_generated_at_bj: ${String(meta.forecast_generated_at_bj || "-")}`,
        `data_updated_at_bj: ${String(meta.data_updated_at_bj || "-")}`,
        `model_version: ${String(meta.model_version || "-")}`,
        `data_version: ${String(meta.data_version || "-")}`,
        `data_source_actual: ${String(meta.data_source_actual || "-")}`,
    ].join("\n");
    scEl("scDataInfo").textContent = info;
}

function renderSessionCryptoDecision(decision) {
    const plan = decision.plan || {};
    const consensus = decision.consensus || {};
    const action = String(plan.action || "WAIT").toUpperCase();
    const actionCn = String(plan.action_cn || "观望");
    const statusText = String(plan.trade_status_text || "-");
    const statusNote = String(plan.trade_status_note || "-");

    scEl("scDecisionCard").innerHTML = `
        <div class="decision-top">
            <div>
                <div class="decision-k">一眼决策卡（交易时段）</div>
                <div class="decision-action ${scActionClass(action)}">${action} / ${actionCn}</div>
                <div class="decision-note">${statusText} · ${statusNote}</div>
                <div class="decision-note">${String(consensus.badge || "-")} · ${String(consensus.detail || "-")}</div>
            </div>
            <div class="decision-right">
                <div class="decision-k">模型健康 / 新鲜度</div>
                <div class="decision-v">${String(decision.model_health_text || "-")}</div>
                <div class="decision-note">${String(decision.threshold_text || "-")}</div>
            </div>
        </div>
        <div class="decision-grid">
            <div><span>执行状态</span><strong>${statusText}</strong></div>
            <div><span>入场价</span><strong>${scFmtPrice(plan.entry)}</strong></div>
            <div><span>止损 SL</span><strong>${scFmtPrice(plan.stop_loss)}</strong></div>
            <div><span>止盈 TP1</span><strong>${scFmtPrice(plan.take_profit)}</strong></div>
            <div><span>止盈 TP2</span><strong>${scFmtPrice(plan.take_profit_2)}</strong></div>
            <div><span>RR(TP1) / RR2(TP2)</span><strong>${scFmtNum(plan.rr_tp1 ?? plan.rr, 2)} / ${scFmtNum(plan.rr_tp2, 2)}</strong></div>
        </div>
    `;

    const q50Signed = String(plan.plan_side || "LONG").toUpperCase() === "SHORT"
        ? -(scNum(plan.q50) || 0)
        : (scNum(plan.q50) || 0);
    const riskPct = (() => {
        const entry = scNum(plan.entry);
        const sl = scNum(plan.stop_loss);
        if (!entry || !sl) return null;
        return Math.abs((entry - sl) / entry);
    })();

    scEl("scTradePlanMetrics").innerHTML = `
        <div class="decision-metric"><span>预期收益(q50)</span><strong>${scFmtSignedPct(q50Signed)}</strong></div>
        <div class="decision-metric"><span>风险（到SL）</span><strong>${scFmtPct(riskPct)}</strong></div>
        <div class="decision-metric"><span>RR(TP1)</span><strong>${scFmtNum(plan.rr_tp1 ?? plan.rr, 2)}</strong></div>
        <div class="decision-metric"><span>RR2(TP2)</span><strong>${scFmtNum(plan.rr_tp2, 2)}</strong></div>
        <div class="decision-metric"><span>扣成本后净Edge</span><strong>${scFmtSignedPct(plan.plan_side === "SHORT" ? plan.edge_short : plan.edge_long)}</strong></div>
    `;
}

function renderSessionCryptoSessions(rows) {
    const wrap = scEl("scSessionCards");
    if (!wrap) return;
    if (!Array.isArray(rows) || rows.length === 0) {
        wrap.innerHTML = '<div class="empty-state"><p>暂无时段数据</p></div>';
        return;
    }
    wrap.innerHTML = rows.map((row) => {
        const pUp = scAsProb(row.p_up);
        const pDown = scAsProb(row.p_down);
        const direction = pUp === null ? "-" : (pUp >= 0.53 ? "看涨" : (pUp <= 0.47 ? "看跌" : "震荡"));
        return `
            <div class="session-card">
                <div class="session-header">
                    <div class="session-info">
                        <h3 class="session-name">${String(row.session_name_cn || row.session_name || "-")}</h3>
                        <span class="session-time">${String(row.session_hours || "-")}</span>
                    </div>
                </div>
                <div class="session-content">
                    <div class="detail-item"><span class="detail-label">P(up)</span><span class="detail-value">${scFmtPct(pUp)}</span></div>
                    <div class="detail-item"><span class="detail-label">P(down)</span><span class="detail-value">${scFmtPct(pDown)}</span></div>
                    <div class="detail-item"><span class="detail-label">q50</span><span class="detail-value">${scFmtSignedPct(row.q50_change_pct)}</span></div>
                    <div class="detail-item"><span class="detail-label">目标价(q50)</span><span class="detail-value">${scFmtPrice(row.target_price_q50)}</span></div>
                    <div class="detail-item"><span class="detail-label">方向</span><span class="detail-value">${direction}</span></div>
                    <div class="detail-item"><span class="detail-label">风险 / 置信度</span><span class="detail-value">${scRiskText(row.risk_level)} / ${scFmtPct(row.confidence_score)}</span></div>
                    <div class="detail-item"><span class="detail-label">策略动作</span><span class="detail-value">${String(row.policy_action || "-")}</span></div>
                </div>
            </div>
        `;
    }).join("");
}

function renderSessionCryptoCompare(compare) {
    const wrap = scEl("scCompareWrap");
    if (!wrap) return;
    const show = scEl("scCompare")?.checked ?? true;
    if (!show) {
        wrap.innerHTML = '<div class="empty-state"><p>已关闭对照视图</p></div>';
        return;
    }

    const rows = Array.isArray(compare?.rows) ? compare.rows : [];
    if (!rows.length) {
        wrap.innerHTML = '<div class="empty-state"><p>对照数据不足</p></div>';
        return;
    }

    const mainMode = String(compare.main_mode || "main");
    const cmpMode = String(compare.compare_mode || "compare");
    wrap.innerHTML = `
        <table class="stock-table">
            <thead>
                <tr>
                    <th>时段</th>
                    <th>${mainMode} p_up</th>
                    <th>${cmpMode} p_up</th>
                    <th>Δp_up</th>
                    <th>${mainMode} q50</th>
                    <th>${cmpMode} q50</th>
                    <th>Δq50</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map((row) => {
                    const mP = row[`${mainMode}_p_up`];
                    const cP = row[`${cmpMode}_p_up`];
                    const mQ = row[`${mainMode}_q50`];
                    const cQ = row[`${cmpMode}_q50`];
                    return `
                        <tr>
                            <td>${String(row.session_name_cn || row.session_name || "-")}</td>
                            <td>${scFmtPct(mP)}</td>
                            <td>${scFmtPct(cP)}</td>
                            <td class="${(scNum(row.delta_p_up) || 0) >= 0 ? "positive" : "negative"}">${scFmtSignedPct(row.delta_p_up)}</td>
                            <td>${scFmtSignedPct(mQ)}</td>
                            <td>${scFmtSignedPct(cQ)}</td>
                            <td class="${(scNum(row.delta_q50) || 0) >= 0 ? "positive" : "negative"}">${scFmtSignedPct(row.delta_q50)}</td>
                        </tr>
                    `;
                }).join("")}
            </tbody>
        </table>
    `;
}

function renderSessionCryptoHeatmap(metric) {
    const grid = scEl("scHeatmapGrid");
    if (!grid) return;
    const rows = Array.isArray(scState.payload?.hourly) ? scState.payload.hourly : [];
    if (!rows.length) {
        grid.innerHTML = '<div class="empty-state"><p>暂无小时级数据</p></div>';
        return;
    }

    const values = rows
        .map((row) => {
            const raw = row?.[metric];
            if (metric === "p_up" || metric === "p_down" || metric === "confidence_score") {
                return scAsPct(raw);
            }
            return scAsPct(raw);
        })
        .filter((v) => scNum(v) !== null);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;
    const span = Math.max(1e-9, max - min);

    grid.innerHTML = rows.map((row) => {
        const hour = Number(row.hour_bj ?? 0) % 24;
        const pUp = scAsPct(metric === "p_up" ? row.p_up : row[metric]);
        const showVal = pUp === null ? "-" : `${pUp.toFixed(2)}%`;
        const ratio = pUp === null ? 0 : (pUp - min) / span;
        const opacity = 0.1 + ratio * 0.5;
        let color = `rgba(0, 212, 170, ${opacity.toFixed(3)})`;
        if (metric === "p_down") color = `rgba(255, 107, 107, ${opacity.toFixed(3)})`;
        if (metric === "volatility_score") color = `rgba(212, 175, 55, ${opacity.toFixed(3)})`;
        return `
            <div class="heatmap-cell" style="background:${color}">
                <div class="heatmap-hour">${hour.toString().padStart(2, "0")}:00</div>
                <div class="heatmap-value">${showVal}</div>
            </div>
        `;
    }).join("");
}

function renderSessionCryptoDaily(rows) {
    const wrap = scEl("scDailyWrap");
    if (!wrap) return;
    if (!Array.isArray(rows) || !rows.length) {
        wrap.innerHTML = '<div class="empty-state"><p>暂无日线数据</p></div>';
        return;
    }

    wrap.innerHTML = `
        <table class="stock-table">
            <thead>
                <tr>
                    <th>日期</th>
                    <th>星期</th>
                    <th>P(up)</th>
                    <th>P(down)</th>
                    <th>q50</th>
                    <th>目标价(q50)</th>
                    <th>波动</th>
                    <th>趋势</th>
                    <th>风险</th>
                    <th>置信度</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map((row) => `
                    <tr>
                        <td>${String(row.date_bj || "-")}</td>
                        <td>${String(row.day_of_week || "-")}</td>
                        <td>${scFmtPct(row.p_up)}</td>
                        <td>${scFmtPct(row.p_down)}</td>
                        <td class="${(scNum(row.q50_change_pct) || 0) >= 0 ? "positive" : "negative"}">${scFmtSignedPct(row.q50_change_pct)}</td>
                        <td>${scFmtPrice(row.target_price_q50)}</td>
                        <td>${scFmtPct(row.volatility_score)}</td>
                        <td>${scTrendText(row.trend_label)}</td>
                        <td>${scRiskText(row.risk_level)}</td>
                        <td>${scFmtPct(row.confidence_score)}</td>
                    </tr>
                `).join("")}
            </tbody>
        </table>
    `;
}

function renderRankTable(containerId, title, rows, side) {
    const wrap = scEl(containerId);
    if (!wrap) return;
    if (!Array.isArray(rows) || !rows.length) {
        wrap.innerHTML = `<div class="empty-state"><p>${title}: 无数据</p></div>`;
        return;
    }

    const keyCol = rows[0].date_bj ? "date_bj" : "hour_label";
    const edgeCol = side === "down" ? "edge_score_short" : "edge_score";
    const riskCol = side === "down" ? "edge_risk_short" : "edge_risk";

    wrap.innerHTML = `
        <h4 class="table-title">${title}</h4>
        <table class="stock-table compact-table">
            <thead>
                <tr>
                    <th>${keyCol === "date_bj" ? "日期" : "小时"}</th>
                    <th>P(up)</th>
                    <th>q50</th>
                    <th>Edge</th>
                    <th>Edge/Risk</th>
                    <th>置信度</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map((row) => `
                    <tr>
                        <td>${String(row[keyCol] || "-")}</td>
                        <td>${scFmtPct(row.p_up)}</td>
                        <td class="${(scNum(row.q50_change_pct) || 0) >= 0 ? "positive" : "negative"}">${scFmtSignedPct(row.q50_change_pct)}</td>
                        <td class="${(scNum(row[edgeCol]) || 0) >= 0 ? "positive" : "negative"}">${scFmtSignedPct(row[edgeCol])}</td>
                        <td class="${(scNum(row[riskCol]) || 0) >= 0 ? "positive" : "negative"}">${scFmtNum(row[riskCol], 3)}</td>
                        <td>${scFmtPct(row.confidence_score)}</td>
                    </tr>
                `).join("")}
            </tbody>
        </table>
    `;
}

function renderSessionCryptoTop(top) {
    renderRankTable("scTopHourlyUp", "小时级：最可能上涨", top?.hourly?.up || [], "up");
    renderRankTable("scTopHourlyDown", "小时级：最可能下跌", top?.hourly?.down || [], "down");
    renderRankTable("scTopHourlyVol", "小时级：最可能大波动", top?.hourly?.vol || [], "vol");
    renderRankTable("scTopDailyUp", "日线级：最可能上涨", top?.daily?.up || [], "up");
    renderRankTable("scTopDailyDown", "日线级：最可能下跌", top?.daily?.down || [], "down");
    renderRankTable("scTopDailyVol", "日线级：最可能大波动", top?.daily?.vol || [], "vol");
}

function renderSessionCryptoNotes(notes) {
    const ul = scEl("scNotes");
    if (!ul) return;
    if (!Array.isArray(notes) || !notes.length) {
        ul.innerHTML = "";
        return;
    }
    ul.innerHTML = notes.map((x) => `<li>${String(x)}</li>`).join("");
}

function renderSessionCryptoSimStats(rows) {
    const panel = scEl("scSimStats");
    if (!panel) return;
    if (!Array.isArray(rows) || !rows.length) {
        panel.innerHTML = '<div class="sim-side-empty">暂无模拟路径数据</div>';
        return;
    }

    const first = rows[0] || {};
    const last = rows[rows.length - 1] || {};
    const keys = ["q10", "q50", "q90"];

    const pctDelta = (fromVal, toVal) => {
        const from = scNum(fromVal);
        const to = scNum(toVal);
        if (from === null || to === null || Math.abs(from) < 1e-12) return null;
        return (to - from) / from;
    };

    const summaryCards = keys.map((k) => {
        const nowVal = scNum(first[k]);
        const endVal = scNum(last[k]);
        const delta = pctDelta(nowVal, endVal);
        const cls = (scNum(delta) || 0) >= 0 ? "positive" : "negative";
        return `
            <div class="sim-summary-card">
                <span class="label"><span class="sim-tag ${k}">${k}</span> 终点</span>
                <span class="value">${scFmtPrice(endVal)}</span>
                <span class="value ${cls}">${scFmtSignedPct(delta)}</span>
            </div>
        `;
    }).join("");

    const rangeNow = (() => {
        const q10 = scNum(first.q10);
        const q90 = scNum(first.q90);
        if (q10 === null || q90 === null) return null;
        return q90 - q10;
    })();
    const rangeEnd = (() => {
        const q10 = scNum(last.q10);
        const q90 = scNum(last.q90);
        if (q10 === null || q90 === null) return null;
        return q90 - q10;
    })();
    const rangeDelta = pctDelta(rangeNow, rangeEnd);

    const rowsHtml = rows.map((r) => `
        <tr>
            <td>${String(r.label || "-")}</td>
            <td>${scFmtPrice(r.q10)}</td>
            <td>${scFmtPrice(r.q50)}</td>
            <td>${scFmtPrice(r.q90)}</td>
        </tr>
    `).join("");

    panel.innerHTML = `
        <div class="sim-side-title">预测数值（q10 / q50 / q90）</div>
        <div class="sim-summary-grid">
            ${summaryCards}
        </div>
        <div class="sim-summary-card">
            <span class="label">价格带宽（q90 - q10）</span>
            <span class="value">${scFmtPrice(rangeEnd)}</span>
            <span class="value ${(scNum(rangeDelta) || 0) >= 0 ? "positive" : "negative"}">${scFmtSignedPct(rangeDelta)}</span>
        </div>
        <div class="sim-mini-table-wrap">
            <table class="sim-mini-table">
                <thead>
                    <tr>
                        <th>时间</th>
                        <th><span class="sim-tag q10">q10</span></th>
                        <th><span class="sim-tag q50">q50</span></th>
                        <th><span class="sim-tag q90">q90</span></th>
                    </tr>
                </thead>
                <tbody>
                    ${rowsHtml}
                </tbody>
            </table>
        </div>
    `;
}

function renderSessionCryptoSim(pathRows) {
    const rows = Array.isArray(pathRows) ? pathRows : [];
    renderSessionCryptoSimStats(rows);
    const canvas = scEl("scSimChart");
    if (!canvas || typeof Chart === "undefined") return;
    if (!rows.length) return;

    const labels = rows.map((r) => String(r.label || "-"));
    const q10 = rows.map((r) => scNum(r.q10));
    const q50 = rows.map((r) => scNum(r.q50));
    const q90 = rows.map((r) => scNum(r.q90));
    const q10Pos = q10.filter((v) => Number.isFinite(v) && v > 0);
    const q90Pos = q90.filter((v) => Number.isFinite(v) && v > 0);
    const minQ10 = q10Pos.length ? Math.min(...q10Pos) : null;
    const maxQ90 = q90Pos.length ? Math.max(...q90Pos) : null;
    const spreadRatio = (minQ10 && maxQ90) ? (maxQ90 / minQ10) : 1;
    const useSplitAxis = Number.isFinite(spreadRatio) && spreadRatio > 30;

    const datasets = [
        {
            label: "q10",
            data: q10,
            borderColor: "rgba(255,107,107,0.92)",
            backgroundColor: "transparent",
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2.0,
            yAxisID: "y",
        },
        {
            label: "q50",
            data: q50,
            borderColor: "rgba(0,212,170,0.98)",
            backgroundColor: "rgba(0,212,170,0.10)",
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2.6,
            fill: true,
            yAxisID: "y",
        },
        {
            label: "q90",
            data: q90,
            borderColor: "rgba(212,175,55,0.98)",
            backgroundColor: "transparent",
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2.0,
            yAxisID: useSplitAxis ? "y1" : "y",
        },
    ];

    if (scState.simChart) {
        scState.simChart.data.labels = labels;
        scState.simChart.data.datasets = datasets;
        scState.simChart.options.scales.y1.display = useSplitAxis;
        scState.simChart.options.plugins.subtitle = {
            display: useSplitAxis,
            text: "q90 波动显著放大，已分离右轴显示（避免 q10/q50 被压扁）",
            color: "#94a3b8",
        };
        scState.simChart.update();
        return;
    }

    scState.simChart = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
            labels,
            datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "#cbd5e1" },
                },
                subtitle: {
                    display: useSplitAxis,
                    text: "q90 波动显著放大，已分离右轴显示（避免 q10/q50 被压扁）",
                    color: "#94a3b8",
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${scFmtPrice(ctx.raw)}`,
                    },
                },
            },
            scales: {
                x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
                y: {
                    ticks: { color: "#94a3b8" },
                    grid: { color: "rgba(255,255,255,0.05)" },
                },
                y1: {
                    display: useSplitAxis,
                    position: "right",
                    ticks: { color: "#d4af37" },
                    grid: { drawOnChartArea: false },
                },
            },
        },
    });
}

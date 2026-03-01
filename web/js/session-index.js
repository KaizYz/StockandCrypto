// ========================================
// Session Index Page (Web parity with dashboard/app.py)
// ========================================

document.addEventListener("DOMContentLoaded", () => {
    initSessionIndexPage().catch((err) => {
        console.error(err);
        showToast(`加载失败: ${err.message || "未知错误"}`, "error");
    });
});

const siState = {
    payload: null,
    metric: "p_up",
    simChart: null,
    controlsReady: false,
    syncingControls: false,
    refreshTimer: null,
    indexOptionsMap: new Map(),
    currency: "$",
};

function siEl(id) {
    return document.getElementById(id);
}

function siNum(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function siAsProb(value) {
    const n = siNum(value);
    if (n === null) return null;
    if (Math.abs(n) > 1) return n / 100;
    return n;
}

function siAsPct(value) {
    const n = siNum(value);
    if (n === null) return null;
    return Math.abs(n) <= 1 ? n * 100 : n;
}

function siFmtPct(value, digits = 2) {
    const p = siAsPct(value);
    if (p === null) return "-";
    return `${p.toFixed(digits)}%`;
}

function siFmtSignedPct(value, digits = 2) {
    const p = siAsPct(value);
    if (p === null) return "-";
    const sign = p > 0 ? "+" : "";
    return `${sign}${p.toFixed(digits)}%`;
}

function siFmtNum(value, digits = 2) {
    const n = siNum(value);
    if (n === null) return "-";
    return n.toFixed(digits);
}

function siFmtPrice(value, digits = 2) {
    const n = siNum(value);
    if (n === null) return "-";
    return `${siState.currency}${n.toLocaleString(undefined, {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
    })}`;
}

function siTrendText(key) {
    const txt = String(key || "").toLowerCase();
    if (txt.includes("bull")) return "看涨";
    if (txt.includes("bear")) return "看跌";
    if (txt.includes("up")) return "看涨";
    if (txt.includes("down")) return "看跌";
    return "震荡";
}

function siRiskText(key) {
    const txt = String(key || "").toLowerCase();
    if (txt === "low") return "低";
    if (txt === "medium") return "中";
    if (txt === "high" || txt === "extreme") return "高";
    return txt || "-";
}

function siActionClass(action) {
    const key = String(action || "").toUpperCase();
    if (key === "LONG") return "long";
    if (key === "SHORT") return "short";
    return "flat";
}

function siSelectOptions(selectEl, options, selected, labelFn = null) {
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
        if (typeof opt === "object" && opt && opt.market) {
            option.dataset.market = String(opt.market);
        }
        if (value === current) option.selected = true;
        selectEl.appendChild(option);
    });
}

function siMarketByIndex(indexKey) {
    const key = String(indexKey || "").trim().toLowerCase();
    const hit = siState.indexOptionsMap.get(key);
    const market = String(hit?.market || "");
    return market.startsWith("cn") ? "cn" : "us";
}

function siBuildParams() {
    const indexKey = String(siEl("siIndex")?.value || "").trim().toLowerCase();
    return {
        market: siMarketByIndex(indexKey),
        index_key: indexKey,
        mode: siEl("siMode")?.value,
        horizon_hours: siEl("siHorizon")?.value,
        lookforward_days: siEl("siLookforward")?.value,
        risk_profile: siEl("siRisk")?.value,
        rank_key: siEl("siRank")?.value,
        top_n: siEl("siTopN")?.value,
        cost_bps: siEl("siCost")?.value,
    };
}

async function initSessionIndexPage() {
    bindSessionIndexControls();
    bindSessionIndexHeatmapTabs();
    await loadSessionIndexData(true);
    if (siState.refreshTimer) clearInterval(siState.refreshTimer);
    siState.refreshTimer = setInterval(() => loadSessionIndexData(false), 5 * 60 * 1000);
}

function bindSessionIndexControls() {
    const ids = [
        "siIndex",
        "siMode",
        "siHorizon",
        "siLookforward",
        "siRisk",
        "siRank",
        "siTopN",
        "siCost",
    ];
    ids.forEach((id) => {
        const node = siEl(id);
        if (!node) return;
        node.addEventListener("change", () => {
            if (siState.syncingControls) return;
            loadSessionIndexData(true).catch((err) => {
                console.error(err);
                showToast(`刷新失败: ${err.message || "未知错误"}`, "error");
            });
        });
    });

    siEl("siRefreshBtn")?.addEventListener("click", () => {
        loadSessionIndexData(true).catch((err) => {
            console.error(err);
            showToast(`刷新失败: ${err.message || "未知错误"}`, "error");
        });
    });

    siEl("siCompare")?.addEventListener("change", () => {
        if (!siState.payload) return;
        renderSessionIndexCompare(siState.payload.compare || {});
    });
}

function bindSessionIndexHeatmapTabs() {
    const tabs = document.querySelectorAll("#siHeatmapTabs .heatmap-tab");
    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const metric = String(tab.dataset.metric || "p_up");
            siState.metric = metric;
            tabs.forEach((x) => x.classList.remove("active"));
            tab.classList.add("active");
            renderSessionIndexHeatmap(metric);
        });
    });
}

async function loadSessionIndexData(showLoading) {
    if (showLoading) siEl("siMetaLine").textContent = "加载中...";
    const payload = await api.session.getIndex(siBuildParams());
    if (!payload || payload.ok === false) {
        throw new Error(payload?.error || "session_index_api_failed");
    }
    siState.payload = payload;
    syncSessionIndexControls(payload.controls || {});
    renderSessionIndexPayload(payload);
    if (showLoading) showToast("数据已更新", "success", 1200);
}

function syncSessionIndexControls(controls) {
    if (!controls || typeof controls !== "object") return;
    const selected = controls.selected || {};
    siState.syncingControls = true;

    siState.indexOptionsMap = new Map(
        (controls.index_options || []).map((x) => [String(x?.key || "").toLowerCase(), x])
    );

    siSelectOptions(siEl("siIndex"), controls.index_options || [], selected.index_key, (opt) => String(opt.label || opt.key || ""));
    siSelectOptions(siEl("siMode"), controls.mode_options || [], selected.mode);
    siSelectOptions(siEl("siHorizon"), controls.horizon_options || [], selected.horizon_hours);
    siSelectOptions(siEl("siRisk"), controls.risk_profile_options || [], selected.risk_profile);

    const rankOptions = Object.entries(controls.rank_options || {}).map(([value, label]) => ({ value, label }));
    siSelectOptions(siEl("siRank"), rankOptions, selected.rank_key, (opt) => String(opt.label || opt.value));

    if (siEl("siLookforward")) siEl("siLookforward").value = String(selected.lookforward_days ?? 14);
    if (siEl("siTopN")) siEl("siTopN").value = String(selected.top_n ?? 5);
    if (siEl("siCost")) siEl("siCost").value = String(selected.cost_bps ?? 8);

    siState.syncingControls = false;
    siState.controlsReady = true;
}

function renderSessionIndexPayload(payload) {
    renderSessionIndexMeta(payload.meta || {});
    renderSessionIndexDecision(payload.decision || {});
    renderSessionIndexSessions(payload.session_cards || payload.sessions || []);
    renderSessionIndexCompare(payload.compare || {});
    renderSessionIndexHeatmap(siState.metric);
    renderSessionIndexDaily(payload.daily || []);
    renderSessionIndexTop(payload.top || {});
    renderSessionIndexNotes(payload.notes || []);
    renderSessionIndexSim(payload.sim_path || []);
}

function renderSessionIndexMeta(meta) {
    const market = String(meta.market || "").toLowerCase();
    siState.currency = market.includes("cn") ? "¥" : "$";
    const indexName = String(meta.index_name_zh || meta.index_name_en || meta.index_key || "-");
    const activeSession = String(meta.active_session || "-");
    const line = [
        `${indexName} (${String(meta.symbol || "-")})`,
        `最新价格: ${siFmtPrice(meta.current_price)}`,
        `更新时间: ${String(meta.data_updated_at_bj || "-")}`,
        `模式/周期: ${String(meta.mode_actual || "-")} / ${String(meta.horizon_hours || "-")}h`,
        `有效交易时段: ${activeSession}`,
    ].join(" | ");
    siEl("siMetaLine").textContent = line;

    const info = [
        `index_key: ${String(meta.index_key || "-")}`,
        `index_name_zh: ${String(meta.index_name_zh || "-")}`,
        `index_name_en: ${String(meta.index_name_en || "-")}`,
        `symbol: ${String(meta.symbol || "-")}`,
        `market: ${String(meta.market || "-")}`,
        `mode_actual: ${String(meta.mode_actual || "-")}`,
        `horizon_hours: ${String(meta.horizon_hours || "-")}`,
        `active_session: ${String(meta.active_session || "-")}`,
        `forecast_generated_at_bj: ${String(meta.forecast_generated_at_bj || "-")}`,
        `data_updated_at_bj: ${String(meta.data_updated_at_bj || "-")}`,
        `model_version: ${String(meta.model_version || "-")}`,
        `data_source_actual: ${String(meta.data_source_actual || "-")}`,
        `data_version: ${String(meta.data_version || "-")}`,
    ].join("\n");
    siEl("siDataInfo").textContent = info;
}

function renderSessionIndexDecision(decision) {
    const plan = decision.plan || {};
    const consensus = decision.consensus || {};
    const action = String(plan.action || "WAIT").toUpperCase();
    const actionCn = String(plan.action_cn || "观望");
    const statusText = String(plan.trade_status_text || "-");
    const statusNote = String(plan.trade_status_note || "-");

    siEl("siDecisionCard").innerHTML = `
        <div class="decision-top">
            <div>
                <div class="decision-k">一眼决策卡（交易时段）</div>
                <div class="decision-action ${siActionClass(action)}">${action} / ${actionCn}</div>
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
            <div><span>入场价</span><strong>${siFmtPrice(plan.entry)}</strong></div>
            <div><span>止损 SL</span><strong>${siFmtPrice(plan.stop_loss)}</strong></div>
            <div><span>止盈 TP1</span><strong>${siFmtPrice(plan.take_profit)}</strong></div>
            <div><span>止盈 TP2</span><strong>${siFmtPrice(plan.take_profit_2)}</strong></div>
            <div><span>RR / 净Edge</span><strong>${siFmtNum(plan.rr, 2)} / ${siFmtSignedPct(plan.plan_side === "SHORT" ? plan.edge_short : plan.edge_long)}</strong></div>
        </div>
    `;

    const q50Signed = String(plan.plan_side || "LONG").toUpperCase() === "SHORT"
        ? -(siNum(plan.q50) || 0)
        : (siNum(plan.q50) || 0);
    const riskPct = (() => {
        const entry = siNum(plan.entry);
        const sl = siNum(plan.stop_loss);
        if (!entry || !sl) return null;
        return Math.abs((entry - sl) / entry);
    })();

    siEl("siTradePlanMetrics").innerHTML = `
        <div class="decision-metric"><span>预期收益(q50)</span><strong>${siFmtSignedPct(q50Signed)}</strong></div>
        <div class="decision-metric"><span>风险（到SL）</span><strong>${siFmtPct(riskPct)}</strong></div>
        <div class="decision-metric"><span>RR</span><strong>${siFmtNum(plan.rr, 2)}</strong></div>
        <div class="decision-metric"><span>扣成本后净Edge</span><strong>${siFmtSignedPct(plan.plan_side === "SHORT" ? plan.edge_short : plan.edge_long)}</strong></div>
    `;
}

function renderSessionIndexSessions(rows) {
    const wrap = siEl("siSessionCards");
    if (!wrap) return;
    if (!Array.isArray(rows) || rows.length === 0) {
        wrap.innerHTML = '<div class="empty-state"><p>暂无时段数据</p></div>';
        return;
    }
    wrap.innerHTML = rows.map((row) => {
        const pUp = siAsProb(row.p_up);
        const pDown = siAsProb(row.p_down);
        const hasRich = pUp !== null || siNum(row.q50_change_pct) !== null || siNum(row.target_price_q50) !== null;
        const direction = hasRich
            ? (pUp === null ? "-" : (pUp >= 0.53 ? "看涨" : (pUp <= 0.47 ? "看跌" : "震荡")))
            : siTrendText(row.direction);
        const volatilityLegacy = siAsPct(row.volatility);
        return `
            <div class="session-card">
                <div class="session-header">
                    <div class="session-info">
                        <h3 class="session-name">${String(row.session_name_cn || row.title || row.session_name || "-")}</h3>
                        <span class="session-time">${String(row.session_hours || row.time || "-")}</span>
                    </div>
                </div>
                <div class="session-content">
                    <div class="detail-item"><span class="detail-label">P(up)</span><span class="detail-value">${hasRich ? siFmtPct(pUp) : "-"}</span></div>
                    <div class="detail-item"><span class="detail-label">P(down)</span><span class="detail-value">${hasRich ? siFmtPct(pDown) : "-"}</span></div>
                    <div class="detail-item"><span class="detail-label">q50</span><span class="detail-value">${hasRich ? siFmtSignedPct(row.q50_change_pct) : "-"}</span></div>
                    <div class="detail-item"><span class="detail-label">目标价(q50)</span><span class="detail-value">${hasRich ? siFmtPrice(row.target_price_q50) : "-"}</span></div>
                    <div class="detail-item"><span class="detail-label">方向</span><span class="detail-value">${direction}</span></div>
                    <div class="detail-item"><span class="detail-label">风险 / 置信度</span><span class="detail-value">${hasRich ? `${siRiskText(row.risk_level)} / ${siFmtPct(row.confidence_score)}` : `波动 ${volatilityLegacy === null ? "-" : `${volatilityLegacy.toFixed(1)}%`}`}</span></div>
                    <div class="detail-item"><span class="detail-label">策略动作</span><span class="detail-value">${String(row.policy_action || row.direction || "-")}</span></div>
                </div>
            </div>
        `;
    }).join("");
}

function renderSessionIndexCompare(compare) {
    const wrap = siEl("siCompareWrap");
    if (!wrap) return;
    const show = siEl("siCompare")?.checked ?? true;
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
                            <td>${siFmtPct(mP)}</td>
                            <td>${siFmtPct(cP)}</td>
                            <td class="${(siNum(row.delta_p_up) || 0) >= 0 ? "positive" : "negative"}">${siFmtSignedPct(row.delta_p_up)}</td>
                            <td>${siFmtSignedPct(mQ)}</td>
                            <td>${siFmtSignedPct(cQ)}</td>
                            <td class="${(siNum(row.delta_q50) || 0) >= 0 ? "positive" : "negative"}">${siFmtSignedPct(row.delta_q50)}</td>
                        </tr>
                    `;
                }).join("")}
            </tbody>
        </table>
    `;
}

function renderSessionIndexHeatmap(metric) {
    const grid = siEl("siHeatmapGrid");
    if (!grid) return;
    const rows = Array.isArray(siState.payload?.hourly) ? siState.payload.hourly : [];
    if (!rows.length) {
        grid.innerHTML = '<div class="empty-state"><p>暂无小时级数据</p></div>';
        return;
    }

    const values = rows
        .map((row) => {
            if (siNum(row?.is_trading_hour) === 0) return null;
            const raw = row?.[metric];
            return siAsPct(raw);
        })
        .filter((v) => siNum(v) !== null);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;
    const span = Math.max(1e-9, max - min);

    grid.innerHTML = rows.map((row) => {
        const hour = Number(row.hour_bj ?? 0) % 24;
        const isTrading = siNum(row?.is_trading_hour) !== 0;
        const val = isTrading ? siAsPct(row?.[metric]) : null;
        const showVal = val === null ? "—" : `${val.toFixed(2)}%`;
        let color = "rgba(148, 163, 184, 0.08)";
        if (val !== null) {
            const ratio = (val - min) / span;
            const opacity = 0.1 + ratio * 0.5;
            color = `rgba(0, 212, 170, ${opacity.toFixed(3)})`;
            if (metric === "p_down") color = `rgba(255, 107, 107, ${opacity.toFixed(3)})`;
            if (metric === "volatility_score") color = `rgba(212, 175, 55, ${opacity.toFixed(3)})`;
        }
        return `
            <div class="heatmap-cell" style="background:${color}">
                <div class="heatmap-hour">${hour.toString().padStart(2, "0")}:00</div>
                <div class="heatmap-value">${showVal}</div>
            </div>
        `;
    }).join("");
}

function renderSessionIndexDaily(rows) {
    const wrap = siEl("siDailyWrap");
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
                        <td>${siFmtPct(row.p_up)}</td>
                        <td>${siFmtPct(row.p_down)}</td>
                        <td class="${(siNum(row.q50_change_pct) || 0) >= 0 ? "positive" : "negative"}">${siFmtSignedPct(row.q50_change_pct)}</td>
                        <td>${siFmtPrice(row.target_price_q50)}</td>
                        <td>${siFmtPct(row.volatility_score)}</td>
                        <td>${siTrendText(row.trend_label)}</td>
                        <td>${siRiskText(row.risk_level)}</td>
                        <td>${siFmtPct(row.confidence_score)}</td>
                    </tr>
                `).join("")}
            </tbody>
        </table>
    `;
}

function renderRankTable(containerId, title, rows, side) {
    const wrap = siEl(containerId);
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
                        <td>${siFmtPct(row.p_up)}</td>
                        <td class="${(siNum(row.q50_change_pct) || 0) >= 0 ? "positive" : "negative"}">${siFmtSignedPct(row.q50_change_pct)}</td>
                        <td class="${(siNum(row[edgeCol]) || 0) >= 0 ? "positive" : "negative"}">${siFmtSignedPct(row[edgeCol])}</td>
                        <td class="${(siNum(row[riskCol]) || 0) >= 0 ? "positive" : "negative"}">${siFmtNum(row[riskCol], 3)}</td>
                        <td>${siFmtPct(row.confidence_score)}</td>
                    </tr>
                `).join("")}
            </tbody>
        </table>
    `;
}

function renderSessionIndexTop(top) {
    renderRankTable("siTopHourlyUp", "小时级：最可能上涨", top?.hourly?.up || [], "up");
    renderRankTable("siTopHourlyDown", "小时级：最可能下跌", top?.hourly?.down || [], "down");
    renderRankTable("siTopHourlyVol", "小时级：最可能大波动", top?.hourly?.vol || [], "vol");
    renderRankTable("siTopDailyUp", "日线级：最可能上涨", top?.daily?.up || [], "up");
    renderRankTable("siTopDailyDown", "日线级：最可能下跌", top?.daily?.down || [], "down");
    renderRankTable("siTopDailyVol", "日线级：最可能大波动", top?.daily?.vol || [], "vol");
}

function renderSessionIndexNotes(notes) {
    const ul = siEl("siNotes");
    if (!ul) return;
    if (!Array.isArray(notes) || !notes.length) {
        ul.innerHTML = "";
        return;
    }
    ul.innerHTML = notes.map((x) => `<li>${String(x)}</li>`).join("");
}

function renderSessionIndexSim(pathRows) {
    const canvas = siEl("siSimChart");
    if (!canvas || typeof Chart === "undefined") return;
    const rows = Array.isArray(pathRows) ? pathRows : [];
    if (!rows.length) return;

    const labels = rows.map((r) => String(r.label || "-"));
    const q10 = rows.map((r) => siNum(r.q10));
    const q50 = rows.map((r) => siNum(r.q50));
    const q90 = rows.map((r) => siNum(r.q90));

    if (siState.simChart) {
        siState.simChart.data.labels = labels;
        siState.simChart.data.datasets[0].data = q10;
        siState.simChart.data.datasets[1].data = q50;
        siState.simChart.data.datasets[2].data = q90;
        siState.simChart.update();
        return;
    }

    siState.simChart = new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
            labels,
            datasets: [
                { label: "q10", data: q10, borderColor: "rgba(255,107,107,0.9)", backgroundColor: "transparent", tension: 0.3, pointRadius: 0, borderWidth: 1.8 },
                { label: "q50", data: q50, borderColor: "rgba(0,212,170,0.95)", backgroundColor: "rgba(0,212,170,0.08)", tension: 0.3, pointRadius: 0, borderWidth: 2.4, fill: true },
                { label: "q90", data: q90, borderColor: "rgba(212,175,55,0.95)", backgroundColor: "transparent", tension: 0.3, pointRadius: 0, borderWidth: 1.8 },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: "#cbd5e1" },
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${siFmtPrice(ctx.raw)}`,
                    },
                },
            },
            scales: {
                x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
                y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255,255,255,0.05)" } },
            },
        },
    });
}

// ========================================
// StockandCrypto - Tracking page
// ========================================

document.addEventListener("DOMContentLoaded", () => {
    initTrackingPage();
});

const trackingState = {
    items: [],
    lastParams: {},
};

function debounce(fn, wait = 250) {
    let timer = null;
    return (...args) => {
        if (timer) clearTimeout(timer);
        timer = setTimeout(() => {
            fn(...args);
        }, wait);
    };
}

function initTrackingPage() {
    bindTrackingEvents();
    loadTrackingOverview();
    setInterval(loadTrackingOverview, 30000);
}

function bindTrackingEvents() {
    const ids = [
        "marketFilter",
        "statusFilter",
        "actionFilter",
        "sortBy",
        "sortDesc",
        "trackingQuery",
    ];
    ids.forEach((id) => {
        const el = document.getElementById(id);
        if (!el) return;
        const evt = el.tagName === "INPUT" ? "input" : "change";
        el.addEventListener(evt, debounce(loadTrackingOverview, 250));
    });

    const refreshBtn = document.getElementById("trackingRefreshBtn");
    if (refreshBtn) {
        refreshBtn.addEventListener("click", () => loadTrackingOverview());
    }

    const tableBody = document.getElementById("trackingTableBody");
    if (tableBody) {
        tableBody.addEventListener("click", async (event) => {
            const btn = event.target.closest("[data-track-key]");
            if (!btn) return;
            const key = btn.getAttribute("data-track-key");
            if (!key) return;
            await loadTrackingDetail(key);
        });
    }
}

function currentParams() {
    const market = document.getElementById("marketFilter")?.value || "all";
    const status = document.getElementById("statusFilter")?.value || "all";
    const action = document.getElementById("actionFilter")?.value || "all";
    const sort_by = document.getElementById("sortBy")?.value || "edge_risk";
    const desc = !!document.getElementById("sortDesc")?.checked;
    const q = document.getElementById("trackingQuery")?.value?.trim() || "";
    return {
        market,
        status,
        action,
        sort_by,
        desc,
        q,
        top_n: 5,
        limit: 200,
        cost_bps: 8,
    };
}

async function loadTrackingOverview() {
    const params = currentParams();
    trackingState.lastParams = params;
    try {
        const res = await api.tracking.getOverview(params);
        trackingState.items = res?.items || [];
        renderTrackingMetrics(res?.metrics || {});
        renderTopTable("topLong", res?.top_long || []);
        renderTopTable("topShort", res?.top_short || []);
        renderTopTable("topWatch", res?.top_watch || []);
        renderTrackingTable(trackingState.items);
        const countLabel = document.getElementById("trackingCountLabel");
        if (countLabel) countLabel.textContent = `${res?.count || 0} 条`;
    } catch (error) {
        ErrorHandler.handle(error);
    }
}

function renderTrackingMetrics(metrics) {
    const container = document.getElementById("trackingMetrics");
    if (!container) return;
    const cards = [
        { label: "总候选", value: metrics.total_candidates ?? 0 },
        { label: "可执行", value: metrics.executable_count ?? 0 },
        { label: "观察", value: metrics.watch_count ?? 0 },
        { label: "暂停", value: metrics.paused_count ?? 0 },
        { label: "预测覆盖率", value: `${pct(metrics.prediction_coverage, 1)}` },
        { label: "硬过滤通过率", value: `${pct(metrics.hard_filter_pass_rate, 1)}` },
        { label: "平均缺失率", value: `${pct(metrics.avg_missing_rate, 1)}` },
        { label: "当前筛选数", value: metrics.filtered_count ?? 0 },
    ];
    container.innerHTML = cards
        .map(
            (c) => `
        <div class="metric-card">
            <div class="metric-label">${escapeHtml(c.label)}</div>
            <div class="metric-value">${escapeHtml(String(c.value))}</div>
        </div>`
        )
        .join("");
}

function renderTopTable(containerId, rows) {
    const container = document.getElementById(containerId);
    if (!container) return;
    if (!Array.isArray(rows) || rows.length === 0) {
        container.innerHTML = '<div class="list-empty">暂无数据</div>';
        return;
    }
    const html = `
        <table class="mini-table">
            <thead>
                <tr>
                    <th>标的</th>
                    <th>动作</th>
                    <th>edge</th>
                    <th>confidence</th>
                </tr>
            </thead>
            <tbody>
                ${rows
                    .map(
                        (r) => `
                    <tr>
                        <td>${escapeHtml(r.display_name || r.symbol || "-")}</td>
                        <td>${escapeHtml(r.action || "-")}</td>
                        <td>${signedPct(r.edge_score)}</td>
                        <td>${pct(r.confidence_score, 1)}</td>
                    </tr>`
                    )
                    .join("")}
            </tbody>
        </table>
    `;
    container.innerHTML = html;
}

function renderTrackingTable(items) {
    const tbody = document.getElementById("trackingTableBody");
    if (!tbody) return;
    if (!Array.isArray(items) || items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="12" class="table-empty">暂无数据</td></tr>';
        return;
    }
    tbody.innerHTML = items
        .map(
            (item) => `
        <tr>
            <td>${escapeHtml(item.market_label || item.market || "-")}</td>
            <td>${escapeHtml(item.display_name || item.symbol || "-")}</td>
            <td>${escapeHtml(item.action || "-")}</td>
            <td>${escapeHtml(item.rule_status || "-")}</td>
            <td>${formatPrice(item.current_price)}</td>
            <td>${formatPrice(item.predicted_price)}</td>
            <td>${signedPct(item.predicted_change_pct)}</td>
            <td>${pct(item.confidence_score, 1)}</td>
            <td>${escapeHtml(item.risk_level || "-")}</td>
            <td>${signedPct(item.edge_score)}</td>
            <td>${num(item.edge_risk, 3)}</td>
            <td><button class="btn btn-secondary btn-sm" data-track-key="${escapeHtml(item.track_key || "")}">查看</button></td>
        </tr>`
        )
        .join("");
}

async function loadTrackingDetail(trackKey) {
    const box = document.getElementById("trackingDetail");
    if (!box) return;
    box.innerHTML = '<p class="section-desc">加载中...</p>';
    try {
        const res = await api.tracking.getDetail(trackKey, {
            market: trackingState.lastParams.market,
            cost_bps: trackingState.lastParams.cost_bps,
        });
        const item = res?.item;
        if (!item) {
            box.innerHTML = '<p class="section-desc">未找到详情</p>';
            return;
        }
        box.innerHTML = `
            <div class="detail-grid">
                <div><span class="k">标的</span><span class="v">${escapeHtml(item.display_name || item.symbol || "-")}</span></div>
                <div><span class="k">市场</span><span class="v">${escapeHtml(item.market_label || item.market || "-")}</span></div>
                <div><span class="k">动作</span><span class="v">${escapeHtml(item.action || "-")}</span></div>
                <div><span class="k">状态</span><span class="v">${escapeHtml(item.rule_status || "-")}</span></div>
                <div><span class="k">当前价</span><span class="v">${formatPrice(item.current_price)}</span></div>
                <div><span class="k">预测价</span><span class="v">${formatPrice(item.predicted_price)}</span></div>
                <div><span class="k">预测涨跌</span><span class="v">${signedPct(item.predicted_change_pct)}</span></div>
                <div><span class="k">置信度</span><span class="v">${pct(item.confidence_score, 1)}</span></div>
                <div><span class="k">风险</span><span class="v">${escapeHtml(item.risk_level || "-")}</span></div>
                <div><span class="k">edge</span><span class="v">${signedPct(item.edge_score)}</span></div>
                <div><span class="k">edge_risk</span><span class="v">${num(item.edge_risk, 3)}</span></div>
                <div><span class="k">仓位</span><span class="v">${pct(item.position_size, 1)}</span></div>
                <div style="grid-column:1 / -1;"><span class="k">原因</span><span class="v">${escapeHtml(item.reason || "-")}</span></div>
                <div style="grid-column:1 / -1;"><span class="k">告警</span><span class="v">${escapeHtml(item.alerts || "-")}</span></div>
                <div style="grid-column:1 / -1;"><span class="k">更新时间</span><span class="v">${escapeHtml(item.timestamp_utc || "-")}</span></div>
            </div>
        `;
    } catch (error) {
        box.innerHTML = '<p class="section-desc">加载详情失败</p>';
        ErrorHandler.handle(error);
    }
}

function pct(v, digits = 1) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return `${(n * 100).toFixed(digits)}%`;
}

function signedPct(v, digits = 2) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    const pctVal = n * 100;
    const sign = pctVal >= 0 ? "+" : "";
    return `${sign}${pctVal.toFixed(digits)}%`;
}

function num(v, digits = 2) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return n.toFixed(digits);
}

function formatPrice(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    if (n >= 1000) return n.toLocaleString("en-US", { maximumFractionDigits: 2 });
    if (n >= 1) return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return n.toFixed(6);
}

function escapeHtml(raw) {
    return String(raw || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

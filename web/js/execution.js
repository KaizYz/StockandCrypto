// ========================================
// StockandCrypto - Execution page
// ========================================

document.addEventListener("DOMContentLoaded", () => {
    initExecutionPage();
});

const executionState = {
    activeTab: "decision_logs",
    payload: null,
};

function initExecutionPage() {
    bindExecutionEvents();
    loadExecutionOverview();
    setInterval(loadExecutionOverview, 30000);
}

function bindExecutionEvents() {
    const refreshBtn = document.getElementById("executionRefreshBtn");
    const clearBtn = document.getElementById("executionClearBtn");
    const tabRow = document.getElementById("executionTabRow");

    if (refreshBtn) {
        refreshBtn.addEventListener("click", () => loadExecutionOverview());
    }
    if (clearBtn) {
        clearBtn.addEventListener("click", async () => {
            const ok = window.confirm("确认清空 execution 日志？");
            if (!ok) return;
            try {
                await api.execution.clearLogs();
                showToast("日志已清空", "success");
                await loadExecutionOverview();
            } catch (error) {
                ErrorHandler.handle(error);
            }
        });
    }
    if (tabRow) {
        tabRow.addEventListener("click", (event) => {
            const btn = event.target.closest("[data-tab]");
            if (!btn) return;
            executionState.activeTab = btn.getAttribute("data-tab") || "decision_logs";
            syncTabButtons();
            renderActiveExecutionTable();
        });
    }
}

function syncTabButtons() {
    const buttons = document.querySelectorAll("#executionTabRow .tab-btn");
    buttons.forEach((btn) => {
        const key = btn.getAttribute("data-tab");
        if (key === executionState.activeTab) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }
    });
}

async function loadExecutionOverview() {
    try {
        const res = await api.execution.getOverview({ limit: 300, log_limit: 200 });
        executionState.payload = res || null;
        renderExecutionMetrics(res?.stats || {});
        syncTabButtons();
        renderActiveExecutionTable();
    } catch (error) {
        ErrorHandler.handle(error);
    }
}

function renderExecutionMetrics(stats) {
    const container = document.getElementById("executionMetrics");
    if (!container) return;
    const cards = [
        { label: "Open Positions", value: stats.open_positions ?? 0 },
        { label: "Closed Positions", value: stats.closed_positions ?? 0 },
        { label: "Win Rate", value: percentValue(stats.win_rate) },
        { label: "Avg Net PnL", value: percentValue(stats.avg_net_pnl_pct) },
        { label: "Total Net PnL", value: percentValue(stats.total_net_pnl_pct) },
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

function renderActiveExecutionTable() {
    const head = document.getElementById("executionTableHead");
    const body = document.getElementById("executionTableBody");
    if (!head || !body) return;

    const payload = executionState.payload || {};
    const rows = getRowsForTab(payload, executionState.activeTab);
    if (!Array.isArray(rows) || rows.length === 0) {
        head.innerHTML = "";
        body.innerHTML = '<tr><td class="table-empty">暂无数据</td></tr>';
        return;
    }

    const columns = pickColumns(rows);
    head.innerHTML = `<tr>${columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}</tr>`;
    body.innerHTML = rows
        .map(
            (row) => `
        <tr>
            ${columns.map((col) => `<td>${escapeHtml(formatCell(row[col]))}</td>`).join("")}
        </tr>`
        )
        .join("");
}

function getRowsForTab(payload, tab) {
    if (tab === "positions") {
        const open = Array.isArray(payload.open_positions) ? payload.open_positions : [];
        const closed = Array.isArray(payload.closed_positions) ? payload.closed_positions : [];
        return [...open, ...closed];
    }
    return Array.isArray(payload[tab]) ? payload[tab] : [];
}

function pickColumns(rows) {
    const preferred = [
        "decision_id",
        "order_id",
        "fill_id",
        "position_id",
        "market",
        "symbol",
        "action",
        "side",
        "status",
        "qty",
        "price",
        "entry_price",
        "exit_price",
        "entry",
        "sl",
        "tp1",
        "rr",
        "expected_edge_pct",
        "net_pnl_pct",
        "created_at_utc",
        "fill_time_utc",
        "entry_time_utc",
        "exit_time_utc",
        "date_utc",
        "timestamp_utc",
        "reason_code",
        "reasons",
    ];
    const allKeys = new Set();
    rows.forEach((r) => Object.keys(r || {}).forEach((k) => allKeys.add(k)));
    const selected = preferred.filter((k) => allKeys.has(k));
    if (selected.length > 0) return selected;
    return Object.keys(rows[0] || {}).slice(0, 16);
}

function formatCell(value) {
    if (value === null || value === undefined) return "-";
    if (typeof value === "number") {
        if (Math.abs(value) >= 1000) return value.toLocaleString("en-US");
        return String(value);
    }
    if (typeof value === "object") {
        try {
            return JSON.stringify(value);
        } catch (_) {
            return String(value);
        }
    }
    return String(value);
}

function percentValue(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    const sign = n > 0 ? "+" : "";
    return `${sign}${n.toFixed(2)}%`;
}

function escapeHtml(raw) {
    return String(raw || "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

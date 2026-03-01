// ========================================
// StockandCrypto - CN Equity 页面脚本
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    initCNEquityPage();
});

let cnIndexCharts = {};
let cnTrendChart = null;
let cnSignalChart = null;
let cnStocksData = [];
let currentStock = '600519.SH';
let cnSignalPeriod = '1D';

const CN_STOCK_ID_MAP = {
    '600519': 'moutai',
    '300750': 'ningde',
    '002594': 'byd',
    '601318': 'pingan',
    '600036': 'cmb',
};

function _safeNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function _cnSymbolKey(symbol) {
    const s = String(symbol || '').toUpperCase().trim();
    if (s.includes('.')) return s.split('.')[0];
    return s;
}

function _formatChartLabel(ts, interval = 'daily') {
    if (!ts) return '';
    const dt = new Date(ts);
    if (Number.isNaN(dt.getTime())) return '';
    if (interval === 'hourly') {
        return `${String(dt.getHours()).padStart(2, '0')}:00`;
    }
    return dt.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' });
}

function _buildSeriesFromBars(bars, interval) {
    const rows = Array.isArray(bars) ? bars : [];
    const cleaned = rows
        .map((row) => ({
            label: _formatChartLabel(row.timestamp, interval),
            close: _safeNumber(row.close),
        }))
        .filter((row) => row.label && row.close !== null);
    return {
        labels: cleaned.map((x) => x.label),
        values: cleaned.map((x) => x.close),
    };
}

// ========================================
// 初始化
// ========================================
async function initCNEquityPage() {
    // 显示加载状态
    showLoadingState();
    try {
        // 加载市场数据
        await loadMarketData();

        // 初始化图表
        initCharts();

        // 加载预测数据
        await loadPredictions();

        // 加载当前选择的股票信号
        await loadSignalData();

        // 加载股票列表
        await loadSymbols();

        // 启动实时更新
        startRealTimeUpdates();

        // 绑定事件
        bindEvents();
    } catch (error) {
        console.error('Failed to initialize CN equity page:', error);
        if (typeof showToast === 'function') {
            showToast(`页面初始化失败: ${error.message || '未知错误'}`, 'error');
        }
    } finally {
        // 隐藏加载状态
        hideLoadingState();
    }
}

// ========================================
// 显示/隐藏加载状态
// ========================================
function showLoadingState() {
    const grid = document.querySelector('.index-grid');
    if (grid) {
        grid.classList.add('loading');
    }
    const table = document.getElementById('stockTable');
    if (table) {
        table.classList.add('loading');
    }
    const signalCard = document.getElementById('signalCardMain');
    if (signalCard) {
        signalCard.classList.add('loading');
    }
}

function hideLoadingState() {
    const grid = document.querySelector('.index-grid');
    if (grid) {
        grid.classList.remove('loading');
    }
    const table = document.getElementById('stockTable');
    if (table) {
        table.classList.remove('loading');
    }
    const signalCard = document.getElementById('signalCardMain');
    if (signalCard) {
        signalCard.classList.remove('loading');
    }
}

function showError(message) {
    const container = document.querySelector('.section-padding .container');
    if (container) {
        container.innerHTML = `<div class="error-message"><p>❌ ${message}</p><button onclick="initCNEquityPage()" class="btn btn-secondary">重试</button></div>`;
    }
}

// ========================================
// 加载市场数据
// ========================================
async function loadMarketData() {
    try {
        const data = await api.market.getIndices('cn');
        if (data) {
            updateIndexCards(data);
        } else {
            useMockData();
        }
    } catch (error) {
        console.error('Failed to load market data:', error);
        useMockData();
    }
}

// ========================================
// 加载预测数据
// ========================================
async function loadPredictions() {
    try {
        const predictions = await api.cn.getPredictions();
        if (predictions && predictions.length > 0) {
            updatePredictionsFromAPI(predictions);
        }
    } catch (error) {
        console.error('Failed to load predictions:', error);
    }
}

// ========================================
// 加载单只股票信号数据
// ========================================
async function loadSignalData() {
    try {
        const symbol = currentStock;
        const signal = await api.cn.getPrediction(symbol);
        if (signal) {
            updateSignalCard(signal);
            await updateSignalChart(symbol, cnSignalPeriod);
        } else {
            useMockSignalData(symbol);
        }
    } catch (error) {
        console.error('Failed to load signal data:', error);
        useMockSignalData(currentStock);
    }
}

// ========================================
// 加载股票列表
// ========================================
async function loadSymbols() {
    try {
        const symbols = await api.cn.getSymbols();
        if (symbols && symbols.length > 0) {
            cnStocksData = symbols;
            const selector = document.getElementById('stockSelector');
            if (selector) {
                const previous = currentStock;
                selector.innerHTML = '';
                symbols.forEach((item) => {
                    const symbol = String(item.symbol || '').toUpperCase();
                    if (!symbol) return;
                    const option = document.createElement('option');
                    const name = item.name || symbol;
                    option.value = symbol;
                    option.textContent = `${name} (${symbol})`;
                    selector.appendChild(option);
                });
                if ([...selector.options].some((o) => o.value === previous)) {
                    selector.value = previous;
                    currentStock = previous;
                } else if (selector.options.length > 0) {
                    selector.selectedIndex = 0;
                    currentStock = selector.value;
                }
            }
        }
    } catch (error) {
        console.error('Failed to load symbols:', error);
    }
}

// ========================================
// 使用模拟数据
// ========================================
function useMockData() {
    const mockData = {
        sh: {
            price: 3085.24,
            change: 0.68,
            open: 3065,
            high: 3092,
            low: 3060,
            volume: '3250亿'
        },
        sz: {
            price: 10256.78,
            change: 1.12,
            open: 10200,
            high: 10280,
            low: 10150,
            volume: '4120亿'
        },
        cyb: {
            price: 2085.36,
            change: -0.45,
            open: 2095,
            high: 2100,
            low: 2080,
            volume: '1850亿'
        },
        stocks: [
            { symbol: 'moutai', name: '贵州茅台', code: '600519.SH', price: 1685.00, change: 1.25, target: 1750 },
            { symbol: 'ningde', name: '宁德时代', code: '300750.SZ', price: 168.50, change: -2.35, target: 160 },
            { symbol: 'byd', name: '比亚迪', code: '002594.SZ', price: 245.80, change: 2.15, target: 265 },
            { symbol: 'pingan', name: '中国平安', code: '601318.SH', price: 42.35, change: 0.25, target: 45 },
            { symbol: 'cmb', name: '招商银行', code: '600036.SH', price: 32.85, change: 0.85, target: 35 }
        ]
    };
    updateIndexCards(mockData);
    updateStockTable(mockData);
}

// ========================================
// 使用模拟信号数据
// ========================================
function useMockSignalData(symbol) {
    const stockNames = {
        '600519.SH': { name: '贵州茅台', price: 1685.00 },
        '300750.SZ': { name: '宁德时代', price: 168.50 },
        '002594.SZ': { name: '比亚迪', price: 245.80 },
        '601318.SH': { name: '中国平安', price: 42.35 },
        '600036.SH': { name: '招商银行', price: 32.85 },
        '000858.SZ': { name: '五粮液', price: 145.20 },
        '601012.SH': { name: '隆基绿能', price: 25.80 },
        '000333.SZ': { name: '美的集团', price: 58.50 }
    };

    const stockInfo = stockNames[symbol] || { name: '未知股票', price: 100 };

    // 生成随机信号数据
    const pUp = 0.45 + Math.random() * 0.15;
    const pDown = 1 - pUp;
    const action = pUp > 0.55 ? 'Long' : pUp < 0.45 ? 'Short' : 'Flat';
    const confidence = 55 + Math.random() * 30;

    const mockSignal = {
        symbol: symbol,
        name: stockInfo.name,
        action: action,
        p_up: pUp,
        p_down: pDown,
        confidence: confidence,
        current_price: stockInfo.price * (1 + (Math.random() - 0.5) * 0.02),
        target_price: stockInfo.price * (1 + (pUp - 0.5) * 0.1),
        q50_change_pct: (pUp - 0.5) * 0.1,
        volatility_score: 0.2 + Math.random() * 0.2,
        signal_strength: confidence > 75 ? 'Strong' : confidence > 60 ? 'Medium' : 'Weak',
        signal_strength_pp: Math.abs((pUp - 0.5) * 100),
        trend_label: pUp > 0.55 ? 'bullish' : pUp < 0.45 ? 'bearish' : 'neutral',
        risk_level: confidence < 60 ? 'high' : confidence < 75 ? 'medium' : 'low',
        position_size: action === 'Flat' ? 0 : 0.05 + Math.random() * 0.15
    };

    updateSignalCard(mockSignal);
    updateSignalChart(symbol, cnSignalPeriod).catch(() => {});
}

// ========================================
// 更新信号卡片
// ========================================
function updateSignalCard(signal) {
    // 更新股票名称
    const symbolEl = document.getElementById('signalSymbol');
    if (symbolEl) {
        symbolEl.textContent = signal.name || signal.symbol;
    }

    // 更新股票代码
    const codeEl = document.getElementById('signalCode');
    if (codeEl) {
        codeEl.textContent = signal.symbol || currentStock;
    }

    // 更新时间
    const timeEl = document.getElementById('signalTime');
    if (timeEl) {
        const now = new Date();
        timeEl.textContent = `更新于 ${now.toLocaleTimeString('zh-CN')}`;
    }

    // 更新策略动作
    const actionEl = document.getElementById('signalAction');
    if (actionEl) {
        const action = signal.action || signal.policy_action || 'Flat';
        actionEl.textContent = _formatActionCN(action);
        actionEl.className = 'action-value ' + action.toLowerCase();
    }

    // 更新价格
    const currentPriceEl = document.getElementById('currentPrice');
    const currentPrice = _safeNumber(signal.current_price);
    if (currentPriceEl && currentPrice !== null) {
        currentPriceEl.textContent = `¥${currentPrice.toFixed(2)}`;
    }

    const targetPriceEl = document.getElementById('targetPrice');
    const targetPrice = _safeNumber(signal.target_price ?? signal.target_price_q50);
    if (targetPriceEl && targetPrice !== null) {
        targetPriceEl.textContent = `¥${targetPrice.toFixed(2)}`;
    }

    // 更新概率
    const pUpValue = signal.p_up || 0.5;
    const pDownValue = signal.p_down || 0.5;

    const pUpBar = document.getElementById('pUpBar');
    const pUpValueEl = document.getElementById('pUpValue');
    if (pUpBar) {
        pUpBar.style.width = `${pUpValue * 100}%`;
    }
    if (pUpValueEl) {
        pUpValueEl.textContent = `${(pUpValue * 100).toFixed(1)}%`;
    }

    const pDownBar = document.getElementById('pDownBar');
    const pDownValueEl = document.getElementById('pDownValue');
    if (pDownBar) {
        pDownBar.style.width = `${pDownValue * 100}%`;
    }
    if (pDownValueEl) {
        pDownValueEl.textContent = `${(pDownValue * 100).toFixed(1)}%`;
    }

    // 更新置信度
    let confidence = signal.confidence ?? signal.confidence_score ?? 60;
    if (confidence <= 1) confidence = confidence * 100;
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    if (confidenceBar) {
        confidenceBar.style.width = `${confidence}%`;
    }
    if (confidenceValue) {
        confidenceValue.textContent = `${confidence.toFixed(0)}%`;
    }

    // 更新详情
    const q50ChangeEl = document.getElementById('q50Change');
    if (q50ChangeEl && signal.q50_change_pct !== undefined) {
        const change = signal.q50_change_pct;
        q50ChangeEl.textContent = `${change >= 0 ? '+' : ''}${(change * 100).toFixed(2)}%`;
        q50ChangeEl.className = 'detail-value ' + (change >= 0 ? 'positive' : 'negative');
    }

    const volatilityEl = document.getElementById('volatilityScore');
    if (volatilityEl && signal.volatility_score !== undefined) {
        volatilityEl.textContent = `${(signal.volatility_score * 100).toFixed(1)}%`;
    }

    const strengthEl = document.getElementById('signalStrength');
    if (strengthEl) {
        const strength = signal.signal_strength || 'Weak';
        const strengthPP = signal.signal_strength_pp || 0;
        strengthEl.textContent = `${_signalStrengthTextCN(strength)} (${strengthPP.toFixed(1)}pp)`;
        strengthEl.className = 'detail-value strength-' + strength.toLowerCase();
    }

    const trendEl = document.getElementById('trendLabel');
    if (trendEl) {
        trendEl.textContent = _trendTextCN(signal.trend_label || 'neutral');
        trendEl.className = 'detail-value trend-' + (signal.trend_label || 'neutral');
    }

    const riskEl = document.getElementById('riskLevel');
    if (riskEl) {
        riskEl.textContent = _riskTextCN(signal.risk_level || 'medium');
        riskEl.className = 'detail-value risk-' + (signal.risk_level || 'medium');
    }

    const posSizeEl = document.getElementById('positionSize');
    if (posSizeEl && signal.position_size !== undefined) {
        posSizeEl.textContent = `${(signal.position_size * 100).toFixed(1)}%`;
    }
}

// ========================================
// 格式化函数
// ========================================
function _formatActionCN(action) {
    const actionMap = {
        'Long': '做多',
        'Short': '做空',
        'Flat': '观望',
        'long': '做多',
        'short': '做空',
        'flat': '观望'
    };
    return actionMap[action] || action;
}

function _signalStrengthTextCN(strength) {
    const strengthMap = {
        'Weak': '弱信号',
        'Medium': '中信号',
        'Strong': '强信号',
        'weak': '弱信号',
        'medium': '中信号',
        'strong': '强信号'
    };
    return strengthMap[strength] || strength;
}

function _trendTextCN(trend) {
    const trendMap = {
        'bullish': '看涨',
        'bearish': '看跌',
        'neutral': '震荡',
        'bull': '看涨',
        'bear': '看跌'
    };
    return trendMap[trend] || trend;
}

function _riskTextCN(risk) {
    const riskMap = {
        'low': '低风险',
        'medium': '中风险',
        'high': '高风险'
    };
    return riskMap[risk] || risk;
}

// ========================================
// 更新指数卡片
// ========================================
function updateIndexCards(data) {
    // 上证指数
    if (data.sh) {
        updateIndexCard('sh', data.sh);
    }
    // 深证成指
    if (data.sz) {
        updateIndexCard('sz', data.sz);
    }
    // 创业板指
    if (data.cyb) {
        updateIndexCard('cyb', data.cyb);
    }
}

function updateIndexCard(symbol, data) {
    const priceEl = document.getElementById(`${symbol}Price`);
    const changeEl = document.getElementById(`${symbol}Change`);
    const openEl = document.getElementById(`${symbol}Open`);
    const highEl = document.getElementById(`${symbol}High`);
    const lowEl = document.getElementById(`${symbol}Low`);
    const volumeEl = document.getElementById(`${symbol}Volume`);

    if (priceEl) {
        animatePrice(priceEl, data.price);
    }
    if (changeEl) {
        changeEl.textContent = `${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)}%`;
        changeEl.className = `index-change ${data.change >= 0 ? 'positive' : 'negative'}`;
    }
    if (openEl) openEl.textContent = data.open.toFixed(2);
    if (highEl) highEl.textContent = data.high.toFixed(2);
    if (lowEl) lowEl.textContent = data.low.toFixed(2);
    if (volumeEl) volumeEl.textContent = data.volume;
}

// ========================================
// 更新股票表格
// ========================================
function updateStockTable(data) {
    if (!data.stocks) return;
    data.stocks.forEach(stock => {
        const priceEl = document.getElementById(`${stock.symbol}Price`);
        const changeEl = document.getElementById(`${stock.symbol}Change`);
        const targetEl = document.getElementById(`${stock.symbol}Target`);
        if (priceEl) priceEl.textContent = `¥${stock.price.toFixed(2)}`;
        if (changeEl) {
            changeEl.textContent = `${stock.change >= 0 ? '+' : ''}${stock.change.toFixed(2)}%`;
            changeEl.className = stock.change >= 0 ? 'positive' : 'negative';
        }
        if (targetEl) targetEl.textContent = `¥${stock.target}`;
    });
}

// ========================================
// 从API更新预测数据
// ========================================
function updatePredictionsFromAPI(predictions) {
    predictions.forEach((pred) => {
        const rowId = CN_STOCK_ID_MAP[_cnSymbolKey(pred.symbol)];
        if (!rowId) return;

        const priceEl = document.getElementById(`${rowId}Price`);
        const changeEl = document.getElementById(`${rowId}Change`);
        const targetEl = document.getElementById(`${rowId}Target`);

        const currentPrice = _safeNumber(pred.current_price);
        const changePercent = _safeNumber(pred.change_percent ?? ((pred.predicted_change_pct ?? 0) * 100));
        const targetPrice = _safeNumber(pred.target_price ?? pred.predicted_price);

        if (priceEl && currentPrice !== null) {
            priceEl.textContent = `¥${currentPrice.toFixed(2)}`;
        }
        if (changeEl && changePercent !== null) {
            changeEl.textContent = `${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;
            changeEl.className = changePercent >= 0 ? 'positive' : 'negative';
        }
        if (targetEl && targetPrice !== null) {
            targetEl.textContent = `¥${targetPrice.toFixed(2)}`;
        }
    });
}

// ========================================
// 价格动画
// ========================================
function animatePrice(element, newPrice) {
    const currentPrice = parseFloat(element.textContent.replace(/,/g, '').replace('¥', '')) || 0;
    const diff = newPrice - currentPrice;
    element.style.transition = 'color 0.3s ease';
    if (diff > 0) {
        element.style.color = '#00d4aa';
    } else if (diff < 0) {
        element.style.color = '#ff6b6b';
    }
    element.textContent = newPrice.toFixed(2);
    setTimeout(() => {
        element.style.color = '';
    }, 500);
}

// ========================================
// 初始化图表
// ========================================
function initCharts() {
    Chart.defaults.color = '#8892a0';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.font.family = 'Inter, sans-serif';

    // 创建指数图表
    createIndexChart('shChart', [], '#00d4aa');
    createIndexChart('szChart', [], '#d4af37');
    createIndexChart('cybChart', [], '#ff6b6b');
    createSignalChart();
    updateCNIndexCharts().catch((error) => {
        console.error('Failed to initialize CN index history charts:', error);
    });
}

function createIndexChart(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const values = Array.isArray(data) ? data : [];
    const gradient = ctx.createLinearGradient(0, 0, 0, 80);
    gradient.addColorStop(0, color + '40');
    gradient.addColorStop(1, color + '00');

    cnIndexCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: values.map((_, i) => i),
            datasets: [{
                data: values,
                borderColor: color,
                backgroundColor: gradient,
                fill: true,
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });
}

async function updateCNIndexCharts() {
    const defs = [
        { canvasId: 'shChart', symbol: '000001.SH' },
        { canvasId: 'szChart', symbol: '399001.SZ' },
        { canvasId: 'cybChart', symbol: '399006.SZ' },
    ];

    await Promise.all(defs.map(async ({ canvasId, symbol }) => {
        const chart = cnIndexCharts[canvasId];
        if (!chart) return;
        try {
            const history = await api.market.getHistory(symbol, {
                period: '1D',
                interval: 'hourly',
                limit: 24,
            });
            const series = _buildSeriesFromBars(history?.bars, 'hourly');
            if (series.values.length > 1) {
                chart.data.labels = series.labels;
                chart.data.datasets[0].data = series.values;
                chart.update('none');
            }
        } catch (error) {
            console.error(`Failed to load CN index history for ${symbol}:`, error);
        }
    }));
}

function createSignalChart() {
    const canvas = document.getElementById('signalChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 350);
    gradient.addColorStop(0, 'rgba(0, 212, 170, 0.2)');
    gradient.addColorStop(1, 'rgba(0, 212, 170, 0)');

    cnSignalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: '股价',
                data: [],
                borderColor: '#00d4aa',
                backgroundColor: gradient,
                fill: true,
                tension: 0.35,
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 4,
                pointHoverBackgroundColor: '#00d4aa',
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(10, 15, 26, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#00d4aa',
                    borderColor: 'rgba(0, 212, 170, 0.3)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12,
                    callbacks: {
                        label: (ctx) => `¥${Number(ctx.parsed.y).toFixed(2)}`,
                    },
                },
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8892a0', maxTicksLimit: 8 },
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#8892a0',
                        callback: (value) => `¥${Number(value).toFixed(0)}`,
                    },
                },
            },
            interaction: {
                intersect: false,
                mode: 'index',
            },
        },
    });

    updateSignalChart(currentStock, cnSignalPeriod).catch((error) => {
        console.error('Failed to initialize CN signal chart:', error);
    });
}

async function updateSignalChart(symbol, period = '1D') {
    if (!cnSignalChart) return;
    cnSignalPeriod = period;

    const periodConfig = {
        '1D': { interval: 'hourly', limit: 48, labelUnit: 'hour' },
        '1W': { interval: 'daily', limit: 14, labelUnit: 'day' },
        '1M': { interval: 'daily', limit: 45, labelUnit: 'day' },
    };
    const cfg = periodConfig[String(period || '').toUpperCase()] || periodConfig['1D'];

    let labels = [];
    let values = [];
    try {
        const history = await api.market.getHistory(symbol, {
            period,
            interval: cfg.interval,
            limit: cfg.limit,
        });
        const series = _buildSeriesFromBars(history?.bars, cfg.interval);
        labels = series.labels;
        values = series.values;
    } catch (error) {
        console.error(`Failed to load CN history for ${symbol}:`, error);
    }

    if (!labels.length || !values.length) {
        return;
    }

    cnSignalChart.data.labels = labels;
    cnSignalChart.data.datasets[0].data = values;
    cnSignalChart.update('none');
}

// ========================================
// 实时更新
// ========================================
function startRealTimeUpdates() {
    setInterval(async () => {
        try {
            await refreshData();
        } catch (error) {
            console.error('Auto refresh failed:', error);
        }
    }, 10000);
}

function updateLivePrices() {
    // no-op: replaced by API polling in startRealTimeUpdates
}

// ========================================
// 刷新数据
// ========================================
async function refreshData() {
    showLoadingState();
    await loadMarketData();
    await updateCNIndexCharts();
    await loadPredictions();
    await loadSignalData();
    await updateSignalChart(currentStock, cnSignalPeriod);
    hideLoadingState();
    if (typeof showToast === 'function') {
        showToast('数据已刷新', 'success');
    }
}

// ========================================
// 事件绑定
// ========================================
function bindEvents() {
    // 股票选择器
    const stockSelector = document.getElementById('stockSelector');
    if (stockSelector) {
        stockSelector.addEventListener('change', async (e) => {
            currentStock = e.target.value;
            await loadSignalData();
        });
    }

    // signal chart period switch
    document.querySelectorAll('.chart-container-enhanced .chart-btn').forEach((btn) => {
        btn.addEventListener('click', async (e) => {
            const controls = e.currentTarget.closest('.chart-controls');
            controls?.querySelectorAll('.chart-btn').forEach((b) => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
            const period = e.currentTarget.dataset.period || '1D';
            await updateSignalChart(currentStock, period);
        });
    });

    // 表格排序
    const tableHeaders = document.querySelectorAll('.stock-table th');
    tableHeaders.forEach((header, index) => {
        header.addEventListener('click', () => {
            sortTable(index);
        });
    });

    // 刷新按钮
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshData);
    }
}

function sortTable(columnIndex) {
    const table = document.getElementById('stockTable');
    if (!table) return;
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
        const aVal = a.cells[columnIndex].textContent;
        const bVal = b.cells[columnIndex].textContent;
        return aVal.localeCompare(bVal, 'zh-CN', { numeric: true });
    });
    rows.forEach(row => tbody.appendChild(row));
}

// ========================================
// 辅助函数
// ========================================
function generateSparklineData(min, max, points = 20) {
    const data = [];
    let current = (min + max) / 2;
    for (let i = 0; i < points; i++) {
        const change = (Math.random() - 0.45) * (max - min) * 0.05;
        current = Math.max(min, Math.min(max, current + change));
        data.push(current);
    }
    return data;
}

function generateTimeLabels(count, unit = 'hour') {
    const labels = [];
    const now = new Date();
    for (let i = count - 1; i >= 0; i--) {
        const time = new Date(now);
        switch (unit) {
            case 'hour':
                time.setHours(time.getHours() - i);
                labels.push(`${String(time.getHours()).padStart(2, '0')}:00`);
                break;
            case 'month':
                time.setMonth(time.getMonth() - i);
                labels.push(time.toLocaleDateString('zh-CN', { month: 'short' }));
                break;
            default:
                time.setDate(time.getDate() - i);
                labels.push(time.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' }));
                break;
        }
    }
    return labels;
}

// 导出刷新函数
window.refreshCNData = refreshData;

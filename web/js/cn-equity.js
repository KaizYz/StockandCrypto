// ========================================
// StockandCrypto - CN Equity 页面脚本
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    initCNEquityPage();
});

let cnIndexCharts = {};
let cnTrendChart = null;
let cnStocksData = [];
let currentStock = '600519.SH';

// ========================================
// 初始化
// ========================================
async function initCNEquityPage() {
    // 显示加载状态
    showLoadingState();

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

    // 隐藏加载状态
    hideLoadingState();
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
    if (currentPriceEl && signal.current_price) {
        currentPriceEl.textContent = `¥${signal.current_price.toFixed(2)}`;
    }

    const targetPriceEl = document.getElementById('targetPrice');
    if (targetPriceEl && signal.target_price) {
        targetPriceEl.textContent = `¥${signal.target_price.toFixed(2)}`;
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
    const confidence = signal.confidence || signal.confidence_score || 60;
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
    predictions.forEach(pred => {
        const symbol = pred.symbol?.toLowerCase();
        const priceEl = document.getElementById(`${symbol}Price`);
        const changeEl = document.getElementById(`${symbol}Change`);
        const targetEl = document.getElementById(`${symbol}Target`);
        if (priceEl && pred.current_price) {
            priceEl.textContent = `¥${pred.current_price.toFixed(2)}`;
        }
        if (changeEl && pred.change_percent !== undefined) {
            changeEl.textContent = `${pred.change_percent >= 0 ? '+' : ''}${pred.change_percent.toFixed(2)}%`;
            changeEl.className = pred.change_percent >= 0 ? 'positive' : 'negative';
        }
        if (targetEl && pred.target_price) {
            targetEl.textContent = `¥${pred.target_price}`;
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
    createIndexChart('shChart', generateSparklineData(3050, 3100), '#00d4aa');
    createIndexChart('szChart', generateSparklineData(10200, 10300), '#d4af37');
    createIndexChart('cybChart', generateSparklineData(2080, 2100), '#ff6b6b');
}

function createIndexChart(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 80);
    gradient.addColorStop(0, color + '40');
    gradient.addColorStop(1, color + '00');

    cnIndexCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map((_, i) => i),
            datasets: [{
                data: data,
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

// ========================================
// 实时更新
// ========================================
function startRealTimeUpdates() {
    setInterval(() => {
        updateLivePrices();
    }, 5000);
}

function updateLivePrices() {
    // 模拟价格更新
    const updates = {
        sh: 3085 + (Math.random() - 0.5) * 10,
        sz: 10256 + (Math.random() - 0.5) * 20,
        cyb: 2085 + (Math.random() - 0.5) * 5
    };
    Object.keys(updates).forEach(symbol => {
        const priceEl = document.getElementById(`${symbol}Price`);
        if (priceEl) {
            animatePrice(priceEl, updates[symbol]);
        }
    });
}

// ========================================
// 刷新数据
// ========================================
async function refreshData() {
    showLoadingState();
    await loadMarketData();
    await loadPredictions();
    await loadSignalData();
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
function generateSparklineData(min, max) {
    const data = [];
    let current = (min + max) / 2;
    for (let i = 0; i < 20; i++) {
        const change = (Math.random() - 0.45) * (max - min) * 0.05;
        current = Math.max(min, Math.min(max, current + change));
        data.push(current);
    }
    return data;
}

// 导出刷新函数
window.refreshCNData = refreshData;

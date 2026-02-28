// ========================================
// StockandCrypto - Crypto 页面脚本
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    initCryptoPage();
});

let cryptoCharts = {};
let trendChart = null;
let symbolsData = [];
let currentSymbol = 'BTCUSDT';
let currentExchange = 'binance';
let currentMarketType = 'perp';

// ========================================
// 初始化
// ========================================
async function initCryptoPage() {
    // 显示加载状态
    showLoadingState();

    // 加载市场数据
    await loadMarketData();

    // 初始化图表
    initCharts();

    // 加载预测数据
    await loadPredictions();

    // 加载当前选择的币种信号
    await loadSignalData();

    // 加载币种列表
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
    const grid = document.getElementById('cryptoGrid');
    if (grid) {
        grid.classList.add('loading');
    }
    const signalCard = document.getElementById('signalCardMain');
    if (signalCard) {
        signalCard.classList.add('loading');
    }
}

function hideLoadingState() {
    const grid = document.getElementById('cryptoGrid');
    if (grid) {
        grid.classList.remove('loading');
    }
    const signalCard = document.getElementById('signalCardMain');
    if (signalCard) {
        signalCard.classList.remove('loading');
    }
}

function showError(message) {
    const grid = document.getElementById('cryptoGrid');
    if (grid) {
        grid.innerHTML = `<div class="error-message"><p>❌ ${message}</p><button onclick="initCryptoPage()" class="btn btn-secondary">重试</button></div>`;
    }
}

function showEmpty(message = '暂无数据') {
    const grid = document.getElementById('cryptoGrid');
    if (grid) {
        grid.innerHTML = `<div class="empty-state"><p>📭 ${message}</p></div>`;
    }
}

// ========================================
// 加载市场数据
// ========================================
async function loadMarketData() {
    try {
        // 尝试从API获取数据
        const data = await api.market.getOverview();
        if (data) {
            updatePriceCards(data);
        } else {
            // 使用模拟数据
            useMockData();
        }
    } catch (error) {
        console.error('Failed to load market data:', error);
        // 使用模拟数据
        useMockData();
    }
}

// ========================================
// 加载预测数据
// ========================================
async function loadPredictions() {
    try {
        const predictions = await api.crypto.getPredictions();
        if (predictions && predictions.length > 0) {
            updatePredictionsFromAPI(predictions);
        }
    } catch (error) {
        console.error('Failed to load predictions:', error);
        // 继续使用模拟数据
    }
}

// ========================================
// 加载单个币种信号数据
// ========================================
async function loadSignalData() {
    try {
        const symbol = currentSymbol;
        // 尝试从API获取单个币种信号
        const signal = await api.crypto.getPrediction(symbol);
        if (signal) {
            updateSignalCard(signal);
        } else {
            // 使用模拟数据
            useMockSignalData(symbol);
        }
    } catch (error) {
        console.error('Failed to load signal data:', error);
        useMockSignalData(currentSymbol);
    }
}

// ========================================
// 加载币种列表
// ========================================
async function loadSymbols() {
    try {
        const symbols = await api.crypto.getSymbols();
        if (symbols && symbols.length > 0) {
            symbolsData = symbols;
            updateSymbolsList(symbols);
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
        btc: {
            price: 67542.89,
            change: 2.34,
            high: 68120,
            low: 65890,
            volume: '28.5B',
            predicted: 68500,
            support: 66500,
            resistance: 68500
        },
        eth: {
            price: 3456.72,
            change: 1.87,
            high: 3520,
            low: 3380,
            volume: '12.3B',
            predicted: 3520,
            support: 3400,
            resistance: 3520
        },
        sol: {
            price: 142.35,
            change: -0.52,
            high: 145.80,
            low: 139.20,
            volume: '2.8B',
            predicted: 138,
            support: 140,
            resistance: 145
        }
    };
    updatePriceCards(mockData);
    updatePredictions(mockData);
}

// ========================================
// 使用模拟信号数据
// ========================================
function useMockSignalData(symbol) {
    const mockSignals = {
        'BTCUSDT': {
            symbol: 'BTCUSDT',
            action: 'Long',
            p_up: 0.58,
            p_down: 0.42,
            confidence: 72,
            current_price: 67542.89,
            target_price: 68500,
            q50_change_pct: 0.024,
            volatility_score: 0.35,
            signal_strength: 'Medium',
            signal_strength_pp: 8.0,
            trend_label: 'bullish',
            risk_level: 'medium',
            position_size: 0.15
        },
        'ETHUSDT': {
            symbol: 'ETHUSDT',
            action: 'Long',
            p_up: 0.55,
            p_down: 0.45,
            confidence: 68,
            current_price: 3456.72,
            target_price: 3520,
            q50_change_pct: 0.018,
            volatility_score: 0.32,
            signal_strength: 'Medium',
            signal_strength_pp: 5.0,
            trend_label: 'bullish',
            risk_level: 'medium',
            position_size: 0.12
        },
        'SOLUSDT': {
            symbol: 'SOLUSDT',
            action: 'Short',
            p_up: 0.42,
            p_down: 0.58,
            confidence: 65,
            current_price: 142.35,
            target_price: 138,
            q50_change_pct: -0.032,
            volatility_score: 0.45,
            signal_strength: 'Weak',
            signal_strength_pp: 8.0,
            trend_label: 'bearish',
            risk_level: 'high',
            position_size: 0.08
        },
        'DOGEUSDT': {
            symbol: 'DOGEUSDT',
            action: 'Flat',
            p_up: 0.50,
            p_down: 0.50,
            confidence: 55,
            current_price: 0.125,
            target_price: 0.128,
            q50_change_pct: 0.015,
            volatility_score: 0.55,
            signal_strength: 'Weak',
            signal_strength_pp: 0.0,
            trend_label: 'neutral',
            risk_level: 'high',
            position_size: 0.0
        },
        'XRPUSDT': {
            symbol: 'XRPUSDT',
            action: 'Long',
            p_up: 0.62,
            p_down: 0.38,
            confidence: 78,
            current_price: 0.52,
            target_price: 0.58,
            q50_change_pct: 0.115,
            volatility_score: 0.28,
            signal_strength: 'Strong',
            signal_strength_pp: 12.0,
            trend_label: 'bullish',
            risk_level: 'low',
            position_size: 0.20
        },
        'ADAUSDT': {
            symbol: 'ADAUSDT',
            action: 'Flat',
            p_up: 0.48,
            p_down: 0.52,
            confidence: 52,
            current_price: 0.45,
            target_price: 0.46,
            q50_change_pct: 0.022,
            volatility_score: 0.38,
            signal_strength: 'Weak',
            signal_strength_pp: 2.0,
            trend_label: 'neutral',
            risk_level: 'medium',
            position_size: 0.0
        }
    };

    const signal = mockSignals[symbol] || mockSignals['BTCUSDT'];
    updateSignalCard(signal);
}

// ========================================
// 更新信号卡片
// ========================================
function updateSignalCard(signal) {
    // 更新币种
    const symbolEl = document.getElementById('signalSymbol');
    if (symbolEl) {
        symbolEl.textContent = signal.symbol || currentSymbol;
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
        actionEl.textContent = _formatAction(action);
        actionEl.className = 'action-value ' + action.toLowerCase();
    }

    // 更新价格
    const currentPriceEl = document.getElementById('currentPrice');
    if (currentPriceEl && signal.current_price) {
        currentPriceEl.textContent = `$${signal.current_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    }

    const targetPriceEl = document.getElementById('targetPrice');
    if (targetPriceEl && signal.target_price) {
        targetPriceEl.textContent = `$${signal.target_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
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
        strengthEl.textContent = `${_signalStrengthText(strength)} (${strengthPP.toFixed(1)}pp)`;
        strengthEl.className = 'detail-value strength-' + strength.toLowerCase();
    }

    const trendEl = document.getElementById('trendLabel');
    if (trendEl) {
        trendEl.textContent = _trendText(signal.trend_label || 'neutral');
        trendEl.className = 'detail-value trend-' + (signal.trend_label || 'neutral');
    }

    const riskEl = document.getElementById('riskLevel');
    if (riskEl) {
        riskEl.textContent = _riskText(signal.risk_level || 'medium');
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
function _formatAction(action) {
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

function _signalStrengthText(strength) {
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

function _trendText(trend) {
    const trendMap = {
        'bullish': '看涨',
        'bearish': '看跌',
        'neutral': '震荡',
        'bull': '看涨',
        'bear': '看跌'
    };
    return trendMap[trend] || trend;
}

function _riskText(risk) {
    const riskMap = {
        'low': '低风险',
        'medium': '中风险',
        'high': '高风险'
    };
    return riskMap[risk] || risk;
}

// ========================================
// 更新价格卡片
// ========================================
function updatePriceCards(data) {
    // BTC
    if (data.btc) {
        updateCryptoCard('btc', data.btc);
    }
    // ETH
    if (data.eth) {
        updateCryptoCard('eth', data.eth);
    }
    // SOL
    if (data.sol) {
        updateCryptoCard('sol', data.sol);
    }
}

function updateCryptoCard(symbol, data) {
    const priceEl = document.getElementById(`${symbol}Price`);
    const changeEl = document.getElementById(`${symbol}Change`);
    const highEl = document.getElementById(`${symbol}High`);
    const lowEl = document.getElementById(`${symbol}Low`);
    const volumeEl = document.getElementById(`${symbol}Volume`);

    if (priceEl) {
        animatePrice(priceEl, data.price);
    }
    if (changeEl) {
        changeEl.textContent = `${data.change >= 0 ? '+' : ''}${data.change.toFixed(2)}%`;
        changeEl.className = `crypto-change ${data.change >= 0 ? 'positive' : 'negative'}`;
    }
    if (highEl) highEl.textContent = `$${data.high.toLocaleString()}`;
    if (lowEl) lowEl.textContent = `$${data.low.toLocaleString()}`;
    if (volumeEl) volumeEl.textContent = `$${data.volume}`;
}

// ========================================
// 更新预测数据
// ========================================
function updatePredictions(data) {
    // BTC 预测
    if (data.btc) {
        const btcPredicted = document.getElementById('btcPredicted');
        const btcSupport = document.getElementById('btcSupport');
        const btcResistance = document.getElementById('btcResistance');
        if (btcPredicted) btcPredicted.textContent = `$${data.btc.predicted.toLocaleString()}`;
        if (btcSupport) btcSupport.textContent = `$${data.btc.support.toLocaleString()}`;
        if (btcResistance) btcResistance.textContent = `$${data.btc.resistance.toLocaleString()}`;
    }
    // ETH 预测
    if (data.eth) {
        const ethPredicted = document.getElementById('ethPredicted');
        const ethSupport = document.getElementById('ethSupport');
        const ethResistance = document.getElementById('ethResistance');
        if (ethPredicted) ethPredicted.textContent = `$${data.eth.predicted.toLocaleString()}`;
        if (ethSupport) ethSupport.textContent = `$${data.eth.support.toLocaleString()}`;
        if (ethResistance) ethResistance.textContent = `$${data.eth.resistance.toLocaleString()}`;
    }
    // SOL 预测
    if (data.sol) {
        const solPredicted = document.getElementById('solPredicted');
        const solSupport = document.getElementById('solSupport');
        const solResistance = document.getElementById('solResistance');
        if (solPredicted) solPredicted.textContent = `$${data.sol.predicted.toLocaleString()}`;
        if (solSupport) solSupport.textContent = `$${data.sol.support.toLocaleString()}`;
        if (solResistance) solResistance.textContent = `$${data.sol.resistance.toLocaleString()}`;
    }
}

// ========================================
// 从API更新预测数据
// ========================================
function updatePredictionsFromAPI(predictions) {
    predictions.forEach(pred => {
        const symbol = pred.symbol?.toLowerCase();
        const predictedEl = document.getElementById(`${symbol}Predicted`);
        const supportEl = document.getElementById(`${symbol}Support`);
        const resistanceEl = document.getElementById(`${symbol}Resistance`);
        if (predictedEl && pred.predicted_price) {
            predictedEl.textContent = `$${pred.predicted_price.toLocaleString()}`;
        }
        if (supportEl && pred.support_level) {
            supportEl.textContent = `$${pred.support_level.toLocaleString()}`;
        }
        if (resistanceEl && pred.resistance_level) {
            resistanceEl.textContent = `$${pred.resistance_level.toLocaleString()}`;
        }
    });
}

// ========================================
// 更新币种列表
// ========================================
function updateSymbolsList(symbols) {
    // 可以在页面上显示支持的币种列表
    console.log('Supported symbols:', symbols);
}

// ========================================
// 价格动画
// ========================================
function animatePrice(element, newPrice) {
    const currentPrice = parseFloat(element.textContent.replace(/,/g, '')) || 0;
    const diff = newPrice - currentPrice;
    element.style.transition = 'color 0.3s ease';
    if (diff > 0) {
        element.style.color = '#00d4aa';
    } else if (diff < 0) {
        element.style.color = '#ff6b6b';
    }
    element.textContent = newPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    setTimeout(() => {
        element.style.color = '';
    }, 500);
}

// ========================================
// 初始化图表
// ========================================
function initCharts() {
    // Chart.js 配置
    Chart.defaults.color = '#8892a0';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.font.family = 'Inter, sans-serif';

    // 创建 Sparkline 图表
    createSparkline('btcChart', generateSparklineData(67000, 68000), '#00d4aa');
    createSparkline('ethChart', generateSparklineData(3400, 3500), '#d4af37');
    createSparkline('solChart', generateSparklineData(140, 145), '#ff6b6b');

    // 创建趋势图表
    createTrendChart();
}

function createSparkline(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    // 创建渐变
    const gradient = ctx.createLinearGradient(0, 0, 0, 100);
    gradient.addColorStop(0, color + '40');
    gradient.addColorStop(1, color + '00');

    cryptoCharts[canvasId] = new Chart(ctx, {
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

function createTrendChart() {
    const canvas = document.getElementById('trendChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const labels = generateTimeLabels(24);

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'BTC',
                    data: generateTrendData(67000, 68000, 24),
                    borderColor: '#00d4aa',
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'ETH',
                    data: generateTrendData(3400, 3500, 24),
                    borderColor: '#d4af37',
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: 'SOL',
                    data: generateTrendData(140, 145, 24),
                    borderColor: '#8892a0',
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    align: 'end',
                    labels: {
                        usePointStyle: true,
                        pointStyle: 'circle',
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 15, 26, 0.9)',
                    titleColor: '#ffffff',
                    bodyColor: '#8892a0',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 12
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#8892a0' }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#8892a0',
                        callback: (value) => '$' + value.toLocaleString()
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
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
        btc: 67000 + Math.random() * 1000,
        eth: 3400 + Math.random() * 100,
        sol: 140 + Math.random() * 5
    };

    // 更新价格显示
    ['btc', 'eth', 'sol'].forEach(symbol => {
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
    // 币种选择器
    const symbolSelector = document.getElementById('symbolSelector');
    if (symbolSelector) {
        symbolSelector.addEventListener('change', async (e) => {
            currentSymbol = e.target.value;
            await loadSignalData();
        });
    }

    // 交易所选择器
    const exchangeSelector = document.getElementById('exchangeSelector');
    if (exchangeSelector) {
        exchangeSelector.addEventListener('change', async (e) => {
            currentExchange = e.target.value;
            await loadSignalData();
        });
    }

    // 市场类型选择器
    const marketTypeSelector = document.getElementById('marketTypeSelector');
    if (marketTypeSelector) {
        marketTypeSelector.addEventListener('change', async (e) => {
            currentMarketType = e.target.value;
            await loadSignalData();
        });
    }

    // 图表周期切换
    document.querySelectorAll('.chart-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            const period = e.target.dataset.period;
            updateChartPeriod(period);
        });
    });

    // 刷新按钮
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshData);
    }
}

function updateChartPeriod(period) {
    if (!trendChart) return;

    const periodConfig = {
        '1D': { points: 24, label: 'hour' },
        '1W': { points: 7, label: 'day' },
        '1M': { points: 30, label: 'day' },
        '1Y': { points: 12, label: 'month' }
    };

    const config = periodConfig[period] || periodConfig['1D'];
    const labels = generateTimeLabels(config.points, config.label);
    trendChart.data.labels = labels;
    trendChart.data.datasets[0].data = generateTrendData(67000, 68000, config.points);
    trendChart.data.datasets[1].data = generateTrendData(3400, 3500, config.points);
    trendChart.data.datasets[2].data = generateTrendData(140, 145, config.points);
    trendChart.update();
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

function generateTrendData(min, max, points) {
    const data = [];
    let current = (min + max) / 2;
    for (let i = 0; i < points; i++) {
        const change = (Math.random() - 0.45) * (max - min) * 0.1;
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
                labels.push(time.getHours() + ':00');
                break;
            case 'day':
                time.setDate(time.getDate() - i);
                labels.push(time.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' }));
                break;
            case 'month':
                time.setMonth(time.getMonth() - i);
                labels.push(time.toLocaleDateString('zh-CN', { month: 'short' }));
                break;
        }
    }
    return labels;
}

// 导出刷新函数
window.refreshCryptoData = refreshData;

// ========================================
// StockandCrypto - Crypto é¡µé¢è„šæœ¬
// ========================================
document.addEventListener('DOMContentLoaded', function() {
 initCryptoPage();
});

let cryptoCharts = {};
let trendChart = null;
let signalChart = null;
let symbolsData = [];
let currentSymbol = 'BTCUSDT';
let currentExchange = 'binance';
let currentMarketType = 'perp';
let signalChartPeriod = '1D';
let marketTrendPeriod = '1D';

function _safeNumber(value) {
 const n = Number(value);
 return Number.isFinite(n) ? n : null;
}

function _symbolToCryptoCardKey(symbol) {
 const s = String(symbol || '').toUpperCase();
 if (s.startsWith('BTC')) return 'btc';
 if (s.startsWith('ETH')) return 'eth';
 if (s.startsWith('SOL')) return 'sol';
 return s.toLowerCase();
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
// åˆå§‹åŒ–
// ========================================
async function initCryptoPage() {
 // æ˜¾ç¤ºåŠ è½½çŠ¶æ€
 showLoadingState();
 try {
 // åŠ è½½å¸‚åœºæ•°æ®
 await loadMarketData();
 // åˆå§‹åŒ–å›¾è¡¨
 initCharts();
 // åŠ è½½é¢„æµ‹æ•°æ®
 await loadPredictions();
 // åŠ è½½å½“å‰é€‰æ‹©çš„å¸ç§ä¿¡å·
 await loadSignalData();
 // åŠ è½½å¸ç§åˆ—è¡¨
 await loadSymbols();
 // å¯åŠ¨å®žæ—¶æ›´æ–°
 startRealTimeUpdates();
 // ç»‘å®šäº‹ä»¶
 bindEvents();
 } catch (error) {
 console.error('Failed to initialize crypto page:', error);
 if (typeof showToast === 'function') {
 showToast(`é¡µé¢åˆå§‹åŒ–å¤±è´¥: ${error.message || 'æœªçŸ¥é”™è¯¯'}`, 'error');
 }
 } finally {
 // éšè—åŠ è½½çŠ¶æ€
 hideLoadingState();
 }
}

// ========================================
// æ˜¾ç¤º/éšè—åŠ è½½çŠ¶æ€
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
 grid.innerHTML = `<div class="error-message"><p>âŒ ${message}</p><button onclick="initCryptoPage()" class="btn btn-secondary">é‡è¯•</button></div>`;
 }
}

function showEmpty(message = 'æš‚æ— æ•°æ®') {
 const grid = document.getElementById('cryptoGrid');
 if (grid) {
 grid.innerHTML = `<div class="empty-state"><p>ðŸ“­ ${message}</p></div>`;
 }
}

// ========================================
// åŠ è½½å¸‚åœºæ•°æ®
// ========================================
async function loadMarketData() {
 try {
 const data = await api.market.getOverview();
 if (data) {
 updatePriceCards(data);
 }
 } catch (error) {
 console.error('Failed to load market data:', error);
 if (typeof showToast === 'function') {
 showToast('实时行情获取失败: ' + (error.message || '未知错误'), 'error');
 }
 }
}

// ========================================
// åŠ è½½é¢„æµ‹æ•°æ®
// ========================================
async function loadPredictions() {
 try {
 const predictions = await api.crypto.getPredictions();
 if (predictions && predictions.length > 0) {
 updatePredictionsFromAPI(predictions);
 }
 } catch (error) {
 console.error('Failed to load predictions:', error);
 // ç»§ç»­ä½¿ç”¨æ¨¡æ‹Ÿæ•°æ®
 }
}

// ========================================
// åŠ è½½å•ä¸ªå¸ç§ä¿¡å·æ•°æ®
// ========================================
async function loadSignalData() {
 try {
 const symbol = currentSymbol;
 const signal = await api.crypto.getPrediction(symbol);
 if (signal) {
 updateSignalCard(signal);
 await updateSignalChart(symbol, signalChartPeriod);
 }
 } catch (error) {
 console.error('Failed to load signal data:', error);
 if (typeof showToast === 'function') {
 showToast('信号数据获取失败: ' + (error.message || '未知错误'), 'error');
 }
 }
}

// ========================================
// åŠ è½½å¸ç§åˆ—è¡¨
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
// ä½¿ç”¨æ¨¡æ‹Ÿæ•°æ®
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
// ä½¿ç”¨æ¨¡æ‹Ÿä¿¡å·æ•°æ®
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
 updateSignalChart(symbol);
}

// ========================================
// æ›´æ–°ä¿¡å·å¡ç‰‡
// ========================================
function updateSignalCard(signal) {
 // æ›´æ–°å¸ç§
 const symbolEl = document.getElementById('signalSymbol');
 if (symbolEl) {
 symbolEl.textContent = signal.symbol || currentSymbol;
 }
 
 // æ›´æ–°ä»£ç æ˜¾ç¤º
 const codeEl = document.getElementById('signalCode');
 if (codeEl) {
 const symbolNames = {
 'BTCUSDT': 'Bitcoin',
 'ETHUSDT': 'Ethereum',
 'SOLUSDT': 'Solana',
 'DOGEUSDT': 'Dogecoin',
 'XRPUSDT': 'Ripple',
 'ADAUSDT': 'Cardano'
 };
 codeEl.textContent = symbolNames[signal.symbol] || signal.symbol;
 }
 
 // æ›´æ–°æ—¶é—´
 const timeEl = document.getElementById('signalTime');
 if (timeEl) {
 const now = new Date();
 timeEl.textContent = `æ›´æ–°äºŽ ${now.toLocaleTimeString('zh-CN')}`;
 }
 
 // æ›´æ–°ç­–ç•¥åŠ¨ä½œ
 const actionEl = document.getElementById('signalAction');
 if (actionEl) {
 const action = signal.action || signal.policy_action || 'Flat';
 actionEl.textContent = _formatAction(action);
 actionEl.className = 'action-value ' + action.toLowerCase();
 }
 
 // æ›´æ–°ä»·æ ¼
 const currentPriceEl = document.getElementById('currentPrice');
 const currentPrice = _safeNumber(signal.current_price);
 if (currentPriceEl && currentPrice !== null) {
 currentPriceEl.textContent = `$${currentPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
 }
 
 const targetPriceEl = document.getElementById('targetPrice');
 const targetPrice = _safeNumber(signal.target_price ?? signal.target_price_q50);
 if (targetPriceEl && targetPrice !== null) {
 targetPriceEl.textContent = `$${targetPrice.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
 }
 
 // æ›´æ–°æ¦‚çŽ‡
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
 
 // æ›´æ–°ç½®ä¿¡åº¦
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
 
 // æ›´æ–°è¯¦æƒ…
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
// æ ¼å¼åŒ–å‡½æ•°
// ========================================
function _formatAction(action) {
 const actionMap = {
 'Long': 'åšå¤š',
 'Short': 'åšç©º',
 'Flat': 'è§‚æœ›',
 'long': 'åšå¤š',
 'short': 'åšç©º',
 'flat': 'è§‚æœ›'
 };
 return actionMap[action] || action;
}

function _signalStrengthText(strength) {
 const strengthMap = {
 'Weak': 'å¼±ä¿¡å·',
 'Medium': 'ä¸­ä¿¡å·',
 'Strong': 'å¼ºä¿¡å·',
 'weak': 'å¼±ä¿¡å·',
 'medium': 'ä¸­ä¿¡å·',
 'strong': 'å¼ºä¿¡å·'
 };
 return strengthMap[strength] || strength;
}

function _trendText(trend) {
 const trendMap = {
 'bullish': 'çœ‹æ¶¨',
 'bearish': 'çœ‹è·Œ',
 'neutral': 'éœ‡è¡',
 'bull': 'çœ‹æ¶¨',
 'bear': 'çœ‹è·Œ'
 };
 return trendMap[trend] || trend;
}

function _riskText(risk) {
 const riskMap = {
 'low': 'ä½Žé£Žé™©',
 'medium': 'ä¸­é£Žé™©',
 'high': 'é«˜é£Žé™©'
 };
 return riskMap[risk] || risk;
}

// ========================================
// æ›´æ–°ä»·æ ¼å¡ç‰‡
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
// æ›´æ–°é¢„æµ‹æ•°æ®
// ========================================
function updatePredictions(data) {
 // BTC é¢„æµ‹
 if (data.btc) {
 const btcPredicted = document.getElementById('btcPredicted');
 const btcSupport = document.getElementById('btcSupport');
 const btcResistance = document.getElementById('btcResistance');
 if (btcPredicted) btcPredicted.textContent = `$${data.btc.predicted.toLocaleString()}`;
 if (btcSupport) btcSupport.textContent = `$${data.btc.support.toLocaleString()}`;
 if (btcResistance) btcResistance.textContent = `$${data.btc.resistance.toLocaleString()}`;
 }
 // ETH é¢„æµ‹
 if (data.eth) {
 const ethPredicted = document.getElementById('ethPredicted');
 const ethSupport = document.getElementById('ethSupport');
 const ethResistance = document.getElementById('ethResistance');
 if (ethPredicted) ethPredicted.textContent = `$${data.eth.predicted.toLocaleString()}`;
 if (ethSupport) ethSupport.textContent = `$${data.eth.support.toLocaleString()}`;
 if (ethResistance) ethResistance.textContent = `$${data.eth.resistance.toLocaleString()}`;
 }
 // SOL é¢„æµ‹
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
// ä»ŽAPIæ›´æ–°é¢„æµ‹æ•°æ®
// ========================================
function updatePredictionsFromAPI(predictions) {
 predictions.forEach((pred) => {
 const key = _symbolToCryptoCardKey(pred.symbol);
 const predictedEl = document.getElementById(`${key}Predicted`);
 const supportEl = document.getElementById(`${key}Support`);
 const resistanceEl = document.getElementById(`${key}Resistance`);
 const predicted = _safeNumber(pred.predicted_price ?? pred.target_price);
 const support = _safeNumber(pred.support_level);
 const resistance = _safeNumber(pred.resistance_level);
 if (predictedEl && predicted !== null) {
 predictedEl.textContent = `$${predicted.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
 }
 if (supportEl && support !== null) {
 supportEl.textContent = `$${support.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
 }
 if (resistanceEl && resistance !== null) {
 resistanceEl.textContent = `$${resistance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
 }
 });
}

// ========================================
// æ›´æ–°å¸ç§åˆ—è¡¨
// ========================================
function updateSymbolsList(symbols) {
 const selector = document.getElementById('symbolSelector');
 if (!selector) return;
 const previous = currentSymbol;
 selector.innerHTML = '';
 symbols.forEach((item) => {
 const symbol = String(item.symbol || '').toUpperCase();
 if (!symbol) return;
 const name = item.name || symbol.replace('USDT', '');
 const option = document.createElement('option');
 option.value = symbol;
 option.textContent = `${symbol} - ${name}`;
 selector.appendChild(option);
 });
 if ([...selector.options].some((o) => o.value === previous)) {
 selector.value = previous;
 currentSymbol = previous;
 } else if (selector.options.length > 0) {
 selector.selectedIndex = 0;
 currentSymbol = selector.value;
 }
}

// ========================================
// ä»·æ ¼åŠ¨ç”»
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
// åˆå§‹åŒ–å›¾è¡¨
// ========================================
function initCharts() {
 // Chart.js é…ç½®
 Chart.defaults.color = '#8892a0';
 Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
 Chart.defaults.font.family = 'Inter, sans-serif';
 // åˆ›å»º Sparkline å›¾è¡¨
 createSparkline('btcChart', [], '#00d4aa');
 createSparkline('ethChart', [], '#d4af37');
 createSparkline('solChart', [], '#ff6b6b');
 // åˆ›å»ºä¿¡å·å›¾è¡¨
 createSignalChart();
 // åˆ›å»ºè¶‹åŠ¿å›¾è¡¨
 createTrendChart();
 updateSparklineCharts().catch((error) => {
 console.error('Failed to initialize crypto sparkline history:', error);
 });
}

// ========================================
// åˆ›å»ºä¿¡å·å›¾è¡¨
// ========================================
function createSignalChart() {
 const canvas = document.getElementById('signalChart');
 if (!canvas) return;
 const ctx = canvas.getContext('2d');
 
 const gradient = ctx.createLinearGradient(0, 0, 0, 350);
 gradient.addColorStop(0, 'rgba(0, 212, 170, 0.2)');
 gradient.addColorStop(1, 'rgba(0, 212, 170, 0)');
 
 signalChart = new Chart(ctx, {
 type: 'line',
 data: {
 labels: [],
 datasets: [{
 label: 'ä»·æ ¼',
 data: [],
 borderColor: '#00d4aa',
 backgroundColor: gradient,
 fill: true,
 tension: 0.4,
 borderWidth: 2,
 pointRadius: 0,
 pointHoverRadius: 4,
 pointHoverBackgroundColor: '#00d4aa'
 }]
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
 label: function(context) {
 return '$' + context.parsed.y.toLocaleString();
 }
 }
 }
 },
 scales: {
 x: {
 grid: { color: 'rgba(255, 255, 255, 0.05)' },
 ticks: { color: '#8892a0', maxTicksLimit: 6 }
 },
 y: {
 grid: { color: 'rgba(255, 255, 255, 0.05)' },
 ticks: {
 color: '#8892a0',
 callback: (value) => '$' + value.toLocaleString()
 }
 }
 },
 interaction: { intersect: false, mode: 'index' }
 }
 });

 // Load real history for the selected symbol.
 updateSignalChart(currentSymbol, signalChartPeriod).catch((error) => {
 console.error('Failed to initialize signal history chart:', error);
 });
}

// ========================================
// æ›´æ–°ä¿¡å·å›¾è¡¨
// ========================================
async function updateSignalChart(symbol, period = '1D') {
 if (!signalChart) return;
 signalChartPeriod = period;

 const chartColor = _symbolToCryptoCardKey(symbol) === 'eth'
 ? '#d4af37'
 : _symbolToCryptoCardKey(symbol) === 'sol'
 ? '#ff6b6b'
 : '#00d4aa';

 const periodConfig = {
 '1H': { interval: 'hourly', limit: 24 },
 '1D': { interval: 'hourly', limit: 72 },
 '1W': { interval: 'daily', limit: 14 },
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
 console.error(`Failed to load signal chart history for ${symbol}:`, error);
 }

 if (!labels.length || !values.length) {
 return;
 }

 signalChart.data.labels = labels;
 signalChart.data.datasets[0].data = values;
 signalChart.data.datasets[0].borderColor = chartColor;
 signalChart.data.datasets[0].pointHoverBackgroundColor = chartColor;

 const canvas = document.getElementById('signalChart');
 if (canvas) {
 const ctx = canvas.getContext('2d');
 const gradient = ctx.createLinearGradient(0, 0, 0, 350);
 gradient.addColorStop(0, chartColor + '33');
 gradient.addColorStop(1, chartColor + '00');
 signalChart.data.datasets[0].backgroundColor = gradient;
 }

 signalChart.update('none');
}

function createSparkline(canvasId, data, color) {
 const canvas = document.getElementById(canvasId);
 if (!canvas) return;
 const ctx = canvas.getContext('2d');
 const values = Array.isArray(data) ? data : [];
 // åˆ›å»ºæ¸å˜
 const gradient = ctx.createLinearGradient(0, 0, 0, 100);
 gradient.addColorStop(0, color + '40');
 gradient.addColorStop(1, color + '00');
 cryptoCharts[canvasId] = new Chart(ctx, {
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

async function updateSparklineCharts() {
 const defs = [
 { canvasId: 'btcChart', symbol: 'BTCUSDT' },
 { canvasId: 'ethChart', symbol: 'ETHUSDT' },
 { canvasId: 'solChart', symbol: 'SOLUSDT' },
 ];

 await Promise.all(defs.map(async ({ canvasId, symbol }) => {
 const chart = cryptoCharts[canvasId];
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
 console.error(`Failed to load sparkline history for ${symbol}:`, error);
 }
 }));
}

function createTrendChart() {
 const canvas = document.getElementById('trendChart');
 if (!canvas) return;
 const ctx = canvas.getContext('2d');
 const labels = [];
 trendChart = new Chart(ctx, {
 type: 'line',
 data: {
 labels: labels,
 datasets: [
 {
 label: 'BTC',
 data: [],
 borderColor: '#00d4aa',
 backgroundColor: 'transparent',
 tension: 0.4,
 borderWidth: 2,
 pointRadius: 0
 },
 {
 label: 'ETH',
 data: [],
 borderColor: '#d4af37',
 backgroundColor: 'transparent',
 tension: 0.4,
 borderWidth: 2,
 pointRadius: 0
 },
 {
 label: 'SOL',
 data: [],
 borderColor: '#cbd5e1',
 backgroundColor: 'transparent',
 tension: 0.4,
 borderWidth: 2.2,
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
 padding: 20,
 color: '#cbd5e1'
 }
 },
 tooltip: {
 backgroundColor: 'rgba(10, 15, 26, 0.9)',
 titleColor: '#ffffff',
 bodyColor: '#cbd5e1',
 borderColor: 'rgba(255, 255, 255, 0.1)',
 borderWidth: 1,
 cornerRadius: 8,
 padding: 12,
 callbacks: {
 label: (ctx) => {
 const v = Number(ctx.parsed.y);
 const sign = Number.isFinite(v) && v > 0 ? '+' : '';
 return `${ctx.dataset.label}: ${sign}${v.toFixed(2)}%`;
 }
 }
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
 callback: (value) => {
 const n = Number(value);
 const sign = Number.isFinite(n) && n > 0 ? '+' : '';
 return `${sign}${n.toFixed(1)}%`;
 }
 }
 }
 },
 interaction: {
 intersect: false,
 mode: 'index'
 }
 }
 });

 updateChartPeriod(marketTrendPeriod).catch((error) => {
 console.error('Failed to initialize market trend chart:', error);
 });
}

// ========================================
// å®žæ—¶æ›´æ–°
// ========================================
function startRealTimeUpdates() {
 setInterval(async () => {
 try {
 await refreshData({ silent: true });
 } catch (error) {
 console.error('Auto refresh failed:', error);
 }
 }, 10000);
}

function updateLivePrices() {
 // no-op: replaced by API polling in startRealTimeUpdates
}

// ========================================
// åˆ·æ–°æ•°æ®
// ========================================
async function refreshData(options = {}) {
 const silent = Boolean(options && options.silent);
 if (!silent) {
 showLoadingState();
 }
 await loadMarketData();
 await updateSparklineCharts();
 await loadPredictions();
 await loadSignalData();
 await updateChartPeriod(marketTrendPeriod);
 if (!silent) {
 hideLoadingState();
 }
 if (!silent && typeof showToast === 'function') {
 showToast('æ•°æ®å·²åˆ·æ–°', 'success');
 }
}

// ========================================
// äº‹ä»¶ç»‘å®š
// ========================================
function bindEvents() {
 // å¸ç§é€‰æ‹©å™¨
 const symbolSelector = document.getElementById('symbolSelector');
 if (symbolSelector) {
 symbolSelector.addEventListener('change', async (e) => {
 currentSymbol = e.target.value;
 await loadSignalData();
 });
 }
 
 // äº¤æ˜“æ‰€é€‰æ‹©å™¨
 const exchangeSelector = document.getElementById('exchangeSelector');
 if (exchangeSelector) {
 exchangeSelector.addEventListener('change', async (e) => {
 currentExchange = e.target.value;
 await loadSignalData();
 });
 }
 
 // å¸‚åœºç±»åž‹é€‰æ‹©å™¨
 const marketTypeSelector = document.getElementById('marketTypeSelector');
 if (marketTypeSelector) {
 marketTypeSelector.addEventListener('change', async (e) => {
 currentMarketType = e.target.value;
 await loadSignalData();
 });
 }
 
 // ä¸Šæ–¹ä¿¡å·å›¾å‘¨æœŸåˆ‡æ¢
 document.querySelectorAll('.chart-container-enhanced .chart-btn').forEach((btn) => {
 btn.addEventListener('click', async (e) => {
 const controls = e.currentTarget.closest('.chart-controls');
 controls?.querySelectorAll('.chart-btn').forEach((b) => b.classList.remove('active'));
 e.currentTarget.classList.add('active');
 const period = e.currentTarget.dataset.period || '1D';
 await updateSignalChart(currentSymbol, period);
 });
 });

 // ä¸‹æ–¹å¸‚åœºè¶‹åŠ¿å›¾å‘¨æœŸåˆ‡æ¢
 document.querySelectorAll('.chart-container .chart-btn').forEach((btn) => {
 btn.addEventListener('click', async (e) => {
 const controls = e.currentTarget.closest('.chart-controls');
 controls?.querySelectorAll('.chart-btn').forEach((b) => b.classList.remove('active'));
 e.currentTarget.classList.add('active');
 const period = e.currentTarget.dataset.period || '1D';
 await updateChartPeriod(period);
 });
 });
 
 // åˆ·æ–°æŒ‰é’®
 const refreshBtn = document.getElementById('refreshBtn');
 if (refreshBtn) {
 refreshBtn.addEventListener('click', refreshData);
 }
}

async function updateChartPeriod(period) {
 if (!trendChart) return;
 marketTrendPeriod = period;

 const cfgMap = {
 '1D': { interval: 'hourly', limit: 24, labelUnit: 'hour' },
 '1W': { interval: 'daily', limit: 7, labelUnit: 'day' },
 '1M': { interval: 'daily', limit: 30, labelUnit: 'day' },
 '1Y': { interval: 'daily', limit: 180, labelUnit: 'month' },
 };
 const cfg = cfgMap[String(period || '').toUpperCase()] || cfgMap['1D'];

 const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'];

 const results = await Promise.all(symbols.map(async (symbol) => {
 try {
 const history = await api.market.getHistory(symbol, {
 period,
 interval: cfg.interval,
 limit: cfg.limit,
 });
 const series = _buildSeriesFromBars(history?.bars, cfg.interval);
 if (series.labels.length > 1 && series.values.length > 1) {
 return series;
 }
 } catch (error) {
 console.error(`Failed to load trend history for ${symbol}:`, error);
 }
 return null;
 }));

 if (results.some((x) => !x)) {
 return;
 }

 const casted = results;
 const minLen = Math.max(2, Math.min(...casted.map((x) => x.values.length)));
 const base = casted[0];

 const toPctSeries = (values) => {
 const trimmed = values.slice(-minLen);
 const baseValue = trimmed.find((v) => Number.isFinite(v) && v > 0);
 if (!Number.isFinite(baseValue) || baseValue <= 0) {
 return trimmed.map(() => 0);
 }
 return trimmed.map((v) => ((v / baseValue) - 1) * 100);
 };

 trendChart.data.labels = base.labels.slice(-minLen);
 trendChart.data.datasets[0].data = toPctSeries(casted[0].values);
 trendChart.data.datasets[1].data = toPctSeries(casted[1].values);
 trendChart.data.datasets[2].data = toPctSeries(casted[2].values);
 trendChart.update();
}

// ========================================
// è¾…åŠ©å‡½æ•°
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

// å¯¼å‡ºåˆ·æ–°å‡½æ•°
window.refreshCryptoData = refreshData;

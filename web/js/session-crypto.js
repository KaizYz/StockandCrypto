// ========================================
// StockandCrypto - Crypto Session 页面脚本
// ========================================
document.addEventListener('DOMContentLoaded', function() {
	initSessionCryptoPage();
});

let hourlyChart = null;

// ========================================
// 初始化
// ========================================
async function initSessionCryptoPage() {
	// 显示加载状态
	showLoadingState();
	
	// 加载时段数据
	await loadSessionData();
	
	// 初始化小时网格
	initHourlyGrid();
	
	// 初始化图表
	initCharts();
	
	// 更新当前时段状态
	updateSessionStatus();
	
	// 定时更新
	startUpdates();
	
	// 隐藏加载状态
	hideLoadingState();
}

// ========================================
// 显示/隐藏加载状态
// ========================================
function showLoadingState() {
	const grid = document.getElementById('hourlyGrid');
	if (grid) {
		grid.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><p>加载数据中...</p></div>';
	}
}

function hideLoadingState() {
	// 加载完成后会重新渲染内容
}

function showError(message) {
	const grid = document.getElementById('hourlyGrid');
	if (grid) {
		grid.innerHTML = `<div class="error-message"><p>❌ ${message}</p><button onclick="initSessionCryptoPage()" class="btn btn-secondary">重试</button></div>`;
	}
}

// ========================================
// 加载时段数据
// ========================================
async function loadSessionData() {
	try {
		const data = await api.session.getCrypto();
		if (data) {
			updateSessionsFromAPI(data);
		}
	} catch (error) {
		console.error('Failed to load session data:', error);
		// 使用模拟数据
	}
}

// ========================================
// 从API更新时段数据
// ========================================
function updateSessionsFromAPI(data) {
	// 更新亚洲时段
	if (data.asian) {
		updateSessionCard('asianSession', data.asian);
	}
	// 更新欧洲时段
	if (data.european) {
		updateSessionCard('europeanSession', data.european);
	}
	// 更新美国时段
	if (data.american) {
		updateSessionCard('americanSession', data.american);
	}
	// 更新总结
	if (data.summary) {
		updateSummary(data.summary);
	}
}

function updateSessionCard(sessionId, data) {
	const session = document.getElementById(sessionId);
	if (!session) return;
	
	// 更新波动预测
	const predictionBar = session.querySelector('.bar-fill');
	if (predictionBar && data.volatility) {
		predictionBar.style.width = data.volatility + '%';
	}
	
	// 更新预测方向
	const directionEl = session.querySelector('.detail-value');
	if (directionEl && data.direction) {
		directionEl.textContent = data.direction;
		directionEl.className = 'detail-value ' + (data.direction.includes('涨') ? 'bullish' : data.direction.includes('跌') ? 'bearish' : '');
	}
}

function updateSummary(summary) {
	const overallTrend = document.getElementById('overallTrend');
	const bestSession = document.getElementById('bestSession');
	const riskSession = document.getElementById('riskSession');
	const confidence = document.getElementById('confidence');
	
	if (overallTrend && summary.overallTrend) {
		overallTrend.textContent = summary.overallTrend;
	}
	if (bestSession && summary.bestSession) {
		bestSession.textContent = summary.bestSession;
	}
	if (riskSession && summary.riskSession) {
		riskSession.textContent = summary.riskSession;
	}
	if (confidence && summary.confidence) {
		confidence.textContent = summary.confidence + '%';
	}
}

// ========================================
// 初始化小时网格
// ========================================
function initHourlyGrid() {
	const grid = document.getElementById('hourlyGrid');
	if (!grid) return;
	
	grid.innerHTML = '';
	
	// 生成24小时预测卡片
	for (let hour = 0; hour < 24; hour++) {
		const prediction = generateHourPrediction(hour);
		const card = createHourCard(hour, prediction);
		grid.appendChild(card);
	}
}

function generateHourPrediction(hour) {
	// 基于时段生成预测
	const isAsianSession = hour >= 8 && hour < 16;
	const isEuropeanSession = hour >= 15 && hour < 23;
	const isAmericanSession = hour >= 21 || hour < 4;
	
	let volatility, direction, confidence;
	
	if (isAmericanSession) {
		volatility = 'high';
		direction = hour % 2 === 0 ? 'bearish' : 'bullish';
		confidence = 70 + Math.random() * 20;
	} else if (isEuropeanSession) {
		volatility = 'medium';
		direction = Math.random() > 0.5 ? 'bullish' : 'bearish';
		confidence = 60 + Math.random() * 25;
	} else if (isAsianSession) {
		volatility = 'low';
		direction = 'neutral';
		confidence = 55 + Math.random() * 30;
	} else {
		volatility = 'low';
		direction = 'neutral';
		confidence = 50 + Math.random() * 20;
	}
	
	return { volatility, direction, confidence };
}

function createHourCard(hour, prediction) {
	const card = document.createElement('div');
	card.className = `hour-card ${prediction.volatility} ${prediction.direction}`;
	
	const timeStr = `${hour.toString().padStart(2, '0')}:00`;
	const directionIcon = prediction.direction === 'bullish' ? '↑' : prediction.direction === 'bearish' ? '↓' : '→';
	const directionText = prediction.direction === 'bullish' ? '看涨' : prediction.direction === 'bearish' ? '看跌' : '震荡';
	
	card.innerHTML = `
		<div class="hour-time">${timeStr}</div>
		<div class="hour-direction">${directionIcon} ${directionText}</div>
		<div class="hour-confidence">${prediction.confidence.toFixed(0)}%</div>
	`;
	
	return card;
}

// ========================================
// 初始化图表
// ========================================
function initCharts() {
	Chart.defaults.color = '#8892a0';
	Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
	Chart.defaults.font.family = 'Inter, sans-serif';
	
	createHourlyChart();
}

function createHourlyChart() {
	const canvas = document.getElementById('hourlyChart');
	if (!canvas) return;
	
	const ctx = canvas.getContext('2d');
	
	// 生成24小时波动数据
	const volatilityData = [];
	for (let i = 0; i < 24; i++) {
		// 美国时段波动最大
		if (i >= 21 || i < 4) {
			volatilityData.push(60 + Math.random() * 30);
		} else if (i >= 15) {
			volatilityData.push(40 + Math.random() * 30);
		} else {
			volatilityData.push(20 + Math.random() * 20);
		}
	}
	
	const labels = [];
	for (let i = 0; i < 24; i++) {
		labels.push(`${i.toString().padStart(2, '0')}:00`);
	}
	
	hourlyChart = new Chart(ctx, {
		type: 'bar',
		data: {
			labels: labels,
			datasets: [{
				label: '波动率',
				data: volatilityData,
				backgroundColor: volatilityData.map(v => 
					v > 70 ? '#ff6b6b' : v > 50 ? '#d4af37' : '#00d4aa'
				),
				borderRadius: 4,
				barThickness: 20
			}]
		},
		options: {
			responsive: true,
			maintainAspectRatio: false,
			plugins: {
				legend: { display: false },
				tooltip: {
					backgroundColor: 'rgba(10, 15, 26, 0.9)',
					titleColor: '#ffffff',
					bodyColor: '#8892a0',
					callbacks: {
						label: (context) => `波动率: ${context.raw.toFixed(1)}%`
					}
				}
			},
			scales: {
				x: {
					grid: { color: 'rgba(255, 255, 255, 0.05)' },
					ticks: { color: '#8892a0', maxRotation: 45, minRotation: 45 }
				},
				y: {
					grid: { color: 'rgba(255, 255, 255, 0.05)' },
					ticks: { 
						color: '#8892a0',
						callback: (value) => value + '%'
					},
					max: 100
				}
			}
		}
	});
}

// ========================================
// 更新时段状态
// ========================================
function updateSessionStatus() {
	const now = new Date();
	const hour = now.getUTCHours();
	
	// 更新亚洲时段
	const asianSession = document.getElementById('asianSession');
	const asianStatus = document.getElementById('asianStatus');
	if (hour >= 0 && hour < 8) {
		asianSession?.classList.add('active');
		if (asianStatus) {
			asianStatus.className = 'session-status active';
			asianStatus.querySelector('.status-text').textContent = '交易中';
		}
	}
	
	// 更新欧洲时段
	const europeanSession = document.getElementById('europeanSession');
	const europeanStatus = document.getElementById('europeanStatus');
	if (hour >= 7 && hour < 15) {
		europeanSession?.classList.add('active');
		if (europeanStatus) {
			europeanStatus.className = 'session-status active';
			europeanStatus.querySelector('.status-text').textContent = '交易中';
		}
	}
	
	// 更新美国时段
	const americanSession = document.getElementById('americanSession');
	const americanStatus = document.getElementById('americanStatus');
	if (hour >= 13 && hour < 21) {
		americanSession?.classList.add('active');
		if (americanStatus) {
			americanStatus.className = 'session-status active';
			americanStatus.querySelector('.status-text').textContent = '交易中';
		}
	}
}

// ========================================
// 定时更新
// ========================================
function startUpdates() {
	// 每分钟更新一次状态
	setInterval(() => {
		updateSessionStatus();
	}, 60000);
	
	// 每小时重新生成预测
	setInterval(() => {
		initHourlyGrid();
	}, 3600000);
}

// ========================================
// 刷新数据
// ========================================
async function refreshData() {
	showLoadingState();
	await loadSessionData();
	initHourlyGrid();
	hideLoadingState();
	if (typeof showToast === 'function') {
		showToast('数据已刷新', 'success');
	}
}

// 导出刷新函数
window.refreshSessionCryptoData = refreshData;

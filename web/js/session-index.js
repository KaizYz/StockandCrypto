// ========================================
// StockandCrypto - Index Session 页面脚本
// ========================================
document.addEventListener('DOMContentLoaded', function() {
	initSessionIndexPage();
});

let sessionChart = null;

// ========================================
// 初始化
// ========================================
async function initSessionIndexPage() {
	// 显示加载状态
	showLoadingState();
	
	// 绑定市场切换事件
	bindMarketSwitch();
	
	// 加载时段数据
	await loadSessionData();
	
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
	const chartContainer = document.querySelector('.chart-container');
	if (chartContainer) {
		chartContainer.classList.add('loading');
	}
}

function hideLoadingState() {
	const chartContainer = document.querySelector('.chart-container');
	if (chartContainer) {
		chartContainer.classList.remove('loading');
	}
}

function showError(message) {
	const container = document.querySelector('.section-padding .container');
	if (container) {
		container.innerHTML = `<div class="error-message"><p>❌ ${message}</p><button onclick="initSessionIndexPage()" class="btn btn-secondary">重试</button></div>`;
	}
}

// ========================================
// 加载时段数据
// ========================================
async function loadSessionData() {
	try {
		const cnData = await api.session.getIndex('cn');
		const usData = await api.session.getIndex('us');
		
		if (cnData) {
			updateCNSessions(cnData);
		}
		if (usData) {
			updateUSSessions(usData);
		}
	} catch (error) {
		console.error('Failed to load session data:', error);
		// 使用模拟数据
	}
}

// ========================================
// 更新A股时段
// ========================================
function updateCNSessions(data) {
	// 可以根据API数据更新时段预测
	if (data.sessions) {
		// 更新各个时段的预测内容
	}
}

// ========================================
// 更新美股时段
// ========================================
function updateUSSessions(data) {
	if (data.sessions) {
		// 更新各个时段的预测内容
	}
}

// ========================================
// 绑定市场切换
// ========================================
function bindMarketSwitch() {
	const marketBtns = document.querySelectorAll('.market-btn');
	marketBtns.forEach(btn => {
		btn.addEventListener('click', (e) => {
			const market = e.target.dataset.market;
			
			// 更新按钮状态
			marketBtns.forEach(b => b.classList.remove('active'));
			e.target.classList.add('active');
			
			// 切换显示内容
			const cnSessions = document.getElementById('cnSessions');
			const usSessions = document.getElementById('usSessions');
			
			if (market === 'cn') {
				cnSessions?.classList.remove('hidden');
				usSessions?.classList.add('hidden');
			} else {
				cnSessions?.classList.add('hidden');
				usSessions?.classList.remove('hidden');
			}
			
			// 更新图表
			updateChart(market);
		});
	});
}

// ========================================
// 初始化图表
// ========================================
function initCharts() {
	Chart.defaults.color = '#8892a0';
	Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
	Chart.defaults.font.family = 'Inter, sans-serif';
	
	createSessionChart();
}

function createSessionChart() {
	const canvas = document.getElementById('sessionChart');
	if (!canvas) return;
	
	const ctx = canvas.getContext('2d');
	
	// 生成时段波动数据
	const cnData = generateSessionVolatility('cn');
	const usData = generateSessionVolatility('us');
	
	const labels = [];
	for (let i = 0; i < 24; i++) {
		labels.push(`${i.toString().padStart(2, '0')}:00`);
	}
	
	sessionChart = new Chart(ctx, {
		type: 'line',
		data: {
			labels: labels,
			datasets: [
				{
					label: 'A股',
					data: cnData,
					borderColor: '#00d4aa',
					backgroundColor: 'transparent',
					tension: 0.4,
					borderWidth: 2,
					pointRadius: 0,
					fill: false
				},
				{
					label: '美股',
					data: usData,
					borderColor: '#d4af37',
					backgroundColor: 'transparent',
					tension: 0.4,
					borderWidth: 2,
					pointRadius: 0,
					fill: false
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
					padding: 12,
					callbacks: {
						label: (context) => `${context.dataset.label}: ${context.raw.toFixed(1)}%`
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
					min: 0,
					max: 100
				}
			},
			interaction: {
				intersect: false,
				mode: 'index'
			}
		}
	});
}

function generateSessionVolatility(market) {
	const data = [];
	
	for (let i = 0; i < 24; i++) {
		if (market === 'cn') {
			// A股交易时段: 9:30-11:30, 13:00-15:00 (北京时间)
			if (i >= 1 && i < 4) {
				// 早盘
				data.push(60 + Math.random() * 30);
			} else if (i >= 5 && i < 7) {
				// 午盘
				data.push(40 + Math.random() * 20);
			} else {
				data.push(10 + Math.random() * 10); // 非交易时段
			}
		} else {
			// 美股交易时段: 21:30-04:00 (北京时间)
			if (i >= 13 && i < 21) {
				// 美股开盘到收盘
				if (i < 15) {
					// 开盘时段波动大
					data.push(70 + Math.random() * 25);
				} else if (i > 18) {
					// 收盘时段
					data.push(50 + Math.random() * 30);
				} else {
					data.push(40 + Math.random() * 20);
				}
			} else {
				data.push(15 + Math.random() * 15); // 非交易时段
			}
		}
	}
	
	return data;
}

function updateChart(market) {
	if (!sessionChart) return;
	
	const cnData = generateSessionVolatility('cn');
	const usData = generateSessionVolatility('us');
	
	sessionChart.data.datasets[0].data = cnData;
	sessionChart.data.datasets[1].data = usData;
	sessionChart.update();
}

// ========================================
// 更新时段状态
// ========================================
function updateSessionStatus() {
	const now = new Date();
	const hour = now.getHours();
	
	// 更新A股时段状态
	const cnSessions = document.querySelectorAll('#cnSessions .session-card');
	cnSessions.forEach((card, index) => {
		card.classList.remove('active');
	});
	
	// A股早盘: 9:30-11:30
	if (hour >= 9 && hour < 12) {
		cnSessions[1]?.classList.add('active');
	}
	// A股午盘: 13:00-15:00
	else if (hour >= 13 && hour < 15) {
		cnSessions[2]?.classList.add('active');
	}
	
	// 更新美股时段状态
	const usSessions = document.querySelectorAll('#usSessions .session-card');
	usSessions.forEach((card, index) => {
		card.classList.remove('active');
	});
	
	// 美股交易时段: 21:30-04:00 (北京时间)
	if (hour >= 21 || hour < 5) {
		if (hour >= 21 && hour < 23) {
			usSessions[1]?.classList.add('active');
		} else {
			usSessions[2]?.classList.add('active');
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
}

// ========================================
// 刷新数据
// ========================================
async function refreshData() {
	showLoadingState();
	await loadSessionData();
	hideLoadingState();
	if (typeof showToast === 'function') {
		showToast('数据已刷新', 'success');
	}
}

// 导出刷新函数
window.refreshSessionIndexData = refreshData;

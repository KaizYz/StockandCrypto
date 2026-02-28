// ========================================
// StockandCrypto - Utility Functions
// ========================================

// ========================================
// Toast Notification
// ========================================
function showToast(message, type = 'info') {
	const toast = document.getElementById('toast');
	if (!toast) return;
	
	toast.textContent = message;
	toast.className = `toast ${type} show`;
	
	setTimeout(() => {
		toast.classList.remove('show');
	}, 3000);
}

// ========================================
// Format Number
// ========================================
function formatNumber(num, decimals = 2) {
	if (num === null || num === undefined) return '--';
	return num.toLocaleString('en-US', {
		minimumFractionDigits: decimals,
		maximumFractionDigits: decimals
	});
}

// ========================================
// Format Currency
// ========================================
function formatCurrency(num, symbol = '$', decimals = 2) {
	if (num === null || num === undefined) return '--';
	return `${symbol}${formatNumber(num, decimals)}`;
}

// ========================================
// Format Percent
// ========================================
function formatPercent(num, decimals = 2) {
	if (num === null || num === undefined) return '--';
	const sign = num >= 0 ? '+' : '';
	return `${sign}${num.toFixed(decimals)}%`;
}

// ========================================
// Debounce Function
// ========================================
function debounce(func, wait) {
	let timeout;
	return function executedFunction(...args) {
		const later = () => {
			clearTimeout(timeout);
			func(...args);
		};
		clearTimeout(timeout);
		timeout = setTimeout(later, wait);
	};
}

// ========================================
// Local Storage Helpers
// ========================================
const Storage = {
	get(key, defaultValue = null) {
		try {
			const item = localStorage.getItem(key);
			return item ? JSON.parse(item) : defaultValue;
		} catch (e) {
			console.error('Error reading from localStorage:', e);
			return defaultValue;
		}
	},
	
	set(key, value) {
		try {
			localStorage.setItem(key, JSON.stringify(value));
			return true;
		} catch (e) {
			console.error('Error writing to localStorage:', e);
			return false;
		}
	},
	
	remove(key) {
		try {
			localStorage.removeItem(key);
			return true;
		} catch (e) {
			console.error('Error removing from localStorage:', e);
			return false;
		}
	},
	
	clear() {
		try {
			localStorage.clear();
			return true;
		} catch (e) {
			console.error('Error clearing localStorage:', e);
			return false;
		}
	}
};

// ========================================
// Date Helpers
// ========================================
const DateUtils = {
	format(date, format = 'YYYY-MM-DD') {
		const d = new Date(date);
		const year = d.getFullYear();
		const month = String(d.getMonth() + 1).padStart(2, '0');
		const day = String(d.getDate()).padStart(2, '0');
		const hours = String(d.getHours()).padStart(2, '0');
		const minutes = String(d.getMinutes()).padStart(2, '0');
		const seconds = String(d.getSeconds()).padStart(2, '0');
		
		return format
			.replace('YYYY', year)
			.replace('MM', month)
			.replace('DD', day)
			.replace('HH', hours)
			.replace('mm', minutes)
			.replace('ss', seconds);
	},
	
	isToday(date) {
		const today = new Date();
		const d = new Date(date);
		return d.toDateString() === today.toDateString();
	},
	
	getTimeAgo(date) {
		const now = new Date();
		const d = new Date(date);
		const seconds = Math.floor((now - d) / 1000);
		
		const intervals = {
			年: 31536000,
			月: 2592000,
			周: 604800,
			天: 86400,
			小时: 3600,
			分钟: 60
		};
		
		for (const [unit, secondsInUnit] of Object.entries(intervals)) {
			const interval = Math.floor(seconds / secondsInUnit);
			if (interval >= 1) {
				return `${interval}${unit}前`;
			}
		}
		
		return '刚刚';
	}
};

// ========================================
// Export Global Functions
// ========================================
window.showToast = showToast;
window.formatNumber = formatNumber;
window.formatCurrency = formatCurrency;
window.formatPercent = formatPercent;
window.debounce = debounce;
window.Storage = Storage;
window.DateUtils = DateUtils;

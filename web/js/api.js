// ========================================
// StockandCrypto - API 封装
// ========================================

// API 基础地址 - 根据环境自动切换
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://127.0.0.1:5001/api'  // 开发环境：直接连接后端 API
  : window.location.origin + '/api';  // 生产环境：同源访问

// API 请求封装
const api = {
    // 基础请求方法
    async request(endpoint, options = {}) {
        const url = API_BASE_URL + endpoint;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        // 添加认证 token
        const token = localStorage.getItem('token');
        if (token) {
            defaultOptions.headers['Authorization'] = `Bearer ${token}`;
        }

        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || '请求失败');
            }

            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },

    // GET 请求
    async get(endpoint) {
        return this.request(endpoint, { method: 'GET' });
    },

    // POST 请求
    async post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data),
        });
    },

    // PUT 请求
    async put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data),
        });
    },

    // DELETE 请求
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    },

    // ========================================
    // 认证相关 API
    // ========================================
    auth: {
        // 用户注册
        async register(userData) {
            return api.post('/auth/register', userData);
        },

        // 用户登录
        async login(credentials) {
            return api.post('/auth/login', credentials);
        },

        // 获取当前用户信息
        async getCurrentUser() {
            return api.get('/auth/me');
        },

        // 退出登录
        async logout() {
            return api.post('/auth/logout', {});
        },

        // 刷新 token
        async refreshToken() {
            return api.post('/auth/refresh', {});
        },
    },

    // ========================================
    // 市场数据 API
    // ========================================
    market: {
        // 获取市场概览
        async getOverview() {
            return api.get('/market/overview');
        },

        // 获取加密货币价格
        async getCryptoPrices(symbols = ['BTC', 'ETH', 'SOL']) {
            return api.get(`/market/crypto?symbols=${symbols.join(',')}`);
        },

        // 获取股票价格
        async getStockPrices(symbols) {
            return api.get(`/market/stocks?symbols=${symbols.join(',')}`);
        },

        // 获取指数数据
        async getIndices(market = 'all') {
            return api.get(`/market/indices?market=${market}`);
        },

        // 获取历史数据
        async getHistory(symbol, period = '1D') {
            return api.get(`/market/history?symbol=${symbol}&period=${period}`);
        },
    },

    // ========================================
    // 预测相关 API
    // ========================================
    predictions: {
        // 获取预测摘要
        async getSummary(market = 'crypto') {
            return api.get(`/predictions/summary?market=${market}`);
        },

        // 获取单个资产预测
        async getAssetPrediction(symbol) {
            return api.get(`/predictions/asset/${symbol}`);
        },

        // 获取时段预测
        async getSessionPrediction(market = 'crypto') {
            return api.get(`/predictions/session?market=${market}`);
        },

        // 获取历史预测准确率
        async getAccuracy(symbol) {
            return api.get(`/predictions/accuracy/${symbol}`);
        },
    },

    // ========================================
    // 回测相关 API
    // ========================================
    backtest: {
        // 运行回测
        async run(params) {
            return api.post('/backtest/run', params);
        },

        // 获取回测结果
        async getResult(id) {
            return api.get(`/backtest/result/${id}`);
        },

        // 获取回测历史
        async getHistory() {
            return api.get('/backtest/history');
        },
    },
};

// ========================================
// 扩展 API 端点 - 直接路径
// ========================================
const directApi = {
	// 加密货币 API
	crypto: {
		// 获取预测数据
		async getPredictions() {
			return api.get('/crypto/predictions');
		},
		// 获取支持的币种列表
		async getSymbols() {
			return api.get('/crypto/symbols');
		},
		// 获取单个币种预测
		async getPrediction(symbol) {
			return api.get(`/crypto/prediction/${symbol}`);
		}
	},
	
	// A股 API
	cn: {
		// 获取预测数据
		async getPredictions() {
			return api.get('/cn/predictions');
		},
		// 获取支持的股票列表
		async getSymbols() {
			return api.get('/cn/symbols');
		},
		// 获取单个股票预测
		async getPrediction(symbol) {
			return api.get(`/cn/prediction/${symbol}`);
		}
	},
	
	// 美股 API
	us: {
		// 获取预测数据
		async getPredictions() {
			return api.get('/us/predictions');
		},
		// 获取支持的股票列表
		async getSymbols() {
			return api.get('/us/symbols');
		},
		// 获取单个股票预测
		async getPrediction(symbol) {
			return api.get(`/us/prediction/${symbol}`);
		}
	},
	
	// 时段预测 API
	session: {
		// 加密货币时段预测
		async getCrypto() {
			return api.get('/session/crypto');
		},
		// 指数时段预测
		async getIndex(market = 'cn') {
			return api.get(`/session/index?market=${market}`);
		}
	}
};

// 合并到 api 对象
Object.assign(api, directApi);

// ========================================
// Token 管理
// ========================================
const TokenManager = {
    TOKEN_KEY: 'token',
    REFRESH_TOKEN_KEY: 'refreshToken',
    USER_KEY: 'user',

    // 保存 token
    saveTokens(token, refreshToken = null) {
        localStorage.setItem(this.TOKEN_KEY, token);
        if (refreshToken) {
            localStorage.setItem(this.REFRESH_TOKEN_KEY, refreshToken);
        }
    },

    // 获取 token
    getToken() {
        return localStorage.getItem(this.TOKEN_KEY);
    },

    // 获取 refresh token
    getRefreshToken() {
        return localStorage.getItem(this.REFRESH_TOKEN_KEY);
    },

    // 检查是否已登录
    isAuthenticated() {
        return !!this.getToken();
    },

    // 保存用户信息
    saveUser(user) {
        localStorage.setItem(this.USER_KEY, JSON.stringify(user));
    },

    // 获取用户信息
    getUser() {
        const user = localStorage.getItem(this.USER_KEY);
        return user ? JSON.parse(user) : null;
    },

    // 清除所有认证信息
    clear() {
        localStorage.removeItem(this.TOKEN_KEY);
        localStorage.removeItem(this.REFRESH_TOKEN_KEY);
        localStorage.removeItem(this.USER_KEY);
    },

    // 检查 token 是否过期
    isTokenExpired() {
        const token = this.getToken();
        if (!token) return true;

        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            const exp = payload.exp * 1000;
            return Date.now() >= exp;
        } catch {
            return true;
        }
    },
};

// ========================================
// 错误处理
// ========================================
const ErrorHandler = {
    handle(error) {
        console.error('Error:', error);
        
        // 根据错误类型处理
        if (error.message.includes('401') || error.message.includes('Unauthorized')) {
            TokenManager.clear();
            window.location.href = '/login.html';
            return;
        }

        if (error.message.includes('403') || error.message.includes('Forbidden')) {
            showToast('您没有权限执行此操作', 'error');
            return;
        }

        if (error.message.includes('404') || error.message.includes('Not Found')) {
            showToast('请求的资源不存在', 'error');
            return;
        }

        if (error.message.includes('500') || error.message.includes('Internal Server Error')) {
            showToast('服务器错误，请稍后重试', 'error');
            return;
        }

        showToast(error.message || '发生未知错误', 'error');
    },
};

// ========================================
// Toast 通知
// ========================================
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.getElementById('toast');
    if (!toast) return;

    toast.textContent = message;
    toast.className = `toast ${type} show`;

    setTimeout(() => {
        toast.classList.remove('show');
    }, duration);
}

// 导出
window.api = api;
window.TokenManager = TokenManager;
window.ErrorHandler = ErrorHandler;
window.showToast = showToast;

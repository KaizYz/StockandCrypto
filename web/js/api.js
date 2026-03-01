// ========================================
// StockandCrypto - API Client
// ========================================

const API_BASE_URL = (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")
    ? "http://127.0.0.1:5001/api"
    : `${window.location.origin}/api`;

function toArray(payload, candidateKeys = []) {
    if (Array.isArray(payload)) return payload;
    if (!payload || typeof payload !== "object") return [];
    for (const key of candidateKeys) {
        if (Array.isArray(payload[key])) return payload[key];
    }
    return [];
}

function normalizeSignal(payload) {
    if (!payload || typeof payload !== "object") return null;
    const signal = payload.signal && typeof payload.signal === "object" ? payload.signal : payload;
    if (signal.confidence !== undefined && signal.confidence !== null && signal.confidence <= 1) {
        signal.confidence = signal.confidence * 100;
    }
    if (signal.confidence_score !== undefined && signal.confidence_score !== null && signal.confidence_score <= 1) {
        signal.confidence_score = signal.confidence_score * 100;
    }
    return signal;
}

function normalizeOverview(payload) {
    if (!payload || typeof payload !== "object") return payload;
    if (payload.btc || payload.eth || payload.sol) return payload;

    const assets = toArray(payload, ["assets", "items", "predictions"]);
    const mapped = {};
    for (const item of assets) {
        const sym = String(item.symbol || "").toUpperCase();
        let key = "";
        if (sym.startsWith("BTC")) key = "btc";
        if (sym.startsWith("ETH")) key = "eth";
        if (sym.startsWith("SOL")) key = "sol";
        if (!key) continue;
        const price = item.current_price ?? null;
        const q50 = item.predicted_change_pct ?? null;
        mapped[key] = {
            symbol: sym,
            name: item.name || key.toUpperCase(),
            price: price,
            change: item.change_percent ?? (q50 !== null ? q50 * 100 : 0),
            high: item.resistance_level ?? item.target_price ?? price,
            low: item.support_level ?? price,
            volume: item.sample_size ?? "--",
            predicted: item.predicted_price ?? item.target_price ?? null,
            support: item.support_level ?? null,
            resistance: item.resistance_level ?? null,
        };
    }
    return { ...payload, ...mapped };
}

const api = {
    async request(endpoint, options = {}) {
        const url = `${API_BASE_URL}${endpoint}`;
        const headers = { "Content-Type": "application/json" };
        const token = localStorage.getItem("token");
        if (token) headers.Authorization = `Bearer ${token}`;

        const config = { ...options, headers: { ...headers, ...(options.headers || {}) } };
        let response;
        try {
            response = await fetch(url, config);
        } catch (error) {
            throw new Error(`Network error: ${error.message}`);
        }

        let data = {};
        try {
            data = await response.json();
        } catch (_) {
            data = {};
        }

        if (!response.ok) {
            throw new Error(data.error || data.message || `HTTP ${response.status}`);
        }
        return data;
    },

    get(endpoint) {
        return this.request(endpoint, { method: "GET" });
    },
    post(endpoint, data) {
        return this.request(endpoint, { method: "POST", body: JSON.stringify(data || {}) });
    },
    put(endpoint, data) {
        return this.request(endpoint, { method: "PUT", body: JSON.stringify(data || {}) });
    },
    delete(endpoint) {
        return this.request(endpoint, { method: "DELETE" });
    },

    auth: {
        register(userData) {
            return api.post("/auth/register", userData);
        },
        login(credentials) {
            return api.post("/auth/login", credentials);
        },
        getCurrentUser() {
            return api.get("/auth/me");
        },
        logout() {
            return api.post("/auth/logout", {});
        },
        refreshToken() {
            return api.post("/auth/refresh", {});
        },
    },

    notes: {
        create(payload) {
            return api.post("/notes", payload || {});
        },
        listMine(params = {}) {
            const qs = new URLSearchParams();
            Object.entries(params || {}).forEach(([k, v]) => {
                if (v === undefined || v === null || v === "") return;
                qs.set(k, String(v));
            });
            const suffix = qs.toString() ? `?${qs.toString()}` : "";
            return api.get(`/notes${suffix}`);
        },
        listPublic(params = {}) {
            const qs = new URLSearchParams();
            Object.entries(params || {}).forEach(([k, v]) => {
                if (v === undefined || v === null || v === "") return;
                qs.set(k, String(v));
            });
            const suffix = qs.toString() ? `?${qs.toString()}` : "";
            return api.get(`/notes/public${suffix}`);
        },
    },

    market: {
        async getOverview() {
            const data = await api.get("/market/overview");
            return normalizeOverview(data);
        },
        async getCryptoPrices(symbols = ["BTC", "ETH", "SOL"]) {
            const data = await api.get(`/market/crypto?symbols=${symbols.join(",")}`);
            return toArray(data, ["items", "predictions", "assets"]);
        },
        async getStockPrices(symbols = []) {
            const data = await api.get(`/market/stocks?symbols=${symbols.join(",")}`);
            return toArray(data, ["items", "predictions", "assets"]);
        },
        async getIndices(market = "all") {
            return api.get(`/market/indices?market=${market}`);
        },
        async getHistory(symbol, options = "1D") {
            let period = "1D";
            let interval = "";
            let limit = "";

            if (typeof options === "string") {
                period = options;
            } else if (options && typeof options === "object") {
                period = options.period || period;
                interval = options.interval || "";
                limit = options.limit ?? "";
            }

            const periodMap = {
                "1H": { interval: "hourly", limit: 24 },
                "1D": { interval: "hourly", limit: 48 },
                "1W": { interval: "daily", limit: 7 },
                "1M": { interval: "daily", limit: 30 },
                "1Y": { interval: "daily", limit: 365 },
            };
            const mapped = periodMap[String(period || "").toUpperCase()] || periodMap["1D"];
            if (!interval) interval = mapped.interval;
            if (limit === "") limit = mapped.limit;

            const qs = new URLSearchParams({
                symbol: String(symbol || ""),
                period: String(period || "1D"),
                interval: String(interval),
                limit: String(limit),
            });
            return api.get(`/market/history?${qs.toString()}`);
        },
    },

    predictions: {
        getSummary(market = "crypto") {
            return api.get(`/predictions/summary?market=${market}`);
        },
        getAssetPrediction(symbol) {
            return api.get(`/predictions/asset/${encodeURIComponent(symbol)}`);
        },
        getSessionPrediction(market = "crypto") {
            return api.get(`/predictions/session?market=${market}`);
        },
        getAccuracy(symbol) {
            return api.get(`/predictions/accuracy/${encodeURIComponent(symbol)}`);
        },
    },

    backtest: {
        run(params) {
            return api.post("/backtest/run", params);
        },
        getResult(id) {
            return api.get(`/backtest/result/${encodeURIComponent(id)}`);
        },
        getHistory() {
            return api.get("/backtest/history");
        },
    },

    tracking: {
        getOverview(params = {}) {
            const qs = new URLSearchParams();
            Object.entries(params || {}).forEach(([k, v]) => {
                if (v === undefined || v === null || v === "") return;
                qs.set(k, String(v));
            });
            const suffix = qs.toString() ? `?${qs.toString()}` : "";
            return api.get(`/tracking/overview${suffix}`);
        },
        getDetail(trackKey, params = {}) {
            const key = encodeURIComponent(String(trackKey || "").trim());
            const qs = new URLSearchParams();
            Object.entries(params || {}).forEach(([k, v]) => {
                if (v === undefined || v === null || v === "") return;
                qs.set(k, String(v));
            });
            const suffix = qs.toString() ? `?${qs.toString()}` : "";
            return api.get(`/tracking/detail/${key}${suffix}`);
        },
    },

    execution: {
        getOverview(params = {}) {
            const qs = new URLSearchParams();
            Object.entries(params || {}).forEach(([k, v]) => {
                if (v === undefined || v === null || v === "") return;
                qs.set(k, String(v));
            });
            const suffix = qs.toString() ? `?${qs.toString()}` : "";
            return api.get(`/execution/overview${suffix}`);
        },
        clearLogs() {
            return api.post("/execution/clear-logs", {});
        },
    },
};

const directApi = {
    crypto: {
        async getPredictions() {
            const data = await api.get("/crypto/predictions");
            return toArray(data, ["predictions", "items"]);
        },
        async getSymbols() {
            const data = await api.get("/crypto/symbols");
            return toArray(data, ["symbols", "items"]);
        },
        async getPrediction(symbol) {
            const data = await api.get(`/crypto/prediction/${encodeURIComponent(symbol)}`);
            return normalizeSignal(data);
        },
    },

    cn: {
        async getPredictions() {
            const data = await api.get("/cn/predictions");
            return toArray(data, ["predictions", "items"]);
        },
        async getSymbols() {
            const data = await api.get("/cn/symbols");
            return toArray(data, ["symbols", "items"]);
        },
        async getPrediction(symbol) {
            const data = await api.get(`/cn/prediction/${encodeURIComponent(symbol)}`);
            return normalizeSignal(data);
        },
    },

    us: {
        async getPredictions() {
            const data = await api.get("/us/predictions");
            return toArray(data, ["predictions", "items"]);
        },
        async getSymbols() {
            const data = await api.get("/us/symbols");
            return toArray(data, ["symbols", "items"]);
        },
        async getPrediction(symbol) {
            const data = await api.get(`/us/prediction/${encodeURIComponent(symbol)}`);
            return normalizeSignal(data);
        },
    },

    session: {
        getCrypto(params = {}) {
            const qs = new URLSearchParams();
            Object.entries(params || {}).forEach(([k, v]) => {
                if (v === undefined || v === null || v === "") return;
                qs.set(k, String(v));
            });
            const suffix = qs.toString() ? `?${qs.toString()}` : "";
            return api.get(`/session/crypto${suffix}`);
        },
        getIndex(marketOrParams = "cn", maybeParams = {}) {
            const params = typeof marketOrParams === "object"
                ? { ...(marketOrParams || {}) }
                : { market: String(marketOrParams || "cn"), ...(maybeParams || {}) };
            const qs = new URLSearchParams();
            Object.entries(params || {}).forEach(([k, v]) => {
                if (v === undefined || v === null || v === "") return;
                qs.set(k, String(v));
            });
            const suffix = qs.toString() ? `?${qs.toString()}` : "";
            return api.get(`/session/index${suffix}`);
        },
    },
};

Object.assign(api, directApi);

const TokenManager = {
    TOKEN_KEY: "token",
    REFRESH_TOKEN_KEY: "refreshToken",
    USER_KEY: "user",

    saveTokens(token, refreshToken = null) {
        localStorage.setItem(this.TOKEN_KEY, token);
        if (refreshToken) localStorage.setItem(this.REFRESH_TOKEN_KEY, refreshToken);
    },
    getToken() {
        return localStorage.getItem(this.TOKEN_KEY);
    },
    getRefreshToken() {
        return localStorage.getItem(this.REFRESH_TOKEN_KEY);
    },
    isAuthenticated() {
        return !!this.getToken();
    },
    saveUser(user) {
        localStorage.setItem(this.USER_KEY, JSON.stringify(user));
    },
    getUser() {
        const raw = localStorage.getItem(this.USER_KEY);
        if (!raw) return null;
        try {
            return JSON.parse(raw);
        } catch (_) {
            return null;
        }
    },
    clear() {
        localStorage.removeItem(this.TOKEN_KEY);
        localStorage.removeItem(this.REFRESH_TOKEN_KEY);
        localStorage.removeItem(this.USER_KEY);
    },
    isTokenExpired() {
        const token = this.getToken();
        if (!token) return true;
        try {
            const payload = JSON.parse(atob(token.split(".")[1]));
            return Date.now() >= payload.exp * 1000;
        } catch (_) {
            return true;
        }
    },
};

const ErrorHandler = {
    handle(error) {
        const message = String(error?.message || "Unknown error");
        console.error("API Error:", message);

        if (message.includes("401") || message.includes("authorization_required") || message.includes("invalid_token")) {
            TokenManager.clear();
            if (!window.location.pathname.includes("login.html")) {
                window.location.href = "login.html";
            }
            return;
        }
        if (message.includes("403")) {
            showToast("权限不足", "error");
            return;
        }
        if (message.includes("404")) {
            showToast("请求资源不存在", "error");
            return;
        }
        if (message.includes("500")) {
            showToast("服务器错误，请稍后重试", "error");
            return;
        }
        showToast(message, "error");
    },
};

function showToast(message, type = "info", duration = 3000) {
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove("show"), duration);
}

window.api = api;
window.TokenManager = TokenManager;
window.ErrorHandler = ErrorHandler;
window.showToast = showToast;

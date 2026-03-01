// ========================================
// StockandCrypto - Main JavaScript
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all modules
    initNavigation();
    // initParticles(); // Disabled
    initCountUp();
    initScrollReveal();
    // initCharts(); // Disabled
    initHomeLiveMarketCards();
});

const HOME_MARKET_REFRESH_MS = 30000;
const homeSparklineCharts = {};
let homeMarketRefreshTimer = null;

// ========================================
// Navigation
// ========================================
function initNavigation() {
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
        
        // Close menu on link click
        navMenu.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('active');
                navToggle.classList.remove('active');
            });
        });

        // Keep core navigation consistent across all html pages.
        ensureExtendedNav(navMenu);

        // Keep active navigation item in sync with current html page.
        setActiveNavLink(navMenu);
    }
    
    // Header scroll effect
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const header = document.querySelector('.header');
        const currentScroll = window.pageYOffset;
        
        if (currentScroll > 100) {
            header.style.background = 'rgba(10, 15, 26, 0.95)';
        } else {
            header.style.background = 'rgba(10, 15, 26, 0.85)';
        }
        
        lastScroll = currentScroll;
    });
}

function ensureExtendedNav(navMenu) {
    const requiredLinks = [
        { href: "session-crypto.html", label: "Crypto时段" },
        { href: "session-index.html", label: "指数时段" },
        { href: "notes.html", label: "Notes" },
        { href: "tracking.html", label: "Selection/Tracking" },
        { href: "execution.html", label: "Paper Trading" },
    ];
    const existing = new Set(
        Array.from(navMenu.querySelectorAll("a.nav-link"))
            .map((a) => String(a.getAttribute("href") || "").toLowerCase())
            .filter(Boolean)
    );
    requiredLinks.forEach((item) => {
        if (existing.has(item.href.toLowerCase())) return;
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = item.href;
        a.className = "nav-link";
        a.textContent = item.label;
        li.appendChild(a);
        navMenu.appendChild(li);
    });
}

function setActiveNavLink(navMenu) {
    if (!navMenu) return;
    const links = navMenu.querySelectorAll(".nav-link");
    if (!links.length) return;

    const current = (window.location.pathname.split("/").pop() || "index.html").toLowerCase();
    links.forEach((link) => link.classList.remove("active"));

    let matched = null;
    links.forEach((link) => {
        const href = String(link.getAttribute("href") || "").toLowerCase();
        if (href && href === current) {
            matched = link;
        }
    });
    if (!matched && current === "") {
        matched = [...links].find((link) => String(link.getAttribute("href") || "").toLowerCase() === "index.html") || null;
    }
    if (matched) {
        matched.classList.add("active");
    }
}

// ========================================
// Floating Particles
// ========================================
function initParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 8 + 's';
        particle.style.animationDuration = (8 + Math.random() * 4) + 's';
        
        // Random colors
        const colors = ['#00d4aa', '#d4af37', '#ffffff'];
        particle.style.background = colors[Math.floor(Math.random() * colors.length)];
        particle.style.width = (2 + Math.random() * 4) + 'px';
        particle.style.height = particle.style.width;
        
        container.appendChild(particle);
    }
}

// ========================================
// Count Up Animation
// ========================================
function initCountUp() {
    const statValues = document.querySelectorAll('.stat-value[data-count]');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const target = parseFloat(el.dataset.count);
                animateValue(el, 0, target, 2000);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.5 });
    
    statValues.forEach(el => observer.observe(el));
}

function animateValue(el, start, end, duration) {
    const startTime = performance.now();
    const isDecimal = end % 1 !== 0;
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (ease out)
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        const current = start + (end - start) * easeProgress;
        
        if (isDecimal) {
            el.textContent = current.toFixed(1);
        } else {
            el.textContent = Math.floor(current).toLocaleString();
        }
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

// ========================================
// Scroll Reveal
// ========================================
function initScrollReveal() {
    const cards = document.querySelectorAll('.feature-card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const delay = entry.target.dataset.delay || 0;
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, delay);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.2 });
    
    cards.forEach(card => observer.observe(card));
}

// ========================================
// Charts
// ========================================
function initCharts() {
    // Chart.js global configuration
    Chart.defaults.color = '#8892a0';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.font.family = 'Inter, sans-serif';
    
    // Price sparkline charts
    const chartConfig = {
        type: 'line',
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
            },
            elements: {
                point: { radius: 0 },
                line: { tension: 0.4, borderWidth: 2 }
            }
        }
    };
    
    // BTC Chart
    createSparkline('btcChart', generateData(67500, 68000, 20), '#00d4aa');
    
    // ETH Chart
    createSparkline('ethChart', generateData(3400, 3500, 20), '#d4af37');
    
    // SOL Chart
    createSparkline('solChart', generateData(140, 145, 20), '#ff6b6b');
    
    // Market Overview Chart
    createMarketChart();
}

function createSparkline(canvasId, data, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 80);
    gradient.addColorStop(0, color + '40');
    gradient.addColorStop(1, color + '00');
    
    new Chart(ctx, {
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
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            }
        }
    });
}

function createMarketChart() {
    const canvas = document.getElementById('marketChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Generate sample data for 7 days
    const labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    
    const btcData = [65000, 66000, 65800, 67000, 67500, 67200, 67542];
    const ethData = [3300, 3350, 3400, 3380, 3420, 3450, 3456];
    const solData = [135, 138, 140, 142, 141, 143, 142];
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'BTC',
                    data: btcData,
                    borderColor: '#00d4aa',
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: '#00d4aa'
                },
                {
                    label: 'ETH',
                    data: ethData,
                    borderColor: '#d4af37',
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: '#d4af37'
                },
                {
                    label: 'SOL',
                    data: solData,
                    borderColor: '#8892a0',
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointBackgroundColor: '#8892a0'
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
                        callback: function(value) {
                            return '$' + value.toLocaleString();
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
}

function generateData(min, max, points) {
    const data = [];
    let current = min + (max - min) / 2;
    
    for (let i = 0; i < points; i++) {
        const change = (Math.random() - 0.45) * (max - min) * 0.1;
        current = Math.max(min, Math.min(max, current + change));
        data.push(current);
    }
    
    return data;
}

// ========================================
// Home Live Market Cards
// ========================================
function initHomeLiveMarketCards() {
    const hasHomeCards = document.getElementById('btcCard') && document.getElementById('ethCard') && document.getElementById('solCard');
    if (!hasHomeCards) return;
    if (!window.api || !api.market || typeof api.market.getOverview !== 'function') return;

    refreshHomeLiveMarketCards({ refreshHistory: true }).catch((error) => {
        console.error('Failed to initialize home live market cards:', error);
    });

    if (homeMarketRefreshTimer) {
        clearInterval(homeMarketRefreshTimer);
    }
    homeMarketRefreshTimer = setInterval(() => {
        refreshHomeLiveMarketCards({ refreshHistory: true }).catch((error) => {
            console.error('Failed to refresh home live market cards:', error);
        });
    }, HOME_MARKET_REFRESH_MS);
}

async function refreshHomeLiveMarketCards({ refreshHistory = true } = {}) {
    const overview = await api.market.getOverview();
    if (!overview || overview.ok === false) return;

    updateSingleHomeCard('btc', overview.btc, '#00d4aa');
    updateSingleHomeCard('eth', overview.eth, '#d4af37');
    updateSingleHomeCard('sol', overview.sol, '#ff6b6b');

    if (!refreshHistory) return;

    await Promise.all([
        updateHomeSparkline('BTCUSDT', 'btcChart', '#00d4aa'),
        updateHomeSparkline('ETHUSDT', 'ethChart', '#d4af37'),
        updateHomeSparkline('SOLUSDT', 'solChart', '#ff6b6b'),
    ]);
}

function updateSingleHomeCard(key, payload, color) {
    const cardId = `${key}Card`;
    const priceId = `${key}Price`;
    const card = document.getElementById(cardId);
    const priceEl = document.getElementById(priceId);
    if (!card || !priceEl || !payload || typeof payload !== 'object') return;

    const price = toFiniteNumber(payload.price);
    if (price !== null) {
        const prev = toFiniteNumber(priceEl.textContent.replace(/[^0-9.-]/g, ''));
        priceEl.textContent = price.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        });
        if (prev !== null) {
            priceEl.style.color = price >= prev ? '#00d4aa' : '#ff6b6b';
            setTimeout(() => {
                priceEl.style.color = '#ffffff';
            }, 450);
        }
    }

    const changeEl = card.querySelector('.price-change');
    const change = toFiniteNumber(payload.change);
    if (changeEl && change !== null) {
        const sign = change > 0 ? '+' : '';
        changeEl.textContent = `${sign}${change.toFixed(2)}%`;
        changeEl.classList.remove('positive', 'negative');
        changeEl.classList.add(change >= 0 ? 'positive' : 'negative');
    }

    const stats = card.querySelectorAll('.price-stats .stat-val');
    if (stats.length >= 3) {
        const high = toFiniteNumber(payload.high);
        const low = toFiniteNumber(payload.low);
        if (high !== null) stats[0].textContent = `$${formatPriceCompact(high)}`;
        if (low !== null) stats[1].textContent = `$${formatPriceCompact(low)}`;
        stats[2].textContent = formatVolume(payload.volume);
    }

    const fallbackSeries = [price, price, price].filter((x) => toFiniteNumber(x) !== null);
    if (!homeSparklineCharts[`${key}Chart`] && fallbackSeries.length >= 2) {
        upsertHomeSparkline(`${key}Chart`, fallbackSeries, color);
    }
}

async function updateHomeSparkline(symbol, canvasId, color) {
    if (!window.api || !api.market || typeof api.market.getHistory !== 'function') return;
    try {
        const history = await api.market.getHistory(symbol, { period: '1D', interval: 'hourly', limit: 24 });
        const bars = Array.isArray(history?.bars) ? history.bars : [];
        const series = bars
            .map((bar) => toFiniteNumber(bar?.close))
            .filter((v) => v !== null);
        if (series.length >= 2) {
            upsertHomeSparkline(canvasId, series, color);
        }
    } catch (error) {
        console.error(`Failed to update sparkline for ${symbol}:`, error);
    }
}

function upsertHomeSparkline(canvasId, data, color) {
    if (typeof Chart === 'undefined') return;
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const labels = data.map((_, i) => String(i));
    const gradient = ctx.createLinearGradient(0, 0, 0, 80);
    gradient.addColorStop(0, `${color}40`);
    gradient.addColorStop(1, `${color}00`);

    const existing = homeSparklineCharts[canvasId];
    if (existing) {
        existing.data.labels = labels;
        existing.data.datasets[0].data = data;
        existing.data.datasets[0].borderColor = color;
        existing.data.datasets[0].backgroundColor = gradient;
        existing.update('none');
        return;
    }

    homeSparklineCharts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data,
                borderColor: color,
                backgroundColor: gradient,
                fill: true,
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                x: { display: false },
                y: { display: false },
            },
        },
    });
}

function toFiniteNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function formatPriceCompact(value) {
    const n = toFiniteNumber(value);
    if (n === null) return '--';
    if (Math.abs(n) >= 1000) return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
    return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatVolume(volume) {
    if (volume === null || volume === undefined || volume === '') return '--';
    const text = String(volume).trim();
    const n = toFiniteNumber(text.replace(/,/g, ''));
    if (n === null) return text;
    if (n >= 1_000_000_000) return `$${(n / 1_000_000_000).toFixed(1)}B`;
    if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
    return `$${n.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
}

// ========================================
// Smooth Scroll
// ========================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

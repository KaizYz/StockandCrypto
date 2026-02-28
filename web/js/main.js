// ========================================
// StockandCrypto - Main JavaScript
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all modules
    initNavigation();
    initParticles();
    initCountUp();
    initScrollReveal();
    initCharts();
    initPriceAnimation();
});

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
// Price Animation
// ========================================
function initPriceAnimation() {
    // Simulate live price updates
    setInterval(() => {
        updatePrice('btcPrice', 67000, 68000, 2);
        updatePrice('ethPrice', 3400, 3500, 2);
        updatePrice('solPrice', 140, 145, 2);
    }, 3000);
}

function updatePrice(elementId, min, max, decimals) {
    const el = document.getElementById(elementId);
    if (!el) return;
    
    const currentPrice = parseFloat(el.textContent.replace(/,/g, ''));
    const change = (Math.random() - 0.5) * (max - min) * 0.01;
    const newPrice = Math.max(min, Math.min(max, currentPrice + change));
    
    // Animate the change
    el.style.transition = 'color 0.3s ease';
    
    if (newPrice > currentPrice) {
        el.style.color = '#00d4aa';
    } else {
        el.style.color = '#ff6b6b';
    }
    
    el.textContent = newPrice.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
    
    setTimeout(() => {
        el.style.color = '#ffffff';
    }, 500);
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

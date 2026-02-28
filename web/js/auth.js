// ========================================
// StockandCrypto - 认证模块
// ========================================

document.addEventListener('DOMContentLoaded', function() {
    initAuthForms();
    checkAuthStatus();
});

// ========================================
// 认证表单初始化
// ========================================
function initAuthForms() {
    // 注册表单
    const registerForm = document.getElementById('registerForm');
    if (registerForm) {
        registerForm.addEventListener('submit', handleRegister);
    }

    // 登录表单
    const loginForm = document.getElementById('loginForm');
    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }
}

// ========================================
// 注册处理
// ========================================
async function handleRegister(e) {
    e.preventDefault();

    const form = e.target;
    const submitBtn = document.getElementById('submitBtn');
    
    // 获取表单数据
    const formData = {
        username: document.getElementById('username').value.trim(),
        email: document.getElementById('email').value.trim(),
        password: document.getElementById('password').value,
    };

    // 表单验证
    if (!validateRegisterForm(formData)) {
        return;
    }

    // 显示加载状态
    setLoadingState(submitBtn, true);

    try {
        const response = await api.auth.register(formData);
        
        // 注册成功
        showToast('注册成功！正在跳转到登录页面...', 'success');
        
        // 跳转到登录页
        setTimeout(() => {
            window.location.href = 'login.html';
        }, 1500);

    } catch (error) {
        ErrorHandler.handle(error);
        setLoadingState(submitBtn, false);
    }
}

// ========================================
// 登录处理
// ========================================
async function handleLogin(e) {
    e.preventDefault();

    const form = e.target;
    const submitBtn = document.getElementById('submitBtn');
    
    // 获取表单数据
    const formData = {
        username: document.getElementById('username').value.trim(),
        password: document.getElementById('password').value,
        rememberMe: document.getElementById('rememberMe')?.checked || false,
    };

    // 表单验证
    if (!validateLoginForm(formData)) {
        return;
    }

    // 显示加载状态
    setLoadingState(submitBtn, true);

    try {
        const response = await api.auth.login(formData);
        
        // 保存 token
        if (response.token) {
            TokenManager.saveTokens(response.token, response.refreshToken);
            
            // 保存用户信息
            if (response.user) {
                TokenManager.saveUser(response.user);
            }
        }

        showToast('登录成功！正在跳转...', 'success');

        // 跳转到首页或指定页面
        const redirectUrl = getRedirectUrl();
        setTimeout(() => {
            window.location.href = redirectUrl;
        }, 1000);

    } catch (error) {
        ErrorHandler.handle(error);
        setLoadingState(submitBtn, false);
    }
}

// ========================================
// 表单验证
// ========================================
function validateRegisterForm(data) {
    let isValid = true;

    // 清除之前的错误
    clearErrors();

    // 用户名验证
    if (!data.username) {
        showError('usernameError', '请输入用户名');
        isValid = false;
    } else if (data.username.length < 3) {
        showError('usernameError', '用户名至少3个字符');
        isValid = false;
    } else if (data.username.length > 20) {
        showError('usernameError', '用户名最多20个字符');
        isValid = false;
    } else if (!/^[a-zA-Z0-9_\u4e00-\u9fa5]+$/.test(data.username)) {
        showError('usernameError', '用户名只能包含字母、数字、下划线和中文');
        isValid = false;
    }

    // 邮箱验证
    if (!data.email) {
        showError('emailError', '请输入邮箱地址');
        isValid = false;
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(data.email)) {
        showError('emailError', '请输入有效的邮箱地址');
        isValid = false;
    }

    // 密码验证
    if (!data.password) {
        showError('passwordError', '请输入密码');
        isValid = false;
    } else if (data.password.length < 8) {
        showError('passwordError', '密码至少8个字符');
        isValid = false;
    } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(data.password)) {
        showError('passwordError', '密码需包含大小写字母和数字');
        isValid = false;
    }

    // 确认密码验证
    const confirmPassword = document.getElementById('confirmPassword')?.value;
    if (confirmPassword !== data.password) {
        showError('confirmPasswordError', '两次输入的密码不一致');
        isValid = false;
    }

    // 协议勾选验证
    const agreement = document.getElementById('agreement');
    if (agreement && !agreement.checked) {
        showToast('请阅读并同意服务条款和隐私政策', 'error');
        isValid = false;
    }

    return isValid;
}

function validateLoginForm(data) {
    let isValid = true;

    // 清除之前的错误
    clearErrors();

    // 用户名验证
    if (!data.username) {
        showError('usernameError', '请输入用户名或邮箱');
        isValid = false;
    }

    // 密码验证
    if (!data.password) {
        showError('passwordError', '请输入密码');
        isValid = false;
    }

    return isValid;
}

// ========================================
// 错误显示
// ========================================
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = message;
        element.style.display = 'block';
        
        // 添加输入框错误样式
        const input = element.previousElementSibling;
        if (input && input.tagName === 'INPUT') {
            input.classList.add('error');
        }
    }
}

function clearErrors() {
    document.querySelectorAll('.form-error').forEach(el => {
        el.textContent = '';
        el.style.display = 'none';
    });
    
    document.querySelectorAll('input.error').forEach(el => {
        el.classList.remove('error');
    });
}

// ========================================
// 加载状态
// ========================================
function setLoadingState(button, isLoading) {
    if (!button) return;

    if (isLoading) {
        button.disabled = true;
        button.classList.add('loading');
        button.dataset.originalText = button.innerHTML;
        button.innerHTML = `
            <span class="spinner"></span>
            <span>处理中...</span>
        `;
    } else {
        button.disabled = false;
        button.classList.remove('loading');
        if (button.dataset.originalText) {
            button.innerHTML = button.dataset.originalText;
        }
    }
}

// ========================================
// 重定向 URL
// ========================================
function getRedirectUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get('redirect') || 'index.html';
}

// ========================================
// 检查认证状态
// ========================================
function checkAuthStatus() {
    // 如果已登录，且在登录/注册页面，则重定向
    if (TokenManager.isAuthenticated()) {
        const currentPage = window.location.pathname;
        if (currentPage.includes('login.html') || currentPage.includes('register.html')) {
            // 检查 token 是否过期
            if (!TokenManager.isTokenExpired()) {
                window.location.href = 'index.html';
            } else {
                TokenManager.clear();
            }
        }
    }
}

// ========================================
// 退出登录
// ========================================
async function logout() {
    try {
        await api.auth.logout();
    } catch (error) {
        console.error('Logout error:', error);
    } finally {
        TokenManager.clear();
        window.location.href = 'login.html';
    }
}

// 导出
window.logout = logout;

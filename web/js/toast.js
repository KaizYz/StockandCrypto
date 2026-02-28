// ========================================
// StockandCrypto - Toast Utility
// ========================================

// Toast Notification
function showToast(message, type = 'info') {
	const toast = document.getElementById('toast');
	if (!toast) {
		console.log(`[${type.toUpperCase()}] ${message}`);
		return;
	}
	
	toast.textContent = message;
	toast.className = `toast ${type} show`;
	
	setTimeout(() => {
		toast.classList.remove('show');
	}, 3000);
}

// Export globally
window.showToast = showToast;

// ========================================
// StockandCrypto - Notes page
// ========================================

document.addEventListener("DOMContentLoaded", () => {
    initNotesPage();
});

function initNotesPage() {
    bindNotesEvents();
    refreshAuthState();
}

function bindNotesEvents() {
    const loginForm = document.getElementById("notesLoginForm");
    const registerForm = document.getElementById("notesRegisterForm");
    const createForm = document.getElementById("createNoteForm");
    const refreshBtn = document.getElementById("notesRefreshBtn");
    const mineSearchBtn = document.getElementById("mineSearchBtn");
    const logoutBtn = document.getElementById("logoutBtn");

    if (loginForm) {
        loginForm.addEventListener("submit", handleNotesLogin);
    }
    if (registerForm) {
        registerForm.addEventListener("submit", handleNotesRegister);
    }
    if (createForm) {
        createForm.addEventListener("submit", handleCreateNote);
    }
    if (refreshBtn) {
        refreshBtn.addEventListener("click", () => {
            loadNotesData();
        });
    }
    if (mineSearchBtn) {
        mineSearchBtn.addEventListener("click", () => {
            loadMineNotes();
        });
    }
    if (logoutBtn) {
        logoutBtn.addEventListener("click", async () => {
            try {
                await api.auth.logout();
            } catch (_) {
                // no-op
            }
            TokenManager.clear();
            refreshAuthState();
        });
    }
}

async function refreshAuthState() {
    const authPanel = document.getElementById("notesAuthPanel");
    const workspace = document.getElementById("notesWorkspace");
    const loginBtn = document.getElementById("loginBtn");
    const logoutBtn = document.getElementById("logoutBtn");
    const userLabel = document.getElementById("notesUserLabel");

    const hasToken = TokenManager.isAuthenticated() && !TokenManager.isTokenExpired();
    if (!hasToken) {
        if (authPanel) authPanel.style.display = "";
        if (workspace) workspace.style.display = "none";
        if (loginBtn) loginBtn.style.display = "";
        if (logoutBtn) logoutBtn.style.display = "none";
        return;
    }

    let user = TokenManager.getUser();
    try {
        const me = await api.auth.getCurrentUser();
        if (me && me.ok && me.user) {
            user = me.user;
            TokenManager.saveUser(user);
        }
    } catch (error) {
        console.error("Failed to validate auth:", error);
    }

    if (authPanel) authPanel.style.display = "none";
    if (workspace) workspace.style.display = "";
    if (loginBtn) loginBtn.style.display = "none";
    if (logoutBtn) logoutBtn.style.display = "";
    if (userLabel) {
        userLabel.textContent = `已登录：${user?.username || "-"}`;
    }

    await loadNotesData();
}

async function handleNotesLogin(event) {
    event.preventDefault();
    const username = document.getElementById("loginUsername")?.value?.trim() || "";
    const email = document.getElementById("loginEmail")?.value?.trim() || "";
    const password = document.getElementById("loginPassword")?.value || "";

    if (!password || (!username && !email)) {
        showToast("请输入用户名/邮箱和密码", "error");
        return;
    }

    try {
        const payload = { username, email, password };
        const res = await api.auth.login(payload);
        if (res?.token) {
            TokenManager.saveTokens(res.token, res.refreshToken);
        }
        if (res?.user) {
            TokenManager.saveUser(res.user);
        }
        showToast("登录成功", "success");
        await refreshAuthState();
    } catch (error) {
        ErrorHandler.handle(error);
    }
}

async function handleNotesRegister(event) {
    event.preventDefault();
    const username = document.getElementById("regUsername")?.value?.trim() || "";
    const email = document.getElementById("regEmail")?.value?.trim() || "";
    const password = document.getElementById("regPassword")?.value || "";

    if (!username || !email || !password) {
        showToast("请填写完整注册信息", "error");
        return;
    }

    try {
        await api.auth.register({ username, email, password });
        showToast("注册成功，请登录", "success");
    } catch (error) {
        ErrorHandler.handle(error);
    }
}

async function handleCreateNote(event) {
    event.preventDefault();
    const title = document.getElementById("noteTitle")?.value?.trim() || "";
    const content = document.getElementById("noteContent")?.value?.trim() || "";
    const tags = document.getElementById("noteTags")?.value?.trim() || "";
    const noteType = document.getElementById("noteType")?.value || "NOTE";
    const isPublic = !!document.getElementById("notePublic")?.checked;

    if (!title && !content) {
        showToast("标题和内容至少填写一项", "error");
        return;
    }

    try {
        const payload = {
            title,
            content,
            tags,
            note_type: noteType,
            is_public: isPublic,
        };
        await api.notes.create(payload);
        showToast("笔记已创建", "success");
        clearCreateForm();
        await loadNotesData();
    } catch (error) {
        ErrorHandler.handle(error);
    }
}

function clearCreateForm() {
    const title = document.getElementById("noteTitle");
    const content = document.getElementById("noteContent");
    const tags = document.getElementById("noteTags");
    const noteType = document.getElementById("noteType");
    const notePublic = document.getElementById("notePublic");
    if (title) title.value = "";
    if (content) content.value = "";
    if (tags) tags.value = "";
    if (noteType) noteType.value = "NOTE";
    if (notePublic) notePublic.checked = false;
}

async function loadNotesData() {
    await Promise.all([loadMineNotes(), loadPublicNotes()]);
}

async function loadMineNotes() {
    const query = document.getElementById("mineQuery")?.value?.trim() || "";
    const container = document.getElementById("myNotesList");
    if (!container) return;

    try {
        const res = await api.notes.listMine({ mine: "true", q: query, page_size: 20 });
        renderNoteList(container, res?.items || [], { includeAuthor: false });
    } catch (error) {
        container.innerHTML = '<div class="list-empty">加载失败</div>';
        ErrorHandler.handle(error);
    }
}

async function loadPublicNotes() {
    const container = document.getElementById("publicNotesList");
    if (!container) return;
    try {
        const res = await api.notes.listPublic({ page_size: 12 });
        renderNoteList(container, res?.items || [], { includeAuthor: true });
    } catch (error) {
        container.innerHTML = '<div class="list-empty">加载失败</div>';
        ErrorHandler.handle(error);
    }
}

function renderNoteList(container, items, options = {}) {
    const includeAuthor = !!options.includeAuthor;
    container.innerHTML = "";
    if (!Array.isArray(items) || items.length === 0) {
        container.innerHTML = '<div class="list-empty">暂无数据</div>';
        return;
    }

    for (const item of items) {
        const card = document.createElement("article");
        card.className = "note-item";

        const title = document.createElement("h4");
        title.className = "note-title";
        title.textContent = `${item.title || "-"} (#${item.id || "-"})`;
        card.appendChild(title);

        const meta = document.createElement("div");
        meta.className = "note-meta";
        const tags = Array.isArray(item.tags) ? item.tags.join(", ") : "";
        const updatedAt = item.updated_at || "-";
        let metaText = `更新时间: ${updatedAt}`;
        if (tags) metaText += ` | 标签: ${tags}`;
        if (includeAuthor) {
            metaText += ` | 作者: ${item.author?.username || "-"}`;
        }
        meta.textContent = metaText;
        card.appendChild(meta);

        const content = document.createElement("p");
        content.className = "note-content";
        content.textContent = item.content || "";
        card.appendChild(content);

        container.appendChild(card);
    }
}

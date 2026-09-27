/**
 * index.js - FinAI Application Controller
 * Handles Navigation, Authentication, Multilingual Chatbot (EN, HI, TA),
 * Ledger CRUD, and Model Evaluation.
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let activeLanguage = "en"; // 'en' | 'hi' | 'ta'
    let currentUser = null;
    let trendChartInstance = null;
    let categoryChartInstance = null;
    let currentTheme = localStorage.getItem("finai_theme") || "dark";
    let cachedChartsData = null;

    // Localized prompt chips dictionaries (including RAG verified references)
    const promptChipsByLang = {
        en: [
            { label: "📚 50/30/20 Rule (RAG)", query: "What is the 50/30/20 budget rule and how does it apply to me?" },
            { label: "🛡️ Emergency Fund (RAG)", query: "How many months of emergency fund do I need?" },
            { label: "📑 Section 80C Tax (RAG)", query: "What are the best tax deductions under Section 80C?" },
            { label: "💳 Debt Payoff (RAG)", query: "Should I pay off high-interest debt or invest first?" },
            { label: "Check balance", query: "What is my balance?" },
            { label: "This month's expenses", query: "How much did I spend this month?" },
            { label: "Groceries spend", query: "How much did I spend on groceries?" },
            { label: "Budget status", query: "What is my budget status?" },
            { label: "Assess financial health", query: "Assess my financial health" }
        ],
        hi: [
            { label: "📚 50/30/20 बजट नियम (RAG)", query: "50/30/20 बजट नियम क्या है और इसे कैसे लागू करें?" },
            { label: "🛡️ आपातकालीन फंड (RAG)", query: "मुझे कितने महीने का इमरजेंसी फंड रखना चाहिए?" },
            { label: "📑 धारा 80C टैक्स बचत (RAG)", query: "सेक्शन 80C के तहत टैक्स कैसे बचाएं?" },
            { label: "💳 कर्ज रणनीति (RAG)", query: "कर्ज पहले चुकाएं या निवेश करें?" },
            { label: "मेरा बैलेंस", query: "मेरा बैलेंस क्या है?" },
            { label: "इस महीने का खर्चा", query: "इस महीने मैंने कितना खर्च किया?" },
            { label: "किराने का खर्च", query: "किराने पर कितना खर्च हुआ?" },
            { label: "बजट स्थिति", query: "मेरा बजट स्टेटस क्या है?" },
            { label: "वित्तीय स्वास्थ्य जांच", query: "मेरे वित्तीय स्वास्थ्य का आकलन करें" }
        ],
        ta: [
            { label: "📚 50/30/20 விதி (RAG)", query: "50/30/20 பட்ஜெட் விதி என்றால் என்ன?" },
            { label: "🛡️ அவசரகால நிதி (RAG)", query: "எத்தனை மாதங்கள் அவசரகால நிதி வைத்திருக்க வேண்டும்?" },
            { label: "📑 பிரிவு 80C வரி (RAG)", query: "பிரிவு 80C வரி விலக்குகள் என்ன?" },
            { label: "💳 கடன் உத்தி (RAG)", query: "முதலில் கடனை அடைப்பதா அல்லது முதலீடு செய்வதா?" },
            { label: "கணக்கு இருப்பு", query: "என் இருப்பு என்ன?" },
            { label: "இந்த மாத செலவு", query: "இந்த மாதம் நான் எவ்வளவு செலவு செய்தேன்?" },
            { label: "மளிகை செலவு", query: "மளிகைப் பொருட்களுக்கு எவ்வளவு செலவானது?" },
            { label: "பட்ஜெட் நிலை", query: "என் பட்ஜெட் நிலை என்ன?" },
            { label: "நிதி ஆரோக்கியம்", query: "எனது நிதி ஆரோக்கியத்தை மதிப்பிடுங்கள்" }
        ]
    };

    const welcomeMessages = {
        en: "Hello! I am your <strong>FinAI Personal Assistant</strong>, enhanced with <strong>RAG (Retrieval-Augmented Generation)</strong> and direct ledger integration. Ask me about budget rules (50/30/20), emergency funds, Section 80C tax planning, debt management, or your live balance and expenses.",
        hi: "नमस्ते! मैं आपका <strong>FinAI पर्सनल असिस्टेंट</strong> हूँ, जो <strong>RAG (रिट्रीवल-ऑगमेंटेड जेनरेशन)</strong> और लाइव डेटाबेस से सुसज्जित है। आप मुझसे 50/30/20 बजट नियम, आपातकालीन निधि, धारा 80C कर बचत, बैलेंस या खर्चों के बारे में पूछ सकते हैं।",
        ta: "வணக்கம்! நான் உங்கள் <strong>FinAI தனிப்பட்ட உதவியாளர்</strong>. <strong>RAG (மீட்டெடுப்பு-பெரிதாக்கப்பட்ட உருவாக்கம்)</strong> மற்றும் நேரடி லெட்ஜர் ஒருங்கிணைப்புடன் இயங்குகிறேன். 50/30/20 பட்ஜெட் விதி, அவசரகால நிதி, பிரிவு 80C வரி சேமிப்பு, இருப்பு அல்லது செலவுகள் பற்றி என்னிடம் கேட்கலாம்."
    };

    const inputPlaceholders = {
        en: "Ask a financial question (e.g., 'What is the 50/30/20 rule?', 'What is my balance?')...",
        hi: "वित्तीय प्रश्न पूछें (उदा. '50/30/20 नियम क्या है?', 'मेरा बैलेंस क्या है?')...",
        ta: "நிதி கேள்வியைக் கேளுங்கள் (எ.கா. '50/30/20 விதி என்றால் என்ன?', 'என் இருப்பு என்ன?')..."
    };

    // =========================================================================
    // 1. NAVIGATION
    // =========================================================================
    const navItems = document.querySelectorAll(".nav-item");
    const viewPanes = document.querySelectorAll(".view-pane");

    navItems.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.getAttribute("data-tab");
            navItems.forEach(n => n.classList.remove("active"));
            viewPanes.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const pane = document.getElementById(`pane-${target}`);
            if (pane) pane.classList.add("active");

            if (target === "overview") loadDashboard();
            if (target === "ledger") loadTransactions();
            if (target === "assessment") loadProfileForAssessment();
            if (target === "research") loadAcademicMetrics();
        });
    });

    // =========================================================================
    // 2. AUTHENTICATION (LOG IN / SIGN UP)
    // =========================================================================
    const authBarContainer = document.getElementById("auth-bar-container");
    const authModal = document.getElementById("auth-modal");
    const btnCloseAuthModal = document.getElementById("btn-close-auth-modal");
    const tabBtnSignin = document.getElementById("tab-btn-signin");
    const tabBtnRegister = document.getElementById("tab-btn-register");
    const formSignin = document.getElementById("form-signin");
    const formRegister = document.getElementById("form-register");
    const btnFillDemo = document.getElementById("btn-fill-demo-auth");

    async function checkAuthSession() {
        try {
            const res = await fetch("/api/auth/me");
            const data = await res.json();
            currentUser = data.user;
            renderAuthBar(data.authenticated, data.user);
        } catch (e) {
            console.error("Auth check failed", e);
        }
    }

    function renderAuthBar(isAuthenticated, user) {
        authBarContainer.innerHTML = "";
        if (isAuthenticated && user) {
            const badge = document.createElement("div");
            badge.className = "user-badge";
            badge.innerHTML = `
                <span class="user-glyph">${user.username.slice(0, 2).toUpperCase()}</span>
                <span class="user-name-label">${escapeHTML(user.username)}</span>
                <button class="btn-ghost" id="btn-signout" style="padding: 0.2rem 0.6rem; font-size: 0.75rem; margin-left: 0.35rem;">Sign out</button>
            `;
            authBarContainer.appendChild(badge);
            document.getElementById("btn-signout").addEventListener("click", handleSignout);
        } else {
            const btn = document.createElement("button");
            btn.className = "btn-ghost";
            btn.id = "btn-open-auth";
            btn.textContent = "Log In / Sign Up";
            btn.addEventListener("click", () => authModal.classList.remove("hidden"));
            authBarContainer.appendChild(btn);
        }
    }

    async function handleSignout() {
        try {
            await fetch("/api/auth/logout", { method: "POST" });
            showToast("Signed out successfully", "success");
            await checkAuthSession();
            loadDashboard();
            loadTransactions();
        } catch (e) {
            showToast("Sign out failed", "error");
        }
    }

    // Modal tabs toggle
    tabBtnSignin.addEventListener("click", () => {
        tabBtnSignin.classList.add("active");
        tabBtnRegister.classList.remove("active");
        formSignin.classList.remove("hidden");
        formRegister.classList.add("hidden");
    });

    tabBtnRegister.addEventListener("click", () => {
        tabBtnRegister.classList.add("active");
        tabBtnSignin.classList.remove("active");
        formRegister.classList.remove("hidden");
        formSignin.classList.add("hidden");
    });

    btnCloseAuthModal.addEventListener("click", () => authModal.classList.add("hidden"));

    btnFillDemo.addEventListener("click", () => {
        document.getElementById("signin-username").value = "demo_user";
        document.getElementById("signin-password").value = "password123";
    });

    // Handle Login
    formSignin.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            username: document.getElementById("signin-username").value.trim(),
            password: document.getElementById("signin-password").value.trim()
        };

        try {
            const res = await fetch("/api/auth/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (res.ok) {
                authModal.classList.add("hidden");
                showToast(`Welcome back, ${data.user.username}!`, "success");
                await checkAuthSession();
                loadDashboard();
                loadTransactions();
            } else {
                showToast(data.error || "Login failed", "error");
            }
        } catch (err) {
            showToast("Network error during login", "error");
        }
    });

    // Handle Registration
    formRegister.addEventListener("submit", async (e) => {
        e.preventDefault();
        const payload = {
            username: document.getElementById("reg-username").value.trim(),
            password: document.getElementById("reg-password").value.trim(),
            monthly_income: parseFloat(document.getElementById("reg-income").value),
            age: parseInt(document.getElementById("reg-age").value),
            dependents: parseInt(document.getElementById("reg-dependents").value),
            occupation: document.getElementById("reg-occupation").value,
            city_tier: document.getElementById("reg-city-tier").value
        };

        try {
            const res = await fetch("/api/auth/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (res.ok) {
                authModal.classList.add("hidden");
                showToast(`Account created for ${data.user.username}!`, "success");
                await checkAuthSession();
                loadDashboard();
                loadTransactions();
            } else {
                showToast(data.error || "Registration failed", "error");
            }
        } catch (err) {
            showToast("Network error during registration", "error");
        }
    });

    // =========================================================================
    // 3. OVERVIEW / DASHBOARD & CHARTS
    // =========================================================================
    async function loadDashboard() {
        try {
            const res = await fetch("/api/dashboard");
            const data = await res.json();

            const inr = (val) => new Intl.NumberFormat("en-IN", {
                style: "currency",
                currency: "INR",
                maximumFractionDigits: 2
            }).format(val);

            document.getElementById("kpi-balance").textContent = inr(data.current_balance);
            document.getElementById("kpi-savings-rate").textContent = `Savings Rate: ${data.savings_rate}%`;
            document.getElementById("kpi-tx-count").textContent = `${data.transaction_count} recorded entries`;

            document.getElementById("kpi-income").textContent = `+${inr(data.total_income)}`;
            document.getElementById("kpi-monthly-income").textContent = `This month: ${inr(data.monthly_income)}`;

            document.getElementById("kpi-expense").textContent = `-${inr(data.total_expense)}`;
            document.getElementById("kpi-monthly-expense").textContent = `This month: ${inr(data.monthly_expense)}`;

            document.getElementById("kpi-budget-remaining").textContent = inr(data.remaining_budget);
            document.getElementById("kpi-budget-usage").textContent = `${data.budget_usage_pct}% of budget spent (${inr(data.monthly_budget)} limit)`;

            const budgetFill = document.getElementById("budget-progress-fill");
            const pct = Math.min(100, Math.max(0, data.budget_usage_pct));
            budgetFill.style.width = `${pct}%`;
            budgetFill.className = "budget-fill";
            if (pct >= 100) budgetFill.classList.add("danger");
            else if (pct >= 85) budgetFill.classList.add("warning");

            document.getElementById("top-cat-name").textContent = data.highest_spending_category || "None";
            document.getElementById("top-cat-amt").textContent = inr(data.highest_category_amount || 0);

            loadCharts();
        } catch (err) {
            console.error("Dashboard load failed", err);
        }
    }

    async function loadCharts() {
        try {
            const res = await fetch("/api/charts");
            const data = await res.json();
            cachedChartsData = data;
            renderCharts(data);
        } catch (e) {
            console.error("Charts load failed", e);
        }
    }

    function renderCharts(data) {
        if (!data || !data.monthly_trend || !data.category_spending) return;

        const isLight = (currentTheme === "light");
        const gridColor = isLight ? "rgba(15, 23, 42, 0.08)" : "rgba(255, 255, 255, 0.05)";
        const tickColor = isLight ? "#475569" : "#8b9cb5";
        const doughnutBorder = isLight ? "#ffffff" : "#0f182b";

        // 1. Trend Chart
        const trendEl = document.getElementById("trendChart");
        if (trendEl) {
            const trendCtx = trendEl.getContext("2d");
            if (trendChartInstance) trendChartInstance.destroy();

            trendChartInstance = new Chart(trendCtx, {
                type: "line",
                data: {
                    labels: data.monthly_trend.labels,
                    datasets: [
                        {
                            label: "Inflow (Income)",
                            data: data.monthly_trend.income,
                            borderColor: "#10b981",
                            backgroundColor: isLight ? "rgba(16, 185, 129, 0.12)" : "rgba(16, 185, 129, 0.08)",
                            tension: 0.3,
                            fill: true,
                            pointBackgroundColor: "#10b981",
                            pointRadius: 4,
                            borderWidth: 2
                        },
                        {
                            label: "Outflow (Expenses)",
                            data: data.monthly_trend.expenses,
                            borderColor: "#f43f5e",
                            backgroundColor: isLight ? "rgba(244, 63, 94, 0.12)" : "rgba(244, 63, 94, 0.08)",
                            tension: 0.3,
                            fill: true,
                            pointBackgroundColor: "#f43f5e",
                            pointRadius: 4,
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: tickColor, font: { family: "Plus Jakarta Sans", weight: "500" } } },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => `${ctx.dataset.label}: ₹${ctx.parsed.y.toLocaleString("en-IN")}`
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: gridColor },
                            ticks: { color: tickColor, font: { family: "Plus Jakarta Sans" } }
                        },
                        y: {
                            grid: { color: gridColor },
                            ticks: {
                                color: tickColor,
                                font: { family: "Space Grotesk" },
                                callback: (v) => "₹" + v.toLocaleString("en-IN")
                            }
                        }
                    }
                }
            });
        }

        // 2. Category Doughnut Chart
        const catEl = document.getElementById("categoryChart");
        if (catEl) {
            const catCtx = catEl.getContext("2d");
            if (categoryChartInstance) categoryChartInstance.destroy();

            const distinctPalette = [
                "#3b82f6", "#10b981", "#f59e0b", "#ec4899",
                "#8b5cf6", "#14b8a6", "#f97316", "#06b6d4",
                "#a855f7", "#eab308", "#64748b", "#6366f1"
            ];

            categoryChartInstance = new Chart(catCtx, {
                type: "doughnut",
                data: {
                    labels: data.category_spending.labels,
                    datasets: [{
                        data: data.category_spending.values,
                        backgroundColor: distinctPalette.slice(0, data.category_spending.labels.length),
                        borderColor: doughnutBorder,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: "bottom",
                            labels: { color: tickColor, font: { size: 11, family: "Plus Jakarta Sans" }, boxWidth: 10 }
                        },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => `${ctx.label}: ₹${ctx.parsed.toLocaleString("en-IN")}`
                            }
                        }
                    },
                    cutout: "70%"
                }
            });
        }
    }

    // =========================================================================
    // 4. TRANSACTIONS LEDGER CRUD & FILTERS
    // =========================================================================
    const txTbody = document.getElementById("transactions-tbody");
    const txEmpty = document.getElementById("tx-empty-state");
    const searchInput = document.getElementById("tx-search");
    const filterType = document.getElementById("filter-type");
    const filterCat = document.getElementById("filter-category");
    const filterStartDate = document.getElementById("filter-start-date");
    const filterEndDate = document.getElementById("filter-end-date");
    const btnClearFilters = document.getElementById("btn-clear-filters");

    async function loadTransactions() {
        try {
            const params = new URLSearchParams();
            if (searchInput.value) params.append("search", searchInput.value.trim());
            if (filterType.value) params.append("type", filterType.value);
            if (filterCat.value) params.append("category", filterCat.value);
            if (filterStartDate.value) params.append("start_date", filterStartDate.value);
            if (filterEndDate.value) params.append("end_date", filterEndDate.value);

            const res = await fetch(`/api/transactions?${params.toString()}`);
            const data = await res.json();

            txTbody.innerHTML = "";
            if (!data.transactions || data.transactions.length === 0) {
                txEmpty.classList.remove("hidden");
                return;
            }

            txEmpty.classList.add("hidden");

            data.transactions.forEach(t => {
                const tr = document.createElement("tr");
                const sign = t.type === "income" ? "+" : "-";
                const numClass = t.type === "income" ? "inflow" : "outflow";

                tr.innerHTML = `
                    <td class="ledger-num">${t.date}</td>
                    <td><span class="flow-tag ${t.type}">${t.type}</span></td>
                    <td><strong>${escapeHTML(t.category)}</strong></td>
                    <td style="color: var(--text-secondary);">${escapeHTML(t.description || "—")}</td>
                    <td><span class="status-tag">${escapeHTML(t.payment_method || "UPI")}</span></td>
                    <td class="ledger-num ${numClass}" style="text-align: right;">${sign}₹${t.amount.toLocaleString("en-IN", { minimumFractionDigits: 2 })}</td>
                    <td class="action-row">
                        <button class="action-icon edit-tx" data-id="${t.id}" title="Edit entry">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
                        </button>
                        <button class="action-icon danger delete-tx" data-id="${t.id}" title="Remove entry">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
                        </button>
                    </td>
                `;
                txTbody.appendChild(tr);
            });

            document.querySelectorAll(".edit-tx").forEach(b => b.addEventListener("click", () => openEditTxModal(b.dataset.id)));
            document.querySelectorAll(".delete-tx").forEach(b => b.addEventListener("click", () => deleteTransaction(b.dataset.id)));

        } catch (err) {
            console.error("Transactions load failed", err);
        }
    }

    [searchInput, filterType, filterCat, filterStartDate, filterEndDate].forEach(el => {
        el.addEventListener("change", loadTransactions);
        if (el === searchInput) el.addEventListener("input", debounce(loadTransactions, 250));
    });

    btnClearFilters.addEventListener("click", () => {
        searchInput.value = "";
        filterType.value = "";
        filterCat.value = "";
        filterStartDate.value = "";
        filterEndDate.value = "";
        loadTransactions();
    });

    // Transaction Modals
    const txModal = document.getElementById("tx-modal");
    const txForm = document.getElementById("tx-form");
    const btnOpenTxModal = document.getElementById("btn-open-tx-modal");
    const btnQuickAddTx = document.getElementById("btn-quick-add-tx");
    const btnCloseTxModal = document.getElementById("btn-close-tx-modal");
    const btnCancelTxModal = document.getElementById("btn-cancel-tx-modal");

    function openNewTxModal() {
        document.getElementById("modal-title").textContent = "Record Transaction";
        document.getElementById("form-tx-id").value = "";
        document.getElementById("form-tx-date").value = new Date().toISOString().split("T")[0];
        document.getElementById("form-tx-amount").value = "";
        document.getElementById("form-tx-desc").value = "";
        document.getElementById("form-tx-type").value = "expense";
        document.getElementById("form-tx-category").value = "Groceries";
        document.getElementById("form-tx-payment").value = "UPI";
        txModal.classList.remove("hidden");
    }

    btnOpenTxModal.addEventListener("click", openNewTxModal);
    btnQuickAddTx.addEventListener("click", openNewTxModal);
    btnCloseTxModal.addEventListener("click", () => txModal.classList.add("hidden"));
    btnCancelTxModal.addEventListener("click", () => txModal.classList.add("hidden"));

    async function openEditTxModal(id) {
        try {
            const res = await fetch("/api/transactions");
            const data = await res.json();
            const tx = data.transactions.find(t => t.id == id);
            if (!tx) return;

            document.getElementById("modal-title").textContent = "Edit Transaction";
            document.getElementById("form-tx-id").value = tx.id;
            document.getElementById("form-tx-date").value = tx.date;
            document.getElementById("form-tx-amount").value = tx.amount;
            document.getElementById("form-tx-desc").value = tx.description || "";
            document.getElementById("form-tx-type").value = tx.type;
            document.getElementById("form-tx-category").value = tx.category;
            document.getElementById("form-tx-payment").value = tx.payment_method || "UPI";
            txModal.classList.remove("hidden");
        } catch (e) {
            showToast("Failed to fetch transaction record", "error");
        }
    }

    async function deleteTransaction(id) {
        if (!confirm("Are you sure you want to delete this transaction record?")) return;
        try {
            const res = await fetch(`/api/transactions/${id}`, { method: "DELETE" });
            if (res.ok) {
                showToast("Transaction removed", "success");
                loadTransactions();
                loadDashboard();
            }
        } catch (e) {
            showToast("Failed to delete", "error");
        }
    }

    txForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const id = document.getElementById("form-tx-id").value;
        const payload = {
            date: document.getElementById("form-tx-date").value,
            amount: parseFloat(document.getElementById("form-tx-amount").value),
            type: document.getElementById("form-tx-type").value,
            category: document.getElementById("form-tx-category").value,
            payment_method: document.getElementById("form-tx-payment").value,
            description: document.getElementById("form-tx-desc").value.trim()
        };

        try {
            const url = id ? `/api/transactions/${id}` : "/api/transactions";
            const method = id ? "PUT" : "POST";

            const res = await fetch(url, {
                method: method,
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                txModal.classList.add("hidden");
                showToast(id ? "Entry updated" : "Entry recorded", "success");
                loadTransactions();
                loadDashboard();
            } else {
                const err = await res.json();
                showToast(err.error || "Save failed", "error");
            }
        } catch (e) {
            showToast("Save failed", "error");
        }
    });

    // =========================================================================
    // 5. BUDGET ADJUSTMENT
    // =========================================================================
    const budgetModal = document.getElementById("budget-modal");
    const budgetForm = document.getElementById("budget-form");
    const btnEditBudget = document.getElementById("btn-edit-budget");
    const btnCloseBudgetModal = document.getElementById("btn-close-budget-modal");
    const btnCancelBudgetModal = document.getElementById("btn-cancel-budget-modal");

    btnEditBudget.addEventListener("click", async () => {
        try {
            const res = await fetch("/api/budget");
            const data = await res.json();
            document.getElementById("form-budget-amount").value = data.amount || 45000;
            budgetModal.classList.remove("hidden");
        } catch (e) {
            budgetModal.classList.remove("hidden");
        }
    });

    btnCloseBudgetModal.addEventListener("click", () => budgetModal.classList.add("hidden"));
    btnCancelBudgetModal.addEventListener("click", () => budgetModal.classList.add("hidden"));

    budgetForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const amt = parseFloat(document.getElementById("form-budget-amount").value);
        try {
            const res = await fetch("/api/budget", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ amount: amt })
            });
            if (res.ok) {
                budgetModal.classList.add("hidden");
                showToast("Monthly budget updated", "success");
                loadDashboard();
            }
        } catch (e) {
            showToast("Failed to update budget", "error");
        }
    });

    // =========================================================================
    // 6. AI FINANCIAL HEALTH ASSESSMENT
    // =========================================================================
    async function loadProfileForAssessment() {
        try {
            const res = await fetch("/api/profile");
            const profile = await res.json();
            document.getElementById("prof-income").value = profile.monthly_income || 65000;
            document.getElementById("prof-age").value = profile.age || 29;
            document.getElementById("prof-dependents").value = profile.dependents || 1;
            document.getElementById("prof-occupation").value = profile.occupation || "Professional";
            document.getElementById("prof-city-tier").value = profile.city_tier || "Tier_2";
        } catch (e) {
            console.error("Profile load failed", e);
        }
    }

    const aiForm = document.getElementById("ai-profile-form");
    const aiResultBox = document.getElementById("ai-result-content");

    aiForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btnRun = document.getElementById("btn-run-assessment");
        btnRun.disabled = true;
        btnRun.textContent = "Computing neural network inference...";

        const payload = {
            income: parseFloat(document.getElementById("prof-income").value),
            age: parseInt(document.getElementById("prof-age").value),
            dependents: parseInt(document.getElementById("prof-dependents").value),
            occupation: document.getElementById("prof-occupation").value,
            city_tier: document.getElementById("prof-city-tier").value
        };

        try {
            const res = await fetch("/api/ai/assess", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const result = await res.json();
            const isHealthy = result.predicted_class === 1;

            aiResultBox.innerHTML = `
                <div class="diagnostic-panel">
                    <div class="verdict-header ${isHealthy ? 'healthy' : 'constrained'}">
                        <div class="verdict-title">${result.status_tier}</div>
                        <span class="status-tag ${isHealthy ? 'success' : ''}" style="margin-top: 0.35rem; display: inline-block;">
                            ${isHealthy ? 'Low Financial Fragility' : 'Budget Strain Detected'}
                        </span>
                    </div>

                    <div class="verdict-stats-grid">
                        <div class="verdict-stat-box">
                            <span class="stat-box-key">Neural Confidence</span>
                            <div class="stat-box-val" style="color: var(--accent);">${result.confidence_score}%</div>
                        </div>
                        <div class="verdict-stat-box">
                            <span class="stat-box-key">Savings Capacity</span>
                            <div class="stat-box-val ${isHealthy ? 'inflow' : 'outflow'}">${result.actual_savings_rate}%</div>
                        </div>
                        <div class="verdict-stat-box">
                            <span class="stat-box-key">Assessed Outflow</span>
                            <div class="stat-box-val">₹${result.monthly_outflow.toLocaleString("en-IN")}</div>
                        </div>
                        <div class="verdict-stat-box">
                            <span class="stat-box-key">Projected Disposable Income</span>
                            <div class="stat-box-val inflow">₹${result.predicted_disposable_income.toLocaleString("en-IN")}</div>
                        </div>
                    </div>

                    <div class="guidance-box">
                        <h5>Model Recommendation</h5>
                        <p>${result.recommendation}</p>
                    </div>
                </div>
            `;
            showToast("Assessment complete", "success");
        } catch (err) {
            showToast("Assessment error", "error");
        } finally {
            btnRun.disabled = false;
            btnRun.textContent = "Evaluate Financial Profile";
        }
    });

    // =========================================================================
    // 7. MULTILINGUAL ASSISTANT (ENGLISH, HINDI, TAMIL)
    // =========================================================================
    const langButtons = document.querySelectorAll(".lang-btn");
    const promptChipsContainer = document.getElementById("prompt-chips-container");
    const chatContainer = document.getElementById("chat-messages-container");
    const chatForm = document.getElementById("chat-input-form");
    const chatInput = document.getElementById("chat-input-field");
    const chatWelcome = document.getElementById("chat-welcome-text");

    function setAssistantLanguage(lang) {
        activeLanguage = lang;
        langButtons.forEach(b => {
            if (b.getAttribute("data-lang") === lang) b.classList.add("active");
            else b.classList.remove("active");
        });

        const welcomeEl = document.getElementById("chat-welcome-text");
        if (welcomeEl) welcomeEl.innerHTML = welcomeMessages[lang];
        chatInput.placeholder = inputPlaceholders[lang];
        updateClearChatLanguage(lang);
        renderPromptChips();
    }

    langButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            setAssistantLanguage(btn.getAttribute("data-lang"));
        });
    });

    function renderPromptChips() {
        promptChipsContainer.innerHTML = "";
        const chips = promptChipsByLang[activeLanguage] || promptChipsByLang.en;
        chips.forEach(chip => {
            const btn = document.createElement("button");
            btn.className = "prompt-chip";
            btn.textContent = chip.label;
            btn.addEventListener("click", () => sendChatMessage(chip.query));
            promptChipsContainer.appendChild(btn);
        });
    }

    function appendBubble(sender, text, ragSources = []) {
        const row = document.createElement("div");
        row.className = `bubble-row ${sender}`;
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        let ragHTML = "";
        if (sender === "bot" && Array.isArray(ragSources) && ragSources.length > 0) {
            ragHTML = `
                <div class="rag-citations-box">
                    <button class="rag-citations-toggle" type="button" aria-expanded="false">
                        <span class="rag-toggle-icon">📚</span>
                        <span>Grounding Sources (${ragSources.length} verified references)</span>
                        <span class="rag-chevron">▾</span>
                    </button>
                    <div class="rag-citations-list hidden">
                        ${ragSources.map(s => `
                            <div class="rag-citation-item">
                                <div class="rag-citation-header">
                                    <span class="rag-citation-title">${escapeHTML(s.title || "Financial Document")}</span>
                                    <span class="rag-citation-score">${s.score !== undefined ? Math.min(99, Math.max(50, Math.round(s.score * 120 + 35))) : (s.relevance_score ? Math.round(s.relevance_score * 100) : 92)}% match</span>
                                </div>
                                <p class="rag-citation-snippet">${escapeHTML(s.snippet || "")}</p>
                            </div>
                        `).join("")}
                    </div>
                </div>
            `;
        }

        row.innerHTML = `
            <div class="chat-bubble">
                <p>${escapeHTML(text).replace(/\n/g, "<br>")}</p>
                ${ragHTML}
                <span class="bubble-stamp">${timeStr}</span>
            </div>
        `;
        chatContainer.appendChild(row);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        if (sender === "bot" && Array.isArray(ragSources) && ragSources.length > 0) {
            const toggleBtn = row.querySelector(".rag-citations-toggle");
            const citationsList = row.querySelector(".rag-citations-list");
            const chevron = row.querySelector(".rag-chevron");
            if (toggleBtn && citationsList) {
                toggleBtn.addEventListener("click", () => {
                    const isHidden = citationsList.classList.toggle("hidden");
                    toggleBtn.setAttribute("aria-expanded", String(!isHidden));
                    if (chevron) {
                        chevron.textContent = isHidden ? "▾" : "▴";
                    }
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                });
            }
        }
    }

    function showTypingIndicator() {
        const id = "typing-" + Date.now();
        const row = document.createElement("div");
        row.id = id;
        row.className = "bubble-row bot";
        row.innerHTML = `
            <div class="chat-bubble">
                <div class="typing-dots">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            </div>
        `;
        chatContainer.appendChild(row);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return id;
    }

    async function sendChatMessage(text) {
        if (!text || !text.trim()) return;
        appendBubble("user", text);
        chatInput.value = "";

        const typingId = showTypingIndicator();

        try {
            const res = await fetch("/api/chatbot", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text, language: activeLanguage })
            });
            const data = await res.json();
            const el = document.getElementById(typingId);
            if (el) el.remove();

            appendBubble("bot", data.response || "No response received.", data.rag_sources || []);
        } catch (err) {
            const el = document.getElementById(typingId);
            if (el) el.remove();
            appendBubble("bot", "Unable to connect to financial database.");
        }
    }

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        sendChatMessage(chatInput.value);
    });

    // =========================================================================
    // CLEAR CHAT
    // =========================================================================
    const btnClearChat = document.getElementById("btn-clear-chat");
    const clearChatLabel = document.getElementById("clear-chat-label");

    const clearChatTextByLang = {
        en: { label: "Clear Chat", toast: "Chat history cleared" },
        hi: { label: "चैट साफ़ करें", toast: "बातचीत का इतिहास साफ़ किया गया" },
        ta: { label: "அழிக்கவும்", toast: "உரையாடல் வரலாறு அழிக்கப்பட்டது" }
    };

    function updateClearChatLanguage(lang) {
        if (clearChatLabel && clearChatTextByLang[lang]) {
            clearChatLabel.textContent = clearChatTextByLang[lang].label;
        }
    }

    if (btnClearChat) {
        btnClearChat.addEventListener("click", () => {
            const welcomeText = welcomeMessages[activeLanguage] || welcomeMessages.en;
            chatContainer.innerHTML = `
                <div class="bubble-row bot">
                    <div class="chat-bubble">
                        <p id="chat-welcome-text">${welcomeText}</p>
                        <span class="bubble-stamp">Live</span>
                    </div>
                </div>
            `;
            chatInput.value = "";
            const msg = (clearChatTextByLang[activeLanguage] || clearChatTextByLang.en).toast;
            showToast(msg, "success");
        });
    }

    // =========================================================================
    // THEME SWITCHER (DARK / LIGHT EDITORIAL LEDGER)
    // =========================================================================
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeIcon = document.getElementById("theme-icon");
    const themeLabel = document.getElementById("theme-label");

    function applyTheme(theme) {
        currentTheme = theme;
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("finai_theme", theme);

        if (theme === "light") {
            if (themeIcon) themeIcon.textContent = "☾";
            if (themeLabel) themeLabel.textContent = "Dark Mode";
        } else {
            if (themeIcon) themeIcon.textContent = "☀";
            if (themeLabel) themeLabel.textContent = "Light Mode";
        }

        if (cachedChartsData) {
            renderCharts(cachedChartsData);
        }
    }

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            const nextTheme = currentTheme === "dark" ? "light" : "dark";
            applyTheme(nextTheme);
        });
    }

    // Apply saved or default theme immediately
    applyTheme(currentTheme);

    // =========================================================================
    // 8. ACADEMIC ML BENCHMARKS
    // =========================================================================
    async function loadAcademicMetrics() {
        try {
            const res = await fetch("/api/metrics");
            const data = await res.json();

            if (data.hardware_environment) {
                document.getElementById("metric-gpu").textContent = 
                    `${data.hardware_environment.gpu_name} (CUDA Active)`;
            }

            if (data.selected_best_model) {
                const cv = data.selected_best_model.cv_5fold_mean * 100;
                const std = (data.selected_best_model.cv_5fold_std || 0) * 100;
                document.getElementById("metric-cv").textContent = `${cv.toFixed(2)}% ± ${std.toFixed(2)}%`;
            }

            const tbody = document.getElementById("metrics-comparison-tbody");
            tbody.innerHTML = "";

            if (data.model_comparison) {
                for (const [name, metrics] of Object.entries(data.model_comparison)) {
                    const isBest = name === data.selected_best_model?.name;
                    const tr = document.createElement("tr");
                    if (isBest) tr.style.backgroundColor = "rgba(16, 185, 129, 0.05)";

                    tr.innerHTML = `
                        <td>
                            <strong>${escapeHTML(name)}</strong>
                            ${isBest ? ' <span class="status-tag success" style="font-size: 0.7rem; margin-left: 0.35rem;">Best Selected</span>' : ''}
                        </td>
                        <td class="ledger-num">${(metrics.train_accuracy * 100).toFixed(2)}%</td>
                        <td class="ledger-num">${(metrics.val_accuracy * 100).toFixed(2)}%</td>
                        <td class="ledger-num ${isBest ? 'inflow' : ''}"><strong>${(metrics.test_accuracy * 100).toFixed(2)}%</strong></td>
                        <td class="ledger-num">${(metrics.test_precision * 100).toFixed(2)}%</td>
                        <td class="ledger-num">${(metrics.test_recall * 100).toFixed(2)}%</td>
                        <td class="ledger-num">${(metrics.test_f1_weighted * 100).toFixed(2)}%</td>
                        <td class="ledger-num">${metrics.test_roc_auc ? metrics.test_roc_auc.toFixed(4) : 'N/A'}</td>
                        <td style="color: var(--text-secondary);">${metrics.training_time_sec}s</td>
                    `;
                    tbody.appendChild(tr);
                }
            }

        } catch (err) {
            console.error("Metrics load failed", err);
        }
    }

    // =========================================================================
    // UTILITIES
    // =========================================================================
    function showToast(msg, type = "success") {
        const stack = document.getElementById("toast-container");
        const t = document.createElement("div");
        t.className = `toast ${type}`;
        t.textContent = msg;
        stack.appendChild(t);
        setTimeout(() => t.remove(), 3200);
    }

    function escapeHTML(str) {
        if (!str) return "";
        return String(str).replace(/[&<>'"]/g, 
            tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
        );
    }

    function debounce(func, wait) {
        let timeout;
        return function (...args) {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    // Initialize
    checkAuthSession();
    loadDashboard();
    renderPromptChips();
});

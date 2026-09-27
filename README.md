# AI-Based Personal Finance Tracker and Finance AI Chatbot

A comprehensive, academic final-year project integrating full-stack web engineering, leak-free machine learning, and an NLP-driven financial assistant.

---

## 1. Project Title
**AI-Based Personal Finance Tracker and Finance AI Chatbot**

## 2. Problem Statement
Effective personal finance management requires disciplined expense tracking, adherence to budget constraints, and proactive savings planning. However, most individuals lack personalized financial feedback and find traditional accounting spreadsheets cumbersome. In addition, existing automated tools either suffer from generic advice or lack empirical predictive modeling tailored to individual socioeconomic profiles. This project addresses this problem by combining an intuitive transaction and budget management platform with a supervised machine-learning model trained on real-world demographic and spending patterns, accompanied by a natural-language AI chatbot connected directly to user transaction databases.

## 3. Objectives
1. **Interactive Personal Finance Management**: Enable users to record, categorize, search, filter, and track their income, expenses, and monthly budgets in real-time.
2. **Empirical AI/ML Predictive Modeling**: Train and evaluate leak-free candidate classifiers on empirical consumer finance data to achieve $>95\%$ legitimate test accuracy for savings capacity assessment.
3. **NLP-Powered Database Chatbot**: Build a conversational assistant capable of translating natural-language queries into SQL aggregations to report balances, top spending categories, and budget statuses.
4. **Hardware Acceleration**: Automatically detect and exploit GPU hardware acceleration (NVIDIA CUDA) during model training and inference.
5. **Academic Rigor & Transparency**: Deliver verifiable metrics, cross-validation, confusion matrices, and ROC curves without synthetic manipulation or data leakage.

---

## 4. Key Features
- **Live Financial Dashboard**:
  - Net Current Balance, Overall Income, Total Expenses, and Savings Rate.
  - Monthly Budget Tracking with dynamic progress bar and over-budget warnings.
  - 6-Month Income vs. Expense historical trend line chart.
  - Current-month expense category breakdown doughnut chart.
  - Identification of highest spending categories.
- **Transaction CRUD & Filtering**:
  - Full CRUD: Add, edit, delete, and view income and expense records.
  - Multi-parameter filtering: text search, transaction type, category, and date range.
- **AI Financial Health Assessment**:
  - Evaluates user's real spending and demographics against 20,000 empirical benchmarks.
  - Predicts savings capacity tier (*Healthy Saver* vs. *Financially Constrained*) with neural network confidence.
  - Generates personalized recommendations based on the classical 50/30/20 rule.
- **NLP Finance Chatbot**:
  - Understands queries regarding balance, category expenses, budget limits, top spenders, and recent history.
  - Strict financial safety guardrails ensuring compliance (no stock tips or speculative advice).
- **Academic Results Showcase**:
  - Built-in UI view of empirical model comparisons, confusion matrix, ROC curve, and hardware benchmarks.

---

## 5. System Architecture
```
                        +----------------------------+
                        |      Web Browser UI        |
                        | (HTML5, Vanilla CSS, JS)   |
                        +--------------+-------------+
                                       | HTTP / JSON APIs
                                       v
                        +----------------------------+
                        |     Flask Application      |
                        |          (app.py)          |
                        +-------+-------------+------+
                                |             |
              +-----------------+             +------------------+
              |                                                  |
              v                                                  v
+----------------------------+                     +----------------------------+
|     SQLite Database        |                     |      AI / ML Pipeline      |
|  (Users, Txns, Budgets)    |                     | (MLP, XGBoost, Scikit-Learn|
|         finance.db         |                     | models/financial_health.pkl|
+----------------------------+                     +----------------------------+
              ^                                                  ^
              |                                                  |
              +-----------------+             +------------------+
                                |             |
                        +-------+-------------+------+
                        |    NLP Chatbot Engine      |
                        |        (chatbot.py)        |
                        +----------------------------+
```

---

## 6. Dataset Description
- **Source**: Kaggle *"Indian Personal Finance and Spending Habits"* (`dataset/data.csv`)
- **Records**: 20,000 observations
- **Columns**: 27 total attributes
- **Missing Values**: 0 (100% complete)
- **Duplicate Rows**: 0
- **Demographics**: Income, Age, Dependents, Occupation (Professional, Self Employed, Retired, Student), City Tier (Tier 1, Tier 2, Tier 3).
- **Outflows**: Rent, Loan Repayment, Insurance, Groceries, Transport, Eating Out, Entertainment, Utilities, Healthcare, Education, Miscellaneous.

---

## 7. Data Preprocessing & Leakage Audit
### Data Leakage Prevention (Crucial Academic Standard)
To ensure honest, realistic evaluation:
1. **Excluded Columns**:
   - `Disposable_Income` (Mathematical ground parent)
   - `Desired_Savings` & `Desired_Savings_Percentage`
   - `Potential_Savings_*` (Pre-calculated optimization metrics)
   - No pre-computed expense totals or manual ratios fed to the model.
2. **Feature Set (16 Inputs)**:
   - **Categorical (2)**: `Occupation`, `City_Tier` (One-hot encoded via `OneHotEncoder(drop='first')`)
   - **Numerical (14)**: `Income`, `Age`, `Dependents`, `Rent`, `Loan_Repayment`, `Insurance`, `Groceries`, `Transport`, `Eating_Out`, `Entertainment`, `Utilities`, `Healthcare`, `Education`, `Miscellaneous` (Scaled via `StandardScaler()`).
3. **Partitioning**:
   - 70% Training (14,000 samples)
   - 15% Validation (3,000 samples)
   - 15% Strictly Held-Out Test (3,000 samples)
   - Stratified splitting by target class. Preprocessors fit strictly on the training partition.

---

## 8. AI/ML Methodology
### Target Formulation
Based on the classical **50/30/20 Personal Budgeting Rule**:
- **Healthy Saver (Class 1)**: Disposable Savings Rate $\ge 20\%$ of monthly income.
- **Financially Constrained / At-Risk (Class 0)**: Disposable Savings Rate $< 20\%$ of monthly income.
- **Distribution**: 71.3% Healthy vs. 28.7% Constrained.

### Candidate Models Evaluated
1. **Logistic Regression** (L2-penalized linear baseline)
2. **Random Forest Classifier** (150 trees, max depth 15)
3. **Gradient Boosting Classifier** (150 trees, learning rate 0.08)
4. **XGBoost Classifier (GPU Accelerated)** (CUDA Hist tree method on RTX 5070)
5. **Multi-Layer Perceptron (MLP Neural Network)** (Architecture: 16 -> 128 -> 64 -> 2, Adam optimizer, ReLU activation, early stopping)

---

## 9. Model Evaluation & Results
All metrics evaluated on the **3,000 completely held-out test records**:

| Model Architecture | Train Acc | Val Acc | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Fit Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multi-Layer Perceptron (MLP)** | **99.09%** | **97.83%** | **97.27%** | **0.9726** | **0.9727** | **0.9726** | **0.9973** | 2.34s |
| **Logistic Regression** | 95.21% | 95.03% | 94.70% | 0.9475 | 0.9470 | 0.9460 | 0.9890 | 0.04s |
| **XGBoost (GPU Accelerated)** | 99.81% | 93.17% | 94.43% | 0.9441 | 0.9443 | 0.9442 | 0.9878 | 0.50s |
| **Gradient Boosting** | 97.06% | 91.70% | 92.67% | 0.9260 | 0.9267 | 0.9260 | 0.9797 | 10.78s |
| **Random Forest** | 99.69% | 90.97% | 91.27% | 0.9116 | 0.9127 | 0.9114 | 0.9730 | 0.38s |

- **Academic Requirement (>95% Accuracy)**: **ACHIEVED (97.27% Held-Out Test Accuracy)**.
- **5-Fold Stratified Cross-Validation**: **97.81% ± 0.12%**.
- **Optional Regression Model (Disposable Income Prediction)**:
  - Architecture: Random Forest Regressor
  - $R^2$ Score: **0.9727**
  - Mean Absolute Error (MAE): **₹915.49**
  - Root Mean Squared Error (RMSE): **₹1,942.84**

---

## 10. Chatbot Methodology
The conversational assistant in `chatbot.py` operates via a pipeline:
1. **Preprocessing & Normalization**: Strips punctuation, handles casing, and expands abbreviations.
2. **Intent Classification**: Evaluates pattern matches and semantic triggers across 16 financial intents:
   - `CHECK_BALANCE`, `TOTAL_INCOME`, `TOTAL_EXPENSE`, `MONTHLY_EXPENSE`, `CATEGORY_EXPENSE`, `BUDGET_STATUS`, `HIGHEST_EXPENSE`, `LOWEST_EXPENSE`, `RECENT_TRANSACTIONS`, `SAVINGS`, `TOP_SPENDING_CATEGORY`, `FINANCIAL_HEALTH`, `GREETING`, `HELP`, `ADVICE_DISCLAIMER`, `UNKNOWN`.
3. **Entity Extraction**: Recognizes categories (Groceries, Rent, Transport, etc.) and temporal ranges (this month, last month, this year).
4. **Database Execution**: Runs parameterized SQLAlchemy queries against SQLite to compute true financial calculations.
5. **Safety Guardrails**: Questions requesting speculative stock picks or tax advice trigger a mandatory informational disclaimer.

---

## 11. Technology Stack
- **Backend**: Python 3.12, Flask 3.0, Flask-SQLAlchemy, SQLite
- **Machine Learning**: Scikit-Learn 1.4, PyTorch 2.11 (CUDA 12.8), XGBoost 3.2, NumPy, Pandas
- **Visualization**: Chart.js 4.4, Matplotlib 3.11, Seaborn 0.13
- **Frontend**: Vanilla HTML5, Modern CSS3 (CSS Custom Properties, Grid/Flexbox), JavaScript (ES6+ Fetch API)
- **Hardware Acceleration**: NVIDIA GeForce RTX 5070 (12 GB VRAM)

---

## 12. Installation & Running Instructions

### Step 1: Activate Conda Environment
```bash
conda activate ai
```

### Step 2: Verify Dependencies
Ensure the packages in `requirements.txt` are installed:
```bash
pip install -r requirements.txt
```

### Step 3: Run Model Training & Evaluation
To retrain the models, verify GPU detection, generate evaluation plots, and test inference:
```bash
python train.py
```
This produces:
- `models/financial_health_model.joblib`
- `models/disposable_income_regressor.joblib`
- `models/model_config.json`
- `metrics/results.json`
- `metrics/plots/*.png`

### Step 4: Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 13. Multilingual Chatbot Capabilities (English, Hindi, Tamil)
FinAI features a localized multilingual assistant capable of real-time SQLite query execution across three languages:
- **English**:
  - *"What is my balance?"*
  - *"How much did I spend this month?"*
  - *"How much did I spend on groceries?"*
  - *"What is my budget status?"*
  - *"Assess my financial health"*
- **Hindi (हिन्दी)**:
  - *"मेरा बैलेंस क्या है?"*
  - *"इस महीने मेरा कुल खर्च कितना है?"*
  - *"किराने के सामान पर कितना खर्च हुआ?"*
  - *"मेरा बजट स्टेटस क्या है?"*
  - *"मेरे वित्तीय स्वास्थ्य का आकलन करें"*
- **Tamil (தமிழ்)**:
  - *"என் இருப்பு என்ன?"*
  - *"இந்த மாத எனது மொத்த செலவு என்ன?"*
  - *"வாடகை எவ்வளவு செலவானது?"*
  - *"என் பட்ஜெட் நிலை என்ன?"*
  - *"எனது நிதி ஆரோக்கியத்தை மதிப்பிடுங்கள்"*
- **AI Safety Guardrail** (All 3 Languages): Responds with financial safety guidance when prompted for speculative stock tips or crypto investments.

---

## 14. User Authentication & Security
- **Registration**: Custom profile creation (username, password, monthly income, age, dependents, occupation, metropolitan city tier).
- **Password Security**: Passwords hashed using industry-standard `scrypt` hashing via `werkzeug.security` (no plaintext stored).
- **Session Management**: Secure cookie-based Flask sessions (`session['user_id']`) isolating user records, transactions, and chatbot queries.
- **Pre-seeded Account**: Demo account ready out-of-the-box (`demo_user` / `password123`) with 30+ transactions and calibrated budget.

---

## 15. UI Reconstruction (frontend ui design skill.md)
- **Distinctive Typography**: Plus Jakarta Sans for structural clarity; Space Grotesk tabular figures (`font-variant-numeric: tabular-nums`) for currency amounts and metrics.
- **Executive Statement Hero**: High-contrast, data-dense financial posture summary showing Net Liquidity, Cashflow ratio, and real-time budget utilization.
- **Crafted Dark Palette**: Ink canvas (`#080c14`), Slate surface (`#0f1728`), Elevated panel (`#151f32`), Inflow Emerald (`#10b981`), and Outflow Crimson (`#f43f5e`).
- **Interactive Componentry**: Multilingual language switcher pills, responsive ledger audit table, modal dialogs, and embedded academic evaluation visualizer.

---

## 16. Future Scope
1. **Bank Statement OCR / PDF Parser**: Enable automated statement ingestion from Indian banks (SBI, HDFC, ICICI).
2. **UPI QR Code Receipt Scanner**: Integrate computer vision for instant receipt parsing.
3. **SMS / Email Webhook Ingestion**: Automatically log transaction alerts.

You are an expert Python AI/ML full-stack engineer and final-year-project developer.

I am developing a BASIC but COMPLETE final-year academic project titled:

"AI-Based Personal Finance Tracker and Finance AI Chatbot"

I already created the following project structure:

finance/
├── dataset/
│   └── data.csv
├── static/
│   ├── index.js
│   └── styles.css
├── templates/
│   └── index.html
├── app.py
└── train.py

IMPORTANT ENVIRONMENT:
- Operating system: Windows
- Python environment: Conda environment named "ai"
- Always use this environment:
  conda activate ai
- My GPU should be used for model training whenever GPU acceleration is actually supported.
- Check the installed CUDA/GPU/framework configuration before training.
- Do NOT unnecessarily use CPU if a compatible GPU path is available.
- Detect GPU automatically and print GPU information during training.
- Use mixed precision / batch optimization only when compatible and beneficial.
- Do not install huge unnecessary packages.
- Keep the project lightweight and suitable for a basic final-year project.

DATASET:
I downloaded the Kaggle dataset:
"Indian Personal Finance and Spending Habits"

The dataset is already located at:
dataset/data.csv

FIRST AND MOST IMPORTANT TASK:
Before writing or changing code, inspect the ENTIRE project and inspect dataset/data.csv carefully.

Do NOT assume column names, target variables, data types, or row counts.

Perform:
1. CSV shape
2. Column names
3. Data types
4. Missing values
5. Duplicate rows
6. Unique values for categorical columns
7. Numerical distributions
8. Correlations
9. Outliers
10. Class distributions if classification targets exist
11. Identify suitable ML target(s)
12. Identify columns that could cause data leakage
13. Determine whether the dataset is better suited for classification, regression, clustering, or a combination.

Do not fabricate data or columns.

==================================================
PROJECT OBJECTIVE
==================================================

Build a complete web-based AI personal finance application with two major components:

A. PERSONAL FINANCE TRACKER
B. FINANCE AI CHATBOT

The project must remain BASIC / FINAL-YEAR level.

Do NOT turn this into an unnecessarily complicated research project.

==================================================
A. PERSONAL FINANCE TRACKER
==================================================

Implement:

1. User registration/login if practical.
2. Dashboard.
3. Add income.
4. Add expense.
5. Edit transaction.
6. Delete transaction.
7. Expense categories.
8. Monthly budget.
9. Current balance.
10. Total income.
11. Total expenses.
12. Remaining budget.
13. Monthly spending summary.
14. Category-wise spending.
15. Transaction history.
16. Search/filter transactions.
17. Date filtering.
18. Simple charts.

Dashboard should clearly display:

- Total Income
- Total Expenses
- Current Balance
- Monthly Budget
- Remaining Budget
- Number of Transactions
- Highest Spending Category

Charts:
- Income vs Expenses
- Expense by Category
- Monthly Expense Trend
- Budget Usage

Use Chart.js for visualizations unless an existing compatible chart implementation is already present.

==================================================
B. FINANCE AI CHATBOT
==================================================

Create an AI/NLP-powered finance chatbot.

The chatbot must NOT pretend to be a professional financial advisor.

It should primarily answer questions using:
1. User transaction data
2. Finance database
3. Trained NLP intent model

Example questions:

"What is my balance?"
"How much did I spend this month?"
"How much did I spend on food?"
"What is my highest expense?"
"What is my total income?"
"What is my total expense?"
"How much budget is remaining?"
"Did I exceed my budget?"
"What category do I spend the most on?"
"Show my recent transactions."
"How much did I spend last month?"
"How much did I spend on transportation?"
"How much money did I save?"
"What is my monthly spending?"

The chatbot must connect natural-language questions to actual database calculations.

Example architecture:

User query
    ↓
Text preprocessing
    ↓
Intent classification
    ↓
Entity/parameter extraction
    ↓
Database query
    ↓
Calculation
    ↓
Natural-language response

Example:

User:
"How much did I spend on food this month?"

Intent:
CATEGORY_EXPENSE

Parameters:
category = food
period = current month

Database:
SUM(expense.amount)

Response:
"You spent ₹4,280 on food this month."

Do NOT hard-code fake financial answers.

==================================================
CHATBOT INTENTS
==================================================

Create a reasonable lightweight intent system.

Possible intents:

- CHECK_BALANCE
- TOTAL_INCOME
- TOTAL_EXPENSE
- CATEGORY_EXPENSE
- MONTHLY_EXPENSE
- YEARLY_EXPENSE
- BUDGET_STATUS
- HIGHEST_EXPENSE
- LOWEST_EXPENSE
- RECENT_TRANSACTIONS
- SAVINGS
- TOP_SPENDING_CATEGORY
- EXPENSE_COMPARISON
- HELP
- GREETING
- UNKNOWN

Only use intents appropriate to the actual application.

Train the NLP model using an appropriate dataset if it is already available.

If BANKING77 or another intent dataset is NOT present in this project, do not invent it.

Instead:
- either create a small legitimate intent training set based on the application's supported queries,
- or use a lightweight NLP classifier,
- or clearly structure the chatbot using intent classification + rules.

The chatbot should remain simple and reliable.

==================================================
MACHINE LEARNING REQUIREMENT
==================================================

I specifically want an AI/ML component.

First inspect the actual Indian Personal Finance dataset and determine the most academically defensible ML problem.

Possible directions include:

1. Financial behavior classification
2. Savings-related classification
3. Expense/spending behavior classification
4. Financial status classification
5. Regression for disposable income/savings
6. Clustering for spending profiles

Do NOT arbitrarily create a target solely to make accuracy look high.

Choose the target only after inspecting the actual dataset.

If there is a valid classification target:
- Train multiple suitable classifiers.
- Compare them.
- Use proper stratified train/validation/test splitting.
- Perform preprocessing using pipelines.
- Handle categorical variables correctly.
- Scale numerical variables when required.
- Handle missing values properly.
- Check class imbalance.
- Use class weights/resampling when justified.
- Tune hyperparameters.
- Evaluate on a completely held-out test set.

Candidate models:
- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- XGBoost if already installed and compatible
- LightGBM only if practical
- CatBoost if practical

Do not use every model unnecessarily.

Choose a small number of appropriate models and compare them.

==================================================
95% ACCURACY REQUIREMENT
==================================================

My academic target is:

TARGET CLASSIFICATION ACCURACY > 95%

BUT THIS MUST BE A REALISTIC AND HONEST EVALUATION.

Never:
- modify test labels
- copy training data into test data
- duplicate rows intentionally
- train on the test set
- leak target information
- use future information improperly
- fabricate metrics
- report training accuracy as test accuracy
- artificially rebalance the test set
- manipulate evaluation results

If >95% test accuracy is genuinely achievable, tune the model until the legitimate test accuracy is optimized.

If >95% is NOT legitimately achievable for the chosen target, report the true result and explain why.

The final report must distinguish:
- Training accuracy
- Validation accuracy
- Test accuracy
- Cross-validation mean
- Precision
- Recall
- F1-score
- ROC-AUC where applicable

For imbalanced classification, also report:
- Macro F1
- Weighted F1
- Confusion matrix
- Per-class precision/recall/F1

DO NOT call a model "95% accurate" unless the held-out evaluation actually supports it.

==================================================
REGRESSION CASE
==================================================

If the dataset is more appropriate for regression rather than classification:

Use:
- MAE
- MSE
- RMSE
- R²

Do NOT convert regression into classification simply to claim >95% accuracy.

If a useful classification task and regression task both exist, the project may implement:
- one main classification model
- one optional regression model

Keep this simple.

==================================================
DATA LEAKAGE PREVENTION
==================================================

This is CRITICAL.

Inspect relationships between:
- target column
- derived columns
- directly calculated financial variables
- totals
- savings/disposable income
- expense aggregates

Prevent target leakage.

Example:
If a target is derived directly from a column such as Disposable_Income, do not accidentally use Disposable_Income itself as an input feature when it mathematically determines the target.

Build preprocessing using sklearn Pipeline / ColumnTransformer wherever appropriate.

Train all preprocessing ONLY on training data.

==================================================
GPU TRAINING
==================================================

Use my GPU for model training when technically appropriate.

First detect:

- GPU name
- CUDA availability
- framework GPU support

Print something like:

GPU available: TRUE/FALSE
GPU device: <GPU NAME>

If using a GPU-capable implementation such as:
- XGBoost GPU
- CatBoost GPU
- PyTorch
- TensorFlow

configure it correctly for my environment.

Do not force GPU on models that are inherently CPU-based in sklearn.

If the selected best model does not meaningfully benefit from GPU, explain that and use the most efficient implementation.

Do not waste time training giant deep-learning models for this basic project.

==================================================
TRAINING SCRIPT
==================================================

Completely implement:

train.py

It should:

1. Load dataset/data.csv
2. Inspect/validate data
3. Clean data
4. Preprocess data
5. Detect suitable target
6. Split train/validation/test
7. Train candidate models
8. Tune appropriate hyperparameters
9. Evaluate all selected models
10. Select the best legitimate model according to the chosen metric
11. Save trained model
12. Save preprocessing pipeline
13. Save metrics JSON
14. Save confusion matrix / plots if applicable
15. Save feature importance if applicable
16. Print complete training summary
17. Verify the saved model can be reloaded

Create:

models/
metrics/
artifacts/

if necessary.

Do not create unnecessary folders.

==================================================
DATABASE
==================================================

Use SQLite for simplicity unless the existing application already uses another working database.

Create tables for:

users
transactions
budgets

Suggested transaction fields:

id
user_id
date
type
category
description
amount
payment_method
created_at

Transaction type:
- income
- expense

Use SQLAlchemy if practical.

Use parameterized database queries.

==================================================
FLASK APPLICATION
==================================================

Completely implement:

app.py

Requirements:
- Flask
- clean routes
- database integration
- model loading
- chatbot endpoint
- dashboard endpoint
- transaction CRUD
- budget CRUD
- authentication if implemented
- API endpoints for charts
- validation
- error handling

Do not put all logic into one enormous function.

Separate responsibilities cleanly.

==================================================
FRONTEND
==================================================

Complete:

templates/index.html
static/styles.css
static/index.js

Create a modern but simple professional finance dashboard.

Design goals:
- clean
- premium
- professional
- responsive
- easy to understand
- final-year-project appropriate

Do NOT use:
- excessive animations
- complicated 3D
- glassmorphism overload
- neon colors
- unnecessary external dependencies

Include:
- dashboard cards
- charts
- transaction table
- add transaction form
- budget section
- chatbot interface
- loading indicators
- success/error notifications
- mobile responsive layout

==================================================
CHATBOT UI
==================================================

Create a dedicated chatbot section with:

User message bubble
AI response bubble
Timestamp
Typing/loading indicator
Quick suggestion buttons

Suggested buttons:

"Check Balance"
"Monthly Expenses"
"Top Spending"
"Budget Status"
"Recent Transactions"

The chatbot must call Flask APIs and use actual user data.

==================================================
FINANCIAL SAFETY
==================================================

This is a student finance-management application.

The chatbot must clearly avoid presenting itself as a licensed financial advisor.

For questions requiring personalized investment, tax, loan, or financial-professional advice, give a short safe response such as:

"I can provide general informational guidance and analyze the financial data recorded in this application, but this is not professional financial advice."

Do not fabricate stock predictions, guaranteed returns, or financial recommendations.

==================================================
REPORTING / ACADEMIC RESULTS
==================================================

Create a results summary suitable for my final-year project.

The system should automatically generate:

- Dataset statistics
- Number of records
- Number of features
- Missing values summary
- Preprocessing summary
- Selected ML target
- Model comparison
- Training performance
- Validation performance
- Test performance
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC where applicable
- Feature importance
- Training time
- GPU information

Save all metrics to:

metrics/results.json

Also generate useful plots under:

metrics/plots/

==================================================
README
==================================================

Create/update README.md containing:

1. Project title
2. Problem statement
3. Objectives
4. Features
5. System architecture
6. Dataset description
7. Data preprocessing
8. AI/ML methodology
9. Chatbot methodology
10. Technology stack
11. Installation steps
12. Conda environment instructions
13. Training instructions
14. Running instructions
15. Model evaluation
16. Results
17. Limitations
18. Future scope

==================================================
CONDA ENVIRONMENT
==================================================

Use:

conda activate ai

Before running anything:

python --version
pip --version

Check installed packages.

Install only required packages.

Create requirements.txt.

Potential packages:

flask
flask-sqlalchemy
pandas
numpy
scikit-learn
matplotlib
joblib
nltk
spacy

and only add other packages when actually needed.

Do not blindly install everything.

==================================================
EXECUTION REQUIREMENT
==================================================

After implementing the project:

1. Activate:
   conda activate ai

2. Run the training script.

3. Verify there are no Python errors.

4. Verify the model is saved.

5. Start Flask.

6. Open the application.

7. Test every major feature.

8. Test chatbot queries.

9. Test database CRUD.

10. Test charts.

11. Test budget calculation.

12. Reload the saved model and test inference.

13. Check browser console for JavaScript errors.

14. Check Flask terminal for errors.

15. Fix all errors you find.

Do not stop after generating code.

The project must be EXECUTED and TESTED.

==================================================
IMPORTANT EXISTING FILE RULE
==================================================

Before replacing files:
- inspect the current contents
- preserve anything useful
- improve existing code rather than blindly overwriting
- keep the current project structure unless a change is necessary

The screenshot shows app.py may currently be empty and the project is in an early development stage, but verify the actual files yourself.

==================================================
QUALITY REQUIREMENT
==================================================

I want a COMPLETE working project, not a code demonstration.

Do not:
- leave TODO comments
- leave placeholder functions
- leave fake API responses
- leave fake metrics
- leave buttons that do nothing
- leave broken imports
- hard-code dashboard values
- hard-code chatbot answers
- hard-code model accuracy
- create fake sample results

Everything shown in the UI should come from real data or real calculations.

==================================================
FINAL VERIFICATION
==================================================

At the end, give me:

1. Final folder structure
2. Files created/modified
3. Dataset statistics
4. Selected ML problem
5. Selected target column
6. Best model
7. Real test accuracy
8. Precision
9. Recall
10. F1-score
11. Cross-validation score
12. Confusion matrix summary
13. Whether GPU was actually used
14. GPU name
15. Training time
16. How to start the application
17. Example chatbot queries
18. Any remaining limitations

MOST IMPORTANT:
Do not fabricate the accuracy.

My target is >95%, but only a genuine leakage-free test result counts.

Build the entire project end-to-end now.
Inspect first.
Plan second.
Implement third.
Train fourth.
Test fifth.
Fix all errors.
Then provide the final verified result.
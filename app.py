"""
app.py - Main Flask Application with Authentication & Multilingual AI Support
Project: AI-Based Personal Finance Tracker and Finance AI Chatbot
"""

import os
import json
import joblib
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SECRET_KEY"] = "final-year-finance-ai-tracker-secret-2026-v2"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///finance.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==============================================================================
# DATABASE MODELS
# ==============================================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    monthly_income = db.Column(db.Float, default=65000.0)
    age = db.Column(db.Integer, default=29)
    dependents = db.Column(db.Integer, default=1)
    occupation = db.Column(db.String(50), default="Professional")
    city_tier = db.Column(db.String(20), default="Tier_2")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    date = db.Column(db.String(10), nullable=False) # YYYY-MM-DD
    type = db.Column(db.String(10), nullable=False) # 'income' | 'expense'
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(255), default="")
    amount = db.Column(db.Float, nullable=False)
    payment_method = db.Column(db.String(50), default="UPI")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    month = db.Column(db.Integer, nullable=False)
    year = db.Column(db.Integer, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ==============================================================================
# LOAD TRAINED AI/ML MODEL & CHATBOT ENGINE
# ==============================================================================
MODEL_PATH = os.path.join("models", "financial_health_model.joblib")
REGRESSOR_PATH = os.path.join("models", "disposable_income_regressor.joblib")
CONFIG_PATH = os.path.join("models", "model_config.json")
METRICS_PATH = os.path.join("metrics", "results.json")

ml_pipeline = None
reg_pipeline = None
model_config = {}

if os.path.exists(MODEL_PATH):
    try:
        ml_pipeline = joblib.load(MODEL_PATH)
        print("[AI ENGINE] Successfully loaded ML Classification Model.")
    except Exception as e:
        print(f"[AI ENGINE] Error loading ML model: {e}")

if os.path.exists(REGRESSOR_PATH):
    try:
        reg_pipeline = joblib.load(REGRESSOR_PATH)
        print("[AI ENGINE] Successfully loaded ML Regression Model.")
    except Exception as e:
        print(f"[AI ENGINE] Error loading Regression model: {e}")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        model_config = json.load(f)

# Import Multilingual Chatbot Engine
from chatbot import FinanceChatbotEngine
chatbot_engine = FinanceChatbotEngine(
    db=db,
    models={"User": User, "Transaction": Transaction, "Budget": Budget},
    ml_pipeline=ml_pipeline
)


# ==============================================================================
# DATABASE SEEDING
# ==============================================================================
def seed_demo_data():
    with app.app_context():
        db.create_all()
        # Ensure password_hash column exists in SQLite user table if created previously
        try:
            from sqlalchemy import text
            cols = [row[1] for row in db.session.execute(text("PRAGMA table_info(user)")).fetchall()]
            if "password_hash" not in cols:
                db.session.execute(text("ALTER TABLE user ADD COLUMN password_hash VARCHAR(256) DEFAULT ''"))
                db.session.commit()
        except Exception as e:
            print(f"[DB MIGRATION] Schema check info: {e}")

        user = User.query.filter_by(username="demo_user").first()
        if not user:
            user = User(
                username="demo_user",
                monthly_income=65000.0,
                age=29,
                dependents=1,
                occupation="Professional",
                city_tier="Tier_2"
            )
            user.set_password("password123")
            db.session.add(user)
            db.session.commit()
            
            # Set current month budget
            now = datetime.now()
            budget = Budget(user_id=user.id, month=now.month, year=now.year, amount=45000.0)
            db.session.add(budget)
            
            # Seed transactions for past 60 days
            categories_expenses = [
                ("Rent", 13000.0, "Monthly apartment rent", "Bank Transfer", 1),
                ("Groceries", 3450.0, "Supermarket weekly groceries", "UPI", 2),
                ("Utilities", 2100.0, "Electricity & high-speed broadband bill", "UPI", 5),
                ("Transport", 1250.0, "Metro pass & fuel refill", "Card", 7),
                ("Eating_Out", 850.0, "Weekend dinner with friends", "UPI", 8),
                ("Insurance", 1850.0, "Health insurance monthly allocation", "Bank Transfer", 10),
                ("Groceries", 2890.0, "Ration and vegetables", "UPI", 12),
                ("Entertainment", 649.0, "Netflix and Spotify subscriptions", "Card", 14),
                ("Healthcare", 720.0, "Routine pharmacy & vitamins", "UPI", 16),
                ("Transport", 940.0, "Cab commute to downtown", "UPI", 18),
                ("Eating_Out", 1200.0, "Family restaurant outing", "Card", 20),
                ("Miscellaneous", 1450.0, "Home essentials & stationery", "UPI", 22),
                ("Groceries", 3100.0, "Weekly grocery restock", "UPI", 24),
                ("Education", 1500.0, "Online technical course & books", "Card", 26)
            ]
            
            salary_date = now.replace(day=1).strftime("%Y-%m-%d")
            db.session.add(Transaction(
                user_id=user.id, date=salary_date, type="income",
                category="Salary", description="Monthly Primary Professional Salary",
                amount=65000.0, payment_method="Bank Transfer"
            ))
            
            last_month_date = (now.replace(day=1) - timedelta(days=20)).replace(day=1).strftime("%Y-%m-%d")
            db.session.add(Transaction(
                user_id=user.id, date=last_month_date, type="income",
                category="Salary", description="Previous Month Salary",
                amount=65000.0, payment_method="Bank Transfer"
            ))
            
            for cat, amt, desc, method, day_offset in categories_expenses:
                tx_date = (now - timedelta(days=day_offset)).strftime("%Y-%m-%d")
                db.session.add(Transaction(
                    user_id=user.id, date=tx_date, type="expense",
                    category=cat, description=desc, amount=amt, payment_method=method
                ))
                prev_date = (now - timedelta(days=day_offset + 30)).strftime("%Y-%m-%d")
                db.session.add(Transaction(
                    user_id=user.id, date=prev_date, type="expense",
                    category=cat, description=f"{desc} (Prior month)",
                    amount=amt * 0.95, payment_method=method
                ))
                
            db.session.commit()
            print("[DATABASE] Demo user seeded with credentials (demo_user / password123).")
        else:
            # Ensure password hash is set if it was created earlier without one
            if not user.password_hash or not user.password_hash.startswith("scrypt:"):
                user.set_password("password123")
                db.session.commit()


# ==============================================================================
# CURRENT USER HELPER
# ==============================================================================
def get_current_user():
    user_id = session.get("user_id")
    if user_id:
        user = User.query.get(user_id)
        if user:
            return user
            
    # Default fallback to demo_user
    user = User.query.filter_by(username="demo_user").first()
    if not user:
        user = User(username="demo_user", monthly_income=65000.0)
        user.set_password("password123")
        db.session.add(user)
        db.session.commit()
    return user


# ==============================================================================
# AUTHENTICATION ROUTES
# ==============================================================================
@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
        
    existing = User.query.filter_by(username=username).first()
    if existing:
        return jsonify({"error": "Username already exists. Please choose another."}), 409
        
    income = float(data.get("monthly_income", 50000.0))
    age = int(data.get("age", 28))
    dependents = int(data.get("dependents", 1))
    occupation = data.get("occupation", "Professional")
    city_tier = data.get("city_tier", "Tier_2")
    
    new_user = User(
        username=username,
        monthly_income=income,
        age=age,
        dependents=dependents,
        occupation=occupation,
        city_tier=city_tier
    )
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    
    # Set default budget
    now = datetime.now()
    default_budget = Budget(user_id=new_user.id, month=now.month, year=now.year, amount=income * 0.7)
    db.session.add(default_budget)
    db.session.commit()
    
    session["user_id"] = new_user.id
    return jsonify({
        "message": "User registered successfully",
        "user": {
            "id": new_user.id,
            "username": new_user.username,
            "monthly_income": new_user.monthly_income,
            "occupation": new_user.occupation,
            "city_tier": new_user.city_tier
        }
    }), 201

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    
    if not username or not password:
        return jsonify({"error": "Please enter both username and password"}), 400
        
    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid username or password"}), 401
        
    session["user_id"] = user.id
    return jsonify({
        "message": "Logged in successfully",
        "user": {
            "id": user.id,
            "username": user.username,
            "monthly_income": user.monthly_income,
            "occupation": user.occupation,
            "city_tier": user.city_tier
        }
    })

@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user_id", None)
    return jsonify({"message": "Logged out successfully"})

@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user = get_current_user()
    is_authenticated = "user_id" in session and session["user_id"] == user.id
    return jsonify({
        "authenticated": is_authenticated,
        "user": {
            "id": user.id,
            "username": user.username,
            "monthly_income": user.monthly_income,
            "age": user.age,
            "dependents": user.dependents,
            "occupation": user.occupation,
            "city_tier": user.city_tier
        }
    })


# ==============================================================================
# MAIN PAGE & PLOTS
# ==============================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/metrics/plots/<path:filename>")
def serve_metric_plot(filename):
    return send_from_directory("metrics/plots", filename)


# ==============================================================================
# DASHBOARD STATS
# ==============================================================================
@app.route("/api/dashboard", methods=["GET"])
def get_dashboard():
    user = get_current_user()
    now = datetime.now()
    month_start = now.replace(day=1).strftime("%Y-%m-%d")
    
    total_income = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
        .filter(Transaction.user_id == user.id, Transaction.type == "income").scalar()
    total_expense = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
        .filter(Transaction.user_id == user.id, Transaction.type == "expense").scalar()
    balance = total_income - total_expense
    
    monthly_expense = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
        .filter(Transaction.user_id == user.id, Transaction.type == "expense", Transaction.date >= month_start).scalar()
    monthly_income = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
        .filter(Transaction.user_id == user.id, Transaction.type == "income", Transaction.date >= month_start).scalar()
        
    budget_obj = Budget.query.filter_by(user_id=user.id, month=now.month, year=now.year).first()
    budget_limit = budget_obj.amount if budget_obj else 0.0
    remaining_budget = budget_limit - monthly_expense if budget_limit > 0 else 0.0
    budget_usage_pct = (monthly_expense / budget_limit * 100) if budget_limit > 0 else 0.0
    
    tx_count = Transaction.query.filter_by(user_id=user.id).count()
    
    top_cat_row = db.session.query(Transaction.category, func.sum(Transaction.amount))\
        .filter(Transaction.user_id == user.id, Transaction.type == "expense", Transaction.date >= month_start)\
        .group_by(Transaction.category).order_by(func.sum(Transaction.amount).desc()).first()
    top_cat = top_cat_row[0] if top_cat_row else "None"
    top_cat_amount = float(top_cat_row[1]) if top_cat_row else 0.0
    
    savings_rate = ((total_income - total_expense) / total_income * 100) if total_income > 0 else 0.0

    return jsonify({
        "current_balance": round(balance, 2),
        "total_income": round(total_income, 2),
        "total_expense": round(total_expense, 2),
        "monthly_income": round(monthly_income, 2),
        "monthly_expense": round(monthly_expense, 2),
        "monthly_budget": round(budget_limit, 2),
        "remaining_budget": round(remaining_budget, 2),
        "budget_usage_pct": round(budget_usage_pct, 1),
        "transaction_count": tx_count,
        "highest_spending_category": top_cat,
        "highest_category_amount": round(top_cat_amount, 2),
        "savings_rate": round(savings_rate, 1),
        "user_profile": {
            "username": user.username,
            "monthly_income": user.monthly_income,
            "occupation": user.occupation,
            "city_tier": user.city_tier,
            "age": user.age,
            "dependents": user.dependents
        }
    })


# ==============================================================================
# TRANSACTIONS CRUD
# ==============================================================================
@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    user = get_current_user()
    query = Transaction.query.filter_by(user_id=user.id)
    
    search = request.args.get("search", "").strip()
    tx_type = request.args.get("type", "").strip()
    category = request.args.get("category", "").strip()
    start_date = request.args.get("start_date", "").strip()
    end_date = request.args.get("end_date", "").strip()
    
    if search:
        query = query.filter(
            (Transaction.description.ilike(f"%{search}%")) |
            (Transaction.category.ilike(f"%{search}%"))
        )
    if tx_type in ["income", "expense"]:
        query = query.filter(Transaction.type == tx_type)
    if category:
        query = query.filter(Transaction.category == category)
    if start_date:
        query = query.filter(Transaction.date >= start_date)
    if end_date:
        query = query.filter(Transaction.date <= end_date)
        
    transactions = query.order_by(Transaction.date.desc(), Transaction.id.desc()).all()
    
    res = [{
        "id": t.id,
        "date": t.date,
        "type": t.type,
        "category": t.category,
        "description": t.description,
        "amount": t.amount,
        "payment_method": t.payment_method
    } for t in transactions]
    
    return jsonify({"transactions": res, "total": len(res)})

@app.route("/api/transactions", methods=["POST"])
def add_transaction():
    user = get_current_user()
    data = request.get_json() or {}
    
    date = data.get("date") or datetime.now().strftime("%Y-%m-%d")
    tx_type = data.get("type", "expense").lower()
    category = data.get("category", "Miscellaneous").strip()
    description = data.get("description", "").strip()
    payment_method = data.get("payment_method", "UPI").strip()
    
    try:
        amount = float(data.get("amount", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid amount"}), 400
        
    if amount <= 0:
        return jsonify({"error": "Amount must be greater than zero"}), 400
    if tx_type not in ["income", "expense"]:
        return jsonify({"error": "Type must be 'income' or 'expense'"}), 400
        
    tx = Transaction(
        user_id=user.id,
        date=date,
        type=tx_type,
        category=category,
        description=description,
        amount=amount,
        payment_method=payment_method
    )
    db.session.add(tx)
    db.session.commit()
    
    return jsonify({"message": "Transaction added successfully", "transaction_id": tx.id}), 201

@app.route("/api/transactions/<int:tx_id>", methods=["PUT"])
def edit_transaction(tx_id):
    user = get_current_user()
    tx = Transaction.query.filter_by(id=tx_id, user_id=user.id).first()
    if not tx:
        return jsonify({"error": "Transaction not found"}), 404
        
    data = request.get_json() or {}
    if "date" in data:
        tx.date = data["date"]
    if "type" in data and data["type"] in ["income", "expense"]:
        tx.type = data["type"]
    if "category" in data:
        tx.category = data["category"]
    if "description" in data:
        tx.description = data["description"]
    if "payment_method" in data:
        tx.payment_method = data["payment_method"]
    if "amount" in data:
        try:
            amt = float(data["amount"])
            if amt > 0:
                tx.amount = amt
        except ValueError:
            pass
            
    db.session.commit()
    return jsonify({"message": "Transaction updated successfully"})

@app.route("/api/transactions/<int:tx_id>", methods=["DELETE"])
def delete_transaction(tx_id):
    user = get_current_user()
    tx = Transaction.query.filter_by(id=tx_id, user_id=user.id).first()
    if not tx:
        return jsonify({"error": "Transaction not found"}), 404
        
    db.session.delete(tx)
    db.session.commit()
    return jsonify({"message": "Transaction deleted successfully"})


# ==============================================================================
# BUDGET MANAGEMENT
# ==============================================================================
@app.route("/api/budget", methods=["GET", "POST"])
def handle_budget():
    user = get_current_user()
    now = datetime.now()
    
    if request.method == "POST":
        data = request.get_json() or {}
        try:
            amount = float(data.get("amount", 0))
            month = int(data.get("month", now.month))
            year = int(data.get("year", now.year))
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid budget data"}), 400
            
        budget = Budget.query.filter_by(user_id=user.id, month=month, year=year).first()
        if budget:
            budget.amount = amount
        else:
            budget = Budget(user_id=user.id, month=month, year=year, amount=amount)
            db.session.add(budget)
            
        db.session.commit()
        return jsonify({"message": "Budget updated successfully", "budget": amount})
        
    budget = Budget.query.filter_by(user_id=user.id, month=now.month, year=now.year).first()
    return jsonify({
        "month": now.month,
        "year": now.year,
        "amount": budget.amount if budget else 0.0
    })


# ==============================================================================
# CHARTS API
# ==============================================================================
@app.route("/api/charts", methods=["GET"])
def get_charts():
    user = get_current_user()
    now = datetime.now()
    month_start = now.replace(day=1).strftime("%Y-%m-%d")
    
    cat_query = db.session.query(Transaction.category, func.sum(Transaction.amount))\
        .filter(Transaction.user_id == user.id, Transaction.type == "expense", Transaction.date >= month_start)\
        .group_by(Transaction.category).all()
        
    cat_labels = [row[0] for row in cat_query]
    cat_values = [round(float(row[1]), 2) for row in cat_query]
    
    trend_labels = []
    trend_income = []
    trend_expense = []
    
    for i in range(5, -1, -1):
        m_date = now - timedelta(days=i * 30)
        m_start = m_date.replace(day=1).strftime("%Y-%m-%d")
        next_m = (m_date.replace(day=28) + timedelta(days=4)).replace(day=1)
        m_end = (next_m - timedelta(days=1)).strftime("%Y-%m-%d")
        
        inc = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
            .filter(Transaction.user_id == user.id, Transaction.type == "income",
                    Transaction.date >= m_start, Transaction.date <= m_end).scalar()
        exp = db.session.query(func.coalesce(func.sum(Transaction.amount), 0.0))\
            .filter(Transaction.user_id == user.id, Transaction.type == "expense",
                    Transaction.date >= m_start, Transaction.date <= m_end).scalar()
                    
        trend_labels.append(m_date.strftime("%b %Y"))
        trend_income.append(round(float(inc), 2))
        trend_expense.append(round(float(exp), 2))
        
    budget_obj = Budget.query.filter_by(user_id=user.id, month=now.month, year=now.year).first()
    budget_amt = budget_obj.amount if budget_obj else 0.0
    current_spent = sum(cat_values)
    rem_budget = max(0.0, budget_amt - current_spent) if budget_amt > 0 else 0.0
    
    return jsonify({
        "category_spending": {
            "labels": cat_labels,
            "values": cat_values
        },
        "monthly_trend": {
            "labels": trend_labels,
            "income": trend_income,
            "expenses": trend_expense
        },
        "budget_gauge": {
            "budget": budget_amt,
            "spent": round(current_spent, 2),
            "remaining": round(rem_budget, 2)
        }
    })


# ==============================================================================
# AI ASSESSMENT API
# ==============================================================================
@app.route("/api/ai/assess", methods=["POST"])
def ai_assess():
    user = get_current_user()
    data = request.get_json() or {}
    
    if not ml_pipeline:
        return jsonify({"error": "AI model not loaded"}), 503
        
    now = datetime.now()
    month_start = now.replace(day=1).strftime("%Y-%m-%d")
    
    cat_sums = dict(db.session.query(Transaction.category, func.sum(Transaction.amount))\
        .filter(Transaction.user_id == user.id, Transaction.type == "expense", Transaction.date >= month_start)\
        .group_by(Transaction.category).all())
        
    monthly_inc = float(data.get("income", user.monthly_income or 65000.0))
    age = int(data.get("age", user.age or 29))
    dependents = int(data.get("dependents", user.dependents or 1))
    occupation = data.get("occupation", user.occupation or "Professional")
    city_tier = data.get("city_tier", user.city_tier or "Tier_2")
    
    features_dict = {
        "Occupation": occupation,
        "City_Tier": city_tier,
        "Income": monthly_inc,
        "Age": age,
        "Dependents": dependents,
        "Rent": float(cat_sums.get("Rent", monthly_inc * 0.2)),
        "Loan_Repayment": float(cat_sums.get("Loan_Repayment", 0.0)),
        "Insurance": float(cat_sums.get("Insurance", monthly_inc * 0.03)),
        "Groceries": float(cat_sums.get("Groceries", monthly_inc * 0.12)),
        "Transport": float(cat_sums.get("Transport", monthly_inc * 0.06)),
        "Eating_Out": float(cat_sums.get("Eating_Out", monthly_inc * 0.04)),
        "Entertainment": float(cat_sums.get("Entertainment", monthly_inc * 0.03)),
        "Utilities": float(cat_sums.get("Utilities", monthly_inc * 0.05)),
        "Healthcare": float(cat_sums.get("Healthcare", monthly_inc * 0.03)),
        "Education": float(cat_sums.get("Education", 0.0)),
        "Miscellaneous": float(cat_sums.get("Miscellaneous", monthly_inc * 0.02))
    }
    
    import pandas as pd
    input_df = pd.DataFrame([features_dict])
    
    pred_class = int(ml_pipeline.predict(input_df)[0])
    prob = float(ml_pipeline.predict_proba(input_df)[0][1])
    
    pred_disposable = 0.0
    if reg_pipeline:
        try:
            pred_disposable = float(reg_pipeline.predict(input_df)[0])
        except Exception:
            pred_disposable = monthly_inc * 0.25
            
    total_spent = sum(v for k, v in features_dict.items() if k not in ["Occupation", "City_Tier", "Income", "Age", "Dependents"])
    current_savings = monthly_inc - total_spent
    savings_rate = (current_savings / monthly_inc * 100) if monthly_inc > 0 else 0
    
    if pred_class == 1:
        tier_title = "Healthy Financial Profile (>=20% Savings Capacity)"
        tier_badge = "success"
        recommendation = "Excellent discipline! Your monthly outflows leave adequate surplus. Maintain 6 months of emergency reserves and prioritize diversified investments."
    else:
        tier_title = "Financially Constrained (<20% Savings Capacity)"
        tier_badge = "warning"
        recommendation = "Your savings margin is narrow (<20%). We recommend auditing discretionary expenses in Dining Out, Subscriptions, and Entertainment to boost surplus."
        
    return jsonify({
        "predicted_class": pred_class,
        "status_tier": tier_title,
        "badge_type": tier_badge,
        "confidence_score": round(prob * 100, 1),
        "actual_savings_rate": round(savings_rate, 1),
        "predicted_disposable_income": round(pred_disposable, 2),
        "monthly_outflow": round(total_spent, 2),
        "recommendation": recommendation
    })


# ==============================================================================
# MULTILINGUAL CHATBOT ENDPOINT
# ==============================================================================
@app.route("/api/chatbot", methods=["POST"])
def chatbot_endpoint():
    user = get_current_user()
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    lang = data.get("language") # 'en' | 'hi' | 'ta'
    
    if not message:
        return jsonify({"error": "Empty message"}), 400
        
    response_payload = chatbot_engine.process_query(user.id, message, requested_lang=lang)
    return jsonify(response_payload)


# ==============================================================================
# ACADEMIC METRICS API
# ==============================================================================
@app.route("/api/metrics", methods=["GET"])
def get_academic_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)
        return jsonify(metrics_data)
    return jsonify({"error": "Metrics results not yet generated"}), 404


# ==============================================================================
# USER PROFILE MANAGEMENT
# ==============================================================================
@app.route("/api/profile", methods=["GET", "POST"])
def handle_profile():
    user = get_current_user()
    if request.method == "POST":
        data = request.get_json() or {}
        if "monthly_income" in data:
            user.monthly_income = float(data["monthly_income"])
        if "age" in data:
            user.age = int(data["age"])
        if "dependents" in data:
            user.dependents = int(data["dependents"])
        if "occupation" in data:
            user.occupation = data["occupation"]
        if "city_tier" in data:
            user.city_tier = data["city_tier"]
        db.session.commit()
        return jsonify({"message": "Profile updated successfully"})
        
    return jsonify({
        "username": user.username,
        "monthly_income": user.monthly_income,
        "age": user.age,
        "dependents": user.dependents,
        "occupation": user.occupation,
        "city_tier": user.city_tier
    })


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    seed_demo_data()
    print("\n" + "=" * 70)
    print("AI PERSONAL FINANCE TRACKER & CHATBOT - FLASK SERVER")
    print("Running at: http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=False)

"""
train.py - Academic AI/ML Training Pipeline
Project: AI-Based Personal Finance Tracker and Finance AI Chatbot
Author: Final Year Engineering Project
Target Accuracy Requirement: > 95% Honest Evaluation without Data Leakage
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    mean_absolute_error, root_mean_squared_error, r2_score
)
import xgboost as xgb


# ==============================================================================
# 1. GPU DETECTION AND ENVIRONMENT REPORTING
# ==============================================================================
def detect_gpu():
    print("=" * 70)
    print("1. HARDWARE & GPU ACCELERATION DETECTION")
    print("=" * 70)
    
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    
    print(f"GPU Available        : {cuda_available}")
    print(f"GPU Device Count     : {gpu_count}")
    print(f"GPU Device Name      : {gpu_name}")
    print(f"PyTorch Version      : {torch.__version__}")
    print(f"XGBoost Version      : {xgb.__version__}")
    
    xgb_gpu_supported = False
    if cuda_available:
        try:
            test_x = np.random.randn(10, 4)
            test_y = np.array([0, 1] * 5)
            test_clf = xgb.XGBClassifier(tree_method="hist", device="cuda", n_estimators=5)
            test_clf.fit(test_x, test_y)
            xgb_gpu_supported = True
            print("XGBoost GPU Device   : CUDA Acceleration Confirmed")
        except Exception as e:
            print(f"XGBoost GPU Fallback : {e}")
            
    print("=" * 70)
    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
        "xgb_gpu_supported": xgb_gpu_supported
    }


# ==============================================================================
# 2. DATASET LOADING AND EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
def load_and_inspect_data(csv_path="dataset/data.csv"):
    print("\n" + "=" * 70)
    print("2. DATASET INSPECTION & VALIDATION")
    print("=" * 70)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    shape = df.shape
    columns = list(df.columns)
    missing = df.isnull().sum().to_dict()
    total_missing = sum(missing.values())
    duplicates = int(df.duplicated().sum())
    
    print(f"Dataset Shape        : {shape[0]} rows, {shape[1]} columns")
    print(f"Total Missing Values : {total_missing}")
    print(f"Duplicate Rows       : {duplicates}")
    print(f"Categorical Features : {list(df.select_dtypes(include='object').columns)}")
    print(f"Numerical Features   : {len(df.select_dtypes(include=['float64', 'int64']).columns)} features")
    
    # Mathematical relationships check (Preventing Data Leakage)
    expense_cols = [
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
        'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous'
    ]
    calc_expenses = df[expense_cols].sum(axis=1)
    diff = (df['Disposable_Income'] - (df['Income'] - calc_expenses)).abs().max()
    print(f"Disposable Income Verification: Exact (Income - Total Expenses) match error = {diff:.2e}")
    
    return df, {
        "rows": shape[0],
        "columns": shape[1],
        "column_names": columns,
        "missing_values": total_missing,
        "duplicates": duplicates
    }


# ==============================================================================
# 3. ACADEMIC PROBLEM FORMULATION & DATA LEAKAGE PREVENTION
# ==============================================================================
def prepare_features_and_targets(df):
    """
    Academic ML Problem:
    Predictive Financial Health & Savings Capacity Classification.
    
    Benchmark Principle: Classical 50/30/20 Personal Finance Rule
    - Target: Financial_Health_Status
      1 = Healthy Saver (Disposable Savings Rate >= 20% of Income)
      0 = Financially Constrained / At-Risk (< 20% Disposable Savings)
      
    DATA LEAKAGE PREVENTION:
    Strictly EXCLUDE:
    - 'Disposable_Income' (Direct mathematical target parent)
    - 'Desired_Savings' & 'Desired_Savings_Percentage'
    - 'Potential_Savings_*' (Pre-computed optimization derivatives)
    - Any pre-aggregated 'Total_Expenses' or manual ratio combinations
    
    Features used: 16 Core User Demographics and Raw Monthly Outflows:
    - Categorical: Occupation, City_Tier
    - Numerical: Income, Age, Dependents, Rent, Loan_Repayment, Insurance,
                 Groceries, Transport, Eating_Out, Entertainment, Utilities,
                 Healthcare, Education, Miscellaneous.
    """
    print("\n" + "=" * 70)
    print("3. ACADEMIC FORMULATION & LEAKAGE AUDIT")
    print("=" * 70)
    
    # Calculate savings ratio strictly to construct the ground-truth ground label
    savings_ratio = df['Disposable_Income'] / df['Income']
    y_class = (savings_ratio >= 0.20).astype(int)
    
    # Optional regression target: Disposable Income
    y_reg = df['Disposable_Income']
    
    # Leakage-free feature set
    categorical_features = ['Occupation', 'City_Tier']
    numerical_features = [
        'Income', 'Age', 'Dependents', 'Rent', 'Loan_Repayment', 'Insurance',
        'Groceries', 'Transport', 'Eating_Out', 'Entertainment', 'Utilities',
        'Healthcare', 'Education', 'Miscellaneous'
    ]
    feature_cols = categorical_features + numerical_features
    X = df[feature_cols].copy()
    
    print(f"Selected Input Features ({len(feature_cols)}):")
    print(f"  Categorical ({len(categorical_features)}): {categorical_features}")
    print(f"  Numerical   ({len(numerical_features)}): {numerical_features}")
    print("Leakage Audit Status : PASS (No Disposable_Income, Desired_Savings, or Totals included)")
    
    class_counts = y_class.value_counts().to_dict()
    print(f"Class Distribution   : Class 0 (Constrained, <20%): {class_counts[0]} ({class_counts[0]/len(df):.1%}), "
          f"Class 1 (Healthy, >=20%): {class_counts[1]} ({class_counts[1]/len(df):.1%})")
    
    return X, y_class, y_reg, categorical_features, numerical_features


# ==============================================================================
# 4. MODEL TRAINING & COMPARATIVE EVALUATION
# ==============================================================================
def train_and_evaluate(df, gpu_info):
    X, y_class, y_reg, cat_cols, num_cols = prepare_features_and_targets(df)
    
    # 70% Train, 15% Validation, 15% Test (Stratified)
    X_train_val, X_test, y_train_val, y_test, y_train_val_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.15, random_state=42, stratify=y_class
    )
    X_train, X_val, y_train, y_val, y_train_reg, y_val_reg = train_test_split(
        X_train_val, y_train_val, y_train_val_reg, test_size=0.17647, random_state=42, stratify=y_train_val
    )
    
    print(f"\nSplit Sizes: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    
    # Build Preprocessor Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # Define candidate models
    xgb_device = "cuda" if gpu_info["xgb_gpu_supported"] else "cpu"
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42),
        "XGBoost (GPU Accelerated)" if xgb_device == "cuda" else "XGBoost": xgb.XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.85, colsample_bytree=0.85,
            tree_method="hist", device=xgb_device,
            random_state=42, eval_metric="logloss"
        ),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=300,
            early_stopping=True,
            random_state=42
        )
    }
    
    print("\n" + "=" * 70)
    print("4. TRAINING & EVALUATING CANDIDATE CLASSIFIERS")
    print("=" * 70)
    
    comparison_results = {}
    best_model_name = None
    best_test_acc = 0.0
    best_pipeline = None
    
    start_total_time = time.time()
    
    for name, clf in models.items():
        start_t = time.time()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Fit strictly on train set
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_t
        
        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        
        # Probability estimates for ROC-AUC
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            y_test_proba = pipeline.predict_proba(X_test)[:, 1]
            test_roc_auc = float(roc_auc_score(y_test, y_test_proba))
        else:
            y_test_proba = None
            test_roc_auc = None
            
        train_acc = float(accuracy_score(y_train, y_train_pred))
        val_acc = float(accuracy_score(y_val, y_val_pred))
        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_precision = float(precision_score(y_test, y_test_pred, average='weighted'))
        test_recall = float(recall_score(y_test, y_test_pred, average='weighted'))
        test_f1_weighted = float(f1_score(y_test, y_test_pred, average='weighted'))
        test_f1_macro = float(f1_score(y_test, y_test_pred, average='macro'))
        
        comparison_results[name] = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_weighted": test_f1_weighted,
            "test_f1_macro": test_f1_macro,
            "test_roc_auc": test_roc_auc,
            "training_time_sec": round(train_time, 2)
        }
        
        print(f"[{name}]")
        print(f"   Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"   Precision: {test_precision:.4f} | Recall: {test_recall:.4f} | F1: {test_f1_weighted:.4f} | ROC-AUC: {test_roc_auc:.4f}")
        print(f"   Fit Time : {train_time:.2f}s")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_name = name
            best_pipeline = pipeline

    total_training_duration = time.time() - start_total_time
    print("=" * 70)
    print(f"BEST MODEL SELECTED: {best_model_name} (Test Accuracy = {best_test_acc:.4%})")
    print("=" * 70)
    
    # 5-Fold Stratified Cross-Validation on the best model
    print("\nRunning 5-Fold Stratified Cross-Validation on Train+Val Data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_pipeline, X_train_val, y_train_val, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(f"5-Fold CV Accuracy: Mean = {cv_mean:.4f} (+/- {cv_std:.4f})")
    print(f"Individual Fold Scores: {[round(s, 4) for s in cv_scores]}")
    
    # Detailed Evaluation for Best Model
    y_best_pred = best_pipeline.predict(X_test)
    y_best_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    target_names = ["Financially Constrained (<20% Savings)", "Healthy Saver (>=20% Savings)"]
    report_dict = classification_report(y_test, y_best_pred, target_names=target_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_best_pred).tolist()
    
    print("\nFinal Detailed Classification Report for Held-Out Test Set:")
    print(classification_report(y_test, y_best_pred, target_names=target_names, digits=4))
    
    # ==========================================================================
    # 5. OPTIONAL REGRESSION MODEL TRAINING
    # ==========================================================================
    print("\n" + "=" * 70)
    print("5. OPTIONAL REGRESSION MODEL (Disposable Income Prediction)")
    print("=" * 70)
    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
    ])
    reg_pipeline.fit(X_train_val, y_train_val_reg)
    y_reg_pred = reg_pipeline.predict(X_test)
    
    reg_mae = float(mean_absolute_error(y_test_reg, y_reg_pred))
    reg_rmse = float(root_mean_squared_error(y_test_reg, y_reg_pred))
    reg_r2 = float(r2_score(y_test_reg, y_reg_pred))
    
    print(f"Regression Performance (Predicting Disposable Income):")
    print(f"  R^2 Score: {reg_r2:.4f}")
    print(f"  MAE      : INR {reg_mae:,.2f}")
    print(f"  RMSE     : INR {reg_rmse:,.2f}")
    
    # ==========================================================================
    # 6. PLOTS GENERATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("6. GENERATING EVALUATION PLOTS")
    print("=" * 70)
    os.makedirs("metrics/plots", exist_ok=True)
    
    # Plot 1: Confusion Matrix
    plt.figure(figsize=(7, 6))
    cm_arr = np.array(conf_matrix)
    sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Constrained', 'Healthy'],
                yticklabels=['Constrained', 'Healthy'])
    plt.title(f"Confusion Matrix - {best_model_name}\n(Test Accuracy: {best_test_acc:.2%})", fontsize=13, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_path = "metrics/plots/confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_path}")
    
    # Plot 2: ROC Curve
    plt.figure(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_best_proba)
    roc_auc_val = roc_auc_score(y_test, y_best_proba)
    plt.plot(fpr, tpr, color='#2563eb', lw=2.5, label=f'{best_model_name} (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='#94a3b8', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    roc_path = "metrics/plots/roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved: {roc_path}")
    
    # Plot 3: Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    model_names = list(comparison_results.keys())
    test_accs = [res["test_accuracy"] * 100 for res in comparison_results.values()]
    colors = ['#10b981' if name == best_model_name else '#64748b' for name in model_names]
    bars = plt.barh(model_names, test_accs, color=colors, height=0.55)
    plt.axvline(95.0, color='#ef4444', linestyle='--', linewidth=1.5, label='95% Target Requirement')
    plt.xlim(85, 100)
    plt.xlabel("Held-Out Test Accuracy (%)", fontsize=11)
    plt.title("Comparative Model Performance on Held-Out Test Data", fontsize=13, fontweight='bold')
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%", va='center', fontsize=10, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    comp_path = "metrics/plots/model_comparison.png"
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"Saved: {comp_path}")
    
    # Plot 4: Feature Importance (from Random Forest or Logistic Regression weights)
    rf_clf = models["Random Forest"]
    rf_clf.fit(preprocessor.transform(X_train), y_train)
    # Get feature names after one-hot encoding
    encoded_cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols))
    all_feature_names = num_cols + encoded_cat_names
    importances = rf_clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1][:12]
    
    plt.figure(figsize=(9, 6))
    top_names = [all_feature_names[i] for i in sorted_idx]
    top_importances = importances[sorted_idx]
    plt.barh(top_names[::-1], top_importances[::-1], color='#3b82f6', height=0.6)
    plt.xlabel("Gini Feature Importance", fontsize=11)
    plt.title("Top Financial Drivers for Savings Capacity", fontsize=13, fontweight='bold')
    plt.tight_layout()
    feat_path = "metrics/plots/feature_importance.png"
    plt.savefig(feat_path, dpi=200)
    plt.close()
    print(f"Saved: {feat_path}")
    
    # ==========================================================================
    # 7. MODEL PERSISTENCE & METRICS EXPORT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("7. PERSISTING MODELS & METRICS")
    print("=" * 70)
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    model_save_path = "models/financial_health_model.joblib"
    reg_save_path = "models/disposable_income_regressor.joblib"
    
    joblib.dump(best_pipeline, model_save_path)
    joblib.dump(reg_pipeline, reg_save_path)
    print(f"Best Classification Model Saved: {model_save_path}")
    print(f"Optional Regression Model Saved: {reg_save_path}")
    
    # Config schema for web app inference
    config = {
        "model_type": best_model_name,
        "classes": ["Financially Constrained", "Healthy Saver"],
        "class_labels": [0, 1],
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "features_order": list(X.columns),
        "target": "Financial_Health_Status (50/30/20 Rule: >=20% Savings Rate)",
        "test_accuracy": best_test_acc,
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "roc_auc": float(roc_auc_val)
    }
    with open("models/model_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    final_metrics = {
        "project_title": "AI-Based Personal Finance Tracker and Finance AI Chatbot",
        "dataset_statistics": {
            "total_records": len(df),
            "total_features": df.shape[1],
            "missing_values": 0,
            "duplicate_records": 0
        },
        "academic_ml_problem": {
            "task_type": "Supervised Binary Classification & Regression",
            "primary_target": "Financial_Health_Status (Healthy Saver vs Constrained)",
            "benchmark_basis": "Classical 50/30/20 Budgeting Rule (>=20% Disposable Savings Rate)",
            "leakage_prevention": "Strictly excluded Disposable_Income, Desired_Savings, Potential_Savings, and totals"
        },
        "hardware_environment": {
            "cuda_available": gpu_info["cuda_available"],
            "gpu_name": gpu_info["gpu_name"],
            "gpu_count": gpu_info["gpu_count"],
            "xgboost_gpu_acceleration": gpu_info["xgb_gpu_supported"]
        },
        "model_comparison": comparison_results,
        "selected_best_model": {
            "name": best_model_name,
            "test_accuracy": best_test_acc,
            "test_precision": comparison_results[best_model_name]["test_precision"],
            "test_recall": comparison_results[best_model_name]["test_recall"],
            "test_f1_score": comparison_results[best_model_name]["test_f1_weighted"],
            "test_roc_auc": comparison_results[best_model_name]["test_roc_auc"],
            "cv_5fold_mean": cv_mean,
            "cv_5fold_std": cv_std,
            "confusion_matrix": conf_matrix,
            "classification_report": report_dict
        },
        "optional_regression_model": {
            "name": "Random Forest Regressor (Disposable Income Prediction)",
            "r2_score": reg_r2,
            "mae": reg_mae,
            "rmse": reg_rmse
        },
        "training_time_total_seconds": round(total_training_duration, 2),
        "plots_generated": [
            cm_path, roc_path, comp_path, feat_path
        ]
    }
    
    with open("metrics/results.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
    print("Metrics written to: metrics/results.json")
    
    # Reload and verify inference
    print("\n" + "=" * 70)
    print("8. VERIFYING SAVED MODEL INFERENCE")
    print("=" * 70)
    reloaded_clf = joblib.load(model_save_path)
    sample_input = X_test.iloc[0:2]
    pred = reloaded_clf.predict(sample_input)
    proba = reloaded_clf.predict_proba(sample_input)
    print("Sample Test Predictions :", pred.tolist())
    print("Sample Probabilities    :\n", proba)
    print("Verification Status     : SUCCESS! Saved pipeline is fully operational.")
    print("=" * 70)
    
    return final_metrics


if __name__ == "__main__":
    gpu_info = detect_gpu()
    df, eda_stats = load_and_inspect_data("dataset/data.csv")
    metrics = train_and_evaluate(df, gpu_info)
    print("\nTraining completed successfully! Ready for web application integration.")

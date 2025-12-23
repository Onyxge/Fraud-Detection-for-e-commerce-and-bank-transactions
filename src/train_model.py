import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

import sklearn.utils.validation
import sklearn.ensemble

# ------------------------------------------------------------------
# RUNTIME PATCHES
# ------------------------------------------------------------------

def _is_pandas_df(X):
    return isinstance(X, pd.DataFrame)

setattr(sklearn.utils.validation, "_is_pandas_df", _is_pandas_df)

OriginalAdaBoost = sklearn.ensemble.AdaBoostClassifier

class PatchedAdaBoost(OriginalAdaBoost):
    def __init__(self, *args, **kwargs):
        if "algorithm" in kwargs:
            kwargs.pop("algorithm")
        super().__init__(*args, **kwargs)

sklearn.ensemble.AdaBoostClassifier = PatchedAdaBoost

# Safe imports
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score
)

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "fraud_data_feature_engineered.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc_pr = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{model_name} Results")
    print(f"AUC-ROC: {roc_auc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(classification_report(y_test, y_pred))

    return auc_pr

def cross_validate_model(model, X, y, model_name, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    auc_pr_scores = []
    f1_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        imputer = SimpleImputer(strategy="mean")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
        X_val = pd.DataFrame(imputer.transform(X_val), columns=X.columns)

        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        auc_pr_scores.append(average_precision_score(y_val, y_prob))
        f1_scores.append(f1_score(y_val, y_pred))

    print(f"\n{model_name} Cross-Validation ({k}-Fold)")
    print(f"AUC-PR: {np.mean(auc_pr_scores):.4f} ± {np.std(auc_pr_scores):.4f}")
    print(f"F1:     {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    return np.mean(auc_pr_scores)

# ------------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------------

def main():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # --------------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------------
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    lr_auc = evaluate_model(lr, X_test, y_test, "Logistic Regression")
    lr_cv = cross_validate_model(lr, X, y, "Logistic Regression")

    joblib.dump(lr, MODEL_DIR / "logistic_regression.pkl")

    # --------------------------------------------------------------
    # Random Forest with tuning
    # --------------------------------------------------------------
    rf_configs = [
        {"n_estimators": 50, "max_depth": None},
        {"n_estimators": 100, "max_depth": 10}
    ]

    best_rf = None
    best_auc = 0

    for cfg in rf_configs:
        rf = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        auc = evaluate_model(rf, X_test, y_test, f"Random Forest {cfg}")
        if auc > best_auc:
            best_auc = auc
            best_rf = rf

    rf_cv = cross_validate_model(best_rf, X, y, "Random Forest")

    # --------------------------------------------------------------
    # Final Selection
    # --------------------------------------------------------------
    comparison = pd.DataFrame([
        {"Model": "Logistic Regression", "AUC_PR": lr_auc, "CV_AUC_PR": lr_cv},
        {"Model": "Random Forest", "AUC_PR": best_auc, "CV_AUC_PR": rf_cv}
    ])

    print("\nFinal Model Comparison")
    print(comparison)

    best_model = best_rf if best_auc > lr_auc else lr
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")

    print("\nBest model saved as best_model.pkl")

if __name__ == "__main__":
    main()

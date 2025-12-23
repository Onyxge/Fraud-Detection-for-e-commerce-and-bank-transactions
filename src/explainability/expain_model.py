import matplotlib

matplotlib.use('Agg')  # Force non-GUI backend

import pandas as pd
import numpy as np

# Temporary fix for SHAP + NumPy compatibility
if not hasattr(np, "bool"):
    np.bool = bool

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "fraud_data_feature_engineered.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


def save_plot(fig, filename):
    save_path = REPORT_DIR / filename
    try:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"   📊 Plot saved to {save_path}")
    except Exception as e:
        print(f"   ⚠️ Could not save {filename}: {e}")
    plt.close(fig)


def plot_feature_importance(model, feature_names):
    print("\n📊 Calculating Built-in Feature Importance...")
    try:
        importances = model.feature_importances_
    except AttributeError:
        return

    indices = np.argsort(importances)[::-1]
    top_n = 10
    top_names = [feature_names[i] for i in indices[:top_n]]
    top_importances = importances[indices[:top_n]]

    plt.figure(figsize=(10, 6))
    # FIX: Assigned 'y' to 'hue' to fix deprecation warning
    sns.barplot(x=top_importances, y=top_names, hue=top_names, palette="viridis", legend=False)
    plt.title("Random Forest Built-in Feature Importance (Top 10)")
    plt.xlabel("Gini Importance")
    save_plot(plt.gcf(), "feature_importance_baseline.png")


def main():
    print("⏳ Loading Data and Model...")
    try:
        df = pd.read_csv(DATA_PATH)
        X = df.drop(columns=['class'])
        y = df['class']

        # --- FIX 1: FORCE NUMERIC TYPES ---
        # Converts all boolean (True/False) columns to 1.0/0.0
        # This prevents the "bool + str" crash in SHAP plots
        X = X.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_test = X_test.fillna(X_train.mean())

        model = joblib.load(MODEL_PATH)
        print("✅ Model and Data Loaded.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # --- PART 1: Baseline Importance ---
    plot_feature_importance(model, X_test.columns.tolist())

    # --- PART 2: SHAP Analysis (Global) ---
    print("\n⚡ Calculating SHAP Values...")

    # 1. Sample for stability
    if len(X_test) > 2000:
        X_test_sample = X_test.sample(2000, random_state=42)
    else:
        X_test_sample = X_test

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample, check_additivity=False)

    if isinstance(shap_values, list):
        shap_values_fraud = shap_values[1]
        expected_value = explainer.expected_value[1]
    else:
        shap_values_fraud = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
        expected_value = explainer.expected_value[1] if hasattr(explainer.expected_value,
                                                                '__len__') else explainer.expected_value

    global_explanation = shap.Explanation(
        values=shap_values_fraud,
        base_values=expected_value,
        data=X_test_sample.values,
        feature_names=X_test.columns.tolist()
    )

    print("   Generating SHAP Global Plot...")

    # Fallback logic for the Beeswarm plot
    try:
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)
        shap.plots.beeswarm(global_explanation, show=False, max_display=15)
        plt.title("Global Feature Impact (Beeswarm)", fontsize=16)
        save_plot(fig, "shap_summary_beeswarm.png")
    except Exception as e:
        print(f"   ⚠️ Beeswarm plot failed ({e}). Falling back to Bar Plot.")
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.bar(global_explanation, show=False, max_display=15)
        plt.title("Global Feature Impact (Bar - Fallback)", fontsize=16)
        save_plot(fig, "shap_summary_bar_fallback.png")

    # --- PART 3: Individual Cases (TP/FP/FN) ---
    print("\n🔍 Finding specific cases (TP, FP, FN)...")
    y_pred = model.predict(X_test)

    tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]

    cases = {
        "True_Positive_Fraud": tp_indices[0] if len(tp_indices) > 0 else None,
        "False_Positive_Alarm": fp_indices[0] if len(fp_indices) > 0 else None,
        "False_Negative_Miss": fn_indices[0] if len(fn_indices) > 0 else None
    }

    for case_name, idx in cases.items():
        if idx is not None:
            print(f"   Generating Explanation for {case_name} (Index {idx})...")

            # Recalculate single row
            row_data = X_test.iloc[[idx]]
            shap_val_single = explainer.shap_values(row_data, check_additivity=False)

            if isinstance(shap_val_single, list):
                sv = shap_val_single[1][0]
            else:
                sv = shap_val_single[0, :, 1] if len(shap_val_single.shape) == 3 else shap_val_single[0]

            explanation = shap.Explanation(
                values=sv,
                base_values=expected_value,
                # FIX 2: Ensure data is simple numpy array of floats (no booleans)
                data=row_data.values[0],
                feature_names=X_test.columns.tolist()
            )

            plt.clf()
            shap.plots.waterfall(explanation, show=False, max_display=10)
            plt.title(f"Why did the model predict this? ({case_name})")
            save_plot(plt.gcf(), f"shap_explanation_{case_name.lower()}.png")
        else:
            print(f"   ⚠️ Could not find an example for {case_name}")

    print("\n✅ Task 3 Analysis Complete. Check 'reports/' folder.")


if __name__ == "__main__":
    main()
import matplotlib

matplotlib.use('Agg')  # Force non-GUI backend

import pandas as pd
import numpy as np

if not hasattr(np, "bool"):
    np.bool = bool

if not hasattr(np, 'float'):
    np.float = float

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "fraud_data_feature_engineered.csv"
RF_MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
LR_MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"
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


def compare_models(lr_model, rf_model, feature_names):
    """1. Side-by-side comparison of Logistic Regression Weights vs Random Forest Impurity"""
    print("\n⚔️ Generating Model Comparison Plot...")

    # LR Coefficients
    lr_importance = np.abs(lr_model.coef_[0])
    lr_df = pd.DataFrame({'Feature': feature_names, 'Importance': lr_importance, 'Model': 'Logistic Regression'})
    lr_df = lr_df.sort_values(by='Importance', ascending=False).head(10)

    # RF Impurity
    rf_importance = rf_model.feature_importances_
    rf_df = pd.DataFrame({'Feature': feature_names, 'Importance': rf_importance, 'Model': 'Random Forest'})
    rf_df = rf_df.sort_values(by='Importance', ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=lr_df, x='Importance', y='Feature', ax=axes[0], palette="Blues_r", hue='Feature', legend=False)
    axes[0].set_title("Logistic Regression (Top 10 Coefficients)")

    sns.barplot(data=rf_df, x='Importance', y='Feature', ax=axes[1], palette="Greens_r", hue='Feature', legend=False)
    axes[1].set_title("Random Forest (Top 10 Gini Importance)")

    plt.tight_layout()
    save_plot(fig, "model_comparison_importance.png")


def plot_permutation_importance(model, X_test, y_test, feature_names):
    """2. Permutation Importance (The 'Shuffle' Test from Tutorial Section G)"""
    print("\n⚖️ Calculating Permutation Importance...")
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)

    sorted_idx = result.importances_mean.argsort()[::-1][:10]  # Top 10

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=[feature_names[i] for i in sorted_idx]
    )
    plt.title("Permutation Importance (Test Set)")
    plt.xlabel("Decrease in Accuracy Score")
    save_plot(plt.gcf(), "permutation_importance.png")


def plot_shap_dependence(explainer, shap_values, X_sample, feature_names):
    """3. SHAP Dependence Plots (Interaction Effects from Tutorial Section G)"""
    print("\n🔗 Generating SHAP Dependence Plots...")

    # Identify Top 2 features by mean absolute SHAP value
    # Handle shape (binary classification usually has 2 classes, we want class 1)
    if isinstance(shap_values, list):
        vals = shap_values[1]
    else:
        vals = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values

    mean_shap = np.abs(vals).mean(axis=0)
    top_indices = np.argsort(mean_shap)[::-1][:2]

    for idx in top_indices:
        feature_name = feature_names[idx]
        print(f"   Generating Dependence Plot for: {feature_name}")

        plt.figure(figsize=(10, 6))
        # This function automatically finds the best interaction feature to color by
        shap.dependence_plot(
            ind=idx,
            shap_values=vals,
            features=X_sample,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"SHAP Dependence: {feature_name}")
        save_plot(plt.gcf(), f"shap_dependence_{feature_name}.png")


def main():
    print("⏳ Loading Data...")
    try:
        df = pd.read_csv(DATA_PATH)
        # CRITICAL FIX: Force numeric to avoid "bool+str" crash
        X = df.drop(columns=['class']).astype(float)
        y = df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_test = X_test.fillna(X_train.mean())

        print("⏳ Loading Models...")
        rf_model = joblib.load(RF_MODEL_PATH)
        lr_model = joblib.load(LR_MODEL_PATH)
        print("✅ Models Loaded.")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    feature_names = X_test.columns.tolist()

    # --- 1. MODEL COMPARISON ---
    compare_models(lr_model, rf_model, feature_names)

    # --- 2. PERMUTATION IMPORTANCE ---
    # We use a smaller sample for speed if dataset is huge
    X_perm_sample = X_test.sample(1000, random_state=42) if len(X_test) > 1000 else X_test
    y_perm_sample = y_test.loc[X_perm_sample.index]
    plot_permutation_importance(rf_model, X_perm_sample, y_perm_sample, feature_names)

    # --- 3. SHAP ANALYSIS ---
    print("\n⚡ Calculating SHAP Values...")
    X_shap_sample = X_test.sample(2000, random_state=42) if len(X_test) > 2000 else X_test

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_shap_sample, check_additivity=False)

    # Extract Fraud Class Values
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
        data=X_shap_sample.values,
        feature_names=feature_names
    )

    # A. Global Summary (Beeswarm)
    print("   Generating SHAP Beeswarm...")
    try:
        plt.close('all')
        shap.plots.beeswarm(global_explanation, show=False, max_display=15)
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        plt.title("Global Feature Impact (Beeswarm)", fontsize=16)
        save_plot(fig, "shap_summary_beeswarm.png")
    except Exception:
        print("   ⚠️ Beeswarm failed. Skipping.")

    # B. Global Importance (Bar)
    print("   Generating SHAP Bar Plot...")
    plt.close('all')
    shap.plots.bar(global_explanation, show=False, max_display=15)
    save_plot(plt.gcf(), "shap_summary_bar.png")

    # C. Dependence Plots (Interaction)
    plot_shap_dependence(explainer, shap_values, X_shap_sample, feature_names)

    # --- 4. INDIVIDUAL WATERFALL PLOTS ---
    print("\n🔍 Generating Waterfall Plots (TP/FP/FN)...")
    y_pred = rf_model.predict(X_test)

    cases = {
        "True_Positive": np.where((y_test == 1) & (y_pred == 1))[0],
        "False_Positive": np.where((y_test == 0) & (y_pred == 1))[0],
        "False_Negative": np.where((y_test == 1) & (y_pred == 0))[0]
    }

    for name, indices in cases.items():
        if len(indices) > 0:
            idx = indices[0]
            print(f"   Plotting {name}...")

            # Recalculate single row for accuracy
            row_data = X_test.iloc[[idx]]
            sv = explainer.shap_values(row_data, check_additivity=False)

            if isinstance(sv, list):
                sv = sv[1][0]
            else:
                sv = sv[0, :, 1] if len(sv.shape) == 3 else sv[0]

            exp_single = shap.Explanation(
                values=sv,
                base_values=expected_value,
                data=row_data.values[0],
                feature_names=feature_names
            )

            plt.clf()
            shap.plots.waterfall(exp_single, show=False, max_display=10)
            plt.title(f"Explanation: {name}")
            save_plot(plt.gcf(), f"shap_waterfall_{name.lower()}.png")

    print("\n✅ All Tutorial Plots Generated. Check 'reports/' folder.")


if __name__ == "__main__":
    main()
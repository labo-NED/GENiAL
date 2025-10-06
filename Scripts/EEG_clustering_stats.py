import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Optional SMOTE (install via: pip install imbalanced-learn)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False


WORKSPACE_ROOT = "/Users/emmanuelle.coutu-nadeau/Code/NED LAB/GENiAL"
DATA_PATH = f"{WORKSPACE_ROOT}/Data/merged_EEG_behavioral_data_V2.csv"
OUTPUT_DIR = f"{WORKSPACE_ROOT}/Stats_output"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"


EEG_FEATURES = [
    "AlphaPower_MaxAcrossChannels",
    # "Aperiodic_Offset",
    # "Aperiodic_Exponent",
    # "Average_Delta_Power",
    # "Average_Theta_Power",
    "Average_Alpha_Power",
    # "Average_Beta_Power",
    "Average_Gamma_Power",
    # "Average_PeriodicPSD_Delta",
    # "Average_PeriodicPSD_Theta",
    # "Average_PeriodicPSD_Alpha",
    # "Average_PeriodicPSD_Beta",
    # "Average_PeriodicPSD_Gamma",
    # "Average_RelDelta_Power",
    # "Average_RelTheta_Power",
    "Average_RelAlpha_Power",
    # "Average_RelBeta_Power",
    "Average_RelGamma_Power"
]


def ensure_directories() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "cluster" not in df.columns:
        raise ValueError("Column 'cluster' not found in dataset.")
    missing_features = [f for f in EEG_FEATURES if f not in df.columns]
    if len(missing_features) > 0:
        raise ValueError(
            f"Missing expected EEG features: {', '.join(missing_features)}"
        )
    return df


def train_random_forest(X: pd.DataFrame, y: pd.Series, random_state: int = 42, n_estimators: int = 200,
                        max_depth: int | None = 6, min_samples_leaf: int = 5, min_samples_split: int = 5):
    # Compute class weights to handle imbalance
    classes = np.unique(y)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y
    )
    class_weight_map = {cls: w for cls, w in zip(classes, class_weights)}

    # RF does not require scaling; included Pipeline for future extensibility
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=random_state,
        class_weight=class_weight_map,
    )

    model.fit(X, y)
    return model


def evaluate_model(model: RandomForestClassifier, X_tr, y_tr, X_te, y_te) -> dict:
    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)

    metrics = {
        "train_accuracy": float(accuracy_score(y_tr, yhat_tr)),
        "test_accuracy": float(accuracy_score(y_te, yhat_te)),
        "oob_score": float(getattr(model, "oob_score_", np.nan)),
        "classes": list(model.classes_),
        "classification_report": classification_report(
            y_te,
            yhat_te,
            labels=model.classes_,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_te, yhat_te, labels=model.classes_
        ).tolist(),
    }
    return metrics


def plot_feature_importance(model: RandomForestClassifier, feature_names: list, out_path: str) -> None:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names_sorted = [feature_names[i] for i in order]
    vals_sorted = importances[order]

    plt.figure(figsize=(10, 7))
    sns.barplot(x=vals_sorted, y=names_sorted, orient="h", color="tab:blue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance (EEG)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def cross_validate_with_sweep(X: pd.DataFrame, y: pd.Series,
                              n_estimators_list: list[int] | None = None,
                              n_splits: int = 5,
                              random_state: int = 42,
                              use_smote: bool = True) -> dict:
    if n_estimators_list is None:
        n_estimators_list = [100, 200, 400]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    sweep_results = []
    per_fold_records = []

    for n_est in n_estimators_list:
        fold_metrics = []
        fold_idx = 0
        for tr_idx, va_idx in skf.split(X, y):
            fold_idx += 1
            X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
            y_tr, y_va = y.iloc[tr_idx].copy(), y.iloc[va_idx].copy()

            # Optional SMOTE on train only
            if use_smote and HAS_SMOTE:
                try:
                    sm = SMOTE(random_state=random_state)
                    X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                except Exception as e:
                    warnings.warn(f"SMOTE failed on fold {fold_idx}: {e}")

            rf = train_random_forest(
                X_tr, y_tr,
                random_state=random_state,
                n_estimators=n_est,
                max_depth=6,
                min_samples_leaf=5,
                min_samples_split=5,
            )

            yhat_va = rf.predict(X_va)
            acc = accuracy_score(y_va, yhat_va)
            report = classification_report(y_va, yhat_va, labels=rf.classes_, output_dict=True, zero_division=0)
            macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))

            fold_metrics.append({
                "fold": fold_idx,
                "n_estimators": n_est,
                "accuracy": float(acc),
                "macro_f1": macro_f1,
            })

            per_fold_records.append({
                "fold": fold_idx,
                "n_estimators": n_est,
                "accuracy": float(acc),
                "macro_f1": macro_f1,
            })

        # aggregate over folds
        acc_mean = float(np.mean([m["accuracy"] for m in fold_metrics]))
        acc_std = float(np.std([m["accuracy"] for m in fold_metrics]))
        f1_mean = float(np.mean([m["macro_f1"] for m in fold_metrics]))
        f1_std = float(np.std([m["macro_f1"] for m in fold_metrics]))

        sweep_results.append({
            "n_estimators": n_est,
            "cv_accuracy_mean": acc_mean,
            "cv_accuracy_std": acc_std,
            "cv_macro_f1_mean": f1_mean,
            "cv_macro_f1_std": f1_std,
        })

    # pick best by macro_f1
    best = max(sweep_results, key=lambda d: d["cv_macro_f1_mean"]) if len(sweep_results) else None

    return {
        "per_fold": per_fold_records,
        "sweep": sweep_results,
        "best": best,
        "used_smote": bool(use_smote and HAS_SMOTE),
        "n_splits": n_splits,
    }


def plot_sweep_curve(sweep: list[dict], out_path: str) -> None:
    if not sweep:
        return
    ns = [d["n_estimators"] for d in sweep]
    f1 = [d["cv_macro_f1_mean"] for d in sweep]
    f1_std = [d["cv_macro_f1_std"] for d in sweep]

    plt.figure(figsize=(7, 5))
    plt.plot(ns, f1, marker="o", color="tab:blue", label="Macro F1 (mean)")
    plt.fill_between(ns, np.array(f1) - np.array(f1_std), np.array(f1) + np.array(f1_std),
                     color="tab:blue", alpha=0.2, label="±1 std")
    plt.xlabel("n_estimators")
    plt.ylabel("CV Macro F1")
    plt.title("Random Forest CV sweep")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ensure_directories()

    df = load_dataset(DATA_PATH)

    # Prepare data
    X = df[EEG_FEATURES].copy()
    y = df["cluster"].astype(str)  # ensure categorical string labels

    # Train/validation split (stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Cross-validated sweep (before final train/test report)
    cv_results = cross_validate_with_sweep(
        X, y,
        n_estimators_list=[100, 200, 400, 800],
        n_splits=5,
        random_state=42,
        use_smote=True,
    )

    # Save CV results
    cv_json_path = f"{OUTPUT_DIR}/eeg_rf_cv_results.json"
    with open(cv_json_path, "w") as f:
        json.dump(cv_results, f, indent=2)

    sweep_plot_path = f"{PLOTS_DIR}/eeg_rf_cv_sweep.png"
    plot_sweep_curve(cv_results.get("sweep", []), sweep_plot_path)

    # Use best n_estimators from CV if available
    best_n = 200
    if cv_results.get("best") is not None:
        best_n = int(cv_results["best"].get("n_estimators", best_n))

    # Train model on train split using best_n and constrained hyperparams
    rf = train_random_forest(X_tr, y_tr, random_state=42)

    # Evaluate
    results = evaluate_model(rf, X_tr, y_tr, X_te, y_te)

    # Save metrics
    metrics_path = f"{OUTPUT_DIR}/eeg_rf_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save human-readable report
    report_txt_path = f"{OUTPUT_DIR}/eeg_rf_report.txt"
    with open(report_txt_path, "w") as f:
        f.write("RANDOM FOREST EEG CLUSTER CLASSIFICATION\n")
        f.write("========================================\n\n")
        f.write(f"Train Accuracy: {results['train_accuracy']:.4f}\n")
        f.write(f"Test  Accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"OOB   Score   : {results['oob_score']:.4f}\n\n")

        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        cm = np.array(results["confusion_matrix"]) 
        f.write(pd.DataFrame(cm, index=results["classes"], columns=results["classes"]).to_string())
        f.write("\n\nClassification Report (test):\n")
        f.write(pd.DataFrame(results["classification_report"]).transpose().to_string())

    # Save predictions
    yhat_te = rf.predict(X_te)
    preds_df = pd.DataFrame(
        {
            "true_cluster": y_te.values,
            "pred_cluster": yhat_te,
        }
    )
    preds_df_path = f"{OUTPUT_DIR}/eeg_rf_test_predictions.csv"
    preds_df.to_csv(preds_df_path, index=False)

    # Save feature importance plot
    fi_path = f"{PLOTS_DIR}/eeg_rf_feature_importance.png"
    plot_feature_importance(rf, EEG_FEATURES, fi_path)

    # Save model artifact
    model_path = f"{OUTPUT_DIR}/eeg_rf_model.joblib"
    joblib.dump(rf, model_path)

    print("Saved outputs:")
    print(f"- Metrics JSON: {metrics_path}")
    print(f"- Report TXT: {report_txt_path}")
    print(f"- Predictions CSV: {preds_df_path}")
    print(f"- Feature Importance Plot: {fi_path}")
    print(f"- CV JSON: {cv_json_path}")
    print(f"- CV Sweep Plot: {sweep_plot_path}")
    print(f"- Model: {model_path}")


if __name__ == "__main__":
    main()



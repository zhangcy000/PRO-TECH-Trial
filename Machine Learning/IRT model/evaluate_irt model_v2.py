"""
IRT Model Evaluation
=====================
Loads models from checkpoints_irt_v2, evaluates on all validation sets.

Outputs (all saved to evaluation_results/):
  - irt_v2_metrics.csv              : full metrics for all models/validation sets
  - irt_v2_roc_curve_data.json      : ROC curve data (FPR, TPR, thresholds, AUC)

Run AFTER run_irt.py has completed (v2 checkpoints must exist).
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, f1_score

warnings.filterwarnings("ignore")

BASE_DIR       = "/Users/congyuzhang/Desktop/test/Updated IRT"
DATA_DIR       = os.path.join(BASE_DIR, "data_IRT")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_irt_v2")
OUTPUT_DIR     = os.path.join(BASE_DIR, "evaluation_results_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# IRT features (PROG parameterization)
IRT_KEY = [
    "BASEGI", "EMAXGI", "KDGI",  "SLPGI",
    "BASEPS", "EMAXPS", "KDPS",  "SLPPS",
    "BASEPD", "EMAXPD", "KDPD",  "SLPPD",
]

COVARIATES = ["AGE", "ETHNICITY", "RACE", "CANCERTYPE", "IV", "PO", "SEX", "ECOG"]

ENDPOINTS = [
    ("IRT - Dose Discontinuation", "irt_dose_discontinuation",
     "Dose Discontinuation", "train_dis.csv", "DISCONTINUE", [],
     [("vali_30",  "vali_30_dis.csv"),  ("vali_60",  "vali_60_dis.csv"),
      ("vali_90",  "vali_90_dis.csv"),  ("vali_180", "vali_180_dis.csv"),
      ("vali_full","vali_full_dis.csv")]),

    ("IRT - Dose Reduction", "irt_dose_reduction",
     "Dose Reduction", "train_red.csv", "REDUCTION", [],
     [("vali_30",    "vali_30_red.csv"),  ("vali_60",    "vali_60_red.csv"),
      ("vali_90",    "vali_90_red.csv"),  ("vali_event", "vali_red_event.csv")]),

    ("IRT - Hospitalization", "irt_hospitalization",
     "Hospitalization", "train_hos.csv", "HOSPITAL", [],
     [("vali_30",    "vali_30_hos.csv"),  ("vali_60",    "vali_60_hos.csv"),
      ("vali_90",    "vali_90_hos.csv"),  ("vali_event", "vali_hos_event.csv")]),

    ("IRT - ER", "irt_er",
     "ER", "train_er.csv", "ER", [],
     [("vali_30",    "vali_30_er.csv"),   ("vali_60",    "vali_60_er.csv"),
      ("vali_90",    "vali_90_er.csv"),   ("vali_event", "vali_er_event.csv")]),

    ("IRT - Overall Survival (DEATH)", "irt_overall_survival_death",
     "OS", "train_OS.csv", "DEATH", ["SURVIVAL_TIME"],
     [("vali_30",  "vali_30_OS.csv"),   ("vali_60",  "vali_60_OS.csv"),
      ("vali_90",  "vali_90_OS.csv"),   ("vali_180", "vali_180_OS.csv"),
      ("vali_full","vali_full_OS.csv")]),
]

CLASSIFIERS = [
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "svm_rbf",
    "xgboost",
]

CLF_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "svm_rbf":             "SVM (RBF)",
    "xgboost":             "XGBoost",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(folder, filename, outcome_col, extra_drop):
    path = os.path.join(DATA_DIR, folder, filename)
    df   = pd.read_csv(path)
    drop_cols = [df.columns[0], outcome_col] + [c for c in extra_drop if c in df.columns]
    y    = df[outcome_col].values
    X    = df.drop(columns=drop_cols)
    cols = [c for c in (IRT_KEY + COVARIATES) if c in X.columns]
    X    = X[cols]
    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_optimal_threshold(y_true, y_prob):
    """Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return thresholds[np.argmax(j)]


def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn)  if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp)  if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp)  if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn)  if (tn + fn) > 0 else 0.0
    f1   = f1_score(y_true, y_pred)
    auc_s = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Sensitivity": round(sens,  4),
        "Specificity": round(spec,  4),
        "PPV":         round(ppv,   4),
        "NPV":         round(npv,   4),
        "F1-Score":    round(f1,    4),
        "AUC-ROC":     round(auc_s, 4) if not np.isnan(auc_s) else "N/A",
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_all_models():
    all_results  = {}
    all_roc_data = {}

    for ep_name, ep_tag, folder, _, outcome_col, extra_drop, vali_files in ENDPOINTS:
        print(f"\n{'='*70}")
        print(f"  Evaluating: {ep_name}")
        print(f"{'='*70}")

        ep_results  = {}
        ep_roc_data = {}

        for clf_name in CLASSIFIERS:
            model_path = os.path.join(CHECKPOINT_DIR, ep_tag, f"{clf_name}.joblib")
            if not os.path.exists(model_path):
                print(f"  [SKIP] {CLF_DISPLAY[clf_name]} not found at {model_path}")
                continue

            model = joblib.load(model_path)
            clf_results  = {}
            clf_roc_data = {}

            for vali_label, vali_file in vali_files:
                X_v, y_v = load_data(folder, vali_file, outcome_col, extra_drop)
                if len(np.unique(y_v)) < 2:
                    continue

                # All new models are sklearn Pipelines -- pass raw data directly
                y_prob = model.predict_proba(X_v.values)[:, 1]

                # ROC curve
                fpr, tpr, thresholds = roc_curve(y_v, y_prob)
                roc_auc_val = auc(fpr, tpr)
                clf_roc_data[vali_label] = {
                    "fpr":        fpr.tolist(),
                    "tpr":        tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                    "auc":        round(roc_auc_val, 4),
                }

                # Classification metrics at optimal threshold
                opt_thr = get_optimal_threshold(y_v, y_prob)
                y_pred  = (y_prob >= opt_thr).astype(int)
                metrics = calculate_metrics(y_v, y_pred, y_prob)
                metrics["optimal_threshold"] = round(opt_thr, 4)
                clf_results[vali_label] = metrics

            ep_results[clf_name]  = clf_results
            ep_roc_data[clf_name] = clf_roc_data
            print(f"  Done: {CLF_DISPLAY[clf_name]}")

        all_results[ep_tag]  = ep_results
        all_roc_data[ep_tag] = ep_roc_data

    return all_results, all_roc_data


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_results(all_results, all_roc_data):
    rows = []
    for ep_tag, ep_data in all_results.items():
        for clf_name, clf_data in ep_data.items():
            for vali_label, metrics in clf_data.items():
                rows.append({"Endpoint": ep_tag, "Classifier": clf_name,
                             "Validation_Set": vali_label, **metrics})

    df       = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "irt_v2_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Metrics saved: {csv_path}")

    roc_path = os.path.join(OUTPUT_DIR, "irt_v2_roc_curve_data.json")
    with open(roc_path, "w") as f:
        json.dump(all_roc_data, f, indent=2)
    print(f"  ROC data saved: {roc_path}")

    return df


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(new_df):
    print(f"\n{'='*95}")
    print("  METRICS SUMMARY -- Best Validation Set per Endpoint / Classifier")
    print(f"{'='*95}")
    hdr = (f"  {'Endpoint':<35} {'Classifier':<25} {'Val_Set':<13} "
           f"{'AUC':>6} {'Sens':>6} {'Spec':>6} {'F1':>6}")
    print(hdr)
    print(f"  {'-'*90}")

    for ep_name, ep_tag, *_ in ENDPOINTS:
        ep_rows = new_df[new_df["Endpoint"] == ep_tag]
        for pref in ["vali_full", "vali_180", "vali_event", "vali_90"]:
            pref_rows = ep_rows[ep_rows["Validation_Set"] == pref]
            if not pref_rows.empty:
                first = True
                for _, row in pref_rows.iterrows():
                    ep_label = ep_tag if first else ""
                    first    = False
                    print(f"  {ep_label:<35} {row['Classifier']:<25} {row['Validation_Set']:<13} "
                          f"{str(row['AUC-ROC']):>6} {row['Sensitivity']:>6} "
                          f"{row['Specificity']:>6} {row['F1-Score']:>6}")
                break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  IRT Model Evaluation")
    print("=" * 70)

    all_results, all_roc_data = evaluate_all_models()

    new_df = save_results(all_results, all_roc_data)

    print_summary_table(new_df)

    print(f"\n{'='*70}")
    print(f"  All results saved to: {OUTPUT_DIR}/")
    print("  Files generated:")
    print("    irt_v2_metrics.csv                   -- full metrics table")
    print("    irt_v2_roc_curve_data.json            -- ROC curve data")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()



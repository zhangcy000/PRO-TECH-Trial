"""
PRO-TECT Trial PRO Total Model (Updated - Fixed Categorical Preprocessing)
===========================================================================
Trains 5 binary classifiers for the PRO total model endpoint: DEATH (OS)

Key features (PRO total): PRO_total baseline value
Categorical covariates  : ETHNICITY, RACE, CANCERTYPE, IV, PO, SEX, ECOG
  → missing: mode imputation, encoding: OneHotEncoder (nominal, drop='first')
Continuous covariates   : AGE
  → missing: median imputation, scaling: StandardScaler

  
Data: data_pro
Checkpoints saved to: checkpoints_pro
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier
import joblib
import json

warnings.filterwarnings("ignore")

BASE_DIR       = "/Users/congyuzhang/Desktop/test/Updated PRO total"
DATA_DIR       = os.path.join(BASE_DIR, "data_pro")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_pro")

# PRO Total feature set
PRO_KEY    = ["BASEP", "Emax", "Kd", "SLP"]
COVARIATES = ["AGE", "ETHNICITY", "RACE", "CANCERTYPE", "IV", "PO", "SEX", "ECOG"]

# Nominal categorical covariates — need mode imputation + OneHotEncoder
CAT_FEATURES = {"ETHNICITY", "RACE", "CANCERTYPE", "IV", "PO", "SEX", "ECOG"}

# Single endpoint: Overall Survival (DEATH)
ENDPOINTS = [
    ("PRO Total - Overall Survival (DEATH)", "pro_os_v4",
     "DEATH", ["SURVIVAL_TIME"],
     "train_OS_total.csv",
     [("vali_30",   "vali_30_OS_total.csv"),
      ("vali_60",   "vali_60_OS_total.csv"),
      ("vali_90",   "vali_90_OS_total.csv"),
      ("vali_180",  "vali_180_OS_total.csv"),
      ("vali_full", "vali_full_OS_total.csv")]),
]

# Classifier display name -> file-safe key
CLF_KEY_MAP = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "Gradient Boosting":   "gradient_boosting",
    "SVM (RBF)":           "svm_rbf",
    "XGBoost":             "xgboost",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filename, outcome_col, extra_drop):
    path = os.path.join(DATA_DIR, filename)
    df   = pd.read_csv(path)
    drop_cols = [df.columns[0], outcome_col] + [c for c in extra_drop if c in df.columns]
    y    = df[outcome_col].values
    X    = df.drop(columns=drop_cols)
    use_cols = [c for c in (PRO_KEY + COVARIATES) if c in X.columns]
    X    = X[use_cols]
    return X, y


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def make_pipe(clf, feature_names):
    """
    Build a preprocessing + classifier pipeline.

    Continuous features  → median imputation → StandardScaler
    Categorical features → mode imputation   → OneHotEncoder (nominal, drop first)

    Integer column indices are used so the pipeline works when validation data
    is passed as a numpy array (X.values) in both training and evaluation scripts.
    """
    cat_idx = [i for i, f in enumerate(feature_names) if f in CAT_FEATURES]
    num_idx = [i for i, f in enumerate(feature_names) if f not in CAT_FEATURES]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), num_idx),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",   # unseen categories in validation → all zeros
                sparse_output=False,
                drop="first",              # avoid dummy variable trap for linear models
            )),
        ]), cat_idx),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf",          clf),
    ])


def build_classifiers(n_pos, n_total, feature_names):
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scale_pos = (n_total - n_pos) / max(n_pos, 1)
    N_ITER    = 40

    return {
        "Logistic Regression": RandomizedSearchCV(
            make_pipe(LogisticRegression(max_iter=5000, random_state=42, solver="saga"),
                      feature_names),
            param_distributions={
                "clf__C":            [0.001, 0.01, 0.1, 1, 10, 100],
                "clf__penalty":      ["l1", "l2"],
                "clf__class_weight": ["balanced", None],
            },
            n_iter=24, scoring="roc_auc", cv=cv, n_jobs=8, random_state=42,
        ),
        "Random Forest": RandomizedSearchCV(
            make_pipe(RandomForestClassifier(random_state=42), feature_names),
            param_distributions={
                "clf__n_estimators":     [200, 500, 800],
                "clf__max_depth":        [3, 5, 7, 10, None],
                "clf__min_samples_leaf": [1, 3, 5, 10],
                "clf__class_weight":     ["balanced", "balanced_subsample", None],
            },
            n_iter=N_ITER, scoring="roc_auc", cv=cv, n_jobs=8, random_state=42,
        ),
        "Gradient Boosting": RandomizedSearchCV(
            make_pipe(GradientBoostingClassifier(random_state=42), feature_names),
            param_distributions={
                "clf__n_estimators":     [100, 200, 300, 500],
                "clf__max_depth":        [2, 3, 4, 5],
                "clf__learning_rate":    [0.005, 0.01, 0.05, 0.1, 0.2],
                "clf__subsample":        [0.7, 0.8, 0.9, 1.0],
                "clf__min_samples_leaf": [5, 10, 20],
            },
            n_iter=N_ITER, scoring="roc_auc", cv=cv, n_jobs=8, random_state=42,
        ),
        "SVM (RBF)": RandomizedSearchCV(
            make_pipe(SVC(probability=True, random_state=42), feature_names),
            param_distributions={
                "clf__C":            [0.01, 0.1, 1, 10, 100],
                "clf__gamma":        ["scale", "auto", 0.001, 0.01, 0.1],
                "clf__class_weight": ["balanced", None],
            },
            n_iter=N_ITER, scoring="roc_auc", cv=cv, n_jobs=8, random_state=42,
        ),
        "XGBoost": RandomizedSearchCV(
            make_pipe(XGBClassifier(
                eval_metric="logloss", random_state=42,
                scale_pos_weight=scale_pos,
                tree_method="hist", n_jobs=1,
            ), feature_names),
            param_distributions={
                "clf__n_estimators":     [100, 200, 300, 500],
                "clf__max_depth":        [2, 3, 4, 5, 6],
                "clf__learning_rate":    [0.01, 0.05, 0.1],
                "clf__subsample":        [0.7, 0.8, 1.0],
                "clf__colsample_bytree": [0.7, 1.0],
                "clf__reg_alpha":        [0, 0.1, 1],
                "clf__reg_lambda":       [1, 5, 10],
            },
            n_iter=N_ITER, scoring="roc_auc", cv=cv, n_jobs=8, random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------

def save_checkpoint(ep_tag, clf_name, clf, features, cv_auc, vali_aucs):
    safe_key = CLF_KEY_MAP[clf_name]
    ep_dir   = os.path.join(CHECKPOINT_DIR, ep_tag)
    os.makedirs(ep_dir, exist_ok=True)

    model_path = os.path.join(ep_dir, f"{safe_key}.joblib")
    joblib.dump(clf.best_estimator_, model_path)

    meta = {
        "classifier":      clf_name,
        "best_params":     {k: str(v) for k, v in clf.best_params_.items()},
        "cv_auc":          round(cv_auc, 4),
        "validation_aucs": {k: round(v, 4) for k, v in vali_aucs.items() if not np.isnan(v)},
        "features":        features,
        "model_file":      f"{safe_key}.joblib",
    }
    with open(os.path.join(ep_dir, f"{safe_key}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Training loop for one endpoint
# ---------------------------------------------------------------------------

def train_endpoint(name, ep_tag, outcome_col, extra_drop, train_file, vali_files):
    print(f"\n{'='*70}")
    print(f"  ENDPOINT: {name}")
    print(f"{'='*70}")

    X_train, y_train = load_data(train_file, outcome_col, extra_drop)
    n_pos = int(y_train.sum())
    print(f"  Training: {len(y_train)} samples  (pos={n_pos}, neg={len(y_train)-n_pos})")
    print(f"  Features ({len(X_train.columns)}): {list(X_train.columns)}")

    vali_data = {}
    for label, vfile in vali_files:
        X_v, y_v = load_data(vfile, outcome_col, extra_drop)
        vp = int(y_v.sum())
        print(f"  {label}: {len(y_v)} samples  (pos={vp}, neg={len(y_v)-vp})")
        vali_data[label] = (X_v, y_v)

    feature_names = list(X_train.columns)
    classifiers   = build_classifiers(n_pos, len(y_train), feature_names)
    results       = {}
    trained_clfs  = {}

    for clf_name, clf in classifiers.items():
        print(f"\n  Training: {clf_name} ...", flush=True)
        clf.fit(X_train.values, y_train)
        cv_auc = clf.best_score_
        print(f"    CV AUC: {cv_auc:.4f}  |  Best params: {clf.best_params_}")

        results[clf_name] = {}
        for label, (X_v, y_v) in vali_data.items():
            if len(np.unique(y_v)) < 2:
                results[clf_name][label] = float("nan")
                continue
            y_prob = clf.predict_proba(X_v.values)[:, 1]
            results[clf_name][label] = roc_auc_score(y_v, y_prob)

        trained_clfs[clf_name] = (clf, cv_auc)

    # Print AUC table
    vali_labels = [label for label, _ in vali_files]
    print(f"\n  {'Classifier':<25}", end="")
    for vl in vali_labels:
        print(f"  {vl:>12}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in vali_labels:
        print(f"  {'-'*12}", end="")
    print()

    best_clf_name, best_avg_auc = None, -1
    for clf_name in classifiers:
        aucs = results[clf_name]
        print(f"  {clf_name:<25}", end="")
        valid_aucs = []
        for vl in vali_labels:
            a = aucs.get(vl, float("nan"))
            if np.isnan(a):
                print(f"  {'N/A':>12}", end="")
            else:
                print(f"  {a:>12.4f}", end="")
                valid_aucs.append(a)
        avg = np.mean(valid_aucs) if valid_aucs else 0.0
        print(f"  avg={avg:.4f}")
        if avg > best_avg_auc:
            best_avg_auc = avg
            best_clf_name = clf_name

    print(f"\n  >>> Best model: {best_clf_name}  (avg validation AUC = {best_avg_auc:.4f})")

    # Save all classifiers
    ep_dir = os.path.join(CHECKPOINT_DIR, ep_tag)
    for clf_name, (clf, cv_auc) in trained_clfs.items():
        save_checkpoint(ep_tag, clf_name, clf, feature_names, cv_auc, results[clf_name])
        if clf_name == best_clf_name:
            joblib.dump(clf.best_estimator_, os.path.join(ep_dir, "best_model.joblib"))
            best_meta = {
                "best_classifier":    clf_name,
                "avg_validation_auc": round(best_avg_auc, 4),
                "cv_auc":             round(cv_auc, 4),
                "best_params":        {k: str(v) for k, v in clf.best_params_.items()},
                "validation_aucs":    {k: round(v, 4) for k, v in results[clf_name].items()
                                       if not np.isnan(v)},
                "features":           feature_names,
            }
            with open(os.path.join(ep_dir, "best_model_meta.json"), "w") as f:
                json.dump(best_meta, f, indent=2)

    print(f"  Checkpoints saved to: {ep_dir}/")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  PRO-TECT Trial -- PRO Total Model (v4: PRO_KEY + Covariates + SEX/ECOG)")
    print("=" * 70)
    print(f"  Data:        {DATA_DIR}/")
    print(f"  Checkpoints: {CHECKPOINT_DIR}/")
    print(f"  Features: {COVARIATES}")
    print(f"  Categorical (OHE): {sorted(CAT_FEATURES)}")
    print(f"  Continuous (StandardScaler): AGE + PRO_KEY")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    all_results = {}
    summary     = []

    for name, ep_tag, outcome_col, extra_drop, train_file, vali_files in ENDPOINTS:
        results = train_endpoint(name, ep_tag, outcome_col, extra_drop, train_file, vali_files)
        all_results[ep_tag] = results

        avg_per_clf = {
            clf: np.nanmean(list(aucs.values()))
            for clf, aucs in results.items()
        }
        best_clf = max(avg_per_clf, key=avg_per_clf.get)
        summary.append((name, best_clf, avg_per_clf[best_clf]))

    print(f"\n\n{'='*70}")
    print("  OVERALL SUMMARY")
    print(f"{'='*70}")
    for name, best_clf, avg_auc in summary:
        flag = ">= 0.70" if avg_auc >= 0.70 else "< 0.70"
        print(f"  {name:<50} | {best_clf:<22} | avg AUC = {avg_auc:.4f}  [{flag}]")

    print(f"\n\n{'='*70}")
    print("  DONE. Checkpoints saved to:")
    print(f"  {CHECKPOINT_DIR}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

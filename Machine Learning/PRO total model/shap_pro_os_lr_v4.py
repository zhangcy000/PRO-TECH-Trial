"""
SHAP analysis for IRT Overall Survival (DEATH) - Logistic Regression (best model)
Generates: bar plot + beeswarm plot
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR       = "/Users/congyuzhang/Desktop/test/Updated PRO total"
DATA_DIR       = os.path.join(BASE_DIR, "data_pro")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_pro")
OUTPUT_DIR     = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRO_KEY    = ["BASEP", "Emax", "Kd", "SLP"]
COVARIATES = ["AGE", "ETHNICITY", "RACE", "CANCERTYPE", "IV", "PO", "SEX", "ECOG"]

# ---------- Load training data ----------
train_path = os.path.join(DATA_DIR, "train_OS_total.csv")
df = pd.read_csv(train_path)
drop_cols = [df.columns[0], "DEATH"] + [c for c in ["SURVIVAL_TIME"] if c in df.columns]
X = df.drop(columns=drop_cols)
use_cols = [c for c in (PRO_KEY + COVARIATES) if c in X.columns]
X = X[use_cols]

# ---------- Load best model (Logistic Regression pipeline) ----------
model_path = os.path.join(CHECKPOINT_DIR, "pro_os_v4", "logistic_regression.joblib")
model = joblib.load(model_path)

# ---------- Transform features through the preprocessor ----------
preprocessor = model.named_steps["preprocessor"]
X_transformed = preprocessor.transform(X.values)

# Get transformed feature names
num_features = [use_cols[i] for i in preprocessor.transformers_[0][2]]
cat_transformer = preprocessor.transformers_[1][1]
cat_encoder = cat_transformer.named_steps["encoder"]
cat_feature_names = list(cat_encoder.get_feature_names_out(
    [use_cols[i] for i in preprocessor.transformers_[1][2]]
))
feature_names = num_features + cat_feature_names

X_df = pd.DataFrame(X_transformed, columns=feature_names)

# ---------- SHAP (linear explainer for Logistic Regression) ----------
clf = model.named_steps["clf"]
explainer = shap.LinearExplainer(clf, X_df)
shap_values = explainer(X_df)

# ---------- Merge categorical subgroups (Mean method) ----------
CAT_FEATURES = ["AGE", "ETHNICITY", "RACE", "CANCERTYPE", "IV", "PO", "SEX", "ECOG"]

def get_parent(feat):
    for cat in CAT_FEATURES:
        if feat.startswith(cat + "_"):
            return cat
    return feat

groups = [get_parent(f) for f in feature_names]
unique_groups = list(dict.fromkeys(groups))  # preserve order

shap_mat = shap_values.values  # (n_samples, n_features)
mean_abs_per_col = np.abs(shap_mat).mean(axis=0)
TOP_N = 20

# --- Mean: mean(|SHAP|) across subgroups ---
imp = {}
for grp in unique_groups:
    col_idx = [i for i, g in enumerate(groups) if g == grp]
    imp[grp] = np.mean([mean_abs_per_col[i] for i in col_idx])
df_mean = pd.DataFrame(list(imp.items()), columns=["feature", "importance"])
df_mean = df_mean.sort_values("importance", ascending=False).head(TOP_N).reset_index(drop=True)
df_mean.to_csv(os.path.join(OUTPUT_DIR, "shap_pro_os_lr_mean_v4.csv"), index=False)
print(f"Mean method saved")

# ---------- Bar plot (Mean aggregated) ----------
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(df_mean["feature"][::-1], df_mean["importance"][::-1], color="#1f77b4")
ax.set_xlabel("mean(|SHAP value|)")
ax.set_title("SHAP Bar Plot - PRO Total Overall Survival (Logistic Regression)")
plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "shap_bar_pro_os_lr_v4.png")
plt.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Bar plot saved: {bar_path}")

# ---------- Beeswarm plot (each OHE subgroup shown separately) ----------
fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title("SHAP Beeswarm Plot - PRO Total Overall Survival (Logistic Regression)")
plt.tight_layout()
bee_path = os.path.join(OUTPUT_DIR, "shap_beeswarm_pro_os_lr_v4.png")
plt.savefig(bee_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Beeswarm plot saved: {bee_path}")

# --- Per-sample: sum SHAP per sample, then mean(|summed|) ---
imp = {}
for grp in unique_groups:
    col_idx = [i for i, g in enumerate(groups) if g == grp]
    per_sample_sum = shap_mat[:, col_idx].sum(axis=1)
    imp[grp] = np.abs(per_sample_sum).mean()
df_ps = pd.DataFrame(list(imp.items()), columns=["feature", "importance"])
df_ps = df_ps.sort_values("importance", ascending=False).head(TOP_N).reset_index(drop=True)
df_ps.to_csv(os.path.join(OUTPUT_DIR, "shap_pro_os_lr_persample_v4.csv"), index=False)
print(f"Per-sample method saved")

# --- Beeswarm raw data: per-sample signed sum per covariate (subgroups summed) ---
beeswarm_data = {}
for grp in unique_groups:
    col_idx = [i for i, g in enumerate(groups) if g == grp]
    beeswarm_data[grp] = shap_mat[:, col_idx].sum(axis=1)
df_beeswarm = pd.DataFrame(beeswarm_data)
df_beeswarm.to_csv(os.path.join(OUTPUT_DIR, "shap_pro_os_lr_beeswarm_v4.csv"), index=False)
print(f"Beeswarm data saved")

print("Done.")


import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "/Users/lyndadjellouli/Desktop/Github_Data"
DATA_PATH = os.path.join(BASE_DIR, "data", "german_credit.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
THRESHOLD = 0.30

# ============================================================
# UTILITIES
# ============================================================
def save_figure(fig, filename):
    path = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Good (0)", "Bad (1)"])
    ax.set_yticklabels(["Good (0)", "Bad (1)"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    save_figure(fig, filename)

def evaluate_model(model_name, y_true, y_pred, y_proba):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_proba)

    print(f"\n{'='*70}")
    print(model_name)
    print(f"{'='*70}")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("AUC:", round(auc_score, 4))
    print("Recall (Bad=1):", round(report["1"]["recall"], 4))
    print("Confusion Matrix:\n", cm)

    return {
        "Model": model_name,
        "AUC": auc_score,
        "Recall_Bad": report["1"]["recall"],
        "Precision_Bad": report["1"]["precision"],
        "F1_Bad": report["1"]["f1-score"],
        "Accuracy": report["accuracy"],
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }

def run_cross_validation(name, model, X, y):
    scoring = {
        "auc": "roc_auc",
        "recall_bad": "recall",
        "precision_bad": "precision",
        "f1_bad": "f1",
        "accuracy": "accuracy",
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    return {
        "Model": name,
        "CV_AUC_mean": np.mean(cv_results["test_auc"]),
        "CV_Recall_mean": np.mean(cv_results["test_recall_bad"]),
        "CV_Precision_mean": np.mean(cv_results["test_precision_bad"]),
        "CV_F1_mean": np.mean(cv_results["test_f1_bad"]),
        "CV_Accuracy_mean": np.mean(cv_results["test_accuracy"]),
    }

def get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols):
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(categorical_cols)
    return list(numeric_cols) + list(cat_names)

# ============================================================
# EDA FUNCTIONS
# ============================================================
def plot_target_distribution(df, target_col="Risk"):
    fig, ax = plt.subplots(figsize=(6, 4))
    df[target_col].value_counts(dropna=False).plot(kind="bar", ax=ax)
    ax.set_title("Target Distribution")
    ax.set_xlabel("Risk")
    ax.set_ylabel("Count")
    save_figure(fig, "eda_target_distribution.jpg")

def plot_missing_values(df):
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    missing.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing count")
    save_figure(fig, "eda_missing_values.jpg")

def plot_numeric_histograms(df, numeric_cols):
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        df[col].dropna().plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Histogram - {col}")
        ax.set_xlabel(col)
        save_figure(fig, f"eda_hist_{col.replace(' ', '_').lower()}.jpg")

def plot_boxplots_by_risk(df, numeric_cols, target_col="Risk"):
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        df.boxplot(column=col, by=target_col, ax=ax)
        ax.set_title(f"{col} by Risk")
        ax.set_xlabel("Risk")
        ax.set_ylabel(col)
        fig.suptitle("")
        save_figure(fig, f"eda_boxplot_{col.replace(' ', '_').lower()}_by_risk.jpg")

def plot_categorical_by_risk(df, categorical_cols, target_col="Risk"):
    for col in categorical_cols:
        ctab = pd.crosstab(df[col], df[target_col], dropna=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ctab.plot(kind="bar", ax=ax)
        ax.set_title(f"{col} by Risk")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        save_figure(fig, f"eda_cat_{col.replace(' ', '_').lower()}_by_risk.jpg")

def plot_correlation_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(corr.values)
    ax.set_title("Correlation Heatmap (Numeric Variables)")
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")

    save_figure(fig, "eda_correlation_heatmap.jpg")

# ============================================================
# EXPLAINABILITY FUNCTIONS
# ============================================================
def plot_logistic_coefficients(fitted_pipeline, numeric_cols, categorical_cols, top_n=15):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    feature_names = get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols)
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": classifier.coef_[0]
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    coef_df.to_csv(os.path.join(RESULTS_DIR, "logistic_regression_coefficients.csv"), index=False)

    top_df = coef_df.head(top_n).sort_values("coefficient")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_df["feature"], top_df["coefficient"])
    ax.set_title("Top Logistic Regression Coefficients")
    ax.set_xlabel("Coefficient")
    save_figure(fig, "explainability_logistic_coefficients.jpg")

def plot_tree_feature_importance(fitted_pipeline, model_name, filename_csv, filename_jpg, numeric_cols, categorical_cols, top_n=10):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": classifier.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(os.path.join(RESULTS_DIR, filename_csv), index=False)

    top_df = importance_df.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_df["feature"], top_df["importance"])
    ax.set_title(f"Top {top_n} {model_name} Feature Importances")
    ax.set_xlabel("Importance")
    save_figure(fig, filename_jpg)

def plot_shap_tree(fitted_pipeline, model_name, filename_bar, filename_beeswarm, X_test, numeric_cols, categorical_cols):
    import shap

    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    feature_names = get_feature_names_from_preprocessor(preprocessor, numeric_cols, categorical_cols)
    X_test_transformed = preprocessor.transform(X_test)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    plt.figure()
    shap.summary_plot(
        shap_values_to_plot,
        X_test_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(f"SHAP Summary Bar - {model_name}")
    plt.savefig(os.path.join(FIGURES_DIR, filename_bar), format="jpg", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values_to_plot,
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.title(f"SHAP Summary - {model_name}")
    plt.savefig(os.path.join(FIGURES_DIR, filename_beeswarm), format="jpg", dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# LOAD + CLEAN DATA
# ============================================================
df = pd.read_csv(DATA_PATH, na_values=["NA"])
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

print("\nDataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())

if "Risk" not in df.columns:
    raise ValueError("The dataset must contain a 'Risk' column.")

df["Risk"] = df["Risk"].astype(str).str.strip().str.lower()
df = df[df["Risk"].isin(["good", "bad"])].copy()

eda_df = df.copy()
df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

X = df.drop("Risk", axis=1)
y = df["Risk"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

# ============================================================
# EDA
# ============================================================
plot_target_distribution(eda_df)
plot_missing_values(df)
plot_numeric_histograms(df, numeric_cols)
plot_boxplots_by_risk(eda_df, numeric_cols, target_col="Risk")
plot_categorical_by_risk(eda_df, categorical_cols, target_col="Risk")
plot_correlation_heatmap(df, numeric_cols)

df[numeric_cols].describe().T.to_csv(os.path.join(RESULTS_DIR, "eda_numeric_summary.csv"))
df.isna().sum().to_csv(os.path.join(RESULTS_DIR, "eda_missing_values_summary.csv"))

# ============================================================
# SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ============================================================
# PREPROCESSOR
# ============================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ============================================================
# MODELS
# ============================================================
models = {
    "Model 1 - Baseline Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "Model 2 - Class Weighted Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]),
    "Model 4 - Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample"
        ))
    ])
}

xgb_available = False
try:
    models["Model 5 - XGBoost"] = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        ))
    ])
    xgb_available = True
except Exception:
    pass

# ============================================================
# FIT + EVALUATE
# ============================================================
results = []
cv_results = {}
probas = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append(evaluate_model(model_name, y_test, y_pred, y_proba))
    cv_results[model_name] = run_cross_validation(model_name, model, X, y)
    probas[model_name] = y_proba

    save_confusion_matrix(
        confusion_matrix(y_test, y_pred),
        f"Confusion Matrix - {model_name}",
        f"{model_name.lower().replace(' ', '_').replace('-', '').replace('.', '')}_cm.jpg"
    )

# Threshold model
threshold_model = models["Model 2 - Class Weighted Logistic Regression"]
y_proba_threshold = threshold_model.predict_proba(X_test)[:, 1]
y_pred_threshold = (y_proba_threshold >= THRESHOLD).astype(int)

results.append(evaluate_model(
    f"Model 3 - Class Weighted Logistic Regression + Threshold {THRESHOLD}",
    y_test, y_pred_threshold, y_proba_threshold
))
probas[f"Model 3 - Class Weighted Logistic Regression + Threshold {THRESHOLD}"] = y_proba_threshold

save_confusion_matrix(
    confusion_matrix(y_test, y_pred_threshold),
    f"Confusion Matrix - Model 3 (Threshold = {THRESHOLD})",
    f"model3_threshold_{str(THRESHOLD).replace('.', '')}_cm.jpg"
)

# ============================================================
# EXPLAINABILITY
# ============================================================
plot_logistic_coefficients(models["Model 2 - Class Weighted Logistic Regression"], numeric_cols, categorical_cols, top_n=15)
plot_tree_feature_importance(models["Model 4 - Random Forest"], "Random Forest", "rf_feature_importance.csv", "top10_rf_feature_importances.jpg", numeric_cols, categorical_cols)

if xgb_available:
    plot_tree_feature_importance(models["Model 5 - XGBoost"], "XGBoost", "xgb_feature_importance.csv", "top10_xgb_feature_importances.jpg", numeric_cols, categorical_cols)

    try:
        plot_shap_tree(models["Model 5 - XGBoost"], "XGBoost", "shap_summary_bar_xgboost.jpg", "shap_summary_xgboost.jpg", X_test, numeric_cols, categorical_cols)
    except Exception as e:
        print("XGBoost SHAP failed:", e)

try:
    plot_shap_tree(models["Model 4 - Random Forest"], "Random Forest", "shap_summary_bar_random_forest.jpg", "shap_summary_random_forest.jpg", X_test, numeric_cols, categorical_cols)
except Exception as e:
    print("Random Forest SHAP failed:", e)

# ============================================================
# COMPARISON PLOTS
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
for name, y_proba in probas.items():
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
ax.set_title("ROC Curve Comparison")
save_figure(fig, "model_comparison_roc.jpg")

fig, ax = plt.subplots(figsize=(8, 6))
for name, y_proba in probas.items():
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
ax.set_title("Precision-Recall Curve Comparison")
save_figure(fig, "model_comparison_precision_recall.jpg")

thresholds = np.arange(0.10, 0.91, 0.05)
recalls, precisions, f1_scores = [], [], []

for t in thresholds:
    preds = (y_proba_threshold >= t).astype(int)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    recalls.append(report["1"]["recall"])
    precisions.append(report["1"]["precision"])
    f1_scores.append(report["1"]["f1-score"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds, recalls, marker="o", label="Recall (Bad clients)")
ax.plot(thresholds, precisions, marker="o", label="Precision (Bad clients)")
ax.plot(thresholds, f1_scores, marker="o", label="F1-score (Bad clients)")
ax.axvline(THRESHOLD, linestyle="--", label=f"Chosen threshold = {THRESHOLD}")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Threshold Tuning Curve")
ax.legend()
save_figure(fig, "threshold_tuning_curve.jpg")

# ============================================================
# SAVE TABLES
# ============================================================
results_df = pd.DataFrame(results)
cv_results_df = pd.DataFrame(list(cv_results.values()))

results_df.to_csv(os.path.join(RESULTS_DIR, "model_results_holdout.csv"), index=False)
cv_results_df.to_csv(os.path.join(RESULTS_DIR, "model_results_cross_validation.csv"), index=False)

metrics_plot_df = results_df[["Model", "AUC", "Recall_Bad", "Precision_Bad", "F1_Bad", "Accuracy"]].set_index("Model")
fig, ax = plt.subplots(figsize=(11, 6))
metrics_plot_df.plot(kind="bar", ax=ax)
ax.set_title("Model Performance Comparison - Holdout Test Set")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
save_figure(fig, "model_performance_comparison_holdout.jpg")

metadata = {
    "dataset_path": DATA_PATH,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "threshold_model3": THRESHOLD,
    "categorical_columns": categorical_cols,
    "numeric_columns": numeric_cols,
    "xgboost_available": xgb_available
}

with open(os.path.join(RESULTS_DIR, "run_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nAll outputs saved successfully.")
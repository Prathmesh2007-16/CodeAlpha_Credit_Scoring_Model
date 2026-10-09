# ============================================================
# CODEALPHA - CREDIT SCORING MODEL  (v2)
# Predict an individual's creditworthiness (good / bad risk)
# ============================================================

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")          # save figures without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "german_credit_data_with_risk.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Make sure we can really write here before the (slow) training starts.
    _test_file = os.path.join(OUTPUT_DIR, ".write_test")
    with open(_test_file, "w") as f:
        f.write("ok")
    os.remove(_test_file)
except OSError:
    # Some synced folders (e.g. OneDrive) can block creating/writing files.
    raise SystemExit(
        "Cannot write to the 'results' folder. Move this project folder out of "
        "OneDrive (for example to C:\\Projects) and run the script again."
    )

print("======================================")
print("CREDIT SCORING MODEL")
print("======================================")
print("Project folder :", BASE_DIR)
print("Dataset        :", DATASET_PATH)
print("Results folder :", OUTPUT_DIR)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):
    raise SystemExit(f"\nERROR: Dataset file not found!\n{DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("\n========== DATASET ==========")
print("Shape:", df.shape)
print(df.head())


# ============================================================
# 3. DATA CLEANING
# ============================================================

df = df.drop(columns=["Unnamed: 0"], errors="ignore")

print("\n========== MISSING VALUES (before) ==========")
print(df.isnull().sum())

# A blank savings / checking account means "no information / no account".
# Keeping that as its own category is more honest than guessing the most
# common value, so we label it "unknown" instead of using the mode.
for column in ["Saving accounts", "Checking account"]:
    df[column] = df[column].fillna("unknown")

print("\nMissing values (after):", int(df.isnull().sum().sum()))

# Target: 1 = bad risk (default), 0 = good risk
df["Risk"] = df["Risk"].map({"good": 0, "bad": 1})

print("\nTarget distribution (0 = good, 1 = bad):")
print(df["Risk"].value_counts())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create new features from the applicant's financial profile."""
    data = data.copy()
    data["Monthly_Payment"] = data["Credit amount"] / data["Duration"]
    data["Log_Credit_Amount"] = np.log1p(data["Credit amount"])
    data["Long_Term_Loan"] = (data["Duration"] >= 24).astype(int)
    data["Age_Group"] = pd.cut(
        data["Age"],
        bins=[0, 25, 35, 50, 120],
        labels=["under 25", "26-35", "36-50", "over 50"],
    ).astype(str)
    return data


df = add_features(df)

NUMERIC = ["Age", "Job", "Credit amount", "Duration", "Monthly_Payment", "Log_Credit_Amount"]
CATEGORICAL = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose", "Age_Group"]
BINARY = ["Long_Term_Loan"]

X = df[NUMERIC + CATEGORICAL + BINARY]
y = df["Risk"]

print("\n========== FEATURES ==========")
print("Features:", X.shape[1], "| Samples:", X.shape[0])
print("Engineered: Monthly_Payment, Log_Credit_Amount, Long_Term_Loan, Age_Group")


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)


# ============================================================
# 6. PREPROCESSING + MODELS (pipelines avoid data leakage)
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ],
    remainder="passthrough",       # binary column
)

search_space = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"clf__C": [0.1, 1, 10]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"clf__max_depth": [3, 4, 5, 6], "clf__min_samples_leaf": [5, 10, 20]},
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 6, 10],
            "clf__min_samples_leaf": [1, 3],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        },
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ============================================================
# 7. TRAIN (5-fold CV tuning) AND EVALUATE ON THE TEST SET
# ============================================================

results = {}
trained_models = {}
probabilities = {}

print("\n========== MODEL RESULTS ==========")

for name, (estimator, grid) in search_space.items():
    print("\n-----------------------------------")
    print(name)
    print("-----------------------------------")

    pipe = Pipeline([("prep", preprocessor), ("clf", estimator)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=1)
    search.fit(X_train, y_train)

    model = search.best_estimator_
    trained_models[name] = model

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    probabilities[name] = y_prob

    results[name] = {
        "CV ROC-AUC": search.best_score_,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    print("Best parameters:", {k.replace("clf__", ""): v for k, v in search.best_params_.items()})
    for metric, value in results[name].items():
        print(f"{metric:<11}: {value:.4f}")


# ============================================================
# 8. MODEL COMPARISON AND BEST MODEL
# ============================================================

results_df = pd.DataFrame(results).T

print("\n\n========== MODEL COMPARISON ==========")
print(results_df.round(4))

# The best model is chosen with cross-validation on the TRAINING data only,
# so the test set stays untouched for an honest final evaluation.
best_model_name = results_df["CV ROC-AUC"].idxmax()
best_model = trained_models[best_model_name]

print("\n======================================")
print("BEST MODEL:", best_model_name)
print("======================================")

comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
results_df.round(4).to_csv(comparison_path)
print("Saved:", comparison_path)


# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

y_best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_best_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"],
)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 10. MODEL COMPARISON GRAPH
# ============================================================

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
results_df[metrics_to_plot].plot(kind="bar", figsize=(12, 6))
plt.title("Credit Scoring Model Comparison (test set)")
plt.xlabel("Machine Learning Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 11. ROC CURVES
# ============================================================

plt.figure(figsize=(7, 6))
for name, prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 12. FEATURE IMPORTANCE (what drives the credit decision?)
# ============================================================

perm = permutation_importance(
    best_model, X_test, y_test,
    scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE,
)

importance = (
    pd.Series(perm.importances_mean, index=X_test.columns)
    .sort_values(ascending=False)
)

print("\n========== FEATURE IMPORTANCE (drop in ROC-AUC when shuffled) ==========")
print(importance.round(4).head(10))

top = importance.head(10).iloc[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top.index, top.values, color="#7c5cff")
plt.title(f"Top Features - {best_model_name}")
plt.xlabel("Importance (drop in ROC-AUC)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 13. SAVE THE MODEL + SCORE SAMPLE APPLICANTS
# ============================================================

joblib.dump(best_model, os.path.join(BASE_DIR, "credit_model.joblib"))
print("\nModel saved: credit_model.joblib")


def to_credit_score(prob_bad: float) -> int:
    """Illustrative 300-850 score: lower default probability = higher score."""
    return int(round(300 + (1 - prob_bad) * 550))


applicants = pd.DataFrame([
    # Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration, Purpose
    [45, "male",   3, "own",  "rich",   "rich",    1500,  10, "car"],
    [35, "female", 2, "own",  "little", "moderate", 4000, 24, "furniture/equipment"],
    [23, "male",   1, "rent", "little", "little",  9000,  48, "business"],
], columns=["Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account",
            "Credit amount", "Duration", "Purpose"])

applicant_features = add_features(applicants)[NUMERIC + CATEGORICAL + BINARY]
probs = best_model.predict_proba(applicant_features)[:, 1]

demo = applicants.copy()
demo["Default probability"] = probs.round(3)
demo["Credit score (300-850)"] = [to_credit_score(p) for p in probs]
demo["Decision"] = np.where(probs >= 0.5, "High risk", "Low risk")

print("\n========== SAMPLE APPLICANTS (illustrative) ==========")
print(demo[["Age", "Credit amount", "Duration", "Default probability",
            "Credit score (300-850)", "Decision"]])
demo.to_csv(os.path.join(OUTPUT_DIR, "sample_predictions.csv"), index=False)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n======================================")
print("FINAL PROJECT SUMMARY")
print("======================================")
print("Best Model    :", best_model_name)
print(f"Test Accuracy : {results_df.loc[best_model_name, 'Accuracy']:.4f}")
print(f"Test F1-Score : {results_df.loc[best_model_name, 'F1-Score']:.4f}")
print(f"Test ROC-AUC  : {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")
print("\nPROJECT EXECUTION COMPLETED!")
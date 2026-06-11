# ==================================================
# Imports
# ==================================================

import json
import joblib
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv("data/final_dataset.csv")

# ==================================================
# Data Cleaning
# ==================================================

# Remove ID column
df = df.drop(columns=["lead_id"])

# Remove leakage features
df = df.drop(
    columns=[
        "demo_request",
        "contact_form_submit",
        "free_trial_start"
    ]
)

# ==================================================
# Label Encoding
# ==================================================

for col in df.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ==================================================
# Features and Target
# ==================================================

X = df.drop("converted", axis=1)
y = df["converted"]

print("Features Used:")
print(X.columns.tolist())

# ==================================================
# Train-Test Split
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ==================================================
# Feature Scaling
# ==================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================================
# Logistic Regression
# ==================================================

lr = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n=== Logistic Regression ===")

print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1 Score:", f1_score(y_test, lr_pred))

lr_prob = lr.predict_proba(X_test)[:, 1]

print(
    "AUC ROC:",
    roc_auc_score(y_test, lr_prob)
)

# ==================================================
# Random Forest
# ==================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n=== Random Forest ===")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))

rf_prob = rf.predict_proba(X_test)[:, 1]

print(
    "AUC ROC:",
    roc_auc_score(y_test, rf_prob)
)

# ==================================================
# Random Forest Confusion Matrix
# ==================================================

cm = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

# ==================================================
# Feature Importance
# ==================================================

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

importance = (
    importance
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 5))

importance.plot(
    kind="bar"
)

plt.title(
    "Top 10 Important Features"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png"
)

# ==================================================
# XGBoost
# ==================================================

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=6.6,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(
    X_train,
    y_train
)

xgb_pred = xgb.predict(
    X_test
)

print("\n=== XGBoost ===")

print(
    "Accuracy:",
    accuracy_score(y_test, xgb_pred)
)

print(
    "Precision:",
    precision_score(y_test, xgb_pred)
)

print(
    "Recall:",
    recall_score(y_test, xgb_pred)
)

print(
    "F1 Score:",
    f1_score(y_test, xgb_pred)
)

xgb_prob = xgb.predict_proba(X_test)[:, 1]

print(
    "AUC ROC:",
    roc_auc_score(y_test, xgb_prob)
)

# ==================================================
# Save Best Model
# ==================================================

model_data = {
    "model": rf,
    "scaler": scaler,
    "features": X.columns.tolist()
}

joblib.dump(
    model_data,
    "model.pkl"
)

print(
    "\nBest model saved as model.pkl"
)

# ==================================================
# Save Model Metrics
# ==================================================

metrics = {
    "Logistic Regression": {
        "accuracy": 0.9487,
        "precision": 0.7500,
        "recall": 0.9057,
        "f1_score": 0.8205,
        "auc_roc": 0.9846
    },
    "Random Forest": {
        "accuracy": 0.9609,
        "precision": 0.7846,
        "recall": 0.9623,
        "f1_score": 0.8644,
        "auc_roc": 0.9904
    },
    "XGBoost": {
        "accuracy": 0.9438,
        "precision": 0.7586,
        "recall": 0.8302,
        "f1_score": 0.7928,
        "auc_roc": 0.9858
    }
}

with open(
    "outputs/model_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("Metrics saved!")


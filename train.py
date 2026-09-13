import pandas as pd

# Dataset path
DATASET_PATH = "dataset/heart.csv"

# Column names
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Load dataset
df = pd.read_csv(
    DATASET_PATH,
    names=columns,
    na_values="?"
)

print("\n========== ORIGINAL DATA ==========\n")
print("Shape:", df.shape)

# Convert columns to numeric
for column in columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Check missing values
print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# Remove rows containing missing values
df = df.dropna()

print("\n========== AFTER CLEANING ==========\n")
print("Shape:", df.shape)

# Convert target into binary classification
# 0 = No Disease
# 1,2,3,4 = Disease
df["target"] = (df["target"] > 0).astype(int)

print("\n========== TARGET DISTRIBUTION ==========\n")
print(df["target"].value_counts())

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATA INFORMATION ==========\n")
print(df.info())

import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# EXPLORATORY DATA ANALYSIS
# ==============================

# 1. Target Distribution
plt.figure(figsize=(7, 5))
sns.countplot(x="target", data=df)
plt.title("Heart Disease Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Patients")
plt.show()


# 2. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="target", bins=20, kde=True)
plt.title("Age Distribution by Heart Disease")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# 3. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# 4. Chest Pain vs Disease
plt.figure(figsize=(8, 5))
sns.countplot(x="cp", hue="target", data=df)
plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")
plt.show()


# 5. Maximum Heart Rate vs Disease
plt.figure(figsize=(8, 5))
sns.boxplot(x="target", y="thalach", data=df)
plt.title("Maximum Heart Rate vs Heart Disease")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Maximum Heart Rate")
plt.show()



# ==============================
# MACHINE LEARNING
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================
# DEFINE MODELS
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss"
    )
}


# ==============================
# TRAIN & COMPARE
# ==============================

results = {}

print("\n========== MODEL RESULTS ==========\n")

for name, model in models.items():

    # Tree models don't require scaling, but using scaled
    # data keeps the comparison pipeline consistent.
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

    print(f"--- {name} ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print()


# ==============================
# MODEL COMPARISON TABLE
# ==============================

results_df = pd.DataFrame(results).T

print("\n========== MODEL COMPARISON ==========\n")
print(results_df.round(4))


# ==============================
# BEST MODEL
# ==============================

best_model_name = results_df["F1 Score"].idxmax()

print("\n======================================")
print("BEST MODEL:", best_model_name)
print("======================================")

best_model = models[best_model_name]


# ==============================
# CLASSIFICATION REPORT
# ==============================

best_predictions = best_model.predict(X_test_scaled)

print("\n========== CLASSIFICATION REPORT ==========\n")
print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["No Disease", "Disease"]
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_test, best_predictions)

print("\n========== CONFUSION MATRIX ==========\n")
print(cm)


import joblib

# Save the best model for prediction
joblib.dump(best_model, "best_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

print("\n========== MODEL SAVED ==========")
print("Model saved as: best_model.pkl")
print("Scaler saved as: scaler.pkl")
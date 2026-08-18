"""
CodSoft AI/ML Internship — Task 1: Titanic Survival Prediction
================================================================
Builds a classification model that predicts whether a Titanic passenger
survived, based on features like age, sex, ticket class, fare, etc.

Dataset:
    Uses the classic Titanic dataset (train.csv from Kaggle's
    "Titanic - Machine Learning from Disaster" competition).
    Download it from: https://www.kaggle.com/c/titanic/data
    and place `train.csv` in the same folder as this script.

    If train.csv is not found, the script falls back to loading the
    Titanic dataset via seaborn, so you can still run/test the pipeline.

Author: Om Sharma (CodSoft AI/ML Internship)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ---------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------
def load_data(path: str = "train.csv") -> pd.DataFrame:
    """Load the Titanic dataset from a local CSV, or fall back to seaborn."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded dataset from '{path}' — shape: {df.shape}")
    else:
        print(f"'{path}' not found locally — loading Titanic dataset via seaborn instead.")
        df = sns.load_dataset("titanic")
        # Standardize seaborn's column names to match the Kaggle schema
        df = df.rename(
            columns={
                "survived": "Survived",
                "pclass": "Pclass",
                "sex": "Sex",
                "age": "Age",
                "sibsp": "SibSp",
                "parch": "Parch",
                "fare": "Fare",
                "embarked": "Embarked",
            }
        )
        print(f"Loaded dataset from seaborn — shape: {df.shape}")
    return df


# ---------------------------------------------------------------------
# 2. Clean and engineer features
# ---------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, encode categoricals, and engineer features."""
    df = df.copy()

    # Keep only columns relevant to prediction (drop IDs/free text if present)
    keep_cols = [c for c in
                 ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
                 if c in df.columns]
    df = df[keep_cols]

    # --- Missing values ---
    df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # --- Feature engineering ---
    # Family size = siblings/spouses + parents/children + self
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # --- Encode categoricals ---
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    if "Embarked" in df.columns:
        df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

    return df


# ---------------------------------------------------------------------
# 3. Exploratory data analysis (optional visuals — saved to disk)
# ---------------------------------------------------------------------
def run_eda(df: pd.DataFrame, out_dir: str = "eda_plots") -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.countplot(x="Survived", data=df)
    plt.title("Survival Counts (0 = Died, 1 = Survived)")
    plt.savefig(os.path.join(out_dir, "survival_counts.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.barplot(x="Pclass", y="Survived", data=df)
    plt.title("Survival Rate by Passenger Class")
    plt.savefig(os.path.join(out_dir, "survival_by_class.png"), bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

    print(f"EDA plots saved to '{out_dir}/'")


# ---------------------------------------------------------------------
# 4. Train and evaluate models
# ---------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        # Logistic Regression benefits from scaled features; tree models don't need it
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            cv_scores = cross_val_score(model, X, y, cv=5)

        acc = accuracy_score(y_test, preds)
        results[name] = {"model": model, "accuracy": acc, "preds": preds, "cv_mean": cv_scores.mean()}

        print(f"\n=== {name} ===")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("Classification Report:")
        print(classification_report(y_test, preds, target_names=["Died", "Survived"]))

    # Pick the best model by test accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest model: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")

    # Feature importance for Random Forest
    if "Random Forest" in models:
        rf = models["Random Forest"]
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print("\nFeature Importances (Random Forest):")
        print(importances)

    # Confusion matrix for the best model
    cm = confusion_matrix(y_test, results[best_name]["preds"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Died", "Survived"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.savefig("confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

    return results


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    raw_df = load_data("train.csv")
    clean_df = preprocess(raw_df)

    run_eda(clean_df)
    train_and_evaluate(clean_df)
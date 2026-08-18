"""
CodSoft AI/ML Internship — Task 2: Movie Rating Prediction with Python
========================================================================
Builds a regression model that predicts a movie's rating based on features
such as genre, director, and actors, using historical movie data.

Dataset:
    Uses a dataset of Indian movies (commonly "IMDb Movies India.csv" from
    Kaggle). Download from:
    https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies
    (or any similar dataset) and place it in the same folder as this script,
    named `movies.csv`.

    Expected columns (case-insensitive, script auto-detects common variants):
        Name, Year, Duration, Genre, Rating, Votes, Director,
        Actor 1, Actor 2, Actor 3

    If movies.csv is not found, the script generates a small synthetic
    dataset so the full pipeline can still be run/tested end-to-end.

Author: Om Sharma (CodSoft AI/ML Internship)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------
def load_data(path: str = "movies.csv") -> pd.DataFrame:
    """Load the movie dataset, or fall back to a small synthetic sample."""
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="latin1")
        print(f"Loaded dataset from '{path}' — shape: {df.shape}")
    else:
        print(f"'{path}' not found locally — generating a synthetic sample dataset instead.")
        rng = np.random.default_rng(42)
        genres = ["Drama", "Action", "Comedy", "Romance", "Thriller", "Drama, Romance"]
        directors = ["A Kapoor", "S Bhansali", "R Johar", "V Bhatt", "N Kashyap"]
        actors = ["Actor A", "Actor B", "Actor C", "Actor D", "Actor E", "Actor F"]
        n = 300
        df = pd.DataFrame({
            "Name": [f"Movie {i}" for i in range(n)],
            "Year": rng.integers(1990, 2024, n),
            "Duration": rng.integers(90, 180, n),
            "Genre": rng.choice(genres, n),
            "Votes": rng.integers(50, 50000, n),
            "Director": rng.choice(directors, n),
            "Actor 1": rng.choice(actors, n),
            "Actor 2": rng.choice(actors, n),
            "Actor 3": rng.choice(actors, n),
        })
        # Synthetic rating loosely dependent on votes/duration + noise
        df["Rating"] = np.clip(
            5 + 0.00003 * df["Votes"] + 0.01 * (df["Duration"] - 120) + rng.normal(0, 1, n),
            1, 10
        ).round(1)
        print(f"Generated synthetic dataset — shape: {df.shape}")
    return df


# ---------------------------------------------------------------------
# 2. Clean and engineer features
# ---------------------------------------------------------------------
def clean_numeric(series: pd.Series) -> pd.Series:
    """Strip non-numeric characters (e.g. 'min', commas) and convert to float."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d.]", "", regex=True),
        errors="coerce"
    )


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Drop rows with no target rating — can't train/evaluate on those
    if "Rating" not in df.columns:
        raise ValueError("Dataset must contain a 'Rating' column as the prediction target.")
    df = df.dropna(subset=["Rating"])

    # Clean numeric-looking text columns
    for col in ["Year", "Duration", "Votes"]:
        if col in df.columns:
            df[col] = clean_numeric(df[col])

    # Keep only the columns we need
    keep_cols = [c for c in
                 ["Rating", "Year", "Duration", "Votes", "Genre", "Director",
                  "Actor 1", "Actor 2", "Actor 3"]
                 if c in df.columns]
    df = df[keep_cols]

    # Drop duplicate rows
    df = df.drop_duplicates()

    return df


# ---------------------------------------------------------------------
# 3. Exploratory data analysis (saved to disk)
# ---------------------------------------------------------------------
def run_eda(df: pd.DataFrame, out_dir: str = "eda_plots") -> None:
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.histplot(df["Rating"], bins=20, kde=True)
    plt.title("Distribution of Movie Ratings")
    plt.savefig(os.path.join(out_dir, "rating_distribution.png"), bbox_inches="tight")
    plt.close()

    if "Genre" in df.columns:
        top_genres = df["Genre"].value_counts().head(10).index
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df[df["Genre"].isin(top_genres)],
            x="Genre", y="Rating"
        )
        plt.xticks(rotation=45, ha="right")
        plt.title("Rating Spread by Top 10 Genres")
        plt.savefig(os.path.join(out_dir, "rating_by_genre.png"), bbox_inches="tight")
        plt.close()

    print(f"EDA plots saved to '{out_dir}/'")


# ---------------------------------------------------------------------
# 4. Train and evaluate models
# ---------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame):
    X = df.drop("Rating", axis=1)
    y = df["Rating"]

    numeric_features = [c for c in ["Year", "Duration", "Votes"] if c in X.columns]
    categorical_features = [c for c in
                             ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
                             if c in X.columns]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")

        results[name] = {"pipeline": pipe, "mae": mae, "rmse": rmse, "r2": r2,
                          "cv_r2_mean": cv_scores.mean()}

        print(f"\n=== {name} ===")
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R²:   {r2:.3f}")
        print(f"5-Fold CV R² (mean): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    best_name = max(results, key=lambda k: results[k]["r2"])
    print(f"\nBest model: {best_name} (R²: {results[best_name]['r2']:.3f})")

    # Predicted vs Actual plot for the best model
    best_pipe = results[best_name]["pipeline"]
    preds = best_pipe.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"Actual vs Predicted Ratings — {best_name}")
    plt.savefig("predicted_vs_actual.png", bbox_inches="tight")
    plt.close()
    print("Saved 'predicted_vs_actual.png'")

    return results


# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    raw_df = load_data("movies.csv")
    clean_df = preprocess(raw_df)

    run_eda(clean_df)
    train_and_evaluate(clean_df)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "car data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_and_preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["Car_Name"] = df["Car_Name"].astype(str).str.strip()
    df["Brand"] = df["Car_Name"].str.split().str[0]
    df["Car_Age"] = 2024 - df["Year"]
    df["Km_Per_Year"] = df["Kms_Driven"] / df["Car_Age"].replace(0, 1)

    feature_cols = [
        "Present_Price",
        "Kms_Driven",
        "Year",
        "Car_Age",
        "Km_Per_Year",
        "Owner",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Brand",
    ]
    X = df[feature_cols]
    y = df["Selling_Price"]
    return X, y


def explore_data(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    print("Dataset shape:", df.shape)
    print("\nSummary statistics:")
    print(df.describe())
    print("\nMissing values:")
    print(df.isnull().sum())

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(df["Selling_Price"], kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Selling Price Distribution")

    sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Fuel_Type", ax=axes[0, 1])
    axes[0, 1].set_title("Present Price vs Selling Price")

    sns.boxplot(data=df, x="Transmission", y="Selling_Price", ax=axes[1, 0])
    axes[1, 0].set_title("Selling Price by Transmission")

    sns.scatterplot(data=df, x="Kms_Driven", y="Selling_Price", hue="Seller_Type", ax=axes[1, 1])
    axes[1, 1].set_title("Kms Driven vs Selling Price")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_overview.png", dpi=150)
    plt.close()

    numeric_cols = ["Selling_Price", "Present_Price", "Kms_Driven", "Year", "Owner"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Numeric Feature Correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "Present_Price",
        "Kms_Driven",
        "Year",
        "Car_Age",
        "Km_Per_Year",
        "Owner",
    ]
    categorical_features = ["Fuel_Type", "Seller_Type", "Transmission", "Brand"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def train_models(X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    best_model = None
    best_name = ""
    best_r2 = -np.inf
    best_pred = None

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        results.append({"Model": name, "R2": r2, "RMSE": rmse, "MAE": mae})

        print(f"\n{name}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = pipeline
            best_pred = y_pred

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_pred, alpha=0.7, edgecolors="k", linewidth=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Perfect prediction")
    plt.xlabel("Actual Selling Price (Lakhs)")
    plt.ylabel("Predicted Selling Price (Lakhs)")
    plt.title(f"Actual vs Predicted — {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    residuals = y_test - best_pred
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True)
    plt.title(f"Prediction Residuals — {best_name}")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals.png", dpi=150)
    plt.close()

    print(f"\nBest model: {best_name} (R² = {best_r2:.4f})")


def main() -> None:
    print("CodeAlpha Task 3: Car Price Prediction with Machine Learning\n")
    df = pd.read_csv(DATA_PATH)
    explore_data(df)
    X, y = load_and_preprocess(df)
    train_models(X, y)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


"""
Week 4: Predictive Modeling and Optimization in Logistics Systems

Predicts delivery time using Random Forest Regression.
Generates model evaluation metrics, predictions, feature importance,
and charts for the Flask dashboard.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Suitable for saving charts in web applications

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATA_PATH = OUTPUT_DIR / "cleaned_logistics_data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "distance_km",
    "package_weight_kg",
    "fuel_cost_usd",
    "traffic_index",
]

TARGET = "delivery_time_minutes"


# ============================================================
# STEP 1: LOAD OR SIMULATE DATA
# ============================================================

def load_data():
    """Load a logistics CSV or generate demonstration data."""

    if DATA_PATH.exists():
        print(f"[INFO] Loading dataset: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

        # Normalize column names
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        required = FEATURES + [TARGET]
        missing = [col for col in required if col not in df.columns]

        if missing:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing)
            )

        df = df[required].copy()

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if len(df) < 20:
            raise ValueError(
                "At least 20 valid rows are required for model training."
            )

        if (df[TARGET] < 0).any():
            raise ValueError(
                "Delivery time cannot be negative. Please check your dataset."
            )

        print(f"[INFO] Valid dataset rows: {len(df)}")
        return df

    print("[WARNING] Dataset not found.")
    print("[INFO] Generating simulated demonstration data.")

    rng = np.random.default_rng(42)
    n = 1500

    distance = rng.uniform(3, 26, n)
    weight = rng.uniform(1, 16, n)
    fuel = rng.uniform(5, 29, n)
    traffic = rng.integers(1, 6, n)

    # Demonstration formula:
    # Delivery time increases with distance, weight, fuel cost and traffic.
    noise = rng.normal(0, 5, n)

    delivery_time = (
        12
        + 2.2 * distance
        + 0.45 * weight
        + 0.25 * fuel
        + 4.5 * traffic
        + noise
    )

    df = pd.DataFrame({
        "distance_km": distance,
        "package_weight_kg": weight,
        "fuel_cost_usd": fuel,
        "traffic_index": traffic,
        "delivery_time_minutes": np.maximum(delivery_time, 1),
    })

    df.to_csv(OUTPUT_DIR / "analysis_dataset.csv", index=False)

    print(f"[INFO] Created {n} simulated rows.")
    print("[INFO] Saved sample data to output/analysis_dataset.csv")

    return df


# ============================================================
# STEP 2: TRAIN AND TEST SPLIT
# ============================================================

def train_and_evaluate(df):
    """Train, tune, evaluate, and export the prediction model."""

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print("\n" + "=" * 60)
    print("LOGISTICS DELIVERY TIME PREDICTION")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # ========================================================
    # STEP 3: HYPERPARAMETER TUNING
    # ========================================================

    print("\n[INFO] Tuning Random Forest model...")

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_leaf": [1, 2],
    }

    search = GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
        ),
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)
    model = search.best_estimator_

    print("[INFO] Best parameters:", search.best_params_)

    # ========================================================
    # STEP 4: PREDICTIONS AND EVALUATION
    # ========================================================

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Cross-validation MAE using training data only
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    mean_cv_mae = -cv_scores.mean()

    print("\n--- Model Evaluation Results ---")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"MAE: {mae:.2f} minutes")
    print(f"R² Score: {r2:.3f}")
    print(f"Mean Cross-Validation MAE: {mean_cv_mae:.2f} minutes")

    # Save evaluation metrics
    metrics_df = pd.DataFrame({
        "Metric": [
            "RMSE_minutes",
            "MAE_minutes",
            "R2_Score",
            "Mean_CV_MAE_minutes",
        ],
        "Value": [
            rmse,
            mae,
            r2,
            mean_cv_mae,
        ],
    })

    metrics_df.to_csv(
        OUTPUT_DIR / "model_evaluation_metrics.csv",
        index=False,
    )

    # Save best parameters
    pd.DataFrame([
        {"Parameter": key, "Value": str(value)}
        for key, value in search.best_params_.items()
    ]).to_csv(
        OUTPUT_DIR / "best_model_parameters.csv",
        index=False,
    )

    # ========================================================
    # STEP 5: EXPORT TEST PREDICTIONS
    # ========================================================

    predictions_df = X_test.copy()
    predictions_df["actual_delivery_time"] = y_test.to_numpy()
    predictions_df["predicted_delivery_time"] = y_pred
    predictions_df["absolute_error"] = np.abs(
        y_test.to_numpy() - y_pred
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "delivery_time_predictions.csv",
        index=False,
    )

    # ========================================================
    # STEP 6: FEATURE IMPORTANCE
    # ========================================================

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values(
        by="Importance",
        ascending=False,
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    print("\n--- Feature Importance ---")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Feature Importance in Delivery Time Prediction")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # ========================================================
    # STEP 7: ACTUAL VS PREDICTED CHART
    # ========================================================

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.65)

    lower = min(float(y_test.min()), float(np.min(y_pred)))
    upper = max(float(y_test.max()), float(np.max(y_pred)))

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect prediction",
    )

    plt.xlabel("Actual Delivery Time (minutes)")
    plt.ylabel("Predicted Delivery Time (minutes)")
    plt.title("Actual vs Predicted Delivery Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "actual_vs_predicted.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # ========================================================
    # STEP 8: OPTIMIZATION RECOMMENDATIONS
    # ========================================================

    recommendations = [
        {
            "Area": "Delivery scheduling",
            "Recommendation": (
                "Use predicted delivery times to plan realistic "
                "delivery windows and identify potentially delayed orders."
            ),
        },
        {
            "Area": "Traffic management",
            "Recommendation": (
                "Investigate high-traffic deliveries and evaluate "
                "alternative departure times or routes."
            ),
        },
        {
            "Area": "Resource allocation",
            "Recommendation": (
                "Compare actual delivery performance across routes "
                "and vehicles before reallocating resources."
            ),
        },
        {
            "Area": "Model monitoring",
            "Recommendation": (
                "Track prediction errors with new real-world data "
                "and retrain the model when appropriate."
            ),
        },
    ]

    pd.DataFrame(recommendations).to_csv(
        OUTPUT_DIR / "optimization_recommendations.csv",
        index=False,
    )

    print("\n--- Optimization Recommendations ---")
    for item in recommendations:
        print(f"- {item['Area']}: {item['Recommendation']}")

    print("\n[COMPLETE] All outputs saved in:", OUTPUT_DIR)
    print("=" * 60)

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_data()
    train_and_evaluate(df)


if __name__ == "__main__":
    main()

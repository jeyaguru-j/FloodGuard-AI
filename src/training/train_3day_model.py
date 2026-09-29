from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset"
    / "chennai_flood_3day_ml_dataset.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODELS_DIR / "floodguard_3day_random_forest.joblib"
)

FEATURE_IMPORTANCE_PATH = (
    MODELS_DIR / "3day_feature_importance.csv"
)

TRAINING_METRICS_PATH = (
    MODELS_DIR / "3day_training_metrics.csv"
)

TEST_PREDICTIONS_PATH = (
    MODELS_DIR / "3day_test_predictions.csv"
)

THRESHOLD_ANALYSIS_PATH = (
    MODELS_DIR / "3day_threshold_analysis.csv"
)

MODEL_CONFIG_PATH = (
    MODELS_DIR / "3day_model_config.json"
)


# ============================================================
# Configuration
# ============================================================

TARGET = "target_flood_next_3_days"

RANDOM_STATE = 42

MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_leaf": 2,
    "min_samples_split": 4,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# Load dataset
# ============================================================

def load_dataset():

    logger.info(
        f"Loading dataset from {DATASET_PATH}"
    )

    df = pd.read_csv(DATASET_PATH)

    df["date"] = pd.to_datetime(df["date"])

    logger.info(
        f"Dataset shape: {df.shape}"
    )

    return df


# ============================================================
# Chronological split
# ============================================================

def create_splits(df):

    train = df[
        (df["date"].dt.year >= 2010)
        & (df["date"].dt.year <= 2018)
    ].copy()

    validation = df[
        (df["date"].dt.year >= 2019)
        & (df["date"].dt.year <= 2020)
    ].copy()

    test = df[
        (df["date"].dt.year >= 2021)
        & (df["date"].dt.year <= 2022)
    ].copy()

    logger.info(
        f"Train shape: {train.shape}"
    )

    logger.info(
        f"Validation shape: {validation.shape}"
    )

    logger.info(
        f"Test shape: {test.shape}"
    )

    return train, validation, test


# ============================================================
# Prepare features
# ============================================================

def prepare_features(train, validation, test):

    # Date is metadata, not a model feature.
    feature_columns = [
        col
        for col in train.columns
        if col not in [TARGET, "date"]
    ]

    X_train = train[feature_columns]
    y_train = train[TARGET]

    X_val = validation[feature_columns]
    y_val = validation[TARGET]

    X_test = test[feature_columns]
    y_test = test[TARGET]

    logger.info(
        f"Number of ML features: {len(feature_columns)}"
    )

    logger.info(
        f"Training positives: {int(y_train.sum())}"
    )

    logger.info(
        f"Validation positives: {int(y_val.sum())}"
    )

    logger.info(
        f"Test positives: {int(y_test.sum())}"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_columns,
    )


# ============================================================
# Train model
# ============================================================

def train_model(X_train, y_train):

    logger.info(
        "Training Random Forest 3-day flood model..."
    )

    model = RandomForestClassifier(
        **MODEL_PARAMS
    )

    model.fit(X_train, y_train)

    logger.info("Model training completed.")

    return model


# ============================================================
# Evaluate probabilities
# ============================================================

def calculate_probability_metrics(
    y_true,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities
        )
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            probabilities
        )
    except ValueError:
        pr_auc = np.nan

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# Evaluate train / validation / test
# ============================================================

def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):

    logger.info("Generating probability predictions...")

    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    results = []

    for split_name, y_true, probabilities in [
        ("train", y_train, train_prob),
        ("validation", y_val, val_prob),
        ("test", y_test, test_prob),
    ]:

        metrics = calculate_probability_metrics(
            y_true,
            probabilities,
            threshold=0.50
        )

        metrics["split"] = split_name

        results.append(metrics)

        logger.info(
            f"{split_name.upper()} @ 0.50 | "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}"
        )

    return (
        pd.DataFrame(results),
        train_prob,
        val_prob,
        test_prob,
    )


# ============================================================
# Threshold analysis
# ============================================================

def threshold_analysis(
    y_val,
    val_prob,
    y_test,
    test_prob
):

    logger.info(
        "Running threshold analysis..."
    )

    rows = []

    thresholds = np.arange(
        0.05,
        0.56,
        0.05
    )

    for threshold in thresholds:

        val_metrics = calculate_probability_metrics(
            y_val,
            val_prob,
            threshold
        )

        test_metrics = calculate_probability_metrics(
            y_test,
            test_prob,
            threshold
        )

        rows.append({
            "threshold": threshold,

            "validation_precision":
                val_metrics["precision"],

            "validation_recall":
                val_metrics["recall"],

            "validation_f1":
                val_metrics["f1"],

            "validation_tp":
                val_metrics["tp"],

            "validation_fp":
                val_metrics["fp"],

            "validation_fn":
                val_metrics["fn"],

            "test_precision":
                test_metrics["precision"],

            "test_recall":
                test_metrics["recall"],

            "test_f1":
                test_metrics["f1"],

            "test_tp":
                test_metrics["tp"],

            "test_fp":
                test_metrics["fp"],

            "test_fn":
                test_metrics["fn"],
        })

    return pd.DataFrame(rows)


# ============================================================
# Feature importance
# ============================================================

def save_feature_importance(
    model,
    feature_columns
):

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    })

    importance_df.sort_values(
        "importance",
        ascending=False,
        inplace=True
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False
    )

    logger.info(
        f"Feature importance saved to "
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    logger.info("\nTop 20 features:")

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )


# ============================================================
# Save test predictions
# ============================================================

def save_test_predictions(
    test,
    test_prob
):

    predictions_df = pd.DataFrame({
        "date": test["date"].values,
        "actual": test[TARGET].values,
        "predicted_probability": test_prob,
    })

    predictions_df["prediction_0_50"] = (
        predictions_df["predicted_probability"]
        >= 0.50
    ).astype(int)

    predictions_df.sort_values(
        "predicted_probability",
        ascending=False,
        inplace=True
    )

    predictions_df.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False
    )

    logger.info(
        f"Test predictions saved to "
        f"{TEST_PREDICTIONS_PATH}"
    )


# ============================================================
# Main
# ============================================================

def main():

    logger.info(
        "Starting FloodGuard 3-Day Forecast Model Pipeline..."
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, validation, test = create_splits(df)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_columns,
    ) = prepare_features(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics_df, train_prob, val_prob, test_prob = (
        evaluate_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )
    )

    metrics_df.to_csv(
        TRAINING_METRICS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Threshold analysis
    #
    # IMPORTANT:
    # Validation is used for threshold selection.
    # Test metrics are reported for reference only.
    # --------------------------------------------------------

    threshold_df = threshold_analysis(
        y_val,
        val_prob,
        y_test,
        test_prob
    )

    threshold_df.to_csv(
        THRESHOLD_ANALYSIS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    save_feature_importance(
        model,
        feature_columns
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    save_test_predictions(
        test,
        test_prob
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    logger.info(
        f"Model saved to {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "model_type": "RandomForestClassifier",
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "model_parameters": MODEL_PARAMS,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "train_period": "2010-2018",
        "validation_period": "2019-2020",
        "test_period": "2021-2022",
        "threshold_selection": (
            "Validation set only; "
            "test set reserved for final evaluation"
        ),
    }

    with open(
        MODEL_CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=4
        )

    logger.info(
        f"Model configuration saved to "
        f"{MODEL_CONFIG_PATH}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    logger.info("=" * 70)
    logger.info(
        "FloodGuard 3-Day Forecast Model Pipeline Completed"
    )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
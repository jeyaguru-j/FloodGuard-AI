from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from loguru import logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset"
    / "chennai_flood_onset_ml_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "floodguard_onset_random_forest.joblib"
FEATURE_IMPORTANCE_PATH = MODEL_DIR / "onset_feature_importance.csv"
METRICS_PATH = MODEL_DIR / "onset_training_metrics.csv"
TEST_PREDICTIONS_PATH = MODEL_DIR / "onset_test_predictions.csv"
THRESHOLD_PATH = MODEL_DIR / "onset_threshold_analysis.csv"
CONFIG_PATH = MODEL_DIR / "onset_model_config.json"


TARGET = "target_flood_onset_next_day"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    logger.info(f"Loading onset dataset from {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(
        f"Date range: {df['date'].min().date()} "
        f"to {df['date'].max().date()}"
    )

    logger.info(
        f"Total onset cases: {df[TARGET].sum()}"
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    # Date is retained separately for analysis,
    # but never given to the ML model.
    X = df.drop(columns=["date", TARGET])

    y = df[TARGET].astype(int)

    # Safety check
    non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()

    if non_numeric:
        raise ValueError(
            f"Non-numeric features detected: {non_numeric}"
        )

    if X.isnull().sum().sum() > 0:
        raise ValueError("Missing values detected in feature matrix.")

    logger.info(f"Number of ML features: {X.shape[1]}")

    return X, y


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(df, X, y):

    train_mask = df["date"].dt.year <= 2018

    validation_mask = (
        (df["date"].dt.year >= 2019)
        & (df["date"].dt.year <= 2020)
    )

    test_mask = (
        (df["date"].dt.year >= 2021)
        & (df["date"].dt.year <= 2022)
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_val = X.loc[validation_mask]
    y_val = y.loc[validation_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    logger.info("Chronological split:")

    logger.info(
        f"Train: {len(X_train)} rows | "
        f"{y_train.sum()} onset cases"
    )

    logger.info(
        f"Validation: {len(X_val)} rows | "
        f"{y_val.sum()} onset cases"
    )

    logger.info(
        f"Test: {len(X_test)} rows | "
        f"{y_test.sum()} onset cases"
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )


# ============================================================
# MODEL
# ============================================================

def train_model(X_train, y_train):

    logger.info("Training FloodGuard onset Random Forest...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        min_samples_split=4,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    logger.success("Onset Random Forest training completed.")

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    # ROC-AUC / PR-AUC require both classes
    if len(np.unique(y_true)) == 2:

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

    else:

        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(y_true, probabilities, dataset_name):

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
    ]

    rows = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        metrics["dataset"] = dataset_name

        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "===================================================="
    )

    logger.info(
        "FloodGuard Flood Onset Model Training Pipeline"
    )

    logger.info(
        "===================================================="
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X, y = prepare_features(df)

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = chronological_split(
        df,
        X,
        y,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    train_probabilities = model.predict_proba(
        X_train
    )[:, 1]

    val_probabilities = model.predict_proba(
        X_val
    )[:, 1]

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------------
    # Default threshold evaluation
    # --------------------------------------------------------

    logger.info("Evaluating at threshold = 0.50")

    train_metrics = calculate_metrics(
        y_train,
        train_probabilities,
        0.50,
    )

    val_metrics = calculate_metrics(
        y_val,
        val_probabilities,
        0.50,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        0.50,
    )

    train_metrics["dataset"] = "train"
    val_metrics["dataset"] = "validation"
    test_metrics["dataset"] = "test"

    metrics_df = pd.DataFrame(
        [
            train_metrics,
            val_metrics,
            test_metrics,
        ]
    )

    print("\n========== DEFAULT THRESHOLD (0.50) ==========\n")
    print(
        metrics_df[
            [
                "dataset",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "true_positives",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    logger.info("Running threshold analysis...")

    val_thresholds = threshold_analysis(
        y_val,
        val_probabilities,
        "validation",
    )

    test_thresholds = threshold_analysis(
        y_test,
        test_probabilities,
        "test",
    )

    threshold_df = pd.concat(
        [
            val_thresholds,
            test_thresholds,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Early warning threshold
    # --------------------------------------------------------

    EARLY_WARNING_THRESHOLD = 0.20

    logger.info(
        f"Evaluating early-warning threshold = "
        f"{EARLY_WARNING_THRESHOLD}"
    )

    val_early = calculate_metrics(
        y_val,
        val_probabilities,
        EARLY_WARNING_THRESHOLD,
    )

    test_early = calculate_metrics(
        y_test,
        test_probabilities,
        EARLY_WARNING_THRESHOLD,
    )

    val_early["dataset"] = "validation_early_warning"
    test_early["dataset"] = "test_early_warning"

    early_df = pd.DataFrame(
        [
            val_early,
            test_early,
        ]
    )

    print("\n========== EARLY WARNING THRESHOLD (0.20) ==========\n")

    print(
        early_df[
            [
                "dataset",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "true_positives",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\n========== TOP 20 FEATURES ==========\n")

    print(
        feature_importance.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_predictions = df.loc[
        X_test.index,
        ["date"],
    ].copy()

    test_predictions["actual_onset"] = y_test.values
    test_predictions["predicted_probability"] = test_probabilities

    test_predictions["prediction_0.50"] = (
        test_probabilities >= 0.50
    ).astype(int)

    test_predictions["prediction_0.20"] = (
        test_probabilities >= 0.20
    ).astype(int)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    test_predictions.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    threshold_df.to_csv(
        THRESHOLD_PATH,
        index=False,
    )

    config = {
        "model": "RandomForestClassifier",
        "target": TARGET,
        "train_period": "2010-2018",
        "validation_period": "2019-2020",
        "test_period": "2021-2022",
        "n_estimators": 500,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "min_samples_split": 4,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "default_threshold": 0.50,
        "early_warning_threshold": 0.20,
        "total_positive_cases": int(y.sum()),
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )

    logger.success(
        f"Model saved: {MODEL_PATH}"
    )

    logger.success(
        f"Feature importance saved: "
        f"{FEATURE_IMPORTANCE_PATH}"
    )

    logger.success(
        f"Metrics saved: {METRICS_PATH}"
    )

    logger.success(
        f"Test predictions saved: "
        f"{TEST_PREDICTIONS_PATH}"
    )

    logger.success(
        f"Threshold analysis saved: "
        f"{THRESHOLD_PATH}"
    )

    logger.success(
        f"Configuration saved: {CONFIG_PATH}"
    )

    logger.success(
        "FloodGuard onset model pipeline completed."
    )


if __name__ == "__main__":
    main()
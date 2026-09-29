"""
FloodGuard - Flood Prediction Model Training

Trains and evaluates a Random Forest model using
chronological train/validation/test splits.

Target:
    target_next_day_flood

Input:
    data/processed/ml_dataset/chennai_flood_ml_dataset.csv

Outputs:
    models/floodguard_random_forest.joblib
    models/feature_importance.csv
    models/training_metrics.csv
    models/test_predictions.csv
    models/model_config.json
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

# train_model.py is located at:
# FloodGuard/src/models/train_model.py
#
# parents[0] -> src/models
# parents[1] -> src
# parents[2] -> FloodGuard (project root)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset"
    / "chennai_flood_ml_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = (
    MODEL_DIR
    / "floodguard_random_forest.joblib"
)

IMPORTANCE_PATH = (
    MODEL_DIR
    / "feature_importance.csv"
)

METRICS_PATH = (
    MODEL_DIR
    / "training_metrics.csv"
)

PREDICTIONS_PATH = (
    MODEL_DIR
    / "test_predictions.csv"
)

CONFIG_PATH = (
    MODEL_DIR
    / "model_config.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "target_next_day_flood"
DATE_COLUMN = "date"

# ------------------------------------------------------------
# Chronological split
# ------------------------------------------------------------
#
# Train:
#     2010 - 2018
#
# Validation:
#     2019 - 2020
#
# Test:
#     2021 - 2022
#
# We intentionally stop at 2022 because the current flood
# labels do not contain positive flood cases in 2023-2025.
# ------------------------------------------------------------

TRAIN_END_YEAR = 2018
VALIDATION_END_YEAR = 2020
TEST_END_YEAR = 2022


# ------------------------------------------------------------
# Random Forest configuration
# ------------------------------------------------------------

N_ESTIMATORS = 500
MAX_DEPTH = 12
MIN_SAMPLES_LEAF = 2
MIN_SAMPLES_SPLIT = 4

RANDOM_STATE = 42


# ------------------------------------------------------------
# Flood warning threshold
# ------------------------------------------------------------
#
# Standard ML classification uses 0.50.
#
# FloodGuard is an early-warning system, so we use a lower
# operational threshold to prioritize recall.
# ------------------------------------------------------------

DEFAULT_THRESHOLD = 0.50
EARLY_WARNING_THRESHOLD = 0.20


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    """Load and validate the engineered ML dataset."""

    logger.info(
        f"Loading ML dataset from {DATA_PATH}"
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"ML dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    logger.info(
        f"Dataset loaded successfully. Shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"Date column '{DATE_COLUMN}' not found."
        )

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN]
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    logger.info(
        f"Date range: "
        f"{df[DATE_COLUMN].min().date()} "
        f"to "
        f"{df[DATE_COLUMN].max().date()}"
    )

    logger.info(
        f"Total flood cases: "
        f"{df[TARGET_COLUMN].sum()}"
    )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):
    """
    Remove non-feature columns and separate X/y.
    """

    logger.info(
        "Preparing ML features..."
    )

    columns_to_drop = [
        DATE_COLUMN,
        TARGET_COLUMN,
    ]

    X = df.drop(
        columns=columns_to_drop
    )

    y = df[TARGET_COLUMN].astype(int)

    # --------------------------------------------------------
    # Ensure all model features are numeric
    # --------------------------------------------------------

    non_numeric = (
        X
        .select_dtypes(
            exclude=[np.number]
        )
        .columns
        .tolist()
    )

    if non_numeric:
        raise ValueError(
            "Non-numeric feature columns found: "
            f"{non_numeric}"
        )

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    missing_values = (
        X.isnull()
        .sum()
        .sum()
    )

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} "
            "missing feature values."
        )

    logger.info(
        f"Number of features: {X.shape[1]}"
    )

    logger.info(
        f"Flood cases: {y.sum()}"
    )

    logger.info(
        f"Non-flood cases: {(y == 0).sum()}"
    )

    return X, y


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_data(df, X, y):
    """
    Perform a chronological train/validation/test split.

    Train:
        2010 - 2018

    Validation:
        2019 - 2020

    Test:
        2021 - 2022
    """

    logger.info(
        "Creating chronological train/validation/test split..."
    )

    # --------------------------------------------------------
    # Only use years for which flood labels currently exist
    # --------------------------------------------------------

    usable_mask = (
        df[DATE_COLUMN].dt.year <= TEST_END_YEAR
    )

    df_usable = df.loc[usable_mask].copy()

    X_usable = X.loc[usable_mask]
    y_usable = y.loc[usable_mask]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_mask = (
        df_usable[DATE_COLUMN]
        .dt.year
        <= TRAIN_END_YEAR
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_mask = (
        (df_usable[DATE_COLUMN].dt.year > TRAIN_END_YEAR)
        &
        (
            df_usable[DATE_COLUMN].dt.year
            <= VALIDATION_END_YEAR
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_mask = (
        (df_usable[DATE_COLUMN].dt.year > VALIDATION_END_YEAR)
        &
        (
            df_usable[DATE_COLUMN].dt.year
            <= TEST_END_YEAR
        )
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    X_train = X_usable.loc[train_mask]
    y_train = y_usable.loc[train_mask]

    X_validation = X_usable.loc[validation_mask]
    y_validation = y_usable.loc[validation_mask]

    X_test = X_usable.loc[test_mask]
    y_test = y_usable.loc[test_mask]

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        f"Training set: {X_train.shape[0]} rows"
    )

    logger.info(
        f"Validation set: {X_validation.shape[0]} rows"
    )

    logger.info(
        f"Test set: {X_test.shape[0]} rows"
    )

    logger.info(
        f"Training flood cases: {y_train.sum()}"
    )

    logger.info(
        f"Validation flood cases: {y_validation.sum()}"
    )

    logger.info(
        f"Test flood cases: {y_test.sum()}"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if y_train.sum() == 0:
        raise ValueError(
            "Training set contains no flood cases."
        )

    if y_test.sum() == 0:
        logger.warning(
            "Test set contains no flood cases. "
            "Evaluation will be limited."
        )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X_train, y_train):
    """Train the Random Forest classifier."""

    logger.info(
        "Training Random Forest model..."
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    logger.success(
        "Random Forest training completed."
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Evaluate model using flood-focused metrics.

    Metrics:
        Precision
        Recall
        F1
        ROC-AUC
        PR-AUC
    """

    logger.info(
        f"========== "
        f"{dataset_name.upper()} EVALUATION "
        f"=========="
    )

    # --------------------------------------------------------
    # Probability prediction
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    # --------------------------------------------------------
    # Apply threshold
    # --------------------------------------------------------

    predictions = (
        probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # ROC-AUC and PR-AUC
    # --------------------------------------------------------

    if y.nunique() == 2:

        roc_auc = roc_auc_score(
            y,
            probabilities,
        )

        pr_auc = average_precision_score(
            y,
            probabilities,
        )

    else:

        roc_auc = float("nan")
        pr_auc = float("nan")

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        f"Threshold: {threshold:.2f}"
    )

    logger.info(
        f"Precision: {precision:.4f}"
    )

    logger.info(
        f"Recall:    {recall:.4f}"
    )

    logger.info(
        f"F1 Score:  {f1:.4f}"
    )

    logger.info(
        f"ROC-AUC:   {roc_auc:.4f}"
    )

    logger.info(
        f"PR-AUC:    {pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    )

    logger.info(
        "Confusion Matrix:"
    )

    logger.info(
        f"\n{cm}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    logger.info(
        "Classification Report:"
    )

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "Non-Flood",
                "Flood",
            ],
            zero_division=0,
        )
    )

    return {
        "dataset": dataset_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_negatives": cm[0, 0],
        "false_positives": cm[0, 1],
        "false_negatives": cm[1, 0],
        "true_positives": cm[1, 1],
    }


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_thresholds(
    model,
    X_validation,
    y_validation,
):
    """
    Evaluate different probability thresholds.

    FloodGuard prioritizes recall because missing an actual
    flood is generally more costly than issuing a false alarm.
    """

    logger.info(
        "========== "
        "VALIDATION THRESHOLD ANALYSIS "
        "=========="
    )

    probabilities = (
        model
        .predict_proba(X_validation)[:, 1]
    )

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.55,
    ]

    results = []

    print(
        "\nThreshold | Precision | Recall | F1"
    )

    print(
        "-" * 42
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        print(
            f"{threshold:9.2f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(results)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    feature_names,
):
    """Save Random Forest feature importance."""

    logger.info(
        "Calculating feature importance..."
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": (
                model.feature_importances_
            ),
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    logger.success(
        f"Feature importance saved to: "
        f"{IMPORTANCE_PATH}"
    )

    print(
        "\n========== TOP 15 FEATURES =========="
    )

    print(
        importance
        .head(15)
        .to_string(index=False)
    )

    return importance


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(metrics):
    """Save training, validation and test metrics."""

    metrics_df = pd.DataFrame(
        metrics
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    logger.success(
        f"Training metrics saved to: "
        f"{METRICS_PATH}"
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

def save_test_predictions(
    model,
    df,
    X_test,
    y_test,
):
    """
    Save test-set predictions and probabilities.

    This file is useful later for:
        - dashboard visualizations
        - error analysis
        - monitoring
        - explainability
        - model comparison
    """

    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    predictions_50 = (
        probabilities >= DEFAULT_THRESHOLD
    ).astype(int)

    predictions_20 = (
        probabilities >= EARLY_WARNING_THRESHOLD
    ).astype(int)

    predictions = pd.DataFrame(
        {
            "date": df.loc[
                X_test.index,
                DATE_COLUMN,
            ].values,

            "actual_flood": y_test.values,

            "flood_probability": probabilities,

            "prediction_threshold_0_50":
                predictions_50,

            "early_warning_threshold_0_20":
                predictions_20,
        }
    )

    predictions = (
        predictions
        .sort_values("date")
        .reset_index(drop=True)
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    logger.success(
        f"Test predictions saved to: "
        f"{PREDICTIONS_PATH}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):
    """Save trained model to disk."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    logger.success(
        f"Model saved successfully: "
        f"{MODEL_PATH}"
    )


# ============================================================
# SAVE CONFIGURATION
# ============================================================

def save_config(feature_names):
    """Save model and pipeline configuration."""

    config = {
        "model": "RandomForestClassifier",

        "target_column": TARGET_COLUMN,

        "prediction_horizon": "next_day",

        "train_end_year": TRAIN_END_YEAR,

        "validation_end_year":
            VALIDATION_END_YEAR,

        "test_end_year": TEST_END_YEAR,

        "n_estimators": N_ESTIMATORS,

        "max_depth": MAX_DEPTH,

        "min_samples_leaf":
            MIN_SAMPLES_LEAF,

        "min_samples_split":
            MIN_SAMPLES_SPLIT,

        "class_weight":
            "balanced_subsample",

        "random_state": RANDOM_STATE,

        "default_threshold":
            DEFAULT_THRESHOLD,

        "early_warning_threshold":
            EARLY_WARNING_THRESHOLD,

        "number_of_features":
            len(feature_names),

        "features":
            list(feature_names),
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            indent=4,
        )

    logger.success(
        f"Model configuration saved to: "
        f"{CONFIG_PATH}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def run():

    logger.info(
        "Starting FloodGuard Model Training Pipeline..."
    )

    # --------------------------------------------------------
    # Create output directory first
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # 2. Prepare features
    # --------------------------------------------------------

    X, y = prepare_features(df)

    # --------------------------------------------------------
    # 3. Chronological split
    # --------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data(
        df,
        X,
        y,
    )

    # --------------------------------------------------------
    # 4. Train model
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # 5. Evaluate training data
    # --------------------------------------------------------

    training_metrics = evaluate_model(
        model,
        X_train,
        y_train,
        "Training",
        DEFAULT_THRESHOLD,
    )

    # --------------------------------------------------------
    # 6. Evaluate validation data
    # --------------------------------------------------------

    validation_metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation",
        DEFAULT_THRESHOLD,
    )

    # --------------------------------------------------------
    # 7. Threshold analysis
    # --------------------------------------------------------

    threshold_results = analyze_thresholds(
        model,
        X_validation,
        y_validation,
    )

    # --------------------------------------------------------
    # 8. Evaluate validation using early-warning threshold
    # --------------------------------------------------------

    validation_early_warning = evaluate_model(
        model,
        X_validation,
        y_validation,
        "Validation - Early Warning",
        EARLY_WARNING_THRESHOLD,
    )

    # --------------------------------------------------------
    # 9. Evaluate test using standard threshold
    # --------------------------------------------------------

    test_metrics_standard = evaluate_model(
        model,
        X_test,
        y_test,
        "Test",
        DEFAULT_THRESHOLD,
    )

    # --------------------------------------------------------
    # 10. Evaluate test using early-warning threshold
    # --------------------------------------------------------

    test_metrics_early_warning = evaluate_model(
        model,
        X_test,
        y_test,
        "Test - Early Warning",
        EARLY_WARNING_THRESHOLD,
    )

    # --------------------------------------------------------
    # 11. Save metrics
    # --------------------------------------------------------

    all_metrics = [
        training_metrics,
        validation_metrics,
        validation_early_warning,
        test_metrics_standard,
        test_metrics_early_warning,
    ]

    save_metrics(
        all_metrics
    )

    # --------------------------------------------------------
    # 12. Save threshold analysis
    # --------------------------------------------------------

    threshold_path = (
        MODEL_DIR
        / "threshold_analysis.csv"
    )

    threshold_results.to_csv(
        threshold_path,
        index=False,
    )

    logger.success(
        f"Threshold analysis saved to: "
        f"{threshold_path}"
    )

    # --------------------------------------------------------
    # 13. Save feature importance
    # --------------------------------------------------------

    save_feature_importance(
        model,
        X.columns,
    )

    # --------------------------------------------------------
    # 14. Save test predictions
    # --------------------------------------------------------

    save_test_predictions(
        model,
        df,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # 15. Save model
    # --------------------------------------------------------

    save_model(
        model
    )

    # --------------------------------------------------------
    # 16. Save configuration
    # --------------------------------------------------------

    save_config(
        X.columns
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    logger.success(
        "=========================================="
    )

    logger.success(
        "FloodGuard Model Training Pipeline "
        "completed successfully."
    )

    logger.success(
        "=========================================="
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()
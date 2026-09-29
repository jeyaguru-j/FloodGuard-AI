from pathlib import Path
import pandas as pd
from loguru import logger


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset"
    / "chennai_flood_onset_ml_dataset.csv"
)

FLOOD_LABELS = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flood_labels"
    / "chennai_daily_flood_labels.csv"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset"
    / "chennai_flood_3day_ml_dataset.csv"
)


# ============================================================
# Load data
# ============================================================

def load_data():
    logger.info("Loading feature dataset...")
    features = pd.read_csv(FEATURE_DATASET)

    logger.info("Loading original flood labels...")
    labels = pd.read_csv(FLOOD_LABELS)

    logger.info(f"Feature dataset shape: {features.shape}")
    logger.info(f"Flood labels shape: {labels.shape}")

    return features, labels


# ============================================================
# Validate inputs
# ============================================================

def validate_inputs(features, labels):

    if "date" not in features.columns:
        raise ValueError("Feature dataset does not contain 'date' column.")

    if "date" not in labels.columns:
        raise ValueError("Flood labels dataset does not contain 'date' column.")

    if "flood_event" not in labels.columns:
        raise ValueError(
            "Original flood labels dataset does not contain 'flood_event'."
        )

    # Convert dates
    features["date"] = pd.to_datetime(features["date"])
    labels["date"] = pd.to_datetime(labels["date"])

    # Check duplicate dates
    if features["date"].duplicated().any():
        raise ValueError("Duplicate dates found in feature dataset.")

    if labels["date"].duplicated().any():
        logger.warning(
            "Duplicate dates found in flood labels. "
            "Please inspect the label dataset."
        )

    logger.info(
        f"Feature date range: "
        f"{features['date'].min().date()} → {features['date'].max().date()}"
    )

    logger.info(
        f"Label date range: "
        f"{labels['date'].min().date()} → {labels['date'].max().date()}"
    )


# ============================================================
# Merge flood labels
# ============================================================

def merge_labels(features, labels):

    logger.info("Merging authoritative flood labels by date...")

    # Only bring the flood_event column.
    # event_id is not required for the ML target.
    labels_small = labels[["date", "flood_event"]].copy()

    df = features.merge(
        labels_small,
        on="date",
        how="left",
        validate="one_to_one"
    )

    missing_labels = df["flood_event"].isna().sum()

    if missing_labels > 0:
        raise ValueError(
            f"{missing_labels} feature rows do not have a corresponding "
            f"flood label."
        )

    df["flood_event"] = df["flood_event"].astype(int)

    logger.info(
        f"Successfully merged flood labels for {len(df)} rows."
    )

    logger.info(
        f"Flood days in merged data: "
        f"{df['flood_event'].sum()}"
    )

    return df


# ============================================================
# Create 3-day target
# ============================================================

def create_3day_target(df):

    logger.info("Creating 3-day forward flood prediction target...")

    flood = df["flood_event"].astype(int)

    # Information available on day t is used to predict
    # whether ANY flood day occurs on t+1, t+2, or t+3.
    future_day_1 = flood.shift(-1)
    future_day_2 = flood.shift(-2)
    future_day_3 = flood.shift(-3)

    df["target_flood_next_3_days"] = (
        (future_day_1 == 1)
        | (future_day_2 == 1)
        | (future_day_3 == 1)
    ).astype(int)

    return df


# ============================================================
# Clean dataset
# ============================================================

def clean_dataset(df):

    logger.info("Cleaning final 3-day ML dataset...")

    # Last 3 rows cannot have complete +1/+2/+3 future information.
    df = df.iloc[:-3].copy()

    # Remove columns that should not be model inputs.
    columns_to_drop = [
        "flood_event",
        "target_flood_onset_next_day",
        "year"
    ]

    existing_columns = [
        col for col in columns_to_drop
        if col in df.columns
    ]

    df.drop(columns=existing_columns, inplace=True)

    # Sort chronologically
    df.sort_values("date", inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


# ============================================================
# Dataset report
# ============================================================

def print_dataset_report(df):

    logger.info("=" * 60)
    logger.info("3-DAY FORECAST DATASET REPORT")
    logger.info("=" * 60)

    logger.info(f"Shape: {df.shape}")

    logger.info(
        f"Date range: "
        f"{df['date'].min().date()} → {df['date'].max().date()}"
    )

    positive = int(df["target_flood_next_3_days"].sum())
    negative = len(df) - positive

    logger.info(f"Positive cases: {positive}")
    logger.info(f"Negative cases: {negative}")

    logger.info(
        f"Positive rate: "
        f"{positive / len(df) * 100:.3f}%"
    )

    logger.info("\nTarget distribution by year:")

    temp = df.copy()
    temp["year_report"] = temp["date"].dt.year

    yearly = (
        temp.groupby("year_report")["target_flood_next_3_days"]
        .agg(["sum", "count"])
    )

    yearly["positive_rate_%"] = (
        yearly["sum"] / yearly["count"] * 100
    )

    print(yearly.to_string())

    logger.info("=" * 60)


# ============================================================
# Main
# ============================================================

def main():

    logger.info(
        "Starting FloodGuard 3-Day Forecast Dataset Pipeline..."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    features, labels = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_inputs(features, labels)

    # --------------------------------------------------------
    # Merge authoritative labels
    # --------------------------------------------------------

    df = merge_labels(features, labels)

    # --------------------------------------------------------
    # Create target
    # --------------------------------------------------------

    df = create_3day_target(df)

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    df = clean_dataset(df)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_dataset_report(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_DATASET,
        index=False
    )

    logger.info(
        f"3-day ML dataset saved to:\n{OUTPUT_DATASET}"
    )

    logger.info(
        "3-Day Forecast Dataset Pipeline completed successfully."
    )


if __name__ == "__main__":
    main()
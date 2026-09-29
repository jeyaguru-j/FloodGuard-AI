"""
FloodGuard - Flood Onset Feature Engineering Pipeline

Purpose:
    Build a leakage-safe ML dataset for predicting whether a
    flood event will START tomorrow.

Target:
    target_flood_onset_next_day

Meaning:
    1 -> Today is not a flood day, but tomorrow starts a flood event
    0 -> No flood event starts tomorrow

Important:
    The original next-day flood dataset is preserved as a baseline.
    This pipeline creates a separate onset dataset.

Input:
    data/raw/weather/weather_chennai.csv
    data/processed/flood_labels/chennai_daily_flood_labels.csv

Output:
    data/processed/ml_dataset/chennai_flood_onset_ml_dataset.csv
"""

from pathlib import Path

import pandas as pd
from loguru import logger


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

WEATHER_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "weather"
    / "weather_chennai.csv"
)

FLOOD_LABEL_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "flood_labels"
    / "chennai_daily_flood_labels.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "ml_dataset"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "chennai_flood_onset_ml_dataset.csv"
)


# ============================================================
# LOAD WEATHER DATA
# ============================================================

def load_weather_data() -> pd.DataFrame:

    logger.info(
        f"Loading weather data from {WEATHER_PATH}"
    )

    if not WEATHER_PATH.exists():
        raise FileNotFoundError(
            f"Weather dataset not found: {WEATHER_PATH}"
        )

    df = pd.read_csv(WEATHER_PATH)

    required_columns = [
        "date",
        "rainfall_mm",
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "relative_humidity_percent",
        "wind_speed_max_kmh",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Weather dataset missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            "Weather dataset contains invalid dates."
        )

    df = (
        df
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    logger.info(
        f"Weather dataset loaded successfully. "
        f"Shape: {df.shape}"
    )

    return df


# ============================================================
# LOAD FLOOD LABELS
# ============================================================

def load_flood_labels() -> pd.DataFrame:

    logger.info(
        f"Loading flood labels from {FLOOD_LABEL_PATH}"
    )

    if not FLOOD_LABEL_PATH.exists():
        raise FileNotFoundError(
            f"Flood label dataset not found: {FLOOD_LABEL_PATH}"
        )

    df = pd.read_csv(FLOOD_LABEL_PATH)

    required_columns = [
        "date",
        "flood_event",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Flood label dataset missing columns: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():
        raise ValueError(
            "Flood label dataset contains invalid dates."
        )

    df["flood_event"] = (
        df["flood_event"]
        .fillna(0)
        .astype(int)
    )

    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    logger.info(
        f"Flood label dataset loaded successfully. "
        f"Shape: {df.shape}"
    )

    return df


# ============================================================
# MERGE DATASETS
# ============================================================

def merge_datasets(
    weather_df: pd.DataFrame,
    flood_df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Merging weather and flood labels..."
    )

    df = weather_df.merge(
        flood_df[["date", "flood_event"]],
        on="date",
        how="left",
    )

    # No recorded flood event = non-flood day
    df["flood_event"] = (
        df["flood_event"]
        .fillna(0)
        .astype(int)
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    logger.info(
        f"Merged dataset shape: {df.shape}"
    )

    return df


# ============================================================
# CALENDAR FEATURES
# ============================================================

def create_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Creating calendar features..."
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek

    # Chennai Northeast Monsoon
    df["northeast_monsoon"] = (
        df["month"]
        .isin([10, 11, 12])
        .astype(int)
    )

    # Southwest Monsoon
    df["southwest_monsoon"] = (
        df["month"]
        .isin([6, 7, 8, 9])
        .astype(int)
    )

    return df


# ============================================================
# RAINFALL FEATURES
# ============================================================

def create_rainfall_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Creating rainfall and rainfall-trajectory features..."
    )

    rainfall = df["rainfall_mm"]

    # --------------------------------------------------------
    # Previous-day rainfall
    # --------------------------------------------------------

    df["rainfall_lag_1d"] = rainfall.shift(1)
    df["rainfall_lag_2d"] = rainfall.shift(2)
    df["rainfall_lag_3d"] = rainfall.shift(3)

    # --------------------------------------------------------
    # Historical rainfall windows
    #
    # These exclude today's rainfall.
    # This preserves the original leakage-safe design.
    # --------------------------------------------------------

    df["rainfall_3d_sum"] = (
        rainfall
        .shift(1)
        .rolling(3)
        .sum()
    )

    df["rainfall_7d_sum"] = (
        rainfall
        .shift(1)
        .rolling(7)
        .sum()
    )

    df["rainfall_14d_sum"] = (
        rainfall
        .shift(1)
        .rolling(14)
        .sum()
    )

    df["rainfall_3d_max"] = (
        rainfall
        .shift(1)
        .rolling(3)
        .max()
    )

    df["rainfall_7d_max"] = (
        rainfall
        .shift(1)
        .rolling(7)
        .max()
    )

    # --------------------------------------------------------
    # Rainfall windows INCLUDING TODAY
    #
    # These are valid because today's weather is known when
    # generating a prediction for tomorrow.
    # --------------------------------------------------------

    df["rainfall_2d_current_sum"] = (
        rainfall
        .rolling(2)
        .sum()
    )

    df["rainfall_3d_current_sum"] = (
        rainfall
        .rolling(3)
        .sum()
    )

    df["rainfall_7d_current_sum"] = (
        rainfall
        .rolling(7)
        .sum()
    )

    df["rainfall_14d_current_sum"] = (
        rainfall
        .rolling(14)
        .sum()
    )

    # --------------------------------------------------------
    # Current-window rainfall peaks
    # --------------------------------------------------------

    df["rainfall_3d_current_max"] = (
        rainfall
        .rolling(3)
        .max()
    )

    df["rainfall_7d_current_max"] = (
        rainfall
        .rolling(7)
        .max()
    )

    df["rainfall_14d_current_max"] = (
        rainfall
        .rolling(14)
        .max()
    )

    # --------------------------------------------------------
    # Rainfall acceleration
    #
    # Measures whether rainfall conditions are intensifying.
    # --------------------------------------------------------

    df["rainfall_3d_change"] = (
        df["rainfall_3d_current_sum"]
        - df["rainfall_3d_sum"]
    )

    df["rainfall_7d_change"] = (
        df["rainfall_7d_current_sum"]
        - df["rainfall_7d_sum"]
    )

    # --------------------------------------------------------
    # Rainfall intensity indicators
    # --------------------------------------------------------

    df["heavy_rain_25mm"] = (
        rainfall >= 25
    ).astype(int)

    df["heavy_rain_50mm"] = (
        rainfall >= 50
    ).astype(int)

    df["heavy_rain_100mm"] = (
        rainfall >= 100
    ).astype(int)

    # --------------------------------------------------------
    # Heavy rainfall day counts
    # --------------------------------------------------------

    df["heavy_rain_days_3d"] = (
        df["heavy_rain_25mm"]
        .shift(1)
        .rolling(3)
        .sum()
    )

    df["heavy_rain_days_7d"] = (
        df["heavy_rain_25mm"]
        .shift(1)
        .rolling(7)
        .sum()
    )

    df["heavy_rain_days_14d"] = (
        df["heavy_rain_25mm"]
        .shift(1)
        .rolling(14)
        .sum()
    )

    # --------------------------------------------------------
    # Current heavy-rain indicators
    # --------------------------------------------------------

    df["heavy_rain_days_3d_current"] = (
        df["heavy_rain_25mm"]
        .rolling(3)
        .sum()
    )

    df["heavy_rain_days_7d_current"] = (
        df["heavy_rain_25mm"]
        .rolling(7)
        .sum()
    )

    # --------------------------------------------------------
    # Rainfall persistence
    #
    # Number of consecutive rainy days ending today.
    # --------------------------------------------------------

    rainy = (
        rainfall > 0
    ).astype(int)

    groups = (
        rainy.eq(0)
        .cumsum()
    )

    df["consecutive_rainy_days"] = (
        rainy
        .groupby(groups)
        .cumsum()
    )

    # --------------------------------------------------------
    # Rainfall ratios
    #
    # Useful for distinguishing recent intense rainfall from
    # long-term accumulated rainfall.
    # --------------------------------------------------------

    df["rainfall_3d_to_14d_ratio"] = (
        df["rainfall_3d_current_sum"]
        / (df["rainfall_14d_current_sum"] + 1e-6)
    )

    df["rainfall_7d_to_14d_ratio"] = (
        df["rainfall_7d_current_sum"]
        / (df["rainfall_14d_current_sum"] + 1e-6)
    )

    # --------------------------------------------------------
    # Recent rainfall dominance
    # --------------------------------------------------------

    df["rainfall_today_fraction_7d"] = (
        rainfall
        / (df["rainfall_7d_current_sum"] + 1e-6)
    )

    # --------------------------------------------------------
    # Rainy days
    # --------------------------------------------------------

    df["rainy_days_7d"] = (
        rainfall
        .shift(1)
        .gt(0)
        .rolling(7)
        .sum()
    )

    df["rainy_days_14d"] = (
        rainfall
        .shift(1)
        .gt(0)
        .rolling(14)
        .sum()
    )

    return df


# ============================================================
# TEMPERATURE FEATURES
# ============================================================

def create_temperature_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Creating temperature features..."
    )

    df["temperature_range_c"] = (
        df["temperature_max_c"]
        - df["temperature_min_c"]
    )

    df["temperature_lag_1d"] = (
        df["temperature_mean_c"]
        .shift(1)
    )

    df["temperature_3d_mean"] = (
        df["temperature_mean_c"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["temperature_7d_mean"] = (
        df["temperature_mean_c"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    return df


# ============================================================
# HUMIDITY FEATURES
# ============================================================

def create_humidity_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Creating humidity features..."
    )

    humidity = df[
        "relative_humidity_percent"
    ]

    df["humidity_lag_1d"] = (
        humidity.shift(1)
    )

    df["humidity_3d_mean"] = (
        humidity
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["humidity_7d_mean"] = (
        humidity
        .shift(1)
        .rolling(7)
        .mean()
    )

    return df


# ============================================================
# WIND FEATURES
# ============================================================

def create_wind_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Creating wind features..."
    )

    wind = df["wind_speed_max_kmh"]

    df["wind_lag_1d"] = (
        wind.shift(1)
    )

    df["wind_3d_mean"] = (
        wind
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["wind_7d_mean"] = (
        wind
        .shift(1)
        .rolling(7)
        .mean()
    )

    return df


# ============================================================
# TARGET
# ============================================================

def create_target(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Creating flood-onset prediction target..."
    )

    # Flood status tomorrow
    next_day_flood = (
        df["flood_event"]
        .shift(-1)
    )

    # A flood onset occurs when:
    #
    # today      = no flood
    # tomorrow   = flood
    #
    df["target_flood_onset_next_day"] = (
        (
            (df["flood_event"] == 0)
            &
            (next_day_flood == 1)
        )
        .astype(int)
    )

    return df


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Cleaning temporal feature dataset..."
    )

    initial_rows = len(df)

    # Largest rolling window is 14 days.
    minimum_history = 14

    df = (
        df
        .iloc[minimum_history:]
        .copy()
    )

    # Last row has no tomorrow.
    df = (
        df
        .iloc[:-1]
        .copy()
    )

    feature_columns = [
        # Rainfall history
        "rainfall_lag_1d",
        "rainfall_lag_2d",
        "rainfall_lag_3d",
        "rainfall_3d_sum",
        "rainfall_7d_sum",
        "rainfall_14d_sum",
        "rainfall_3d_max",
        "rainfall_7d_max",

        # Current rainfall windows
        "rainfall_2d_current_sum",
        "rainfall_3d_current_sum",
        "rainfall_7d_current_sum",
        "rainfall_14d_current_sum",
        "rainfall_3d_current_max",
        "rainfall_7d_current_max",
        "rainfall_14d_current_max",

        # Rainfall trajectory
        "rainfall_3d_change",
        "rainfall_7d_change",

        # Heavy rainfall
        "heavy_rain_days_3d",
        "heavy_rain_days_7d",
        "heavy_rain_days_14d",
        "heavy_rain_days_3d_current",
        "heavy_rain_days_7d_current",

        # Rainfall persistence
        "consecutive_rainy_days",

        # Rainfall ratios
        "rainfall_3d_to_14d_ratio",
        "rainfall_7d_to_14d_ratio",
        "rainfall_today_fraction_7d",

        # Existing rainy-day features
        "rainy_days_7d",
        "rainy_days_14d",

        # Temperature
        "temperature_lag_1d",
        "temperature_3d_mean",
        "temperature_7d_mean",

        # Humidity
        "humidity_lag_1d",
        "humidity_3d_mean",
        "humidity_7d_mean",

        # Wind
        "wind_lag_1d",
        "wind_3d_mean",
        "wind_7d_mean",
    ]

    df = df.dropna(
        subset=feature_columns
    )

    df["target_flood_onset_next_day"] = (
        df["target_flood_onset_next_day"]
        .astype(int)
    )

    removed_rows = (
        initial_rows - len(df)
    )

    logger.info(
        f"Removed {removed_rows} rows due to "
        f"temporal history/target requirements."
    )

    return (
        df
        .reset_index(drop=True)
    )


# ============================================================
# SELECT ML FEATURES
# ============================================================

def select_ml_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info(
        "Selecting ML features..."
    )

    feature_columns = [

        # Date
        "date",

        # Raw weather available today
        "rainfall_mm",
        "temperature_max_c",
        "temperature_min_c",
        "temperature_mean_c",
        "relative_humidity_percent",
        "wind_speed_max_kmh",

        # Calendar
        "year",
        "month",
        "day_of_year",
        "day_of_week",
        "northeast_monsoon",
        "southwest_monsoon",

        # Rainfall history
        "rainfall_lag_1d",
        "rainfall_lag_2d",
        "rainfall_lag_3d",
        "rainfall_3d_sum",
        "rainfall_7d_sum",
        "rainfall_14d_sum",
        "rainfall_3d_max",
        "rainfall_7d_max",

        # Current rainfall accumulation
        "rainfall_2d_current_sum",
        "rainfall_3d_current_sum",
        "rainfall_7d_current_sum",
        "rainfall_14d_current_sum",

        # Current rainfall peaks
        "rainfall_3d_current_max",
        "rainfall_7d_current_max",
        "rainfall_14d_current_max",

        # Rainfall trajectory
        "rainfall_3d_change",
        "rainfall_7d_change",

        # Heavy rainfall
        "heavy_rain_25mm",
        "heavy_rain_50mm",
        "heavy_rain_100mm",
        "heavy_rain_days_3d",
        "heavy_rain_days_7d",
        "heavy_rain_days_14d",
        "heavy_rain_days_3d_current",
        "heavy_rain_days_7d_current",

        # Rainfall persistence
        "consecutive_rainy_days",

        # Rainfall ratios
        "rainfall_3d_to_14d_ratio",
        "rainfall_7d_to_14d_ratio",
        "rainfall_today_fraction_7d",

        # Rainy days
        "rainy_days_7d",
        "rainy_days_14d",

        # Temperature
        "temperature_range_c",
        "temperature_lag_1d",
        "temperature_3d_mean",
        "temperature_7d_mean",

        # Humidity
        "humidity_lag_1d",
        "humidity_3d_mean",
        "humidity_7d_mean",

        # Wind
        "wind_lag_1d",
        "wind_3d_mean",
        "wind_7d_mean",

        # Target
        "target_flood_onset_next_day",
    ]

    df = df[
        feature_columns
    ].copy()

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
) -> None:

    logger.info(
        "Running final dataset validation..."
    )

    if df.empty:
        raise ValueError(
            "Final ML dataset is empty."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates found."
        )

    if df.isna().any().any():

        missing = df.isna().sum()
        missing = missing[
            missing > 0
        ]

        raise ValueError(
            f"Missing values found:\n{missing}"
        )

    if (
        "target_flood_onset_next_day"
        not in df.columns
    ):
        raise ValueError(
            "Onset target column missing."
        )

    target_values = set(
        df[
            "target_flood_onset_next_day"
        ].unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Invalid target values: "
            f"{target_values}"
        )

    logger.info(
        f"Final dataset shape: {df.shape}"
    )

    logger.info(
        f"Date range: "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    logger.info(
        "Onset target distribution:\n"
        +
        df[
            "target_flood_onset_next_day"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    onset_count = (
        df[
            "target_flood_onset_next_day"
        ]
        .sum()
    )

    total = len(df)

    onset_percentage = (
        onset_count / total * 100
    )

    logger.info(
        f"Flood onset target percentage: "
        f"{onset_percentage:.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

def save_dataset(
    df: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    logger.success(
        f"Onset ML dataset saved successfully: "
        f"{OUTPUT_PATH}"
    )


# ============================================================
# PIPELINE
# ============================================================

def run() -> None:

    logger.info(
        "Starting FloodGuard Flood Onset "
        "Feature Engineering Pipeline..."
    )

    weather_df = (
        load_weather_data()
    )

    flood_df = (
        load_flood_labels()
    )

    df = merge_datasets(
        weather_df,
        flood_df,
    )

    df = create_calendar_features(df)

    df = create_rainfall_features(df)

    df = create_temperature_features(df)

    df = create_humidity_features(df)

    df = create_wind_features(df)

    df = create_target(df)

    df = clean_dataset(df)

    df = select_ml_features(df)

    validate_dataset(df)

    save_dataset(df)

    logger.success(
        "FloodGuard Flood Onset Feature "
        "Engineering Pipeline completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()
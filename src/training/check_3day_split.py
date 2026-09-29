from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset"
    / "chennai_flood_3day_ml_dataset.csv"
)


def main():

    print("=" * 70)
    print("FloodGuard 3-Day Forecast Split Validation")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    # --------------------------------------------------------
    # Chronological splits
    # --------------------------------------------------------

    train = df[
        (df["date"].dt.year >= 2010) &
        (df["date"].dt.year <= 2018)
    ]

    validation = df[
        (df["date"].dt.year >= 2019) &
        (df["date"].dt.year <= 2020)
    ]

    test = df[
        (df["date"].dt.year >= 2021) &
        (df["date"].dt.year <= 2022)
    ]

    # --------------------------------------------------------
    # Print split information
    # --------------------------------------------------------

    splits = {
        "TRAIN": train,
        "VALIDATION": validation,
        "TEST": test
    }

    for name, split in splits.items():

        positives = int(
            split["target_flood_next_3_days"].sum()
        )

        negatives = len(split) - positives

        positive_rate = (
            positives / len(split) * 100
            if len(split) > 0
            else 0
        )

        print()
        print(f"{name}")
        print("-" * 50)

        print(f"Rows          : {len(split)}")
        print(f"Positive cases: {positives}")
        print(f"Negative cases: {negatives}")
        print(f"Positive rate : {positive_rate:.3f}%")

        if len(split) > 0:
            print(
                f"Date range    : "
                f"{split['date'].min().date()} → "
                f"{split['date'].max().date()}"
            )

    # --------------------------------------------------------
    # Year-wise distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Positive Cases by Year")
    print("=" * 70)

    yearly = (
        df.assign(year=df["date"].dt.year)
        .groupby("year")["target_flood_next_3_days"]
        .agg(["sum", "count"])
    )

    yearly["positive_rate_%"] = (
        yearly["sum"] / yearly["count"] * 100
    )

    print(yearly.to_string())

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Validation Checks")
    print("=" * 70)

    print(
        "Train/Validation overlap:",
        len(set(train.index) & set(validation.index))
    )

    print(
        "Train/Test overlap:",
        len(set(train.index) & set(test.index))
    )

    print(
        "Validation/Test overlap:",
        len(set(validation.index) & set(test.index))
    )

    print()
    print("Split validation completed.")


if __name__ == "__main__":
    main()
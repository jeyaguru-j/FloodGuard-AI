from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flood_labels"
    / "chennai_daily_flood_labels.csv"
)


def main():

    df = pd.read_csv(LABEL_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    flood = df["flood_event"].astype(int)

    print("=" * 70)
    print("FloodGuard Forecast-Horizon Analysis")
    print("=" * 70)

    horizons = [1, 3, 5, 7]

    results = []

    for horizon in horizons:

        future_flood = pd.Series(
            0,
            index=df.index,
            dtype=int,
        )

        for step in range(1, horizon + 1):

            future_flood = (
                future_flood
                | flood.shift(-step).fillna(0).astype(int)
            )

        positive_count = int(
            future_flood.sum()
        )

        percentage = (
            positive_count / len(df)
        ) * 100

        results.append(
            {
                "horizon_days": horizon,
                "positive_days": positive_count,
                "negative_days": len(df) - positive_count,
                "positive_percentage": percentage,
            }
        )

    result_df = pd.DataFrame(results)

    print("\nForecast Horizon Summary\n")

    print(
        result_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("Examples around November 2021")
    print("=" * 70)

    display_start = pd.Timestamp("2021-11-01")
    display_end = pd.Timestamp("2021-11-15")

    display_df = df[
        (df["date"] >= display_start)
        & (df["date"] <= display_end)
    ][
        ["date", "flood_event"]
    ].copy()

    for horizon in horizons:

        future_flood = pd.Series(
            0,
            index=df.index,
            dtype=int,
        )

        for step in range(1, horizon + 1):

            future_flood = (
                future_flood
                | flood.shift(-step).fillna(0).astype(int)
            )

        display_df[
            f"next_{horizon}d_flood"
        ] = future_flood.loc[
            display_df.index
        ].values

    print(
        display_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
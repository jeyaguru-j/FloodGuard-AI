import pandas as pd

FILE_PATH = (
    "data/raw/flood_events/ifi/"
    "India_Flood_Inventory_v3.csv"
)


def main():

    df = pd.read_csv(FILE_PATH)

    print("\n========== STATE ANALYSIS ==========\n")

    print("Unique states:")
    print(df["State"].unique())

    print("\nFlood events by state:")
    print(
        df["State"]
        .value_counts()
        .to_string()
    )

    print("\n========== TAMIL NADU EVENTS ==========\n")

    tamil_nadu = df[
        df["State"]
        .astype(str)
        .str.contains(
            "Tamil Nadu",
            case=False,
            na=False
        )
    ]

    print(
        f"Tamil Nadu events: {len(tamil_nadu)}"
    )

    print("\nTamil Nadu districts:")
    print(
        tamil_nadu["Districts"]
        .dropna()
        .astype(str)
        .value_counts()
        .to_string()
    )
    print("\n========== CHENNAI SEARCH ==========\n")

    chennai_matches = tamil_nadu[
        tamil_nadu["Districts"]
        .astype(str)
        .str.contains(
            "Chennai|Madras",
            case=False,
            na=False,
            regex=True
        )
    ]

    print(
        f"Potential Chennai events: "
        f"{len(chennai_matches)}"
    )

    print("\nMatching records:")

    print(
        chennai_matches[
            [
                "UEI",
                "Start Date",
                "End Date",
                "Main Cause",
                "Districts",
                "State",
                "Human fatality",
                "Human injured",
                "Human Displaced",
                "Event Source",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
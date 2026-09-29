import pandas as pd


FILE_PATH = (
    "data/raw/flood_events/ifi/"
    "India_Flood_Inventory_v3.csv"
)


def load_data():
    return pd.read_csv(FILE_PATH)


def identify_chennai_events(df):
    """
    Identify events that explicitly mention Chennai
    in the district field.
    """

    district_match = (
        df["Districts"]
        .astype(str)
        .str.contains(
            r"\bChennai\b",
            case=False,
            na=False,
            regex=True
        )
    )

    historical_madras = (
        df["State"]
        .astype(str)
        .str.contains(
            r"\bMadras\b",
            case=False,
            na=False,
            regex=True
        )
    )

    tamil_nadu = (
        df["State"]
        .astype(str)
        .str.contains(
            r"\bTamil Nadu\b",
            case=False,
            na=False,
            regex=True
        )
    )

    return df[
        (tamil_nadu & district_match)
        | historical_madras
    ].copy()


def analyze_dates(df):

    df["start_date"] = pd.to_datetime(
        df["Start Date"],
        errors="coerce",
        dayfirst=True
    )

    df["end_date"] = pd.to_datetime(
        df["End Date"],
        errors="coerce",
        dayfirst=True
    )

    df["calculated_duration"] = (
        df["end_date"] - df["start_date"]
    ).dt.days + 1

    print("\n========== DATE ANALYSIS ==========\n")

    print(
        df[
            [
                "UEI",
                "start_date",
                "end_date",
                "Duration(Days)",
                "calculated_duration"
            ]
        ].to_string(index=False)
    )

    print("\n========== SUSPICIOUS DURATIONS ==========\n")

    suspicious = df[
        df["calculated_duration"] > 30
    ]

    if suspicious.empty:
        print("No events longer than 30 days.")
    else:
        print(
            suspicious[
                [
                    "UEI",
                    "start_date",
                    "end_date",
                    "Duration(Days)",
                    "Main Cause"
                ]
            ].to_string(index=False)
        )


def analyze_causes(df):

    print("\n========== MAIN CAUSES ==========\n")

    print(
        df["Main Cause"]
        .fillna("Unknown")
        .value_counts()
        .to_string()
    )


def analyze_years(df):

    df["start_date"] = pd.to_datetime(
        df["Start Date"],
        errors="coerce",
        dayfirst=True
    )

    print("\n========== EVENTS BY YEAR ==========\n")

    print(
        df["start_date"]
        .dt.year
        .value_counts()
        .sort_index()
        .to_string()
    )


def main():

    print("\n========== CHENNAI FLOOD ANALYSIS ==========\n")

    df = load_data()

    chennai = identify_chennai_events(df)

    print(
        f"Total Chennai-related events: {len(chennai)}"
    )

    print("\nSelected Events:")

    print(
        chennai[
            [
                "UEI",
                "Start Date",
                "End Date",
                "Main Cause",
                "Districts",
                "State",
                "Human fatality",
                "Human injured"
            ]
        ].to_string(index=False)
    )

    analyze_dates(chennai)

    analyze_causes(chennai)

    analyze_years(chennai)


if __name__ == "__main__":
    main()
import pandas as pd

FILE_PATH = "data/raw/weather/weather_chennai.csv"


def main():

    print("\n========== WEATHER DATA QUALITY CHECK ==========\n")

    df = pd.read_csv(FILE_PATH)

    # Basic information
    print("Dataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # Date range
    df["date"] = pd.to_datetime(df["date"])

    print("\nDate Range:")
    print("Start:", df["date"].min())
    print("End  :", df["date"].max())

    # Duplicate rows
    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Unique cities
    print("\nCities:")
    print(df["city"].unique())

    # Coordinate consistency
    print("\nLatitude:")
    print(df["latitude"].unique())

    print("\nLongitude:")
    print(df["longitude"].unique())

    # Basic statistics
    print("\nNumerical Summary:")
    print(
        df[
            [
                "rainfall_mm",
                "temperature_max_c",
                "temperature_min_c",
                "temperature_mean_c",
                "relative_humidity_percent",
                "wind_speed_max_kmh",
            ]
        ].describe()
    )

    print("\n========== CHECK COMPLETED ==========\n")


if __name__ == "__main__":
    main()
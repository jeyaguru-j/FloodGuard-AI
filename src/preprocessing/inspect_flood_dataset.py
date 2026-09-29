import pandas as pd

FILE_PATH = (
    "data/raw/flood_events/ifi/"
    "India_Flood_Inventory_v3.csv"
)


def main():

    print("\n========== IFI DATASET INSPECTION ==========\n")

    df = pd.read_csv(FILE_PATH)

    print("Dataset Shape:")
    print(df.shape)

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\n========== INSPECTION COMPLETED ==========\n")


if __name__ == "__main__":
    main()
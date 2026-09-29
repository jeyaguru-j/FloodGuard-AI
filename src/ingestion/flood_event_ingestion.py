from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import logger


class FloodEventIngestion:
    """
    Extracts and validates Chennai-related historical
    flood events from the India Flood Inventory.
    """

    def __init__(
        self,
        input_path: str = (
            "data/raw/flood_events/ifi/"
            "India_Flood_Inventory_v3.csv"
        ),
        output_directory: str = (
            "data/raw/flood_events/chennai"
        ),
        output_filename: str = "chennai_flood_events.csv",
    ):

        self.input_path = Path(input_path)
        self.output_directory = Path(output_directory)
        self.output_filename = output_filename

    def load_data(self) -> pd.DataFrame:

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"IFI dataset not found: {self.input_path}"
            )

        logger.info(
            f"Loading IFI dataset from {self.input_path}"
        )

        dataframe = pd.read_csv(self.input_path)

        logger.info(
            f"IFI dataset loaded successfully. "
            f"Shape: {dataframe.shape}"
        )

        return dataframe

    @staticmethod
    def normalize_text(value: Any) -> str:

        if pd.isna(value):
            return ""

        return " ".join(
            str(value).strip().split()
        )

    def identify_chennai_events(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "Identifying Chennai-related flood events..."
        )

        district_text = (
            dataframe["Districts"]
            .fillna("")
            .astype(str)
        )

        state_text = (
            dataframe["State"]
            .fillna("")
            .astype(str)
        )

        # Chennai explicitly mentioned in district field
        chennai_match = district_text.str.contains(
            r"\bChennai\b",
            case=False,
            na=False,
            regex=True
        )

        # Historical Madras record
        madras_match = state_text.str.contains(
            r"\bMadras\b",
            case=False,
            na=False,
            regex=True
        )

        chennai_events = dataframe[
            chennai_match | madras_match
        ].copy()

        logger.info(
            f"Chennai-related events identified: "
            f"{len(chennai_events)}"
        )

        return chennai_events

    @staticmethod
    def parse_dates(
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        dataframe["start_date"] = pd.to_datetime(
            dataframe["Start Date"],
            errors="coerce",
            dayfirst=True
        )

        dataframe["end_date"] = pd.to_datetime(
            dataframe["End Date"],
            errors="coerce",
            dayfirst=True
        )

        return dataframe

    @staticmethod
    def calculate_duration(
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        dataframe["calculated_duration_days"] = (
            dataframe["end_date"]
            - dataframe["start_date"]
        ).dt.days + 1

        return dataframe

    @staticmethod
    def classify_date_quality(
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        conditions = [
            dataframe["start_date"].isna()
            | dataframe["end_date"].isna(),

            dataframe["end_date"]
            < dataframe["start_date"],

            dataframe["calculated_duration_days"] > 30,

            (
                dataframe["Duration(Days)"].notna()
                &
                (
                    dataframe["Duration(Days)"]
                    != dataframe["calculated_duration_days"]
                )
            ),
        ]

        choices = [
            "missing_date",
            "invalid_date_order",
            "suspicious_long_event",
            "duration_field_mismatch",
        ]

        dataframe["date_quality"] = (
            pd.Series(
                "valid",
                index=dataframe.index
            )
        )

        for condition, choice in zip(
            conditions,
            choices
        ):
            dataframe.loc[
                condition,
                "date_quality"
            ] = choice

        return dataframe

    @staticmethod
    def normalize_state(
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        dataframe["state_original"] = (
            dataframe["State"]
            .apply(
                lambda value:
                str(value).strip()
                if not pd.isna(value)
                else ""
            )
        )

        dataframe["state_normalized"] = (
            dataframe["state_original"]
            .replace(
                {
                    "Madras": "Tamil Nadu"
                }
            )
        )

        return dataframe

    @staticmethod
    def build_output_dataframe(
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        output = pd.DataFrame()

        output["event_id"] = dataframe["UEI"]

        output["start_date"] = dataframe["start_date"]

        output["end_date"] = dataframe["end_date"]

        output["duration_days"] = (
            dataframe["calculated_duration_days"]
        )

        output["duration_source"] = (
            "calculated_from_dates"
        )

        output["date_quality"] = (
            dataframe["date_quality"]
        )

        output["main_cause"] = (
            dataframe["Main Cause"]
        )

        output["districts"] = (
            dataframe["Districts"]
        )

        output["state_original"] = (
            dataframe["state_original"]
        )

        output["state_normalized"] = (
            dataframe["state_normalized"]
        )

        output["human_fatality"] = (
            dataframe["Human fatality"]
        )

        output["human_injured"] = (
            dataframe["Human injured"]
        )

        output["event_source"] = (
            dataframe["Event Source"]
        )

        return output

    def save_data(
        self,
        dataframe: pd.DataFrame
    ) -> Path:

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            self.output_directory
            / self.output_filename
        )

        dataframe.to_csv(
            output_path,
            index=False
        )

        logger.success(
            f"Chennai flood events saved to: "
            f"{output_path}"
        )

        return output_path

    def run(self) -> Path:

        logger.info(
            "Starting Chennai Flood Event Ingestion..."
        )

        dataframe = self.load_data()

        dataframe = self.identify_chennai_events(
            dataframe
        )

        dataframe = self.parse_dates(
            dataframe
        )

        dataframe = self.calculate_duration(
            dataframe
        )

        dataframe = self.classify_date_quality(
            dataframe
        )

        dataframe = self.normalize_state(
            dataframe
        )

        output = self.build_output_dataframe(
            dataframe
        )

        logger.info(
            "Flood event validation summary:"
        )

        total_events = len(output)

        valid_events = (
            output["date_quality"] == "valid"
        ).sum()

        suspicious_events = (
            output["date_quality"] == "suspicious_long_event"
        ).sum()

        duration_mismatches = (
            output["date_quality"] == "duration_field_mismatch"
        ).sum()

        invalid_dates = (
            output["date_quality"]
            .isin([
                "missing_date",
                "invalid_date_order"
            ])
            .sum()
        )

        logger.info(
            f"Total events: {total_events}"
        )

        logger.info(
            f"Valid events: {valid_events}"
        )

        logger.info(
            f"Suspicious long events: {suspicious_events}"
        )

        logger.info(
            f"Duration mismatches: {duration_mismatches}"
        )

        logger.info(
            f"Missing/invalid dates: {invalid_dates}"
        )

        return self.save_data(output)


if __name__ == "__main__":

    ingestion = FloodEventIngestion()

    ingestion.run()
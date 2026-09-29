from pathlib import Path

import pandas as pd

from src.utils.logger import logger


class FloodLabelGenerator:

    def __init__(
        self,
        weather_path: str = (
            "data/raw/weather/weather_chennai.csv"
        ),
        flood_events_path: str = (
            "data/raw/flood_events/chennai/"
            "chennai_flood_events.csv"
        ),
        output_directory: str = (
            "data/processed/flood_labels"
        ),
        output_filename: str = (
            "chennai_daily_flood_labels.csv"
        ),
    ):

        self.weather_path = Path(weather_path)
        self.flood_events_path = Path(flood_events_path)

        self.output_directory = Path(
            output_directory
        )

        self.output_filename = output_filename

    def load_weather_data(self) -> pd.DataFrame:

        if not self.weather_path.exists():
            raise FileNotFoundError(
                f"Weather dataset not found: "
                f"{self.weather_path}"
            )

        logger.info(
            f"Loading weather data from "
            f"{self.weather_path}"
        )

        weather = pd.read_csv(
            self.weather_path
        )

        weather["date"] = pd.to_datetime(
            weather["date"],
            errors="coerce"
        )

        logger.info(
            f"Weather dataset loaded. "
            f"Shape: {weather.shape}"
        )

        return weather

    def load_flood_events(self) -> pd.DataFrame:

        if not self.flood_events_path.exists():
            raise FileNotFoundError(
                f"Flood event dataset not found: "
                f"{self.flood_events_path}"
            )

        logger.info(
            f"Loading flood events from "
            f"{self.flood_events_path}"
        )

        events = pd.read_csv(
            self.flood_events_path
        )

        events["start_date"] = pd.to_datetime(
            events["start_date"],
            errors="coerce"
        )

        events["end_date"] = pd.to_datetime(
            events["end_date"],
            errors="coerce"
        )

        logger.info(
            f"Flood event dataset loaded. "
            f"Shape: {events.shape}"
        )

        return events

    @staticmethod
    def select_high_confidence_events(
        events: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "Selecting high-confidence flood events..."
        )

        # We only use validated events.
        #
        # duration_field_mismatch is excluded because
        # the source duration conflicts with the dates.
        #
        # suspicious_long_event is also excluded.

        valid_events = events[
            events["date_quality"] == "valid"
        ].copy()

        # Our weather dataset starts in 2010.
        # IFI contains historical events before that,
        # but they cannot be matched to our weather data.

        valid_events = valid_events[
            valid_events["end_date"] >=
            pd.Timestamp("2010-01-01")
        ]

        valid_events = valid_events[
            valid_events["start_date"] <=
            pd.Timestamp("2025-12-31")
        ]

        logger.info(
            f"High-confidence events selected: "
            f"{len(valid_events)}"
        )

        return valid_events

    @staticmethod
    def create_daily_event_labels(
        weather: pd.DataFrame,
        events: pd.DataFrame
    ) -> pd.DataFrame:

        logger.info(
            "Creating daily flood labels..."
        )

        # Start with the complete weather date index.
        labels = weather[
            ["date"]
        ].copy()

        labels["flood_event"] = 0

        labels["event_id"] = None

        for _, event in events.iterrows():

            start_date = event["start_date"]
            end_date = event["end_date"]

            if pd.isna(start_date) or pd.isna(end_date):
                continue

            mask = (
                (labels["date"] >= start_date)
                &
                (labels["date"] <= end_date)
            )

            labels.loc[
                mask,
                "flood_event"
            ] = 1

            # Preserve the event that generated the label.
            labels.loc[
                mask,
                "event_id"
            ] = event["event_id"]

        return labels

    @staticmethod
    def calculate_label_statistics(
        labels: pd.DataFrame
    ):

        total_days = len(labels)

        flood_days = (
            labels["flood_event"] == 1
        ).sum()

        non_flood_days = (
            labels["flood_event"] == 0
        ).sum()

        flood_percentage = (
            flood_days / total_days
        ) * 100

        logger.info(
            "========== LABEL STATISTICS =========="
        )

        logger.info(
            f"Total days: {total_days}"
        )

        logger.info(
            f"Flood days: {flood_days}"
        )

        logger.info(
            f"Non-flood days: {non_flood_days}"
        )

        logger.info(
            f"Flood percentage: "
            f"{flood_percentage:.2f}%"
        )

    def save_data(
        self,
        labels: pd.DataFrame
    ) -> Path:

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            self.output_directory
            / self.output_filename
        )

        labels.to_csv(
            output_path,
            index=False
        )

        logger.success(
            f"Daily flood labels saved to: "
            f"{output_path}"
        )

        return output_path

    def run(self):

        logger.info(
            "Starting Daily Flood Label Generation..."
        )

        weather = self.load_weather_data()

        events = self.load_flood_events()

        events = self.select_high_confidence_events(
            events
        )

        labels = self.create_daily_event_labels(
            weather,
            events
        )

        self.calculate_label_statistics(
            labels
        )

        self.save_data(
            labels
        )

        logger.success(
            "Daily Flood Label Generation completed."
        )


if __name__ == "__main__":

    generator = FloodLabelGenerator()

    generator.run()
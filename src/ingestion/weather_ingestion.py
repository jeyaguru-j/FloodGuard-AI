from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from src.utils.logger import logger


class WeatherIngestion:
    """
    Handles historical weather data ingestion from Open-Meteo.
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the weather ingestion service.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.
        """

        self.config = self._load_config(config_path)

        self.city = self.config["city"]
        self.weather_config = self.config["weather"]

        self.latitude = self.city["latitude"]
        self.longitude = self.city["longitude"]

        self.start_date = self.weather_config["start_date"]
        self.end_date = self.weather_config["end_date"]

        self.timezone = self.weather_config["timezone"]

        self.daily_variables = self.weather_config["daily_variables"]

        self.output_directory = Path(
            self.weather_config["output_directory"]
        )

        self.filename = self.weather_config["filename"]

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """
        Load project configuration from YAML.
        """

        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}"
            )

        with config_file.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        logger.info("Configuration loaded successfully.")

        return config

    def _build_params(self) -> dict[str, Any]:
        """
        Build parameters required by the Open-Meteo API.
        """

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "daily": ",".join(self.daily_variables),
            "timezone": self.timezone,
        }

        return params

    def fetch_data(self) -> dict[str, Any]:
        """
        Fetch historical weather data from Open-Meteo.
        """

        params = self._build_params()

        logger.info(
            f"Fetching weather data for {self.city['name']} "
            f"from {self.start_date} to {self.end_date}."
        )

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            logger.info(
                f"Open-Meteo request successful. "
                f"Status code: {response.status_code}"
            )

            return response.json()

        except requests.exceptions.Timeout:
            logger.error("Open-Meteo request timed out.")
            raise

        except requests.exceptions.ConnectionError:
            logger.error(
                "Unable to connect to Open-Meteo."
            )
            raise

        except requests.exceptions.HTTPError as error:
            logger.error(
                f"Open-Meteo returned an HTTP error: {error}"
            )
            raise

        except requests.exceptions.RequestException as error:
            logger.error(
                f"Weather API request failed: {error}"
            )
            raise

    @staticmethod
    def validate_response(data: dict[str, Any]) -> None:
        """
        Validate the API response before processing it.
        """

        if not data:
            raise ValueError("Open-Meteo returned an empty response.")

        if "daily" not in data:
            raise ValueError(
                "The API response does not contain daily weather data."
            )

        daily_data = data["daily"]

        if "time" not in daily_data:
            raise ValueError(
                "The API response does not contain date information."
            )

        if len(daily_data["time"]) == 0:
            raise ValueError(
                "Open-Meteo returned zero weather records."
            )

        logger.info(
            f"API response validated successfully. "
            f"Records received: {len(daily_data['time'])}"
        )

    def convert_to_dataframe(
        self,
        data: dict[str, Any]
    ) -> pd.DataFrame:
        """
        Convert the Open-Meteo response into a DataFrame.
        """

        self.validate_response(data)

        daily_data = data["daily"]

        dataframe = pd.DataFrame(daily_data)

        dataframe.rename(
            columns={
                "time": "date",
                "precipitation_sum": "rainfall_mm",
                "temperature_2m_max": "temperature_max_c",
                "temperature_2m_min": "temperature_min_c",
                "temperature_2m_mean": "temperature_mean_c",
                "relative_humidity_2m_mean": "relative_humidity_percent",
                "wind_speed_10m_max": "wind_speed_max_kmh",
            },
            inplace=True
        )

        dataframe["date"] = pd.to_datetime(
            dataframe["date"]
        )

        dataframe["city"] = self.city["name"]

        dataframe["latitude"] = self.latitude

        dataframe["longitude"] = self.longitude

        logger.info(
            f"Weather data converted to DataFrame. "
            f"Shape: {dataframe.shape}"
        )

        return dataframe

    def save_data(self, dataframe: pd.DataFrame) -> Path:
        """
        Save raw weather data to the configured directory.
        """

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = self.output_directory / self.filename

        dataframe.to_csv(
            output_path,
            index=False
        )

        logger.success(
            f"Weather data saved successfully: {output_path}"
        )

        return output_path

    def run(self) -> Path:
        """
        Execute the complete weather ingestion workflow.
        """

        logger.info("Starting Weather Data Ingestion...")

        data = self.fetch_data()

        dataframe = self.convert_to_dataframe(data)

        output_path = self.save_data(dataframe)

        logger.success(
            "Weather Data Ingestion completed successfully."
        )

        return output_path


if __name__ == "__main__":
    ingestion = WeatherIngestion()

    ingestion.run()
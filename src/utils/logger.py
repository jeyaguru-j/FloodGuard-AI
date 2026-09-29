from loguru import logger
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.add(
    LOG_DIR / "floodguard.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Settings for data
    DATA_PROVIDER: str = "yfinance"
    STORAGE_BACKEND: str = "parquet"

    # Market settings
    R: float = 0.045    # Risk-free rate - May be made more advanced later to adjust to market data
    Q: float = 0.012

    # LOB collection interval (seconds)
    LOB_INTERVAL_SECONDS: int = 60

# Create a single, global instance of the settings
# Should be imported when using this config file
settings = Settings()
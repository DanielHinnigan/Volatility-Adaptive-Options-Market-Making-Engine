from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATA_PROVIDER: str = "yfinance"


# Create a single, global instance of the settings
# Should be imported when using this config file
settings = Settings()
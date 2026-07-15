# Functionality for choosing the data provider.

from .base_connector import DataConnector
from ..config import settings


class ConnectorFactory:
    @staticmethod
    def get_connector(provider: str = None, symbol:str = "SPY") -> DataConnector:
        # Determine how to store the data
        if settings.STORAGE_BACKEND == "parquet":
            from ..storage.parquet_storage import ParquetStorage
            storage = ParquetStorage()
        else:
            raise ValueError(f"settings.STORAGE_BACKEND is not known: {settings.STORAGE_BACKEND}")

        # Determine which data provider to use
        if provider is None:
            provider = settings.DATA_PROVIDER

        if provider == "yfinance":
            from .yfinance_connector import YFinanceConnector
            return YFinanceConnector(symbol, storage=storage)
        else:
            raise ValueError(f"Unknown provider: {provider}")
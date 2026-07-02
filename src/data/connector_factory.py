# Functionality for choosing the data provider.

from .base_connector import DataConnector

class ConnectorFactory:
    @staticmethod
    def get_connector(provider: str, **kwargs) -> DataConnector:
        if provider == "yfinance":
            from .yfinance_connector import YFinanceConnector
            return YFinanceConnector()
        else:
            raise ValueError(f"Unknown provider: {provider}")
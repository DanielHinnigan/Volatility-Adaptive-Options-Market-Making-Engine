# Base connector that any data connector must follow (yfinance, alpaca, etc.).
# Provides abstraction such that any other module can be agnostic w.r.t the connector used.

from abc import ABC, abstractmethod
from typing import Dict, Literal
from dataclasses import dataclass

@dataclass
class OptionQuote:
    """Standardized format for option quotes."""
    strike: float
    bid: float
    ask: float
    mid: float
    implied_vol: float
    volume:int
    open_interest:int
    option_type: Literal['call', 'put']

class DataConnector(ABC):
    @abstractmethod
    def get_surface_data(self, symbol: str, max_expiries: int) -> Dict:
        """Returns standardized surface data for SVI/SSVI calibration."""
        pass
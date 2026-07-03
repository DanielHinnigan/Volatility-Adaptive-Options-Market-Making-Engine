from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ..data.base_connector import OptionQuote

class StorageBackend(ABC):
    """Abstract interface for storing and retrieving option data."""
    
    @abstractmethod
    def save_expiry(self, symbol: str, expiry: str, calls: List[OptionQuote], puts: List[OptionQuote]) -> None:
        pass
    
    @abstractmethod
    def load_expiry(self, symbol: str, expiry: str) -> Optional[Dict[str, List[OptionQuote]]]:
        pass
    
    @abstractmethod
    def expiry_exists(self, symbol: str, expiry: str) -> bool:
        pass
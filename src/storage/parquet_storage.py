import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from .base import StorageBackend
from ..data.base_connector import OptionQuote

class ParquetStorage(StorageBackend):
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            # Set the cache_dir to the root node of the project (storage->src->root)
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent

            self.cache_dir = project_root / "data" / "cache"
        else:
            self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_path(self, symbol: str, expiry: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol}_{expiry}.parquet")
    
    def save_expiry(self, symbol: str, expiry: str, calls: List[OptionQuote], puts: List[OptionQuote]) -> None:
        records = []
        for q in calls + puts:
            records.append({
                'symbol': symbol,
                'expiry': expiry,
                'strike': q.strike,
                'bid': q.bid,
                'ask': q.ask,
                'mid': q.mid,
                'implied_vol': q.implied_vol,
                'volume': q.volume,
                'open_interest': q.open_interest,
                'type': q.option_type,
                'cached_at': datetime.now().isoformat()
            })
        df = pd.DataFrame(records)
        df.to_parquet(self._get_path(symbol, expiry), index=False)
    
    def load_expiry(self, symbol: str, expiry: str) -> Optional[Dict[str, List[OptionQuote]]]:
        path = self._get_path(symbol, expiry)
        if not os.path.exists(path):
            return None
        
        df = pd.read_parquet(path)
        calls, puts = [], []
        for _, row in df.iterrows():
            q = OptionQuote(
                strike=float(row['strike']),
                bid=float(row['bid']),
                ask=float(row['ask']),
                mid=float(row['mid']),
                implied_vol=float(row['implied_vol']),
                volume=int(row['volume']),
                open_interest=int(row['open_interest']),
                option_type=str(row['type'])
            )
            if q.option_type == 'call':
                calls.append(q)
            else:
                puts.append(q)
        return {"calls": calls, "puts": puts}
    
    def expiry_exists(self, symbol: str, expiry: str) -> bool:
        return os.path.exists(self._get_path(symbol, expiry))
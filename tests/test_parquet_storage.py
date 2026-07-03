# Unit tests for parquet storage
import pytest
from pathlib import Path

from src.data.base_connector import OptionQuote
from src.storage.parquet_storage import ParquetStorage

# Sample inputs
@pytest.fixture
def sample_calls():
    """Returns a list of sample OptionQuote objects (Calls)."""
    return [
        OptionQuote(
            strike=100.0,
            bid=9.50,
            ask=10.50,
            mid=10.00,
            implied_vol=0.20,
            volume=100,
            open_interest=200,
            option_type="call"
        ),
        OptionQuote(
            strike=105.0,
            bid=5.50,
            ask=6.50,
            mid=6.00,
            implied_vol=0.22,
            volume=50,
            open_interest=150,
            option_type="call"
        ),
    ]


@pytest.fixture
def sample_puts():
    """Returns a list of sample OptionQuote objects (Puts)."""
    return [
        OptionQuote(
            strike=95.0,
            bid=2.50,
            ask=3.50,
            mid=3.00,
            implied_vol=0.25,
            volume=30,
            open_interest=80,
            option_type="put"
        ),
    ]


# Unit tests
class TestParquetStorage:
    """Unit tests for ParquetStorage."""

    def test_roundtrip_save_load(self, tmp_path, sample_calls, sample_puts):
        """
        Test that saving data to Parquet and loading it back returns the exact same objects.
        """
        # Arrange: Create storage pointing to a temporary directory
        storage = ParquetStorage(cache_dir=str(tmp_path))
        symbol = "TEST"
        expiry = "2025-12-19"

        # Act: Save the data
        storage.save_expiry(symbol, expiry, sample_calls, sample_puts)

        # Assert: File was created
        expected_file = tmp_path / f"{symbol}_{expiry}.parquet"
        assert expected_file.exists(), "Parquet file was not created."

        # Act: Load the data
        loaded = storage.load_expiry(symbol, expiry)

        # Assert: Data was loaded successfully
        assert loaded is not None, "load_expiry returned None."
        assert "calls" in loaded
        assert "puts" in loaded

        # Assert: Number of calls/puts matches
        assert len(loaded["calls"]) == len(sample_calls)
        assert len(loaded["puts"]) == len(sample_puts)

        # Assert: Detailed field comparison for the first call
        original = sample_calls[0]
        loaded_call = loaded["calls"][0]

        assert original.strike == loaded_call.strike
        assert original.bid == loaded_call.bid
        assert original.ask == loaded_call.ask
        assert original.mid == loaded_call.mid
        assert original.implied_vol == loaded_call.implied_vol
        assert original.volume == loaded_call.volume
        assert original.open_interest == loaded_call.open_interest
        assert original.option_type == loaded_call.option_type

        # Assert: Put comparison (first put)
        original_put = sample_puts[0]
        loaded_put = loaded["puts"][0]
        assert original_put.strike == loaded_put.strike
        assert original_put.option_type == loaded_put.option_type

    def test_expiry_exists(self, tmp_path, sample_calls):
        """Test that expiry_exists returns True for saved data, False otherwise."""
        storage = ParquetStorage(cache_dir=str(tmp_path))
        symbol = "SPY"
        expiry = "2026-01-16"

        # Initially, should not exist
        assert storage.expiry_exists(symbol, expiry) is False

        # Save the data
        storage.save_expiry(symbol, expiry, sample_calls, [])

        # Now, should exist
        assert storage.expiry_exists(symbol, expiry) is True

    def test_load_missing_expiry(self, tmp_path):
        """Test that loading a non-existent expiry returns None gracefully."""
        storage = ParquetStorage(cache_dir=str(tmp_path))
        result = storage.load_expiry("AAPL", "2099-12-31")
        assert result is None

    def test_save_empty_lists(self, tmp_path):
        """Test that saving empty lists creates an empty Parquet file."""
        storage = ParquetStorage(cache_dir=str(tmp_path))
        symbol = "EMPTY"
        expiry = "2025-01-01"

        # Save empty lists
        storage.save_expiry(symbol, expiry, [], [])

        # File should still exist (it contains just the schema/metadata)
        expected_file = tmp_path / f"{symbol}_{expiry}.parquet"
        assert expected_file.exists()

        # Loading it should return empty lists
        loaded = storage.load_expiry(symbol, expiry)
        assert loaded is not None
        assert len(loaded["calls"]) == 0
        assert len(loaded["puts"]) == 0
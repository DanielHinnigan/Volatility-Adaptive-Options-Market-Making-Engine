"""
Unit tests for the Backtester engine.

Uses synthetic data and mocks to test the backtester's core logic
without running a full day's worth of data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.backtest.engine import Backtester, Fill
from src.data.historical_connector import HistoricalConnector
from src.data.option_spec import OptionSpec
from src.data.lob_snapshot import LOBQuote
from src.quoting.lucic_tse import Quote
from src.risk.risk_manager import RiskStatus


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_option_specs():
    """Sample OptionSpec objects for testing."""
    return [
        OptionSpec(strike=100.0, expiry="2026-07-13", T=0.02, option_type="call", id="C_100"),
        OptionSpec(strike=105.0, expiry="2026-07-13", T=0.02, option_type="call", id="C_105"),
    ]


@pytest.fixture
def sample_lob_data():
    """Generate synthetic LOB data for a single day."""
    base = datetime(2026, 7, 13, 9, 30, 0)
    timestamps = [base + timedelta(minutes=i) for i in range(10)]
    
    records = []
    for ts in timestamps:
        spot = 100.0 + np.random.randn() * 0.5
        for expiry in ["2026-07-13", "2026-07-20"]:
            for strike in [95, 100, 105]:
                for opt_type in ["call", "put"]:
                    iv = 0.2 + 0.1 * ((strike / 100.0) - 1) ** 2
                    records.append({
                        "timestamp": ts,
                        "spot_price": spot,
                        "expiry": expiry,
                        "strike": float(strike),
                        "type": opt_type,
                        "bid": max(0.01, 10.0 * (1 - 0.02 * np.random.rand())),
                        "ask": 10.0 * (1 + 0.02 * np.random.rand()),
                        "bid_size": int(np.random.randint(1, 100)),
                        "ask_size": int(np.random.randint(1, 100)),
                        "volume": int(np.random.randint(0, 1000)),
                        "open_interest": int(np.random.randint(100, 10000)),
                        "implied_vol": iv,
                        "last_price": 10.0,
                    })
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


@pytest.fixture
def mock_pricing_engine():
    """Mock PricingEngine."""
    engine = MagicMock()
    engine.get_price.return_value = 10.0
    engine.get_iv.return_value = 0.2
    engine.get_spot.return_value = 100.0
    engine.r = 0.05
    engine.q = 0.01
    engine._connector = None
    return engine


@pytest.fixture
def mock_quoting_engine():
    """Mock QuotingEngine."""
    engine = MagicMock()
    engine.generate_quotes.return_value = {
        "C_100": Quote(
            option_id="C_100",
            strike=100.0,
            expiry="2026-07-13",
            T=0.02,
            option_type="call",
            bid=9.80,
            ask=10.20,
            bid_size=5,
            ask_size=5,
            fair_value=10.0,
            spread=0.4,
        ),
        "C_105": Quote(
            option_id="C_105",
            strike=105.0,
            expiry="2026-07-13",
            T=0.02,
            option_type="call",
            bid=9.50,
            ask=9.90,
            bid_size=5,
            ask_size=5,
            fair_value=9.70,
            spread=0.4,
        ),
    }
    engine.aggregate_inventory.return_value = MagicMock(
        delta=0.0,
        gamma=0.0,
        vega=0.0,
        theta=0.0,
        vega_by_tenor={},
    )
    return engine


@pytest.fixture
def mock_risk_manager(mock_quoting_engine):
    """Mock RiskManager."""
    manager = MagicMock()
    manager.get_quotes.return_value = RiskStatus(
        halted=False,
        reason=None,
        quotes=mock_quoting_engine.generate_quotes(),
    )
    return manager


# ============================================================================
# Test Cases
# ============================================================================

class TestBacktesterInit:
    """Test initialization and validation."""

    def test_init_with_valid_data(self, sample_lob_data, sample_option_specs,
                                   mock_pricing_engine, mock_quoting_engine,
                                   mock_risk_manager, tmp_path):
        """Test that backtester initialises correctly with valid data."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
            initial_capital=100000.0,
        )

        assert backtester.option_specs == sample_option_specs
        assert backtester.initial_capital == 100000.0
        assert len(backtester.data) > 0

    def test_init_with_directory_raises_error(self, sample_lob_data,
                                               mock_pricing_engine,
                                               mock_quoting_engine,
                                               mock_risk_manager,
                                               sample_option_specs, tmp_path):
        """Test that passing a directory raises ValueError."""
        with pytest.raises(ValueError, match="must be a single Parquet file"):
            Backtester(
                pricing_engine=mock_pricing_engine,
                quoting_engine=mock_quoting_engine,
                risk_manager=mock_risk_manager,
                option_specs=sample_option_specs,
                data_path=tmp_path,
            )


class TestFillSimulation:
    """Test the fill simulation logic."""

    def test_fill_when_improving_bid(self, sample_option_specs, sample_lob_data,
                                     mock_pricing_engine, mock_quoting_engine,
                                     mock_risk_manager, tmp_path):
        """Test that improving the bid gives high fill probability."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
        )

        # Create a quote that improves the market bid
        quote = Quote(
            option_id="C_100",
            strike=100.0,
            expiry="2026-07-13",
            T=0.02,
            option_type="call",
            bid=10.50,  # Higher than market bid
            ask=11.00,
            bid_size=5,
            ask_size=5,
            fair_value=10.0,
            spread=0.4,
        )

        lob_quote = LOBQuote(
            bid=10.00,
            ask=10.50,
            bid_size=10,
            ask_size=10,
            volume=100,
            open_interest=1000,
            implied_vol=0.2,
            last_price=10.0,
        )

        # Run the simulation many times to check probability
        bid_filled_count = 0
        for _ in range(100):
            bid_filled, _ = backtester._simulate_fill(quote, lob_quote)
            if bid_filled:
                bid_filled_count += 1

        # Should fill > 80% of the time
        assert bid_filled_count > 80

    def test_fill_when_worse_than_market(self, sample_option_specs, sample_lob_data,
                                         mock_pricing_engine, mock_quoting_engine,
                                         mock_risk_manager, tmp_path):
        """Test that a worse quote never fills."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
        )

        quote = Quote(
            option_id="C_100",
            strike=100.0,
            expiry="2026-07-13",
            T=0.02,
            option_type="call",
            bid=9.50,  # Lower than market bid
            ask=10.80,  # Higher than market ask
            bid_size=5,
            ask_size=5,
            fair_value=10.0,
            spread=0.4,
        )

        lob_quote = LOBQuote(
            bid=10.00,
            ask=10.50,
            bid_size=10,
            ask_size=10,
            volume=100,
            open_interest=1000,
            implied_vol=0.2,
            last_price=10.0,
        )

        for _ in range(100):
            bid_filled, ask_filled = backtester._simulate_fill(quote, lob_quote)
            assert bid_filled is False
            assert ask_filled is False

    def test_fill_when_market_bid_zero(self, sample_option_specs, sample_lob_data,
                                       mock_pricing_engine, mock_quoting_engine,
                                       mock_risk_manager, tmp_path):
        """Test that filling when market bid is zero (only liquidity provider)."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
        )

        quote = Quote(
            option_id="C_100",
            strike=100.0,
            expiry="2026-07-13",
            T=0.02,
            option_type="call",
            bid=10.00,
            ask=10.50,
            bid_size=5,
            ask_size=5,
            fair_value=10.0,
            spread=0.4,
        )

        lob_quote = LOBQuote(
            bid=0.0,  # No other bids
            ask=10.50,
            bid_size=0,
            ask_size=10,
            volume=100,
            open_interest=1000,
            implied_vol=0.2,
            last_price=10.0,
        )

        bid_filled_count = 0
        for _ in range(100):
            bid_filled, _ = backtester._simulate_fill(quote, lob_quote)
            if bid_filled:
                bid_filled_count += 1

        # Should fill > 80% of the time (we are the only bid)
        assert bid_filled_count > 80


class TestInventoryAndPnL:
    """Test inventory and PnL tracking."""

    def test_buy_updates_inventory_and_cash(self, sample_option_specs, sample_lob_data,
                                            mock_pricing_engine, mock_quoting_engine,
                                            mock_risk_manager, tmp_path):
        """Test that a BUY fill updates inventory (+), cash (-), and PnL."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        transaction_cost_per_contract = 0.5
        quantity = 5
        price = 10.0

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
            transaction_cost_per_contract=transaction_cost_per_contract
        )

        fill = Fill(
            timestamp=datetime.now(),
            option_id="C_100",
            strike=100.0,
            expiry="2026-07-13",
            option_type="call",
            side="BUY",
            price=price,
            quantity=quantity,
            pnl=0.0,
        )

        backtester._update_inventory_and_pnl(fill)

        assert backtester._positions["C_100"] == quantity
        assert backtester._cash == -quantity*price-transaction_cost_per_contract*5  # 5 * $10, minus transaction cost
        assert backtester._daily_pnl == -quantity*price-transaction_cost_per_contract*5

    def test_sell_updates_inventory_and_cash(self, sample_option_specs, sample_lob_data,
                                             mock_pricing_engine, mock_quoting_engine,
                                             mock_risk_manager, tmp_path):
        """Test that a SELL fill updates inventory (-), cash (+), and PnL."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        transaction_cost_per_contract = 0.5
        quantity = 5
        price = 10.0

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
            transaction_cost_per_contract=transaction_cost_per_contract
        )

        fill = Fill(
            timestamp=datetime.now(),
            option_id="C_100",
            strike=100.0,
            expiry="2026-07-13",
            option_type="call",
            side="SELL",
            price=price,
            quantity=quantity,
            pnl=0.0,
        )

        backtester._update_inventory_and_pnl(fill)

        assert backtester._positions["C_100"] == -quantity
        assert backtester._cash == price*quantity-transaction_cost_per_contract*quantity  # 5 * $10, minus transaction cost
        assert backtester._daily_pnl == price*quantity-transaction_cost_per_contract*quantity

    def test_mark_to_market(self, sample_option_specs, sample_lob_data,
                            mock_pricing_engine, mock_quoting_engine,
                            mock_risk_manager, tmp_path):
        """Test mark-to-market PnL calculation."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
            initial_capital=100000.0,
        )

        # Set up positions
        backtester._positions["C_100"] = 5  # Long 5 calls
        backtester._cash = 95000.0  # Spent $5,000 on 5 calls at $10 each

        fair_values = {"C_100": 12.0}  # Current fair value is $12

        total_pnl = backtester._mark_to_market(fair_values)

        # PnL = Cash + (quantity * fair) - initial_capital
        # = 95000 + (5 * 12) - 100000 = -5000 + 60 = -4940
        assert total_pnl == -4940.0

    def test_daily_pnl_accumulates(self, sample_option_specs, sample_lob_data,
                                   mock_pricing_engine, mock_quoting_engine,
                                   mock_risk_manager, tmp_path):
        """Test that daily PnL accumulates correctly across multiple fills."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        trans_cost = 0.5

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
            transaction_cost_per_contract=trans_cost
        )

        # Buy 5 at $10
        fill1 = Fill(timestamp=datetime.now(), option_id="C_100", strike=100.0,
                     expiry="2026-07-13", option_type="call", side="BUY",
                     price=10.0, quantity=5, pnl=0.0)
        backtester._update_inventory_and_pnl(fill1)
        assert backtester._daily_pnl == -50.0-5*trans_cost

        # Sell 3 at $12
        fill2 = Fill(timestamp=datetime.now(), option_id="C_100", strike=100.0,
                     expiry="2026-07-13", option_type="call", side="SELL",
                     price=12.0, quantity=3, pnl=0.0)
        backtester._update_inventory_and_pnl(fill2)
        assert backtester._daily_pnl == -50.0 + 36.0-(5+3)*trans_cost  # -14.0
        assert backtester._positions["C_100"] == 2


class TestRiskUnwind:
    """Test risk management and unwinding logic."""

    def test_delta_unwind(self, sample_option_specs, sample_lob_data,
                          mock_pricing_engine, mock_quoting_engine,
                          mock_risk_manager, tmp_path):
        """Test that Delta excess is unwound correctly."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        mock_pricing_engine.get_spot.return_value = 100.0

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
        )

        # Create a RiskStatus with Delta excess
        risk_status = RiskStatus(
            halted=True,
            reason="Delta",
            excess_delta=1000.0,  # Too long Delta
        )

        # Need a snapshot for the unwinding
        historical_connector = HistoricalConnector(sample_lob_data)
        snapshot = historical_connector.get_current_snapshot()

        backtester._unwind_excess_risk(risk_status, datetime.now(), snapshot)

        # Cash should increase (we sold SPY to reduce Delta)
        # 1000 excess_delta * $100 spot = $100,000
        assert backtester._cash > 0
        assert backtester._risk_breaches == 1

    def test_fire_sale_on_gamma_breach(self, sample_option_specs, sample_lob_data,
                                       mock_pricing_engine, mock_quoting_engine,
                                       mock_risk_manager, tmp_path):
        """Test that Gamma breach triggers a fire sale."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
        )

        # Set up some inventory
        backtester._positions["C_100"] = 10

        risk_status = RiskStatus(
            halted=True,
            reason="Gamma",
            excess_gamma=50.0,
        )

        historical_connector = HistoricalConnector(sample_lob_data)
        snapshot = historical_connector.get_current_snapshot()

        backtester._unwind_excess_risk(risk_status, datetime.now(), snapshot)

        # Positions should be flattened
        assert backtester._positions["C_100"] == 0
        assert backtester._risk_breaches == 1


class TestBacktesterRun:
    """Test the full backtester run."""

    def test_run_returns_metrics(self, sample_option_specs, sample_lob_data,
                                 mock_pricing_engine, mock_quoting_engine,
                                 mock_risk_manager, tmp_path):
        """Test that the backtester run() returns metrics."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
            initial_capital=100000.0,
        )

        results = backtester.run()

        # Check that all expected metrics are present
        expected_keys = {'total_pnl', 'sharpe_ratio', 'max_drawdown',
                         'win_rate', 'total_trades', 'risk_breaches',
                         'drawdown_breaches', 'equity_curve'}
        assert all(key in results for key in expected_keys)

        # Check that equity curve is a DataFrame
        assert isinstance(results['equity_curve'], pd.DataFrame)

    def test_run_with_risk_breach(self, sample_option_specs, sample_lob_data,
                                  mock_pricing_engine, mock_quoting_engine,
                                  mock_risk_manager, tmp_path):
        """Test that risk breaches are tracked correctly."""
        data_path = tmp_path / "test_data.parquet"
        sample_lob_data.to_parquet(data_path)

        # Make RiskManager return a halted status
        mock_risk_manager.get_quotes.return_value = RiskStatus(
            halted=True,
            reason="Delta limit breached",
            excess_delta=1000.0,
        )

        backtester = Backtester(
            pricing_engine=mock_pricing_engine,
            quoting_engine=mock_quoting_engine,
            risk_manager=mock_risk_manager,
            option_specs=sample_option_specs,
            data_path=data_path,
            initial_capital=100000.0,
        )

        results = backtester.run()

        # Should have recorded at least one risk breach
        assert results['risk_breaches'] > 0
        assert results['drawdown_breaches'] == 0
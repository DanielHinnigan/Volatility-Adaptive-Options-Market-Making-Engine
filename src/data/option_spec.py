"""
Definition of an Option Specification (OptionSpec) data container.

This class provides both attribute-style (spec.T) and dictionary-style (spec["T"]) access
to its fields, ensuring backward compatibility while enforcing a standard interface.

The class is frozen (immutable) and validates all fields on creation.
"""

from dataclasses import dataclass
from typing import Optional, Any


@dataclass(frozen=True)
class OptionSpec:
    """
    Immutable specification for a single option contract.

    Attributes:
        id: Unique identifier (e.g., 'SPY_750_2026-07-13').
        strike: Strike price.
        expiry: Expiration date as string (YYYY-MM-DD).
        T: Time to expiry in years.
        option_type: 'call' or 'put'.
        spot: Optional spot price at creation time (for reference).

    Supports both attribute and key access.
    """
    strike: float
    expiry: str
    T: float
    option_type: str
    id: Optional[str] = None
    spot: Optional[float] = None

    # Map of field names to their expected types (for validation)
    _FIELD_TYPES = {
        'strike': float,
        'expiry': str,
        'T': float,
        'option_type': str,
        'id': str,
        'spot': float,
    }

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive: {self.strike}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry must be positive: {self.T}")
        if self.option_type not in ('call', 'put'):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type}")

        if self.id is None:
            object.__setattr__(self, 'id', f"SPY_{int(self.strike)}_{self.expiry}")

    # ------------------------------------------------------------------------
    # Dictionary-Like Access
    # ------------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access: spec["strike"] -> 750.0

        Supports all fields: strike, expiry, T, option_type, id, spot.
        """
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"OptionSpec has no field: {key}")

    def __contains__(self, key: str) -> bool:
        """Allow 'strike' in spec checks."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Safe dictionary-style access with default value."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Return an iterable of field names."""
        return self._FIELD_TYPES.keys()

    def values(self):
        """Return an iterable of field values."""
        return [self[k] for k in self.keys()]

    def items(self):
        """Return an iterable of (key, value) pairs."""
        return [(k, self[k]) for k in self.keys()]

    def to_dict(self) -> dict:
        """Convert to a standard dictionary."""
        return {k: self[k] for k in self.keys()}

    # ------------------------------------------------------------------------
    # Factory Methods
    # ------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> 'OptionSpec':
        """Create an OptionSpec from a dictionary."""
        return cls(
            strike=data['strike'],
            expiry=data['expiry'],
            T=data['T'],
            option_type=data['option_type'],
            id=data.get('id'),
            spot=data.get('spot'),
        )
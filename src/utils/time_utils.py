import datetime
from zoneinfo import ZoneInfo

def compute_time_to_expiry(expiry_str: str, current_time=None, raise_on_expired: bool = True) -> float:
    """
    Compute time-to-expiry in YEARS for a US equity option.
    
    Args:
        expiry_str: Expiry date as 'YYYY-MM-DD'
        current_time: Optional datetime (naive or timezone-aware). 
                      Defaults to now in US/Eastern.
        raise_on_expired: If True, raises ValueError for expired options.
                          If False, returns NaN.
    
    Returns:
        float: Time to expiry in years (ACT/365 convention).
    """
    # 1. Set timezone to US/Eastern (where SPY trades)
    eastern = ZoneInfo("US/Eastern")
    
    # 2. Parse expiry date and set expiration time to 4:00 PM ET (market close)
    expiry_date = datetime.datetime.strptime(expiry_str, "%Y-%m-%d").date()
    expiry_dt = datetime.datetime.combine(
        expiry_date, 
        datetime.time(16, 0, 0)  # 4:00 PM ET
    ).replace(tzinfo=eastern)
    
    # 3. Get current time in US/Eastern
    if current_time is None:
        current_time = datetime.datetime.now(eastern)
    elif current_time.tzinfo is None:
        # If naive, assume it's in ET (or convert)
        current_time = current_time.replace(tzinfo=eastern)
    
    # 4. Calculate difference in seconds
    diff_seconds = (expiry_dt - current_time).total_seconds()
    
    # 5. Convert to years (ACT/365 convention)
    # Check if the option has expired
    if diff_seconds <= 0:
            if raise_on_expired:
                raise ValueError(
                    f"Option expired on {expiry_str} at 4:00 PM ET. "
                    f"Current time: {current_time.isoformat()}. "
                    "Filter out expired expiries before calling this function."
                )
            else:
                return float('nan')  # Let the caller handle it
    
    return diff_seconds / (365 * 24 * 3600)  # ACT/365
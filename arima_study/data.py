"""Data: simulator matching the report's stylized facts, plus a CSV loader."""
import numpy as np
import pandas as pd


def simulate_prices(n=1500, seed=42, start="2019-01-01", p0=100.0, mu=0.0004,
                    phi=0.08, omega=4.4e-6, alpha=0.10, beta=0.85):
    """Daily price with drift mu, AR(1) returns (phi) and GARCH(1,1) errors."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    sigma2 = omega / (1 - alpha - beta)
    r = np.zeros(n)
    for t in range(n):
        eps = np.sqrt(sigma2) * z[t]
        prev = r[t - 1] - mu if t else 0.0
        r[t] = mu + phi * prev + eps
        sigma2 = omega + alpha * eps ** 2 + beta * sigma2
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(p0 * np.exp(np.cumsum(r)), index=idx, name="price")


def load_csv(path_or_buf, date_col=None, price_col=None):
    """Load real prices. Defaults: first column = date, 'Close' (or last numeric) = price."""
    df = pd.read_csv(path_or_buf)
    date_col = date_col or df.columns[0]
    if price_col is None:
        price_col = "Close" if "Close" in df.columns else df.select_dtypes("number").columns[-1]
    s = pd.Series(df[price_col].values, index=pd.to_datetime(df[date_col]), name="price")
    s = s.sort_index().dropna()
    if (s <= 0).any():
        raise ValueError("Prices must be positive (log transform).")
    return s.asfreq("B").ffill()

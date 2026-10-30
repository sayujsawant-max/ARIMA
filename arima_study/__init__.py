"""Box-Jenkins ARIMA case study (simulated daily asset price)."""
from .data import simulate_prices, load_csv
from .model import (
    stationarity_report, select_order, fit_arima, diagnostics,
    walk_forward, forecast_interval, run_case_study,
)

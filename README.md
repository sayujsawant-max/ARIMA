# ARIMA for Financial Time-Series Forecasting

Working implementation of the CIAP report *ARIMA for Financial Time-Series Forecasting: A Conceptual Case Study*
(SAKEC, AI & Data Science, Module 3): the full Box-Jenkins cycle on a simulated daily asset price.

**Live app:** see the link at the top of the GitHub repo description (Streamlit Community Cloud).

## What it does
1. **Stationarity**: ADF + KPSS on log price and log return, picks d = 1.
2. **Identification**: ACF / PACF plots, grid search over p, q in 0..N.
3. **Estimation**: AIC/BIC ranking, parameter table, optional manual order.
4. **Diagnostics**: Ljung-Box on residuals and squared residuals, Jarque-Bera, kurtosis, ARCH flag.
5. **Forecast & evaluation**: walk-forward one-step forecasts vs. naive and drift benchmarks (RMSE, MAE,
   directional accuracy, Diebold-Mariano), plus a prediction-interval fan.

Data: simulated series (drift 0.04%/day, AR(1) phi = 0.08, GARCH(1,1) errors, seed 42), or upload your own CSV
(date column + `Close`/price column) to repeat the study on real tickers.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
pytest
```

## Notes
The report's exact figures depend on its unpublished random draws, so numbers here differ slightly; the
qualitative findings reproduce: d = 1, ARIMA(0,1,1) selected, mean diagnostics pass, squared-residual
test fails (volatility clustering), and ARIMA does not beat the naive forecast out of sample.
Educational only; not investment advice.

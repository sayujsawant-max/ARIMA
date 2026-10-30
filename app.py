"""Streamlit app: interactive Box-Jenkins ARIMA case study."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf

from arima_study import (simulate_prices, load_csv, stationarity_report, select_order,
                         fit_arima, diagnostics, walk_forward, forecast_interval)

st.set_page_config(page_title="ARIMA Case Study", layout="wide")
st.title("ARIMA for Financial Time-Series Forecasting")
st.caption("Interactive Box-Jenkins case study. Simulated data by default; "
           "educational only, not investment advice.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    src = st.radio("Source", ["Simulated (report setup)", "Upload CSV"])
    if src.startswith("Sim"):
        n = st.slider("Observations", 600, 3000, 1500, 100)
        seed = st.number_input("Seed", 0, 10_000, 42)
        phi = st.slider("AR(1) in returns (phi)", 0.0, 0.3, 0.08, 0.01)
        mu = st.slider("Daily drift (%)", 0.0, 0.15, 0.04, 0.01) / 100
        price = simulate_prices(n, int(seed), phi=phi, mu=mu)
    else:
        demo_path = Path(__file__).parent / "data" / "demo_prices.csv"
        st.download_button("Download demo CSV", demo_path.read_bytes(), "demo_prices.csv", "text/csv")
        use_demo = st.checkbox("Use the demo CSV (1,500 simulated days, Date + Close)", value=True)
        up = demo_path if use_demo else st.file_uploader(
            "CSV with a date column and a Close/price column", type="csv")
        if up is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        try:
            price = load_csv(up)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.stop()
    st.header("Model")
    n_test = st.slider("Hold-out days", 50, min(500, len(price) // 3), min(250, len(price) // 4), 10)
    max_pq = st.slider("Max p and q in grid", 1, 4, 3)
    manual = st.checkbox("Choose order manually")
    if manual:
        mp, mq = st.number_input("p", 0, 5, 0), st.number_input("q", 0, 5, 1)

lp = np.log(price)
train = lp.iloc[:-n_test]


@st.cache_data(show_spinner="Searching model orders...")
def _grid(series, mx):
    return select_order(series, 1, mx, mx)


cands = _grid(train, max_pq)
order = (int(mp), 1, int(mq)) if manual else (int(cands.iloc[0].p), 1, int(cands.iloc[0].q))
res = fit_arima(train, order)
diag = diagnostics(res)
fc, metrics, (dm_stat, dm_p) = walk_forward(lp, order, n_test)

t1, t2, t3, t4, t5 = st.tabs(["1 Data & stationarity", "2 Identification", "3 Estimation",
                              "4 Diagnostics", "5 Forecast & evaluation"])

with t1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Price", "Log return"))
    fig.add_scatter(x=price.index, y=price, name="Price", row=1, col=1)
    fig.add_scatter(x=price.index[1:], y=lp.diff().dropna(), name="Log return", row=2, col=1)
    fig.update_layout(height=520, showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.subheader("ADF / KPSS (training sample)")
    st.dataframe(stationarity_report(train).style.format(
        {"ADF statistic": "{:.3f}", "ADF p-value": "{:.4f}", "KPSS p-value": "{:.3f}"}),
        hide_index=True, width='stretch')
    st.caption("ADF null: unit root. KPSS null: stationary. KPSS p-values are truncated "
               "to [0.01, 0.10] by the test tables. Both agree on d = 1.")

with t2:
    r = train.diff().dropna()
    lags = 30
    ci = 1.96 / np.sqrt(len(r))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF of log returns", "PACF of log returns"))
    for c, vals in enumerate((acf(r, nlags=lags)[1:], pacf(r, nlags=lags)[1:]), 1):
        fig.add_bar(x=list(range(1, lags + 1)), y=vals, row=1, col=c, marker_color="#2a6f97")
        for s in (ci, -ci):
            fig.add_hline(y=s, line_dash="dash", line_color="gray", row=1, col=c)
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, width='stretch')
    st.write("Little structure and no clean cut-off, so a grid of (p, q) is searched and "
             "information criteria arbitrate.")

with t3:
    top = cands.head(8).copy()
    top.insert(0, "Rank", range(1, len(top) + 1))
    st.dataframe(top[["Rank", "Model", "AIC", "BIC"]].style.format({"AIC": "{:.1f}", "BIC": "{:.1f}"}),
                 hide_index=True, width='stretch')
    spread = cands.head(5).AIC.max() - cands.AIC.min()
    st.caption(f"Top five lie within {spread:.1f} AIC points (differences under ~2 are negligible), "
               "so parsimony is a fair tie-breaker.")
    st.subheader(f"Selected: ARIMA{order} with drift")
    st.dataframe(pd.DataFrame({"coef": res.params, "std err": res.bse, "p-value": res.pvalues}),
                 width='stretch')

with t4:
    e = pd.Series(np.asarray(res.resid)).iloc[1:]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ljung-Box p (lag 10)", f"{diag['ljung_box_p_lag10']:.3f}")
    c2.metric("Ljung-Box p (lag 20)", f"{diag['ljung_box_p_lag20']:.3f}")
    c3.metric("LB on squared resid. (lag 10)", f"{diag['ljung_box_sq_p_lag10']:.1e}")
    c4.metric("Kurtosis", f"{diag['kurtosis']:.2f}")
    (st.success if diag["mean_adequate"] else st.error)(
        "Mean equation " + ("adequate: no leftover linear autocorrelation." if diag["mean_adequate"]
                            else "NOT adequate: residuals still autocorrelated, revisit identification."))
    (st.warning if diag["arch_effects"] else st.success)(
        "ARCH effects detected (volatility clustering). ARIMA assumes constant variance; "
        "consider ARIMA-GARCH." if diag["arch_effects"] else "No ARCH effects detected.")
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Residuals", "ACF of residuals",
                                                        "ACF of squared residuals"))
    fig.add_scatter(y=e.values, mode="lines", row=1, col=1)
    fig.add_bar(x=list(range(1, 31)), y=acf(e, nlags=30)[1:], row=1, col=2)
    fig.add_bar(x=list(range(1, 31)), y=acf(e ** 2, nlags=30)[1:], row=1, col=3)
    fig.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig, width='stretch')

with t5:
    st.dataframe(metrics.style.format({"RMSE": "{:.5f}", "MAE": "{:.5f}",
                                       "Directional accuracy": "{:.1%}"}, na_rep="n/a"),
                 hide_index=True, width='stretch')
    rel = metrics.RMSE[0] / metrics.RMSE[1] - 1
    verdict = "beats" if rel < 0 else "does not beat"
    st.info(f"ARIMA RMSE is {rel:+.1%} versus the naive forecast, so it **{verdict}** the benchmark. "
            f"Diebold-Mariano: stat = {dm_stat:.2f}, p = {dm_p:.3f} "
            f"({'significant' if dm_p < 0.05 else 'not significant'} at 5%).")
    f1 = go.Figure()
    f1.add_scatter(x=train.index[-150:], y=np.exp(train.iloc[-150:]), name="Train", line_color="gray")
    f1.add_scatter(x=fc.index, y=np.exp(fc.actual), name="Actual")
    f1.add_scatter(x=fc.index, y=np.exp(fc.arima), name="ARIMA 1-step", line_dash="dash")
    f1.update_layout(title="One-step-ahead walk-forward", height=380)
    st.plotly_chart(f1, width='stretch')
    h = st.slider("Forecast horizon (days)", 10, 120, 60)
    hz = forecast_interval(res, h)
    f2 = go.Figure()
    f2.add_scatter(x=train.index[-150:], y=np.exp(train.iloc[-150:]), name="History", line_color="gray")
    f2.add_scatter(x=hz.index, y=np.exp(hz.upper), line_width=0, showlegend=False)
    f2.add_scatter(x=hz.index, y=np.exp(hz.lower), fill="tonexty", line_width=0,
                   name="95% interval", fillcolor="rgba(220,80,80,0.2)")
    f2.add_scatter(x=hz.index, y=np.exp(hz["mean"]), name="Forecast", line_color="crimson")
    held = lp.iloc[len(train):len(train) + h]
    f2.add_scatter(x=held.index, y=np.exp(held), name="Realised", line_color="#2a6f97")
    f2.update_layout(title=f"{h}-day forecast from end of training sample", height=400)
    st.plotly_chart(f2, width='stretch')
    inside = ((held.values >= hz.lower.values[:len(held)]) & (held.values <= hz.upper.values[:len(held)])).mean()
    st.caption(f"{inside:.0%} of realised prices fell inside the 95% interval. "
               "Intervals assume constant variance, so they are too narrow in turbulent regimes.")

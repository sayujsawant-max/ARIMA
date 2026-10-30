"""Box-Jenkins pipeline: identify, estimate, diagnose, forecast, evaluate."""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")


def stationarity_report(log_price):
    """ADF + KPSS on the level and first difference (Table 3)."""
    rows = []
    for name, s in [("Log price (level)", log_price),
                    ("Log return (1st difference)", log_price.diff().dropna())]:
        adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
        kpss_p = kpss(s, regression="c", nlags="auto")[1]
        stationary = adf_p < 0.05 and kpss_p > 0.05
        rows.append({"Series": name, "ADF statistic": adf_stat, "ADF p-value": adf_p,
                     "KPSS p-value": kpss_p,
                     "Verdict": "Stationary" if stationary else "Non-stationary"})
    return pd.DataFrame(rows)


def _spec(order):
    """Drift for d=1 is trend='t'; for d=0 a constant."""
    return dict(order=order, trend="t" if order[1] == 1 else "c")


def select_order(log_price, d=1, max_p=3, max_q=3):
    """Grid search (Table 4), sorted by AIC."""
    out = []
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        try:
            r = ARIMA(log_price, **_spec((p, d, q))).fit()
            out.append({"Model": f"ARIMA({p},{d},{q})", "p": p, "d": d, "q": q,
                        "AIC": r.aic, "BIC": r.bic})
        except Exception:
            continue
    return pd.DataFrame(out).sort_values("AIC").reset_index(drop=True)


def fit_arima(log_price, order):
    return ARIMA(log_price, **_spec(order)).fit()


def diagnostics(res):
    """Ljung-Box on residuals and squared residuals, Jarque-Bera, kurtosis."""
    e = pd.Series(np.asarray(res.resid)).iloc[1:]  # first residual is the diffuse start-up
    out = {}
    for lag in (10, 20):
        out[f"ljung_box_p_lag{lag}"] = float(acorr_ljungbox(e, [lag])["lb_pvalue"].iloc[0])
        out[f"ljung_box_sq_p_lag{lag}"] = float(acorr_ljungbox(e ** 2, [lag])["lb_pvalue"].iloc[0])
    out["jarque_bera_p"] = float(stats.jarque_bera(e).pvalue)
    out["kurtosis"] = float(stats.kurtosis(e, fisher=False))
    out["mean_adequate"] = out["ljung_box_p_lag10"] > 0.05 and out["ljung_box_p_lag20"] > 0.05
    out["arch_effects"] = out["ljung_box_sq_p_lag10"] < 0.05 or out["ljung_box_sq_p_lag20"] < 0.05
    return out


def dm_test(e1, e2):
    """Diebold-Mariano (h=1, squared error). Returns (stat, two-sided p)."""
    d = (np.asarray(e1) ** 2 - np.asarray(e2) ** 2)
    stat = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def walk_forward(log_price, order, n_test=250):
    """One-step-ahead forecasts on the hold-out, no re-estimation (Table 5)."""
    train, test = log_price.iloc[:-n_test], log_price.iloc[-n_test:]
    res = fit_arima(train, order)
    full = res.apply(log_price)  # same parameters, filtered through the whole sample
    arima = np.asarray(full.get_prediction(start=len(train)).predicted_mean)
    naive = log_price.shift(1).iloc[-n_test:]
    drift = naive + train.diff().mean()
    fc = pd.DataFrame({"actual": test, "arima": arima, "naive": naive, "drift": drift})
    names = {"arima": f"ARIMA{order} with drift", "naive": "Naive (random walk)",
             "drift": "Random walk with drift"}
    rows = []
    for key, label in names.items():
        err = fc["actual"] - fc[key]
        acc = None
        if key == "arima":
            move = np.sign(fc["actual"].diff()).iloc[1:]
            call = np.sign(fc["arima"] - fc["actual"].shift(1)).iloc[1:]
            acc = float((move == call).mean())
        rows.append({"Forecast": label, "RMSE": float(np.sqrt((err ** 2).mean())),
                     "MAE": float(err.abs().mean()), "Directional accuracy": acc})
    dm = dm_test(fc["actual"] - fc["arima"], fc["actual"] - fc["naive"])
    return fc, pd.DataFrame(rows), dm


def forecast_interval(res, steps=60, alpha=0.05):
    f = res.get_forecast(steps)
    ci = np.asarray(f.conf_int(alpha=alpha))
    return pd.DataFrame({"mean": np.asarray(f.predicted_mean),
                         "lower": ci[:, 0], "upper": ci[:, 1]},
                        index=pd.bdate_range(f.predicted_mean.index[0], periods=steps)
                        if not isinstance(f.predicted_mean.index, pd.DatetimeIndex)
                        else f.predicted_mean.index)


def run_case_study(price, n_test=250, max_p=3, max_q=3):
    """Full pipeline on a price series; returns a dict of results."""
    lp = np.log(price)
    train = lp.iloc[:-n_test]
    cands = select_order(train, 1, max_p, max_q)
    best = cands.iloc[0]
    order = (int(best.p), 1, int(best.q))
    res = fit_arima(train, order)
    fc, metrics, dm = walk_forward(lp, order, n_test)
    return dict(log_price=lp, train=train, stationarity=stationarity_report(train),
                candidates=cands, order=order, result=res, diagnostics=diagnostics(res),
                forecasts=fc, metrics=metrics, dm=dm, horizon=forecast_interval(res, 60))

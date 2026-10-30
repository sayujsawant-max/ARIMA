from arima_study import simulate_prices, run_case_study


def test_simulation_is_reproducible():
    a, b = simulate_prices(300), simulate_prices(300)
    assert a.equals(b) and len(a) == 300 and (a > 0).all()


def test_pipeline_end_to_end():
    out = run_case_study(simulate_prices(1500), max_p=2, max_q=2)
    s = out["stationarity"]
    assert s.iloc[0]["Verdict"] == "Non-stationary" and s.iloc[1]["Verdict"] == "Stationary"
    assert out["order"][1] == 1
    assert len(out["forecasts"]) == 250 and len(out["horizon"]) == 60
    assert out["metrics"]["RMSE"].between(0.005, 0.03).all()
    w = out["horizon"]["upper"] - out["horizon"]["lower"]
    assert w.iloc[-1] > w.iloc[0]  # intervals widen with horizon

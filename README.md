# Distributionally robust Kalman filtering with volatility uncertainty

* `tracking.py` implements the target tracking example. The flag variable `CAUSAL_FLAG` controls
whether the bi-causal constraint is used. With the log files generated, `RMSE.py` produces the RMSE
summary statistics and figures.

* `pairstrading.py` shows the pairs trading example. There are two flag variables, `CAUSAL` and `ROBUST`.
Set `ROBUST = False` and `CAUSAL = False` to reproduce the non-robust benchmark. `ROBUST = True`
together with `CAUSAL = False` is for the ordinary OT framework. `ROBUST = True` with `CAUSAL = True`
generates the bi-causal OT results. The stock price data attached are from Yahoo! Finance.
`Ratios.ipynb` can be used to compare Sharpe and Sortino ratios.

* `utils.py` implements the optimization algorithm with the LDL decomposition.

* `tracking.py` and `pairstrading.py` can work independently.

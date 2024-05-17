# Distributionally robust Kalman filtering with volatility uncertainty

* `tracking.py` implements the target tracking example. The flag variable `ALGO_FLAG` controls which algorithm is used:
- 'Nonrobust': No robustness
- 'KL': KL divergence
- 'OT': The classical OT framework
- 'BCOT': Our proposed bicausal OT constraint. 

With the log files generated, `RMSE.py` and `KL_RMSE.py` produce the RMSE summary statistics and figures.

* `pairstrading.py` shows the pairs trading example. The flag `ALGO_FLAG` is interpreted the same as above. 
The stock price data attached are from Yahoo! Finance. `Ratios.ipynb` can be used to compare Sharpe and Sortino ratios.

* `utils.py` implements the optimization algorithm with the LDL decomposition.

* `tracking.py` and `pairstrading.py` can work independently.

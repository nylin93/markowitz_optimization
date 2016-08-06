# Mean Variance Optimization 
This is an implementation of [Mean Variance Optimization] 
(https://en.wikipedia.org/wiki/Modern_portfolio_theory) 
and was built with much inspiration from the one found at [quantandfinancial] 
(http://www.quantandfinancial.com/2013/07/mean-variance-portfolio-optimization.html). 

This implementation consists of a simpler interface that exposes two top-level 
methods `optimize_sharpe_ratio` which optimizes the portfolio in regards to 
the Sharpe Ratio, and `optimize_return_amount` which optimizes the portfolio 
with the minimum variance for a requested return on the portfolio.

## Sample Usage
```python
stocks = ['BA', 'WFC', 'CSCO', 'GOOG', 'FL']
start = 2013
rf_return = .015

optimized = optimize_sharpe_ratio(stocks, start, rf_return)
# => {'CSCO': 50.0, 'BA': 0.0, 'GOOG': 0.0, 'WFC': 50.0, 'FL': 0.0} 
```

Which means you should invest 50% into CSCO and 50% into WFC. If you don't want to
invest that much into just two stocks change the `max_weight` parameter to a lower
value, e.g.

```python
optimized = optimize_sharpe_ratio(stocks, start, rf_return, max_weight=0.25)
# => {'CSCO': 25.0, 'BA': 25.0, 'GOOG': 0.0, 'WFC': 25.0, 'FL': 25.0}
```

## Dependencies
* scipy 0.16.0
* numpy 1.11.1
* pandas 0.18.1
* requests 2.9.1

# Mean Variance Optimization 
This is an implementation of [Mean Variance Optimization] 
(https://en.wikipedia.org/wiki/Modern_portfolio_theory) 
and was built with much inspiration from the one found at [quantandfinancial] (http://www.quantandfinancial.com/2013/07/mean-variance-portfolio-optimization.html). 

This implementation consists of a simpler interface that exposes two top-level methods `optimize_sharpe_ratio` 
which optimizes the portfolio in regards to the Sharpe Ratio, and `optimize_return_amount` which optimizes the 
portfolio with the minimum variance for a requested return on the portfolio.

## Sample Usage

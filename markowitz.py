import standardandpoor, yahoo
import numpy as np
import scipy.optimize

def optimize_sharpe(symbols, start, rf_return, max_weight=0.5):
	'''
	optimize_sharpe_ratio

	params
	------
		symbols: list of stock symbols 
		start: the starting year to gather stock returns
		rf_return: the risk free return
		max_weight: the maximum weight to be allocated to a single stock

	returns: dictionary of strings mapped to floats (stock -> weight)
	'''

	returns = _returns(symbols, start)
	mu, sigma = _mu_sigma(returns)

	initial_X = np.ones(len(mu))/len(mu)
	bounds = [(0., max_weight) for i in range(len(mu))]
	constraints = ({'type': 'eq', 'fun': lambda X: sum(X) - 1.})

	optimized_weights = map(lambda x: round(100*x), 
			scipy.optimize.minimize(_sharpe_ratio, initial_X, 
			(mu, sigma, rf_return), method='SLSQP', constraints=constraints, 
			bounds=bounds).x)

	return dict(zip(returns.keys(), optimized_weights))
	
def optimize_return(symbols, start, min_return, max_weight=0.5):
	'''
	optimize_return_amount

	params
	------
		symbols: list of stock symbols
		start: the starting year to gather stock returns
		min_return: the return that you would like the portfolio to generate
		max_weight: the maximum to be allocated to a single stock
	
	returns: dictionary of strings mapped to floats (stock -> weight)
	'''

	returns = _returns(symbols, start)
	mu, sigma = _mu_sigma(returns)
	
	initial_X = np.ones(len(mu))/len(mu)
	bounds = [(0., max_weight) for i in range(len(mu))]
	constraints = ({'type': 'eq', 'fun': lambda X: sum(X) - 1.})

	optimized_weights = map(lambda x: round(100*x), 
			scipy.optimize.minimize(_maximize_ret, initial_X, 
			(mu, sigma, min_return), method='SLSQP', constraints=constraints, 
			bounds=bounds).x)

	return dict(zip(returns.keys(), optimized_weights))

def _mu_sigma(returns):
	'''
	_mu_sigma

	Returns the expected return and the covariance matrix 

	params:
	------
		returns: dictionary mapping symbols to Pandas series

	returns: ([float], np.array of len(returns))
	'''
	return ([_annualize_monthly_returns(returns[k]) for k in returns], 
			np.cov(np.array(list(returns.values()))))

def _returns(symbols, start):
	'''
	_returns

	Fetches the monthly returns from the specified start date. Since
	some stocks have been on the market for different periods of time
	we remove stocks that have not been on the market since specified 
	`start` year

	params:
	------
		symbols: list of str
		start: int

	returns: dictionary (str -> Pandas series)
	'''
	raw_returns = {sym: ret for (sym, ret) in 
			zip(symbols, [yahoo.monthly_returns(sym, start) for sym in symbols])} 
	max_len = len(raw_returns[max(raw_returns, key=lambda x: len(x))])
	return {k: raw_returns[k] for k in raw_returns 
			if len(raw_returns[k]) == max_len}	
	

def _sharpe_ratio(X, mu, sigma, rf_return):
	'''
	_sharpe_ratio

	An objective function for the SLSQP optimization function to minimize
	Because SLSQP minimizes a convex function we take the inverse of
	the Sharpe ratio value	
	
	params
	------
		X: the vector of weights we are optimizing
		mu: the expected return of each stock
		sigma: the sample covariance matrix of each stock's returns
		rf_return: the risk free return 

	returns: float
	'''

	portfolio_return = np.dot(X, mu)
	portfolio_variance = np.dot(np.dot(X, sigma), X)
	util = (portfolio_return - rf_return)/np.sqrt(portfolio_variance)

	return 1/util

def _maximize_ret(X, mu, sigma, min_return):
	'''
	_maximize_ret

	An objective function for the SLSQP optimization function to minimize.
	We penalize the optimizer for a portfolio with high variance and 
	a return much greater or less than the minimum return 

	params
	------
		X: the vector of weights
		mu: the expected return of each stock
		sigma: the sample covariance matrix of each stock's returns
		min_return: the expected return

	returns: float
	'''

	portfolio_return = np.dot(X, mu)
	portfolio_variance = np.dot(np.dot(X, sigma), X)
	
	return portfolio_variance + (100 * abs(portfolio_return - min_return))

def _annualize_monthly_returns(prices):
	'''
	_annualize_monthly_returns

	Is used to compute the annualized monthly returns. 

	params
	------
		prices: pandas Series

	returns: float 
	'''

	diff = np.array([prices.iloc[i]/prices.iloc[i-1] - 1 
			for i in range(1, len(prices))])
	return (1 + diff.mean())**12 - 1


if __name__ == "__main__":
	stocks = ['GM', 'CSCO', 'WFM', 'SPY']
	result = optimize_return(stocks, 2013, .10, .40)
	filtered = {k: result[k] for k in result if result[k] > 0}
	print(filtered)
	

import yahoo
import numpy as np
import scipy.optimize

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

def optimize_sharpe_ratio(symbols, start, rf_return, max_weight=0.5):
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

	raw_returns = [yahoo.monthly_returns(sym, start) for sym in symbols]	
	mu = np.array([ret.mean() for ret in raw_returns])
	sigma = np.cov(raw_returns)

	initial_X = np.ones(len(mu))/len(mu)
	bounds = [(0., max_weight) for i in range(len(mu))]
	constraints = ({'type': 'eq', 'fun': lambda X: sum(X) - 1.})

	optimized_weights = map(lambda x: round(100*x), 
			scipy.optimize.minimize(_sharpe_ratio, initial_X, 
			(mu, sigma, rf_return), method='SLSQP', constraints=constraints, 
			bounds=bounds).x)

	return dict(zip(symbols, optimized_weights))
	

def optimize_return_amount(symbols, start, min_return, max_weight=0.5):
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

	raw_returns = [yahoo.monthly_returns(sym, start) for sym in symbols]	
	mu = np.array([ret.mean() for ret in raw_returns])
	sigma = np.cov(raw_returns)
	
	initial_X = np.ones(len(mu))/len(mu)
	bounds = [(0., max_weight) for i in range(len(mu))]
	constraints = ({'type': 'eq', 'fun': lambda X: sum(X) - 1.})

	optimized_weights = map(lambda x: round(100*x), 
			scipy.optimize.minimize(_maximize_ret, initial_X, 
			(mu, sigma, min_return), method='SLSQP', constraints=constraints, 
			bounds=bounds).x)

	return dict(zip(symbols, optimized_weights))

if __name__ == "__main__":
	stocks = ['BEN', 'WFM', 'SPY', 'CSCO', 'NKE', 'SBUX']	
	weights = optimize_sharpe_ratio(stocks, 2013, .015)
	print(weights)

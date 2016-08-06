import requests
import pandas as pd
from io import StringIO

BASE_URL = 'http://ichart.yahoo.com/table.csv?s='

def monthly_returns(symbol, start, end=None):
	'''
	monthly_returns	
		
	Gets historical returns by year. If end is not specified 
	then we query Yahoo for the monthly returns until the most
	recent month

	params
	------
		start: starting year (int)
		end: ending year (int)
		symbol: stock symbol (str)

	returns: Pandas series corresponding to the stock's adjusted close
	'''		

	end_str = '&d=01&e=01=f={:d}'.format(end) if end else ''
	end_str += '&g=m'
		
	prices = pd.read_csv(StringIO(requests.get(
			BASE_URL + '{:s}&a=01&b=01&c={:d}'
			.format(symbol, start) + end_str).text), 
			usecols=['Date', 'Adj Close'])
	return prices.sort_values('Date')['Adj Close']

if __name__ == "__main__":
	print(monthly_returns('WFC', 2013))

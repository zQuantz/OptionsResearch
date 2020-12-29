import pandas as pd
import numpy as np

###################################################################################################

options = pd.read_csv("data/small_options.csv")
options = options[options.tdays_to_expiry > 0]
options['mid_price'] = (options.bid_price + options.ask_price) / 2

start = "2019-01-01"
end = "2029-01-01"

fridays = pd.date_range(start, end, freq="WOM-3FRI").astype(str)
thursdays = pd.date_range(start, end, freq="WOM-3THU").astype(str)
regulars = list(fridays) + list(thursdays)

time_anchors = [
    1,
    3,
    6,
    9,
    12,
    18,
    24
]
time_anchors = np.array(time_anchors) * 21
time_df = pd.DataFrame(time_anchors, columns = ['expiration'])

moneyness_anchors = list(range(80, 125, 5))
moneyness_anchors = np.array(moneyness_anchors)
moneyness_df = pd.DataFrame(moneyness_anchors, columns = ['moneyness'])

TDF_COLS = ['expiration'] + [f"m{m}" for m in range(80, 125, 5)]
MDF_COLS = ['moneyness', 'm1', 'm2', 'w1', 'w2', 'iv1', 'iv2']
COLS = [
	f"m{m1}m{m2}"
	for m2 in [1,3,6,9,12,18,24]
	for m1 in range(80, 125, 5)
]

###################################################################################################

def pre_filters(options):

	options = options[options.implied_volatility != 0]
	options = options[options.bid_price != 0]
	return options[options.ask_price != 0]

def post_filters(options):

	ticker_exp = options.date_current + " " + options.expiration_date
	x = ticker_exp.value_counts()
	x = x[x >= 20]

	return options[ticker_exp.isin(x.index)]

def calculate_implied_forward(options):

	def by_ticker(ticker_exp):

		cols = ['strike_price', 'mid_price']
		calls = ticker_exp[ticker_exp.option_type == "C"][cols]
		puts = ticker_exp[ticker_exp.option_type != "C"][cols]

		if len(calls) == 0 or len(puts) == 0:
			return None

		prices = calls.merge(puts, on="strike_price", how="outer")
		prices = prices.reset_index(drop=True)

		diff = (prices.mid_price_x - prices.mid_price_y)
		idc = diff.abs().argmin()

		r = np.log(1 + ticker_exp.rate.values[0])
		T = ticker_exp.tdays_to_expiry.values[0] / 252
		K = ticker_exp.strike_price.values[idc]

		return K + np.exp(-r * T) * diff.iloc[idc]

	cols = ["date_current", "expiration_date"]
	forwards = options.groupby(cols).apply(by_ticker)
	forwards = forwards.reset_index(name="F")
	return options.merge(forwards, on=cols, how="inner").dropna()

def brackets(values, anchors):
    
    N = len(values)
    M = len(anchors)
    
    values = np.array(values)
    matrix = values.repeat(M).reshape(N, M)
    
    matrix -= anchors
    signed_matrix = np.sign(matrix)
    dsigned_matrix = np.diff(signed_matrix, axis=0)
    
    return matrix, signed_matrix, dsigned_matrix

def calculate_bracket_coords(values, anchors, idx, extra_values=None):

	v1 = values[idx[0]]
	v2 = values[idx[0] + 1]
	v = anchors[idx[1]]

	p1 = 1 / abs(v1 - v)
	p2 = 1 / abs(v2 - v)
	d = p1 + p2

	p1 /= d
	p2 /= d

	coords = [v, v1, v2, p1, p2]

	if extra_values is not None:
		coords.extend([
			extra_values[idx[0]],
			extra_values[idx[0] + 1],
		])

	return coords

def calculate_brackets(values, anchors, sm, dsm, extra_values=None):

	brackets = [
		calculate_bracket_coords(values, anchors, idx, extra_values)
		for idx in np.argwhere(dsm == 2)
	]

	for idx in np.argwhere(sm == 0):

		bracket = calculate_bracket_coords(values, anchors, idx, extra_values)
		bracket[2] = bracket[1]
		bracket[3:5] = [0.5]*2

		if extra_values is not None:
			bracket[6] = bracket[5]

		brackets.append(bracket)

	return brackets

def by_ticker(options):

	expirations = options[options.expiration_date.isin(regulars)]
	expirations = expirations.tdays_to_expiry.unique()

	m, sm, dsm = brackets(expirations, time_anchors)
	time_brackets = calculate_brackets(expirations, time_anchors, sm, dsm)

	idc = np.argwhere(dsm.sum(axis=0) == 0)
	if len(idc) != 0:

		idc = idc.reshape(-1, )
		expirations = options.tdays_to_expiry.unique()
		m, sm, dsm = brackets(expirations, time_anchors[idc])
		time_brackets.extend(
			calculate_brackets(expirations, time_anchors[idc], sm, dsm)
		)

	time_brackets = np.array(sorted(time_brackets))
	expirations = np.unique(time_brackets[:, 1:3].reshape(-1, ))

	ivs = {}
	for expiration in expirations:

		_options = options[options.tdays_to_expiry == expiration]
		moneyness = _options.moneyness.values
		iv = _options.implied_volatility.values

		m, sm, dsm = brackets(moneyness, moneyness_anchors)
		moneyness_brackets = calculate_brackets(
			moneyness,
			moneyness_anchors,
			sm,
			dsm,
			iv
		)

		df = pd.DataFrame(moneyness_brackets, columns = MDF_COLS)
		df['iv'] = df.iv1 * df.w1 + df.iv2 * df.w2
		
		df = df[['moneyness', 'iv']]
		df = df.groupby("moneyness").mean().reset_index()
		df = moneyness_df.merge(df, on='moneyness', how='outer')

		df['expiration'] = expiration
		df = df.fillna(0)
		ivs[expiration] = df.iv.values

	surface = [
		[b[0]] + (ivs[b[1]] * b[3] + ivs[b[2]] * b[4]).tolist()
		for b in time_brackets
	]
	surface = pd.DataFrame(surface, columns = TDF_COLS)
	surface = time_df.merge(surface, on="expiration", how="outer")
	surface = surface.fillna(0).values[:, 1:].reshape(1, -1)
	return pd.DataFrame(surface, columns = COLS)

def calculate_surface(options):

	options = pre_filters(options)	
	options = calculate_implied_forward(options)

	omap = options.option_type.map({
		"C" : 1,
		"P" : -1
	})	
	options = options[omap * (options.F - options.strike_price) <= 0]
	options['moneyness'] = options.strike_price / options.F * 100

	options = post_filters(options)

	cols = ["date_current", "tdays_to_expiry", "strike_price"]
	options = options.sort_values(cols)

	surface = options.groupby("date_current").apply(by_ticker)
	surface = surface.reset_index(level=0)

if __name__ == "__main__":
	
	surface = calculate_surface(options)

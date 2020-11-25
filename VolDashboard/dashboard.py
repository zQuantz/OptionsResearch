from scipy.stats import percentileofscore as pctrank
from pandas.io.formats.style import Styler
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import numpy as np
import sys, os

###################################################################################################

DATE = "2010-01-01"
URL = "https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={p1}&period2={p2}"
URL += "&interval=1d&events=history&includeAdjustedClose=true"

LCOLS = [
	'Ticker', 'spread', 'rvspread', 'Carry', 'Corr6M', 'Corr3M',
	'Mean3Y', 'ZScore3Y', 'Min3Y', 'Max3Y', 'Rank3Y', 'PctRank3Y'
]
SCOLS = [
	'Ticker', 'spread', 'rvspread', 'Carry', 'Corr6M', 'Corr3M',
	'Mean', 'ZScore', 'Min', 'Max', 'Rank', 'PctRank'
]
FCOLS = [
	'Ticker', "Implied Spread", "RVol Spread", "Carry", "6M Corr.", "3M Corr.",
	"Mean", "Z-Score", "Min", "Max", "Rank", "Pct. Rank"
]

VICOLS = ["Ticker", "val", "1D Perf.", "3M Perf.", "52W Perf.", "ATL Rank", "1M RVol", "3M RVol"]
VIFCOLS = ["Ticker", "Price"] + VICOLS[2:]

ICOLS = ["Ticker", "adjclose", "1D Perf.", "3M Perf.", "52W Perf.", "ATH Rank", "Rel. Volume", "1M RVol", "3M RVol"]
IFCOLS = ["Ticker", "Price"] + ICOLS[2:]

CMAP = "RdBu"

###################################################################################################

PAGE = """
	<!DOCTYPE html>
	<html>
	<head>

		<title>Volatility Monitor</title>

		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">

		<!-- Font -->
		<link rel="preconnect" href="https://fonts.gstatic.com">
		<link href="https://fonts.googleapis.com/css2?family=Ubuntu&display=swap" rel="stylesheet">

		<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
		<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ho+j7jyWK8fNQe+A12Hb8AhRq26LrZ/JpcUGGOn+Y7RsweNrtN/tE3MoK7ZeZDyx" crossorigin="anonymous"></script>

	</head>
	<body>

		<style type="text/css">

			.jumbotron {
				background-image: url(https://media.giphy.com/media/S8IIUwihhFu7O90I8l/giphy.gif);
				background-size: 20rem;
				background-repeat: no-repeat;
				background-color: black;
				background-position: right;
			}
			.jumbotronHeader {
				font-family: 'Ubuntu', sans-serif;
				color: white;
				font-size: 5rem
			}
			.jumbotronLabel {
				color:white;
				font-size: 2rem;
				font-family: 'Ubuntu', sans-serif;
				padding-top: 1rem
			}
			.list-group-item.active {
				background-color: black;
				border-color: black;
				color:white;
			}
			.centerAlign {
				text-align: center;
			}
			.tableCol {
				width: 100%;
			}
			.typeRow {
				width: 100%;
				margin-left: 0;
				margin-right: 0;
			}
			body {
				line-height: 0.9
			}
		</style>
		TABLE_STYLES_HERE

		<div class="container-fluid" style="max-width: 50%; padding:0;">

			<div class="jumbotron jumbotron-fluid" style="padding-left: 3rem; padding-right: 3rem">

				<div class="container">

					<h1 class="jumbotronHeader">
						<b>O p t i Q s</b>
					</h1>
					<h6 class="jumbotronLabel">
						- Vol Monitor
					</h6>

				</div>

			</div>

			<hr>

			<h4>Indices</h4>
			<div class="row typeRow">

				<div class="tableCol col-12-xl col-12-lg col-12-md col-12-sm">
					EQUITY_INDEX_TABLE_HERE
				</div>

			</div>

			<hr>
			<h4>Volatility Indices</h4>
			<div class="row typeRow">

				<div class="tableCol col-12-xl col-12-lg col-12-md col-12-sm">
					VOL_INDEX_TABLE_HERE
				</div>

			</div>

			<hr>

			<div class="row typeRow" style="margin-bottom: -1.8rem; display: block;">

					<div class="list-group list-group-horizontal" id="list-tab" role="tablist" style="line-height: 0.25; margin-left: 70%; height:1.75rem;">
						<a class="list-group-item list-group-item-action centerAlign active" id="list-1ystats" data-toggle="list" href="#list-1y" role="tab" aria-controls="home">1 Yr Stats</a>
						<a class="list-group-item list-group-item-action centerAlign" id="list-3ystats" data-toggle="list" href="#list-3y" role="tab" aria-controls="profile">3 Yr Stats</a>
					</div>
					
			</div>
			
			<h4>Spreads</h4>
			<div class="row typeRow" style="margin-bottom: 2rem">
			
				<div class="tableCol col-12-xl col-12-lg col-12-md col-12-sm">
					<div class="tab-content" id="nav-tabContent">
						<div class="tab-pane fade show active" id="list-1y" role="tabpanel" aria-labelledby="list-1ystats">
							SHORT_TABLE_HERE
						</div>
						<div class="tab-pane fade" id="list-3y" role="tabpanel" aria-labelledby="list-3ystats">
							LONG_TABLE_HERE
						</div>
					</div>
				</div>
				
			</div>

		</div>

	</body>
	</html>

"""

###################################################################################################

def rvol(key):
	rv = pd.read_csv(products[key])
	rv['Date'] = pd.to_datetime(rv.Date.values)
	rv = rv.sort_values('Date', ascending=True)
	rv.columns = ['date', 'val']
	return rv

def get_vixm(key):
	v = pd.read_csv(products[key], skiprows=2)
	v.columns = ['date', 'open', 'high', 'low', 'val']
	v['date'] = pd.to_datetime(v.date.values)
	return v[['date', 'val']]

def get_major(key, name, skip):
	v = pd.read_csv(products[key], skiprows=skip)
	v['Date'] = pd.to_datetime(v.Date.values)
	v = v[['Date', name]]
	v.columns = ['date', 'val']
	return v

def get_index(ticker, p1, p2):
	index = pd.read_csv(URL.format(ticker=ticker, p1=p1, p2=p2))
	index.columns = ['date', 'open', 'high', 'low', 'close', 'adjclose', 'volume']
	index['date'] = pd.to_datetime(index.date.values)
	return index[index.date >= DATE].reset_index(drop=True)

def calculate_rv(x, days):
	x = np.log(x / x.shift()) ** 2
	x = x.rolling(days, min_periods=1).sum()
	return np.sqrt(x * (252 / days)) * 100

###################################################################################################

def spread_stats(ticker, ticker1, ticker2, rvol1, rvol2):
	
	data = pd.DataFrame(zip(
		ticker1.date,
		ticker1.val,
		ticker2.val,
		ticker1.val - ticker2.val,
		rvol1 - rvol2
	), columns = ['date', 'v1', 'v2', 'spread', 'rvspread'])
	
	r3 = data.spread.rolling(252*3, min_periods=1)
	r1 = data.spread.rolling(252, min_periods=1)

	data['Ticker'] = ticker
	data['Carry'] = data.rvspread - data.spread
	data['Corr6M'] = data.v1.rolling(126, min_periods=1).corr(data.v2) * 100
	data['Corr3M'] = data.v1.rolling(63, min_periods=1).corr(data.v2) * 100
	
	data['Mean3Y'] = r3.mean()
	data['ZScore3Y'] = (data.spread - data.Mean3Y) / r3.std()
	data['Min3Y'] = r3.min()
	data['Max3Y'] = r3.max()
	data['Rank3Y'] = (data.spread - data.Min3Y) / (data.Max3Y - data.Min3Y) * 100
	data['PctRank3Y'] = r3.apply(lambda x: pctrank(x, x.values[-1]))

	data['Mean'] = r1.mean()
	data['ZScore'] = (data.spread - data.Mean) / r1.std()
	data['Min'] = r1.min()
	data['Max'] = r1.max()
	data['Rank'] = (data.spread - data.Min) / (data.Max - data.Min) * 100
	data['PctRank'] = r1.apply(lambda x: pctrank(x, x.values[-1]))
	
	return data

def style_spread(data):
	
	styler = Styler(data.iloc[-1:], precision=2)
	
	def z_score(x):
		m = x.mean()
		return m - x.std() * 3, m + x.std() * 3
	
	keys = ['Mean', '6M Corr.', '3M Corr.', 'Carry', 'Implied Spread', 'RVol Spread']
	for key in keys:
		l, h = z_score(data[key])
		styler = styler.background_gradient(cmap=CMAP, vmin=l, vmax=h, subset=[key])
				
	keys = ['Rank', 'Pct. Rank']
	styler = styler.background_gradient(cmap=CMAP, vmin=-10, vmax=110, subset=keys)
	styler = styler.background_gradient(cmap=CMAP, vmin=-3, vmax=3, subset=["Z-Score"])
	
	return styler

###################################################################################################

def index_stats(ticker, index):
	
	index['Ticker'] = ticker
	index['1D Perf.'] = index.adjclose.pct_change() * 100
	index['3M Perf.'] = index.adjclose.pct_change(periods=63) * 100
	index['52W Perf.'] = index.adjclose.pct_change(periods=252) * 100
	index['ATH Rank'] = index.adjclose.max() / index.adjclose * 100
	index['Rel. Volume'] = index.volume / index.volume.rolling(21, min_periods=1).mean()
	index['1M RVol'] = calculate_rv(index.adjclose, 21)
	index['3M RVol'] = calculate_rv(index.adjclose, 63)
	
	return index

def style_index(data):
	
	styler = Styler(data.iloc[-1:], precision=2)
	
	def z_score(x):
		m = x.mean()
		return m - x.std() * 3, m + x.std() * 3
	
	keys = ["1D Perf.", "3M Perf.", "52W Perf.", "Rel. Volume", "1M RVol", "3M RVol"]
	for key in keys:
		l, h = z_score(data[key])
		styler = styler.background_gradient(cmap=CMAP, vmin=l, vmax=h, subset=[key])
				
	keys = ['ATH Rank']
	styler = styler.background_gradient(cmap=CMAP, vmin=-10, vmax=110, subset=keys)
	
	return styler

###################################################################################################

def vindex_stats(ticker, index):

	index['Ticker'] = ticker
	index['1D Perf.'] = index.val.pct_change() * 100
	index['3M Perf.'] = index.val.pct_change(periods=63) * 100
	index['52W Perf.'] = index.val.pct_change(periods=252) * 100
	index['ATL Rank'] = index.val.min() / index.val * 100
	index['1M RVol'] = calculate_rv(index.val, 21)
	index['3M RVol'] = calculate_rv(index.val, 63)

	return index

def style_vindex(data):
	
	styler = Styler(data.iloc[-1:], precision=2)
	
	def z_score(x):
		m = x.mean()
		return m - x.std() * 3, m + x.std() * 3
	
	keys = ["1D Perf.", "3M Perf.", "52W Perf.", "1M RVol", "3M RVol"]
	for key in keys:
		l, h = z_score(data[key])
		styler = styler.background_gradient(cmap=CMAP, vmin=l, vmax=h, subset=[key])
				
	keys = ['ATL Rank']
	styler = styler.background_gradient(cmap=CMAP, vmin=-10, vmax=110, subset=keys)
	
	return styler

###################################################################################################

def filter_cols(data, columns, new_cols):
	data = data[columns]
	data.columns = new_cols
	return data

def merge(items):
	
	html = items[0]
	for item in items[1:]:
		html.find("style").insert_after(item.find("style"))
		html.find("tbody").append(item.find_all("tr")[1])
		
	return html

if __name__ == '__main__':

	## Prep
	dt = datetime.now()

	p1 = datetime(dt.year-15, dt.month, dt.day)
	p1 = int(p1.timestamp() / 1000) * 1000

	p2 = datetime(dt.year, dt.month, dt.day)
	p2 = int(p2.timestamp() / 1000) * 1000

	products = pd.read_csv("data/vol_products.csv")
	products = products.set_index("Ticker")["Link"].to_dict()

	rv, rv3m, rv6m = rvol("RVOL"), rvol("RVOL3M"), rvol("RVOL6M")
	vix3m, vix6m = get_vixm("VIX3M"), get_vixm("VIX6M")
	vix = get_major("VIX", "VIX Close", 1)
	vxn = get_major("VXN", "Close", 2)
	rvx = get_major("RVX", "Close", 2)
	vxd = get_major("VXD", "Close", 4)

	spx = get_index("%5EGSPC", p1, p2)
	rut = get_index("%5ERUT", p1, p2)
	dji = get_index("%5EDJI", p1, p2)
	ndx = get_index("%5ENDX", p1, p2)

	spxrv = calculate_rv(spx.adjclose, 21)
	spxrv3m = calculate_rv(spx.adjclose, 63)
	spxrv6m = calculate_rv(spx.adjclose, 126)

	rutrv = calculate_rv(rut.adjclose, 21)
	ndxrv = calculate_rv(ndx.adjclose, 21)
	djirv = calculate_rv(dji.adjclose, 21)

	rv = rv[rv.date >= DATE].reset_index(drop=True)
	rv3m = rv3m[rv3m.date >= DATE].reset_index(drop=True)
	rv6m = rv6m[rv6m.date >= DATE].reset_index(drop=True)
	vix = vix[vix.date >= DATE].reset_index(drop=True)
	vix3m = vix3m[vix3m.date >= DATE].reset_index(drop=True)
	vix6m = vix6m[vix6m.date >= DATE].reset_index(drop=True)
	vxn = vxn[vxn.date >= DATE].reset_index(drop=True)
	rvx = rvx[rvx.date >= DATE].reset_index(drop=True)
	vxd = vxd[vxd.date >= DATE].reset_index(drop=True)

	## Spreads
	rvx_vix = spread_stats("RVX VIX", rvx, vix, rutrv, spxrv)
	vxd_vix = spread_stats("VXD VIX", vxd, vix, djirv, spxrv)
	vxn_vix = spread_stats("VXN VIX", vxn, vix, ndxrv, spxrv)
	vix3m_vix = spread_stats("VIX3M VIX", vix3m, vix, spxrv3m, spxrv)
	vix6m_vix = spread_stats("VIX6M VIX", vix6m, vix, spxrv6m, spxrv)
	vxd_vxn = spread_stats("VXD VXN", vxd, vxn, djirv, ndxrv)
	rvx_vxn = spread_stats("RVX VXN", rvx, vxn, rutrv, ndxrv)
	rvx_vxd = spread_stats("RVX VXD", rvx, vxd, rutrv, djirv)

	## Indices
	spx = index_stats("S&P 500", spx)[ICOLS]
	spx.columns = IFCOLS

	rut = index_stats("Russell 2000", rut)[ICOLS]
	rut.columns = IFCOLS

	ndx = index_stats("Nasdaq 100", ndx)[ICOLS]
	ndx.columns = IFCOLS

	dji = index_stats("Dow Jones", dji)[ICOLS]
	dji.columns = IFCOLS

	## Volatility Indices
	vix = vindex_stats("VIX", vix)[VICOLS]
	vix.columns = VIFCOLS

	vix3m = vindex_stats("3M VIX", vix3m)[VICOLS]
	vix3m.columns = VIFCOLS

	vix6m = vindex_stats("6M VIX", vix6m)[VICOLS]
	vix6m.columns = VIFCOLS

	rvx = vindex_stats("RVX", rvx)[VICOLS]
	rvx.columns = VIFCOLS

	vxn = vindex_stats("VXN", vxn)[VICOLS]
	vxn.columns = VIFCOLS

	vxd = vindex_stats("VXD", vxd)[VICOLS]
	vxd.columns = VIFCOLS

	## HTML
	items = [
		style_spread(filter_cols(rvx_vix.copy(), SCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vxd_vix.copy(), SCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vxn_vix.copy(), SCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vix3m_vix.copy(), SCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vix6m_vix.copy(), SCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vxd_vxn.copy(), SCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(rvx_vxn.copy(), SCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(rvx_vxd.copy(), SCOLS, FCOLS)).hide_index().render()
	]
	items = [
		BeautifulSoup(item, features="lxml")
		for item in items
	]
	html_short = merge(items)
	short_table = html_short.find("table")
	short_table.attrs['class'] = "table table-sm table-hover centerAlign"

	items = [
		style_spread(filter_cols(rvx_vix.copy(), LCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vxd_vix.copy(), LCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vxn_vix.copy(), LCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vix3m_vix.copy(), LCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vix6m_vix.copy(), LCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(vxd_vxn.copy(), LCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(rvx_vxn.copy(), LCOLS, FCOLS)).hide_index().render(),
		style_spread(filter_cols(rvx_vxd.copy(), LCOLS, FCOLS)).hide_index().render()
	]
	items = [
		BeautifulSoup(item, features="lxml")
		for item in items
	]
	html_long = merge(items)
	long_table = html_long.find("table")
	long_table.attrs['class'] = "table table-sm table-hover centerAlign"

	items = [
		style_vindex(vix).hide_index().render(),
		style_vindex(vix3m).hide_index().render(),
		style_vindex(vix6m).hide_index().render(),
		style_vindex(rvx).hide_index().render(),
		style_vindex(vxn).hide_index().render(),
		style_vindex(vxd).hide_index().render()
	]
	items = [
		BeautifulSoup(item, features="lxml")
		for item in items
	]
	html_vindex = merge(items)
	vindex_table = html_vindex.find("table")
	vindex_table.attrs['class'] = "table table-sm table-hover centerAlign"

	items = [
		style_index(spx).hide_index().render(),
		style_index(ndx).hide_index().render(),
		style_index(rut).hide_index().render(),
		style_index(dji).hide_index().render()
	]
	items = [
		BeautifulSoup(item, features="lxml")
		for item in items
	]
	html_index = merge(items)
	index_table = html_index.find("table")
	index_table.attrs['class'] = "table table-sm table-hover centerAlign"

	table_styles = ''.join(
		list(map(str, html_short.find_all("style"))) + 
		list(map(str, html_long.find_all("style"))) + 
		list(map(str, html_index.find_all("style"))) + 
		list(map(str, html_vindex.find_all("style")))
	)

	PAGE = PAGE.replace("EQUITY_INDEX_TABLE_HERE", str(index_table))
	PAGE = PAGE.replace("VOL_INDEX_TABLE_HERE", str(vindex_table))
	PAGE = PAGE.replace("LONG_TABLE_HERE", str(long_table))
	PAGE = PAGE.replace("SHORT_TABLE_HERE", str(short_table))
	PAGE = PAGE.replace("TABLE_STYLES_HERE", str(table_styles))

	with open("auto.html", "w") as file:
		file.write(PAGE)

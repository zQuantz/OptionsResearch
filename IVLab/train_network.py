from fastai.tabular.all import *
from scipy.stats import norm
import pandas as pd
import numpy as np
import sys, os

import seaborn as sns
import matplotlib.pyplot as plt

def bs_price(S, K, T, r, q, v, t):
    
    d1 = np.log(S / K) + (r + 0.5 * v * v) * T
    d1 /= np.sqrt(T) * v
    d2 = d1 - np.sqrt(T) * v
    
    d1 *= t
    d2 *= t
    
    return t * S * np.exp(-q * T) * norm.cdf(d1) - t * K * np.exp(-r * T) * norm.cdf(d2)

def convert_act_cls(model, layer_type_old, layer_type_new):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_act_cls(module, layer_type_old, layer_type_new)

        if type(module) == layer_type_old:
            layer_old = module
            layer_new = layer_type_new
            model._modules[name] = layer_new

    return model

rates = np.log(1 + np.arange(0, 2e-1, 0.001))
rmin, rmax = rates.min(), rates.max()

divs = np.log(1 + np.arange(0, 2e-1, 0.001))
qmin, qmax = divs.min(), divs.max()

stock_prices = np.arange(1, 200, 1)
strike_prices = np.arange(50, 150, 1)

time_to_expiries = np.arange(1, 1000, 1)
tmin, tmax = time_to_expiries.min(), time_to_expiries.max()

implied_volatilities = np.arange(5e-2, 1.25, 0.0005).tolist()
implied_volatilities += np.arange(1.25, 5, 0.05).tolist()
implied_volatilities = np.array(implied_volatilities)

types = np.array([1, -1])

data = [
    rates,
    divs,
    stock_prices,
    strike_prices,
    time_to_expiries,
    implied_volatilities,
    types
]

mods = np.array([
    len(d)
    for d in data
]).reshape((1, -1))
idc = np.random.randint(0, mods.max(), size=(750_000, len(data)))
idc = idc % mods

rates = rates[idc[:, 0]]
divs = divs[idc[:, 1]]
stock_prices = stock_prices[idc[:, 2]]
strike_prices = strike_prices[idc[:, 3]]
time_to_expiries = time_to_expiries[idc[:, 4]]
implied_volatilities = implied_volatilities[idc[:, 5]]
types = types[idc[:, 6]]
prices = bs_price(
    stock_prices,
    strike_prices,
    time_to_expiries / 252,
    rates,
    divs,
    implied_volatilities,
    types
)

rdata = pd.DataFrame(np.array([
    stock_prices,
    strike_prices,
    time_to_expiries,
    rates,
    divs,
    implied_volatilities,
    types,
    prices
]).T, columns = [
    'S',
    'K',
    'TT',
    'r',
    'q',
    'v',
    't',
    'p'
])
rdata = rdata[rdata.p > 0]

rdata['M'] = np.log(rdata.S / rdata.K)

x = rdata.t * (rdata.S - rdata.K)
x[x < 0] = 0
rdata['tv'] = rdata.p - x
rdata['rtv'] = rdata.tv / rdata.K

x = rdata.t.values
x[x < 0] = 0
rdata['t'] = x

rdata['TT'] = np.log(rdata.TT.values / 252)

rdata['r'] = (rdata.r - rmin) / (rmax - rmin)
rdata['q'] = (rdata.q - qmin) / (qmax - qmin)

cols = [
    "M",
    "TT",
    "r",
    "q",
    "t",
    "rtv",
    "v"
]
rdata = rdata[cols]

options = pd.read_csv("data/options.csv")
options['mid_price'] = (options.ask_price + options.bid_price) / 2

cols = [
    'stock_price',
    'strike_price',
    'tdays_to_expiry',
    'rate',
    'dividend_yield',
    'option_type',
    'mid_price',
    'bivs'
]
odata = options[cols]
odata = odata[odata.bivs < 10]
odata = odata[odata.tdays_to_expiry > 0]
odata = odata.reset_index(drop=True)

odata['M'] = np.log(odata.stock_price / odata.strike_price)

odata['r'] = np.log(1 + odata.rate / 100)
x = (odata.r - rmin) / (rmax - rmin)
x[x > 1] = 1
odata['r'] = x

odata['q'] = np.log(1 + odata.dividend_yield / 100)
x = (odata.q - qmin) / (qmax - qmin)
x[x > 1] = 1
odata['q'] = x

odata['TT'] = np.log(odata.tdays_to_expiry / 252)

omap = odata.option_type.map({"C" : 1, "P" : -1})
x = omap * (odata.stock_price - odata.strike_price)
x[x < 0] = 0
x = odata.mid_price - x
odata['rtv'] = x / odata.strike_price

odata['v'] = odata.bivs

omap[omap < 0] = 0
odata['t'] = omap

cols = [
    "M",
    "TT",
    "r",
    "q",
    "t",
    "rtv",
    'v'
]
odata = odata[cols]

if __name__ == "__main__":
    
    splits = RandomSplitter(valid_pct=0.35)(range_of(odata))
    dls = TabularDataLoaders.from_df(
        df=rdata,
        y_names='v',
        splits=splits,
        bs=1024
    )
    
    learn = tabular_learner(
        dls,
        metrics=mse,
        layers=[400, 400, 400, 400],
        cbs=[
            SaveModelCallback()
        ]
    )

    learn.model = convert_act_cls(learn.model, torch.nn.ReLU, Mish())
    learn.lr = 10e-4
    learn.fit_flat_cos(3000)

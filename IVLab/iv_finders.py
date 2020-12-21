from scipy.optimize import newton, brentq
from scipy.stats import norm
from tqdm import tqdm
import pandas as pd
import numpy as np

import sys, os

options = pd.read_csv("data/options.csv")
print(options.head())
rates = pd.read_csv("data/cubic_ratemap.csv")
print(rates.head())

moneyness = abs(options.stock_price / options.strike_price - 1)
options = options[moneyness <= 0.35]
options = options[options.bid_price != 0]
options = options[options.ask_price != 0]
options = options.merge(rates, on=['date_current', 'tdays_to_expiry'], how="inner")
options = options.reset_index(drop=True)
print(options.head())

Ss = options.stock_price.values
Ks = options.strike_price.values
rs = options.rate.values / 100
Ts = options.tdays_to_expiry.values / 252
qs = options.dividend_yield.values / 100
ts = options.option_type.map({"C" : 1, "P" : -1}).values
Ms = (options.ask_price + options.bid_price).values * 0.5

def bs_price(v, S, K, T, r, q, t, M):
    
    d1 = np.log(S / K) + (r + 0.5 * v * v) * T
    d1 /= np.sqrt(T) * v
    d2 = d1 - np.sqrt(T) * v
    
    if t == 1:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    
def vega(v, S, K, T, r, q, t, M):
    
    d1 = np.log(S / K) + (r + 0.5 * v * v) * T
    d1 /= np.sqrt(T) * v
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

def root(v, *args):
    return bs_price(v, *args) - M

if __name__ == '__main__':
    
    brent_ivs = []
    for i, (S, K, T, r, q, t, M) in enumerate(zip(Ss, Ks, Ts, rs, qs, ts, Ms)):
        
        if i % 10000 == 0:
            print(i / len(Ss) * 100)
        
        try:
            iv = brentq(root, 0, 100, args=(S, K, T, r, q, t, M))
        except Exception as e:
            iv = 0
        brent_ivs.append(iv)
        
    brent_ivs = np.array(brent_ivs)
    print("Non-zero", brent_ivs[brent_ivs > 0].shape[0])
    print("Zero", brent_ivs[brent_ivs == 0].shape[0])
    
    newton_ivs = []
    for i, (S, K, T, r, q, t, M) in enumerate(zip(Ss, Ks, Ts, rs, qs, ts, Ms)):
        
        if i % 10000 == 0:
            print(i / len(Ss) * 100)
            
        try:
            iv = newton(root, 0.5, fprime=vega, args=(S, K, T, r, q, t, M))
        except Exception as e:
            iv = 0
        newton_ivs.append(iv)
        
    newton_ivs = np.array(newton_ivs)
    print("Non-zero", newton_ivs[newton_ivs > 0].shape[0])
    print("Zero", newton_ivs[newton_ivs == 0].shape[0])
    
    options['bivs'] = brent_ivs
    options['nivs'] = newton_ivs
    options.to_csv("data/options.csv", index=False)
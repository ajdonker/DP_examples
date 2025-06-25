import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
import numba
df = pd.read_csv(
    #'GUI_ENERGY_PRICES_202412312300-202512312300.csv',
    'GUI_ENERGY_PRICES_202312312300-202412312300.csv',
    index_col=0,
    parse_dates=True
)
prices = df['Day-ahead Price (EUR/MWh)']
price_arr = prices.to_numpy(dtype=np.float64)
@numba.njit
def solve_dp_numba(prices, C, R, η, N,init_soc_frac):
    T = prices.shape[0]
    soc_grid = np.linspace(0.0, C, N)
    V = np.zeros((T+1, N))
    actions = np.array([-R, 0.0, R])
    for t in range(T-1, -1, -1):
        p = prices[t]
        for i in range(N):
            soc = soc_grid[i]
            best = -1e20
            for a in actions:
                if a > 0 and soc >= C: continue
                if a < 0 and soc <= 0: continue
                if a > 0:
                    actual  = min(a, C - soc)
                    new_soc = soc + actual * η
                    rew     = -p * actual
                elif a < 0:
                    actual  = min(-a, soc)
                    new_soc = soc - actual
                    rew     =  p * actual * η
                else:
                    new_soc = soc
                    rew     = 0.0
                j = int(round(new_soc / C * (N-1)))
                val = rew + V[t+1, j]
                if val > best:
                    best = val
            V[t, i] = best
    #return V[0, N//2]
    

def solve_dp(price, C=1.0, R=0.25, eff=0.95, N=100, init_soc_frac=0.5):
    T = len(price)
    soc_grid = np.linspace(0, C, N)
    V = np.zeros((T+1, N))
    actions = np.array([-R, 0.0, +R])
    for t in reversed(range(T)):
        p = price[t]
        for i, soc in enumerate(soc_grid):
            best = -np.inf
            for a in actions:
                if a > 0:
                    actual = min(a, C - soc)
                    new_soc = soc + actual * eff
                    rew = -p * actual
                elif a < 0:
                    actual = min(-a, soc)
                    new_soc = soc - actual
                    rew = p * actual * eff
                else:
                    new_soc = soc
                    rew = 0.0
                j = int(np.round(new_soc/C*(N-1)))
                val = rew + V[t+1, j]
                if val > best:
                    best = val
            V[t, i] = best
    #return V[0, int((N-1)/2)]
    init_index = int(np.round(init_soc_frac * (N-1)))
    return V[0, init_index]

def solve_lp_sparse(prices, C=1.0, R=0.25, η=0.95, soc0=None):
    T = len(prices)
    if soc0 is None: soc0 = C/2
    n = 2*T + (T+1)
    c = np.concatenate([prices, -prices, np.zeros(T+1)])
    A_eq = lil_matrix((T+1, n))
    b_eq = np.zeros(T+1)
    A_eq[0, 2*T] = 1.0
    b_eq[0] = soc0
    for t in range(T):
        A_eq[t+1, t] = -η
        A_eq[t+1, T+t] = 1/η
        A_eq[t+1, 2*T+t] = -1.0
        A_eq[t+1, 2*T+t+1] = 1.0
    A_eq = A_eq.tocsc()
    bounds = [(0, R)]*T + [(0, R)]*T + [(0, C)]*(T+1)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return -res.fun
horizons = [96, 288, 672, 1440, 2880,5760,11520,35040]  
results = []
for T in horizons:
    price_slice = price_arr[:T]
    start = time.time()
    dp_profit = solve_dp_numba(price_slice,1.0,0.25,0.95,100,0.5)
    dp_time = time.time() - start
    start = time.time()
    lp_profit = solve_lp_sparse(price_slice)
    lp_time = time.time() - start
    results.append({'Хоризонт (број на 15мин периоди)': T, 'Време Д.П. (s)': dp_time, 'Време Л.П. (s)': lp_time,
                    'Профит Д.П.':dp_profit,'Профит Л.П.':lp_profit})

df_results = pd.DataFrame(results)
print(df_results)
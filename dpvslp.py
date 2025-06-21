import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
import numba
@numba.njit
def solve_dp_numba(prices, C, R, η, N):
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
    return V[0, N//2]

from scipy.sparse import lil_matrix
# DP solver from above
def solve_dp(price, C=1.0, R=0.25, eff=0.99, N=41):
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
    return V[0, int((N-1)/2)]

# Sparse LP solver
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
def solve_lp(price, C=1.0, R=0.25, η=0.95, soc0=None):
    T = len(price)
    if soc0 is None:
        soc0 = C/2
    # decision vars: c0..c_{T-1}, d0..d_{T-1}, s0..s_T
    n = 2*T + (T+1)
    # Objective: minimize price*c - price*d (i.e. maximize price*d - price*c)
    c = np.concatenate([price, -price, np.zeros(T+1)])
    # Equality constraints
    A_eq = np.zeros((T+1, n))
    b_eq = np.zeros(T+1)
    # s0 = soc0
    A_eq[0, 2*T] = 1
    b_eq[0] = soc0
    # SoC dynamics
    for t in range(T):
        A_eq[t+1,     t    ] = -η
        A_eq[t+1,   T + t  ] =  1/η
        A_eq[t+1, 2*T + t  ] = -1
        A_eq[t+1, 2*T + t+1] =  1
    # Bounds for each variable
    bounds = [(0, R)]*T + [(0, R)]*T + [(0, C)]*(T+1)
    # Solve
    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    profit = -res.fun
    return profit
horizons = [96, 288, 672, 1440, 2880,5760,11520,35040]  
results = []
for T in horizons:
    price = np.random.rand(T) * 50 + 20
    start = time.time()
    dp_profit = solve_dp_numba(price,1.0,0.25,0.95,99)
    dp_time = time.time() - start
    start = time.time()
    lp_profit = solve_lp_sparse(price)
    lp_time = time.time() - start
    results.append({'Horizon (periods)': T, 'DP time (s)': dp_time, 'LP time (s)': lp_time})

df_results = pd.DataFrame(results)
print(df_results)
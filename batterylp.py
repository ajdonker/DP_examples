from scipy.optimize import linprog
from scipy.sparse import lil_matrix
import numpy as np
import pandas as pd
price = pd.read_csv(
    'GUI_ENERGY_PRICES_202412312300-202512312300.csv',
    index_col=0,
    parse_dates=True
)
price_series = pd.to_numeric(
    price['Day-ahead Price (EUR/MWh)'], 
    errors='coerce'
).dropna()
prices = price_series.values
def solve_lp(price, C=1.0, R=0.25, η=0.95, soc0=None):
    T = len(price)
    if soc0 is None: soc0 = C/2
    # decision vars order: [c0..c_{T-1}, d0..d_{T-1}, s0..s_T]
    n = 2*T + (T+1)
    # Objective: maximize p·d - p·c => linprog minimizes, so c = [price, -price, zeros]
    c = np.concatenate([ -price, price, np.zeros(T+1) ])
    # Constraints: equality for SoC dynamics
    A_eq = np.zeros((T+1, n))
    b_eq = np.zeros(T+1)
    # s0 = soc0
    A_eq[0, 2*T + 0] = 1
    b_eq[0] = soc0
    # For t=0..T-1: s_{t+1} - s_t - η c_t + d_t/η = 0
    for t in range(T):
        # c_t coeff
        A_eq[t+1, 0 + t]      = -η
        # d_t coeff
        A_eq[t+1, T + t]      = +1/η
        # s_t coeff
        A_eq[t+1, 2*T + t]    = -1
        # s_{t+1} coeff
        A_eq[t+1, 2*T + t+1]  = +1
    # Bounds for each var
    bounds = (
      [(0, R)] * T          # c_t
    + [(0, R)] * T          # d_t
    + [(0, C)] * (T+1)      # s_t
    )
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    profit = -res.fun  # since we minimized -objective
    return profit, res

def solve_lp_sparse(prices, C=1.0, R=0.25, η=0.95, soc0=None):
    T = len(prices)
    if soc0 is None: soc0 = C/2
    n = 2*T + (T+1)

    # Objective vector (still dense)
    c = np.concatenate([ prices, -prices, np.zeros(T+1) ])

    # Build A_eq as a sparse LIL matrix
    A_eq = lil_matrix((T+1, n), dtype=float)
    b_eq = np.zeros(T+1)

    # s0 = soc0
    A_eq[0, 2*T + 0] = 1.0
    b_eq[0]         = soc0

    # SoC dynamics: s_{t+1} = s_t + η·c_t − d_t/η
    for t in range(T):
        A_eq[t+1,     t    ] = -η     # c_t
        A_eq[t+1,   T + t  ] = +1/η   # d_t
        A_eq[t+1, 2*T + t  ] = -1.0   # s_t
        A_eq[t+1, 2*T + t+1] = +1.0   # s_{t+1}

    # Convert to CSR/CSC for the solver
    A_eq = A_eq.tocsc()

    # Bounds
    bounds = [(0, R)]*T + [(0, R)]*T + [(0, C)]*(T+1)

    res = linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )
    profit = -res.fun
    return profit, res


lp_profit, lp_res = solve_lp_sparse(prices, 1.0, 0.25, 0.75)
print("LP optimal profit:", lp_profit)
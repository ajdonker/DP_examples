import pandas as pd
import numpy as np
# Load the CSV with a datetime index
df = pd.read_csv(
    'GUI_ENERGY_PRICES_202412312300-202512312300.csv',
    index_col=0,
    parse_dates=True
)

# State: (SoCt,  pt)(SoCt​,pt​)

# Action: charge / hold / discharge (discrete) or continuous power level.

# Reward:   rt=pt×Pdischarge,t−pt×Pcharge,trt​=pt​×Pdischarge,t​−pt​×Pcharge,t​ (minus any cycling cost).

# Display the DataFrame to the user
print(df.columns)
# i want the column AT_price_day_ahead 

df = df['Day-ahead Price (EUR/MWh)']

print(len(df))


def solve_dp_tracking(price_series, 
                      capacity_mwh=1.0, 
                      max_rate_mw=0.25, 
                      eff=0.90, 
                      discrete_levels=41,
                      init_soc_frac=0.0):
    """
    Returns:
      - optimal_profit (float)
      - schedule_df (DataFrame with columns: soc_before, action_mw, action_type, price)
    """
    prices = price_series.values
    times  = price_series.index
    T      = len(prices)
    C, R   = capacity_mwh, max_rate_mw

    soc_grid = np.linspace(0, C, discrete_levels)
    V        = np.zeros((T+1, discrete_levels))
    policy   = np.zeros((T,   discrete_levels))  

    base_actions = np.array([-R, 0.0, +R])

    for t in reversed(range(T)):
        p = prices[t]
        for i, soc in enumerate(soc_grid):
            best_val, best_a = -np.inf, 0.0

            for a in base_actions:
                # enforce feasibility
                if a > 0 and soc >= C:   # can't charge if full
                    continue
                if a < 0 and soc <= 0:   # can't discharge if empty
                    continue

                if a > 0:   # CHARGE
                    actual  = min(a, C - soc)
                    new_soc = soc + actual * eff
                    rew     = -p * actual
                elif a < 0: 
                    actual  = min(-a, soc)
                    new_soc = soc - actual
                    rew     = +p * actual * eff
                else:     
                    actual  = 0.0
                    new_soc = soc
                    rew     = 0.0

                # map back onto grid
                j = int(np.round(new_soc/C*(discrete_levels-1)))
                val = rew + V[t+1, j]

                if val > best_val:
                    best_val, best_a = val, a

            V[t, i]      = best_val
            policy[t, i] = best_a

    start_idx      = int(np.round(init_soc_frac*(discrete_levels-1)))
    optimal_profit = V[0, start_idx]

    soc     = soc_grid[start_idx]
    soc_idx = start_idx
    records = []

    for t in range(T):
        a = policy[t, soc_idx]
        if a > 0:
            actual = min(a, C - soc)
        elif a < 0:
            actual = -min(-a, soc)  
        else:
            actual = 0.0

        # decide label
        if actual > 0:
            atype = 'charge'
        elif actual < 0:
            atype = 'discharge'
        else:
            atype = 'hold'

        records.append({
            'time':        times[t],
            'soc_before':  soc,
            'action_mw':   actual,
            'action_type': atype,
            'price':       prices[t]
        })
        soc = soc + actual * eff if actual>0 else soc + actual / eff if actual<0 else soc
        soc = max(0, min(C, soc))
        soc_idx = int(np.round(soc/C*(discrete_levels-1)))

    schedule_df = pd.DataFrame(records).set_index('time')
    return optimal_profit, schedule_df

def solve_dp(price_series, 
            capacity_mwh=1.0, 
            max_rate_mw=0.25, 
            eff=0.90, 
            discrete_levels=41,
            init_soc_frac=0.0):
    """
    Returns:
      - optimal_profit (float)
      - schedule_df (DataFrame with columns: soc_before, action_mw, action_type, price)
    """
    prices = price_series.values
    times  = price_series.index
    T      = len(prices)
    C, R   = capacity_mwh, max_rate_mw

    soc_grid = np.linspace(0, C, discrete_levels)
    V        = np.zeros((T+1, discrete_levels))
    policy   = np.zeros((T,   discrete_levels))  

    base_actions = np.array([-R, 0.0, +R])

    for t in reversed(range(T)):
        p = prices[t]
        for i, soc in enumerate(soc_grid):
            best_val, best_a = -np.inf, 0.0

            for a in base_actions:
                # enforce feasibility
                if a > 0 and soc >= C:   # can't charge if full
                    continue
                if a < 0 and soc <= 0:   # can't discharge if empty
                    continue

                if a > 0:   # CHARGE
                    actual  = min(a, C - soc)
                    new_soc = soc + actual * eff
                    rew     = -p * actual
                elif a < 0: 
                    actual  = min(-a, soc)
                    new_soc = soc - actual
                    rew     = +p * actual * eff
                else:     
                    actual  = 0.0
                    new_soc = soc
                    rew     = 0.0

                # map back onto grid
                j = int(np.round(new_soc/C*(discrete_levels-1)))
                val = rew + V[t+1, j]

                if val > best_val:
                    best_val, best_a = val, a

            V[t, i]      = best_val
            policy[t, i] = best_a

    start_idx      = int(np.round(init_soc_frac*(discrete_levels-1)))
    optimal_profit = V[0, start_idx]
    return optimal_profit


df = pd.read_csv(
    'GUI_ENERGY_PRICES_202412312300-202512312300.csv',
    index_col=0, parse_dates=True
)
prices = df['Day-ahead Price (EUR/MWh)']

# profit, schedule = solve_dp_tracking(
#     prices,
#     capacity_mwh=1.0,
#     max_rate_mw=0.25,
#     eff=0.75,
#     discrete_levels=100,
#     init_soc_frac=0.0
# )
profit = solve_dp(
    prices,
    capacity_mwh=1.0,
    max_rate_mw=0.25,
    eff=0.75,
    discrete_levels=100,
    init_soc_frac=0.0
)
print("DP optimal profit:", profit)
#print("\nFirst 10 steps:")
#print(schedule.head(10))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(
    #'GUI_ENERGY_PRICES_202412312300-202512312300.csv',
    'GUI_ENERGY_PRICES_202312312300-202412312300.csv',
    index_col=0,
    parse_dates=True
)

# State: (SoCt, t)(SoCt​,t​)

# Action: charge / hold / discharge (discrete) or continuous power level.

# Reward:   rt=pt×Pdischarge,t−pt×Pcharge,trt​=pt​×Pdischarge,t​−pt​×Pcharge,t​ (minus any cycling cost).

print(df.columns)

df = df['Day-ahead Price (EUR/MWh)']

print(len(df))


def solve_dp_tracking(price_series, 
                      capacity_mw=1.0, 
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
    C, R   = capacity_mw, max_rate_mw

    soc_grid = np.linspace(0, C, discrete_levels)
    V        = np.zeros((T+1, discrete_levels)) # state is (time_step,SoC discrete_level)
    policy   = np.zeros((T,   discrete_levels))  

    base_actions = np.array([-R, 0.0, +R])

    for t in reversed(range(T)):
        p = prices[t]
        for i, soc in enumerate(soc_grid):
            best_val, best_a = -np.inf, 0.0
            # find best state val and action for this state of charge at time t 
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
            step_profit = -p * actual
        elif a < 0:
            actual = -min(-a, soc)  
            step_profit = p * (-actual) * eff
        else:
            actual = 0.0
            step_profit = 0.0

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
            'price':       prices[t],
            'step_profit':  step_profit
        })
        soc = soc + actual * eff if actual>0 else soc + actual / eff if actual<0 else soc
        soc = max(0, min(C, soc))
        soc_idx = int(np.round(soc/C*(discrete_levels-1)))

    schedule_df = pd.DataFrame(records).set_index('time')
    return optimal_profit, schedule_df

def solve_dp(price_series, 
            capacity_mw=1.0, 
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
    C, R   = capacity_mw, max_rate_mw

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
    #'GUI_ENERGY_PRICES_202412312300-202512312300.csv',
    'GUI_ENERGY_PRICES_202312312300-202412312300.csv',
    index_col=0, parse_dates=True
)
prices = df['Day-ahead Price (EUR/MWh)']

profit, schedule = solve_dp_tracking(
    prices,
    capacity_mw=1.0,
    max_rate_mw=0.25,
    eff=0.75,
    discrete_levels=100,
    init_soc_frac=0.0
)
# profit = solve_dp(
#     prices,
#     capacity_mw=1.0,
#     max_rate_mw=0.25,
#     eff=0.75,
#     discrete_levels=100,
#     init_soc_frac=0.0
# )
print("DP optimal profit:", profit)
times = schedule.index
plt.plot(times, schedule.soc_before, label='SoC')
charge = schedule.action_mw.clip(lower=0)
dischg = -schedule.action_mw.clip(upper=0)
plt.bar(times, charge,  bottom=schedule.soc_before, color='green',   width=0.01)
plt.bar(times, dischg, bottom=schedule.soc_before, color='red',     width=0.01)
plt.legend()
plt.savefig('profit_timeline.jpg')

#print("\nFirst 10 steps:")
#print(schedule.head(10))
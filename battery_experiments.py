import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from batterydp import solve_dp_tracking  # replace with your import path
df = pd.read_csv(
    #'GUI_ENERGY_PRICES_202412312300-202512312300.csv',
    'GUI_ENERGY_PRICES_202312312300-202412312300.csv',
    index_col=0,
    parse_dates=True
)
prices = df['Day-ahead Price (EUR/MWh)']

# Compute schedule with step_profit
_, schedule = solve_dp_tracking(
    prices,
    capacity_mw=1.0,
    max_rate_mw=0.25,
    eff=0.75,
    discrete_levels=100,
    init_soc_frac=0.0
)
profit_10 = schedule['step_profit'].iloc[::10]
# Plot step_profit only
plt.figure(figsize=(10, 4))
plt.plot(schedule.index, schedule['step_profit'], alpha=0.4, label='All steps')
plt.plot(profit_10.index, profit_10.values, 'o', label='Every 10th step')
plt.xlabel("Време")
plt.ylabel("Профит по чекор (EUR)")
plt.title("Профит по чекор со маркери на секој 10-ти чекор")
plt.legend()
plt.tight_layout()
plt.savefig('step_profit_highlight_10th.jpg')
plt.show()


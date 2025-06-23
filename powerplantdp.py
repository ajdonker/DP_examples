import pandas as pd
import numpy as np

# --- 1) Load & clean power prices (15-min intervals) ---
power_dp = pd.read_csv(
    'GUI_ENERGY_PRICES_202312312300-202412312300.csv',
    index_col=0,
    parse_dates=True
)
power_dp['price_elec'] = pd.to_numeric(
    power_dp['Day-ahead Price (EUR/MWh)'], errors='coerce'
)
# drop NaNs and assume index is a proper DatetimeIndex at 15-min freq
power_dp = power_dp.dropna(subset=['price_elec'])
price_elec = power_dp['price_elec']

# --- 2) Load weekly gas, upsample to 15-min, and align ---
gas_dp = pd.read_excel(
    'pswrgvwall.xls',
    sheet_name='Data 1',
    skiprows=2,
    usecols=[0, 1],
)
gas_dp.columns = ['Date', 'Weekly_Gas_Price']
gas_dp = gas_dp[gas_dp['Date'] != 'Date'].copy()
gas_dp['Date'] = pd.to_datetime(gas_dp['Date'])
gas_dp = gas_dp.set_index('Date')
gas_2024    = gas_dp[gas_dp.index.year == 2024]
gas_hourly  = gas_2024['Weekly_Gas_Price'].resample('H').ffill()
print(gas_hourly.head(24))
# Upsample weekly → 15-min
price_gas = gas_hourly.resample('15T').ffill()

# --- 3) Define DP machinery ---
next_states = {
    0: [0, 5], 5: [6], 6: [7], 7: [8], 8: [9],
    9: [10], 10: [11], 11: [4],
    4: [4,1], 1: [2], 2: [3], 3: [0]
}
P_output = {
    0:0, 1:250, 2:175, 3:20, 4:250,
    5:0, 6:30, 7:80, 8:80, 9:110,
    10:110, 11:160
}
alpha, beta, gamma = 630.0, 7.7, 0.0004   # Btu/kWh
startup_cost = 100000.0                     # EUR

# --- 4) Backward DP ---
times = price_elec.index
T = len(times)
states = list(next_states.keys())
S = len(states)

V = np.zeros((T+1, S))
policy = np.zeros((T, S), dtype=int)
dt = 0.25  # 15 minutes = 0.25 hours

for t in range(T-1, -1, -1):
    pe = price_elec.iloc[t]
    pg = price_gas.iloc[t]
    for i, s in enumerate(states):
        best_val, best_next = -np.inf, s
        for sn in next_states[s]:
            P = P_output[sn]
            rev = pe * P * dt
            fuel_mmbtu = (alpha + beta*P + gamma*P**2) / 1e3
            cost_fuel = fuel_mmbtu * pg * P * dt
            sc = startup_cost if (sn == 5 and s != 5) else 0.0
            val = rev - cost_fuel - sc + V[t+1, states.index(sn)]
            if val > best_val:
                best_val, best_next = val, sn
        V[t, i] = best_val
        policy[t, i] = best_next

# --- 5) Forward simulate schedule ---
records = []
s = 0
for t in range(T):
    sn = policy[t, states.index(s)]
    P = P_output[sn]
    records.append({
        'time': times[t],
        'state': s,
        'dispatch_MW': P,
        'price_EUR_MWh': price_elec.iloc[t],
        'gas_price_per_MMBtu': price_gas.iloc[t],
        'step_profit': (
            price_elec.iloc[t]*P*dt
            - (alpha+beta*P+gamma*P**2)/1e3 * price_gas.iloc[t] * P * dt
            - (startup_cost if (sn==5 and s!=5) else 0)
        )
    })
    s = sn

schedule_df = pd.DataFrame(records).set_index('time')

# --- 6) Output ---
print("Optimal profit (EUR):", V[0, states.index(0)])
print(schedule_df.head())

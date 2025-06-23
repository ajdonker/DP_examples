import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
power_df = pd.read_csv(
    'GUI_ENERGY_PRICES_202312312300-202412312300.csv',
    index_col=0, parse_dates=True
)
power_df['price_elec'] = pd.to_numeric(
    power_df['Day-ahead Price (EUR/MWh)'], errors='coerce'
)
price_elec = power_df['price_elec'].dropna()
price_elec = power_df['price_elec'].groupby(level=0).max().sort_index()
full_15T = price_elec.index 
print("price_elec entries:", len(price_elec))
print("price_elec spans", price_elec.index.min(), "to", price_elec.index.max())
print("price_elec spans", price_elec.min(), "to", price_elec.index.max())


df = pd.read_excel(
    'CMO-Historical-Data-Monthly.xlsx', 
    sheet_name='Monthly Prices',
    skiprows=4,
    header=0                             
).iloc[1:]
#drop first row 
#print(df.columns)
#print("columns:", df.columns.tolist())
date_col = df.columns[0]
df = df.rename(columns={
    df.columns[0]: 'YearMonth',
})

df['Date'] = pd.to_datetime(
    df['YearMonth'].str.replace('M', '-'),
    format='%Y-%m'
)
df = df.set_index('Date')
#print(df['YearMonth'].head)
#print(df.columns.tolist())
ng2024 = df['Natural gas, Europe'].loc['2024']

# 6) Upsample monthly → hourly → 15-min
ng_hourly = ng2024.resample('h').ffill()
price_gas  = ng_hourly .resample('15min').ffill()
print(price_gas.tail)
print("price_gas spans", price_gas.index.min(), "to", price_gas.index.max())
print("price_gas spans", price_gas.min(), "to", price_gas.max())

print("price_gas  entries:", len(price_gas))
print("price_gas spans", price_gas.index.min(), "to", price_gas.index.max())
# --- 3) Define DP start-up/shutdown machine for a single plant ---
next_states = {
    0: [0, 6],  # OFF -> OFF or U1
    6: [7], 7: [8], 8: [9],
    9: [10], 10: [11], 11: [4],  # U1->U7->ON
    4: [4,12,2],  # ON -> ON or D1
    #1: [2], 2: [3], 3: [0],       # D1->D3->OFF
    2: [3], 3: [0],
    12: [4,12,13],
    13: [12,13,14],
    14: [13,14]
}
state_names = {
    0: 'OFF',
    1: 'Shutdown D1',
    2: 'Shutdown D2',
    3: 'Shutdown D3',
    4: 'ON',
    #5: 'Startup U1',
    6: 'Startup U2',
    7: 'Startup U3',
    8: 'Startup U4',
    9: 'Startup U5',
    10: 'Startup U6',
    11: 'Startup U7',
    12: 'Ramp up R1',
    13: 'Ramp up R2',
    14: 'Max power'
}
P_output = {
    0: 0,1: 250, 2: 175, 3: 20, 4: 250,
    #5: 0,    
    6: 30,  7: 80,  8: 80,  9: 110,
    10: 110, 11: 160, 12: 400, 13: 550, 14: 700
}
P_max = 700
def dp_plant_tracking(price_elec,price_gas,startup_cost=100000,alpha=630,beta=7.7,gamma=0.0004,partial_load_penalty=0.5):
    times = price_gas.index
    T = len(times)
    states = list(next_states.keys())
    S = len(states)
    V = np.zeros((T+1, S))
    dt = 0.25  # hours per step
    policy = np.zeros((T, S), dtype=int)
    for t in range(T-1, -1, -1):
        pe = price_elec.iloc[t]
        pg = price_gas.iloc[t] * 2.97 # convert dollar/MMbtu to eur/MWh
        for i, s in enumerate(states):
            best_val = -np.inf
            best_next = s
            for sn in next_states[s]:
                P = P_output[sn]
                rev = pe * P * dt
                # Fuel use: convert MMBtu/kWh to MMBtu/MWh
                fuel_use_mmbtu = (alpha + beta*P + gamma*P**2) / 1e3
                energy_mwh = P * dt
                cost_fuel = fuel_use_mmbtu * pg * energy_mwh
                # Optional : efficiency penalty to low power states  
                if s < 11:
                    cost_fuel = cost_fuel * (1 + partial_load_penalty*(1 - (P/P_max)))
                sc = startup_cost if (sn == 6 and s != 6) else 0.0
                val = rev - cost_fuel - sc + V[t+1, states.index(sn)]
                if val > best_val:
                    best_val, best_next = val, sn
            V[t, i] = best_val
            policy[t, i] = best_next

    # --- 5) Forward simulate to build schedule ---
    records = []
    s = 0  # start OFF
    for t in range(T):
        sn = policy[t, states.index(s)]
        P = P_output[sn]
        records.append({
            'time': times[t],
            'state': s,
            'next_state': sn,
            'dispatch_MW': P,
            'price_EUR_MWh': price_elec.iloc[t],
            'gas_price_per_MWh': price_gas.iloc[t] * 2.97, 
            'step_profit_EUR': (price_elec.iloc[t] * P - 
                                (alpha + beta*P + gamma*P**2)/1e3 * price_gas.iloc[t] * P -
                                (startup_cost if (sn==5 and s!=5) else 0))
        })
        s = sn
    schedule = pd.DataFrame(records).set_index('time')    
    return V,schedule
def dp_plant_profit(price_elec,price_gas,startup_cost=100000,alpha=630,beta=7.7,gamma=0.0004,
                    partial_load_penalty=0.5):
    times = price_gas.index
    T = len(times)
    states = list(next_states.keys())
    S = len(states)
    V = np.zeros((T+1, S))
    dt = 0.25  # hours per step
    policy = np.zeros((T, S), dtype=int)
    for t in range(T-1, -1, -1):
        pe = price_elec.iloc[t]
        pg = price_gas.iloc[t] * 2.97 # convert dollar/MMbtu to eur/MWh
        for i, s in enumerate(states):
            best_val = -np.inf
            best_next = s
            for sn in next_states[s]:
                P = P_output[sn]
                rev = pe * P * dt
                # Fuel use: convert MMBtu/kWh to MMBtu/MWh
                fuel_use_mmbtu = (alpha + beta*P + gamma*P**2) / 1e3
                energy_mwh = P * dt
                cost_fuel = fuel_use_mmbtu * pg * energy_mwh
                if s < 11:
                    cost_fuel = cost_fuel * (1 + partial_load_penalty*(1 - (P/P_max)))
                sc = startup_cost if (sn == 6 and s != 6) else 0.0
                val = rev - cost_fuel - sc + V[t+1, states.index(sn)]
                if val > best_val:
                    best_val, best_next = val, sn
            V[t, i] = best_val
            policy[t, i] = best_next  
    return V[0, states.index(0)]



startup_cost = 100000.0  # EUR per start
# Heat-rate (Btu/kWh) coefficients for fuel use
alpha, beta, gamma = 630.0, 7.7, 0.0004

# --- 4) Backward DP ---
V,schedule_df = dp_plant_tracking(price_elec,price_gas,startup_cost,alpha,beta,gamma)
step_profits = schedule_df['step_profit_EUR']
plt.figure(figsize=(8, 4))
plt.hist(
    step_profits, 
    bins=50, 
    range=(0, 100000), 
    edgecolor='black'
)
plt.xlabel('Step Profit Magnitude (EUR)')
plt.ylabel('Frequency')
plt.title('Distribution of Step Profits')
plt.tight_layout()
plt.savefig('profit_distr.jpg')

counts = schedule_df['state'].value_counts().sort_index()
labels = [state_names[s] for s in counts.index]
values = counts.values

plt.figure(figsize=(10, 6))
plt.bar(labels, values)
plt.xticks(rotation=45, ha='right')
plt.xlabel('State')
plt.ylabel('Number of 15-minute Intervals')
plt.title('Time Spent in Each State')
plt.tight_layout()
plt.savefig('states_with_eff_penalty.jpg')

print("Optimal total income (EUR):", V[0, 0])
print(schedule_df.tail(50))
fig , ax = plt.subplots(figsize=(12,4))
ax.plot(
    schedule_df.index,
    schedule_df['price_EUR_MWh'],
    label='Elec Price (€/MWh)',
    color='C0',
    linewidth=1
)

# Gas price (converted)
ax.plot(
    schedule_df.index,
    schedule_df['gas_price_per_MWh'],
    label='Gas Price (€/MWh)',
    color='C1',
    linewidth=1
)
# ax2.set_ylabel('$/MMBtu')

# Legends
ax.fill_between(
    schedule_df.index,
    0,
    schedule_df['dispatch_MW'],
    where=schedule_df['dispatch_MW']>0,
    color='green',
    alpha=0.3,
    label='Dispatch (MW)'
)

ax.set_ylabel('€/MWh & MW')
ax.set_xlabel('Time')
ax.set_ylim(bottom=0)
ax.legend(loc='upper right')
plt.title('Dispatch vs. Elec & Gas Prices (Same Scale)')
plt.tight_layout()
plt.savefig('states_ramp.jpg')


schedule_costs = np.linspace(2000000,10000000,10)
profits = [dp_plant_profit(price_elec, price_gas, sc) for sc in schedule_costs]
for income in profits:
    print(f'income is {income}')
plt.figure(figsize=(8,4))
plt.plot(schedule_costs, profits, marker='o')
plt.xlabel('Startup Cost (EUR)')
plt.ylabel('Optimal income (EUR)')
plt.title('DP income vs. Startup Cost')
plt.grid(True)
plt.tight_layout()
plt.savefig('profit_vs_startup_cost.jpg')
import pandas as pd

# --- 1) Properly parse your power_dp index into a DatetimeIndex ---
power_dp = pd.read_csv(
    'GUI_ENERGY_PRICES_202312312300-202412312300.csv',
    index_col=0,
    parse_dates=False  # read raw strings
)
# Extract the start time before the " - " and parse:
power_dp.index = pd.to_datetime(
    power_dp.index.to_series().str.split(' - ').str[0],
    dayfirst=True
)
price_elec = pd.to_numeric(
    power_dp['Day-ahead Price (EUR/MWh)'], errors='coerce'
).dropna()

print("After parsing:")
print("  price_elec length:", len(price_elec))
print("  price_elec period:", price_elec.index.min(), "to", price_elec.index.max())

# --- 2) Ensure your gas_15min is a 15-min DatetimeIndex series (as before) ---
# (Assuming you’ve already got ng_15min from the TTF sheet)
gas_15min = ng_15min  # should be a pd.Series with a proper DatetimeIndex at 15T

print("Before aligning:")
print("  gas_15min length:", len(gas_15min))
print("  gas_15min period:", gas_15min.index.min(), "to", gas_15min.index.max())

# --- 3) Reindex gas onto exactly the same timestamps as elec ---
gas_aligned = gas_15min.reindex(price_elec.index, method='ffill')
print("After aligning:")
print("  gas_aligned length:", len(gas_aligned))
print("  # gaps (NaN):", gas_aligned.isna().sum())

# Now price_elec.index == gas_aligned.index, and len(...) is the same.

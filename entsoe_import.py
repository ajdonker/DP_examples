from entsoe import EntsoePandasClient
import pandas as pd

client = EntsoePandasClient(api_key="YOUR_API_KEY")

# 3) Define your target – the “EIC” code of your generator or BA
resource   = "10Y1001A1001A83F"   # example EIC for Austria
start, end = pd.Timestamp("2010-01-01", tz="Europe/Brussels"), pd.Timestamp("2020-12-31", tz="Europe/Brussels")

# 4) This single call will internally page by year for you:
df = client.query_day_ahead_prices(resource, start=start, end=end)

print(df)  # a pandas Series indexed to the full 11-year span
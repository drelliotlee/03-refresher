import pandas as pd

# Read the CSV, parse the first column as dates (format: DD-MM-YYYY)
df = pd.read_csv('files/climate_data.csv', parse_dates=[0], dayfirst=True)

# Filter for stations 96741, 96745, 96747
station_ids = [96741, 96745, 96747]
df = df[df['station_id'].isin(station_ids)]

date_col = df.columns[0]
dates = df[date_col]

# Sort dates just in case
dates_sorted = dates.sort_values().reset_index(drop=True)

# Generate expected date range
expected = pd.date_range(start=dates_sorted.iloc[0], end=dates_sorted.iloc[-1], freq='D')

# Check for missing and duplicate dates
missing = expected.difference(dates_sorted)
duplicates = dates_sorted[dates_sorted.duplicated()]

print(f"Total rows: {len(dates_sorted)}")
print(f"Expected rows: {len(expected)}")
print(f"Missing dates: {len(missing)}")
if not missing.empty:
    print(missing)
print(f"Duplicate dates: {len(duplicates)}")
if not duplicates.empty:
    print(duplicates)

if len(dates_sorted) != len(expected) or not missing.empty or not duplicates.empty:
    print("There is something strange with the date column.")
else:
    print("Date column looks good: one row per day, no missing or duplicate dates.")

print("\nFiltered DataFrame (final df):")
print(df)

# Write the final filtered DataFrame to a CSV file
df.to_csv('files/climate_filtered_stations.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# 1. Read the CSV file, parse dates in the first column
df = pd.read_csv('files/climate_data.csv', parse_dates=['date'], dayfirst=True)

# Only keep rows for stations 96741, 96745, 96747
station_ids = [96741, 96745, 96747]
df = df[df['station_id'].isin(station_ids)].copy()

# 2. Extract day and month as a new column (e.g., "01-01")
df['DayMonth'] = df['date'].dt.strftime('%d-%m')

# Use columns RR and ss by name for clarity
rr_col = 'RR'
ss_col = 'ss'

# 3. Group by day and month, calculate mean for columns RR and ss
daily_means = df.groupby('DayMonth')[[rr_col, ss_col]].mean()

# --- Fix sorting for plotting ---
# Convert DayMonth to datetime using a dummy year (e.g., 2000)
def safe_parse(dm):
    try:
        return datetime.strptime(dm, "%d-%m")
    except ValueError:
        return None  # skip invalid dates like 31-02

daily_means = daily_means.copy()
daily_means['plot_date'] = daily_means.index.map(safe_parse)
daily_means = daily_means.dropna(subset=['plot_date'])
daily_means = daily_means.sort_values('plot_date')

# Add 7-day rolling means as explicit columns to daily_means
daily_means['RR_7d_rolling_mean'] = daily_means[rr_col].rolling(window=7, center=True, min_periods=1).mean()
daily_means['ss_7d_rolling_mean'] = daily_means[ss_col].rolling(window=7, center=True, min_periods=1).mean()

# 4. Print or save the result
print(daily_means)
daily_means.drop(columns=['plot_date']).to_csv('files/climate_daily_means.csv')

# Write out table of dates and rolling means to a CSV
daily_means[['plot_date', 'RR_7d_rolling_mean', 'ss_7d_rolling_mean']].to_csv(
    'files/climate_rolling_means.csv', index=False
)

# Save just the date column to a new CSV for inspection
df[['date']].to_csv('files/climate_dates_only.csv', index=False)

# Write the final filtered DataFrame to a CSV file
df.to_csv('files/climate_filtered_stations.csv', index=False)

# 5. Plotting with dual y-axes and fixed x-axis order
fig, ax1 = plt.subplots(figsize=(12, 5))

color1 = 'tab:blue'
color2 = 'tab:orange'

ax1.set_xlabel('Day-Month')
ax1.set_ylabel(rr_col, color=color1)
ax1.plot(daily_means['plot_date'], daily_means[rr_col], color=color1, label=rr_col)
# Plot 7-day rolling mean for RR
ax1.plot(
    daily_means['plot_date'],
    daily_means[rr_col].rolling(window=7, center=True, min_periods=1).mean(),
    color='black', linestyle='--', linewidth=2, label=f'{rr_col} 7-day rolling mean'
)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
ax2.set_ylabel(ss_col, color=color2)
ax2.plot(daily_means['plot_date'], daily_means[ss_col], color=color2, label=ss_col)
# Plot 7-day rolling mean for ss
ax2.plot(
    daily_means['plot_date'],
    daily_means[ss_col].rolling(window=7, center=True, min_periods=1).mean(),
    color='red', linestyle='--', linewidth=2, label=f'{ss_col} 7-day rolling mean'
)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Daily Mean Climate Data for Stations 96741, 96745, 96747')

# Set major ticks to the 1st of each month and format as "DD-MM"
ax1.xaxis.set_major_locator(mdates.MonthLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

fig.autofmt_xdate()
plt.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()

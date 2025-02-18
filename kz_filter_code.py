

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = 'Tesla_data_raw.csv'
data = pd.read_csv(file_path, parse_dates=True, index_col='Date')

# Remove the dollar sign and convert to float
data['High'] = data['High'].replace('[\$,]', '', regex=True).astype(float)

# Applying KZ filter
def kz_filter(series, m, k):
    for i in range(k):
        series = series.rolling(window=m, min_periods=1, center=True).mean()
    return series

# Parameters
m_30 = 30   # Window size for 30 days
m_180 = 180 # Window size for 180 days
m_365 = 365 # Window size for 365 days
k = 5       # Number of iterations for 30 days
k_3 = 3     # Number of iterations for 180 and 365 days

# Apply KZ filter
data['kz1_30'] = kz_filter(data['High'], m_30, 1)
data['kz2_30'] = kz_filter(data['High'], m_30, 2)
data['kz3_30'] = kz_filter(data['High'], m_30, 3)
data['kz4_30'] = kz_filter(data['High'], m_30, 4)
data['kz5_30'] = kz_filter(data['High'], m_30, k)
data['kz_180'] = kz_filter(data['High'], m_180, k_3)
data['kz_365'] = kz_filter(data['High'], m_365, k_3)

# Plotting the data
plt.figure(figsize=(14, 8))
plt.plot(data.index, data['High'], label='High')
plt.plot(data.index, data['kz1_30'], label='kz1_30')
plt.plot(data.index, data['kz2_30'], label='kz2_30')
plt.plot(data.index, data['kz3_30'], label='kz3_30')
plt.plot(data.index, data['kz4_30'], label='kz4_30')
plt.plot(data.index, data['kz5_30'], label='kz5_30')
plt.plot(data.index, data['kz_180'], label='kz_180')
plt.plot(data.index, data['kz_365'], label='kz_365')
plt.title('KZ Filtered Moving Averages')
plt.legend()
plt.show()

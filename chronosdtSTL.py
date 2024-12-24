import pickle
import datasets
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

period_map = {
    'ercot': 24,                      # Daily cycle (hourly data)
    'm4_daily': 7,                    # Weekly cycle (daily data)
    'm4_hourly': 24,                  # Daily cycle (hourly data)
    'm4_monthly': 12,                 # Yearly cycle (monthly data)
    'm4_quarterly': 4,                # Quarterly cycle within a year
    'm4_yearly':2,
    'm4_weekly': 52,                  # Yearly cycle (weekly data)
    'mexico_city_bikes': 24,          # Daily cycle (hourly bike usage)
    'monash_australian_electricity': 24,   # Daily cycle (hourly electricity usage)
    'monash_car_parts': 12,           # Yearly cycle (likely monthly data)
    'monash_cif_2016': 7,             # Weekly cycle (daily data)
    'monash_covid_deaths': 7,         # Weekly cycle (daily COVID data)
    'monash_electricity_hourly': 24,  # Daily cycle (hourly electricity data)
    'monash_electricity_weekly': 52,  # Yearly cycle (weekly data)
    'monash_fred_md': 12,             # Yearly cycle (monthly data)
    'monash_hospital': 7,             # Weekly cycle (daily data)
    'monash_kdd_cup_2018': 24,        # Daily cycle (hourly data)
    'monash_london_smart_meters': 24, # Daily cycle (hourly meter readings)
    'monash_m1_monthly': 12,          # Yearly cycle (monthly data)
    'monash_m1_quarterly': 4,         # Quarterly cycle within a year
    'monash_m1_yearly': 2,            # Set to 2 to meet minimum (originally yearly)
    'monash_m3_monthly': 12,          # Yearly cycle (monthly data)
    'monash_m3_quarterly': 4,         # Quarterly cycle within a year
    'monash_m3_yearly': 2,            # Set to 2 to meet minimum (originally yearly)
    'monash_nn5_weekly': 52,          # Yearly cycle (weekly data)
    'monash_pedestrian_counts': 7,    # Weekly cycle (daily pedestrian counts)
    'monash_rideshare': 24,           # Daily cycle (hourly rideshare data)
    'monash_saugeenday': 7,           # Weekly cycle (daily water levels)
    'monash_temperature_rain': 30,   # Yearly cycle (daily temperature/rainfall)
    'monash_tourism_monthly': 12,     # Yearly cycle (monthly tourism data)
    'monash_tourism_quarterly': 4,    # Quarterly cycle within a year
    'monash_tourism_yearly': 2,       # Set to 2 to meet minimum (originally yearly)
    'monash_weather': 30,            # Yearly cycle (daily weather data)
    'nn5': 7,                         # Weekly cycle (daily data)
    'ushcn_daily': 30,               # Yearly cycle (daily climate data)
    'wind_farms_daily': 7,            # Weekly cycle (daily wind data)
    'wind_farms_hourly': 24           # Daily cycle (hourly wind data)
}


all_names = [
   # 'ercot', 'm4_daily', 'm4_hourly', 'm4_monthly', 'm4_quarterly', 'm4_yearly','m4_weekly',
     'mexico_city_bikes', 
    'monash_australian_electricity', 'monash_car_parts', 'monash_cif_2016', 'monash_covid_deaths',
    'monash_electricity_hourly', 'monash_electricity_weekly', 'monash_fred_md', 'monash_hospital', 'monash_kdd_cup_2018',
    'monash_london_smart_meters', 'monash_m1_monthly', 'monash_m1_quarterly', 'monash_m1_yearly', 
    'monash_m3_monthly',
    # 'monash_m3_quarterly', 'monash_m3_yearly', 'monash_nn5_weekly', 
    'monash_pedestrian_counts', 'monash_rideshare',
    'monash_saugeenday', 
    # 'monash_temperature_rain', 'monash_tourism_monthly', 'monash_tourism_quarterly',
    #'monash_tourism_yearly', 
    'monash_weather', 'nn5', 'ushcn_daily', 'wind_farms_daily', 'wind_farms_hourly'
]


datasets_dict = {}

def read_dataset(dataset_name):
    print(f"Loading dataset: {dataset_name}")
    try:
        ds = datasets.load_dataset("autogluon/chronos_datasets", dataset_name, split="train")
        ds.set_format("numpy")
        return ds
    except Exception as e:
        print(f"Failed to load dataset {dataset_name}: {e}")
        return None

def perform_stl_decomposition(series, dataset_name):
    period = period_map.get(dataset_name, 24)
    stl = STL(series, period=period)
    result = stl.fit()
    return result.trend, result.seasonal, result.resid

def create_sliding_windows(dataset_name, ds, window_size=432, x_size=336, y_size=96):
    windows_dict = {}
    for entry in ds:
        data_id = entry['id']
        timestamps = entry['timestamp']

        # import numpy as np

        # 假设targets是一个numpy数组
        # 假设targets是一个numpy数组
        targets = np.array(entry['target'])
        if len(targets) < window_size:
            continue
        # 计算均值，忽略NaN值
        mean = np.nanmean(targets)
        std = np.nanstd(targets)

        # 用均值填充NaN值
        targets = np.where(np.isnan(targets), mean, targets)

        # 进行z-score标准化
        z_scores = (targets - mean) / std
        
        targets = z_scores #entry['target']

        trend, seasonal, resid = perform_stl_decomposition(targets, dataset_name)

        windows_list = []

        for i in range(0, len(targets) - window_size + 1):
            window_timestamps = timestamps[i:i + window_size]
            window_targets = targets[i:i + window_size]

            x = {'timestamp': window_timestamps[:x_size], 'target': window_targets[:x_size]}
            y = {'timestamp': window_timestamps[x_size:], 'target': window_targets[x_size:]}
            # x = window_targets[:x_size]
            # y = window_targets[x_size:]
            x_trend = trend[i:i + x_size]
            x_seasonal = seasonal[i:i + x_size]
            x_resid = resid[i:i + x_size]
            y_trend = trend[i + x_size:i + window_size]
            y_seasonal = seasonal[i + x_size:i + window_size]
            y_resid = resid[i + x_size:i + window_size]


            # x_trend, x_seasonal, x_resid = perform_stl_decomposition(x['target'], dataset_name)
            # y_trend, y_seasonal, y_resid = perform_stl_decomposition(y['target'], dataset_name)

            windows_list.append({
                'x': x, 'y': y,
                'x_trend': x_trend, 'x_seasonal': x_seasonal, 'x_resid': x_resid,
                'y_trend': y_trend, 'y_seasonal': y_seasonal, 'y_resid': y_resid,
                'mean': mean, 'std': std
            })
        if len(windows_list) > 0:
            windows_dict[data_id] = windows_list
        else:

            print(f"Skipping data_id: {data_id} due to insufficient data")
            windows_list.append({
                'x': {'timestamp': np.zeros(x_size), 'target': np.zeros(x_size)},
                'y': {'timestamp': np.zeros(y_size), 'target': np.zeros(y_size)},
                'x_trend': np.zeros(x_size), 'x_seasonal': np.zeros(x_size), 'x_resid': np.zeros(x_size),
                'y_trend': np.zeros(y_size), 'y_seasonal': np.zeros(y_size), 'y_resid': np.zeros(y_size),
                'mean': 0, 'std': 0
            })

    return windows_dict


all_data_windows = {}

for dataset_name in all_names:
    print(f"Processing dataset: {dataset_name}")
    ds = read_dataset(dataset_name)
    if ds is not None:
        sliding_windows = create_sliding_windows(dataset_name, ds)
        # if len(sliding_windows) > 0:
        # all_data_windows[dataset_name] = sliding_windows

        with open(f"datasets/chronos_2/{dataset_name}.pkl", "wb") as f:
            pickle.dump(sliding_windows, f)


# with open("datasets/all_datasets_windows.pkl", "wb") as f:
#     pickle.dump(all_data_windows, f)

print("All datasets processed and saved to all_datasets_windows.pkl")
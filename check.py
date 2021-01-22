import copy

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import netCDF4
import tqdm as tqdm
from netCDF4 import Dataset

path_confirmed_global = "data/time_series_covid19_confirmed_global.csv"
path_confirmed_usa = "data/time_series_covid19_confirmed_US.csv"
path_deaths_global = "data/time_series_covid19_deaths_global.csv"
path_deaths_usa = "data/time_series_covid19_deaths_US.csv"
path_recovery_global = "data/time_series_covid19_recovered_global.csv"

def confirmed_per_day():
    ## CONFIRMED global
    df_confirmed_global = pd.read_csv(path_confirmed_global)

    df_confirmed_global_temp = pd.concat([df_confirmed_global.iloc[:, 1:2], df_confirmed_global.iloc[:, 4:]], axis=1)
    # check if there are any nans
    df_confirmed_global_temp = df_confirmed_global_temp.fillna(0)
    # sum by country
    df_country_confirmed = df_confirmed_global_temp.groupby('Country/Region').agg('sum')

    df_world_confirmed = df_country_confirmed.T

    df_world_confirmed.index.name = "Date"
    df_world_confirmed.index = pd.to_datetime(df_world_confirmed.index)
    return df_world_confirmed

def deaths_per_day():
    ## DEaTHS global
    df_deaths_global = pd.read_csv(path_deaths_global)
    # print(df_deaths_global.head())

    df_deaths_global_temp = pd.concat([df_deaths_global.iloc[:, 1:2], df_deaths_global.iloc[:, 4:]], axis=1)
    # check if there are any nans
    df_deaths_global_temp = df_deaths_global_temp.fillna(0)
    # sum by country
    df_country_deaths = df_deaths_global_temp.groupby('Country/Region').agg('sum')

    df_world_deaths = df_country_deaths.T

    df_world_deaths.index.name = "Date"
    df_world_deaths.index = pd.to_datetime(df_world_deaths.index)
    return df_world_deaths

def recovery_per_day_no_usa():
    ## recovery global
    df_recovery_global = pd.read_csv(path_recovery_global)

    df_recovery_global_temp = pd.concat([df_recovery_global.iloc[:, 1:2], df_recovery_global.iloc[:, 4:]], axis=1)
    # check if there are any nans
    df_recovery_global_temp = df_recovery_global_temp.fillna(0)
    # sum by country (no USA yet)
    df_country_recovery = df_recovery_global_temp.groupby('Country/Region').agg('sum')
    df_country_recovery = df_country_recovery.T
    df_country_recovery.index.name = "Date"
    df_country_recovery.index = pd.to_datetime(df_country_recovery.index)

    return df_country_recovery

def lon_lat_countries():
    df_countries = pd.read_csv(path_confirmed_global)
    df_countries = df_countries.iloc[:, 1:4]
    df_c = df_countries.groupby('Country/Region').agg('mean')
    return df_c

def weather():
    weather_min = Dataset('./weather/TerraClimate_tmin_2018.nc')
    weather_max = Dataset('./weather/TerraClimate_tmax_2018.nc')
    # w_max_lat = weather_max['lat'][:]
    # w_max_lon = weather_max['lon'][:]
    # pos_lon = 75.45
    # geo_idx = (np.abs(pos_lon - w_max_lon)).argmin()
    # print(geo_idx)
    return weather_min, weather_max

def GET_IDX(elem,vect):
    idx = (np.abs(elem-vect)).argmin()
    return idx

def GET_TEMP(elem_x, elem_y,w_min,w_max,date):
    month = date.month - 1
    temp_min = w_min[month][int(elem_x)][int(elem_y)]
    temp_max = w_max[month][int(elem_x)][int(elem_y)]
    temp_avg = (temp_max + temp_min) / 2.0
    #return temp_min
    return temp_avg

def main():
    # confirmed cases
    df_world_confirmed = confirmed_per_day()
    df_world_confirmed_14 = df_world_confirmed.shift(14,fill_value=0)
    ## deaths
    df_world_deaths = deaths_per_day()
    df_world_deaths_14 = df_world_deaths.shift(14,fill_value=0)
    df_usa_recovered = df_world_confirmed_14['US'] - df_world_deaths_14['US']
    df_world_recovered = recovery_per_day_no_usa()
    df_world_recovered['US']=df_usa_recovered
    df_world_active = df_world_confirmed - df_world_recovered - df_world_deaths
    df_world_recovered_monthly = df_world_recovered.resample('1M').sum()
    df_world_deaths_monthly = df_world_deaths.resample('1M').sum()
    df_mortality_rate = df_world_deaths_monthly/df_world_recovered_monthly
    df_mortality_rate = df_mortality_rate.replace([np.inf, -np.inf], np.nan)
    df_mortality_rate = df_mortality_rate.fillna(0)
    #print(df_mortality_rate['Poland'])

    df_M = df_world_active.rolling(7).sum().fillna(0)

    df_M_5 = df_M.shift(5,fill_value=0)
    df_R = (df_M/df_M_5).fillna(0)
    w_min, w_max = weather()
    lat_lon = lon_lat_countries()
    lat_lon['Lat_idx'] = lat_lon.apply(lambda row: GET_IDX(row['Lat'],w_min['lat'][:]),axis=1)
    lat_lon['Long_idx'] = lat_lon.apply(lambda row: GET_IDX(row['Long'],w_min['lon'][:]),axis=1)

    #print(lat_lon)

    temps = pd.DataFrame().reindex_like(df_world_deaths_monthly)
    avg_temps = temps.T
    dates = avg_temps.columns

    w_min_np = w_min.variables['tmin'][:]
    w_max_np = w_max.variables['tmax'][:]

    for dt in dates:
        avg_temps[dt] = lat_lon.apply(lambda row: GET_TEMP(row['Lat_idx'],row['Long_idx'],w_min_np,w_max_np,dt),axis=1)
    print(avg_temps)



if __name__=='__main__':
    main()

















































































































































































































































































































































































































































































































































    # Nobody expected that
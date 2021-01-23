import copy
import time

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

    df_confirmed_global_temp = pd.concat([df_confirmed_global.iloc[:, 0:2], df_confirmed_global.iloc[:, 4:]], axis=1)
    df_confirmed_global_temp['Country/Region'] = df_confirmed_global_temp['Country/Region'] + ('_' + df_confirmed_global_temp['Province/State']).fillna('')
    df_confirmed_global_temp.drop(columns={'Province/State'}, inplace=True)
    df_confirmed_global_temp = df_confirmed_global_temp.set_index('Country/Region')
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

    df_deaths_global_temp = pd.concat([df_deaths_global.iloc[:, 0:2], df_deaths_global.iloc[:, 4:]], axis=1)
    df_deaths_global_temp['Country/Region'] = df_deaths_global_temp['Country/Region'] + ('_' + df_deaths_global_temp['Province/State']).fillna('')
    df_deaths_global_temp.drop(columns={'Province/State'}, inplace=True)
    df_deaths_global_temp = df_deaths_global_temp.set_index('Country/Region')
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

    df_recovery_global_temp = pd.concat([df_recovery_global.iloc[:, 0:2], df_recovery_global.iloc[:, 4:]], axis=1)
    df_recovery_global_temp['Country/Region'] = df_recovery_global_temp['Country/Region'] + ('_' + df_recovery_global_temp['Province/State']).fillna('')
    df_recovery_global_temp.drop(columns={'Province/State'}, inplace=True)
    df_recovery_global_temp = df_recovery_global_temp.set_index('Country/Region')
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
    df_countries = df_countries.iloc[:, 0:4]
    df_countries['Country/Region'] = df_countries['Country/Region'] + ('_' + df_countries['Province/State']).fillna('')
    df_countries.drop(columns={'Province/State'}, inplace=True)
    df_countries = df_countries.set_index('Country/Region')
    #df_c = df_countries.groupby('Country/Region').agg('mean')
    #return df_c
    return df_countries

def weather():
    weather_min = Dataset('./weather/TerraClimate_tmin_2018.nc')
    weather_max = Dataset('./weather/TerraClimate_tmax_2018.nc')
    return weather_min, weather_max

def GET_IDX(elem,vect):
    idx = (np.abs(elem-vect)).argmin()
    return idx

def GET_TEMP(elem_x, elem_y,w_min,w_max,date):
    month = date.month - 1
    temp_min = w_min[month][int(elem_x)][int(elem_y)]
    temp_max = w_max[month][int(elem_x)][int(elem_y)]
    temp_avg = (temp_max + temp_min) / 2.0
    return temp_min
    #return temp_avg

def GET_SINGLE_TEMP(elem_x, elem_y,w_data,date):
    month = date.month - 1
    temp = w_data[month][int(elem_x)][int(elem_y)]
    #print(temp)
    temp = temp * 1.0
    return temp

def custom_resampler(array_like):
    return array_like[-1]-array_like[0]



def main():
    # confirmed cases
    df_world_confirmed = confirmed_per_day()
    df_world_confirmed_14 = df_world_confirmed.shift(14,fill_value=0)
    # deaths
    df_world_deaths = deaths_per_day()
    df_world_deaths_14 = df_world_deaths.shift(14,fill_value=0)
    # recovered
    # USA case
    df_usa_recovered = df_world_confirmed_14['US'] - df_world_deaths_14['US']
    df_world_recovered = recovery_per_day_no_usa()
    df_world_recovered['US']=df_usa_recovered

    # Canada case
    canada_names = [col for col in df_world_confirmed if col.startswith('Canada')]
    df_canada_recovered = df_world_confirmed_14[canada_names] - df_world_deaths_14[canada_names]
    df_world_recovered = df_world_recovered.drop(columns={'Canada'})
    df_world_recovered[canada_names]=df_canada_recovered

    df_world_recovered = df_world_recovered.reindex(sorted(df_world_recovered.columns), axis=1)

    # active
    df_world_active = df_world_confirmed - df_world_recovered - df_world_deaths

    df_world_active = df_world_active.dropna(axis=1)

    df_world_recovered_monthly = df_world_recovered.resample('1M').apply(custom_resampler)
    df_world_deaths_monthly = df_world_deaths.resample('1M').apply(custom_resampler)

    df_mortality_rate_month = df_world_deaths_monthly/df_world_recovered_monthly
    df_mortality_rate_month = df_mortality_rate_month.replace([np.inf, -np.inf], np.nan)
    df_mortality_rate_month = df_mortality_rate_month.fillna(0)

    df_M = df_world_active.rolling(7).mean().fillna(0)

    df_M_5 = df_M.shift(5,fill_value=0)
    df_R = (df_M/df_M_5)
    df_R = df_R.replace([np.inf, -np.inf], np.nan)

    df_R_max = df_R.max(axis=0)
    df_R_norm = df_R / df_R_max
    df_R_norm = df_R_norm.resample('1M').mean()
    df_R_norm = df_R_norm.T
    #print(df_R_norm)

    # w_min, w_max = weather()
    # lat_lon = lon_lat_countries()
    # lat_lon['Lat_idx'] = lat_lon.apply(lambda row: GET_IDX(row['Lat'],w_min['lat'][:]),axis=1)
    # lat_lon['Long_idx'] = lat_lon.apply(lambda row: GET_IDX(row['Long'],w_min['lon'][:]),axis=1)
    #
    # temps = pd.DataFrame().reindex_like(df_world_deaths_monthly)
    # temps_min = temps.T
    # temps_max = temps.T
    #
    # dates = temps_min.columns
    #
    # w_min_np = w_min.variables['tmin'][:]
    #
    # for dt in dates:
    #     temps_min[dt] = lat_lon.apply(lambda row: GET_SINGLE_TEMP(row['Lat_idx'],row['Long_idx'],w_min_np,dt),axis=1)
    # w_min_np = None
    # #
    # w_max_np = w_max.variables['tmax'][:]
    # for dt in dates:
    #     temps_max[dt] = lat_lon.apply(lambda row: GET_SINGLE_TEMP(row['Lat_idx'], row['Long_idx'], w_max_np, dt),axis=1)
    # w_max_np = None
    #
    # avg_temp = (temps_min + temps_max)/2.0
    # print(avg_temp)

    buckets = pd.DataFrame().reindex_like(df_R_norm)
    buckets = buckets.T
    buckets.insert(0, '<0', [0 for i in range(len(buckets.index))])
    buckets.insert(1, '0-10', [0 for i in range(len(buckets.index))])
    buckets.insert(2, '10-20', [0 for i in range(len(buckets.index))])
    buckets.insert(3, '20-30', [0 for i in range(len(buckets.index))])
    buckets.insert(4, '>30', [0 for i in range(len(buckets.index))])
    buckets = buckets.iloc[:,0:5]
    buckets = buckets.T
    print(buckets)
    buckets.loc['<0']['2020-01-31'] += 1
    print(buckets)






if __name__=='__main__':
    main()

















































































































































































































































































































































































































































































































































    # Nobody expected that
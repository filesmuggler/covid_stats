import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import netCDF4
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

def main():
    ## confirmed cases
    df_world_confirmed = confirmed_per_day()
    df_world_confirmed_14 = df_world_confirmed.shift(14,fill_value=0)
    ## deaths
    df_world_deaths = deaths_per_day()
    df_world_deaths_14 = df_world_deaths.shift(14,fill_value=0)
    df_usa_recovered = df_world_confirmed_14['US'] - df_world_deaths_14['US']
    df_world_recovered = recovery_per_day_no_usa()
    df_world_recovered['US']=df_usa_recovered
    df_world_active = df_world_confirmed - df_world_recovered - df_world_deaths




if __name__=='__main__':
    main()
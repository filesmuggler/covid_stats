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
    # sum by country (no USA yet)
    df_country_confirmed = df_confirmed_global_temp.groupby('Country/Region').agg('sum')
    # print(df_country_confirmed.shape)
    # print(df_country_confirmed.head(5))

    ## CONFIRMED dumbfuckinstan
    df_confirmed_usa = pd.read_csv(path_confirmed_usa)
    df_confirmed_usa.rename(columns={'Country_Region': 'Country/Region'}, inplace=True)
    df_confirmed_usa_temp = pd.concat([df_confirmed_usa.iloc[:, 7:8], df_confirmed_usa.iloc[:, 11:]], axis=1)
    df_confirmed_usa_total = df_confirmed_usa_temp.groupby('Country/Region').agg('sum')
    # print(df_confirmed_usa_total.shape)
    # print(df_confirmed_usa_total.head(5))

    ## CONFIRMED merge
    df_world_confirmed = pd.concat([df_country_confirmed, df_confirmed_usa_total])

    df_world_confirmed = df_world_confirmed.T

    #df_world_confirmed = df_world_confirmed.rename(columns={'Country/Region':'Date'})

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
    # sum by country (no USA yet)
    df_country_deaths = df_deaths_global_temp.groupby('Country/Region').agg('sum')
    # print(df_country_deaths.shape)
    # print(df_country_deaths.head(5))

    ## DEATH dumbfuckinstan
    df_deaths_usa = pd.read_csv(path_deaths_usa)
    df_deaths_usa.rename(columns={'Country_Region': 'Country/Region'}, inplace=True)
    df_deaths_usa_temp = pd.concat([df_deaths_usa.iloc[:, 7:8], df_deaths_usa.iloc[:, 12:]], axis=1)
    df_deaths_usa_total = df_deaths_usa_temp.groupby('Country/Region').agg('sum')
    # print(df_deaths_usa_total.head(5))

    ## DEATH merge
    df_world_deaths = pd.concat([df_country_deaths, df_deaths_usa_total])


    df_world_deaths = df_world_deaths.T

    # df_world_confirmed = df_world_confirmed.rename(columns={'Country/Region':'Date'})

    df_world_deaths.index.name = "Date"
    df_world_deaths.index = pd.to_datetime(df_world_deaths.index)
    return df_world_deaths


def recovery_per_day():
    ## recovery global
    df_recovery_global = pd.read_csv(path_recovery_global)
    print(df_recovery_global.head())

    df_recovery_global_temp = pd.concat([df_recovery_global.iloc[:, 1:2], df_recovery_global.iloc[:, 4:]], axis=1)
    # check if there are any nans
    df_recovery_global_temp = df_recovery_global_temp.fillna(0)
    # sum by country (no USA yet)
    df_country_recovery = df_recovery_global_temp.groupby('Country/Region').agg('sum')
    print(df_country_recovery.shape)
    print(df_country_recovery.head(5))
    df_country_recovery = df_country_recovery.T
    print(df_country_recovery.head(5))

    # ## CONFIRMED dumbfuckinstan
    # df_deaths_usa = pd.read_csv(path_deaths_usa)
    # df_deaths_usa.rename(columns={'Country_Region': 'Country/Region'}, inplace=True)
    # df_deaths_usa_temp = pd.concat([df_deaths_usa.iloc[:, 7:8], df_deaths_usa.iloc[:, 12:]], axis=1)
    # df_deaths_usa_total = df_deaths_usa_temp.groupby('Country/Region').agg('sum')
    # # print(df_deaths_usa_total.head(5))
    #
    # ## CONFIRMED merge
    # df_world_deaths = pd.concat([df_country_deaths, df_deaths_usa_total])

    # print(df_world_deaths.tail(10))

    #return df_world_deaths

def main():
    # ## confirmed cases
    df_world_confirmed = confirmed_per_day()
    print(df_world_confirmed)
    print(type(df_world_confirmed.index))
    ## deaths
    df_world_deaths = deaths_per_day()
    print(df_world_deaths)
    print(type(df_world_deaths.index))

if __name__=='__main__':
    main()
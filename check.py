import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import netCDF4
from netCDF4 import Dataset

path_confirmed_global = "data/time_series_covid19_confirmed_global.csv"
path_confirmed_usa = "data/time_series_covid19_confirmed_US.csv"

def main():
    ## global
    df_confirmed_global = pd.read_csv(path_confirmed_global)

    df_confirmed_global_temp = pd.concat([df_confirmed_global.iloc[:,1:2],df_confirmed_global.iloc[:,4:]],axis=1)
    # check if there are any nans
    df_confirmed_global_temp = df_confirmed_global_temp.fillna(0)
    # sum by country (no USA yet)
    df_country_confirmed = df_confirmed_global_temp.groupby('Country/Region').agg('sum')
    print(df_country_confirmed.shape)
    print(df_country_confirmed.head(5))

    ## dumbfuckinstan
    df_confirmed_usa = pd.read_csv(path_confirmed_usa)
    df_confirmed_usa.rename(columns={'Country_Region':'Country/Region'},inplace=True)
    df_confirmed_usa_temp = pd.concat([df_confirmed_usa.iloc[:,7:8],df_confirmed_usa.iloc[:,11:]],axis=1)
    #print(df_confirmed_usa_temp.shape)
    df_confirmed_usa_total = df_confirmed_usa_temp.groupby('Country/Region').agg('sum')
    print(df_confirmed_usa_total.shape)
    print(df_confirmed_usa_total.head(5))

    ## merge
    df_world = pd.concat([df_country_confirmed,df_confirmed_usa_total])

    print(df_world.shape)
    print(df_world.tail(10))



if __name__=='__main__':
    main()
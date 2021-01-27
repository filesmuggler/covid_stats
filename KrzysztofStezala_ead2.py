import copy
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import netCDF4
import tqdm as tqdm
from netCDF4 import Dataset

from scipy.stats import normaltest, f_oneway, chisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

def BUCKET_TEMP(temp,dt):
    print(temp[dt])
    return 3

def remove_nan(a):
    a = np.asarray(a)
    b = a[np.logical_not(np.isnan(a))]
    return b

def task_1():
    # confirmed cases
    df_world_confirmed = confirmed_per_day()
    df_world_confirmed_14 = df_world_confirmed.shift(14, fill_value=0)
    # deaths
    df_world_deaths = deaths_per_day()
    df_world_deaths_14 = df_world_deaths.shift(14, fill_value=0)
    # recovered
    # USA nie publikuje danych o ozdrowieńcach, więc założono czas infekcji 14 dni
    # Po tym czasie chory mógł albo wyzdrowieć albo umrzeć, co powoduje, że można uznać wzór
    # L_recovered = L_confirmed_14 - L_death_14
    # L_recovered -> liczba ozdrowieńców do 14 dni wstecz od obecnego dnia
    # L_confirmed_14 -> liczba potwierdzonych przypadków do 14 dnia wstecz od teraz
    # L_death_14 -> liczba zgonów przypadków do 14 dnia wstecz od teraz
    # Daje to przybliżony pogląd na rzeczywistą liczbę, zazwyczaj różnice w porównaniu z innymi
    # źródłami, jak np. Worldometers czy dane statystyczne na stronach rządowych poszczególnych krajów
    df_usa_recovered = df_world_confirmed_14['US'] - df_world_deaths_14['US']
    df_world_recovered = recovery_per_day_no_usa()
    df_world_recovered['US'] = df_usa_recovered

    # Canada nie publikuje ozdrowieńców dla poszczególnych prowincji, tylko dla cało kraju, więc
    # żeby utrzymać dane o prowincjach dokonano przekształcenia jak w wypadku USA
    canada_names = [col for col in df_world_confirmed if col.startswith('Canada')]
    df_canada_recovered = df_world_confirmed_14[canada_names] - df_world_deaths_14[canada_names]
    df_world_recovered = df_world_recovered.drop(columns={'Canada'})
    df_world_recovered[canada_names] = df_canada_recovered

    df_world_recovered = df_world_recovered.reindex(sorted(df_world_recovered.columns), axis=1)

    # active
    df_world_active = df_world_confirmed - df_world_recovered - df_world_deaths

    df_world_active = df_world_active.dropna(axis=1)

    df_world_recovered_monthly = df_world_recovered.resample('1M').apply(custom_resampler)
    df_world_deaths_monthly = df_world_deaths.resample('1M').apply(custom_resampler)

    df_mortality_rate_month = df_world_deaths_monthly / df_world_recovered_monthly
    df_mortality_rate_month = df_mortality_rate_month.replace([np.inf, -np.inf], np.nan)
    df_mortality_rate_month = df_mortality_rate_month.fillna(0)

    df_M = df_world_active.rolling(7).mean().fillna(0)

    df_M_5 = df_M.shift(5, fill_value=0)
    df_R = (df_M / df_M_5)
    df_R = df_R.replace([np.inf, -np.inf], np.nan)

    df_R_max = df_R.max(axis=0)
    df_R_norm = df_R / df_R_max
    df_R_norm_month = df_R_norm.resample('1M').mean()
    df_R_norm_month = df_R_norm_month.T

    w_min, w_max = weather()
    lat_lon = lon_lat_countries()
    lat_lon['Lat_idx'] = lat_lon.apply(lambda row: GET_IDX(row['Lat'], w_min['lat'][:]), axis=1)
    lat_lon['Long_idx'] = lat_lon.apply(lambda row: GET_IDX(row['Long'], w_min['lon'][:]), axis=1)

    temps = pd.DataFrame().reindex_like(df_world_deaths_monthly)
    temps_min = temps.T
    temps_max = temps.T

    dates = temps_min.columns

    w_min_np = w_min.variables['tmin'][:]

    for dt in dates:
        temps_min[dt] = lat_lon.apply(lambda row: GET_SINGLE_TEMP(row['Lat_idx'], row['Long_idx'], w_min_np, dt),
                                      axis=1)
    w_min_np = None
    #
    w_max_np = w_max.variables['tmax'][:]
    for dt in dates:
        temps_max[dt] = lat_lon.apply(lambda row: GET_SINGLE_TEMP(row['Lat_idx'], row['Long_idx'], w_max_np, dt),
                                      axis=1)
    w_max_np = None

    avg_temp = (temps_min + temps_max) / 2.0

    below_zero = []
    zero_to_ten = []
    ten_to_twenty = []
    twenty_to_thirty = []
    over_thirty = []

    for dt in dates:
        below_zero.extend(list(df_R_norm_month[dt].loc[avg_temp.index[avg_temp[dt] < 0]].values))
        zero_to_ten.extend(
            list(df_R_norm_month[dt].loc[avg_temp.index[(avg_temp[dt] > 0) & (avg_temp[dt] < 10)]].values))
        ten_to_twenty.extend(
            list(df_R_norm_month[dt].loc[avg_temp.index[(avg_temp[dt] > 10) & (avg_temp[dt] < 20)]].values))
        twenty_to_thirty.extend(
            list(df_R_norm_month[dt].loc[avg_temp.index[(avg_temp[dt] > 20) & (avg_temp[dt] < 30)]].values))
        over_thirty.extend(list(df_R_norm_month[dt].loc[avg_temp.index[avg_temp[dt] > 30]].values))

    below_zero = remove_nan(below_zero)
    zero_to_ten = remove_nan(zero_to_ten)
    ten_to_twenty = remove_nan(ten_to_twenty)
    twenty_to_thirty = remove_nan(twenty_to_thirty)
    over_thirty = remove_nan(over_thirty)

    print(normaltest(below_zero))
    print(normaltest(zero_to_ten))
    print(normaltest(ten_to_twenty))
    print(normaltest(twenty_to_thirty))
    print(normaltest(over_thirty))

    f_value, p_value = f_oneway(below_zero, zero_to_ten, ten_to_twenty, twenty_to_thirty, over_thirty)
    print("F-stat: ", f_value, " p-val: ", p_value)

    print(pairwise_tukeyhsd(np.concatenate([below_zero, zero_to_ten, ten_to_twenty, twenty_to_thirty, over_thirty]),
                            np.concatenate([['below_zero'] * len(below_zero),
                                            ['zero_to_ten'] * len(zero_to_ten),
                                            ['ten_to_twenty'] * len(ten_to_twenty),
                                            ['twenty_to_thirty'] * len(twenty_to_thirty),
                                            ['over_thirty'] * len(over_thirty)])))

    ## WNIOSKI
    # Wyniki testu na normalność rozkładów dla poszczegolnych przedziałów temperatur sygnalizują, że
    # ich rozkłady odbiegają od normalnego (w każdym wypadku p << 0.05)
    #
    # Po podaniu danych do testu anova wynik p jest również bliski zero co każe nam odrzucić hipotezę o niezależnosci
    # zmiennych. Wynika z że rozprzestrzenianie się wirusa jest zależne od temperatury, co stoi w sprzeczności z danymi
    # publikowanymi na stronie pacjent.gov.pl.

def task_2_1():
    print("task2_1")
    # confirmed cases
    df_world_confirmed = confirmed_per_day()
    df_world_confirmed = df_world_confirmed.T
    df_world_confirmed_last = df_world_confirmed.iloc[:,-1:]
    # deaths
    df_world_deaths = deaths_per_day()
    df_world_deaths = df_world_deaths.T
    df_world_deaths_last = df_world_deaths.iloc[:,-1:]

    lat_lon = lon_lat_countries()

    # europe 70 > lat > 35 ; -30 < long < 50

    europe = lat_lon.loc[(lat_lon['Lat']<70) & (lat_lon['Lat']>35) & (lat_lon['Long']<60) & (lat_lon['Long']>-30)]

    df_world_confirmed_last_eu = df_world_confirmed_last.loc[europe.index]
    df_world_deaths_last_eu = df_world_deaths_last.loc[europe.index]

    df_eu = df_world_deaths_last_eu/df_world_confirmed_last_eu

    sum_eu = df_eu.sum().values

    exp = np.array([1/len(df_eu) for x in range(len(df_eu))]) * sum_eu

    obs = df_eu.values

    chi2, p = chisquare(obs, exp)
    print(chi2,p)
    ## WNIOSKI
    # Porównując do rozkładu równomiernego p=1 dla każdego przypadku , wiec zachowujemy hipoteze zerowa, czyli
    # nie istnieją istotne różnice w śmiertelności z powodu Covid-19

def task_2_2():
    print("task_2_2")
    # confirmed cases
    df_world_confirmed = confirmed_per_day()
    df_world_confirmed_t = df_world_confirmed.T
    # deaths
    df_world_deaths = deaths_per_day()
    df_world_deaths_t = df_world_deaths.T
    # recovered
    df_world_recovered = recovery_per_day_no_usa()
    df_world_recovered_t = df_world_recovered.T

    lat_lon = lon_lat_countries()

    # europe 70 > lat > 35 ; -30 < long < 50

    europe_filter = lat_lon.loc[(lat_lon['Lat'] < 70) & (lat_lon['Lat'] > 35) & (lat_lon['Long'] < 60) & (lat_lon['Long'] > -30)]

    df_eu_confirmed = df_world_confirmed_t.loc[europe_filter.index]
    df_eu_deaths = df_world_deaths_t.loc[europe_filter.index]
    df_eu_recovered = df_world_recovered_t.loc[europe_filter.index]

    # active
    df_eu_active = df_eu_confirmed - df_eu_deaths - df_eu_recovered
    df_eu_active = df_eu_active.T
    df_eu_active = df_eu_active.dropna(axis=1)

    df_eu_recovered = df_eu_recovered.T
    df_eu_deaths = df_eu_deaths.T

    df_eu_recovered_monthly = df_eu_recovered.resample('1M').apply(custom_resampler)
    df_eu_deaths_monthly = df_eu_deaths.resample('1M').apply(custom_resampler)


    df_mortality_rate_month = df_eu_deaths_monthly / df_eu_recovered_monthly
    df_mortality_rate_month = df_mortality_rate_month.replace([np.inf, -np.inf], np.nan)
    df_mortality_rate_month = df_mortality_rate_month.fillna(0)

    df_M = df_eu_active.rolling(7).mean().fillna(0)

    df_M_5 = df_M.shift(5, fill_value=0)
    df_R = (df_M / df_M_5)
    df_R = df_R.replace([np.inf, -np.inf], np.nan)

    df_R_max = df_R.max(axis=0)
    df_R_norm = df_R / df_R_max
    df_R_norm_month = df_R_norm.resample('1M').mean()

    df_R_nm_np = df_R_norm_month.to_numpy()

    R = df_R_nm_np.tolist()

    R_new = []
    for c in R:
        R_new.append(remove_nan(c))

    R_new = [x for x in R_new if x != []]

    f_value, p_value = f_oneway(*[col for col in R_new])
    print(f'F-stat: {f_value}, p-val: {p_value}')

    ## WNIOSKI
    # Porównano zależność smiertelności pomiędzy krajami w Europie, p<<0, więc można odrzucić hipoteze zerową,
    # że nie ma zależności i zbadać na czym polegam taka zależność

def main():
    ## UWAGA task_1 może zabierać dużo pamięci RAM ze względu na szybkie ładowanie danych o temperaturze
    task_1()
    task_2_1()
    task_2_2()

if __name__=='__main__':
    main()

















































































































































































































































































































































































































































































































































    # Nobody expected that
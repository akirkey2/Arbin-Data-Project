 # -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 09:30:03 2023

@author: Aaron Kirkey
"""

#%% Environment setup

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
import scipy as sp
import glob
import sys
from matplotlib import cm
import pathlib
import os
from sklearn import preprocessing
import matplotlib.cm as cm
from pathlib import Path
from IPython.display import display
import re
import math
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, WeekdayLocator
display()
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.interactive(True)
mpl.rcParams['lines.markersize'] = 2
plt.rc('axes', axisbelow=True)
plt.rcParams['figure.constrained_layout.use'] = True

#%%

df = pd.read_csv('facilitymonitoring-2023-08-15T00_00_00_2023-11-10T23_55_00.csv')
df = df.drop('Unnamed: 0',axis = 1)
lst = df['location'].unique().tolist()
# df.index = df.index.astype('datetime64[ns]')
#%%
def ppmv(location):
    #Separating df's, locations and values into new columns to simplify doing math later
    df.index = df.index.astype('datetime64[ns]')
    dfl = df.loc[(df['location']==location)]
    dfl['rh'] = dfl.loc[(dfl['type']=='humidity')]['val']
    dfl['temp'] = dfl.loc[(dfl['type']=='temperature')]['val']
    dfl = dfl.groupby('time_stamp').max()
    dfl = dfl.dropna()


    #dewpoint + ppmv calcs (the aforementioned math)
    a = 17.368
    b = 238.88
    
    #dfl['dewpoint'] = (b * ((a * dfl['temp']) / (b + dfl['temp']) + np.log(dfl['rh'] / 100.0))) / (a - ((a * dfl['temp']) / (b + dfl['temp']) + np.log(dfl['rh'] / 100.0)))
    #dfl['ppmv'] =(pow(10,6)*(6.139243551*np.exp(22.452*dfl['dewpoint']/(272.55+dfl['dewpoint']))))/(1014.25-(6.139243551*np.exp(22.452*dfl['dewpoint']/(272.55+dfl['dewpoint']))))
    #dfl['ps'] = 6.1121*np.exp((18.678-(dfl['temp']/234.5))*(dfl['temp']/(257.14+ dfl['temp'])))
    #dfl['ah'] = (dfl['rh']*(dfl['ps']*1000))/(461.5*(dfl['temp']+273.15))
    dfl['ah2'] = (6.112*np.exp((17.67*dfl['temp'])/(dfl['temp']+243.5))*dfl['rh']*18.02)/((273.15+dfl['temp'])*100*0.08314)

    
    dfl.index = pd.to_datetime(dfl.index)
    dfl['time']= dfl.index
    hourly = dfl.asfreq(freq = '60T').between_time('9:00:00','17:00:00')
    daily = hourly.groupby(pd.Grouper(freq='D')).mean()
    daily['time'] = daily.index
    
    for i in range(len(daily)):
        daily['time'][i] = daily.index[i].replace(hour = 12)
        
        
    
    return daily, hourly

#%%
list_of_daily = []
list_of_hourly = []
for location in lst:
    tmp_daily, tmp_hourly = ppmv(location)
    list_of_daily.append(tmp_daily)
    list_of_hourly.append(tmp_hourly)

#%%
# start_date = '2023-09-14'
# end_date = '2023-10-09'
# dfl = dfl.reset_index()
# dfste['time_stamp'] = dfste['time_stamp'].astype(str)

#%%
# dfste['time_stamp'] = dfste['time_stamp'].astype('datetime64[ns]')
def plot():
    plt.figure()
    fig_ax = plt.subplot(1,1,1)
    count = 0
    colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]
    for dfd,dfh in zip(list_of_daily, list_of_hourly):
        lbl = list_of_hourly[count]['location'][0]
        plt.scatter(dfd.index,dfd['ah2'],s=30, marker='D', label = f'{lbl} daily',color=colors[count])
        plt.scatter(dfh.index,dfh['ah2'],s=5,marker='D', label = f'{lbl} hourly',color=colors[count])
        loc = WeekdayLocator(byweekday=0)
        fig_ax.xaxis.set_major_locator(loc)
        # ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        # ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
        plt.grid()
        plt.xticks(dfd.index)
        plt.xlabel('Day')
        plt.ylabel('Absolute Humidity ($g/m^3$)')
        plt.title('Concentration of $H_2O$ vapour in air')
        plt.xticks(rotation = 25)
        plt.legend()
        count += 1
    plt.show()

# ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
# ax.xaxis.set_major_formatter(DateFormatter("%m-%d"))
#%%

for dfh in list_of_hourly:
    dfh['day'] = ""
    for i in dfh.index:
        dfh['day'][i] = i.date()
#%%
count = 0
for dfh in list_of_hourly:
    plt.figure()
    ax = sns.boxplot(data=dfh, x='day',y='ah2')
    ax.set(xlabel='Date', ylabel='Absolute Humidity ($g/m^3$)')
    plt.xticks(rotation=30)
    loc = lst[count]
    plt.title(f'{loc}')
    count += 1




#%%
"""
def convert_dew_point_to_ppm(dew_point):
    if math.isnan(dew_point) or dew_point is None:
        return -1

    lower_bound = math.floor(dew_point)

    if lower_bound % 2:
        lower_bound -= 1

    upper_bound = lower_bound + 2

    result = linear_interpolation(dew_point, lower_bound, DEW_POINT_PPM_DICT.get(lower_bound, -1),
                                  upper_bound, DEW_POINT_PPM_DICT.get(upper_bound, -1))

    return result


DEW_POINT_PPM_DICT = {
        -110: 0.00132,
        -108: 0.00237,
        -106: 0.00368,
        -104: 0.00566,
        -102: 0.00855,
        -100: 0.013,
        -98: 0.0197,
        -96: 0.0289,
        -94: 0.0434,
        -92: 0.0632,
        -90: 0.0921,
        -88: 0.132,
        -86: 0.184,
        -84: 0.263,
        -82: 0.382,
        -80: 0.562,
        -78: 0.737,
        -76: 1.01,
        -74: 1.38,
        -72: 1.88,
        -70: 2.55,
        -68: 3.43,
        -66: 4.59,
        -64: 6.11,
        -62: 8.08,
        -60: 10.6,
        -58: 13.9,
        -56: 18.2,
        -54: 23.4,
        -52: 30.3,
        -50: 38.8,
        -48: 49.7,
        -46: 63.3,
        -44: 80,
        -42: 101,
        -40: 127,
        -38: 159,
        -36: 198,
        -34: 246,
        -32: 305,
        -30: 376,
        -28: 452,
        -26: 566,
        -24: 692,
        -22: 842,
        -20: 1020,
        -18: 1240,
        -16: 1490,
        -14: 1790,
        -12: 2160,
        -10: 2570,
        -8: 3050,
        -6: 3640,
        -4: 4320,
        -2: 5100,
        0: 6020,
        2: 6970,
        4: 8030,
        6: 9230,
        8: 10590,
        10: 12120,
        12: 13840,
        14: 15780,
        16: 17930,
        18: 20370,
        20: 23080,
        22: 26088,
        24: 29443,
        26: 33169,
        28: 37301,
        30: 41874
    }
"""
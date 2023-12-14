# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 09:43:14 2023

@author: Aaron Kirkey
"""

#%% Environment setup

import pandas as pd
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
display()
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.interactive(True)
mpl.rcParams['lines.markersize'] = 2
plt.rc('axes', axisbelow=True)
plt.rcParams['figure.constrained_layout.use'] = True

#%%

df = pd.DataFrame()
for file in os.listdir('C:\\Users\\Aaron Kirkey\\PythonProjects\\UV_Vis'):
    if 'Sample.Raw' in file:
        print(file)
        data = pd.read_csv('UV_Vis\\'+file)
        dc = re.split('\.',file)[0]
        data.rename(columns = {'nm':str(dc)+'nm',' %T':str(dc)+'%T'}, inplace = True)
        df = pd.concat([df,data],axis = 1)
for i in df.columns:
    if '%T' in i:
        df[re.split('%',str(i))[0]+'abs']= 2 - np.log10(df[str(i)])
df.columns = df.columns.str.replace("2023090067", "bg", regex=True)

dfbg = pd.read_csv('UV_Vis\\UV_Vis backgrounds.csv',skiprows=5)
#%%
dfbg.drop(dfbg.iloc[:,2:6],axis=1,inplace=True)
dfbg.drop(dfbg.iloc[:,-2:],axis = 1, inplace = True)
#%%
dfbg.columns=['nm','liq_t','acr_t']
dfbg['liq_abs'],dfbg['acr_abs'] = '',''
dfbg['liq_abs'], dfbg['acr_abs']=(2-np.log10(dfbg['liq_t'])),(2-np.log10(dfbg['acr_t']))
# %%
plt.figure()
df = pd.DataFrame()
for file in os.listdir('C:\\Users\\Aaron Kirkey\\PythonProjects\\UV_Vis'):
    if 'Sample.Raw' in file:
        print(file)
        data = pd.read_csv('UV_Vis\\'+file)
        dc = re.split('\.',file)[0]
        data.rename(columns = {' %T':str(dc)}, inplace = True)
        plt.grid()
        plt.xlabel('Wavelength(nm)')
        plt.ylabel('Absorbance')
        plt.title('Absorbance')
        plt.ylim(0.15,2.25)
        plt.xlim(300,1100)
        plt.plot(data['nm'],(2 - np.log10(data.iloc[:,1])),label = dc)
        #plt.plot(data['nm'],data.iloc[:,1],label = dc)  
        plt.legend()
        df = pd.concat([df,data],axis = 1)
        
#%%
# plt.figure()
# plt.grid()
# plt.xlabel('Wavelength(nm)')
# plt.ylabel('Transmission (%)')
# plt.title('Transmission (%)')
# plt.plot(data['nm'],data.iloc[:,1],label = '2023060122') 

plt.figure()
plt.grid()
plt.xlabel('Wavelength(nm)')
plt.ylabel('Transmission (%)')
plt.title('UV-Vis of various failed devices')
plt.plot(df.iloc[:,0],df['2023060122'],label = '2023060122',color='orange')
plt.plot(df.iloc[:,0],df['2023090002'],label = '2023090002',color='C5') 
plt.plot(df.iloc[:,0],df['2023090067'],label = '2023090067',color='black') 
plt.plot(df.iloc[:,0],df['2023090065'],label = '2023090065',color='C0')
plt.plot(df.iloc[:,0],df['2023060204'],label = '2023090065',color='grey')
plt.legend()

#%%
plt.plot(df.iloc[:,0],df['2023060122abs']-dfbg['liq_abs'],label = '2023060122',color='orange')
plt.plot(df.iloc[:,0],df['2023090002abs']-dfbg['liq_abs'],label = '2023090002',color='C5') 
plt.plot(df.iloc[:,0],dfbg['liq_abs']-dfbg['liq_abs'],label = 'bg',color='black') 
plt.plot(df.iloc[:,0],df['2023090065abs']-dfbg['liq_abs'],label = '2023090065',color='C0')
plt.plot(df.iloc[:,0],df['2023060204abs']-dfbg['liq_abs'],label = '2023060204',color='grey')
plt.grid()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Diff. in abs. (a.u.)')
# plt.ylim(-0.25,0.75)
plt.legend()
plt.show()
#%%

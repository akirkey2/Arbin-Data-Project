# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:28:09 2023

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
from matplotlib.cm import get_cmap

name = "tab20"
cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list


#%%

df = pd.DataFrame()
list_of_dfs = []
device_codes = []
filepath = 'C:\\Users\\Aaron Kirkey\\Miru\\FA_Pilot\\IR'
for file in os.listdir(filepath):
    if 'Ni' in file:
        print(file)
        device_codes.append(re.split('_',file)[3])
        temp_df = pd.read_csv(filepath + '\\' + file,delimiter = '\t', names = ['wavenumber','absorbance'])
        list_of_dfs.append(temp_df)
# %%
fig, ax = plt.subplots(1,1)
for i in range(len(list_of_dfs)):
    ax.plot(list_of_dfs[i]['wavenumber'],list_of_dfs[i]['absorbance'], label = device_codes[i],color=colors[i])
    ax.set_xlim(4000,400)
    ax.set_xlabel('Wavenumber $(cm^{-1})$')
    ax.set_ylabel('Absorbance (a.u.)')
    # ax.set_prop_cycle(color=colors)
    fig.legend()

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:03:56 2023

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
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.interactive(True)
mpl.rcParams['lines.markersize'] = 2
plt.rc('axes', axisbelow=True)
plt.rcParams['figure.constrained_layout.use'] = True

device_names = [
    '110001',
    # '050047',
    #'2023050123, 2023050125'
    ]
ls = []
folder = ""
dfall = pd.DataFrame()
arbin_data_folder = Path('C:\\Users\\Aaron Kirkey\\PythonProjects\\Arbin Data Project')

for folder in os.listdir(arbin_data_folder): #Loops through folder in Arbin Data Project
    for x in device_names: # Checks device names for names
        if x in folder: #Checks if device name is in any folders in Arbin Data Project
            ls = []
            ffolder = folder
            for i in os.listdir(str(arbin_data_folder)+'\\'+folder):#Checks for .csv files in the folder matching device name
                if 'Wb' in i and '.CSV' in i: #Appends .csv file to a list named lst
                    ls.append(i)
            print(ls)
            for i in range(len(ls)): #Makes a dict with key df1,df2,df3... and values are actual dataframes
                dfchunk = pd.read_csv(str(arbin_data_folder)+'\\'+folder+'\\'+str(ls[i]))#,usecols=['Test Time (s)','Step Time (s)',
                                                                      # 'Cycle Index', 'Step Index', 
                                                                      # 'Current (A)', 'Voltage (V)',
                                                                      # 'Charge Capacity (Ah)', 'Discharge Capacity (Ah)', 
                                                                      # 'Internal Resistance (Ohm)'])
#If you only want specific columns, use^^^
                #key=str('df'+str(i+1))
                #file = str(os.getcwd()+'\\'+folder+'\\'+ls[x])
                #dfchunk= pd.read_csv(folder+'\\'+str(ls[i])) #os.getcwd()+'\\'+ #FOR FULL RAW CSV
                dfall = dfall.append([dfchunk], ignore_index=True)
                
        #print(ls)
    else:
        continue
dfchunk = 0
device_code = ls[0][:ls[0].find('_')]
df = dfall


'''dfstat=pd.DataFrame()
for folder in os.listdir(arbin_data_folder): #Loops through folder in Arbin Data Project
    for x in device_names: # Checks device names for names
        if x in folder: #Checks if device name is in any folders in Arbin Data Project
            lss = []
            ffolder = folder
            for i in os.listdir(str(arbin_data_folder)+'\\'+folder):#Checks for .csv files in the folder matching device name
                if 'Stat' in i: #Appends .csv file to a list named lst
                    lss.append(i)
            print(lss)
            for i in range(len(lss)): #Makes a dict with key df1,df2,df3... and values are actual dataframes
                dfchunk = pd.read_csv(str(arbin_data_folder)+'\\'+folder+'\\'+str(lss[i]))
                """,usecols=['Date_Time                    ', 'Test Time (s)', 'Step Time (s)',
                       'Cycle Index', 'Step Index', 'Current (A)', 'Voltage (V)', 'Power (W)',
                       'Charge Capacity (Ah)', 'Discharge Capacity (Ah)', 'Charge Energy (Wh)',
                       'Discharge Energy (Wh)', 'Charge Time (s)', 'Discharge Time (s)',
                       'V_Max_On_Cycle (V)', 'Coulombic Efficiency (%)', 'mAh/g'])"""
#If you only want specific columns, use^^^
                #key=str('df'+str(i+1))
                #file = str(os.getcwd()+'\\'+folder+'\\'+ls[x])
                #dfchunk= pd.read_csv(folder+'\\'+str(ls[i])) #os.getcwd()+'\\'+ #FOR FULL RAW CSV
                dfstat = dfstat.append([dfchunk], ignore_index=True)
                
        #print(ls)
    else:
        continue
dfchunk = 0
device_code = ls[0][:ls[0].find('_')]

if '/' not in str(dfstat.iloc[0,0]):
    dfstat.columns = ['Test Time (s)', 'Step Time (s)',
            'Cycle Index', 'Step Index', 'Current (A)', 'Voltage (V)', 'Power (W)',
            'Charge Capacity (Ah)', 'Discharge Capacity (Ah)', 'Charge Energy (Wh)',
            'Discharge Energy (Wh)', 'Charge Time (s)', 'Discharge Time (s)',
            'V_Max_On_Cycle (V)', 'Coulombic Efficiency (%)', 'mAh/g','blank']'''
# Making df's grouped by cycle num + finding max and min values at each cycle num
dfmax = dfall.groupby(dfall['Cycle Index']).max()
dfmin = dfall.groupby(dfall['Cycle Index']).min()
#Setting up dfs for bleach and darken max voltages & TTB, TTD
dfdark = dfall.loc[(dfall['Step Index'] == 6)]
dfdarkmax = dfdark.groupby(dfdark['Cycle Index']).max()
dfbleach = dfall.loc[(dfall['Step Index'] == 9)]
dfbleachmax = dfbleach.groupby(dfbleach['Cycle Index']).max()
dfdarkmin = dfdark.groupby(dfdark['Cycle Index']).min()
# Making dfs for IR bleach & IR dark
dfbleach_ir = dfall.loc[(dfall['Step Index'] == 5)]
# dfdark_ir = dfall.loc[(dfall['Step Index'] == 8)]
# dfdark_ir = dfdark_ir.groupby(dfdark_ir['Cycle Index']).max()

# df's for Voltage Cutoffs - EOD July 13
dfdarkmin = dfdark.groupby(dfdark['Cycle Index']).min()
dfbleachmax = dfbleach.groupby(dfbleach['Cycle Index']).max()

# To split dfdark into CC and CV - EOD July 13
if dfdarkmin['Voltage (V)'].iloc[115] < -1.9:
    desired_v = -1.895
else:
    desired_v = -1.395
dfdarkcc = dfdark.loc[(dfdark['Voltage (V)'] > desired_v)]
dfdarkcv = dfdark.loc[(dfdark['Voltage (V)'] < desired_v)]

dfdarkcc = dfdarkcc.groupby(dfdarkcc['Cycle Index']).max()
dfdarkcv = dfdarkcv.groupby(dfdarkcv['Cycle Index']).max()
dfdarkcc['cv'] = dfdarkcv['Discharge Capacity (Ah)']
dfdarkcc['pqcc'] = (dfdarkcc['Discharge Capacity (Ah)'] / dfdarkcv['Discharge Capacity (Ah)'].quantile(0.99)) * 100

# dfs + columns for % charge from cc and cv
for i in (range(len(dfdarkcc))):
    if pd.isna(dfdarkcc['cv'].iloc[i]) == True:
        dfdarkcc['pqcc'].iloc[i] = 100 
    else:
        dfdarkcc['pqcc'].iloc[i] = (dfdarkcc['Discharge Capacity (Ah)'].iloc[i] / dfdarkcc['cv'].iloc[i]) * 100

# To split dfbleach into CC and CV
if dfbleachmax['Voltage (V)'].iloc[110] > 1.6:
    desired_v = 1.695
else:
    desired_v = 1.395  
dfbleachcc = dfbleach.loc[(dfbleach['Voltage (V)'] < desired_v)]
dfbleachcv = dfbleach.loc[(dfbleach['Voltage (V)'] > desired_v)]
dfbleachcc = dfbleachcc.groupby(dfbleachcc['Cycle Index']).max()
dfbleachcv = dfbleachcv.groupby(dfbleachcv['Cycle Index']).max()
dfbleachcc['pqcc'] = (dfbleachcc['Charge Capacity (Ah)'] / dfbleachcv['Charge Capacity (Ah)'].quantile(0.995)) * 100

#dfs for charge delivered per cycle
dfq = dfall.groupby(dfall['Cycle Index']).max()
dfq['%maxdq'] = ""
dfq['%maxcq'] = ""
dq999= dfq['Discharge Capacity (Ah)'].quantile(0.999)
cq999=dfq['Charge Capacity (Ah)'].quantile(0.999)
for i in range(len(dfq)): 
    dfq['%maxdq'].iloc[i] = dfq['Discharge Capacity (Ah)'].iloc[i] / dq999
    dfq['%maxcq'].iloc[i] = dfq['Charge Capacity (Ah)'].iloc[i] / cq999
dfq['%qdiff'] = dfq['%maxdq'] - dfq['%maxcq']
#%%
dfc1300 = df.loc[(df['Cycle Index']>1300)]
#%% IR and TT to D and B

fig, axs = plt.subplots(2,layout='constrained')
axs[0].set_ylabel('Resistance (Ohms)')
fig.suptitle('IR, TTB & TTD (%s)'%(device_code))
plt.xlabel('Cycle')
plt.xlim()
#axs[0].set_ylim(0,(dfstep1['Internal Resistance (Ohm)'].mean() + 200))
axs[0].scatter(dfbleach_ir['Cycle Index'], dfbleach_ir['Internal Resistance (Ohm)'],label = 'IR @ Bleach',  color = '#1f77b4',s=2)
# axs[0].scatter(dfdark_ir.index, dfdark_ir['Internal Resistance (Ohm)'], label = 'IR @ Dark', color = '#ff7f0e',s=1)
axs[0].legend()
axs[0].set_ylim(0,dfall['Internal Resistance (Ohm)'].quantile(0.99)+50)

axs[0].set_xlim(0,)
axs[1].set_ylabel('Time (s)')
axs[1].scatter(dfdarkmax.index.values,dfdarkmax['Step Time (s)'],label = 'TTD',color = '#ff7f0e',s=2, marker='D')
axs[1].scatter(dfbleachmax.index.values, dfbleachmax['Step Time (s)'],label = 'TTB',color = '#1f77b4',s=1)
axs[1].set_ylim(0,dfall['Step Time (s)'].quantile(0.995)+50)
axs[1].set_xlim(0,)
axs[1].legend()

fig.set_figheight(5)
fig.set_figwidth(9)
axs[0].grid()
axs[1].grid()
axs[0].set_axisbelow(True)
axs[1].set_axisbelow(True)

#%%
# Select only step 2
df_steps = df[df['Step Index'] == 2]
# Group by cycle index
df_steps = df_steps.groupby('Cycle Index')
# Loop through groups of cycle index
for cycle_index, cycle_df in df_steps:
    # logger.info(f"Cycle index: {cycle_index}")
    # Compute time difference between rows
    cycle_df['Time Difference'] = cycle_df['Test Time (s)'].diff().fillna(0)

    # logger.info(f"cycle_df['Time Difference']: {cycle_df['Time Difference']}")
    # Compute instantaneous charge = current * time difference / minutes / seconds
    cycle_df["Instantaneous Charge"] = cycle_df["Current (A)"] * cycle_df["Time Difference"] / (60 * 60)

    # logger.info(f"cycle_df['Instantaneous Charge']: {cycle_df['Instantaneous Charge']}")
    # Compute charge by integrating instantaneous charge values
    cycle_df["Discharge"] = cycle_df["Instantaneous Charge"].sum()

    # logger.info(f"cycle_df['Charge']: {cycle_df['Charge']}")
    # Write values back to dataframe
    condition = (df["Cycle Index"] == cycle_index) & (df["Step Index"] == 2)
    df.loc[condition, "Discharge"] = -cycle_df["Discharge"]
    print(cycle_index)

print(f"df[df['Step Index'] == 2][['Discharge', 'Discharge Capacity']].tail(): {df[df['Step Index'] == 2][['Discharge', 'Discharge Capacity (Ah)']].tail(20)}")
# dfa['DCsum'] = ""
#%%
# Calc discharge capacity
# Select only step 2
df_steps = df[df['Step Index'] == 5]
# Group by cycle index
df_steps = df_steps.groupby('Cycle Index')
# Loop through groups of cycle index
for cycle_index, cycle_df in df_steps:
    # logger.info(f"Cycle index: {cycle_index}")
    # Compute time difference between rows
    cycle_df['Time Difference'] = cycle_df['Test Time (s)'].diff().fillna(0)

    # logger.info(f"cycle_df['Time Difference']: {cycle_df['Time Difference']}")
    # Compute instantaneous charge = current * time difference / minutes / seconds
    cycle_df["Instantaneous Charge"] = cycle_df["Current (A)"] * cycle_df["Time Difference"] / (60 * 60)

    # logger.info(f"cycle_df['Instantaneous Charge']: {cycle_df['Instantaneous Charge']}")
    # Compute charge by integrating instantaneous charge values
    cycle_df["Charge"] = cycle_df["Instantaneous Charge"].sum()

    # logger.info(f"cycle_df['Charge']: {cycle_df['Charge']}")
    # Write values back to dataframe
    condition = (df["Cycle Index"] == cycle_index) & (df["Step Index"] == 5)
    df.loc[condition, "Charge"] = cycle_df["Charge"]
    print(cycle_index)

print(f"df[df['Step Index'] == 5][['Charge', 'Charge Capacity']].tail(): {df[df['Step Index'] == 5][['Charge', 'Charge Capacity (Ah)']].tail(20)}")
#%%
dfdarkcc2 = dfdark.loc[(dfdark['Voltage (V)'] > -1.7)]
dfdarkcc2 = dfdarkcc2.groupby(dfdarkcc2['Cycle Index']).min()

#%%
global cycles
cycles = [110,160,210,260,310,360,410,460,510]
# if len(cycles) == 1:
#     for i in np.linspace(101,(dfall['Cycle Index'].max()-1),20):
#         irr = int(round(i,-2))
#         cycles.append(irr)
# else:
#     cycles = cycles
# for i in cycles:
#     if i >= dfall['Cycle Index'].max():
#         cycles.remove(i)
#     elif i <= 100:
#         cycles.remove(i)
# cycles.append((dfall['Cycle Index'].max()-2))
print(cycles)
plt.figure(figsize=(10,8))
plt.grid()
color_map = cm.viridis
cycle_norm = plt.Normalize(min(cycles), max(cycles))
for i in cycles:
    x = dfall.loc[(dfall['Cycle Index']== i)]
    plt.scatter((x['Test Time (s)']-x['Test Time (s)'].min()),x['Current (A)'],label = f'Cycle {i}',color=color_map(cycle_norm(i)))
    plt.legend(loc = 4)
    plt.ylabel('Current')
    plt.xlabel('Cycle Time')
    plt.title(f'Current Profile for various cycles ({device_code})')
#%% Voltage reached plot


fig, (ax1, ax2) = plt.subplots(2, sharex=True)
#fig.suptitle('Voltage reached on Darken or Bleach')
#ax1.scatter(dfmin.index.values, dfmin['Voltage (V)'],color = '#ff7f0e') #darken
ax1.scatter(dfmin.index.values, dfmin['Voltage (V)'],color = '#ff7f0e',s=2) #Step2 min V. Slightly cleaner plots but removes tail that shows switching.
ax1.set_title('Voltage reached on Darken (%s)' %(device_code))
ax1.set_ylim(dfmin['Voltage (V)'].quantile(0.01)-0.1,dfmin['Voltage (V)'].quantile(0.99)+0.1)
#ax2.scatter(dfmax.index.values, dfmax['Voltage (V)']) # bleach
ax2.scatter(dfmax.index.values, dfmax['Voltage (V)'],s=2) #Step5 max V. Cleaner, but removes tails.
ax2.set_title('Voltage reached on  Bleach (%s)' %(device_code))
# ax2.set_ylim(dfmax['Voltage (V)'].quantile(0.01)-0.1,dfmax['Voltage (V)'].quantile(0.99)+0.1)
plt.xlabel('Cycle Number')
fig.tight_layout()
ax1.grid()
ax2.grid()
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)
# plt.savefig(f'{ffolder}/{device_code} Vs_on_Dark_and_Bleach', dpi = 300)
# plt.savefig(f'Figures/{device_code} Vs_on_Dark_and_Bleach', dpi = 300)
  # otherwise the right y-label is slightly clipped
plt.show()
#%%
dfmax['FE Arbin'] = (dfmax['Charge Capacity (Ah)'] / dfmax['Discharge Capacity (Ah)'])*100
dfmax['FE Miru'] = (dfall.groupby(dfall['Cycle Index']).max()['Charge'] / dfall.groupby(dfall['Cycle Index']).max()['Discharge'])*100
plt.figure()
plt.scatter(dfmax.index.values,dfmax['FE Arbin'],label='FE-arbin')
plt.scatter(dfmax.index.values,dfmax['FE Miru'], label = 'FE-miru')
# plt.ylim(5,105)
plt.xlabel('Cycle')
plt.ylabel('Faradaic Efficiency (%)')
plt.title(f'Faradaic Efficiency ({device_code})')
plt.legend()
plt.grid()
plt.show()
# #%%
# dfa = dfall.loc[(dfall['Step Index']==2)]
# dfa['DC'] = dfa['Current (A)'] / 3600
# dfa['DCsum'] = ""
# #%%
# for i in range(0,len(dfa)):
#     if dfa['Cycle Index'].iloc[i] == dfa['Cycle Index'].iloc[i-1]:
#         dfa['DCsum'].iloc[i] = dfa['DC'].iloc[i] + dfa['DCsum'].iloc[i-1]
#         print(i)
#     else:
#         dfa['DCsum'].iloc[i] = 0
# #%%
# dfa['DCsum'].iloc[1] = dfa['DC'].iloc[1] + dfa['DC'].iloc[0]
#%%
# dfstat['Coulombic Efficiency (%)'] = (dfstat['Charge Capacity (Ah)'] / dfstat['Discharge Capacity (Ah)']) * 100
plt.figure()
plt.scatter(dfstat['Cycle Index'], dfstat['Coulombic Efficiency (%)'])
plt.axhline(y=100,color='red')
# plt.scatter(dfmax.index.values,dfmax['FE Arbin'],label='FE-arbin')
plt.grid()
plt.ylim(98,102)
# plt.ylim(dfstat['Coulombic Efficiency (%)'].quantile(0.1)-1,dfstat['Coulombic Efficiency (%)'].quantile(0.9)+1)
plt.xlabel('Cycle Number')
plt.ylabel('<-- More Qbleach     |     More QDark -->')
plt.title(f'Coloumbic Efficiency (%) ({device_code})')
plt.show()
plt.savefig(f'{arbin_data_folder}/{ffolder}/{device_code} Coulombic Efficiency', dpi = 300)
plt.savefig(f'{arbin_data_folder}/Figures/{device_code} Coulombic Efficiency', dpi = 300)

#%%
dfstat['%cqd'] = (dfstat['Charge Capacity (Ah)'] / dfstat['Charge Capacity (Ah)'].iloc[0] ) *100
dfstat['%dqd'] = (dfstat['Discharge Capacity (Ah)'] / dfstat['Discharge Capacity (Ah)'].iloc[0] ) *100
#%%
plt.figure()
plt.scatter(dfstat['Cycle Index'], dfstat['Charge Capacity (Ah)'], label = 'Charge')
plt.scatter(dfstat['Cycle Index'], dfstat['Discharge Capacity (Ah)'], label = 'Discharge')
# plt.scatter(dfmax.index.values,dfmax['FE Arbin'],label='FE-arbin')
plt.grid()
plt.ylim()
# plt.ylim(dfstat['Coulombic Efficiency (%)'].quantile(0.1)-1,dfstat['Coulombic Efficiency (%)'].quantile(0.9)+1)
plt.xlabel('Cycle Number')
plt.ylabel('Q delivered (C)')
plt.title(f'Charge delivered ({device_code})')
plt.legend()
plt.show()
plt.savefig(f'{arbin_data_folder}/{ffolder}/{device_code} Norm Q delivered', dpi = 300)
plt.savefig(f'{arbin_data_folder}/Figures/{device_code}  Norm Q delivered', dpi = 300)
#%%
plt.figure()
plt.scatter(dfstat['Cycle Index'],dfstat['Coulombic Efficiency (%)'])
plt.ylim(80,105)
plt.show()
#%% Plotting %Q from CC step

fig, (ax1,ax2) = plt.subplots(2, sharex = True)
ax1.set_title('Percent Q from CC in Darken (%s)'%(device_code))
ax1.scatter(dfdarkcc.index.values,dfdarkcc['pqcc'],label = 'Percent Q from CC', color = '#ff7f0e')
ax2.set_ylabel('                                                   Percent Charge')
#fig.text(0.01, 0.5, 'Percent', ha='center', va='center', rotation='vertical')
ax1.set_ylim(0,110)
ax1.grid()
ax1.set_axisbelow(True)
ax2.set_title('Percent Q from CC in Bleach (%s)'%(device_code))
ax2.scatter(dfbleachcc.index.values,dfbleachcc['pqcc'],label = 'Percent Q from CC', color = '#1f77b4')
ax2.grid()
ax2.set_ylim(0,110)
ax2.set_xlabel('Cycle Number')
ax2.set_axisbelow(True)
# fig.tight_layout()
#ax2 = ax1.twinx()
#ax2.scatter(dfq.index.values,dfq['%qdiff'],label = 'Diff. Q/disQ', color = 'g',alpha = 0.1)
ax2.set_ylim(20,105)
#ax1.legend()
plt.show()
#%%
aa = dfdarkmin
aa['deriv']=aa['Voltage (V)'].diff()
plt.scatter(aa.index,aa['deriv'])

    
    
#%% EOD Oct 17

aa = dfdarkmin
wind = 30
aa['SmoothV'] = aa['Voltage (V)'].rolling(window=wind).mean()
fit_start = 100
cut_off = 0
num_rows = len(aa)
for index in range(num_rows -1, -1, -1):
    row = aa.iloc[index]
    if row['SmoothV'] < -1.695:
        fit_cut_off = index
fit_cut_off = 200    

coefficients = np.polyfit(np.array(aa.index[fit_start:fit_cut_off]),np.array(aa['SmoothV'])[fit_start:fit_cut_off],1)
# Create a polynomial function using the coefficients
poly = np.poly1d(coefficients)

# Evaluate the polynomial at any x value
y_fit = poly(np.array(aa.index))

plt.figure()
plt.plot(aa.index,aa['SmoothV'], label = 'rolling window average')
plt.plot(aa.index,aa['Voltage (V)'], label = 'raw data')
plt.plot(aa.index,y_fit,label = 'polyfit')
plt.legend()
plt.grid()
plt.show()

#%%
plt.scatter(df['Cycle Index'],df['Internal Resistance (Ohm)'])
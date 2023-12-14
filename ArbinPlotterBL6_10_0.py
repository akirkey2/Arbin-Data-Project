#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:31:43 2023

@author: aaronkirkey
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
display()
mpl.rcParams.update(mpl.rcParamsDefault)

mpl.interactive(True)
mpl.rcParams['lines.markersize'] = 2
plt.rc('axes', axisbelow=True)
plt.rcParams['figure.constrained_layout.use'] = True

device_names = [
    '',
    # '050047',
    #'2023050123, 2023050125'
    ]

ls = []
folder = ""

#Do I define all of my sub-df's globally? or do I put the plotter functions inside initdf

# To make a batch file work you may need to export dfbleach_ir, dfs2..... and compile those files in
# another .py in it's own df. Like dfs2all (for dfs2 for all devices you want to import)
# Another option is to do that here with with different device_codes + extract the columns you want.
# i.e. df.append(dfall.index + dfall['IR'] and give them unique names and keep doing that).
# import matplotlib.pyplot as plt
# x=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
# y=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
# for i in range(len(x)):
#     plt.figure()
#     plt.plot(x[i],y[i])
#     # Show/save figure as desired.
#     plt.show()
# # Can show all four figures at once by calling plt.show() here, outside the loop.
#plt.show()

arbin_folder = Path('C:\\Users\\Aaron Kirkey\\PythonProjects\\Arbin Data Project')
def idf(dev_name):
    # arbin_data_folder = Path('C:\\Users\\Aaron Kirkey\\PythonProjects\\Arbin Data Project')
    global dfall
    dfall = pd.DataFrame()
    for folder in os.listdir(arbin_folder): #Loops through folder in Arbin Data Project
        if str(dev_name) in folder: #Checks if device name is in any folders in Arbin Data Project
            global ls
            ls = []
            global ffolder
            ffolder = folder
            for i in os.listdir(str(arbin_folder)+'\\'+folder):#Checks for .csv files in the folder matching device name
                if 'Wb' in i and '.CSV' in i: #Appends .csv file to a list named lst
                    ls.append(i)
            print(ls)
            for i in range(len(ls)): #Makes a dict with key df1,df2,df3... and values are actual dataframes
                dfchunk = pd.read_csv(str(arbin_folder)+'\\'+folder+'\\'+str(ls[i]),usecols=['Test Time (s)','Step Time (s)',
                                                                      'Cycle Index', 'Step Index', 
                                                                      'Current (A)', 'Voltage (V)',
                                                                      'Charge Capacity (Ah)', 'Discharge Capacity (Ah)', 
                                                                      'Internal Resistance (Ohm)'])
#If you only want specific columns, use^^^
                #key=str('df'+str(i+1))
                #file = str(os.getcwd()+'\\'+folder+'\\'+ls[x])
                #dfchunk= pd.read_csv(folder+'\\'+str(ls[i])) #os.getcwd()+'\\'+ #FOR FULL RAW CSV
                dfall = dfall.append([dfchunk], ignore_index=True)

    # return dfall
    dfchunk = 0
    global device_code
    device_code = ls[0][:ls[0].find('_')]
    
    # Make dfstat dataframe
    # arbin_data_folder = Path('C:\\Users\\Aaron Kirkey\\PythonProjects\\Arbin Data Project')
    """
    global dfstat
    dfstat = pd.DataFrame()
    for folder in os.listdir(arbin_folder): #Loops through folder in Arbin Data Project
        if str(dev_name) in folder: #Checks if device name is in any folders in Arbin Data Project
            ls = []
            ffolder = folder
            for i in os.listdir(str(arbin_folder)+'\\'+folder):#Checks for .csv files in the folder matching device name
                if 'Stat' in i: #Appends .csv file to a list named lst
                    ls.append(i)
            print(ls)
            for i in range(len(ls)): #Makes a dict with key df1,df2,df3... and values are actual dataframes
                dfchunk = pd.read_csv(str(arbin_folder)+'\\'+folder+'\\'+str(ls[i]))
#If you only want specific columns, use^^^
                #key=str('df'+str(i+1))
                #file = str(os.getcwd()+'\\'+folder+'\\'+ls[x])
                #dfchunk= pd.read_csv(folder+'\\'+str(ls[i])) #os.getcwd()+'\\'+ #FOR FULL RAW CSV
                dfstat = dfstat.append([dfchunk], ignore_index=True)

    # return dfall
    dfchunk = 0
    #device_code = ls[0][:ls[0].find('_')]
    # dfstat.columns = ['Test Time (s)', 'Step Time (s)', 'Cycle Index', 'Step Index', 
                      # 'Current (A)', 'Voltage (V)', 'Power (W)','Charge Capacity (Ah)', 
                      # 'Discharge Capacity (Ah)', 'Charge Energy (Wh)','Discharge Energy (Wh)', 
                      # 'Charge Time (s)', 'Discharge Time (s)','V_Max_On_Cycle (V)', 
                      # 'Coulombic Efficiency (%)', 'mAh/g','blank']
    print(dfstat.head(5))
    if '/' not in str(dfstat.iloc[0,0]):
        dfstat.columns = ['Test Time (s)', 'Step Time (s)',
                'Cycle Index', 'Step Index', 'Current (A)', 'Voltage (V)', 'Power (W)',
                'Charge Capacity (Ah)', 'Discharge Capacity (Ah)', 'Charge Energy (Wh)',
                'Discharge Energy (Wh)', 'Charge Time (s)', 'Discharge Time (s)',
                'V_Max_On_Cycle (V)', 'Coulombic Efficiency (%)', 'mAh/g','blank']
        """
# Making df's grouped by cycle num + finding max and min values at each cycle num
    global dfmax
    dfmax = dfall.groupby(dfall['Cycle Index']).max()
    global dfmin
    dfmin = dfall.groupby(dfall['Cycle Index']).min()
    
    #Setting up dfs for bleach and darken max voltages & TTB, TTD
    global dfdark
    dfdark = dfall.loc[(dfall['Step Index'] == 6)]
    global dfdarkmax
    dfdarkmax = dfdark.groupby(dfdark['Cycle Index']).max()
    global dfbleach
    dfbleach = dfall.loc[(dfall['Step Index'] == 9)]
    global dfbleachmax
    global dfdarkmin
    dfbleachmax = dfbleach.groupby(dfbleach['Cycle Index']).max()
    dfdarkmin = dfdark.groupby(dfdark['Cycle Index']).min()
    # Making dfs for IR bleach & IR dark
    global dfbleach_ir
    dfbleach_ir = dfall.loc[(dfall['Step Index'] == 5)]
    global dfdark_ir
    dfdark_ir = dfall.loc[(dfall['Step Index'] == 5)]
    dfdark_ir = dfdark_ir.groupby(dfdark_ir['Cycle Index']).max()

# df's for Voltage Cutoffs - EOD July 13
    dfdarkmin = dfdark.groupby(dfdark['Cycle Index']).min()
    dfbleachmax = dfbleach.groupby(dfbleach['Cycle Index']).max()

# To split dfdark into CC and CV - EOD July 13
    if min(dfdarkmin['Voltage (V)']) < -1.75:
        dark_cc_cutoff = -1.898
        bleach_cc_cutoff = 0.798
    else:
        dark_cc_cutoff = -1.698
        bleach_cc_cutoff = 0.998
    global dfdarkcc
    global dfdarkcv
    dfdarkcc = dfdark.loc[(dfdark['Voltage (V)'] > dark_cc_cutoff)]
    dfdarkcv = dfdark.loc[(dfdark['Voltage (V)'] < dark_cc_cutoff)]
    dfdarkcc = dfdarkcc.groupby(dfdarkcc['Cycle Index']).max()
    dfdarkcv = dfdarkcv.groupby(dfdarkcv['Cycle Index']).max()
    dfdarkcc['cv'] = dfdarkcv['Discharge Capacity (Ah)']
    dfdarkcc['pqcc'] = (dfdarkcc['Discharge Capacity (Ah)'] / dfdarkcv['Discharge Capacity (Ah)'].quantile(0.99)) * 100
# To split dfbleach into CC and CV
    global dfbleachcc
    global dfbleachcv    
    dfbleachcc = dfbleach.loc[(dfbleach['Voltage (V)'] < bleach_cc_cutoff)]
    dfbleachcv = dfbleach.loc[(dfbleach['Voltage (V)'] > bleach_cc_cutoff)]
    dfbleachcc = dfbleachcc.groupby(dfbleachcc['Cycle Index']).max()
    dfbleachcv = dfbleachcv.groupby(dfbleachcv['Cycle Index']).max()
    dfbleachcc['pqcc'] = (dfbleachcc['Charge Capacity (Ah)'] / dfbleachcv['Charge Capacity (Ah)'].quantile(0.995)) * 100
# dfs + columns for % charge from cc and cv
    for i in (range(len(dfdarkcc))):
        if pd.isna(dfdarkcc['cv'].iloc[i]) == True:
            dfdarkcc['pqcc'].iloc[i] = 100 
        else:
            dfdarkcc['pqcc'].iloc[i] = (dfdarkcc['Discharge Capacity (Ah)'].iloc[i] / dfdarkcc['cv'].iloc[i]) * 100
    
    #dfs for charge delivered per cycle from wb.csv's
    global dfq
    dfq = dfmax.copy()
    dfq['%maxdq'] = dfq['Discharge Capacity (Ah)']/ (dfq['Discharge Capacity (Ah)'].quantile(0.99))
    dfq['%maxcq'] = dfq['Charge Capacity (Ah)'] / (dfq['Charge Capacity (Ah)'].quantile(0.99))
    #Charge delivered from Stats files
    dfstat['%cqd'] = (dfstat['Charge Capacity (Ah)'] / dfstat['Charge Capacity (Ah)'].iloc[0] ) *100
    dfstat['%dqd'] = (dfstat['Discharge Capacity (Ah)'] / dfstat['Discharge Capacity (Ah)'].iloc[0] ) *100
    print(dfstat.columns)
    print('IR@DARK AVG:' + str(dfdark_ir['Internal Resistance (Ohm)'].mean()))
    # return dfall, dfstat, device_code
#%%
# Plotting %Q from CC step
def qcc():
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
def save_qcc():
    qcc()
    plt.savefig(f'{arbin_folder}/{ffolder}/{device_code} %Q from CC', dpi = 300)
    plt.savefig(f'{arbin_folder}/Figures/{device_code} %Q from CC', dpi = 300)

# Voltage on Darken & Bleach vertically stacked
#dfstep2['Voltage (V)'].min()
def v():
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    #fig.suptitle('Voltage reached on Darken or Bleach')
    #ax1.scatter(dfmin.index.values, dfmin['Voltage (V)'],color = '#ff7f0e') #darken
    ax1.scatter(dfdarkmin.index.values, dfdarkmin['Voltage (V)'],color = '#ff7f0e',s=2) #Step2 min V. Slightly cleaner plots but removes tail that shows switching.
    ax1.set_title('Voltage reached on Darken (%s)' %(device_code))
    ax1.set_ylim(dfdarkmin['Voltage (V)'].quantile(0.01)-0.1,dfdarkmin['Voltage (V)'].quantile(0.99)+0.1)
    #ax2.scatter(dfmax.index.values, dfmax['Voltage (V)']) # bleach
    ax2.scatter(dfbleachmax.index.values, dfbleachmax['Voltage (V)'],s=2) #Step5 max V. Cleaner, but removes tails.
    ax2.set_title('Voltage reached on  Bleach (%s)' %(device_code))
    ax2.set_ylim(dfbleachmax['Voltage (V)'].quantile(0.01)-0.1,dfbleachmax['Voltage (V)'].quantile(0.99)+0.1)
    plt.xlabel('Cycle Number')
    fig.tight_layout()
    ax1.grid()
    ax2.grid()
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
      # otherwise the right y-label is slightly clipped
    plt.show()
    
def save_v():
    v()
    plt.savefig(f'{arbin_folder}/{ffolder}/{device_code} Vs_on_Dark_and_Bleach', dpi = 300)
    plt.savefig(f'{arbin_folder}/Figures/{device_code} Vs_on_Dark_and_Bleach', dpi = 300)
# IR, TTB, TTD all in one plot (working June 6, 2023)
def ir():
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
def save_ir():
    ir()
    plt.savefig(f'{arbin_folder}/{ffolder}/{device_code} IR, TTB & TTD', dpi = 300)
    plt.savefig(f'{arbin_folder}/Figures/{device_code} IR, TTB & TTD', dpi = 300)
    
# Plots for Normalized charge delivered per cycle
def q():
    fig, ax1 = plt.subplots()
    plt.title('Normalized Q Delivered (%s)'%(device_code))
    ax1.scatter(dfq.index.values,dfq['%maxdq'],label = '% max Q delivered on Darken', color = '#ff7f0e',s=2,marker="D")
    ax1.scatter(dfq.index.values,dfq['%maxcq'],label = '% max Q delivered on Bleach', color = '#1f77b4',s=1)
    ax1.set_ylabel('Normalized Charge')
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylim(dfq['%maxcq'].quantile(0.01)-0.05,1.05) #(dfq['%maxcq'].iloc[-1]-0.05)
    ax1.grid()
    ax1.set_axisbelow(True)
    #ax2 = ax1.twinx()
    #ax2.scatter(dfq.index.values,dfq['%qdiff'],label = 'Diff. Q/disQ', color = 'g',alpha = 0.1)
    #ax2.set_ylim(-0.03,0.03)
    ax1.legend(loc = 3)
    plt.show()
def save_q():
    q()
    plt.savefig(f'{arbin_folder}/{ffolder}/{device_code} %Q delievered', dpi = 300)
    plt.savefig(f'{arbin_folder}/Figures/{device_code} %Q delievered', dpi = 300)
def vp():
    global cycles
    cycles = [110]
    if len(cycles) == 1:
        for i in np.linspace(101,(dfall['Cycle Index'].max()-1),8):
            irr = int(round(i,-1))
            cycles.append(irr)
    else:
        cycles = cycles
    for i in cycles:
        if i >= dfall['Cycle Index'].max():
            cycles.remove(i)
        elif i <= 100:
            cycles.remove(i)
    cycles.append((dfall['Cycle Index'].max()-2))
    print(cycles)
    
    #Plotting voltage profiles
    plt.figure(figsize = (8,6))
    plt.grid()
    color_map = cm.viridis
    cycle_norm = plt.Normalize(min(cycles), max(cycles))
    for i in cycles:
        x = dfall.loc[(dfall['Cycle Index']== i)]
        plt.scatter((x['Test Time (s)']-x['Test Time (s)'].min()),x['Voltage (V)'],label = f'Cycle {i}',color=color_map(cycle_norm(i)))
        plt.legend(loc = 'best')
        plt.ylabel('Voltage')
        plt.xlabel('Cycle Time')
        plt.title(f'Voltage Profile for various cycles ({device_code})')
        plt.ylim()
    plt.show()
def save_vp():
    vp()
    plt.savefig(f'{arbin_folder}/{ffolder}/{device_code} Voltage Profiles at various cycles', dpi = 300)
    plt.savefig(f'{arbin_folder}/Figures/{device_code} Voltage Profiles at various cycles', dpi = 300)
def ip():
    global cycles
    cycles = [110]
    if len(cycles) == 1:
        for i in np.linspace(101,(dfall['Cycle Index'].max()-1),10):
            irr = int(round(i,-1))
            cycles.append(irr)
    else:
        cycles = cycles
    for i in cycles:
        if i >= dfall['Cycle Index'].max():
            cycles.remove(i)
        elif i <= 100:
            cycles.remove(i)
    cycles.append((dfall['Cycle Index'].max()-2))
    print(cycles)
    plt.figure(figsize=(8,6))
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
        plt.ylim()
    plt.show()
def save_ip():
    ip()
    plt.savefig(f'{arbin_folder}/{ffolder}/{device_code} Current Profiles at various cycles', dpi = 300)
    plt.savefig(f'{arbin_folder}/Figures/{device_code} Current Profiles at various cycles', dpi = 300)
def fe():
    plt.figure()
    plt.scatter(dfstat['Cycle Index'], dfstat['Coulombic Efficiency (%)'])  
    # plt.scatter(dfmax.index.values,dfmax['FE Arbin'],label='FE-arbin')
    plt.grid()
    plt.ylim(dfstat['Coulombic Efficiency (%)'].quantile(0.05)-2,dfstat['Coulombic Efficiency (%)'].quantile(0.95)+2)
    # plt.ylim(dfstat['Coulombic Efficiency (%)'].quantile(0.1)-1,dfstat['Coulombic Efficiency (%)'].quantile(0.9)+1)
    plt.axhline(y=100,color='r')
    plt.xlabel('Cycle Number')
    plt.ylabel('<-- More Qbleach     |     More QDark -->')
    plt.title(f'Coloumbic Efficiency (%) ({device_code})')
    plt.show()
def save_fe():
    fe()
    plt.savefig(f'{arbin_folder}/{ffolder}/{device_code} Coulombic Efficiency', dpi = 300)
    plt.savefig(f'{arbin_folder}/Figures/{device_code} Coulombic Efficiency', dpi = 300)

#%%
def pq():    
    plt.figure()
    plt.scatter(dfstat['Cycle Index'], dfstat['Charge Capacity (Ah)']*3600, label = 'Charge', s= 2, marker ='D')
    plt.scatter(dfstat['Cycle Index'], dfstat['Discharge Capacity (Ah)']*3600, label = 'Discharge', s = 1)
    # plt.scatter(dfmax.index.values,dfmax['FE Arbin'],label='FE-arbin')
    plt.grid()
    plt.ylim(dfstat['Charge Capacity (Ah)'].quantile(0.05)*0.95*3600,dfstat['Discharge Capacity (Ah)'].quantile(0.99)*1.05*3600)
    plt.xlabel('Cycle Number')
    plt.ylabel('Charge delivered')
    plt.title(f'Charge Delivered by Cycle ({device_code})')
    plt.legend()
    plt.show()
    # plt.savefig(f'{arbin_data_folder}/{ffolder}/{device_code} Norm Q delivered', dpi = 300)
    # plt.savefig(f'{arbin_data_folder}/Figures/{device_code}  Norm Q delivered', dpi = 300)
def save_pq():
    pq()
    plt.savefig(f'{arbin_folder}/{ffolder}/{device_code} Q delivered', dpi = 300)
    plt.savefig(f'{arbin_folder}/Figures/{device_code} Q delivered', dpi = 300)
# dfc101 = dfall.loc[(dfall['Cycle Index']==1755)]
# print(dfc101['Test Time (s)'].max()-dfc101['Test Time (s)'].iloc[0])
print('\n Initialize data: \tinitdf("Device number")\n \
V on D or B:\t\tv()\n \
IR + TTD/TTB:\t\tir()\n \
Charge by CC:\t\tqcc()\n \
Norm. Charge:\t\tq()\n \
Voltage Profile:\tvp()\n \
Current Profile:\tip()\n \
Plot all:\t\t\tplots()\n \
Save desired plot: save_v(),etc. \n \
Plot + save all:\tsave_plots()\n')
def plots():
    ir()
    v()
    qcc()
    vp()
    ip()
    fe()
    pq()

def save_plots():
    save_ir()
    save_v()
    save_qcc()
    save_vp()
    save_ip()
    save_pq()
    save_fe()
    plt.close('all')
#%% Code to recycle at some point
#dfir = dfall.loc[(dfall['Internal Resistance (Ohm)'].isna()==False)]
# dfc100 = dfall.loc[(dfall['Cycle Index']==2300)]
# data_folder = Path('C:\\Users\\Aaron Kirkey\\PythonProjects\\Arbin Data Project')
"""
for i in range(dfq['Cycle Index'].min(),dfq['Cycle Index'].max()):
    dfq['qcont']=dfq['Discharge Capacity (Ah)']/dfq['Discharge Capacity (Ah)'].iloc[-1]
    
dfq['qcont'].iloc[i] = dfq['Discharge Capacity (Ah)'].iloc[i]
    #%%
dfqmax = dfq.groupby(dfq['Cycle Index']).max()
dfdes = []
#%%
desired_v =  -1.69
for i in range(dfq['Cycle Index'].min(),dfq['Cycle Index'].max()):
    dfdes = dfq.loc[(dfq['Cycle Index'] == i) & (dfq['Voltage (V)'] > desired_v)]
    dfdes2 = dfq.loc[(dfq['Cycle Index'] == i) & (dfq['Voltage (V)'] < desired_v)]
    print(dfdes['Discharge Capacity (Ah)'].iloc[-1] / dfdes2['Discharge Capacity (Ah)'].iloc[-1])
#%%

#%% To match indices to graph using same x (cycle num)
for i in range(dfdark.index.values[-1]):
    if i in dfdark.index.values and i in dfbleach.index.values:
        continue
    elif i in dfdark.index.values:
        dfdark = dfdark.drop(i)
    elif i in dfdark.index.values:
        dfbleach = dfbleach.drop(i)
        
#%% Plotting Min voltage at each cycle num

plt.plot(dfmin.index.values,dfmin['Voltage (V)'])
#plt.ylim(-2.5,-1.0)
plt.title('Voltage Reached on Darken (%s)'%(device_code))
plt.xlabel('Cycle')
plt.ylabel('Voltage (V)')
plt.grid()
plt.show()
plt.savefig('%s_V_on_Dark'%(device_code), dpi = 300)
plt.plot

#%% Plotting Max voltage at each cycle num
plt.figure()
plt.plot(dfmax.index.values,dfmax['Voltage (V)'])
#plt.ylim(1.6,1.9)
plt.title('Voltage Reached on Bleach (%s)'%(device_code))
plt.xlabel('Cycle')
plt.ylabel('Voltage (V)')
plt.grid()
plt.show()
plt.savefig('%s_V_on_Bleach'%(device_code), dpi = 300)
"""
#%% Plotting internal resistance vs. test time
"""#dfall.loc[dfall['Internal Resistance (Ohm)'] > 300, 'Internal Resistance (Ohm)'] = 300
print(dfall['Internal Resistance (Ohm)'].max())

plt.figure()
plt.plot(dfall['Cycle Index'],dfall['Internal Resistance (Ohm)'])
#plt.ylim(10,750)
plt.title('Internal Resistance over time (%s)'%(device_code))
plt.xlabel('Cycle')
plt.ylabel('Internal Resistance (Ohm)')
plt.grid()
plt.show()
plt.savefig('%s_IR_vs_Time'%(device_code), dpi = 300)

#%% Plotting time to Darken Plot
plt.figure()
plt.plot(dfdark.index.values,dfdark['Step Time (s)'])
#plt.ylim(140,300)
plt.title('Time to Darken to 8 mC (%s)'%(device_code))
plt.xlabel('Cycle')
plt.ylabel('Time (s)')
plt.grid()
plt.show()
plt.savefig('%s_Time_to_darken'%(device_code), dpi = 300)

#%% Plotting time to Bleach Plot
plt.figure()
plt.plot(dfbleach.index.values,dfbleach['Step Time (s)'])
#plt.ylim(140,250)
plt.title('Time to Bleach to 8 mC (%s)'%(device_code))
plt.xlabel('Cycle')
plt.ylabel('Time (s)')
plt.grid()
plt.show()
plt.savefig('%s_Time_to_bleach'%(device_code), dpi = 300)

#%% Plotting TTD + TTB 
plt.figure()
plt.plot(dfdark.index.values,dfdark['Step Time (s)'])
plt.plot(dfbleach.index.values,dfbleach['Step Time (s)'])
#plt.ylim(140,300)
plt.title('TTB & TTD (%s) '%(device_code))
plt.xlabel('Cycle')
plt.ylabel('Time (s)')
plt.grid()
plt.show()
plt.savefig('%s TTB & TTD'%(device_code), dpi = 300)

#%% Dual y-axis TTB & TTD

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Cycle')
ax1.set_ylabel('TTD', color=color)
ax1.plot(dfdark.index.values,dfdark['Step Time (s)'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('TTB', color=color)  # we already handled the x-label with ax1
ax2.plot(dfdark.index.values,dfbleach['Step Time (s)'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()"""


#%% #Working on this code to plot Voltage for Darken or Bleach on same plot (dif y-axis) OR on two subplots

"""
plt.plot(dfmin.index.values,dfmin['Voltage (V)'])
plt.ylim(-2.5,-1.0)
plt.title('Voltage Reached on Darken (%s)'%(device_code))
plt.xlabel('Cycle')
plt.ylabel('Voltage (V)')
plt.grid()
plt.show()
plt.savefig('%s_V_on_Dark'%(device_code), dpi = 300)
plt.plot

#%% Plotting Max voltage at each cycle num
plt.figure()
plt.plot(dfmax.index.values,dfmax['Voltage (V)'])
plt.ylim(1.6,1.9)
plt.title('Voltage Reached on Bleach (%s)'%(device_code))
plt.xlabel('Cycle')
plt.ylabel('Voltage (V)')
plt.grid()
plt.show()
plt.savefig('%s_V_on_Bleach'%(device_code), dpi = 300)
"""

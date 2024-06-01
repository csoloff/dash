# import packages

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import matplotlib.dates as mdates

# define functions

def reader(path, first_var):
    '''
    Reads SEMS/DASH data, adds datetime columns

    :param path: path to data file
    :param first_var: the name of the first column label
    :return: pandas DataFrame
    '''
    # Open the file and read the lines
    skip=1
    with open(path, "r") as file:
        # Iterate over the lines
        for line in file:
            # Strip leading and trailing whitespace
            line = line.strip()
            # Check if the line contains column names
            if line.startswith(first_var):
                # Split the line by whitespace and append to the columns list
                columns = line[1:].strip().split("\t")
                break  # Stop reading lines after finding column names
            skip+=1
    # Read the data into a DataFrame, skipping the first 6 rows of comments
    d = pd.read_csv(path, sep='\t', skiprows=skip, names=columns)

    # Creates datetime columns
    if 'DOY.Frac' in d.keys():
        d['dt'] = pd.to_datetime('2024-1-1') + pd.to_timedelta(d['DOY.Frac'], unit='D') - pd.Timedelta(days=1)
    if 'StartTimeSt' in d.keys():
        d['st_dt'] = pd.to_datetime('2024-1-1') + pd.to_timedelta(d['StartTimeSt'], unit='D') - pd.Timedelta(days=1)
    if 'EndTimeSt' in d.keys():
        d['end_dt'] = pd.to_datetime('2024-1-1') + pd.to_timedelta(d['EndTimeSt'], unit='D') - pd.Timedelta(days=1)
    return d

def glob_reader(file_key, first_var, subfolder = './data/'):
    '''
    Reads groups of data files and merges them into one

    :param file_key: shared key in filenames
    :param first_var: the name of the first column label
    :param subfolder: name of the subfolder containing the data
    :return: pandas DataFrame
    '''
    paths = sorted(glob.glob(subfolder+'*'+file_key+'*'))
    d = []
    for i in range(0, len(paths)):
        d.append(reader(paths[i], first_var))
    d = pd.concat(d).reset_index()
    return d

def find_neg_flow(d, t_name = 'untitled'):
    '''
    Finds periods of negative flow out of the SEMS inlet

    :param d: pandas DataFrame containing SEMS data
    :param t_name: name of csv table to save data as
    :return: pandas DataFrame containing time periods of negative flow
    '''

    below_zero = d[d['UpSt_Samp'] < 0] # all 'UpSt_Samp' values below zero

    start_points = below_zero.index[~(below_zero.index.to_series().diff() == 1)]
    end_points = below_zero.index[~(below_zero.index.to_series().diff(-1) == -1)]
    
    # Calculate the duration of each block where the values are below zero
    durations = (d.loc[end_points, 'DOY.Frac'].values - d.loc[start_points, 'DOY.Frac'].values)*24*60*60 # units: seconds

    # Calculate the total time below zero
    total_time_below_zero = durations.sum()
    
    print('Total minutes below zero:', round(total_time_below_zero/60, 2))

    nf_d = pd.DataFrame(data={'Start_EST':d['dt'].iloc[start_points].reset_index(drop=True), 'End_EST':d['dt'].iloc[end_points].reset_index(drop=True), 'Start_index':start_points, 'End_index':end_points, 'Duration_s':pd.Series(durations)})

    nf_d.to_csv('./tables/nf-' + t_name+'.csv')

    return nf_d

def nf_plotter(sems, nf_d, param, dash, f_name = 'untitled'):
    '''
    Plots areas of negative flow and plots RH set point and measured values within DASH

    :param sems: sems pandas DataFrame
    :param nf_d: negative flow DataFrame
    :param param: sample parameters DataFrame
    :param dash: dash flow DataFrame
    :param f_name: name to add to end of saved figure
    :return: None
    '''

    fig, axes = plt.subplots(4, sharex=True)

    axes[0].plot(sems['dt'], sems['UpSt_Samp'], c='black')
    axes[0].set_ylabel('UpSt_Samp')
    for i in range(0,4):
            for j in range(0, len(nf_d)):
                    axes[i].axvspan(sems['dt'].loc[nf_d.loc[j]['Start_index']], sems['dt'].loc[nf_d.loc[j]['End_index']], color='red', alpha=0.2)
    
    x,y = [],[]
    for i in range(0, len(param)):
            row = param.iloc[i]
            x.extend([row['st_dt'], row['end_dt']])
            y.extend([row['RH_Targ'], row['RH_Targ']])

    axes[1].plot(x,y, c='black')
    axes[1].scatter(x,y, c='r', s=.5, zorder=100)

    axes[1].set_ylabel('RH_Targ')
    axes[2].plot(dash['dt'], dash['HM_RH'], c='black')
    axes[2].set_ylabel('HM_RH')
    axes[3].plot(dash['dt'], dash['HO_RH'], c='black')
    axes[3].set_ylabel('HO_RH')
    axes[3].set_xlabel('EST Time')
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.tight_layout()
    fig.savefig('./figures/neg_flow-'+f_name+'.png', dpi=300, bbox_inches='tight')

    plt.show()


def run_all_functions(subfolder='./data/', glob_append = '', xprt_f_name = 'untitled'):
    '''
    Finds negative flows, saves a table of times with negative flows, and plots the negative flow along with RH in DASH

    :param subfolder: specify the subfolder which contains the data (put all data of interest in same directory)
    :param xprt_f_name: the str you would like to append to the table and figure saved
    '''

    sems = glob_reader('SEMS_DATA'+glob_append, '#DOY.Frac', subfolder=subfolder)

    sems_nf = find_neg_flow(sems, t_name=xprt_f_name)

    param = glob_reader('SAMP_PARAM'+glob_append, '#StartTimeSt', subfolder=subfolder)

    dash = glob_reader('DASH_FLOW'+glob_append, '#DOY.Frac', subfolder=subfolder)

    nf_plotter(sems, sems_nf, param, dash, f_name=xprt_f_name)

# Calls function to run all other negative flow functions
# Put all the data including SEMS, SAMP_PARAM, and DASH_FLOW files in same directory and specify it in the function below after "subfolder=".
# Specify the name you would like to add at the end of the negative flow table and figure which will be saved

run_all_functions(subfolder='./data/DASH-test_fight-2024_05_17/', xprt_f_name = '2024_05_24')
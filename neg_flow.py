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

def glob_reader(file_key, first_var):
    '''
    Reads groups of data files and merges them into one

    :param file_key: shared key in filenames
    :param first_var: the name of the first column label
    :return: pandas DataFrame
    '''
    paths = sorted(glob.glob('./data/*'+file_key+'*'))
    d = []
    for i in range(0, len(paths)):
        d.append(reader(paths[i], first_var))
    d = pd.concat(d).reset_index()
    return d

def find_neg_flow(d):
    
    below_zero = d[d['UpSt_Samp'] < 0]

    start_points = below_zero.index[~(below_zero.index.to_series().diff() == 1)]
    end_points = below_zero.index[~(below_zero.index.to_series().diff(-1) == -1)]
    #print(start_points)
    # Calculate the duration of each block where the values are below zero
    durations = d.loc[end_points, 'DOY.Frac'].values - d.loc[start_points, 'DOY.Frac'].values

    # Calculate the total time below zero
    total_time_below_zero = durations.sum()
    print([d['dt'].iloc[start_points], d['dt'].iloc[end_points], durations])
    nf_d = pd.DataFrame(data=np.array([d['dt'].iloc[start_points], d['dt'].iloc[end_points], durations]).T, columns=['Start', 'Stop', 'Duration'])

    return nf_d

sems = glob_reader('SEMS', '#DOY.Frac')

sems_nf = find_neg_flow(sems)
#test
print(sems_nf)
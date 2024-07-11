# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:44:58 2024

@author: hilario

Version 3
    - 20240601: This version of the script outputs retrieved data per chunk of 
    data so progress is not all-or-nothing. Copy pasted from v2_RESAMP then made edits.

Version 2_RESAMP: Applied to DASH_SAMP_PARAM_RESAMP
    - Adjusted filter values so they don't do anything
    - This is so I can use DASH_SAMP_PARAM_RESAMP (from Cassidy) with minimal changes to my own script
    
Jug implementation on RI and GF retrieval. This script processes all DASH data in a folder based on DASH_SAMP_PARAM files in folder. 
Processing is done per instance of DASH_SAMP_PARAM
After processing all data in folder, the script also outputs a combined version of the data with all data in the folder combined into one file.
This script was formerly named process_DASH_jug.py

Timing/speed:
    - 42 s (v2, 4 parallel processes)
    - 205 s (v2, no parallel processing)
    - 179 s (v1, no parallel processing)

Links:
    - Parallelization applied chunk-wise instead of file-wise.
    - Combine jug outputs to save in one file: https://github.com/grleung/satlcc/blob/main/bootstrap_mean_deforest_effect.py
    - Use argparse instead of sys.argv to read arguments
        - https://stackoverflow.com/questions/47637987/how-to-provide-command-line-arguments-to-save-a-file-to-a-particular-location
        - https://docs.python.org/3/library/argparse.html
    
To do:
    - Add overwrite argument for skipping files already processed

Relevant jug commands:
    Open Anaconda Prompt
    Change directory
        cd C:\\Users\hilario\OneDrive - University of Arizona\Python\\ARCSIX_DASH\scripts
    Clear old jug runs
        jug invalidate "C:\\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240524\\"  --target process_data
        jug cleanup "C:\\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240524\\" 
    Execute jug run (1 on each Anaconda Prompt):
        jug execute "C:\\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240524\\" 
        For multiple executes at once, see below (L~40)
    Check status of jug run
        jug status "C:\\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240524\\" 
        
"""
# %%
'''To print a statement that runs multiple commands at once'''
# folder = '240613'
# script_dir = 'C:\\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\\2_run_retrieval_jug_v3.py'
# folder_dir = f'C:\\Data\ARCSIX\Raw\\{folder}\\' #If flight data # folder_dir = f'C:\\Data\DASH\Lab\Raw\\{folder}\\' #If lab data
# cmd = f'start jug execute "{script_dir}" "{folder_dir}" & '#  & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240524\\" & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240524\\" & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240524\\" & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240524\\" & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240524\\" 
# print([cmd*6])

'''Backup command. Need to manually change folder date'''
# start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240517\\" & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240517\\" & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240517\\" & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240517\\" & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240517\\" & start jug execute "C:\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\scripts\run_retrieval_jug_v3.py" "C:\\Data\ARCSIX\Raw\\240517\\" 

#%%
proj_dir = 'C:\\Users\hilario\OneDrive - University of Arizona\Python\ARCSIX_DASH\\'
inputs_dir = f'{proj_dir}inputs\\' #Folder with parameters, OPC bins, etc.
meta_dir = f'{inputs_dir}meta\\'    
outputs_dir = f'{proj_dir}outputs\\retrieved_data\\' 
repo_dir = 'C:\\Users\hilario\OneDrive - University of Arizona\Python\\repository\\DASH\\' 
    #repo_dir: Contains DASH functions and calibration surface

import datetime as dt
from datetime import time
import sys
sys.path.append(f'{repo_dir}scripts\\')
import dash_functions_arcsix as fx
import os
import argparse
import pandas as pd
import numpy as np
from jug import TaskGenerator
import math

def get_params():
    df = pd.read_csv(f'{meta_dir}params.txt').set_index('variable')
    return df['value'].to_dict()

'''Reading metadata, parameters'''
# save_name_list = ['OPC_DATA', 'QA_STATS', 'ARCSIX-DASH_P3B'] #Saved files will start with these strings
params = get_params()
bins = fx.read_bins(meta_dir)
flight_times = pd.read_csv(f'{meta_dir}ARCSIX_takeoff_landing_times.txt',
                           parse_dates = ['Takeoff_UTC', 'Landing_UTC']
                           ).set_index('FltNum')
line_breaks = pd.read_csv(f'{meta_dir}ARCSIX_DASH_SEMS_switch_times.txt',
                            parse_dates = ['Switch_Start_UTC', 'Switch_Stop_UTC']
                            )[['FltNum', 'Switch_Start_UTC', 'Switch_Stop_UTC', 
                               'LARGE_From', 'LARGE_To']]

#Defining parameters
#20250530: Adjusted filters so they are essentially non-functional. Will rely on Cassidy's post-processing sampling script
max_RHstd = 1e4 #float(params['max_RHstd'])
low_conc_rel_thresh = 0 #float(params['low_conc_rel_thresh'])
low_conc_abs_thresh = 0 #float(params['low_conc_abs_thresh'])
RH_col = params['RH_col']
opc_dict = {k: params[k] for k in ('DOPC', 'HOPC')}
percentile = 0.5 #For now, use only 50th percentile
percentile_list = [0.25, 0.5, 0.75] #Percentiles to calculate for RI
#%%
'''Importing calibration surfaces for both HOPC and DOPC'''
full_idw_dict, compressed_idw_dict = {}, {}
    #Full IDW is used for retrieving HOPC distribution given SEMS Dp and a wet RI
    #Compressed IDW is used for retrieving dry RI

print('Reading calibration surface')
for opc_name in ['DOPC', 'HOPC']:
    idw = pd.read_csv(f'{repo_dir}inputs\\idw_cdf_{opc_name}.csv') #Only DOPC calib needed for RI
    full_idw_dict[opc_name] = idw.drop_duplicates().reset_index()
    
    idw_3d = pd.read_csv(f'{repo_dir}inputs\\idw_cdf_{opc_name}_3d.csv') #Only DOPC calib needed for RI
    compressed_idw_dict[opc_name] = idw_3d
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='Folder with raw DASH data')
    return parser.parse_args()

def total_seconds(t): #https://stackoverflow.com/questions/44823073/convert-datetime-time-to-seconds
    return (t.hour * 60 + t.minute) * 60 + t.second

def calc_mid_time(min_time, max_time):
    return ((max_time - min_time) / 2) + min_time
    
def export(output, save_dir):
    # 20240601: 2_run_retieval_jug_V3 saves the outputs of each parallelized process so progress is less all-or-nothing
    if not os.path.exists(save_dir): #Makes subfolder
        os.mkdir(save_dir)

    output.drop(columns = [i for i in output if '_rounded' in i], inplace = True) #Removes SEMS_Dp_rounded
    output.sort_values(by = 'Start_Date_Time_UTC', inplace = True) #Sort rows by time and OPC

    #Calculating midpoint time
    output['Mid_Date_Time_UTC'] = [calc_mid_time(*i) for i in output[['Start_Date_Time_UTC', 'Stop_Date_Time_UTC']].values]
    #Calculating seconds since UTC midnight for Start, Mid, Stop
    for time_label in ['Start', 'Mid', 'Stop']:
        output[f'UTC_{time_label}_Time'] = output[
            f'{time_label}_Date_Time_UTC'].apply(total_seconds) #https://pynative.com/python-datetime-to-seconds/
    assert all(output.UTC_Start_Time <= output.UTC_Mid_Time)
    assert all(output.UTC_Mid_Time <= output.UTC_Stop_Time)
    
    start_t, stop_t = output.Start_Date_Time_UTC.min(), output.Stop_Date_Time_UTC.max()
    save_name = f'{start_t.strftime("%Y%m%d%H%M%S")}-{stop_t.strftime("%Y%m%d%H%M%S")}'
    output.to_csv(f'{save_dir}retrieval_output_{save_name}.csv', index = None)

@TaskGenerator
def retrieve(chunk):
    
    if chunk is None: 
        #If no usable chunk from read_all()
        return None
    else:
        dopc = chunk.loc[chunk.index.get_level_values('OPC') == 'DOPC']
        hopc = chunk.loc[chunk.index.get_level_values('OPC') == 'HOPC']
        
        if len(dopc) > 0: #Making sure chunk is not empty
            '''Retrieve dry RI'''
            # bin_cols = [i for i in dopc if type(i) == float]
            RI_dry = dopc.apply(fx.retrieve_dry_RI, idw_3d = compressed_idw_dict['DOPC'], 
                                apply_filters = True, axis = 1, result_type='expand')
            RI_dry.columns = [f'RI_q{int(i*100)}' for i in percentile_list]
            
            # for j in range(len(dopc.index)):
            #     fx.retrieve_dry_RI(dopc.iloc[j], idw_3d = compressed_idw_dict['DOPC'])
                # dopc[dopc.sum(axis=1) > 0]
                # dopc.iloc[j].plot(); hopc.iloc[j].plot()
        
            if not RI_dry.dropna().empty: #Making sure chunk is not empty
                '''Retrieve GF'''
                l = []
                for i in range(len(RI_dry.index)): #Running on data where dry RI was retrieved
                    print(100*i/len(RI_dry.index))
                    #To do: for now I just use the median, but I want to apply this to the 
                    #other percentiles so we have a range of GF. For later, after I apply jug
                    l.append([*RI_dry.iloc[i].name, 
                              RI_dry.iloc[i]['RI_q50'], 
                              fx.retrieve_GF(dopc.iloc[i], 
                                              hopc.iloc[i],  
                                              RI_dry.iloc[i]['RI_q50'], 
                                              calib_sfc = full_idw_dict['HOPC'])
                              ])
                output = pd.DataFrame(l)
                output.columns = [*RI_dry.index.names, 'RI_dry', 'GF']
                output.drop(columns = 'OPC', inplace = True)
                assert len(output['RI_dry'].dropna()) >= len(output['GF'].dropna()), 'GF was somehow retrieved without RI'
                
                save_dir = f'{outputs_dir}{folder_date_str}\\partials\\'
                export(output, save_dir)#combining and saving is done in export()
                return output 
            
def chunk_data(data):
    if data is None: #If no usable data from read_all()
        return [None]
    else:
        #Chunking is based on Start_Date_Time_UTC to keep DOPC and HOPC data together
        unique_index = data.index.get_level_values('Start_Date_Time_UTC').unique()

        #Calculating chunk length (len_chunks; i.e., number of rows in a chunk)
        len_chunks = min(len(unique_index), math.ceil(len(unique_index)/8), 25)
        N_chunks = math.ceil(len(unique_index)/len_chunks)
            #minimum chunk length is len(df)
            #max chunk length is 
        #Converts dataframe into a list of chunks (type dataframe)
        indx_list = np.array_split(unique_index, N_chunks)
        return [data.loc[data.index.get_level_values('Start_Date_Time_UTC').isin(indx)] for indx in indx_list]

'''Reading SEMS+DASH periods'''
def get_sems_periods(flight_date):

    #Filtering for RF
    #This one uses exclusive or (^) but that gives True ^ True = False. Changed to line below it
    # line_breaks_RF = line_breaks.loc[(line_breaks['Switch_Start_UTC'].dt.date == flight_date.date()) ^
    #                                  (line_breaks['Switch_Stop_UTC'].dt.date == flight_date.date())]
    line_breaks_RF = line_breaks.loc[[((line_breaks.loc[i,'Switch_Start_UTC'].date() == flight_date.date()) 
                                       or (line_breaks.loc[i,'Switch_Stop_UTC'].date() == flight_date.date())) 
                                      for i in line_breaks.index]]
    
    #Making sure rows are ordered before we reset index for easier querying
    # line_breaks_RF.sort_values('Switch_Start_UTC', inplace = True)
    line_breaks_RF.reset_index(drop = True, inplace = True)
    
    #Replace NaT with dummy value based on available timestamp
    for i in line_breaks_RF.index:
        if type(line_breaks_RF.loc[i, 'Switch_Stop_UTC']) != pd.Timestamp:
            line_breaks_RF.loc[i, 'Switch_Stop_UTC'] = line_breaks_RF.loc[i, 'Switch_Start_UTC']+dt.timedelta(seconds = 1)
        if type(line_breaks_RF.loc[i, 'Switch_Start_UTC']) != pd.Timestamp:
            line_breaks_RF.loc[i, 'Switch_Start_UTC'] = line_breaks_RF.loc[i, 'Switch_Stop_UTC']-dt.timedelta(seconds = 1)
        
    #Adding rows for first (before first switch) and last (after last switch) timestamps 
    first_dt = dt.datetime.combine(line_breaks_RF.loc[0, 'Switch_Start_UTC'], time.min) 
    last_dt = dt.datetime.combine(line_breaks_RF.loc[len(line_breaks_RF)-1, 'Switch_Stop_UTC'], time.max)
    line_breaks_RF.loc[-1] = [np.nan, np.nan, first_dt, 'OFF', 'SEMS']
    line_breaks_RF.loc[len(line_breaks_RF)-1] = [np.nan, last_dt, np.nan, 'SEMS', 'OFF']
    line_breaks_RF.index += 1
    line_breaks_RF.sort_index(inplace = True)
    # line_breaks_RF.sort_values(by = ['Switch_Start_UTC', 'Switch_Stop_UTC'], inplace = True)
    
    #Filtering only for SEMS switches
    sems_switches = (line_breaks_RF[['LARGE_From', 'LARGE_To']] == 'SEMS').values.sum(1)
    line_breaks_RF = line_breaks_RF[sems_switches==1] #These are the switches involving the SEMS
    
    output = [(math.floor(i/2), 
               line_breaks_RF.loc[i, f'Switch_{"Stop" if i%2==0 else "Start"}_UTC']) for i in line_breaks_RF.index]
    groups = {}
    for l in output:
        groups.setdefault(l[0], []).append(l[1])
    
    output_sorted = list(groups.values())
    return output_sorted
    #Output is a list of [[start1, stop1], [start2, stop2], ...]
    
def make_mask(data_dt, start_stop_times):
    #start_stop_times should be [[start1, stop1], [start2, stop2], ...]
    #Loops over start, stop times in sems_periods and looks for timestamps in data that fall between them
    # data_dt = opc_data['DOPC'].Date_Time_UTC #for testing
    return pd.concat([data_dt.between(*i) for i in start_stop_times], axis=1).apply(any, axis = 1)

def read_all(folder, samp_param_str = 'DASH_SAMP_PARAM_RESAMP', skip_done = True):
    # folder = "C:\\Data\ARCSIX\Raw\\240528\\"  #For testing
    #Reads all data in folder and concatenates into dataframes for later processing
    opc_data = dict([(i, fx.read_mfdata(folder, f'OPC_{opc_dict[i]}')) for i in opc_dict.keys()])
    
    if 'RESAMP' in samp_param_str:
        samp_param = fx.read_mfdata(f'{inputs_dir}resamp\\', folder.split('\\')[-2])
    else:
        samp_param = fx.read_mfdata(folder, samp_param_str)
    
    print('Using samples from', samp_param_str) #So I know which SAMP_PARAM is being used
    
    '''Assert that stop of previous row is not after the start of the current row'''
    overlap_check = sum(~(samp_param.Stop_Date_Time_UTC.shift(1) <= samp_param.Start_Date_Time_UTC)[1:])
    assert overlap_check == 0, f'Overlap found for {overlap_check} samples'
    # all((samp_param.Stop_Date_Time_UTC.shift(1) <= samp_param.Start_Date_Time_UTC)[1:])
    # samp_param.loc[(samp_param.Stop_Date_Time_UTC.shift(1) <= samp_param.Start_Date_Time_UTC)]
    
    '''Cropping for SEMS+DASH data. Removing periods without SEMS (i.e., DASH only)'''
    try:
        sems_periods = get_sems_periods(folder_date) 
            #Each element of sems_periods is start, stop times of when SEMS was being used
            #To remove DASH only data, remove anything NOT between start, stop times
            
        #Apply mask function separately to HOPC and DOPC because we can still use DOPC even without HOPC (i.e., RI retrieval)
        for opc_name in opc_data.keys():
            opc_data[opc_name] = opc_data[opc_name][
                make_mask(opc_data[opc_name].Date_Time_UTC, sems_periods)]
            
        #Apply mask function to samp_param such that we account for sample start/stop times
        samp_mask = pd.concat([make_mask(samp_param.Start_Date_Time_UTC, sems_periods),
                              make_mask(samp_param.Stop_Date_Time_UTC, sems_periods)], 
                             axis = 1).apply(all, axis = 1)
        samp_param = samp_param[samp_mask]
        assert opc_data['DOPC'].shape[0] > 0, 'Check SEMS-DASH timestamps'
        print('Data cropped for SEMS-DASH timestamps. DASH only (no SEMS) periods removed.')
    except KeyError:
        print('No sample line breaks found. Including all data')
    
    '''Cropping for LARGE filter'''
    try:
        date = pd.to_datetime(os.path.basename(folder[:-1]), format = '%y%m%d').date()
        flight = flight_times[[date==i for i in flight_times.Takeoff_UTC.dt.date.values]]
        start, stop = flight.LARGE_Filter_Off_UTC.values[0], flight.LARGE_Filter_On_UTC.values[0]
        for opc_name in opc_dict.keys():
            opc_data[opc_name] = opc_data[opc_name].loc[(start <= opc_data[opc_name].Date_Time_UTC) &
                                  (opc_data[opc_name].Date_Time_UTC <= stop)]
        samp_param = samp_param.loc[(start <= samp_param.Start_Date_Time_UTC) &
                                    (samp_param.Stop_Date_Time_UTC < stop)]
        print('Data cropped for takeoff and landing')
    except IndexError: #if no takeoff/landing times found in flight_times (ARCSIX_takeoff_landing_times.csv)
        print('No LARGE filter times found. Including all data')
    
    try:
        '''Filter out samples with high RH std'''
        samp_param = samp_param[(samp_param['HM_RH_Sdev'] < max_RHstd) & 
                                (samp_param['HO_RH_Sdev'] < max_RHstd)] 
        
        '''Filter out samples with erroneous SEMS Dp'''
        #This happened on 240528_115034 data
        samp_param = samp_param[samp_param['UpSt_Dia'] > 150]
        
        if not samp_param.empty: #Check if samp_param is empty
            print('Before filtering/cleaning\n'
                  f'N Samples: {len(samp_param)}\n'
                  f'N DOPC distributions: {len(opc_data["DOPC"])/2}\n'
                  f'N HOPC distributions: {len(opc_data["HOPC"])/2}')
            
            #%%
            # '''Creating a new column for rounded RH values'''
            # samp_param[f'{RH_col}_rounded'] = samp_param[RH_col].round()#apply(fx.round_RH)
        
            '''Cleaning OPC data'''
            opc_clean, qa_stats = [], []
            for opc_name in opc_data:
                data = opc_data[opc_name]

                '''Processing data'''
                #Sum per sample
                data = fx.calc_dNdlogDp_persamp(data, bins, samp_param)

                #Appending RH, Dp columns
                data = pd.concat([data, samp_param[f'{RH_col}'], samp_param['UpSt_Dia']], axis = 1)
                data.rename(columns = {f'{RH_col}': 'RH', 'UpSt_Dia': 'SEMS_Dp'}, inplace = True)
                
                #OPC column
                data['OPC'] = opc_name
                
                data.set_index(['Start_Date_Time_UTC', 'Stop_Date_Time_UTC', 
                                'RH', 'SEMS_Dp', 'OPC'], inplace = True)
                
                '''Filtering low-count data'''
                '''20240528: commented out OPC data filters because I let Cassidy do that (Min Counts)'''
                #Create mask for relatively low concentration scans (relative to scans at other RHs; same species, size)
                #Assumes that concentrations should be similar per size across different RHs
                #If total concentration for some size bin is < 2% of mean total concentration for that size across different samples, then replace with NaN
                
                # data_sum = data.sum(axis=1).unstack()
                # data_sum /= data_sum.mean()/100 #Converting to mean-normalized value
                # rel_conc_mask = (data_sum < low_conc_rel_thresh).stack() #False if concentration is high enough for good signal
                # n_remv_mask = len(data[rel_conc_mask[rel_conc_mask.index.isin(data.index)]])
                
                #Replace scans that have low concentrations with NaN
                #To do: revisit this and make sure it works. 
                # 20240529: I had to add a minimum conc threshold to retrieve_dry_RI because of an all-zero OPC dist
                abs_conc_mask = data.sum(axis=1, numeric_only = True) < low_conc_abs_thresh
                data[abs_conc_mask] = np.nan
                n_remv_tot = sum(abs_conc_mask)
                
                # abs_conc_mask = abs_conc_mask[abs_conc_mask]
                # n_remv_abs = len(data[data.index.isin(abs_conc_mask.index)])
                # n_remv_tot = sum(abs_conc_mask) #sum((rel_conc_mask[rel_conc_mask.index.isin(data.index)]) & (data.index.isin(abs_conc_mask.index)))
                # print(f'{opc_name} low count: {np.round(100*(n_remv_tot)/len(data))}%')
                
                # data[rel_conc_mask[rel_conc_mask.index.isin(data.index)]] = np.nan
                
                #In case I want to plot the OPC distributions in the file
                # ax=data.T.plot();ax.set_xlim([150,200]);ax.set_title(opc_ID)
                
                #Rounding size and cleaning up multiindex order
                data.reset_index(inplace = True)
                # data['SEMS_Dp_rounded'] = [fx.round_to_base(i, base = 10) for i in data['SEMS_Dp']] 
                # data.set_index('TC', inplace = True)
                data.set_index(['Start_Date_Time_UTC',
                                'Stop_Date_Time_UTC', 'RH', 'SEMS_Dp', 
                                # 'SEMS_Dp_rounded', 
                                'OPC'], 
                                inplace = True)
                
                #Record data quality per scan
                qa = pd.DataFrame([opc_name, 
                                   len(data),
                                   np.nanmax(data.to_numpy()),
                                   np.nanmean(data.to_numpy()),
                                   np.nanmedian(data.to_numpy()),
                                   data.sum(axis=1).quantile(0.75),
                                   n_remv_tot, 
                                   (n_remv_tot)/len(data)],
                                  index = ['OPC_ID', 'N', 'MaxConc', 'MeanConc', 'Q50Conc', 
                                           'Q75Conc', 'Num_remv', 'Prc_remv']).T
                    #To do: calculate sum N properly (undo dlogDp)
                opc_clean.append(data)
                qa_stats.append(qa)
            
            '''Setting up output folder'''
            save_dir = f'{outputs_dir}{folder_date_str}\\'
            if not os.path.exists(save_dir): #Makes subfolder
                os.mkdir(save_dir)
                
            '''Saving outputs'''
            #20240603: Since I added the save progress, the opc_clean here is no longer the complete CSV. Only those that haven't been done
            opc_clean = pd.concat(opc_clean)
            qa_stats = pd.concat(qa_stats).set_index('OPC_ID')
            # opc_clean.replace(np.nan, -9999).to_csv(f'{save_dir}{save_name_list[0]}_{folder_date_str}.csv') # DC3-DASH-HYGRO_DC8_20120518_R2.ICT
            # qa_stats.to_csv(f'{save_dir}{save_name_list[1]}_{folder_date_str}.csv')
            
            '''Cropping for data rows that have already been processed/retieved'''
            '''If skip done, then remove data that's already been retrieved'''
            if skip_done:
                partials_dir = f'{save_dir}partials\\'
                if not os.path.exists(partials_dir): #Makes subfolder
                    os.mkdir(partials_dir)
                skip_dt = []
                for i in os.listdir(partials_dir):
                    skip_dt.append([*pd.to_datetime(i.split('_')[-1][:-4].split('-'), format = "%Y%m%d%H%M%S")])
                if len(skip_dt) > 0:
                    done_list = make_mask(
                        opc_clean.index.get_level_values('Start_Date_Time_UTC')
                        .to_frame()['Start_Date_Time_UTC'], skip_dt)
                    opc_clean = opc_clean[list(~done_list)]
                    
            #For testing
            # opc_clean = opc_clean.loc[(opc_clean.index.get_level_values('Start_Date_Time_UTC') >= '2024-05-30 15:04:00') &
            #               (opc_clean.index.get_level_values('Start_Date_Time_UTC') <= '2024-05-30 15:45:00')]
            # assert opc_clean.shape[0] > 0, 'Comment out testing filter'
            
            print('After filtering/cleaning\n'
                  f'N Samples: {len(samp_param)}\n'
                  f'N DOPC distributions: {len(opc_clean.index.get_level_values("OPC")=="DOPC")}\n'
                  f'N HOPC distributions: {len(opc_clean.index.get_level_values("OPC")=="HOPC")}')
            
            if len(opc_clean) == 0:
                print('All OPC data has been processed. Retrieval done')
                return None
            elif all(opc_clean.sum(axis=1)==0):
                print('All remaining OPC data has zero counts. Retrieval done')
                return None
                            
            return opc_clean #Return cleaned OPC data for RI and GF retrieval
        else:
            print(f'RHstd too high for all data in {folder}.')
            return None
    except TypeError: #If samp_param has no samples in it, then return None
        print(f'No samples found in {folder}.')
        return None

'''Reading data in folder'''
# folder = "C:\\Data\ARCSIX\Raw\\240612\\" #For testing; comment out when using command prompt
# folder = "C:\\Data\DASH\Lab\Raw\\240214\\"
args = get_args()
folder = args.folder
folder_date = dt.datetime.strptime(folder.split('\\')[-2], '%y%m%d')
folder_date_str = folder_date.strftime('%Y%m%d')

'''Applying retrieval to data. Using RESAMP file from Cassidy'''
[retrieve(i) for i in chunk_data(read_all(folder))]

'''Applying retrieval if I use original DASH_SAMP_PARAM'''
# [retrieve(i) for i in chunk_data(read_all(folder,
                                          # samp_param_str = 'DASH_SAMP_PARAM'))]

'''For troubleshooting'''
# data_all = read_all(folder, samp_param_str = 'DASH_SAMP_PARAM')

# #For testing
# data_all = data_all.loc[(data_all.index.get_level_values('Start_Date_Time_UTC') >= '2024-05-30 15:04:00') &
#              (data_all.index.get_level_values('Start_Date_Time_UTC') <= '2024-05-30 15:07:00')]

# chunks = chunk_data(data_all)
# for n, chunk in enumerate(chunks):
#     print(n, len(chunk))
#     output = retrieve(chunk) #v3: export included in this

'''Archived'''
# def skip_done(data, save_dir):
#     '''If skip done, then remove rows that have already been retrieved'''
#     # save_dir = f'{outputs_dir}{folder_date_str}\\partials\\'
#     skip_dt = []
#     for i in os.listdir(save_dir):
#         skip_dt.append([*pd.to_datetime(i.split('_')[-1][:-4].split('-'), format = "%Y%m%d%H%M%S")])
#     done_list = make_mask(
#         data.index.get_level_values('Start_Date_Time_UTC')
#         .to_frame()['Start_Date_Time_UTC'], skip_dt)
#     return data[list(~done_list)]
        

#     #nan eror happens at start_time = 2024-05-30 17:26:56.497920
#     # break

    # For testing; shortens data
    # start = data.index.get_level_values('Start_Date_Time_UTC')[50]
    # stop = data.index.get_level_values('Start_Date_Time_UTC')[53]
    # data = data.loc[(start <= data.index.get_level_values('Start_Date_Time_UTC')) &
    #           (data.index.get_level_values('Start_Date_Time_UTC') <= stop)]
    
#     #This is one of the rows that gives the error, maybe
#     # output = retrieve(data_all.loc[(data_all.index.get_level_values('Start_Date_Time_UTC') >= '2024-05-30 17:26:56.497920') &
#     #                       ('2024-05-30 18:15:19.200960' < data_all.index.get_level_values('Start_Date_Time_UTC'))])
    

# import matplotlib.pyplot as plt
# plt.style.use('custom')
# data600 = data_all.loc[data_all.index.get_level_values('SEMS_Dp') == 600]
# data600.loc[data600.index.get_level_values('OPC') == 'DOPC'].T.plot(legend = None)
# data600.loc[data600.index.get_level_values('OPC') == 'HOPC'].T.plot(legend = None)

# groups = data_all.loc[(data_all.index.get_level_values('Start_Date_Time_UTC') >= '2024-05-17 18:30') & 
#              (data_all.index.get_level_values('Start_Date_Time_UTC') <= '2024-05-17 18:35')
#              ].groupby('OPC')
# for opc_name, group in groups:
#     ax = group.sum().plot(label = opc_name)
#     ax.set_xlim([150,500])

'''Below is from version 1'''
#This works correctly
# chunks = chunk(read_all(folder))
# output = [retrieve(i[::2]) for i in chunks[:2]]
# export([output, output, output], f'{outputs_dir}{folder_date_str}\\')

# '''Processing data and saving cleaned data and data quality summary'''
# export(
#     [process_data(datetime_tag) 
#           for datetime_tag in unique_datetime], 
#     f'{outputs_dir}{folder_date_str}\\')

# print(time()-start)

# '''Some lines useful for troubleshooting'''
# save_dir = f'{outputs_dir}{folder_date_str}\\'

# l = []
# for datetime_tag in unique_datetime[:2]:
#     l.append(process_data(datetime_tag))
    
# export(l, f'{outputs_dir}{folder_date_str}\\')


# out, dq = process_data(unique_datetime[0], 'HOPC')
# output = ([process_data(datetime_tag, 'DOPC') for datetime_tag in unique_datetime], 
#           [process_data(datetime_tag, 'HOPC') for datetime_tag in unique_datetime])
# output = [process_data(datetime_tag, opc_name) for datetime_tag in unique_datetime for opc_name in opc_dict.keys()]
# output = list(map(list, zip(*output))) #https://stackoverflow.com/questions/6473679/transpose-list-of-lists
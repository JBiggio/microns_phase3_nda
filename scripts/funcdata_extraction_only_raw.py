import os
import gc
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
#from sklearn.metrics import r2_score
from ccmodels.preprocessing.connectomics import client_version, subset_v1l234
from ccmodels.preprocessing.selectivity import orientation_extractor, von_mises, von_mises_single, is_selective, cell_area_identifiers, fpd_assignment, identify_multiscan


#Identify desired neurons, in this case v1 neurons from L234
client = client_version(661)
area_v1_neurons = cell_area_identifiers('V1')
v1l234_neur = subset_v1l234(client, table_name = 'coregistration_manual_v3', area_df = area_v1_neurons)
v1l234_neur = v1l234_neur[v1l234_neur['pt_root_id'] != 0]

#Identify neurons that have been scaned multiple vs one time only
multiscan_ids = identify_multiscan(v1l234_neur)
v1l234_singlescan = v1l234_neur[~v1l234_neur['pt_root_id'].isin(multiscan_ids)]
v1l234_multiscan = v1l234_neur[v1l234_neur['pt_root_id'].isin(multiscan_ids)]

session_scan_pairs = [(8,7), (4, 7), (5, 3), (5, 6),(5, 7),(6, 2),(6, 4),(6, 6),(6, 7),(7, 3),(7, 4),(7, 5),(8, 5), (9, 3), (9, 4), (9, 6)]

#Check if storage folders for temporary data exist, if not make them
if os.path.isdir('../data/in_processing/orientation_fits') != True:
    os.makedirs('../data/in_processing/orientation_fits')


print('Starting Extraction of Single Scan...')
for pair in tqdm(session_scan_pairs, desc = 'Session and Scan Loop'):
    #change the frame per directions according to the one used for each session and scan
    fpd = fpd_assignment(pair[0], pair[1])

    #Container with saved data
    data = []

    #Subset the cells for server limitation reasons
    sub = v1l234_singlescan[(v1l234_singlescan['session'] == pair[0]) & (v1l234_singlescan['scan_idx'] == pair[1])]
        
    #loop through cells
    for i in tqdm(range(sub.shape[0]), desc = f'Extracting neurons of session: {pair[0]}, scan: {pair[1]}' ):
        unit_key = {'session':sub.iloc[i, 2], 'scan_idx':sub.iloc[i, 3], 'unit_id':sub.iloc[i, 4]}
        df = orientation_extractor(unit_key, fpd)
        data.append(df)
    #Save the data
    data_df =  pd.concat(data, axis = 1)
    data_df.to_pickle(f'./data/in_processing/activities/activities_{pair[0]}_{pair[1]}.pkl')
        
    #Clean RAM
    del sub, unit_key, df
    gc.collect()    

print('Extraction single scan finished')
print('Starting extraction Multiscan...')

#Container with saved data
data = []
            
#Von Mises
columns = ['root_id', 'session', 'scan_idx','cell_id', 'pvalue','tuning_type', 
        'r_squared_diff', 'mean_r_sqrd', 'A', 'phi', 'k', 'activity', 'orientations']

for mult_id in tqdm(multiscan_ids, desc = f'Extracting multiscan neurons data') :
    subset_multiscan = v1l234_multiscan[v1l234_multiscan['pt_root_id'] == mult_id]
    scan_acts = []
    
    for row in range(subset_multiscan.shape[0]):
        unit_key = {'session':subset_multiscan['session'].values[row], 'scan_idx':subset_multiscan['scan_idx'].values[row], 'unit_id':subset_multiscan['unit_id'].values[row]}
        fpd = fpd_assignment(subset_multiscan['session'].values[row], subset_multiscan['scan_idx'].values[row])
        df = orientation_extractor(unit_key, fpd)
        scan_acts.append(df)
    
    all_acts = pd.concat(scan_acts, axis = 1)
    data.append(all_acts)

    
#Save the data
data_df =  pd.concat(data, axis = 1)
data_df.to_pickle(f'./data/in_processing/activities/activities_{pair[0]}_{pair[1]}.pkl')
#NOTE: Session, scan_idx and unit_id of these neurons refer to the values in the last row of the subset_multiscan df above
#Clean RAM
del sub, unit_key, df
gc.collect()   

print('Extraction multi scan finished, saving data')

#Joining all of the DataFrames
#Loading the first file in the directory
ors_all = pd.read_pickle(f"./data/in_processing/activities/{os.listdir('./data/in_processing/activities')[0]}")

#Loading the rest iteratively and concatenating
for file in tqdm(os.listdir('./data/in_processing/activities/')[1:], desc='Aggregating session, scan files'):
    if file!='.DS_Store':
        cont_df = pd.read_pickle(f'./data/in_processing/activities/{file}')
        ors_all = pd.concat([ors_all, cont_df], axis = 0)
    else:
        continue

#Saving the data
ors_all.to_pickle('./data/in_processing/activities.pkl')

#Delete unnecessary data repeats
#shutil.rmtree('../data/in_processing/orientation_fits')

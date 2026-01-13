#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 13:22:56 2024

@author: denizyilmaz

IMPORTANT!!!!!!

Before you run this script, make sure you disable auto-pop-up raphs othrwise you'll overload the system.

"""


# %%  0. Import Packages & Load Data

import mne
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
from mne_icalabel import label_components
from autoreject import AutoReject # for rejecting bad channels
from autoreject import get_rejection_threshold  
from collections import Counter
from pyprep.find_noisy_channels import NoisyChannels
import datetime
import json

#from pyprep import PreprocessingPipeline
def compute_heps_dp(data_dir,  
                 heart_epoch_tmin = -0.25, 
                 heart_epoch_tmax = 0.55, 
                 baseline = (-0.125, -0.025), 
                 epoch_reject_criteria=dict(eeg=150e-6),
                 bad_epoch_threshold=33,
                 time_window = (0.45, 0.50), 
                 roi = ['Fp2','F4', 'F8'],
                 double_rpeak_exclusion = True
                 ):

    """
    This function computes Heart Evoked Potentials (HEPs) for each participant in the given directory.
    The function reads in preprocessed and ICA cleaned data files, epochs the data around R-peaks,
    and computes the average HEP for each participant. The function saves the HEPs as evoked objects,
    plots the HEPs, and saves the plots and evoked objects in the specified directory.

    Parameters
    ----------
    data_dir : str
        The directory where the preprocessed and ICA cleaned data files are stored.
    epoch_reject_criteria : dict, optional
        The rejection criteria for epochs. The default is dict(eeg=150e-6).
    bad_epoch_threshold : int, optional
        The threshold percentage of bad epochs to interpolate channels. The default is 33.
    heart_epoch_tmin : float, optional
        The start time of the HEP epoch. The default is -0.25.
    heart_epoch_tmax : float, optional
        The end time of the HEP epoch. The default is 0.55.
    baseline : tuple, optional
        The baseline for the HEP epoch. The default is (-0.125, -0.025).
    time_window : tuple, optional
        The time window to highlight on the HEP plot. The default is (0.45, 0.50).
    roi : list, optional
        The region of interest for the HEP analysis. The default is ['F4', 'F8', 'Fp2'].

    Returns
    -------
    None.
    Only saves the HEPs, plots, and outputs in the specified directory.

    """

    #  Dir where data preprocessed and ICA cleaned is stored
    os.chdir(data_dir)

    ##### define default vars for the function
    # define epoch parameters    
    epoch_time_window = (heart_epoch_tmin, heart_epoch_tmax)
     # baseline = (-0.25, -0.20) # same as: KOREKI: -250 - -200
    highlight = time_window #time window to highlight on plot  
    # in seconds   # we are interested in a time window from 450 to 500 ms
    frontal_central_regions= ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8','Fz', 'FC1', 'FC2', 'FC5', 'FC6', 'C3', 'C4', 'Cz']

    ##### define necess vars
    tasks = ['eyes-closed', 'eyes-open', 'hct']
    sz_id = "BTSCZ"
    hc_id = "BHC"

    # if a filename in data_dir includes BHC then is_hc is True, if it includes BTSCZ then is_sz is True, if none then raise error saying no participants found in dir
    is_hc = False
    is_sz = False
    for filename in os.listdir():
        if hc_id in filename:
            is_hc = True      
            sessions  = None  
            print("processing HC data")
        elif sz_id in filename:
            is_sz = True
            sessions= ['V1']  # sessions= ['V1', 'V3']
            print("processing SZ data")
    if not is_hc and not is_sz:
        raise ValueError("No participants found in the directory.")
  

    # %%  1. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

    # Define the column names for the DF (act.ly not necess but I like to have it in the end I added some more columns)
    column_names = ['subject_id', 'session', 'task', 'event_type',  'event_times', 'total_heart_events_nr',
                    'total_epochs_nr', 'good_epoch_count', 'percentage_dropped_epochs_double_peaks_not_cleaned','percentage_dropped_epochs_double_peaks_cleaned', 
                    'epoch_time_window', 'baseline_correction', 
                    'hep_time_window',  'channels', 'hep_max_amplitudes', 'hep_max_latencies', 'hep_min_amplitudes', 'hep_min_latencies',
                    'hep_mean_amplitudes', 'hep_amplitudes_sd', 'Fp2_mean_amplitude', 'F4_mean_amplitude', 'F8_mean_amplitude',
                    'Fp2_max_amplitude', 'F4_max_amplitude', 'F8_max_amplitude', 'Fp2_min_amplitude', 'F8_min_amplitude', 
                    'Fp2_max_latency', 'F4_max_latency', 'F8_max_latency', 'Fp2_min_latency', 'F4_min_latency', 'F8_min_latency',
                    'mean_HEP_accross_channels',
                    'start_time_of_analysis', 'analysis_duration'
                    ]


    # Initialize an empty DF with columns
    hep_outputs = pd.DataFrame(columns=column_names)

    # Lists to store evoked objects, where each sub will be one item in the list (to be able to save them all in a single file)
    hep_list_eyes_closed_V1 = []
    evoked_metadata_eyes_closed_V1 = [] # Store metadata for each evoked object
    hep_list_eyes_open_V1 = []
    evoked_metadata_eyes_open_V1 = []
    hep_list_hct_V1 = []
    evoked_metadata_hct_V1 = []
    hep_list_eyes_closed_V3 = []
    evoked_metadata_eyes_closed_V3 = []
    hep_list_eyes_open_V3 = []
    evoked_metadata_eyes_open_V3 = []
    hep_list_hct_V3 = []
    evoked_metadata_hct_V3 = []
    hep_list_eyes_closed_no_session = []
    evoked_metadata_eyes_closed_no_session = []
    hep_list_eyes_open_no_session = []
    evoked_metadata_eyes_open_no_session = []
    hep_list_hct_no_session = []
    evoked_metadata_hct_no_session = []


    # Initialize list to store excluded participants and the reason for exclusion, do NOT proceed with further analysis if > {bad_epoch_threshold} % of epochs are bad
    excluded_files = []

    # %% 2. loopidiebooo

    # Get a list of filenames from the current directory
    all_files = os.listdir()

    # Filter filenames that end with ".eeg"
    eeg_file_names = [filename for filename in all_files if filename.endswith('.fif')]

    # Initialize Participant, Session, Task
    participant_numbers = []
    if is_sz:
        for file in eeg_file_names:
            num = file[5:8]
            print(num)
            participant_numbers.append(num)
    elif is_hc:
        for file in eeg_file_names:
            num = file[3:6]
            print(num)
            participant_numbers.append(num)
        
    participant_numbers =  list(set(participant_numbers))
    participant_numbers = np.array(sorted(participant_numbers))


    for participant_no in participant_numbers:
        for session in (sessions if sessions else [None]): 
            for task in tasks: 

                if is_sz:
                    filename = f"BTSCZ{participant_no}_{session}_{task}_prep_ICA.fif"
                elif is_hc:
                    filename = f"BHC{participant_no}_{task}_prep_ICA.fif"

                
                try:
                    prep_ica_data = mne.io.read_raw(filename, preload=True)
                    print(f"Successfully loaded: {filename}")
                except FileNotFoundError:
                    print(f"File not found: {filename}. Skipping...")
                    continue
                            
                # track time
                start_time = datetime.datetime.now()
                
                # Filter out irrelevant annotations 
                irrelevant_annotations = ['Comment/actiCAP Data On', 'New Segment/', 'Comment/actiCAP USB Power On', 'ControlBox is not connected via USB', 'actiCAP USB Power On', 'Comment/ControlBox is not connected via USB', 'BAD boundary', 'EDGE boundary']  # Adjust this list as needed
                
                # find indices of irrelevant annots
                irrelevant_indices = np.where(np.isin(prep_ica_data.annotations.description, irrelevant_annotations))[0]
                
                # delete irrelevant ones
                prep_ica_data.annotations.delete(irrelevant_indices)
                
                # check whether it worked
                # prep_ica_data.annotations.description
                print("Remaining annotations:", np.unique(prep_ica_data.annotations.description, return_counts=True))
                
                ### Continue with epoching & creating evoked
                
                if task == 'hct':

                    # Get event ids of heartbeat_events
                    events, event_id = mne.events_from_annotations(prep_ica_data)
                    R_peak_id = event_id['R-peak']
                    heartbeat_events = events[events[:, 2] == R_peak_id]
                    
                else: 
                    
                    # This below works ONLY for non-HCT tasks (RESTING-STATE) because in HCT we also have the other annots, which wont be dropped, just by dropping the first 2 irrelevants!
                    heartbeat_events, R_peak_id = mne.events_from_annotations(prep_ica_data)
                
                total_heart_events_nr = len(heartbeat_events)

                # create epochs
                heartbeat_epochs = mne.Epochs(
                    prep_ica_data, heartbeat_events,
                    event_id=R_peak_id, tmin=heart_epoch_tmin, tmax=heart_epoch_tmax,  # here I select the timing same as Koreki et al., 23
                    baseline=baseline, 
                    preload=True, event_repeated='drop' # or event_repeated='error'?
                )
                
                # do I need to apply_baseline again ?! no because it is already done above as you create epochs
                total_epochs_nr = len(heartbeat_epochs)

                # drop bad epochs
                heartbeat_epochs.drop_bad(reject=epoch_reject_criteria)
                #percentage_dropped_epochs = heartbeat_epochs.drop_log_stats()
                n_dropped_bad_epochs = sum(1 if len(x) > 0 else 0 for x in heartbeat_epochs.drop_log)
                #percentage_dropped_epochs = (n_dropped_bad_epochs / total_epochs_nr) * 100
                percentage_dropped_epochs = ((total_epochs_nr - len(heartbeat_epochs)) / total_epochs_nr) * 100



                
                # plot epochs
                # heartbeat_epochs.plot(events = heartbeat_events_v1, event_id=event_id_v1)

                # %% ### !!: If more than {bad_epoch_threshold} % of the epochs are bad, check whether that is due to a single channel, if so interpolate that channel


                # If > bad_epoch_threshold % of epochs are dropped, analyze the responsible channels
                if percentage_dropped_epochs > bad_epoch_threshold:
                    print(f"\nMore than {bad_epoch_threshold} % of epochs are dropped. Analyzing bad channels...")
                    
                    # Analyze which channels are causing the dropped epochs
                    drop_log = heartbeat_epochs.drop_log
                    bad_channels = [entry for log in drop_log for entry in log if entry != "IGNORED"]

                    # Count occurrences of each bad channel
                    bad_channel_counts = {ch: bad_channels.count(ch) for ch in set(bad_channels)}

                    # Find channels present in > {bad_epoch_threshold} % of all epochs
                    chans_to_interpolate_epochs = [
                        ch for ch, count in bad_channel_counts.items() if (count / total_epochs_nr) * 100 > bad_epoch_threshold
                    ]

                    if len(chans_to_interpolate_epochs) == 1:
                        
                        channel_interpolated_due_bad_epochs = chans_to_interpolate_epochs

                        print(f"\nChannels to interpolate due to causing many bad epochs: {chans_to_interpolate_epochs}")
                        # Interpolate bad channels before dropping epochs
                        # create epochs
                        heartbeat_epochs = mne.Epochs(
                        prep_ica_data, heartbeat_events,
                        event_id=R_peak_id, tmin=heart_epoch_tmin, tmax=heart_epoch_tmax,  # here I select the timing same as Koreki et al., 23
                        baseline=baseline, 
                        preload=True, event_repeated='drop')

                        # interpolate channel that causes more than {bad_epoch_threshold} % of epochs to be bad
                        heartbeat_epochs.info['bads'] = chans_to_interpolate_epochs
                        heartbeat_epochs.interpolate_bads()

                        # now drop bad epochs
                        heartbeat_epochs.drop_bad(reject=epoch_reject_criteria)
                        n_dropped_bad_epochs = sum(1 if len(x) > 0 else 0 for x in heartbeat_epochs.drop_log)
                        #percentage_dropped_epochs = heartbeat_epochs.drop_log_stats()
                        #percentage_dropped_epochs = (n_dropped_bad_epochs / total_epochs_nr) * 100
                        percentage_dropped_epochs = ((total_epochs_nr - len(heartbeat_epochs)) / total_epochs_nr) * 100




                    else:
                        # indicate that no single channel was responsible for > {bad_epoch_threshold} % of dropped epochs
                        channel_interpolated_due_bad_epochs = ''
                        print(f"\nNo single channel is responsible for > {bad_epoch_threshold}  % of dropped epochs. Proceeding without interpolation.")
                else:
                    channel_interpolated_due_bad_epochs = ''
                    print(f"\nLess than {bad_epoch_threshold}  % of epochs are dropped. No interpolation needed.")

                # Add exclusion criterion: exclude files if > 33% epochs are bad after interpolation
                if percentage_dropped_epochs > bad_epoch_threshold: # this may not work if the last file is excluded?? bc it wont be in the csv?
                    print(f"\nMore than {bad_epoch_threshold}% of epochs are bad ({percentage_dropped_epochs:.2f}%). Excluding this file from further analysis.")
                    excluded_files.append({
                            'subject_id': participant_id,
                            'session': session,
                            'task': task,
                            'reason': f"More than {bad_epoch_threshold}% epochs dropped"
                        })
                    continue # Skip the rest of the loop and move to the next file






                # Add exclusion criterion: if there are 2 peaks in a remaining good epoch (denoted by multiple triggers with an R-peak label),  exclude that epoch
                # -----------------------------
                # Step: Exclude epochs with double R-peaks
                # -----------------------------

                if double_rpeak_exclusion:

                    print("\nChecking for epochs with multiple R-peaks...")
                    # List to store indices of epochs with >1 R-peak
                    double_rpeak_epochs = []

                    sfreq = heartbeat_epochs.info['sfreq']
                    heart_epoch_tmin

                    # Epoch starts and ends in samples (adjust start by tmin)
                    epoch_starts = heartbeat_epochs.events[:, 0] + int(heart_epoch_tmin * sfreq)
                    epoch_ends   = epoch_starts + len(heartbeat_epochs.times)

                    # All R-peak event samples
                    rpeak_samples = heartbeat_events[:, 0]  # 1D array of sample indices

                    # Boolean mask: True if epoch has multiple R-peaks
                    extra_rpeak_mask = np.array([
                        np.sum((rpeak_samples >= start) & (rpeak_samples < end)) > 1  # >1 means multiple R-peaks
                        for start, end in zip(epoch_starts, epoch_ends)
                    ])

                    # Keep only epochs with 1 or 0 R-peaks
                    clean_mask = ~extra_rpeak_mask
                    heartbeat_epochs_clean = heartbeat_epochs[clean_mask]

                    # how many double peak epochs were found?
                    num_double_rpeak_epochs = np.sum(extra_rpeak_mask)

                    # updated percentage dropped epochs
                    percentage_dropped_epochs_including_double_peaks = ((total_epochs_nr - len(heartbeat_epochs_clean)) / total_epochs_nr) * 100


                    print(f"Kept {len(heartbeat_epochs_clean)} / {len(heartbeat_epochs)} epochs")

                else:
                    percentage_dropped_epochs_including_double_peaks = percentage_dropped_epochs
                    num_double_rpeak_epochs = 0
                    heartbeat_epochs_clean = heartbeat_epochs
                    








                        
                
                # %% ### create evoked
                
                heartbeat_evoked = heartbeat_epochs_clean.average()

                # Append evoked data to appropriate list
                if task == 'eyes-closed' and session == 'V1':
                    hep_list_eyes_closed_V1.append(heartbeat_evoked)
                    evoked_metadata_eyes_closed_V1.append(filename)
                    # add debug print
                    print(f"HEP for {filename} appended to metadata list for eyes-closed V1")
                elif task == 'eyes-open' and session == 'V1':
                    hep_list_eyes_open_V1.append(heartbeat_evoked)
                    evoked_metadata_eyes_open_V1.append(filename)
                    #debug
                    print(f"HEP for {filename} appended to metadata list for eyes-open V1")
                elif task == 'hct' and session == 'V1':
                    hep_list_hct_V1.append(heartbeat_evoked)
                    evoked_metadata_hct_V1.append(filename)
                    #debug
                    print(f"HEP for {filename} appended to metadata list for hct V1")
                elif task == 'eyes-closed' and session == 'V3':
                    hep_list_eyes_closed_V3.append(heartbeat_evoked)
                    evoked_metadata_eyes_closed_V3.append(filename)
                elif task == 'eyes-open' and session == 'V3':
                    hep_list_eyes_open_V3.append(heartbeat_evoked)
                    evoked_metadata_eyes_open_V3.append(filename)
                elif task == 'hct' and session == 'V3':
                    hep_list_hct_V3.append(heartbeat_evoked)
                    evoked_metadata_hct_V3.append(filename)
                elif task == 'eyes-closed' and session is None:
                    hep_list_eyes_closed_no_session.append(heartbeat_evoked)
                    evoked_metadata_eyes_closed_no_session.append(filename)
                elif task == 'eyes-open' and session is None:
                    hep_list_eyes_open_no_session.append(heartbeat_evoked)
                    evoked_metadata_eyes_open_no_session.append(filename)
                elif task == 'hct' and session is None:
                    hep_list_hct_no_session.append(heartbeat_evoked)
                    evoked_metadata_hct_no_session.append(filename)
                


                ##############     Plot evoked ###############
                              
                hep_joint_plot = heartbeat_evoked.plot_joint(show=False)
                hep_plot = mne.viz.plot_evoked(heartbeat_evoked, show=False, highlight=highlight)
                hep_plot_3chans = mne.viz.plot_evoked(heartbeat_evoked, highlight=highlight, picks = roi, show=False)
                hep_psd_plot = heartbeat_evoked.plot_psd(show=False)
                
                # Compute the average HEP across all fronto-central channels
                picks = mne.pick_channels(heartbeat_evoked.ch_names, frontal_central_regions)
                average_frontal_central = heartbeat_evoked.data[picks, :].mean(axis=0) # axis=0 to average across channels
                # Create an Info object for the averaged data
                info_avg = mne.create_info(['Frontal-Central Average'], heartbeat_evoked.info['sfreq'], ch_types='eeg')
                # Create an Evoked object for the averaged data
                average_frontal_central = mne.EvokedArray(average_frontal_central[np.newaxis, :], info_avg, tmin=heartbeat_evoked.times[0])
                average_frontal_central_plot = mne.viz.plot_evoked(average_frontal_central,  highlight=highlight, show=False)

                    
                # heartbeat_evoked.plot(gfp=True)
                # heartbeat_evoked_v1.plot_joint().show()
                # heartbeat_evoked_v1.plot()
                # heartbeat_evoked_v1.plot_psd()
                # heartbeat_evoked_v1.plot(picks=['F4'])
                # heartbeat_evoked_v1.plot(picks=['F8'])
                # heartbeat_evoked_v1.plot(picks=['Fp2'])
                
                ########       Extract relevant variables and values.... ##############
                
                # Select the time indices for this range
                mask = (heartbeat_evoked.times >= time_window[0]) & (heartbeat_evoked.times <= time_window[1])
                
                # Extract data for all channels within this time window, then average across the time window also get peak values
                hep_mean_amplitudes = heartbeat_evoked.data[:, mask].mean(axis=1) # axis=1 to average across time
                hep_amplitudes_sd = heartbeat_evoked.data[:, mask].std(axis=1)
                
                # Mean HEP accross channels
                mean_HEP_accross_channels = hep_mean_amplitudes.mean()
                
                ## get peak values
                evoked_times = heartbeat_evoked.times # define all timepoints of the evoked object
                
                # max
                max_amplitudes = heartbeat_evoked.data[:, mask].max(axis=1)
                max_indices = heartbeat_evoked.data[:, mask].argmax(axis=1)
                max_latencies = evoked_times[mask][max_indices]
                
                # min
                min_amplitudes = heartbeat_evoked.data[:, mask].min(axis=1)
                min_indices = heartbeat_evoked.data[:, mask].argmin(axis=1)
                min_latencies = heartbeat_evoked.times[mask][min_indices]
                
                # You need to add channel names too, to be able to match mean amps to chans...
                channels = heartbeat_evoked.ch_names
                
                # Focus on specific channels, e.g., 'Fz', 'Cz', 'Pz'
                # Find indices of the channels
                channel_indices = [heartbeat_evoked.ch_names.index(ch) for ch in roi]
                selected_mean_amplitude_data = hep_mean_amplitudes[channel_indices]
                selected_max_amplitude_data = max_amplitudes[channel_indices]
                selected_min_amplitude_data = min_amplitudes[channel_indices]
                selected_max_latency_data = max_latencies[channel_indices]
                selected_min_latency_data = min_latencies[channel_indices]


                Fp2_mean_amplitude = selected_mean_amplitude_data[0]
                F4_mean_amplitude = selected_mean_amplitude_data[1]
                F8_mean_amplitude = selected_mean_amplitude_data[2]
                
                Fp2_max_amplitude = selected_max_amplitude_data[0]
                F4_max_amplitude = selected_max_amplitude_data[1]
                F8_max_amplitude = selected_max_amplitude_data[2]
                
                Fp2_min_amplitude = selected_min_amplitude_data[0]
                F4_min_amplitude = selected_min_amplitude_data[1]
                F8_min_amplitude = selected_min_amplitude_data[2]
                
                Fp2_max_latency = selected_max_latency_data[0]
                F4_max_latency = selected_max_latency_data[1]
                F8_max_latency = selected_max_latency_data[2]
                
                Fp2_min_latency = selected_min_latency_data[0]
                F4_min_latency = selected_min_latency_data[1]
                F8_min_latency = selected_min_latency_data[2]


                ### Add also the mean HEP for each channel as a column
                channel_mean_hep_values = {}
                for channel in heartbeat_evoked.ch_names:
                    channel_idx = heartbeat_evoked.ch_names.index(channel)
                    channel_mean_hep_values[channel] = hep_mean_amplitudes[channel_idx]

                #########  Save the data to a CSV file #########
                
                ### Prepare the CSVto Save 
                
                # participant id should be BTSCZ...
                participant_id = f"BTSCZ{participant_no}" if "BTSCZ" in filename else f"BHC{participant_no}"
                
                # record end_time 
                end_time = datetime.datetime.now()
                
                # Calculate the duration of the analysis
                duration = end_time - start_time
                
                # Create a dictionary representing the new row
                new_row = pd.Series({'subject_id': participant_id, 
                                    'session': session, 
                                    'task': task,
                                    'event_type': 'R-peaks for HEP',
                                    'event_times': heartbeat_events[:,0],
                                    'total_heart_events_nr': total_heart_events_nr,
                                    'total_epochs_nr': total_epochs_nr,
                                    'good_epoch_count': len(heartbeat_epochs_clean), 
                                    'percentage_dropped_epochs_double_peaks_not_cleaned': percentage_dropped_epochs, 
                                    'percentage_dropped_epochs_double_peaks_cleaned': percentage_dropped_epochs_including_double_peaks,
                                    'threshold_percentage_bad_epochs': bad_epoch_threshold ,
                                    'channel_interpolated_due_too-many-bad-epochs': channel_interpolated_due_bad_epochs,
                                    'epoch_strategy': f'if > {bad_epoch_threshold}% bad epochs due to a single channel, interpolate that channel. If due to multiple channels, exclude the file',
                                    'number_of_double_R-peak_epochs_excluded': num_double_rpeak_epochs,
                                    'epoch_time_window': epoch_time_window,
                                    'baseline_correction': baseline,
                                    'hep_time_window': time_window,
                                    'channels': channels,
                                    'hep_max_amplitudes': max_amplitudes,
                                    'hep_max_latencies': max_latencies,
                                    'hep_min_amplitudes': min_amplitudes,
                                    'hep_min_latencies': min_latencies,
                                    'hep_mean_amplitudes': hep_mean_amplitudes,
                                    'hep_amplitudes_sd': hep_amplitudes_sd,
                                    'Fp2_mean_amplitude': Fp2_mean_amplitude, 
                                    'F4_mean_amplitude': F4_mean_amplitude, 
                                    'F8_mean_amplitude': F8_mean_amplitude,
                                    'Fp2_max_amplitude': Fp2_max_amplitude, 
                                    'F4_max_amplitude': F4_max_amplitude, 
                                    'F8_max_amplitude': F8_max_amplitude, 
                                    'Fp2_min_amplitude': Fp2_min_amplitude, 
                                    'F4_min_amplitude': F4_min_amplitude,
                                    'F8_min_amplitude': F8_min_amplitude, 
                                    'Fp2_max_latency': Fp2_max_latency, 
                                    'F4_max_latency': F4_max_latency, 
                                    'F8_max_latency': F8_max_latency, 
                                    'Fp2_min_latency': Fp2_min_latency, 
                                    'F4_min_latency': F4_min_latency, 
                                    'F8_min_latency': F8_min_latency,
                                    'mean_HEP_accross_channels': mean_HEP_accross_channels,
                                    'start_time_of_analysis': start_time,
                                    'analysis_duration': duration
                                    })
                
                # Add each channel's mean HEP value to the new_row dictionary
                for channel, mean_hep_value in channel_mean_hep_values.items():
                    new_row[f'{channel}_mean_HEP'] = mean_hep_value

                # convert row to df
                new_row =  new_row.to_frame().T
                
                # add to existing df the current data outputs
                hep_outputs = pd.concat([hep_outputs, new_row], ignore_index=True)
                
                # Print the DataFrame
                print(hep_outputs)
                            
                ### Save the Data 
                
                #### Save individual's file!   

                # create dynamic folder name for processing style   
                processing_style_epoch = f"hep-epoch-{heart_epoch_tmin:.2f}-{heart_epoch_tmax:.2f}"    
                processing_style_baseline = f"baseline-{baseline[0]:.2f}-{baseline[1]:.2f}"   
                eeg_reject_value = epoch_reject_criteria.get('eeg', 'NA')  # Default to 'NA' if not found
                processing_style_reject = f"epoch-reject-{eeg_reject_value:.2e}"
                processing_style_bad_epoch = f"{bad_epoch_threshold}%-bad-epochs-threshold"
                processing_style_time_window = f"HEP-time_window-{time_window[0]:.2f}-{time_window[1]:.2f}"
                processing_subfolder_name = f"HEPs_{processing_style_epoch}_{processing_style_baseline}_{processing_style_reject}_{processing_style_bad_epoch}_{processing_style_time_window}"
                double_rpeak_exclusion_str = "with-double-R-peak-exclusion" if double_rpeak_exclusion else ""

                # SAVE the Preprocessed Data & Prep Ouputs in a CSV
                folder_name = 'HEPs' # this is the main folder where all the outputs will be saved"
                output_folder_path = os.path.join(os.getcwd(), folder_name, processing_subfolder_name, double_rpeak_exclusion_str)
                print(f"Creating directory: {output_folder_path}")
                os.makedirs(output_folder_path, exist_ok=True)

                # Save the individual's file

                # Define the particpant folder path
                participant_folder = os.path.join(output_folder_path, participant_id)
                os.makedirs(participant_folder, exist_ok=True) # Create the participant folder if it doesn't exist
                # Session-specific folder within the participant folder
                participant_session_folder = os.path.join(participant_folder, f"ses-{session}") if "BTSCZ" in filename else participant_folder
                os.makedirs(participant_session_folder, exist_ok=True)  # Create session folder if it doesn't exist
                # Task-specific folder within the participant folder
                participant_session_task_folder = os.path.join(participant_session_folder, f"task-{task}") 
                os.makedirs(participant_session_task_folder, exist_ok=True)  # Create session folder if it doesn't exist


                # Define the file name and path
                hep_file_name = filename.replace('_prep_ICA', '_hep-ave')
                individual_file_path = os.path.join(participant_session_task_folder, hep_file_name)
                #individual_file_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/HEPs', hep_file_name)
                mne.write_evokeds(individual_file_path, heartbeat_evoked, on_mismatch='raise', overwrite=True,)   # evokedEvoked instance, or list of Evoked instance; to load it back: evokeds_list = mne.read_evokeds(evk_file, verbose=False)


                # Save the individual's plots

                # directory for plots
                plot_dir = os.path.join(output_folder_path, 'plots', participant_id, f"ses-{session}" if "BTSCZ" in filename else "", f"task-{task}")
                os.makedirs(plot_dir, exist_ok=True)
                #joint plot
                hep_joint_plot_path = os.path.join(plot_dir, f'{participant_id}_{f'ses-{session}_' if 'BTSCZ' in filename else ''}{task}_joint_plot.jpg' )
                hep_joint_plot.savefig(hep_joint_plot_path, format='jpg')
                # hep plot
                hep_plot_path = os.path.join(plot_dir, f'{participant_id}_{f'ses-{session}_' if 'BTSCZ' in filename else ''}{task}_hep_plot.jpg' )
                hep_plot.savefig(hep_plot_path, format='jpg')
                # plot 3 chans
                hep_plot_3chans_path = os.path.join(plot_dir, f'{participant_id}_{f'ses-{session}_' if 'BTSCZ' in filename else ''}{task}_hep_plot_3chans.jpg' )
                hep_plot_3chans.savefig(hep_plot_3chans_path, format='jpg')
                # psd plot
                hep_psd_plot_path = os.path.join(plot_dir, f'{participant_id}_{f'ses-{session}_' if 'BTSCZ' in filename else ''}{task}_hep_psd_plot.jpg' )
                hep_psd_plot.savefig(hep_psd_plot_path, format='jpg')
                # avg. frontal central plot
                average_frontal_central_plot_path = os.path.join(plot_dir, f'{participant_id}_{f'ses-{session}_' if 'BTSCZ' in filename else ''}{task}_hep_average_frontal_central_plot.jpg' )
                average_frontal_central_plot.savefig(average_frontal_central_plot_path, format='jpg')

                
                ### Save all subs Evokeds in a single file as a list
                if task == 'eyes-closed' and session == 'V1':
                    mne.write_evokeds(os.path.join(output_folder_path, 'heps_list_V1_eyes_closed-ave.fif'), hep_list_eyes_closed_V1, overwrite=True) 
                    evoked_metadata_eyes_closed_V1_path = os.path.join(output_folder_path, 'evoked_metadata_eyes_closed_V1.csv')
                    evoked_metadata_eyes_closed_V1_df = pd.DataFrame(evoked_metadata_eyes_closed_V1, columns=["filename"])
                    evoked_metadata_eyes_closed_V1_df.to_csv(evoked_metadata_eyes_closed_V1_path, index=False)
                elif task == 'eyes-open' and session == 'V1':
                    mne.write_evokeds(os.path.join(output_folder_path, 'heps_list_V1_eyes_open-ave.fif'), hep_list_eyes_open_V1, overwrite=True)
                    evoked_metadata_eyes_open_V1_path = os.path.join(output_folder_path, 'evoked_metadata_eyes_open_V1.csv')
                    evoked_metadata_eyes_open_V1_df = pd.DataFrame(evoked_metadata_eyes_open_V1, columns=["filename"])
                    evoked_metadata_eyes_open_V1_df.to_csv(evoked_metadata_eyes_open_V1_path, index=False)
                elif task == 'hct' and session == 'V1':
                    mne.write_evokeds(os.path.join(output_folder_path, 'heps_list_V1_hct-ave.fif'), hep_list_hct_V1, overwrite=True)
                    evoked_metadata_hct_V1_path = os.path.join(output_folder_path, 'evoked_metadata_hct_V1.csv')
                    evoked_metadata_hct_V1_df = pd.DataFrame(evoked_metadata_hct_V1, columns=["filename"])
                    evoked_metadata_hct_V1_df.to_csv(evoked_metadata_hct_V1_path, index=False)
                elif task == 'eyes-closed' and session == 'V3':
                    mne.write_evokeds(os.path.join(output_folder_path, 'heps_list_V3_eyes_closed-ave.fif'), hep_list_eyes_closed_V3, overwrite=True)
                    evoked_metadata_eyes_closed_V3_path = os.path.join(output_folder_path, 'evoked_metadata_eyes_closed_V3.csv')
                    evoked_metadata_eyes_closed_V3_df = pd.DataFrame(evoked_metadata_eyes_closed_V3, columns=["filename"])
                    evoked_metadata_eyes_closed_V3_df.to_csv(evoked_metadata_eyes_closed_V3_path, index=False)
                elif task == 'eyes-open' and session == 'V3':
                    mne.write_evokeds(os.path.join(output_folder_path, 'heps_list_V3_eyes_open-ave.fif'), hep_list_eyes_open_V3, overwrite=True)
                    evoked_metadata_eyes_open_V3_path = os.path.join(output_folder_path, 'evoked_metadata_eyes_open_V3.csv')
                    evoked_metadata_eyes_open_V3_df = pd.DataFrame(evoked_metadata_eyes_open_V3, columns=["filename"])
                    evoked_metadata_eyes_open_V3_df.to_csv(evoked_metadata_eyes_open_V3_path, index=False)
                elif task == 'hct' and session == 'V3':
                    mne.write_evokeds(os.path.join(output_folder_path, 'heps_list_V3_hct-ave.fif'), hep_list_hct_V3, overwrite=True)
                    evoked_metadata_hct_V3_path = os.path.join(output_folder_path, 'evoked_metadata_hct_V3.csv')
                    evoked_metadata_hct_V3_df = pd.DataFrame(evoked_metadata_hct_V3, columns=["filename"])
                    evoked_metadata_hct_V3_df.to_csv(evoked_metadata_hct_V3_path, index=False)
                if task == 'eyes-closed' and session == None:
                    mne.write_evokeds(os.path.join(output_folder_path, 'HC_heps_list_V1_eyes_closed-ave.fif'), hep_list_eyes_closed_no_session, overwrite=True)  
                    evoked_metadata_eyes_closed_no_session_path = os.path.join(output_folder_path, 'evoked_metadata_eyes_closed_no_session.csv')
                    evoked_metadata_eyes_closed_no_session_df = pd.DataFrame(evoked_metadata_eyes_closed_no_session, columns=["filename"])
                    evoked_metadata_eyes_closed_no_session_df.to_csv(evoked_metadata_eyes_closed_no_session_path, index=False)
                elif task == 'eyes-open' and session == None:
                    mne.write_evokeds(os.path.join(output_folder_path, 'HC_heps_list_V1_eyes_open-ave.fif'), hep_list_eyes_open_no_session, overwrite=True)
                    evoked_metadata_eyes_open_no_session_path = os.path.join(output_folder_path, 'evoked_metadata_eyes_open_no_session.csv')
                    evoked_metadata_eyes_open_no_session_df = pd.DataFrame(evoked_metadata_eyes_open_no_session, columns=["filename"])
                    evoked_metadata_eyes_open_no_session_df.to_csv(evoked_metadata_eyes_open_no_session_path, index=False)
                elif task == 'hct' and session == None:
                    mne.write_evokeds(os.path.join(output_folder_path, 'HC_heps_list_V1_hct-ave.fif'), hep_list_hct_no_session, overwrite=True)
                    evoked_metadata_hct_no_session_path = os.path.join(output_folder_path, 'evoked_metadata_hct_no_session.csv')
                    evoked_metadata_hct_no_session_df = pd.DataFrame(evoked_metadata_hct_no_session, columns=["filename"])
                    evoked_metadata_hct_no_session_df.to_csv(evoked_metadata_hct_no_session_path, index=False)        

                ### CSV : can be in or out of the loop, in the loop it keeps adding and overwriting the prvs csv
                
                # Construct the csv path and filename including the date of the current run
                current_date = datetime.datetime.now().strftime("%Y%m%d")
                csv_filename = f'hep_outputs_{current_date}.csv' if "BTSCZ" in filename else f'HC_hep_outputs_{current_date}.csv'
                csv_path = os.path.join(output_folder_path, csv_filename)
                hep_outputs.to_csv(csv_path, mode='w', sep=',', index=False)
                
                # Save excluded participants to CSV at the end
                excluded_csv_filename = f'excluded_files_{current_date}.csv'
                excluded_csv_path = os.path.join(output_folder_path, excluded_csv_filename)
                excluded_files_df = pd.DataFrame(excluded_files, columns=["subject_id", "session", "task", "reason"])
                excluded_files_df.to_csv(excluded_csv_path, index=False)
                print(f"Excluded files logged to {excluded_csv_path}")
    
    print("HEP computation completed.")
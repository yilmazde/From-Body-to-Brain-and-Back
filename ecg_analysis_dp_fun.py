
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 09:43:47 2024

@author: denizyilmaz
"""

# %%  0. Import Packages 
import datetime
import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
import neurokit2 as nk
import matplotlib
from collections import Counter
import pandas as pd

def ecg_analysis_fun(data_dir,
                     new_sfreq = 250,
                     baseline = (-0.125, -0.025),
                     ecg_epochs_tmin = -0.25,
                     ecg_epochs_tmax = 0.55,
                     time_window = (0.45, 0.50),
                     double_rpeak_exclusion = True
):

    """
    Function to extract ECG parameters from EEG data and save the outputs in a CSV file.

    Parameters
    ----------  
    data_dir : str
        Directory where the raw EEG data is stored.
    
    Returns
    -------
    Outputs in a CSV file.
    Evoked ECG data saved in a fif file.
    
    """

    #  Dir where raw EEG data is stored
    os.chdir(data_dir)

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
        elif sz_id in filename:
            is_sz = True
            sessions= ['V1', 'V3']
    if not is_hc and not is_sz:
        raise ValueError("No participants found in the directory.")

    # %%  1. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

    # Define the column names for the DF
    column_names = ['subject_id', 'session', 'task', 
                                    'heart_rate_bpm',
                                    'hrv_rmssd_ms',
                                    'R_peak_amplitude_mV', 
                                    'QT_interval_ms',
                                    'QTc_interval_ms', 
                                    'baseline',
                                    'ecg_epochs_tmin_tmax',
                                    'time_window',
                                    'percentage_dropped_epochs_double_peaks_cleaned',
                    ]


    # Initialize an empty DF with columns
    ecg_outputs = pd.DataFrame(columns=column_names)

    # Lists to store evoked objects, where each sub will be one item in the list (to be able to save them all in a single file)
    ecg_list_eyes_closed_V1 = []
    ecg_evoked_metadata_eyes_closed_V1 = [] # Store metadata for each evoked object
    ecg_list_eyes_open_V1 = []
    ecg_evoked_metadata_eyes_open_V1 = []
    ecg_list_hct_V1 = []
    ecg_evoked_metadata_hct_V1 = []
    ecg_list_eyes_closed_V3 = []
    ecg_evoked_metadata_eyes_closed_V3 = []
    ecg_list_eyes_open_V3 = []
    ecg_evoked_metadata_eyes_open_V3 = []
    ecg_list_hct_V3 = []
    ecg_evoked_metadata_hct_V3 = []
    ecg_list_eyes_closed_no_session = []
    ecg_evoked_metadata_eyes_closed_no_session = []
    ecg_list_eyes_open_no_session = []
    ecg_evoked_metadata_eyes_open_no_session = []
    ecg_list_hct_no_session = []
    ecg_evoked_metadata_hct_no_session = []



    # %% 2. loopidiebooo

    # Get a list of filenames from the current directory
    all_files = os.listdir()

    # Filter filenames that end with ".eeg"
    eeg_file_names = [filename for filename in all_files if filename.endswith('.eeg')]

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
                    filename = f"BTSCZ{participant_no}_{session}_{task}.vhdr"
                elif is_hc:
                    filename = f"BHC{participant_no}_{task}.vhdr"

                
                try:
                    raw = mne.io.read_raw(filename, preload=True)
                    print(f"Successfully loaded: {filename}")
                except FileNotFoundError:
                    print(f"File not found: {filename}. Skipping...")
                    continue
                
                ### OR you can directly downsample EEG then extract ECG
                
                # Resample the EEG data to the new sampling rate
                # raw_resampled = raw.copy()
                # raw_resampled.resample(sfreq=new_sfreq)
                # print(raw_resampled.info["sfreq"])
                
                
                #%% Extract ecg parameters: HR, HRV (RMSSD),  Amplitude of R wave,  QT interval, QTc interval,
                
                # Extract ECG signal (assuming the channel name is 'ECG')
                ecg_data = raw.copy().pick_channels(['ECG']).get_data()[0]
                sfreq = raw.info['sfreq']  # current sampling frequency
                
                # Downsample ECG data, Convert the ECG signal to a NeuroKit2-compatible format
                ecg_downsampled = nk.signal_resample(ecg_data, sampling_rate=sfreq, desired_sampling_rate=new_sfreq) 
                time_vector = np.arange(len(ecg_downsampled)) / new_sfreq
                ecg_df = pd.DataFrame({'ECG': ecg_downsampled}, index=time_vector)
                ecg_df['ECG'] = ecg_df['ECG'].astype(float)

                
                # Process the downsampled ECG signal with NeuroKit2
                ecg_processed  = nk.ecg_process(ecg_df['ECG'], sampling_rate=new_sfreq)
                
                ecg_processed_df = ecg_processed[0]
                #ecg_processed_df.columns
                ecg_processed_dict = ecg_processed[1]
                #ecg_processed_dict.keys()
                
                # plot
                # nk.ecg_plot(ecg_processed_df)
                # ecg_analyzed = nk.ecg_analyze(ecg_processed, method = "interval-related")
    
                
                #### 1. HR
                
                hr = ecg_processed_df['ECG_Rate']
                # Calculate the average heart rate
                average_heart_rate = hr.mean()
                average_heart_rate = float(average_heart_rate)
                print(f"Average Heart Rate: {average_heart_rate} bpm")
                
                
                #### 2. HRV
                
                # Extract R-peaks (in samples)
                R_peaks = ecg_processed_dict['ECG_R_Peaks'] # Note: Using ecg_processed_dict for consistency

                # Convert R-peaks into R-R intervals (in seconds)
                #rri_sec = np.diff(R_peaks) / new_sfreq # NEW: Calculates intervals in seconds

                # Clean the R-R interval series using NeuroKit2's signal_fix
                #rri_cleaned, _ = nk.signal_fixpeaks(
                #    rri_sec, 
                #    method="neurokit", # This method is effective for identifying ectopic beats
                #    sampling_rate=1.0 # NEW: Ensures time unit is interpreted as seconds
                #)

                # Calculate HRV on the cleaned series
                # hrv = nk.hrv(rri_cleaned, sampling_rate=1.0) # CHANGED: Input is rri_cleaned

                # Alternatively, calculate HRV directly from R-peaks
                hrv = nk.hrv(R_peaks)
                # Extract RMSSD value
                rmssd = hrv['HRV_RMSSD'].values[0]
                rmssd_value_ms = float(rmssd) * 1000 # NEW: Correctly converts result from seconds to milliseconds
                
                print(f"HRV (RMSSD): {rmssd_value_ms:.2f} ms") # Added unit to print                
                
                
                #### 3. Amplitude of the R-wave
                
                
                ecg_data_resampled = raw.copy().pick_channels(['ECG'])
                #create a downsampled ecg data using mne
                ecg_data_resampled.resample(sfreq=new_sfreq) # you need to do this bc ecg_downsampled is not an mne object

                # Create epochs around R-peaks
                ecg_events,_,_ = mne.preprocessing.find_ecg_events(ecg_data_resampled, ch_name='ECG')
                ecg_epochs = mne.preprocessing.create_ecg_epochs(ecg_data_resampled, ch_name='ECG',picks = 'ECG', tmin = ecg_epochs_tmin, tmax = ecg_epochs_tmax, baseline = baseline) # creates epochs around R-peaks
                # did not drop bads so far ! could add a reject=XX, parameter above?!


                if double_rpeak_exclusion:

                    print("\nChecking for epochs with multiple R-peaks...")
                    # List to store indices of epochs with >1 R-peak
                    double_rpeak_epochs = []

                    sfreq = ecg_epochs.info['sfreq']
                

                    # Epoch starts and ends in samples (adjust start by tmin)
                    epoch_starts = ecg_epochs.events[:, 0] + int(ecg_epochs_tmin * sfreq)
                    epoch_ends   = epoch_starts + len(ecg_epochs.times)

                    # All R-peak event samples
                    rpeak_samples = ecg_events[:, 0]  # 1D array of sample indices

                    # Boolean mask: True if epoch has multiple R-peaks
                    extra_rpeak_mask = np.array([
                        np.sum((rpeak_samples >= start) & (rpeak_samples < end)) > 1  # >1 means multiple R-peaks
                        for start, end in zip(epoch_starts, epoch_ends)
                    ])

                    # Keep only epochs with 1 or 0 R-peaks
                    clean_mask = ~extra_rpeak_mask
                    heartbeat_epochs_clean = ecg_epochs[clean_mask]

                    # how many double peak epochs were found?
                    num_double_rpeak_epochs = np.sum(extra_rpeak_mask)

                    # updated percentage dropped epochs
                    total_epochs_nr = len(ecg_epochs)
                    percentage_dropped_epochs_double_peaks_cleaned = ((total_epochs_nr - len(heartbeat_epochs_clean)) / total_epochs_nr) * 100


                    print(f"Kept {len(heartbeat_epochs_clean)} / {len(ecg_epochs)} epochs")

                else:
                    percentage_dropped_epochs_double_peaks_cleaned = 0
                    num_double_rpeak_epochs = 'N/A'
                    heartbeat_epochs_clean = ecg_epochs


                # Access the epochs data
                epoch_data = heartbeat_epochs_clean.get_data()
                
                # Define the time point within the epoch to extract the R-peak amplitude, Calculate index for time point 0s, since we're focusing on the peak amplitude in the epoch
                r_peak_amplitude_index = int(abs(ecg_epochs_tmin) * ecg_epochs.info['sfreq'])  # Index for the R-peak position within the epoch
                
                # Extract R-peak amplitudes from all epochs
                # Assuming single channel, adjust channel index if multiple channels are used
                r_peak_amplitudes = epoch_data[:, 0, r_peak_amplitude_index]
                
                # Calculate the average R-peak amplitude
                average_r_peak_amplitude = np.mean(r_peak_amplitudes)
                
                average_r_peak_amplitude_microvolts = average_r_peak_amplitude * 1e6
                average_r_peak_amplitude_microvolts = float(average_r_peak_amplitude_microvolts)
                average_r_peak_amplitude_mV = average_r_peak_amplitude_microvolts/1000
                print(f'Average R Peak Amplitude in mV: {average_r_peak_amplitude_mV} mV') 
                print(f'Average R Peak Amplitude in µV: {average_r_peak_amplitude_microvolts:.6f} µV')  # Corrected this line
                
                """
                from neurokit:
                    # Baseline correction
                ecg_baseline_corrected = ecg_downsampled - np.mean(ecg_downsampled)
                
                # Process the baseline-corrected ECG signal
                ecg_processed_corrected = nk.ecg_process(ecg_baseline_corrected, sampling_rate=new_sfreq)
                
                # Extract R-wave amplitudes
                r_wave_amplitude = ecg_processed_corrected[0]['ECG_Clean'].loc[ecg_processed_corrected[1]['ECG_R_Peaks'] == 1].mean()
                print(f'Average Amplitude of R Wave: {r_wave_amplitude:.2f} µV')
                """

                ##### 0. store the evoked ECG amplitude for the time of interest for all runs

                
                ecg_mean = heartbeat_epochs_clean.average(picks = 'ECG')
                # ecg_mean.plot
                
                # Append to list
                if task == 'eyes-closed' and session == 'V1':
                    ecg_list_eyes_closed_V1.append(ecg_mean)
                    ecg_evoked_metadata_eyes_closed_V1.append(filename)
                elif task == 'eyes-open' and session == 'V1':
                    ecg_list_eyes_open_V1.append(ecg_mean)
                    ecg_evoked_metadata_eyes_open_V1.append(filename)
                elif task == 'hct' and session == 'V1':
                    ecg_list_hct_V1.append(ecg_mean)
                    ecg_evoked_metadata_hct_V1.append(filename)
                elif task == 'eyes-closed' and session == 'V3':
                    ecg_list_eyes_closed_V3.append(ecg_mean)
                    ecg_evoked_metadata_eyes_closed_V3.append(filename)
                elif task == 'eyes-open' and session == 'V3':
                    ecg_list_eyes_open_V3.append(ecg_mean)
                    ecg_evoked_metadata_eyes_open_V3.append(filename)
                elif task == 'hct' and session == 'V3':
                    ecg_list_hct_V3.append(ecg_mean)
                    ecg_evoked_metadata_hct_V3.append(filename)
                elif task == 'eyes-closed' and session == None:
                    ecg_list_eyes_closed_no_session.append(ecg_mean)
                    ecg_evoked_metadata_eyes_closed_no_session.append(filename)
                elif task == 'eyes-open' and session == None:
                    ecg_list_eyes_open_no_session.append(ecg_mean)
                    ecg_evoked_metadata_eyes_open_no_session.append(filename)
                elif task == 'hct' and session == None:
                    ecg_list_hct_no_session.append(ecg_mean)
                    ecg_evoked_metadata_hct_no_session.append(filename)
                
                # Select the time indices for this range and calculate the mean amplitude
                mask = (ecg_mean.times >= time_window[0]) & (ecg_mean.times <= time_window[1])
                # calculate the mean amplitude of the ECG signal within the time window
                ecg_mean_amplitude_time_window = ecg_mean.data[:, mask].mean(axis=1) # axis 1 is the time axis, axis 0 is the channel axis
                ecg_mean_amplitude_time_window = ecg_mean_amplitude_time_window[0] # un-array the value
                ecg_sd_amplitude_time_window = ecg_mean.data[:, mask].std(axis=1)
                ecg_sd_amplitude_time_window = ecg_sd_amplitude_time_window[0] # un-array the value

                
                #### 4. QT Interval
                
                # Calculate QT interval
                # Assuming 'ECG_Q_Peaks' and 'ECG_T_Offsets' are in the same unit (e.g., seconds or samples)
                T_offsets = np.array(ecg_processed_dict['ECG_T_Offsets'])
                #T_offsets = float(T_offsets)
                len(T_offsets)
                
                Q_peaks = np.array(ecg_processed_dict['ECG_Q_Peaks'])
                #Q_peaks = float(Q_peaks)
                len(Q_peaks)
                
                ecg_processed_dict['QT_Interval'] = T_offsets - Q_peaks
                QT_intervals = ecg_processed_dict['QT_Interval']
                # Convert the QT intervals from samples to milliseconds
                QT_intervals = (QT_intervals / new_sfreq) * 1000
                average_QT_interval_ms = np.nanmean(QT_intervals)
                average_QT_interval_ms = float(average_QT_interval_ms)
                print(f"average_QT_interval: {average_QT_interval_ms}") 
                
                
                #### 5. QTc Interval
                
                # The corrected QT interval (QTc) was calculated using the Bazett formula (Bazett, 1997).
                        
                # Calculate RR intervals
                R_peaks_list = list(R_peaks)
                RRs = []
                for i in range(len(R_peaks_list)-1):
                    RRs.append(R_peaks_list[i+1]- R_peaks_list[i])
                
                # Convert RR intervals to numpy array
                RRs = np.array(RRs)
                
                # Calculate average RR interval in samples
                average_RR = np.mean(RRs)
                average_RR = float(average_RR)
                
                # Convert average RR interval to milliseconds
                average_RR_msec = (average_RR / new_sfreq) * 1000
                
                qtc_seconds = (average_QT_interval_ms / 1000) / np.sqrt(average_RR_msec / 1000)
                
                qtc_seconds = float(qtc_seconds)
                
                qtc_ms = qtc_seconds*1000

                
                #### 6. Prepare the CSVto Save 
                
                # participant id should be BTSCZ...
                participant_id = f"BTSCZ{participant_no}" if "BTSCZ" in filename else f"BHC{participant_no}"

                
                # Create a dictionary representing the new row
                new_row = pd.Series({'subject_id': participant_id, 
                                    'session': session, 
                                    'task': task,
                                    'heart_rate_bpm':average_heart_rate,
                                    'hrv_rmssd_ms': rmssd_value_ms, 
                                    'R_peak_amplitude_mV': average_r_peak_amplitude_mV, 
                                    'QT_interval_ms': average_QT_interval_ms, 
                                    'QTc_interval_ms': qtc_ms, 
                                    'baseline': baseline,
                                    'new_sampling_freq': new_sfreq,
                                    'ecg_epochs_tmin': ecg_epochs_tmin,
                                    'ecg_epochs_tmax': ecg_epochs_tmax,
                                    'time_window': time_window,
                                    'ecg_mean_amplitude_time_window': ecg_mean_amplitude_time_window,
                                    'ecg_sd_amplitude_time_window': ecg_sd_amplitude_time_window,
                                    'ecg_epochs_tmin_tmax' : (ecg_epochs_tmin, ecg_epochs_tmax),
                                    'num_double_rpeak_epochs': num_double_rpeak_epochs,
                                    'percentage_dropped_epochs_double_peaks_cleaned': percentage_dropped_epochs_double_peaks_cleaned,
                                    'time_window' : time_window
                                    })
                
                # convert row to df
                new_row =  new_row.to_frame().T
                
                # add to existing df the current data outputs
                ecg_outputs = pd.concat([ecg_outputs, new_row], ignore_index=True)
                
                # Print the DataFrame
                print(ecg_outputs)
                            
                ### Save the Data 
                
                #### Save !                        
                #ecg_file_name = filename.replace('_prep_ICA', '_ecg-ave')
                #file_path = os.path.join('/Users/denizyilmaz/Desktop/BrainTrain/BrainTrain_EEG_data/ECG', ecg_file_name)
                #mne.write_evokeds(file_path, ecg_mean, on_mismatch='raise', overwrite=True,)   # evokedEvoked instance, or list of Evoked instance; to load it back: evokeds_list = mne.read_evokeds(evk_file, verbose=False)

                # create dynamic folder name for processing style   
                processing_style_epoch = f"ecg-epoch-{ecg_epochs_tmin:.2f}-{ecg_epochs_tmax:.2f}"    
                processing_style_baseline = f"baseline-{baseline[0]:.2f}-{baseline[1]:.2f}"   
                processing_style_time_window = f"time_window-{time_window[0]:.2f}-{time_window[1]:.2f}"
                processing_subfolder_name = f"ECG_{processing_style_epoch}_{processing_style_baseline}_{processing_style_time_window}"
                double_rpeak_exclusion_str = "with-double-R-peak-exclusion" if double_rpeak_exclusion else ""

                # SAVE the Preprocessed Data & Prep Ouputs in a CSV
                folder_name = 'ECG' # this is the main folder where all the outputs will be saved"
                output_folder_path = os.path.join(os.getcwd(), folder_name, processing_subfolder_name, double_rpeak_exclusion_str)
                print(f"Creating directory: {output_folder_path}")
                os.makedirs(output_folder_path, exist_ok=True)
                
                # Save all Mean ECGs in a single file per run including all subs
                if task == 'eyes-closed' and session == 'V1':
                    mne.write_evokeds(os.path.join(output_folder_path, 'SSD_ecg_mean_list_V1_eyes_closed-ave.fif'), ecg_list_eyes_closed_V1, overwrite=True)
                    ecg_evoked_metadata_eyes_closed_V1_path = os.path.join(output_folder_path, 'ecg_evoked_metadata_eyes_closed_V1.csv')
                    ecg_evoked_metadata_eyes_closed_V1_df = pd.DataFrame(ecg_evoked_metadata_eyes_closed_V1, columns=["filename"])
                    ecg_evoked_metadata_eyes_closed_V1_df.to_csv(ecg_evoked_metadata_eyes_closed_V1_path, index=False)
                elif task == 'eyes-open' and session == 'V1':
                    mne.write_evokeds(os.path.join(output_folder_path, 'SSD_ecg_mean_list_V1_eyes_open-ave.fif'), ecg_list_eyes_open_V1, overwrite=True)
                    ecg_evoked_metadata_eyes_open_V1_path = os.path.join(output_folder_path, 'ecg_evoked_metadata_eyes_open_V1.csv')
                    ecg_evoked_metadata_eyes_open_V1_df = pd.DataFrame(ecg_evoked_metadata_eyes_open_V1, columns=["filename"])
                    ecg_evoked_metadata_eyes_open_V1_df.to_csv(ecg_evoked_metadata_eyes_open_V1_path, index=False)
                elif task == 'hct' and session == 'V1':
                    mne.write_evokeds(os.path.join(output_folder_path, 'SSD_ecg_mean_list_V1_hct-ave.fif'), ecg_list_hct_V1, overwrite=True)
                    ecg_evoked_metadata_hct_V1_path = os.path.join(output_folder_path, 'ecg_evoked_metadata_hct_V1.csv')
                    ecg_evoked_metadata_hct_V1_df = pd.DataFrame(ecg_evoked_metadata_hct_V1, columns=["filename"])
                    ecg_evoked_metadata_hct_V1_df.to_csv(ecg_evoked_metadata_hct_V1_path, index=False)
                elif task == 'eyes-closed' and session == 'V3':
                    mne.write_evokeds(os.path.join(output_folder_path, 'SSD_ecg_mean_list_V3_eyes_closed-ave.fif'), ecg_list_eyes_closed_V3, overwrite=True)
                    ecg_evoked_metadata_eyes_closed_V3_path = os.path.join(output_folder_path, 'ecg_evoked_metadata_eyes_closed_V3.csv')
                    ecg_evoked_metadata_eyes_closed_V3_df = pd.DataFrame(ecg_evoked_metadata_eyes_closed_V3, columns=["filename"])
                    ecg_evoked_metadata_eyes_closed_V3_df.to_csv(ecg_evoked_metadata_eyes_closed_V3_path, index=False)
                elif task == 'eyes-open' and session == 'V3':
                    mne.write_evokeds(os.path.join(output_folder_path, 'SSD_ecg_mean_list_V3_eyes_open-ave.fif'), ecg_list_eyes_open_V3, overwrite=True)
                    ecg_evoked_metadata_eyes_open_V3_path = os.path.join(output_folder_path, 'ecg_evoked_metadata_eyes_open_V3.csv')
                    ecg_evoked_metadata_eyes_open_V3_df = pd.DataFrame(ecg_evoked_metadata_eyes_open_V3, columns=["filename"])
                    ecg_evoked_metadata_eyes_open_V3_df.to_csv(ecg_evoked_metadata_eyes_open_V3_path, index=False)
                elif task == 'hct' and session == 'V3':
                    mne.write_evokeds(os.path.join(output_folder_path, 'SSD_ecg_mean_list_V3_hct-ave.fif'), ecg_list_hct_V3, overwrite=True)
                    ecg_evoked_metadata_hct_V3_path = os.path.join(output_folder_path, 'ecg_evoked_metadata_hct_V3.csv')
                    ecg_evoked_metadata_hct_V3_df = pd.DataFrame(ecg_evoked_metadata_hct_V3, columns=["filename"])
                    ecg_evoked_metadata_hct_V3_df.to_csv(ecg_evoked_metadata_hct_V3_path, index=False)
                elif task == 'eyes-closed' and session == None:
                    mne.write_evokeds(os.path.join(output_folder_path, 'HC_ecg_mean_list_V1_eyes_closed-ave.fif'), ecg_list_eyes_closed_no_session, overwrite=True)
                    ecg_evoked_metadata_eyes_closed_no_session_path = os.path.join(output_folder_path, 'ecg_evoked_metadata_eyes_closed_no_session.csv')
                    ecg_evoked_metadata_eyes_closed_no_session_df = pd.DataFrame(ecg_evoked_metadata_eyes_closed_no_session, columns=["filename"])
                    ecg_evoked_metadata_eyes_closed_no_session_df.to_csv(ecg_evoked_metadata_eyes_closed_no_session_path, index=False)
                elif task == 'eyes-open' and session == None:
                    mne.write_evokeds(os.path.join(output_folder_path, 'HC_ecg_mean_list_V1_eyes_open-ave.fif'), ecg_list_eyes_open_no_session, overwrite=True)
                    ecg_evoked_metadata_eyes_open_no_session_path = os.path.join(output_folder_path, 'ecg_evoked_metadata_eyes_open_no_session.csv')
                    ecg_evoked_metadata_eyes_open_no_session_df = pd.DataFrame(ecg_evoked_metadata_eyes_open_no_session, columns=["filename"])
                    ecg_evoked_metadata_eyes_open_no_session_df.to_csv(ecg_evoked_metadata_eyes_open_no_session_path, index=False)
                elif task == 'hct' and session == None:
                    mne.write_evokeds(os.path.join(output_folder_path, 'HC_ecg_mean_list_V1_hct-ave.fif'), ecg_list_hct_no_session, overwrite=True)
                    ecg_evoked_metadata_hct_no_session_path = os.path.join(output_folder_path, 'ecg_evoked_metadata_hct_no_session.csv')
                    ecg_evoked_metadata_hct_no_session_df = pd.DataFrame(ecg_evoked_metadata_hct_no_session, columns=["filename"])
                    ecg_evoked_metadata_hct_no_session_df.to_csv(ecg_evoked_metadata_hct_no_session_path, index=False)

                
                ### CSV : can be in or out of the loop
                
                # Construct the csv path
                current_date = datetime.datetime.now().strftime("%Y%m%d")
                csv_filename = f'SSD_ecg_outputs_{current_date}.csv' if "BTSCZ" in filename else f'HC_ecg_outputs_{current_date}.csv'
                csv_path = os.path.join(output_folder_path, csv_filename)
                ecg_outputs.to_csv(csv_path, mode='w', sep=',', index=False)
                
                
                
                
                
                
                
                


import os
import pandas as pd
import numpy as np
import datetime
import gc
import logging
import mne
from pyprep.find_noisy_channels import NoisyChannels
from pandas import concat
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
from mne_icalabel import label_components
from autoreject import get_rejection_threshold  
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from autoreject import AutoReject # for rejecting bad channels

# Set up logging
logging.basicConfig(
    log_filename='prep_until_ica.log',   # log file
    filemode='a',                    # overwrite each run; 'a' to append
    level=logging.INFO,              # INFO level; DEBUG for more details
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("Logging started")  # test entry


def prep_until_ica(data_dir, behavioral_data_path, new_sampling_rate=250, low_freq=0.3, high_freq=45.0, line_freq=50 ):

    """
    Look at this great function I wrote! It does a lot of things.

    Preprocess EEG data until ICA.

    Parameters:
        data_dir (str): Path to the directory containing EEG files.
        behavioral_data_path (str): Path to the behavioral data CSV.
        new_sampling_rate (int): Target sampling rate for resampling.
        low_freq (float): Lower cutoff frequency for bandpass filtering.
        high_freq (float): Upper cutoff frequency for bandpass filtering.
        line_freq (float): Frequency for line noise removal.
        tasks (list): List of task names.
        exclude_criteria (dict): Custom criteria for exclusion (optional).

    Returns:
        Preprocessed data files: EEG data files preprocessed until ICA.
        DataFrame: Summary of preprocessing steps and outcomes.
        PSD images

    """

    # %%  00. Define WD and other vars


    # change dir to where the data is
    os.chdir(data_dir)

    ##### define default vars for the function
    #new_sampling_rate = 250
    #low_freq = 0.3 # Lower cutoff frequency (in Hz)
    #high_freq = 45.0 # Upper cutoff frequency (in Hz)
    bandpass_filter = (low_freq, high_freq) # Define the bandpass filter frequency range
    #line_freq = 50  # Apply notch filter to remove line noise (e.g., 50 Hz from Antonin's manuscript)


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
            #print("processing HC data")
            logging.info("processing HC data") 
        elif sz_id in filename:
            is_sz = True
            sessions= ['V1'] #sessions= ['V1', 'V3']
            #print("processing SZ data")
            logging.info("processing SZ data")
    if not is_hc and not is_sz:
        raise ValueError("No participants found in the directory.")
    

    # %%  0. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

    column_names = ['subject_id', 'session', 'task', 
                    'montage', 'new_sampling_rate', 
                    'removed_line_noise_freq', 'rereferencing', 
                    'total_electrode_nr', 'interpolated_electrode_nr', 'percent_interpolated_electrode', 
                    'interpolated_chans', 'bad_chan_detect_method',
                    'bandpass_filtering', 'bandpass_filter_method',
                    'start_time_of_analysis', 'analysis_duration'
                    ]
    
    # Initialize an empty DF with columns
    prep_outputs = pd.DataFrame(columns=column_names)

    # %% 1. Import data & define excluded files
    """
    .eeg: This file contains the actual EEG data. It's the raw EEG signal data you want to analyze.

    .vhdr: This is the header file, and it contains metadata and information about the EEG recording, such as channel names, sampling rate, and electrode locations. This file is essential for interpreting the EEG data correctly.

    .vmrk: This file contains event markers or annotations that correspond to events in the EEG data. Event markers can be used to mark the timing of specific events or stimuli during the EEG recording. This file is useful for event-related analyses.
    """
    # Get a list of filenames from the current directory
    all_files = os.listdir()
    # Filter filenames that end with ".eeg"
    eeg_file_names = [filename for filename in all_files if filename.endswith('.eeg')]

    # Initialize Participant, Task, Session : define all file relevant variables

    participant_numbers = []
    if is_sz:
        for file in eeg_file_names:
            num = file[5:8]
            #print(num)
            participant_numbers.append(num)
    elif is_hc:
        for file in eeg_file_names:
            num = file[3:6]
            #print(num)
            participant_numbers.append(num)
        
    participant_numbers =  list(set(participant_numbers))
    participant_numbers = np.array(sorted(participant_numbers))

    ########## define and exclude the excluded participants
    # read data from behavioral_data_path
    if behavioral_data_path.endswith('.csv'):
        beh_data = pd.read_csv(behavioral_data_path)
    elif behavioral_data_path.endswith(('.xls', '.xlsx')):
        beh_data = pd.read_excel(behavioral_data_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
    
    # If the columns contain unexpected string representations
    # Strip whitespace from column names
    beh_data.columns = beh_data.columns.str.strip()

    excluded_files = []
    for _, row in beh_data.iterrows():
        if is_sz:
            session = row['Session']
            participant_no = row['subject'][2:]
            if row['excluded'] == 1:
                excluded_files.append(f"BTSCZ{participant_no}_{session}_eyes-closed.vhdr")
                excluded_files.append(f"BTSCZ{participant_no}_{session}_eyes-open.vhdr")
                excluded_files.append(f"BTSCZ{participant_no}_{session}_hct.vhdr")
            if row['excluded_ec'] == 1:
                excluded_files.append(f"BTSCZ{participant_no}_{session}_eyes-closed.vhdr")
            if row['excluded_eo'] == 1:
                excluded_files.append(f"BTSCZ{participant_no}_{session}_eyes-open.vhdr")
            if row['excluded_hct_eeg'] == 1:
                excluded_files.append(f"BTSCZ{participant_no}_{session}_hct.vhdr")
        elif is_hc:
            participant_no = row['subject'][3:6]
            if row['excluded'] == 1:
                excluded_files.append(f"BHC{participant_no}_eyes-closed.vhdr")
                excluded_files.append(f"BHC{participant_no}_eyes-open.vhdr")
                excluded_files.append(f"BHC{participant_no}_hct.vhdr")
            if row['excluded_ec'] == 1:
                excluded_files.append(f"BHC{participant_no}_eyes-closed.vhdr")
            if row['excluded_eo'] == 1:
                excluded_files.append(f"BHC{participant_no}_eyes-open.vhdr")
            if row['excluded_hct_eeg'] == 1:
                excluded_files.append(f"BHC{participant_no}_hct.vhdr")
    
    #print(f"Excluded files are: {excluded_files}")
    logging.info(f"Excluded files are: {excluded_files}")  # Log excluded files

 

    # %% 2. Loop over all participants, sessions, and tasks (if file is not excluded) and cut the data to remove bad segments

    for participant_no in participant_numbers:
        logging.info(f"Starting participant {participant_no}")
        # flush to make sure it's written immediately
        for handler in logging.getLogger().handlers:
            handler.flush()
        for session in (sessions if sessions else [None]):            
            for task in tasks: 
                try:
                    if is_sz:
                        filename = f"BTSCZ{participant_no}_{session}_{task}.vhdr"
                        #print(f"Successfully loaded: {filename}")
                        logging.info(f"Successfully loaded: {filename}")
                    elif is_hc:
                        filename = f"BHC{participant_no}_{task}.vhdr"
                        #print(f"Successfully loaded: {filename}")
                        logging.info(f"Successfully loaded: {filename}")
                    if filename in excluded_files:
                        #print(f"Skipping file {filename} because they are excluded.")
                        logging.info(f"Skipping file {filename} because they are excluded.")
                        continue # skip this iteration if participant is excluded

                    try:
                        raw = mne.io.read_raw(filename, preload=True)
                    except FileNotFoundError:
                        #print(f"File not found: {filename}. Skipping...")
                        logging.warning(f"File not found: {filename}. Skipping...")
                        continue
                    
                    start_time = datetime.datetime.now()
                    
                    # do some specific data manipulations and cut bad segments..

                    last_sec = raw.times[-1]

                    if filename == "BTSCZ002_V3_eyes-closed.vhdr":
                        raw.crop(tmin=5)
                    elif filename == "BTSCZ006_V3_eyes-closed.vhdr":
                        tmax = last_sec - 4
                        raw.crop(tmax=tmax)
                    elif filename == "BTSCZ007_V3_eyes-closed.vhdr":
                        tmax = last_sec - 10
                        raw.crop(tmax=tmax)
                    elif filename == "BTSCZ009_V1_eyes-closed.vhdr":
                        raw.crop(tmin=20)
                    elif filename == "BTSCZ017_V1_eyes-closed.vhdr":
                        raw.crop(tmin=12)
                    elif filename == "BTSCZ026_V1_eyes-open.vhdr":
                        raw.crop(tmin=42)
                    elif filename == "BTSCZ026_V3_eyes-open.vhdr":
                        tmax = last_sec - 231
                        raw.crop(tmax=tmax)
                    elif filename == "BTSCZ043_V1_eyes-open.vhdr":
                        raw_part1 = raw.copy().crop(tmax=256)
                        raw_part2 = raw.copy().crop(tmin=280)
                        raw = mne.concatenate_raws([raw_part1, raw_part2])

                    # for   "BHC003_eyes-open.vhdr" delete first 15 seconds
                    elif filename == "BHC003_eyes-open.vhdr":
                        #print("delete first 15secs")
                        logging.info("delete first 15secs")
                        raw.crop(tmin=15)
                    # BHC023_eyes-closed.vhdr a bit of noise coming from outside at min 3:15; delete 3:12-3:25
                    elif filename == "BHC023_eyes-closed.vhdr":
                        #print("delete the seconds 192-205")
                        logging.info("delete the seconds 192-205")
                        # Create two copies of the raw data
                        raw_part1 = raw.copy().crop(tmax=192)
                        raw_part2 = raw.copy().crop(tmin=205)
                        # Concatenate the two parts
                        raw = mne.concatenate_raws([raw_part1, raw_part2])
                    # BHC054 Eyes-open, ECG electrode fell off from 91. second to the end of the recording, delete that segment
                    elif filename == "BHC054_eyes-open.vhdr": 
                        #print("delete the seconds 91-300")
                        logging.info("delete the seconds 91-300")
                        raw.crop(tmax=91)  # Keep the first 91 seconds
                                        
                    # %% 3. Montage
                    # check our channel names
                    #print(raw.ch_names)
                    
                    # some channels are named according to the old nomenclature, change them
                    # Create a dictionary to map old channel names to new channel names
                    channel_renaming = {
                        'T3': 'T7',
                        'T4': 'T8',
                        'T5': 'P7',
                        'T6': 'P8',
                    }           
                                            
                    raw.rename_channels(channel_renaming)

                    if 'RESP' in raw.ch_names:
                        channel_type_mapping = {
                            'ECG': 'ecg',
                            'RESP': 'resp'
                        }
                        raw.set_channel_types(channel_type_mapping)
                    else:
                        channel_type_mapping = {
                            'ECG': 'ecg'
                        }
                        raw.set_channel_types(channel_type_mapping)

                    montage = mne.channels.make_standard_montage('easycap-M1')
                    raw.set_montage(montage)
                    # raw.plot_sensors(show_names=True)

                    # %% 4. Resampling to 250 Hz, by 2 bc the initial sampling is 500
                                        
                    raw_resampled = raw.copy()
                    raw_resampled.resample(sfreq=new_sampling_rate) # In MNE-Python, the resampling methods (raw.resample(), epochs.resample() and evoked.resample()) apply a low-pass filter to the signal to avoid aliasing, so you don’t need to explicitly filter it yourself first. 
                    #print(raw_resampled.info["sfreq"])

                    # %% 5. Find ECG events & add annotations: some r peaks not detected, reduce treshold &try again!

                    ecg_events,_,_ = mne.preprocessing.find_ecg_events(raw_resampled, ch_name='ECG')
                    descriptions = ['R-peak'] * len(ecg_events)
                    R_peak_annotations = mne.Annotations(onset=ecg_events[:, 0] / raw_resampled.info['sfreq'], duration=0.0, description=descriptions)  # Create annotations time-locked to R-peak with event descriptions; for onset: Convert sample indices to seconds & Set the duration to zero for point events

                    existing_annotations = raw.annotations.copy()
                    combined_annotations = mne.Annotations(
                        onset=np.concatenate((existing_annotations.onset, R_peak_annotations.onset)),
                        duration=np.concatenate((existing_annotations.duration, R_peak_annotations.duration)),
                        description=np.concatenate((existing_annotations.description, R_peak_annotations.description))
                    )
                    raw_resampled.set_annotations(combined_annotations)

                    irrelevant_annotations = ['Comment/actiCAP Data On', 'New Segment/', 'Comment/actiCAP USB Power On', 'ControlBox is not connected via USB', 'actiCAP USB Power On', 'Comment/ControlBox is not connected via USB', 'BAD boundary', 'EDGE boundary']  # Adjust this list as needed
                    irrelevant_indices = np.where(np.isin(raw_resampled.annotations.description, irrelevant_annotations))[0]  # find indices of irrelevant annots
                    raw_resampled.annotations.delete(irrelevant_indices)

                    # Cut out the breaks in HCT
                    if task == 'hct':
                        events, event_id = mne.events_from_annotations(raw_resampled)   # Get event ids of HCT triggers
                        R_peak_id = event_id['R-peak']
                        hct_events = events[events[:, 2] != R_peak_id]

                        eeg_segments = [] # Create an empty list to store the EEG data segments
                        for i in range(0, len(hct_events), 2):
                            raw_copy = raw_resampled.copy()
                            raw_part = raw_copy.crop(tmin=hct_events[i][0] / raw_resampled.info['sfreq'], tmax=hct_events[i+1][0] / raw_resampled.info['sfreq']) # get tmin, tmax in seconds
                            eeg_segments.append(raw_part)
                        mne.concatenate_raws(eeg_segments) # Combine raw_parts into a single raw object
                        raw_resampled = eeg_segments[0]


                    # %% 6. Separate the EEG from ECG & RESP data

                    raw_eeg = raw_resampled.copy()
                    eeg_chans = raw_eeg.copy().pick(picks=["eeg"])
                    total_electrode_nr = len(eeg_chans.ch_names) # extract total nr of eeg channels 

                    # %% 7. Preprocess the EEG data: DO NOT INCLUDE the ECG ND RESP channel here !!

                    #print("Starting preprocessing block...!!!!!!")
                    logging.info("Starting preprocessing block...!!!!!!")  # Log the start of preprocessing


                    # ### A. Bandpass filter [0.3  45]
                    raw_filt = raw_eeg.copy()
                    raw_filt.filter(l_freq=low_freq, h_freq=high_freq, method='fir', phase='zero', picks=["eeg"])  # Apply the bandpass filter

                    # ### B. Remove line noise, 50 Hz
                    raw_filt_line = raw_filt.copy()
                    raw_filt_line.notch_filter(freqs=line_freq, picks=["eeg"])


                    # ### C. Detect & interpolate noisy channels
                    raw_filt_line_interp = raw_filt_line.copy()             # for NoisyChannels this Version: only EEG channels are supported and any non-EEG channels in the provided data will be ignored.
                    noisy_data = NoisyChannels(raw_filt_line_interp, random_state=1337) # Assign the mne object to the NoisyChannels class. The resulting object will be the place where all following methods are performed.
                    noisy_data.find_all_bads(ransac=True, channel_wise=False, max_chunk_size=None)

                    raw_filt_line_interp.info["bads"] = noisy_data.get_bads() # get channel names marked as bad and assign them into bads of the data from the step before
                    bads = noisy_data.get_bads()
                    rejected_electrode_nr = len(bads)
                    interpolated_bads_str = ', '.join(bads) # all bad in a string to record them in csv 
                    percent_rejected_electrode = (rejected_electrode_nr / total_electrode_nr) * 100  # calculate % bad electrodes

                    raw_filt_line_interp.interpolate_bads()  # Interpolate noisy Channels

                    del noisy_data  # Free memory used by NoisyChannels
                    gc.collect()

                    # ### D.Robust average rereferencing
                    raw_filt_line_interp_reref = raw_filt_line_interp.copy()
                    raw_filt_line_interp_reref.set_eeg_reference(ref_channels='average') # method is specifically designed for setting the EEG reference, and it will automatically apply to EEG channels.




                    #%% 8. Save the preprocessed data & prep outputs in a CSV

                    participant_id = f"BTSCZ{participant_no}" if "BTSCZ" in filename else f"BHC{participant_no}"
                    end_time = datetime.datetime.now() # record end_time 
                    duration = end_time - start_time   # Calculate the duration of the analysis

                    # Create a dictionary representing the new row
                    new_row = pd.Series({'subject_id': participant_id, 
                                            'session': session if "BTSCZ" in filename else None, 
                                            'task': task,
                                            'montage': "make_standard_montage('easycap-M1')",
                                            'new_sampling_rate': new_sampling_rate,
                                            'removed_line_noise_freq': line_freq,
                                            'rereferencing': 'robust average rereferencing',
                                            'total_electrode_nr': total_electrode_nr,
                                            'interpolated_electrode_nr': rejected_electrode_nr,
                                            'percent_interpolated_electrode': percent_rejected_electrode,
                                            'interpolated_chans': interpolated_bads_str,
                                            'bad_chan_detect_method': 'find_all_bads()',
                                            'bandpass_filtering': bandpass_filter,
                                            'bandpass_filter_method': 'fir',
                                            'start_time_of_analysis': start_time,
                                            'analysis_duration': duration
                                            })
                    new_row = new_row.to_frame().T  # convert row to df
                    prep_outputs = pd.concat([prep_outputs, new_row], ignore_index=True)  # add to existing df the current data outputs
                    

                    # SAVE the Preprocessed Data & Prep Ouputs in a CSV
    

                    folder_name = 'Preprocessed_until_ICA' if "BTSCZ" in filename else 'HC_Preprocessed_until_ICA'
                    processing_style_sampling = f"sampling-{new_sampling_rate}"                     # create dynamic folder name for processing style   
                    processing_style_bandpass = f"bandpass-{bandpass_filter[0]:.2f}-{bandpass_filter[1]:.2f}"
                    processing_style_line = f"line-{line_freq}" 
                    processing_style_interp = "find_all_bads"
                    processing_subfolder_name = f"{processing_style_sampling}_{processing_style_bandpass}_{processing_style_line}_{processing_style_interp}" # this sequence also tells in which order the processing steps were applied

                    output_folder_path = os.path.join(os.getcwd(), folder_name, processing_subfolder_name)
                    #print(f"Creating directory: {output_folder_path}")
                    logging.info(f"Creating directory: {output_folder_path}")  # Log the creation of the output folder
                    os.makedirs(output_folder_path, exist_ok=True)

                    ### Data
                    filename = os.path.splitext(os.path.basename(raw.filenames[0]))[0]
                    file_path = os.path.join(output_folder_path, filename + '_prep_until_ICA.fif')
                    raw_filt_line_interp_reref.save(file_path, overwrite=True)

                    ### CSV 
                    current_date = datetime.datetime.now().strftime("%Y%m%d")
                    csv_filename = f'prep_interp_output_{processing_style_sampling}_{processing_style_bandpass}_{processing_style_line}_{processing_style_interp}_{current_date}.csv' if "BTSCZ" in filename else f'HC_prep_interp_output_{processing_style_sampling}_{processing_style_bandpass}_{processing_style_line}_{processing_style_interp}_{current_date}.csv'
                    csv_path = os.path.join(output_folder_path, csv_filename)
                    prep_outputs.to_csv(csv_path, mode='w', sep=',', index=False)

                    ### PSD Plot : save psd_plot by putting it in a session folder in a participant folder; create that dir
                    psd_plot = raw_filt_line_interp_reref.plot_psd()

                    # Define base folder for saving images
                    images_base_path = os.path.join(output_folder_path, 'PSDs')
                    # Participant-specific folder
                    participant_folder = os.path.join(images_base_path, participant_id)
                    os.makedirs(participant_folder, exist_ok=True)  # Create folder if it doesn't exist
                    # Session-specific folder within the participant folder
                    participant_session_folder = os.path.join(participant_folder, f"ses-{session}") if "BTSCZ" in filename else participant_folder
                    os.makedirs(participant_session_folder, exist_ok=True)  # Create session folder if it doesn't exist
                    # Task-specific folder within the participant folder
                    participant_session_task_folder = os.path.join(participant_session_folder, f"task-{task}") 
                    os.makedirs(participant_session_task_folder, exist_ok=True)  # Create session folder if it doesn't exist
                    # save file
                    psd_file_path = os.path.join(participant_session_task_folder, filename + f"_prep_until_ICA_psd_{processing_style_sampling}_{processing_style_bandpass}_{processing_style_line}_{processing_style_interp}.jpg")
                    os.makedirs(os.path.dirname(psd_file_path), exist_ok=True)
                    psd_plot.savefig(psd_file_path, format='jpg')
                    plt.close(psd_plot)  # free memory

                    # Free memory
                    del raw, raw_resampled, raw_filt, raw_filt_line, raw_filt_line_interp, raw_filt_line_interp_reref
                    gc.collect()
                    if 'eeg_segments' in locals():
                        del eeg_segments
                    gc.collect()

                    logging.info(f"Finished processing {filename}")
                    for handler in logging.getLogger().handlers:
                        handler.flush()


                    # archive of PSD
                    #psd_folder_name = 'PSDs'
                    #psd_file_path = os.path.join(output_folder_path, psd_folder_name, filename + f"_prep_until_ICA_psd_{processing_style_sampling}_{processing_style_bandpass}_{processing_style_line}_{processing_style_interp}.jpg")
                    #os.makedirs(os.path.dirname(psd_file_path), exist_ok=True)
                    #psd_plot.savefig(psd_file_path, format='jpg')
                                

                except Exception as e:
                    #print(f"An error occurred while processing {filename}: {e}")
                    logging.error(f"An error occurred while processing {filename}: {e}")
                    continue
    gc.collect()
    logging.info("Final garbage collection complete, function finished.")
    logging.shutdown()


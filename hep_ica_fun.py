#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 16:22:45 2024

@author: denizyilmaz
"""


# %%  0. Import Packages 

import mne
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs, find_bad_channels_maxwell
from mne_icalabel import label_components
from mne.viz import plot_ica_sources
from autoreject import AutoReject # for rejecting bad channels
from autoreject import get_rejection_threshold  
from collections import Counter
from pyprep.find_noisy_channels import NoisyChannels
import matplotlib

mne.set_log_level("ERROR")

def hep_ica_fun(data_dir):

    os.chdir(data_dir)

    ##### define default vars for the function, turns out theres none lol, rendering processing_subfolder_name useless



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
            sessions= ['V1']  # sessions= ['V1', 'V3']
    if not is_hc and not is_sz:
        raise ValueError("No participants found in the directory.")


    # %%  1. Initialize a DF to store all prep outcomes to be able to report them later and exclude participants

    # Define the column names for the DF
    column_names = ['subject_id', 'session', 'task', 'total_explained_var_ratio',
                    'cont_ICA_labels', 'cont_ICA_percentage', 
                    'cont_ICA_heart_found', 'cont_ICA_blink_found', 'cont_ICA_muscle_found', 
                    'bads_ecg_indices', 'bads_ecg_scores_corr', 'n_bads_ecg', 
                    'num_components_total', 'num_components_excluded', 'rejection_criteria'
                    ]

    # Initialize an empty DF with columns
    ica_raw_outputs = pd.DataFrame(columns=column_names)


    # %% Perform ICA on RAW data

    ### Find Participant numbers

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

    # %% Loop through all participants, sessions, and tasks


    for participant_no in participant_numbers:
        for session in (sessions if sessions else [None]): 
            for task in tasks: 
                            
                if is_sz:
                    filename = f"BTSCZ{participant_no}_{session}_{task}_prep_until_ICA.fif"
                elif is_hc:
                    filename = f"BHC{participant_no}_{task}_prep_until_ICA.fif"

                
                try:
                    prep_data = mne.io.read_raw(filename, preload=True)
                    print(f"Successfully loaded: {filename}")
                except FileNotFoundError:
                    print(f"File not found: {filename}. Skipping...")
                    continue



                # For ICA to perform better we need filter 1, THEN copy weights back to filter .3
                raw_ica_filtered = prep_data.copy().filter(l_freq=1.0, h_freq=100)  # you can try and see lower lowpass (e.g. 0.1, 0.3,..)for heart artifacts but other components may get worse
                
                """
                Richard: 
                    you can simply stick with the default values here – we automatically account for rank deficiency, 
                    which seems to work in almost all cases. So: don’t set n_components, don’t set n_pca_components, 
                    and you should be good in 99% of the cases. 
                    After fitting, check ica.n_components_ (mind the trailing underscore) to find out how many components were kept.
                    
                    ARCHIVE: 

                    # decide on component number for ICA
                    good_channels = mne.pick_types(raw_ica_filtered.info, meg=False, eeg=True, exclude='bads')
                    best_n_components = len(good_channels) - 1 # -1 for the acerage rereferencing beforehand
                    print("Components after having accounted for rank deficiency: ", best_n_components)
                    
                    # now reset bads because theyve been interpolated anyway
                    raw_ica_filtered.info["bads"] = []
                    prep_data.info["bads"] = []
                    print("Number of bad channels for raw_ica_filtered: ", raw_ica_filtered.info["bads"])
                    print("Number of bad channels for prep_data: ", prep_data.info["bads"])
                """
                
                
                # Set up and fit the ICA
                ica = ICA(             #  n_components=best_n_components, AUTOMATICALLY DONE reduce the dim (by 1 for average reference and 1 for each interpolated channel) for optimal ICA performance
                    max_iter="auto", 
                    method="infomax", 
                    random_state=97,
                    fit_params=dict(extended=True)
                    ) # n_components should be fit to the # interpolated channels ... ICLabel requires extended infomax!
                ica.fit(raw_ica_filtered) # fit the ICA on the ICA-filtered data, not the original data!
                ica
                
                # Print explained var for ICA
                explained_var_ratio = ica.get_explained_variance_ratio(raw_ica_filtered)
                for channel_type, ratio in explained_var_ratio.items():
                    print(f"Fraction of {channel_type} variance explained by all components: " f"{ratio}")
                
                # %%  #####  Plot ICs: From here use the original prep data!
                
                # load data
                prep_data.load_data()

                # Plot topographies
                # ica.plot_components(inst = prep_data)
                topo_figs = ica.plot_components(inst=prep_data, show=False)  # Topo plot (add 'show=False' for batch saving)
                plt.close("all")  # Close all open Matplotlib figures to free memory


                # Plot time courses
                # ica.plot_sources(prep_data, show_scrollbars=False) # you can call the original unfiltered raw object
                timecourse_fig = ica.plot_sources(prep_data, show_scrollbars=False, show=False)  # Timecourse plot
                timecourse_fig.close() # Close figure to free memory

                
                # %% ### Automatically label components using the 'iclabel' method

                ic_labels = label_components(prep_data, ica, method='iclabel')
                component_labels = ic_labels['labels']
                predicted_probabilities = ic_labels['y_pred_proba']
                                    
                # Print the results
                print("Predicted Probabilities:", ic_labels['y_pred_proba'])
                print("Component Labels:", ic_labels['labels'])
                # Maybe: Create a dictionary mapping component labels to their probabilities
                
                # Check whether heart component was found 
                nr_heart_components = 0
                nr_blink_components = 0
                nr_muscle_components = 0

                for label in component_labels:
                    if label == 'heart beat':
                        nr_heart_components = nr_heart_components + 1
                    elif label == 'eye blink':
                        nr_blink_components = nr_blink_components + 1
                    elif label == 'muscle artifact':
                        nr_muscle_components = nr_muscle_components + 1
                
                
                # Extract non-brain labels' index to exclude them from original data
                # only those labels where algorithm assigns above chance probability to the label, as per Berkan's suggestion
                labels = ic_labels["labels"]
                exclude_index = [
                    index for index, label in enumerate(labels) if label not in ["brain", "other"] and predicted_probabilities[index] > 0.50
                ]
                
                # # ADD: CORRELATION WITH ECG signal!!  find which ICs match the ECG pattern and exclude those too
                ecg_indices, ecg_scores = ica.find_bads_ecg(prep_data, method="correlation",  threshold="auto") # leave measure as the default (z-score) 'zscore' can be more sensitive to subtle ECG artifacts because it accounts for relative deviations across all components.
                # convert all elements in the ecg_indices list to integers
                ecg_indices = [int(index) for index in ecg_indices]
                n_bads_ecg = len(ecg_indices)
                exclude_index.extend(ecg_indices)
                #exclude_index.extend([int(index) for index in ecg_indices]) 
                # remove duplicates
                exclude_index = list(set(exclude_index))

                # Assign those bads ICs to ica.exclude
                ica.exclude = exclude_index
                print(f"Excluding these ICA components: {exclude_index}")
                
                
                # Exclude the bad Components: Reconstruct the original data without noise components
                # ica.apply() changes the Raw object in-place, so let's make a copy first:
                prep_ica_data = prep_data.copy()
                ica.apply(prep_ica_data, exclude=exclude_index) #  no need:  n_pca_components=best_n_components
                
                # # compare ica cleaned and before
                # prep_data.plot()
                # prep_ica_data.plot()
                
                ###  Plot the overlay of raw vs. cleaned data for inspection
                overlay_fig = ica.plot_overlay(prep_data, exclude=exclude_index, picks='eeg', show=False)  # Optional: Pick specific channels
                plt.close(overlay_fig) 
                # ica.plot_overlay(prep_data)
                # ica.plot_properties(prep_data, picks=[4])  # visualize a randomly selected component
                
                # %% ## Prepare the CSVto Save 
                
                # participant id ..
                participant_id = f"BTSCZ{participant_no}" if "BTSCZ" in filename else f"BHC{participant_no}"
                
                # Create a dictionary representing the new row
                new_row = pd.Series({'subject_id': participant_id, 
                                    'session': session, 
                                    'task': task,
                                    'total_explained_var_ratio': explained_var_ratio['eeg'],
                                    'cont_ICA_labels': component_labels,
                                    'cont_ICA_percentage': predicted_probabilities, 
                                    'cont_ICA_heart_found': nr_heart_components,
                                    'cont_ICA_blink_found': nr_blink_components,
                                    'cont_ICA_muscle_found': nr_muscle_components,
                                    'bads_ecg_indices': ecg_indices, 
                                    'bads_ecg_scores_corr': ecg_scores,
                                    'n_bads_ecg': n_bads_ecg,
                                    'num_components_total': ica.n_components_, # best_n_components,
                                    'num_components_excluded': len(exclude_index),
                                    'rejection_criteria': 'iclabel =! brain or other (with predicted_probabilities > .50)'
                                    })
            
                # convert row to df
                new_row =  new_row.to_frame().T
                # add to existing df the current data outputs
                ica_raw_outputs = pd.concat([ica_raw_outputs, new_row], ignore_index=True)
                # Print the DataFrame
                print(ica_raw_outputs)
                
                
                # SAVE the Preprocessed Data & Prep Ouputs in a CSV
                folder_name = 'Preprocessed_ICA_applied_on_raw' if "BTSCZ" in filename else 'HC_Preprocessed_ICA_applied_on_raw'
                #processing_subfolder_name = f"{processing_style_sampling}_{processing_style_bandpass}_{processing_style_line}_{processing_style_interp}" # this sequence also tells in which order the processing steps were applied

                output_folder_path = os.path.join(os.getcwd(), folder_name)
                print(f"Creating directory: {output_folder_path}")
                os.makedirs(output_folder_path, exist_ok=True)
                
                ### Save the Data 
                
                # IF u need to save data again uncomment below!
                
                # Save the data, yay!
                ica_file_name = filename.replace('_until_', '_')
                file_path = os.path.join(output_folder_path, ica_file_name)
                prep_ica_data.save(file_path, overwrite=True)
                
                
                ### CSV including all subs : can be in or out of the loop
                
                # Construct the csv path
                current_date = datetime.datetime.now().strftime("%Y%m%d")
                csv_filename = f'ica_output_{current_date}.csv' if "BTSCZ" in filename else f'HC_ica_output_{current_date}.csv'
                csv_path = os.path.join(output_folder_path, csv_filename)
                ica_raw_outputs.to_csv(csv_path, mode='w', sep=',', index=False)

                ###  SAVE ICA IMAGES

                # Define base folder for saving images
                images_base_path = os.path.join(output_folder_path, 'ICA_images')

                # Participant-specific folder
                participant_folder = os.path.join(images_base_path, participant_id)
                os.makedirs(participant_folder, exist_ok=True)  # Create folder if it doesn't exist

                # Session-specific folder within the participant folder
                participant_session_folder = os.path.join(participant_folder, f"ses-{session}") if "BTSCZ" in filename else participant_folder
                os.makedirs(participant_session_folder, exist_ok=True)  # Create session folder if it doesn't exist

                # Task-specific folder within the participant folder
                participant_session_task_folder = os.path.join(participant_session_folder, f"task-{task}") 
                os.makedirs(participant_session_task_folder, exist_ok=True)  # Create session folder if it doesn't exist

                # Save each component's figure
                if isinstance(topo_figs, list):  # If it returns a list of Matplotlib figures
                    for idx, fig in enumerate(topo_figs):
                        topo_path = os.path.join(participant_session_task_folder, f"{participant_id}_{f'ses-{session}_' if 'BTSCZ' in filename else ''}{task}_ica_topography_component_{idx}.png") 
                        fig.savefig(topo_path, dpi=300)
                        plt.close(fig)
                elif isinstance(topo_figs, plt.Figure):  # If it returns a single MNEFigure object
                    topo_path = os.path.join(participant_session_task_folder, f"{participant_id}_{f'ses-{session}_' if 'BTSCZ' in filename else ''}{task}_ica_topographies.png")
                    topo_figs.savefig(topo_path, dpi=300)  # Save the figure
                    plt.close(topo_figs)  # Close the figure to release memory

                # Save overlay plot
                overlay_path = os.path.join(participant_session_task_folder, f"{participant_id}_{f'ses-{session}_' if 'BTSCZ' in filename else ''}{task}_prep-only_vs_ica-cleaned_overlay.png")
                overlay_fig.savefig(overlay_path, dpi=300)
                plt.close(overlay_fig)  # Close figure to free memory

                # Save properties of specific components
                for i in range(ica.n_components_):  # Loop through all ICA components
                    prop_figs = ica.plot_properties(prep_data, picks=[i], show=False)  # Returns a list of figures
                    for fig in prop_figs:  # Iterate through the list of figures
                        prop_path = os.path.join(participant_session_task_folder, f"{participant_id}_{f'ses-{session}_' if 'BTSCZ' in filename else ''}{task}_ica_component_{i}_properties.png")
                        fig.savefig(prop_path, dpi=300)  # Save each figure
                        plt.close(fig)  # Close figure to free memory
                            # Define the particpant folder path




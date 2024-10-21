###############################################################################
##                     EYE DATA: EXTRACTION                                   ##
###############################################################################
# by Camille Fakche 07/08/2024

import numpy as np

dirEyeData = 'myeyedatapath'
dirImageInfo = 'myimageinfopath'
dirBehavData = 'mybehaviouraldatapath'

# EDF files have been converted to ASCII file with edfconverter available at https://www.sr-research.com/
# There are one EDF file per block. 
ListBlock = ['1','2','3','4','5','6','7','8','9','10'] 
nBlock = len(ListBlock)
nImages = 7

# Example One Participant
suj = 'B3EB'

# %% Load Behavioural Data 

BehavData_filename = dirBehavData+suj+'_BehavData_matrix.npy'
BehavData_matrix_allTrials = np.load(BehavData_filename, allow_pickle=True)
# One Row = One Trial
# Column 0: Target Image Position (1:7)
# Column 1: Target Image Position Response (1:7)
# Column 2: Correct (1) or Incorrect (0) Response
# Column 3: Reaction Times (s)
# Column 4: ITI (s)
# Column 5: Greyscale (0) or Colorscale (1) Target Image
# Column 6: Animal (1), Food (2), Object (3) Target Image
# Column 7: Target Image Number (1:1500)

# Extract index Correct/Incorrect trials
indexTrialsCorrIncorr = BehavData_matrix_allTrials[:,2]

# %% Eye Data Extraction from ASCII file 

trial_counter = 0
for block in range(nBlock):

    # Load Image Infos Data 
    ImagesInfos_filename = dirImageInfo+suj+'_ImageInfos_allTrials_block'+ListBlock[block]+'.npy'
    ImagesInfos_allTrials = np.load(ImagesInfos_filename, allow_pickle=True)
    # One Row = One Trial = One Object
    # Each Object:
    # One Row = One Image
    # Column 0: X Position Start
    # Column 1: Y Position Start
    # Column 2: X Position End
    # Column 3: Y Position End
    # Column 4: Animal (1), Food (2) or Object (3)
    # Column 5: Image Number (1:1500)
    # Column 6: Greyscale (0) or Colorscale (1)
    # Column 7: Number of Target Image 
    
    
    # Load ASCII file
    edf_file_name = dirEyeData+suj+'b'+ListBlock[block]+'.asc' # One ASCII file per block
    edf_file = open(edf_file_name, 'r')
    
    trial = 0
    trial_counter_per_block = 0
    
    # Read the ASCII file line by line
    for line in edf_file:
        
        # Initialize one trial
        if 'TRIAL_START' in line:
            trial = 1
            info_start = line.split()
            start_time = float(info_start[1]) # Define the starting time of the trial 
            AllEyeEvent = np.zeros([1,11]) # Initialize AllEyeEvent array - Reference all the eye events in one trial 
            AllEyeEvent[:] = np.nan
            AllEyeEvent[0,0] = 0
            AllEyeEvent[0,1] = float(info_start[1])
            AddEyeEventLine = np.zeros([1,11])
            AddEyeEventLine[:] = np.nan
            event_counter = 1
            ImagesInfos_Trial = ImagesInfos_allTrials[trial_counter_per_block] # Define the images displayed in the trial
        
        # Complete one trial 
        if 'ITI' in line:
            trial = 0
            # Note events interrupt by a blink as bad events (Event = 0)
            for ev in range(len(AllEyeEvent)-2):
                if AllEyeEvent[ev,0] == 3:
                    if AllEyeEvent[ev-1,0] < 11111:
                        AllEyeEvent[ev-1,0] = 0
                    else:
                        AllEyeEvent[ev-2,0] = 0
                if AllEyeEvent[ev,0] == 30:
                    if AllEyeEvent[ev+1,0] < 11111:
                        AllEyeEvent[ev+1,0] = 0
                    else:
                        AllEyeEvent[ev+2,0] = 0
            
            AllEyeEvent_filename = dirEyeData+suj+'_AllEyeEvent_trial'+str(trial_counter)+'.npy' # Save AllEyeEvent array
            np.save(AllEyeEvent_filename,AllEyeEvent)
            trial_counter += 1
            trial_counter_per_block += 1
            
        if trial == 1:
            
            if 'ENCODING' in line:
                info_encoding = line.split()
                encoding_time = float(info_encoding[1]) - start_time # Define the starting time according to trial starting time
                AllEyeEvent = np.append(AllEyeEvent,AddEyeEventLine,axis=0)
                AllEyeEvent[event_counter,0] = 11111
                AllEyeEvent[event_counter,1] = encoding_time
                AllEyeEvent[event_counter,10] = indexTrialsCorrIncorr[trial_counter]
                event_counter += 1
            if 'MASK' in line:
                info_mask = line.split()
                mask_time = float(info_mask[1]) - start_time # Define the starting time according to trial starting time
                AllEyeEvent = np.append(AllEyeEvent,AddEyeEventLine,axis=0)
                AllEyeEvent[event_counter,0] = 22222
                AllEyeEvent[event_counter,1] = mask_time
                AllEyeEvent[event_counter,10] = indexTrialsCorrIncorr[trial_counter]
                event_counter += 1
            if 'TASK' in line:
                info_task = line.split()
                task_time = float(info_task[1]) - start_time # Define the starting time according to trial starting time
                AllEyeEvent = np.append(AllEyeEvent,AddEyeEventLine,axis=0)
                AllEyeEvent[event_counter,0] = 33333
                AllEyeEvent[event_counter,1] = task_time
                AllEyeEvent[event_counter,10] = indexTrialsCorrIncorr[trial_counter]
                event_counter += 1
                
                
            if 'SFIX' in line:
                info_startfix = line.split()
                startfix_time = float(info_startfix[2]) - start_time # Define the starting time according to trial starting time
                AllEyeEvent = np.append(AllEyeEvent,AddEyeEventLine,axis=0)
                AllEyeEvent[event_counter,0] = 1
                AllEyeEvent[event_counter,1] = startfix_time
                AllEyeEvent[event_counter,10] = indexTrialsCorrIncorr[trial_counter]
                event_counter += 1
            if 'SSACC' in line:
                info_startsac = line.split()
                startsac_time = float(info_startsac[2]) - start_time # Define the starting time according to trial starting time
                AllEyeEvent = np.append(AllEyeEvent,AddEyeEventLine,axis=0)
                AllEyeEvent[event_counter,0] = 2
                AllEyeEvent[event_counter,1] = startsac_time
                AllEyeEvent[event_counter,10] = indexTrialsCorrIncorr[trial_counter]
                event_counter += 1
            if 'SBLINK' in line:
                info_startblink = line.split()
                startblink_time = float(info_startblink[2]) - start_time # Define the starting time according to trial starting time
                AllEyeEvent = np.append(AllEyeEvent,AddEyeEventLine,axis=0)
                AllEyeEvent[event_counter,0] = 3
                AllEyeEvent[event_counter,1] = startblink_time
                AllEyeEvent[event_counter,10] = indexTrialsCorrIncorr[trial_counter]
                event_counter += 1
                
                
            if 'EFIX' in line:
                info_endfix = line.split()
                endfixstart_time = float(info_endfix[2]) - start_time # Define the starting time according to trial starting time
                endfix_time = float(info_endfix[3]) - start_time
                AllEyeEvent = np.append(AllEyeEvent,AddEyeEventLine,axis=0)
                AllEyeEvent[event_counter,0] = 10
                AllEyeEvent[event_counter,1] = endfixstart_time
                AllEyeEvent[event_counter,2] = endfix_time 
                AllEyeEvent[event_counter,3] = float(info_endfix[4])/1000 # Duration - Convert to second 
                AllEyeEvent[event_counter,4] = float(info_endfix[5]) # X pos
                AllEyeEvent[event_counter,5] = float(info_endfix[6]) # Y pos
                # Find on which image is the fixation
                for im in range(nImages):
                    xStart = ImagesInfos_Trial[im,0]
                    yStart = ImagesInfos_Trial[im,1]
                    xEnd = ImagesInfos_Trial[im,2]
                    yEnd = ImagesInfos_Trial[im,3]
                    if (float(info_endfix[5]) >= xStart) and (float(info_endfix[5]) <= xEnd) and (float(info_endfix[6]) >= yStart) and (float(info_endfix[6]) <= yEnd):
                        AllEyeEvent[event_counter,6] = ImagesInfos_Trial[im,4] # Image Category
                        AllEyeEvent[event_counter,7] = im+1 # Image Position
                        AllEyeEvent[event_counter,8] = ImagesInfos_Trial[im,6] # Image GreyScale (0) vs Color (1) 
                AllEyeEvent[event_counter,10] = indexTrialsCorrIncorr[trial_counter]
                event_counter += 1
                
            if 'ESACC' in line:
                info_endsac = line.split()
                endsacstart_time = float(info_endsac[2]) - start_time # Define the starting time according to trial starting time
                endsac_time = float(info_endsac[3]) - start_time
                AllEyeEvent = np.append(AllEyeEvent,AddEyeEventLine,axis=0)
                AllEyeEvent[event_counter,0] = 20
                AllEyeEvent[event_counter,1] = endsacstart_time
                AllEyeEvent[event_counter,2] = endsac_time
                AllEyeEvent[event_counter,3] = float(info_endsac[4])/1000 # Duration - Convert to second 
                AllEyeEvent[event_counter,4] = float(info_endsac[5]) # X pos start
                AllEyeEvent[event_counter,5] = float(info_endsac[6]) # Y pos start
                AllEyeEvent[event_counter,6] = float(info_endsac[7]) # X pos end
                AllEyeEvent[event_counter,7] = float(info_endsac[8]) # Y pos end 
                AllEyeEvent[event_counter,8] = float(info_endsac[9]) # Amplitude
                AllEyeEvent[event_counter,9] = float(info_endsac[10]) # Speed
                AllEyeEvent[event_counter,10] = indexTrialsCorrIncorr[trial_counter]
                event_counter += 1
                
            if 'EBLINK' in line:
                info_endblink = line.split()
                endblinkstart_time = float(info_endblink[2]) - start_time # Define the starting time according to trial starting time
                endblink_time = float(info_endblink[3]) - start_time
                AllEyeEvent = np.append(AllEyeEvent,AddEyeEventLine,axis=0)
                AllEyeEvent[event_counter,0] = 30
                AllEyeEvent[event_counter,1] = endblinkstart_time
                AllEyeEvent[event_counter,2] = endblink_time
                AllEyeEvent[event_counter,3] = float(info_endblink[4])/1000 # Duration - Convert to second 
                AllEyeEvent[event_counter,10] = indexTrialsCorrIncorr[trial_counter]
                event_counter += 1
                
# %% Have a look to the structure of the extracted data 

AllEyeEvent_filename = dirEyeData+suj+'_AllEyeEvent_trial0.npy'
AllEyeEvent = np.load(AllEyeEvent_filename, allow_pickle=True) 

# One row = One Eye Event
# Column 0: Event Type
# TRIAL ONSET (0), VISUAL SEARCH (11111), MASK (22222), TASK (33333)
# Fixation Start (1) End (10), Saccade Start (2) End (20), Blink Start (3) End (30)
# Column 1: Event Start Time  (frame from trial onset)
# Column 2: Event End Time (frame from trial onset) - for Event Type (10) (20) (30)
# Column 3: Event Duration (s) - for Event Type (10) (20) (30)

# If Eye Event = (10) Fixation End
# Column 4: Eye X position 
# Column 5: Eye Y position 
# Column 6: Image Category Animal (1) Food (2) Object (3) of the image on which the eye is - nan if background 
# Column 7: Image Position (1 to 7) of the image on which the eye is - nan if background 
# Column 8: Image Greyscale (0) vs Colourscale (1) - of the image on which the eye is - nan if background 

# If Eye Event = (20) Saccade End
# Column 4: Eye X position at Saccade Start
# Column 5: Eye Y position at Saccade Start
# Column 6: Eye X position at Saccade End
# Column 7: Eye Y position at Saccade End
# Column 8: Amplitude (deg)
# Column 9: Peak Velocity (deg/s)

# Column 10: Correct (1) or Incorrect (0) trial


##############################################################################
##                     MEG DATA ANALYSIS: EPOCHING                           ##
###############################################################################
# by Camille Fakche 07/08/2024

import numpy as np
import mne

dirMEGData = 'myMEGdatapath'
dirEyeData = 'myEyedatapath'
dirRejectTrialData = 'myRejectTrialdatapath'

ListFov = ['Foveal','ParaFoveal','Previous','Remaining']
# Select Foveal Condition
fov = 0

# Example One Participant
suj = 'B3EB'

nTrials = 300

#%% 

# Load MEG data
DataSujFile = dirMEGData+suj+'/'+suj+'_Data_tSSS_AA_ICA_RC.fif'
Data = mne.io.read_raw(DataSujFile, allow_maxshield=True, preload=True, verbose=True)

# Find events
events = mne.find_events(Data, stim_channel='STI101', min_duration=0.001001,consecutive=(True))

# Select trials onset
index_trials_onset = np.array(np.where(events[:,2]==2))
TrialsOnsetEvents = np.squeeze(events[index_trials_onset,:])

# Load Trial to Reject because Trigger Issue
RejectTrialFilename = dirRejectTrialData+suj+'_RejectTrialEpoching.npy'
index_RejectTrial = np.load(RejectTrialFilename, allow_pickle=True)

# Create new events based on EyeLink data
events_FixOnsetEnc = np.zeros([1, 3], dtype=int)
trial_counter = 0
event_counter = 0
for trial in range(nTrials):
    if  np.isin(trial,index_RejectTrial):
        print('Trial Rejected')
    else:
        # Load data from EyeLink
        FixationsEpoch_filename = dirEyeData+suj+'_'+ListFov[fov]+'_FixationsOnsetEpoch_trial'+str(trial)+'.npy'
        FixationsEpoch = np.load(FixationsEpoch_filename, allow_pickle=True) 
        # Trial onset info
        add_trial_event = np.zeros([1, 3], dtype=int)
        add_trial_event[0,0] = TrialsOnsetEvents[trial_counter,0]
        add_trial_event[0,2] = TrialsOnsetEvents[trial_counter,2]
        if trial == 0:
            events_FixOnsetEnc = add_trial_event
        else:  
            events_FixOnsetEnc = np.append(events_FixOnsetEnc, add_trial_event, axis=0)
        # Add Fixation Onset 
        if FixationsEpoch.size != 0:
            nevents_trial = len(FixationsEpoch)
            for ev in range(nevents_trial):
                # Compute new time according to trial onset
                trial_onset_time = TrialsOnsetEvents[trial_counter,0]
                add_event = np.zeros([1,3], dtype=int)
                add_event[0,0] = FixationsEpoch[ev,2]+trial_onset_time
                # Add trigger value
                add_event[0,2] = FixationsEpoch[ev,0]
                events_FixOnsetEnc = np.append(events_FixOnsetEnc, add_event, axis=0)
        trial_counter += 1
        
mapping = {2: 'StartTrial', 
               11: 'AnimalColor',
               10: 'AnimalGrey',
               21: 'FoodColor',
               20: 'FoodGrey', 
               31: 'ObjectColor',
               30: 'ObjectGrey'}
    
annot_from_events = mne.annotations_from_events(
            events=events_FixOnsetEnc, event_desc=mapping, sfreq=Data.info['sfreq'],
            orig_time=Data.info['meas_date'])
Data.set_annotations(annot_from_events)   

# Epoching 
reject = dict(grad=5000e-13,    # T / m (gradiometers)
              mag=5e-12,        # T (magnetometers)
              )              

for cond in range(5):
    if cond == 0:
        epoch_dict = {'AnimalColor': 11, 'AnimalGrey': 10}
        cond_name = 'Animal'
    elif cond == 1:
        epoch_dict = {'FoodColor': 21, 'FoodGrey': 20}     
        cond_name = 'Food'
    elif cond == 2:
        epoch_dict = {'ObjectColor': 31, 'ObjectGrey': 30}
        cond_name = 'Object'
    elif cond == 3:
        epoch_dict = {'AnimalColor': 11, 'FoodColor': 21, 'ObjectColor': 31}
        cond_name = 'Color'
    elif cond == 4:
        epoch_dict = {'AnimalGrey': 10, 'FoodGrey': 20, 'ObjectGrey': 30}
        cond_name = 'Grey'
             
    Epochs = mne.Epochs(Data,
            events_FixOnsetEnc, event_id=epoch_dict,
            tmin=-1, tmax=1,
            baseline=None,
            proj=True,
            picks = 'all',
            detrend = 1,
            reject=reject,
            reject_by_annotation=True,
            preload=True,
            verbose=True) 
    
    Epochs.resample(500)

    # Save 
    EpochFile = dirMEGData+suj+'/'+suj+'_Epochs_'+ListFov[fov]+'_FixationOnset_'+cond_name+'.fif'
    Epochs.save(EpochFile, overwrite=True) 


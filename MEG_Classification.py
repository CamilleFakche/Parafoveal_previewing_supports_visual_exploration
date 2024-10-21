##############################################################################
##                     MEG DATA ANALYSIS: CLASSIFICATION                   ##
###############################################################################
# by Camille Fakche 07/08/2024

import mne
import numpy as np
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, LinearModel,
                          Vectorizer, cross_val_multiscore)
import sklearn.svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

dirMEGData = 'myMEGdatapath'
dirScript = 'myscriptpath'

os.chdir(dirScript)
import MEG_Functions

ListFov = ['Foveal','ParaFoveal','Previous','Remaining']
# Select Foveal Condition
fov = 0

# Select Temporal Generalization
tempgen = 0 

# Example One Participant
suj = 'B3EB'

# Example One Classif
condname1 = 'Color'
condname2 = 'Grey' 
filename = condname1 + '_vs_' + condname2

#%% Step1: Load and Prepare Epoch Data

# Load Data
Epoch1File = dirMEGData+suj+'/'+suj+'_Epochs_'+ListFov[fov]+'_FixationOnset_'+condname1+'.fif'
Epoch1 = mne.read_epochs(Epoch1File, preload=True, verbose=True)
Epoch2File = dirMEGData+suj+'/'+suj+'_Epochs_'+ListFov[fov]+'_FixationOnset_'+condname2+'.fif'
Epoch2 = mne.read_epochs(Epoch2File, preload=True, verbose=True)

# Prepare Data for Classification: Select Times + MEG sensors
tmin = -0.5
tmax = 0.5
Epoch1Class, times = MEG_Functions.MEG_Prepare_Data_Classification(Epoch1, tmin, tmax)
Epoch2Class, times = MEG_Functions.MEG_Prepare_Data_Classification(Epoch2, tmin, tmax)

# Save times
times_Filename = dirMEGData+'times_'+ListFov[fov]+'.npy'
np.save(times_Filename, times)

#%% Step2: Delay-Embedding

Epoch1Embedded = MEG_Functions.MEG_DelayEmbedding(Epoch1Class)
Epoch2Embedded = MEG_Functions.MEG_DelayEmbedding(Epoch2Class)

#%% Step3: Classification

# %% Classifier settings 

clf = make_pipeline(Vectorizer(),StandardScaler(),  
               LinearModel(sklearn.svm.SVC(kernel = 'linear'))) 

if tempgen == 0:     
    time_decod = SlidingEstimator(clf, n_jobs=-1, scoring='roc_auc', verbose=True)
    
elif tempgen == 1:
    time_gen = GeneralizingEstimator(clf, n_jobs=-1, scoring="roc_auc", verbose=True)

# %% Initialize scores array

nTimes = len(times)
nChans = int(Epoch1Embedded.shape[1])
nRepet = 10
if tempgen == 0:
     scoresall = np.zeros((nRepet, nTimes))
if tempgen == 1:
     scoresall = np.zeros((nRepet, nTimes, nTimes))
scoresall[:] = np.nan

# %% Classification with Supertrials

nEpoch1 = len(Epoch1Embedded)
nEpoch2 = len(Epoch2Embedded) 
print(nEpoch1)
print(nEpoch2)

# nSupertrials = 10 
# But reduce if not enough trials 
if 40 < nEpoch1 < 50 or 40 < nEpoch2 < 50:
    nFewTrials = 5
elif 10 < nEpoch1 < 40 or 10 < nEpoch2 < 40:
    nFewTrials = 2
elif nEpoch1 < 10 or nEpoch2 < 10:
    nFewTrials = 1
else:
    nFewTrials = 10   
print(nFewTrials)

# nCrossValidation = 10
# But reduce if not enough trials 
if nEpoch1 < 100 and nEpoch1 <= nEpoch2:
    cv = int(np.floor(nEpoch1/nFewTrials))
elif nEpoch2 < 100 and nEpoch2 < nEpoch1:
    cv = int(np.floor(nEpoch2/nFewTrials))
else:
    cv = 10
print('cv='+str(cv))

nRepet = 10
for i_rep in range(nRepet):
    
    # Subsampling part 
    # if nEpoch1 > nEpoch2:
    #     Epoch2sub = Epoch2Embedded
    #     # Shuffle Trials List
    #     TrialsListEpoch1sub = np.arange(0,nEpoch1,1)
    #     np.random.shuffle(TrialsListEpoch1sub)
    #     # Select nEpoch2 trials
    #     TrialsListEpoch1sub = TrialsListEpoch1sub[0:nEpoch2]
    #     # Subtrials selection
    #     Epoch1sub = Epoch1Embedded[TrialsListEpoch1sub,:,:]
    # elif nEpoch1 < nEpoch2:
    #     Epoch1sub = Epoch1Embedded
    #     # Shuffle Trials List
    #     TrialsListEpoch2sub = np.arange(0,nEpoch2,1)
    #     np.random.shuffle(TrialsListEpoch2sub)
    #     # Select nEpoch1 trials
    #     TrialsListEpoch2sub = TrialsListEpoch2sub[0:nEpoch1]
    #     # Subtrials selection
    #     Epoch2sub = Epoch2Embedded[TrialsListEpoch2sub,:,:]
    # elif nEpoch1 == nEpoch2:
    #     Epoch1sub = Epoch1Embedded
    #     Epoch2sub = Epoch2Embedded     
    # nEpoch1sub = len(Epoch1sub)
    # nEpoch2sub = len(Epoch2sub)
    
    # Epoch1
    # Shuffle Trials List
    TrialsListEpoch1 = np.arange(0,nEpoch1,1)
    np.random.shuffle(TrialsListEpoch1)
    # Split Trials List
    nChunksEpoch1 = int(np.floor(nEpoch1/nFewTrials))
    nTrialsEpoch1 = int(nChunksEpoch1*nFewTrials)
    TrialsListEpoch1 = TrialsListEpoch1[0:nTrialsEpoch1]
    TrialsListEpoch1_Split = np.split(TrialsListEpoch1,nChunksEpoch1)
    # Average to create Supertrials
    epoch1_allsupertrial = np.zeros((nChunksEpoch1, nChans, nTimes))
    epoch1_allsupertrial[:] = np.nan
    for new_trial_epoch1 in range(nChunksEpoch1):
        # Average
        supertrial_index_epoch1 = TrialsListEpoch1_Split[new_trial_epoch1]
        supertrial_epoch1 = np.mean(Epoch1Embedded[supertrial_index_epoch1],axis=0)
        epoch1_allsupertrial[new_trial_epoch1,:,:] = supertrial_epoch1
        
    # Epoch2
    # Shuffle Trials List
    TrialsListEpoch2 = np.arange(0,nEpoch2,1)
    np.random.shuffle(TrialsListEpoch2)
    # Split Trials List
    nChunksEpoch2 = int(np.floor(nEpoch2/nFewTrials))
    nTrialsEpoch2 = int(nChunksEpoch2*nFewTrials)
    TrialsListEpoch2 = TrialsListEpoch2[0:nTrialsEpoch2]
    TrialsListEpoch2_Split = np.split(TrialsListEpoch2,nChunksEpoch2)
    # Average to create Supertrials
    epoch2_allsupertrial = np.zeros((nChunksEpoch2, nChans, nTimes))
    epoch2_allsupertrial[:] = np.nan
    for new_trial_epoch2 in range(nChunksEpoch2):
        # Average
        supertrial_index_epoch2 = TrialsListEpoch2_Split[new_trial_epoch2]
        supertrial_epoch2 = np.mean(Epoch2Embedded[supertrial_index_epoch2],axis=0)
        epoch2_allsupertrial[new_trial_epoch2,:,:] = supertrial_epoch2
    
    # Matrix X (features) is a three-dimensional matrix (trials x channel x time points)
    X = np.vstack((epoch1_allsupertrial, epoch2_allsupertrial))
    # Vector Y (targets)
    nEpoch1rep = len(epoch1_allsupertrial)
    nEpoch2rep = len(epoch2_allsupertrial)
    iEpoch1rep = np.ones([1,nEpoch1rep])
    iEpoch2rep = 2*np.ones([1,nEpoch2rep])
    Y = np.append(iEpoch1rep, iEpoch2rep)
    
    if tempgen == 0:
        # Fit and test on the same dataset
        scoresrep = cross_val_multiscore(time_decod, X, Y, cv=cv, n_jobs=-1)
        scoresrep = np.mean(scoresrep, axis=0)
        # Indent scores all
        scoresall[i_rep,:] = scoresrep
        
    elif tempgen == 1:
        # Fit and test on the same dataset
        scoresrep = cross_val_multiscore(time_gen, X, Y, cv=cv, n_jobs=-1)
        # Mean scores across cross-validation splits
        scoresrep = np.mean(scoresrep, axis=0)
        # Indent scores all
        scoresall[i_rep,:, :] = scoresrep


# Average 
scores = np.mean(scoresall, axis=0)

# Save
if tempgen == 0:
    Classif_Filename = dirMEGData+suj+'/'+suj+'_Classification_'+ListFov[fov]+'_FixationOnset_'+filename+'.npy'
elif tempgen == 1:
    Classif_Filename = dirMEGData+suj+'/'+suj+'_ClassificationTempGen_'+ListFov[fov]+'_FixationOnset_'+filename+'.npy'
np.save(Classif_Filename, scores)



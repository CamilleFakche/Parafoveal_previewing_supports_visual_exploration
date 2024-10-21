The codes provided here allow to perform the main analyses (Figures 3, 4, and 5) from Fakche C., Hickey C., and Jensen O., (2024), Fast feature- and category-related parafoveal previewing support free visual exploration, The Journal of Neuroscience,  with Python 3.10.9. and MNE 1.0.3..

RAW DATA
Note that one participant has only 9 blocks and 270 trials. 

BEHAVIORAL DATA
There is one file per participant, ParticipantCode_BehavData_matrix.npy.

EYE DATA
There are 20 files per participant, one file per block (10 blocks) in EDF and in ASCII format, ParticipantCodebX.

IMAGES INFOS DATA
There are 10 files per participant, one file per block, ParticipantCode_ImageInfos_allTrials_blockX.npy.

MEG DATA
There is one folder per participant, with 3 files in FIF format, ParticipantCode.fif, -1.fif, -2.fif. 
RejectTrial folder: There is one file per participant, ParticipantCode_RejectTrialEpoching.npy, with the trial indexes to be rejected
because of trigger issues during the recording. 

CODES

- EyeData_Extraction.py
This code allows to transform the ASCII file into a numpy array with all eye events. 

- EyeData_Epoch_Selection_MEG.py
This code allows to create a new stucture of MEG events, based on Eye Data, that will be used for Epoching. 

- MEG_Preprocessing.py
# Step0: Compute Continuous Head Position 
# Step1: Low-Pass Filter 150 Hz, MaxFilter, tSSS, Movement compensation
# Step2: Artifact Annotations
# Step3: ICA

- MEG_Epoching.py
This code allows to compute Epoch [-1; +1]s according to Foveal Fixation Onset, with the structure of MEG events created previously. 

- MEG_Classification.py
This code allows to perform MVPA. 
# Step1: Select all sensors + Crop data [-0.5; +0.5]s
# Step2: Delay-Embedding
# Step3: Classification / Temporal Generalization with Supertrials 
(Linear Support Vector Machine, 10-fold cross-validation, Supertrial = 10 trials, 10 repetitions, Area Under the Curve, AUC)

- MEG_Statistics_Figures.py
This code allows to perform Figures 3, 4, and 5, with associated statistics.
# Classification: Figure + Permutation across times
# Temporal Generalization: Figure + Permutation across times*times
# Foveal vs Parafoveal: Peak Latency Difference + Figure + T-test 
(Statistics Details reported in the Publication)


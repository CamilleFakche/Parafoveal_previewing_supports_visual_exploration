###############################################################################
##                     MEG DATA ANALYSIS: PREPROCESSING                      ##
###############################################################################
# by Camille Fakche 07/08/2024

import mne
import matplotlib.pyplot as plt

dirRawData = 'myrawdatapath'
dirData = 'mydatapath'

# Example One Participant
suj = 'B3EB'

# %% Step0: Compute Head Position

# Load raw data
RawDataSujFile = dirRawData+suj+'/'+suj+'.fif'
Data = mne.io.read_raw(RawDataSujFile, allow_maxshield=True, preload=True, verbose=True)

# cHPI frequencies
chpi_freqs, ch_idx, chpi_codes = mne.chpi.get_chpi_info(info=Data.info)
print(f'cHPI coil frequencies extracted from raw: {chpi_freqs} Hz')
# cHPI amplitudes
chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(Data)
# Compute time-varying HPI coil locations from these amplitudes 
chpi_locs = mne.chpi.compute_chpi_locs(Data.info, chpi_amplitudes)
# Compute head position 
head_pos = mne.chpi.compute_head_pos(Data.info, chpi_locs, verbose=True)
# Plot
mne.viz.plot_head_positions(head_pos, mode='traces')
mne.viz.plot_head_positions(head_pos, mode='field')
# Save head positions
HeadPos_File = dirData+suj+'/'+suj+'_HeadPosition.fif'
mne.chpi.write_head_pos(HeadPos_File, head_pos)

# %% Step1: tSSS

CrosstalkFile = 'ct_sparse_triux2.fif'
CalibrationFile = 'sss_cal_3140_60_190213.dat'

# Load raw data
RawDataSujFile = dirRawData+suj+'/'+suj+'.fif'
Data = mne.io.read_raw(RawDataSujFile, allow_maxshield=True, preload=True, verbose=True)

# Load head position
HeadPos_File = dirData+suj+'/'+suj+'_HeadPosition.fif'
HeadPos = mne.chpi.read_head_pos(HeadPos_File)

# Identify faulty sensors automatically
Data.info['bads'] = []
Data_Check = Data.copy()
auto_noisy_chs, auto_flat_chs, auto_scores = mne.preprocessing.find_bad_channels_maxwell(
        Data_Check, 
        cross_talk=CrosstalkFile, 
        calibration=CalibrationFile,
        return_scores=True, 
        verbose=True)
print('noisy =', auto_noisy_chs)
print('flat = ', auto_flat_chs)
Data.info['bads'].extend(auto_noisy_chs + auto_flat_chs)
print('bads = ', Data.info['bads'])
# Change MEGIN magnetometer coil types to ensure compatibility across systems
Data.fix_mag_coil_types()

# Low pass filter at 150 Hz
Data = Data.filter(None,150)

# Apply MaxFilter, tSSS, Calibration and CrossTalk reduction
# With movement compensation
Data_SSS = mne.preprocessing.maxwell_filter(
         Data,
         cross_talk=CrosstalkFile,
         calibration=CalibrationFile,
         verbose=True,
         head_pos=HeadPos,
         st_duration=6)
SSSFile = dirData+suj+'/'+suj+'_Data_tSSS.fif'

# %% Step2: Artifact Annotations

from mne.preprocessing import annotate_muscle_zscore

# Load data
DataSujFile = dirData+suj+'/'+suj+'_Data_tSSS.fif'
Data = mne.io.read_raw(DataSujFile, allow_maxshield=True, preload=True, verbose=True)

# Find muscle artefacts
threshold_muscle = 10  
annotations_muscle, scores_muscle = annotate_muscle_zscore(
    Data, ch_type="mag", threshold=threshold_muscle, min_length_good=0.2,
    filter_freq=[110, 140])

# Check that the threshold used for the muscle annotation is appropriate 
fig1, ax = plt.subplots()
ax.plot(Data.times, scores_muscle)
ax.axhline(y=threshold_muscle, color='r')
ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity (threshold = %s)' % threshold_muscle)

# Include annotations in the dataset
Data.set_annotations(annotations_muscle)
# Save the new dataset
SSS_AA_File = dirData+suj+'/'+suj+'_Data_tSSS_AA.fif'
Data.save(SSS_AA_File)

# %% Step3: ICA

from mne.preprocessing import ICA

# Load data
DataSujFile = dirData+suj+'/'+suj+'_Data_tSSS_AA.fif'
Data = mne.io.read_raw(DataSujFile, allow_maxshield=True, preload=True, verbose=True)

# Downsample and filter the data to improve ICA 
Data_Downsample = Data.copy().pick_types(meg=True)
Data_Downsample.resample(200) # dowsample to 200 Hz
Data_Downsample.filter(1, 40) # band-pass filter from 1 to 40 Hz

# Perform ICA
ica = ICA(method='fastica', random_state=97, n_components=30, verbose=True)
ica.fit(Data_Downsample, verbose=True)

# Plot ICA timecourse
ica.plot_sources(Data_Downsample,title='ICA')
# Plot ICA topography
ica.plot_components()

# Set the components to exclude
ica.exclude = [0, 1, 2] # Selected from visual inspection

# Apply ICA to data 
Data_ICA = Data.copy()
ica.apply(Data_ICA)

# Save
Data_SSS_AA_ICA_RC_File = dirData+suj+'/'+suj+'_Data_tSSS_AA_ICA_RC.fif'
Data_ICA.save(Data_SSS_AA_ICA_RC_File, overwrite=True)
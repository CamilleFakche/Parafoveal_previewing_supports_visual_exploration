##############################################################################
##                     MEG DATA ANALYSIS: STATISTICS & FIGURES             ##
###############################################################################
# by Camille Fakche 07/08/2024

import numpy as np 
import os 
import math
import matplotlib.colors as mcolors
colors = mcolors.CSS4_COLORS
import matplotlib.pyplot as plt
import pingouin as pg

dirMEGData = 'myMEGdatapath'
dirScript = 'myscriptpath'

os.chdir(dirScript)
import MEG_Functions

ListFov = ['Foveal','ParaFoveal','Previous','Remaining']
# Select Foveal Condition
fov = 0

# Select Temporal Generalization
tempgen = 0 

ListSuj = ['B57D', 'B3EB', 'B5EB','B60D','B4E9','B5B3','B51C','B44F','B51F','B57B','B453','B457','B515','B3EE','B3F8','B4B8','B4B9','B4BB','B4BE', 'B4C1', 'B4C4', 'B4C6', 'B4C8','B57A','B57C','B58A', 'B329','B580','B581','B585','B588', 'B590', 'B3F3', 'B5E8', 'B591', 'B85B']
nsuj = len(ListSuj)

#%% Category: Average Conditions

for suj in range(nsuj):
    if tempgen == 0:
        filename = '_Classification_'+ListFov[fov]+'_FixationOnset_'
    elif tempgen == 1:
        filename = '_ClassificationTempGen_'+ListFov[fov]+'_FixationOnset_'
    MEG_Functions.MEG_Conditions_Averaging(suj, filename, dirMEGData)

#%% Classification: Gather Participants

# Load times data
times_Filename = dirMEGData+'times_'+ListFov[fov]+'.npy'
times = np.load(times_Filename, allow_pickle=True)
ntimes = len(times)

# Initialize
scores_all_Color = np.zeros([nsuj,ntimes])
scores_all_Cat = np.zeros([nsuj,ntimes])

# Gather Participants
for suj in range(nsuj): 
    Classif_Filename_Color = dirMEGData+suj+'/'+suj+'_Classification_'+ListFov[fov]+'_FixationOnset_Color_vs_Grey.npy'
    Classif_Filename_Cat = dirMEGData+suj+'/'+suj+'_Classification_'+ListFov[fov]+'_FixationOnset_Categories.npy' 
    scores_Color = np.load(Classif_Filename_Color, allow_pickle=True)
    scores_Cat = np.load(Classif_Filename_Cat, allow_pickle=True)
    scores_all_Color[suj,:] = scores_Color
    scores_all_Cat[suj,:] = scores_Cat
    
# Compute Mean
scores_avg_Color = np.mean(scores_all_Color, axis=0)
scores_avg_Cat = np.mean(scores_all_Cat, axis=0)
# Compute SEM
scores_sem_Color = (np.std(scores_all_Color, axis=0))/(math.sqrt(nsuj))
scores_sem_Cat = (np.std(scores_all_Cat, axis=0))/(math.sqrt(nsuj)) 
    
#%% Classification: Plot

# Peak 
t_start = 0
t_end = 0.250
index_t_start = int(np.array(np.where(times[:]==t_start)))
index_t_end = int(np.array(np.where(times[:]==t_end)))
times_timewin = times[index_t_start:index_t_end]

scores_avg_Color_timewin = scores_avg_Color[index_t_start:index_t_end]
peak_avg_Color = np.max(scores_avg_Color_timewin)
index_peak_avg_Color = np.argmax(scores_avg_Color_timewin)
print('Peak Color')
print(str(peak_avg_Color)+' AUC')
print(str((times_timewin[index_peak_avg_Color]))+' ms')

scores_avg_Cat_timewin = scores_avg_Cat[index_t_start:index_t_end]
peak_avg_Cat = np.max(scores_avg_Cat_timewin)
index_peak_avg_Cat = np.argmax(scores_avg_Cat_timewin)
print('Peak Category')
print(str(peak_avg_Cat)+' AUC')
print(str((times_timewin[index_peak_avg_Cat]))+' ms')

# Plot
color1 = colors['darkblue']
color2 = colors['crimson']
fig, ax = plt.subplots()
line1 = ax.plot(times, scores_avg_Color, label='score', color=color1, linestyle ='-')
line2 = ax.plot(times, scores_avg_Cat, label='score', color=color2, linestyle ='-')
ax.fill_between(times, scores_avg_Color-scores_sem_Color, scores_avg_Color+scores_sem_Color, color=color1, alpha=0.2)
ax.fill_between(times, scores_avg_Cat-scores_sem_Cat, scores_avg_Cat+scores_sem_Cat,  color=color2, alpha=0.2)
ax.axvline(times_timewin[index_peak_avg_Color], color=color1, linestyle='--')
ax.axvline(times_timewin[index_peak_avg_Cat], color=color2, linestyle='--')
plt.xlim([-0.25, 0.25]) 
ax.axhline(.5, color='k', linestyle='-', label='chance')
ax.axvline(.0, color='k', linestyle='-')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
plt.ylim([0.4, 0.7]) 
fig.set_figheight(8)
fig.set_figwidth(7)    

#%% Classification: Statistics

# Select window of interest 
tstart = -0.25
tend = 0.25
# Permutations across times 
[nulls_Color, thresh_Color] = MEG_Functions.MEG_Permutations_Statistics_Times(tstart, tend, times, scores_all_Color, nsuj)
[nulls_Cat, thresh_Cat] = MEG_Functions.MEG_Permutations_Statistics_Times(tstart, tend, times, scores_all_Cat, nsuj)


#%% Temporal Generalization: Gather Participants

# Load times data
times_Filename = dirMEGData+'times_'+ListFov[fov]+'.npy'
times = np.load(times_Filename, allow_pickle=True)
ntimes = len(times)

# Initialize
scores_all_Color = np.zeros([nsuj,ntimes, ntimes])
scores_all_Cat = np.zeros([nsuj,ntimes, ntimes])

# Gather Participants
for suj in range(nsuj): 
    Classif_Filename_Color = dirMEGData+suj+'/'+suj+'_ClassificationTempGen_'+ListFov[fov]+'_FixationOnset_Color_vs_Grey.npy'
    Classif_Filename_Cat = dirMEGData+suj+'/'+suj+'_ClassificationTempGen_'+ListFov[fov]+'_FixationOnset_Categories.npy' 
    scores_Color = np.load(Classif_Filename_Color, allow_pickle=True)
    scores_Cat = np.load(Classif_Filename_Cat, allow_pickle=True)
    scores_all_Color[suj,:,:] = scores_Color
    scores_all_Cat[suj,:,:] = scores_Cat

# Compute Mean
scores_avg_Color = np.mean(scores_all_Color, axis=0)
scores_avg_Cat = np.mean(scores_all_Cat, axis=0)

#%% Temporal Generalization: Plot

fig, ax = plt.subplots(1, 1)
plt.imshow(scores_avg_Color, interpolation='nearest', origin='lower', cmap='RdBu_r',
            vmin=0.4, vmax=0.6, extent = [-0.5 , 0.5, -0.5, 0.5])
ax.set_xlabel('Times Test (ms)')
ax.set_ylabel('Times Train (ms)')
plt.colorbar()


fig, ax = plt.subplots(1, 1)
plt.imshow(scores_avg_Cat, interpolation='nearest', origin='lower', cmap='RdBu_r',
            vmin=0.4, vmax=0.6, extent = [-0.5 , 0.5, -0.5, 0.5])
ax.set_xlabel('Times Test (ms)')
ax.set_ylabel('Times Train (ms)')
plt.colorbar()

#%% Temporal Generalization: Statistics

# Permutations across times*times
scores_t_Color, thresh_Color =MEG_Functions.MEG_Permutations_Statistics_TimesTimes(scores_all_Color, nsuj, ntimes)
scores_t_Cat, thresh_Cat = MEG_Functions.MEG_Permutations_Statistics_TimesTimes(scores_all_Cat, nsuj, ntimes)

#%% Foveal vs Parafoveal: Extract Peak Latencies

ListCond = ['Color_vs_Grey','Categories']
ncond = len(ListCond)
icond = 0

# Load times data
times_Filename = dirMEGData+'times_Foveal.npy'
times = np.load(times_Filename, allow_pickle=True)
ntimes = len(times)


# Initialize
scores_all_Foveal = np.zeros([nsuj,ntimes])
scores_all_ParaFoveal = np.zeros([nsuj,ntimes])

# Gather Participants
for suj in range(nsuj): 
    Classif_Filename_Foveal = dirMEGData+suj+'/'+suj+'_Classification_Foveal_FixationOnset_'+ListCond[icond]+'.npy'
    Classif_Filename_ParaFoveal = dirMEGData+suj+'/'+suj+'_Classification_ParaFoveal_FixationOnset_'+ListCond[icond]+'.npy'
    scores_Foveal = np.load(Classif_Filename_Foveal, allow_pickle=True)
    scores_ParaFoveal = np.load(Classif_Filename_ParaFoveal, allow_pickle=True)
    scores_all_Foveal[suj,:] = scores_Foveal
    scores_all_ParaFoveal[suj,:] = scores_ParaFoveal

# Select Time-Window of interest
if icond==0: # Color
    t_start = 0.06 # 60 ms
    t_end = 0.236
elif icond==1: # Category
    t_start = 0.16 # 160 ms
    t_end = 0.2 # 200 ms
    
index_t_start = int(np.array(np.where(times[:]==t_start)))
index_t_end = int(np.array(np.where(times[:]==t_end)))
times_timewin = times[index_t_start:index_t_end]
scores_all_Foveal_timewin = scores_all_Foveal[:,index_t_start:index_t_end]
scores_all_ParaFoveal_timewin = scores_all_ParaFoveal[:,index_t_start:index_t_end]

# Extract Peak Latency
peak_latency_Foveal = times_timewin[np.argmax(scores_all_Foveal_timewin,axis=1)]
peak_latency_ParaFoveal = times_timewin[np.argmax(scores_all_ParaFoveal_timewin,axis=1)]

# Compute Individual Difference
peak_latency_diff = peak_latency_ParaFoveal - peak_latency_Foveal

#%% Foveal vs Parafoveal: Plot

# Scatter Plot   
x_axis = np.ones([nsuj])
fig, ax = plt.subplots(1, 1)
ax.scatter(x_axis, peak_latency_diff, 40)
ax.scatter(1, np.mean(peak_latency_diff), 120)

    
# Violin plot
FovParaFov = np.zeros([nsuj,1])
FovParaFov[:,0] = peak_latency_diff
fig, ax = plt.subplots(1, 1)
ax.violinplot(FovParaFov, showmeans = True, showmedians=False, showextrema=True, widths=0.1)     
ax.set_xticks([]) 

#%% Foveal vs Parafoveal: Statistics

ttest_results = pg.ttest(peak_latency_diff, np.zeros(nsuj), alternative='two-sided', paired=True)

print('Diff-----------------------------')
print('Mean '+str(np.mean(peak_latency_diff)))
print('Median '+str(np.median(peak_latency_diff)))
print('STD '+str(np.std(peak_latency_diff)))

ttest_results = pg.ttest(peak_latency_Foveal, peak_latency_ParaFoveal, alternative='two-sided', paired=True)

print('Foveal-----------------------------')
print('Mean '+str(np.mean(peak_latency_Foveal)))
print('Median '+str(np.median(peak_latency_Foveal)))
print('STD '+str(np.std(peak_latency_Foveal)))

print('ParaFoveal-----------------------------')
print('Mean '+str(np.mean(peak_latency_ParaFoveal)))
print('Median '+str(np.median(peak_latency_ParaFoveal)))
print('STD '+str(np.std(peak_latency_ParaFoveal)))


def MEG_Prepare_Data_Classification(epoch, tmin, tmax):
    
    # Crop
    epoch_crop = epoch.crop(tmin=tmin, tmax=tmax)
    
    # Get times
    times = epoch_crop.times
    
    # Select MEG sensors
    epoch_classifier = epoch_crop.get_data(picks='meg') 
   
    return epoch_classifier, times

#%%

def MEG_DelayEmbedding(epoch):
    
    import numpy as np
    
    # Settings
    nTrials = epoch.shape[0]
    nElecs = epoch.shape[1]
    nTimePoints = epoch.shape[2]
    TimeLag = 25
    nTimePointsEmbedded = nTimePoints
    nTimePoints_2 = int(nTimePointsEmbedded + (TimeLag-1)/2)
    
    # Initialize new epoch 
    embedded_epoch = np.zeros((nTrials,nElecs,nTimePointsEmbedded))
    # Loop on Trials
    for trial in range(nTrials):
        # Loop on Elecs
        for elec in range(nElecs):
            # Select initial time series
            time_serie = epoch[trial, elec, 0:nTimePointsEmbedded]
            time_serie = time_serie - np.mean(time_serie)
            # Compute shifted time series
            time_serie2 = epoch[trial, elec, 0:nTimePoints_2]
            time_serie2 = time_serie2 - np.mean(time_serie2)
            # Initialize embedded time series
            time_serie_3 = np.zeros(nTimePointsEmbedded)
            j = 0
            tstart = int(((TimeLag-1)/2)+1)
            tend = int(nTimePoints_2 - ((TimeLag-1)/2) -1)
            for m in range(tstart, tend):
                time_serie_3[j] = (np.sum(time_serie2[int((m-(TimeLag-1)/2)):int((m+(TimeLag-1)/2))]))/(TimeLag)
                j += 1
                j
            # Indent new epoch
            embedded_epoch[trial, elec, :] = time_serie_3
    
    return embedded_epoch

#%% 

def MEG_Conditions_Averaging(suj, filename, dirMEGData):
    
    import numpy as np
    
    # Load scores
    # Animal vs Food
    AnimalvsFood_Filename = dirMEGData+suj+'/'+suj+filename+'Animal_vs_Food.npy'
    AnimalvsFood_scores = np.load(AnimalvsFood_Filename, allow_pickle=True)
    # Animal vs Object
    AnimalvsObject_Filename = dirMEGData+suj+'/'+suj+filename+'Animal_vs_Object.npy'
    AnimalvsObject_scores = np.load(AnimalvsObject_Filename, allow_pickle=True)
    # Food vs Object
    FoodvsObject_Filename = dirMEGData+suj+'/'+suj+filename+'Food_vs_Object.npy'
    FoodvsObject_scores = np.load(FoodvsObject_Filename, allow_pickle=True)
    
    # Conditions averaging
    AnimalvsOthers_scores = (AnimalvsFood_scores+AnimalvsObject_scores)/2
    FoodvsOthers_scores = (AnimalvsFood_scores+FoodvsObject_scores)/2
    ObjectvsOthers_scores = (AnimalvsObject_scores+FoodvsObject_scores)/2
    Categories_scores = (AnimalvsFood_scores+AnimalvsObject_scores+FoodvsObject_scores)/3
    
    # Save
    Categories_Filename = dirMEGData+suj+'/'+suj+filename+'Categories.npy'
    np.save(Categories_Filename, Categories_scores)
    
#%%

def MEG_Permutations_Statistics_Times(tstart, tend, times, scores, nsuj):
    
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    
    index_t_start = int(np.array(np.where(times[:]==tstart)))
    index_t_end = int(np.array(np.where(times[:]==tend)))
    scores_timewin = scores[:,index_t_start:index_t_end]
    ntimes_perm = scores_timewin.shape[1]
    times_perm = times[index_t_start:index_t_end]
    
    # Remove chance level from real scores - put the null hypothesis at 0
    scores_perm = scores_timewin - 0.5
    # Initiate
    nperms = 1500
    nulls = np.zeros((nperms,))
    # Compute t-test against 0 for each time point
    scores_t, scores_p = stats.ttest_1samp(scores_perm, 0, axis=0)
    # Get t-max
    nulls[0] = scores_t.max()  # First null is the same as the observed data
    # Permutation Loop
    for ii in range(1, nperms):
        # Randomly select 1 or -1 for each subject each time point and multiply scores 
        # Aim: Attribute randomly according to chance level (50%) a plus or a minus to the data 
        # Because according to the null hypothesis, mean = 0, so equivalent number of plus and minus  
        null_scores = np.random.choice([1, -1], (nsuj, ntimes_perm)) * scores_perm
        # Compute t-test against 0 for each time point
        t, p = stats.ttest_1samp(null_scores, 0, axis=0)
        # Get t-max
        nulls[ii] = t.max()
    # Compute threshold at 95, 99 and 99.9
    thresh = np.percentile(nulls, [95, 99, 99.9])
    
    # Plot Statistics 
    fig, ax = plt.subplots()
    plt.plot(times_perm, scores_t)
    ax.axhline(thresh[0], color='k', linestyle='-', label='p=0.05')
    ax.axhline(thresh[1], color='k', linestyle='--', label='p=0.01')
    ax.axhline(thresh[2], color='k', linestyle=':', label='p=0.001') 
    plt.legend(['T-Value', 'p=0.05', 'p=0.01', 'p=0.001'])
    plt.xlabel('Times')
    plt.ylabel('t-value')
    plt.xlim([tstart, tend])
    fig.set_figheight(8)
    fig.set_figwidth(7)
    
    return nulls, thresh

#%% 
def MEG_Permutations_Statistics_TimesTimes(scores, nsuj, ntimes):
    
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Permutations statistics -------------------------------------------------
    # Remove chance level from real scores - put the null hypothesis at 0
    scores_perm = scores - 0.5
    # Initiate
    nperms = 1500
    nulls = np.zeros((nperms,))
    # Compute t-test against 0 for each time point
    scores_t, scores_p = stats.ttest_1samp(scores_perm, 0)
    # Remove NaN
    indexnan = np.isnan(scores_t)
    scores_t_nonan = scores_t
    scores_t_nonan[indexnan] = 0
    # Get t-max
    nulls[0] = scores_t.max()  # First null is the same as the observed data
    
    # Permutation Loop
    for ii in range(1, nperms):
        # Randomly select 1 or -1 for each subject each time point and multiply scores 
        # Aim: Attribute randomly according to chance level (50%) a plus or a minus to the data 
        # Because according to the null hypothesis, mean = 0, so equivalent number of plus and minus  
        null_scores = np.random.choice([1, -1], (nsuj, ntimes, ntimes)) * scores_perm
        # Compute t-test against 0 for each time point
        t, p = stats.ttest_1samp(null_scores, 0)
        t[indexnan] = 0
        # Get t-max
        nulls[ii] = t.max()
    # Compute threshold at 95, 99 and 99.9
    thresh = np.percentile(nulls, [95, 99, 99.9])
    
    # Figure
    scores_pval = scores_t
    for ii in range(ntimes):
        for iii in range(ntimes):
            if scores_pval[ii,iii] >= thresh[2]:
                scores_pval[ii,iii] = 3 
            elif thresh[2] > scores_pval[ii,iii] >= thresh[1]:
                scores_pval[ii,iii] = 2
            elif thresh[1] > scores_pval[ii,iii] >= thresh[0]:
                  scores_pval[ii,iii] = 1
            else:
                scores_pval[ii,iii] = 0
    
    fig, ax = plt.subplots(1, 1) 
    cmap = mpl.cm.binary
    plt.imshow(scores_pval, origin='lower', cmap=cmap)
    plt.colorbar()
    
    return scores_t, thresh


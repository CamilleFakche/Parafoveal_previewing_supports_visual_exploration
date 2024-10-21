###############################################################################
###                EYE DATA: FIXATION ONSET EPOCH SELECTION                  ###
###############################################################################
# by Camille Fakche 07/08/2024

import numpy as np

dirEyeData = 'myeyedatapath'
dirImageInfo = 'myimageinfopath'

ListFov = ['Foveal','ParaFoveal','Previous','Remaining']
# Select Foveal Condition
fov = 0

# Example One Participant
suj = 'B3EB'

#%% 

# Define Fixation Event
FixEvent = 10
nTrials = 300

# Load Images Info
ListBlock = ['1','2','3','4','5','6','7','8','9','10']
nblock = len(ListBlock)
for block in range(nblock):
    ImagesInfos_filename = dirImageInfo+suj+'_ImageInfos_allTrials_block'+ListBlock[block]+'.npy'
    ImagesInfos_block = np.load(ImagesInfos_filename, allow_pickle=True)  
    if block == 0:
        ImagesInfos = ImagesInfos_block
    else:
        ImagesInfos = np.append(ImagesInfos, ImagesInfos_block)

for trial in range(nTrials):
    # Select Image Infos
    ImagesInfos_Trial = ImagesInfos[trial]
    
    # Load Eye Events data
    AllEyeEvent_filename = dirEyeData+suj+'_AllEyeEvent_trial'+str(trial)+'.npy'
    AllEyeEvent = np.load(AllEyeEvent_filename, allow_pickle=True) 
    
    # Define VISUAL SEARCH Limits
    index_visualsearch = np.array(np.where(AllEyeEvent[:,0]==11111))  
    index_mask = np.array(np.where(AllEyeEvent[:,0]==22222)) 
    start = index_visualsearch[0,0]
    end = index_mask[0,0]
    
    # Select VISUAL SEARCH Events
    AllEyeEventCond = AllEyeEvent[start:end,:] 
    
    # Do not consider the end of the fixation that starts before 
    if AllEyeEventCond.shape[0] > 1:
        if AllEyeEventCond[1,0] == FixEvent:
            AllEyeEventCond[1,0] = np.nan 
            
    # Select Fixations Events
    index = np.array(np.where(AllEyeEventCond[:,0]==FixEvent))
    FixationsEvents = np.squeeze(AllEyeEventCond[index,:])

    if index.shape[1] < 2:
        FixationsEvents = np.array([])
        
    if FixationsEvents.size != 0:
        
        # Find first fixations 
        nFix = len(FixationsEvents)
        FixationsEvents[:,9] = np.ones([1,nFix])
        for fix1 in range(nFix):
            for fix2 in range(nFix):
                if FixationsEvents[fix1,7] == FixationsEvents[fix2,7] and fix2 > fix1:
                    FixationsEvents[fix2,9] = FixationsEvents[fix2,9] + 1
        
        first_fix = 1
        # Loop over fixations
        if ListFov[fov] == 'Foveal':
            LoopFix = range(nFix)
        elif ListFov[fov] == 'ParaFoveal':
            LoopFix = range(nFix-1)
        elif ListFov[fov] == 'Previous': 
            LoopFix = range(1, nFix)
        elif ListFov[fov] == 'Remaining':
            LoopFix = range(nFix-1)  
        for fix in LoopFix:
            keep_fix = 1
            match_im = 1
            
            # We do not analyze Fixation < 80 ms and > 1000 ms
            if FixationsEvents[fix,3] < 0.08 or FixationsEvents[fix,3] > 1:
                keep_fix = 0
            # We do not analyze Fixation on the background
            if np.isnan(FixationsEvents[fix,6]):
                keep_fix = 0
            
            # FOVEAL Criteria
            if ListFov[fov] == 'Foveal':
                # We do not analyze Fixation if the image has been already seen 
                if FixationsEvents[fix,9] > 1:
                    keep_fix = 0
            
            # PARAFOVEAL Criteria
            elif ListFov[fov] == 'ParaFoveal':
                # We do not analyze Fixation if the image fix+1 is on the background
                if np.isnan(FixationsEvents[fix+1,6]):
                    keep_fix = 0
                # We do not analyze Fixation if the image fix+1 has been already seen 
                if FixationsEvents[fix+1,9] > 1:
                    keep_fix = 0    
                # We do not analyze Fixation if the image fix+1 is not consecutive or on the same image
                if FixationsEvents[fix,7] == 1:
                    if FixationsEvents[fix+1,7] == 3 or FixationsEvents[fix+1,7] == 4 or FixationsEvents[fix+1,7] == 5 or FixationsEvents[fix+1,7] == 1:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 2:
                    if FixationsEvents[fix+1,7] == 4 or FixationsEvents[fix+1,7] == 5 or FixationsEvents[fix+1,7] == 6 or FixationsEvents[fix+1,7] == 2:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 3:
                    if FixationsEvents[fix+1,7] == 5 or FixationsEvents[fix+1,7] == 6 or FixationsEvents[fix+1,7] == 1 or FixationsEvents[fix+1,7] == 3:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 4:
                    if FixationsEvents[fix+1,7] == 6 or FixationsEvents[fix+1,7] == 1 or FixationsEvents[fix+1,7] == 2 or FixationsEvents[fix+1,7] == 4:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 5:
                    if FixationsEvents[fix+1,7] == 1 or FixationsEvents[fix+1,7] == 2 or FixationsEvents[fix+1,7] == 3 or FixationsEvents[fix+1,7] == 5:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 6: 
                    if FixationsEvents[fix+1,7] == 2 or FixationsEvents[fix+1,7] == 3 or FixationsEvents[fix+1,7] == 4 or FixationsEvents[fix+1,7] == 6:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 7:
                    if FixationsEvents[fix+1,7] == 7:
                        keep_fix = 0        
                # We do not analyze Fixation if Color Parafoveal == Color Foveal
                if FixationsEvents[fix+1,8] == FixationsEvents[fix,8]:
                    keep_fix = 0 
                # We do not analyze Fixation if Category Parafoveal == Category Foveal
                if FixationsEvents[fix+1,6] == FixationsEvents[fix,6]:
                    keep_fix = 0 
                    
            # PREVIOUS Criteria 
            elif ListFov[fov] == 'Previous':
                # We do not analyze Fixation if the image fix-1 is on the background
                if np.isnan(FixationsEvents[fix-1,6]):
                    keep_fix = 0
                # We do not analyze Fixation if the image fix-1 has been already seen 
                if FixationsEvents[fix-1,9] > 1:
                    keep_fix = 0
                # We do not analyze Fixation if the image fix-1 is not consecutive or on the same image
                if FixationsEvents[fix,7] == 1:
                    if FixationsEvents[fix-1,7] == 3 or FixationsEvents[fix-1,7] == 4 or FixationsEvents[fix-1,7] == 5 or FixationsEvents[fix-1,7] == 1:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 2:
                    if FixationsEvents[fix-1,7] == 4 or FixationsEvents[fix-1,7] == 5 or FixationsEvents[fix-1,7] == 6 or FixationsEvents[fix-1,7] == 2:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 3:
                    if FixationsEvents[fix-1,7] == 5 or FixationsEvents[fix-1,7] == 6 or FixationsEvents[fix-1,7] == 1 or FixationsEvents[fix-1,7] == 3:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 4:
                    if FixationsEvents[fix-1,7] == 6 or FixationsEvents[fix-1,7] == 1 or FixationsEvents[fix-1,7] == 2 or FixationsEvents[fix-1,7] == 4:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 5:
                    if FixationsEvents[fix-1,7] == 1 or FixationsEvents[fix-1,7] == 2 or FixationsEvents[fix-1,7] == 3 or FixationsEvents[fix-1,7] == 5:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 6: 
                    if FixationsEvents[fix-1,7] == 2 or FixationsEvents[fix-1,7] == 3 or FixationsEvents[fix-1,7] == 4 or FixationsEvents[fix-1,7] == 6:
                        keep_fix = 0
                elif FixationsEvents[fix,7] == 7:
                    if FixationsEvents[fix-1,7] == 7:
                        keep_fix = 0         
                # We do not analyze Fixation if Color Parafoveal == Color Foveal
                if FixationsEvents[fix-1,8] == FixationsEvents[fix,8]:
                    keep_fix = 0 
                # We do not analyze Fixation if Category Parafoveal == Category Foveal
                if FixationsEvents[fix-1,6] == FixationsEvents[fix,6]:
                    keep_fix = 0 
            
            # REMAINING Criteria 
            elif ListFov[fov] == 'Remaining' and keep_fix == 1:
                # For each foveal image, determine the parafoveal images
                if FixationsEvents[fix,7] == 1:
                    parafoveal_im = [2, 7, 6]
                elif FixationsEvents[fix,7] == 2:
                    parafoveal_im = [1, 7, 3]
                elif FixationsEvents[fix,7] == 3:
                    parafoveal_im = [2, 7, 4]
                elif FixationsEvents[fix,7] == 4:
                    parafoveal_im = [3, 7, 5]
                elif FixationsEvents[fix,7] == 5:
                    parafoveal_im = [4, 6, 7]
                elif FixationsEvents[fix,7] == 6:
                    parafoveal_im = [1, 5, 7]
                elif FixationsEvents[fix,7] == 7:
                    parafoveal_im = [1, 2, 3, 4, 5, 6]    
                # If one parafoveal image is fix+1, remove this image from the list
                next_fix = FixationsEvents[fix+1,7]
                if np.isin(next_fix, parafoveal_im) == True:
                    parafoveal_im = np.delete(parafoveal_im, np.where(parafoveal_im == next_fix))
                # If one parafoveal image is fix-1, remove this image from the list
                if fix > 0:
                    previous_fix = FixationsEvents[fix-1,7]
                    if np.isin(previous_fix, parafoveal_im) == True:
                        parafoveal_im = np.delete(parafoveal_im, np.where(parafoveal_im == previous_fix))
                # Select parafoveal image that do not match features of the foveal image
                features_foveal = float(str(FixationsEvents[fix,6])[0]+str(FixationsEvents[fix,8])[0]) 
                for im in range(len(parafoveal_im)):
                    color = ImagesInfos_Trial[parafoveal_im[im]-1,6]
                    cat = ImagesInfos_Trial[parafoveal_im[im]-1,4]
                    features_parafoveal = float(str(cat)+str(color))
                    if features_foveal != features_parafoveal:
                        parafoveal_im = parafoveal_im[im]
                        break
                    else: 
                        match_im = 0
            
            if keep_fix == 1 and match_im ==1: 
               # Define Trigger
               if ListFov[fov] == 'Foveal':
                   trigger = float(str(FixationsEvents[fix,6])[0]+str(FixationsEvents[fix,8])[0])
               elif ListFov[fov] == 'ParaFoveal':
                   trigger = float(str(FixationsEvents[fix+1,6])[0]+str(FixationsEvents[fix+1,8])[0])
               elif ListFov[fov] == 'Previous': 
                   trigger = float(str(FixationsEvents[fix-1,6])[0]+str(FixationsEvents[fix-1,8])[0])
               elif ListFov[fov] == 'Remaining':
                   trigger = features_parafoveal
               # Define time onset
               time_onset = FixationsEvents[fix,1]
               # Epoch to add
               Epoch = np.zeros([1,3])
               Epoch[0,0] = trigger
               Epoch[0,2] = time_onset  
               # Create array for Fixations Epochs
               if first_fix == 1:
                   first_fix = 0
                   FixationsEpoch = Epoch 
               else:
                   FixationsEpoch = np.append(FixationsEpoch,Epoch,axis=0)   
    else:
         FixationsEpoch = np.array([])
     # Save
    FixationsEpoch_filename = dirEyeData+suj+'_'+ListFov[fov]+'_FixationsOnsetEpoch_trial'+str(trial)+'.npy'
    np.save(FixationsEpoch_filename, FixationsEpoch)   
             
             
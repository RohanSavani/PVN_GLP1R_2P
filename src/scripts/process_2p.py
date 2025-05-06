import os
import numpy as np
from scipy.ndimage import uniform_filter1d


def load_s2p_data(path_to_suite2p_folder):
    """
    Load 2p data from the specified path. Fneu not loaded as it is now. 
    """
    # Load the data
    f = np.load(os.path.join(path_to_suite2p_folder, 'F.npy'))
    # fn = np.load(os.path.join(path_to_suite2p_folder, "Fneu.npy"))
    iscell=np.load(os.path.join(path_to_suite2p_folder,"iscell.npy"))
    ops=np.load(os.path.join(path_to_suite2p_folder,"ops.npy"), allow_pickle=True)


    return f, iscell, ops


def filter_cells(F, iscell, filter_by='first'):
    """
    Filters the cells in F based on the iscell array.
    
    Parameters:
        F (np.ndarray): Fluorescence data array.
        iscell (np.ndarray): Array indicating cell classification.
        filter_by (str): 'first' to filter by iscell[:, 0] == 1,
                        'second' to filter by iscell[:, 1] == 1.
    Returns:
        np.ndarray: Filtered fluorescence data. 
    """
    if filter_by == 'first':
        mask = iscell[:, 0] == 1
    elif filter_by == 'second':
        mask = iscell[:, 1] == 1
    else:
        raise ValueError("filter_by must be 'first' or 'second'")
    return F[mask]


def reshape_data(F, n_trials = 30, framespertrial = 375, fps = 15):
    """
    Rearranges the fluorescence data into a 3D array.
    
    Parameters:
        F (np.ndarray): Fluorescence data array.
        n_trials (int): Number of trials.
        framespertrial (int): Number of frames per trial.
        fps (int): Frames per second.
        
    Returns:
        np.ndarray: Rearranged fluorescence data.
    """
    n_cells = F.shape[0]

    # Reshape F to have dimensions (n_cells, n_trials, framespertrial)
    F_reshaped = F.reshape(n_cells, n_trials, framespertrial)
    
    return F_reshaped


def normalize_data(F, use_baseline = True, baseline_period = [0, 105], trial_by_trial = True):
    """
    Normalizes the fluorescence data.
    
    Parameters:
        F (np.ndarray): Fluorescence data array.
        use_baseline (bool): Whether to use baseline normalization.
        baseline_period (list): Time period for baseline normalization.
        
    Returns:
        np.ndarray: Normalized fluorescence data.
    """
    n_cells, n_trials, frames = F.shape
    F_normalized = np.zeros_like(F)

    if trial_by_trial:
        for i in range(n_cells):
            for j in range(n_trials):
                if use_baseline:
                    baseline = np.mean(F[i, j, baseline_period[0]:baseline_period[1]])
                else:
                    baseline = np.mean(F[i, j, :])
            
                F_normalized[i, j, :] = (F[i, j, :]) / baseline
    else:
        for i in range(n_cells):
            if use_baseline:
                baseline = np.mean(F[i, :, baseline_period[0]:baseline_period[1]])
            else:
                baseline = np.mean(F[i, :, :])
            
            F_normalized[i, :, :] = (F[i, :, :]) / baseline
        
    return F_normalized 


def align_2p_to_licks(filt_f, bout_start_frame, pre_time = 75, post_time = 150, reward_frame = 150):
    """
    Aligns 2p data to lick bout start frames.
    
    Parameters
    ----------
    filt_f : array
        Filtered 2p data.
    bout_start_frame : array
        Lick bout start frames.
    pre_time : int
        Number of frames before the lick bout to include.
    post_time : int
        Number of frames after the lick bout to include.
    reward_frame : int
        Frame number of reward delivery.
        
    Returns
    -------
    aligned_data : array
        Aligned 2p data.
    """
    if len(bout_start_frame) != filt_f.shape[1]:
        raise ValueError("Number of bout start frames does not match number of trials in 2p data.")
    if np.any(bout_start_frame < reward_frame) and np.any(bout_start_frame >= reward_frame):
        raise ValueError("Inconsistent bout_start_frame values: some are before and some are after reward_frame.")

    
    aligned_data = np.zeros((filt_f.shape[0], len(bout_start_frame), pre_time + post_time))

    if all(bout_start_frame < reward_frame):
        bout_start_frame = bout_start_frame + reward_frame

    for i, start_frame in enumerate(bout_start_frame):
        aligned_data[:, i, :] = filt_f[:, i, start_frame - pre_time:start_frame + post_time]

    return aligned_data


def align_2p_to_cues(filt_f, cue_frame = 105, pre_time = 75, post_time = 150):
    """
    Aligns 2P data to cue frames.
    
    Parameters
    ----------
    filt_f : array
        Filtered 2P data.
    cue_frame : int
        Frame number of the cue.
    pre_time : int
        Time before the cue to include in the aligned data.
    post_time : int
        Time after the cue to include in the aligned data.

    Returns
    -------
    aligned_data : array
        Aligned 2P data.
    """
    num_trials = filt_f.shape[1]
    aligned_data = np.zeros((filt_f.shape[0], num_trials, pre_time + post_time))
    
    for i in range(num_trials):
        aligned_data[:, i, :] = filt_f[:, i, cue_frame - pre_time:cue_frame + post_time]
    
    return aligned_data


def get_baseline_filt_f(f, baseline_start_frame = 30, baseline_end_frame = 105):
    """
    Get the baseline fluorescence array for each cell.
    """
    return f[:, :, baseline_start_frame:baseline_end_frame]


def average_trials(f):
    """
    Average across trials for each cell
    """
    if f.ndim != 3:
        raise ValueError("Input array must be 3D (cells x trials x time)")
    return np.nanmean(f, axis=1)


def moving_average(data, window_size = 5):
    """
    Computes the moving average of the data.
    
    Parameters:
        data (np.ndarray): Data to compute the moving average on.
        window_size (int): Size of the moving window.
        
    Returns:
        np.ndarray: Moving average of the data.
    """
    return uniform_filter1d(data, size=window_size)
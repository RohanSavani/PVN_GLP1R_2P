import os
import numpy as np
from scipy.ndimage import uniform_filter1d
import lick_behav_analysis as behav
import scipy.io as sio
from scipy.integrate import simpson


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
    if trial_by_trial:
        n_cells, n_trials, frames = F.shape
    else:
        n_cells, frames = F.shape
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
                baseline = np.mean(F[i, :])
            
            F_normalized[i, :] = (F[i, :]) / baseline
        
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

def get_auc(data, start_frame, end_frame, normalize = 1, ms_per_frame = int(1000//15), dim = 2):
    """
    Computes the area under the curve (AUC) for the given data.
    
    Parameters:
        data (np.ndarray): Data to compute the AUC on.
        start_frame (int): Start frame for AUC calculation.
        end_frame (int): End frame for AUC calculation.
        normalize (bool): Whether to normalize the AUC.
        ms_per_frame (int): Milliseconds per frame, default is 1000//15.
        
    Returns:
        float: Area under the curve.
    """
    if dim == 2:
        auc = simpson(data[:, start_frame:end_frame], dx=ms_per_frame)
    elif dim == 3:
        auc = simpson(data[:, :, start_frame:end_frame], dx=ms_per_frame, axis=2)
    return auc / normalize


def process_2p_folder(folder, fps = 15, align = 'lick', success = 'success'):
    """
    Process 2p data for a given folder.

    Parameters:
    - folder: str, path to the folder containing the data
    - fps: int, frames per second
    - align: str, alignment type ('lick' or 'cue')
    - success: bool, whether to include only successful trials or only unsuccessful trials
    
    Returns:
    - None
    """

    all_avg_f = []
    all_baseline_data = []
    all_aligned_f = []
    all_successful_f = []

    for path in [f for f in os.listdir(folder) if not f.startswith('.')]:
        matpath = os.path.join(folder, path, 'suite2p', 'plane0', 'behaviordata.mat')
        cyto_suite = os.path.join(folder, path, 'suite2p', 'plane0')

        # Load data
        f, iscell, ops = load_s2p_data(cyto_suite)
        filt_f = filter_cells(f, iscell)
        filt_f_reshaped = reshape_data(filt_f)
        filt_f_norm = normalize_data(filt_f_reshaped)

        # Process behavior data
        bout_start, bout_end, bout_start_frames, bout_end_frames = behav.og_lickprocessing(matpath)
        if success == 'success':
            successful_trial_idx = [i for i, x in enumerate(bout_start) if (x > 0) & (x < 5000)]
        elif success == 'unsuccess':
            successful_trial_idx = [i for i, x in enumerate(bout_start) if not (x > 0) & (x < 5000)]
        elif success == 'all':
            successful_trial_idx = [i for i, _ in enumerate(bout_start)]
        successful_bout_start_frames = bout_start_frames[successful_trial_idx]

        # Combine lick data with imaging data
        successful_filt_f = filt_f_norm[:, successful_trial_idx, :]
        if align == 'lick':
            filt_f_aligned = align_2p_to_licks(successful_filt_f, successful_bout_start_frames)
        elif align == 'cue':
            filt_f_aligned = align_2p_to_cues(successful_filt_f)

        # Get baseline data
        baseline_data = get_baseline_filt_f(successful_filt_f)

        # Average across trials
        avg_f = average_trials(filt_f_aligned)

        # Store data 
        all_avg_f.append(avg_f)
        all_baseline_data.append(baseline_data)
        all_aligned_f.append(filt_f_aligned)
        all_successful_f.append(successful_filt_f)
    
    # Concatenate avg data (no trials axis )
    all_avg_f = np.concatenate(all_avg_f, axis=0)

    return all_avg_f, all_baseline_data, all_aligned_f, all_successful_f


def process_2p_folder_mt(folder, n_trials, fps = 15, align = 'lick', success = 'success'):
    """
    Process 2p data for a given folder.

    Parameters:
    - folder: str, path to the folder containing the data
    - multiple_tastants: bool, whether to process multiple tastants
    - fps: int, frames per second
    - align: str, alignment type ('lick' or 'cue')
    - success: bool, whether to include only successful trials or only unsuccessful trials
    
    Returns:
    - None
    """
    
    all_avg_f_suc = []
    all_avg_f_alt = []
    all_baseline_data_suc = []
    all_baseline_data_alt = []
    all_aligned_f_suc = []
    all_aligned_f_alt = []
    all_successful_f_suc = []
    all_successful_f_alt = []

    for path in [f for f in os.listdir(folder) if not f.startswith('.')]:
        matpath = os.path.join(folder, path, 'suite2p', 'plane0', 'behaviordata.mat')
        cyto_suite = os.path.join(folder, path, 'suite2p', 'plane0')
        beh = sio.loadmat(matpath)

        # Load data
        f, iscell, ops = load_s2p_data(cyto_suite)
        filt_f = filter_cells(f, iscell)
        filt_f_reshaped = reshape_data(filt_f, n_trials = n_trials)
        filt_f_norm = normalize_data(filt_f_reshaped)

        # Process behavior data
        cues      = np.squeeze(beh['cues'])    / 1000.0
        cues      = cues[cues > 0]

        alt_ts = np.squeeze(beh['cuesminus']) / 1000.0
        alt_idx, suc_idx = [], []
        for idx, t in enumerate(cues):
            (alt_idx if t in alt_ts else suc_idx).append(idx)

        bout_start, bout_end, bout_start_frames, bout_end_frames = behav.og_lickprocessing(matpath)
        if success == 'success':
            successful_trial_idx = [i for i, x in enumerate(bout_start) if (x > 0) & (x < 5000)]
        elif success == 'unsuccess':
            successful_trial_idx = [i for i, x in enumerate(bout_start) if not (x > 0) & (x < 5000)]
        elif success == 'all':
            successful_trial_idx = [i for i, _ in enumerate(bout_start)]

        suc_trials = [x for x in successful_trial_idx if x in suc_idx]
        alt_trials = [x for x in successful_trial_idx if x in alt_idx]

        successful_bout_start_frames_suc = bout_start_frames[suc_trials]
        successful_bout_start_frames_alt = bout_start_frames[alt_trials]

        # Combine lick data with imaging data

        successful_filt_f_suc = filt_f_norm[:, suc_trials, :]
        successful_filt_f_alt = filt_f_norm[:, alt_trials, :]

        if align == 'lick':
            filt_f_aligned_suc = align_2p_to_licks(successful_filt_f_suc, successful_bout_start_frames_suc)
            filt_f_aligned_alt = align_2p_to_licks(successful_filt_f_alt, successful_bout_start_frames_alt)
        elif align == 'cue':
            filt_f_aligned_suc = align_2p_to_cues(successful_filt_f_suc)
            filt_f_aligned_alt = align_2p_to_cues(successful_filt_f_alt)

        # Get baseline data
        baseline_data_suc = get_baseline_filt_f(successful_filt_f_suc)
        baseline_data_alt = get_baseline_filt_f(successful_filt_f_alt)

        # Average across trials
        avg_f_suc = average_trials(filt_f_aligned_suc)
        avg_f_alt = average_trials(filt_f_aligned_alt)

        # Store data 
        all_avg_f_suc.append(avg_f_suc)
        all_avg_f_alt.append(avg_f_alt)
        all_baseline_data_suc.append(baseline_data_suc)
        all_baseline_data_alt.append(baseline_data_alt)
        all_aligned_f_suc.append(filt_f_aligned_suc)
        all_aligned_f_alt.append(filt_f_aligned_alt)
        all_successful_f_suc.append(successful_filt_f_suc)
        all_successful_f_alt.append(successful_filt_f_alt)
    
    # Concatenate avg data (no trials axis )
    all_avg_f_suc = np.concatenate(all_avg_f_suc, axis=0)
    all_avg_f_alt = np.concatenate(all_avg_f_alt, axis=0)


    return all_avg_f_suc, all_avg_f_alt, all_baseline_data_suc, all_baseline_data_alt, all_aligned_f_suc, all_aligned_f_alt, all_successful_f_suc, all_successful_f_alt

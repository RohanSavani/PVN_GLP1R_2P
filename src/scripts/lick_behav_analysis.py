import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d

def og_lickprocessing(matfile, fps = 15, return_more = False, bout_limit = 5000, filter_pre_rew=True, filter_pre_licknum = 20): 
    """
    Function to process lick data from a .mat file. It calculates bout start, end, length, and lick rate for each trial.
    Args:
        matfile (str): Path to the .mat file containing lick data.
        fps (int): Frames per second for the video.
        return_more (bool): If True, returns additional data.
        bout_limit (int): Time limit for considering licks in a trial.
        filter_pre_rew (bool): If True, filters out pre-reward licks.
    Returns:
        bout_start (np.ndarray): Start times of licks in each trial.
        bout_end (np.ndarray): End times of licks in each trial.
        bout_start_frames (np.ndarray): Start times of licks in frames.
        bout_end_frames (np.ndarray): End times of licks in frames.
        firstlickafterrew (np.ndarray): Time of the first lick after reward.
        latency (list): Latency of the first lick after reward for each trial.
        lick_rate (list): Lick rate for each trial.
        bout_length (list): Length of the lick bout for each trial.
        all_licks (list): List of all licks in each trial.
    """
    behaviordata = sio.loadmat(matfile)
    cues=np.squeeze(behaviordata['cues'])
    cues=cues[cues>0] ## remove cue artifact at 0
    licks=np.squeeze(behaviordata['licks'])
    t_fxd=np.squeeze(behaviordata['fxdpumps'])
    num_trials = len(t_fxd)
    ms_per_frame=int(1000//fps)
 
    firstlickafterrew=np.nan*np.zeros((len(t_fxd)))
    latency = [0]*len(t_fxd)
    lick_rate = [0]*len(t_fxd)
    bout_length = [0]*len(t_fxd)
    bout_start = [0]*len(t_fxd)
    bout_end = [0]*len(t_fxd)
    bout_start_frames = np.asarray([0]*len(t_fxd))
    bout_end_frames = np.asarray([0]*len(t_fxd))
    all_licks = []

    for a in range(0,len(t_fxd)):
        ##Calculate reward and the number of licks within the trial. Limited to within 5 seconds of reward
        rew_t = t_fxd[a]
        licks_total = licks[licks>100+t_fxd[a]]#[1:]
        licks_in_trial = licks_total[licks_total<(bout_limit+t_fxd[a])]
        licks_in_trial = list(licks_in_trial)
        all_licks.append(licks_in_trial)
        pre_rew_deliv = licks[(licks > (t_fxd[a] - 3000)) & (licks < t_fxd[a])]
        lick_differences = [0]*(len(licks_in_trial)-1)
        for m in range(0,len(licks_in_trial)-1):
                lick_differences[m] = licks_in_trial[m+1]-licks_in_trial[m]
        for h in range(0,len(lick_differences)):
            if lick_differences[h] < 30:
                licks_in_trial[h+1] = 0
        licks_in_trial = list(licks_in_trial)
        while 0 in licks_in_trial: licks_in_trial.remove(0)

        if filter_pre_rew: 
            if len(pre_rew_deliv) >= filter_pre_licknum:
                licks_in_trial = [0]

        #If the mouse did lick in the trial, proceed. If not, cast all the relevant variables as 0
        #Because we know that physiologically, a lick rate, a bout start, end, or length should not be 0 if a mouse did lick.
        if len(licks_in_trial) > 2:
            lick_diffs = [0]*(len(licks_in_trial)-1)
            fist_lick_t = licks[licks>100+t_fxd[a]][1]
            latency[a] = fist_lick_t-rew_t
            ##To find bouts, first we calculate the difference in time between licks in the trial
            for j in range(0,len(licks_in_trial)-1):
                lick_diffs[j] = licks_in_trial[j+1]-licks_in_trial[j]
            for b in range(0,len(lick_diffs)):
                if lick_diffs[b] < 30:
                    licks_in_trial[b+1] = 0
            while 0 in licks_in_trial: licks_in_trial.remove(0)
            lick_diffs = np.asarray(lick_diffs)
            if len(licks_in_trial) <= 2:
                bout_start[a] = 0
                bout_end[a] = 0
                bout_length[a] = 0
                lick_rate[a] = 0
            #We exclude lick differences greater than 300,casting them as zero, to search for our bouts (i.e. licks occuring within 300 ms)
            for k in range(0,len(lick_diffs)):
                if lick_diffs[k] > 500:
                    lick_diffs[k] = 0
            lick_diffs = list(lick_diffs)
            #We find the lick differences that pass our test and find the last lick that occurs within that bout
            nonzero_values = [i for i,x in enumerate(lick_diffs) if x != 0]
            if (len(nonzero_values) > 0):
                first_nonzero_val = nonzero_values[0]
                lick_diff_fromnonzero = lick_diffs[first_nonzero_val:]
                if 0 in lick_diff_fromnonzero:
                    last_lick = lick_diff_fromnonzero.index(0)
                else:
                    last_lick = lick_diff_fromnonzero[len(lick_diff_fromnonzero)-1]
                bout = licks_in_trial[first_nonzero_val:last_lick+1]
            #If there are more than 4 licks (i.e. enough for a 3 lick bout), and if the length of the bout is less than 2,
            #loop through all licks that could be part of a bout and try to find bouts that are greater than length 2 (i.e. 3 licks or longer)
            if len(licks_in_trial)>5:
                zeroes = [i for i, x in enumerate(lick_diffs) if x == 0]
                if len(lick_diffs)-1 in zeroes:
                    zeroes.remove(len(lick_diffs)-1)
                zeroes = [0] + zeroes
                nonzero_diffs = [(a+1)-a for b,a in enumerate(zeroes[:len(zeroes)-1])]
                #print(nonzero_diffs)
                if len(nonzero_diffs)>0:
                    max_nonzero_diffs = max(nonzero_diffs)
                    #print(max_nonzero_diffs)
                if len(bout) < 3 and len(nonzero_diffs)>0 and max_nonzero_diffs>2:
                    while len(bout) < 3:
                        for u in range(0,len(zeroes)-1): 
                            lick_diff_fromzero = lick_diffs[zeroes[u]:]
                            nonzero_values = [i for i,x in enumerate(lick_diff_fromzero) if x != 0]
                            next_nonzero_val = nonzero_values[0]
                            #print(next_nonzero_val)
                            lick_diff_fromnonzero = lick_diff_fromzero[next_nonzero_val:]
                            if 0 in lick_diff_fromnonzero:
                                last_lick = lick_diff_fromnonzero.index(0)
                                bout = licks_in_trial[next_nonzero_val:last_lick+1]
                            elif 0 not in lick_diff_fromnonzero:
                                bout = licks_in_trial[next_nonzero_val:]
            #We calculate the bout start,end,and length relative to the trial
            if len(bout) != 0:
                bout_start[a] = bout[0]-t_fxd[a]
                bout_length[a] = len(bout)
                bout_end[a] = bout[len(bout)-1]-t_fxd[a]
                lick_rate[a] = (bout_length[a]/((bout_end[a]/1000)-(bout_start[a])/1000))
                #print(bout[0])

            ##If there are fewer than 2 licks in trial, if there is an artifact (i.e. licks occurring within 50 ms of each other),
            #cast those bouts/trials as zeros.

            if bout_end[a]-bout_start[a] <= 50:
                bout_start[a] = 0
                bout_end[a] = 0
                bout_length[a] = 0
                lick_rate[a] = 0
            #Calculate first lick after reward, and convert the bout start and ends (in ms) to frames for the activity

            firstlickafterrew[a] = licks[licks>t_fxd[a]+100][0]
            #firstlickafterrew[a] = licks[licks>t_fxd[a]][1]

            bout_start_frames[a] = bout_start[a]/ms_per_frame
            bout_end_frames[a] = bout_end[a]/ms_per_frame
            if bout_start_frames[a] == 0:
                bout_start_frames[a] = 1
        ##If there are no licks in the trial, cast everything to 0
        elif len(licks_in_trial) < 2:
            bout_start[a] = 0
            bout_end[a] = 0
            bout_length[a] = 0
            lick_rate[a] = 0

            bout_start_frames[a] = 0
            bout_end_frames[a] = 0
    bout_start = np.asarray(bout_start)
    bout_end = np.asarray(bout_end)
    
    if return_more:
        return bout_start, bout_end, bout_start_frames, bout_end_frames, firstlickafterrew, latency, lick_rate, bout_length, all_licks

    return bout_start, bout_end, bout_start_frames, bout_end_frames


def plot_lick_raster_with_psth(
    mat_file: str,
    align_to: str = 'reward',
    pre_window: float = 5,
    post_window: float = 10,
    bin_size_psth: float = 0.1,
    smooth_psth: bool = True,
    psth_smoothing_sigma: float = 1.0,
    xlabel: str = 'Time (s)',
    ylabel_raster: str = 'Trial #',
    ylabel_psth: str = 'Licks/sec',
    fps: int = 15,
    plot_bouts: bool = True,
    multiple_tastant: bool = False,
    filter_pre_licknum: int = 5, 
    artifact_thresh: float = 0.100,
    normalize_histo: bool = False,
    lw = 1.0
):
    """
    Load your .mat file and plot a lick raster + PSTH, aligned to cue, reward, or first lick.
    Raster goes top down. Earlier trials are at the top. 
    Parameters
    ----------
    mat_file
        Path to the .mat file (must contain 'licks', 'cues', 'fxdpumps' and if multiple_tastant=True, 'cuesminus').
    align_to
        'cue'       → align to each cue time
        'reward'    → align to each reward delivery (fxdpumps)
        'firstlick' → align to first lick after reward (within trial)
    pre_window, post_window
        Seconds before/after the alignment event to show.
    bin_size_psth
        (s) width for PSTH bars.
    smooth_psth
        If True, applies Gaussian smoothing to the PSTH.
    psth_smoothing_sigma
        Sigma in bins for the Gaussian filter.
    xlabel, ylabel_raster, ylabel_psth
        Axis labels.
    fps
        Frames-per-second for bout overlay.
    plot_bouts
        If True, overlays bouts returned by og_lickprocessing().
    multiple_tastant
        If True, colors the vertical zero-line by tastant type (requires 'cuesminus').
    filter_pre_licknum
        If True, filters out trials with pre-reward licks.
    artifact_thresh
        (s) threshold for artifact removal. If a lick occurs within this time of the previous lick, it is removed.
    """
    # 1) load from .mat
    beh       = sio.loadmat(mat_file)
    lick_ts   = np.squeeze(beh['licks'])   / 1000.0
    cues      = np.squeeze(beh['cues'])    / 1000.0
    cues      = cues[cues > 0]
    rwd_ts    = np.squeeze(beh['fxdpumps'])/ 1000.0

    if multiple_tastant:
        alt_ts = np.squeeze(beh['cuesminus']) / 1000.0
        alt_idx, suc_idx = [], []
        for idx, t in enumerate(cues):
            (alt_idx if t in alt_ts else suc_idx).append(idx)

    # 2) pick alignment times & trial mapping
    a = align_to.lower()
    if a == 'cue':
        event_ts  = cues
        trial_map = np.arange(len(cues))
    elif a == 'reward':
        event_ts  = rwd_ts
        trial_map = np.arange(len(rwd_ts))
    elif a in ('firstlick', 'lick'):
        # get first-lick times (ms) from og_lickprocessing
        *_, firstlick_ms, _lat, _rate, _bl, _all = og_lickprocessing(mat_file, fps, return_more=True, filter_pre_licknum=filter_pre_licknum)
        fl_s_all = firstlick_ms / 1000.0
        valid    = ~np.isnan(fl_s_all)
        event_ts = fl_s_all[valid]
        trial_map = np.where(valid)[0]
    else:
        raise ValueError("align_to must be 'cue', 'reward', or 'firstlick'")

    n_trials = len(event_ts)

    # 3) build per-trial aligned lists
    aligned = []
    for row, t0 in enumerate(event_ts, start=0):
        # 1) pick all licks in your plotting window
        mask = (lick_ts >= t0 - pre_window) & (lick_ts <= t0 + post_window)

        # 2) figure out this trial’s reward time and drop everything
        #    in [rwd, rwd + artifact_thresh)
        rwd_time = rwd_ts[ trial_map[row] ]
        mask &= ~((lick_ts >= rwd_time) & (lick_ts < rwd_time + artifact_thresh))

        # 3) shift into the aligned frame and store
        rel = lick_ts[mask] - t0
        aligned.append(rel)

    # 4) figure setup
    fig, (ax_raster, ax_psth) = plt.subplots(
        2,1,
        sharex=True,
        gridspec_kw={'height_ratios':[3,1]},
        figsize=(12,8)
    )

    # 5) raster
    for row, times in enumerate(aligned, start=1):
        ax_raster.vlines(times, row-0.4, row+0.4, color='black', lw = lw)

    # zero line (colored if multis)
    # if multiple_tastant and a=='cue':
    #     for row in range(1, n_trials+1):
    #         color = 'green' if (row-1 in alt_idx) else 'blue'
    #         ax_raster.vlines(0, row-0.4, row+0.4, linestyle='--', color=color)
    # else:
    #     ax_raster.axvline(0, linestyle='--', color='blue')
    if multiple_tastant:
        for row, trial_idx in enumerate(trial_map, start=1):
            color = 'green' if trial_idx in alt_idx else 'blue'
            ax_raster.vlines(0, row-0.4, row+0.4,
                            linestyle='--', color=color, lw = lw)
    else:
        ax_raster.axvline(0, linestyle='--', color='blue', lw = lw)

    ax_raster.set_ylabel(ylabel_raster)
    ax_raster.set_ylim(0.5, n_trials+0.5)
    ax_raster.invert_yaxis()
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)

    # 6) overlay bouts
    if plot_bouts:
        bs_ms, be_ms, *_ = og_lickprocessing(mat_file, fps, return_more=False, filter_pre_licknum=filter_pre_licknum)
        bs_s, be_s = bs_ms/1000.0, be_ms/1000.0

        for row_idx, trial_idx in enumerate(trial_map, start=1):
            # absolute times
            rwd_t = rwd_ts[trial_idx]           # reward time (sec)
            evt_t = event_ts[row_idx-1]         # alignment time (sec)

            # shift to get bout relative to your alignment
            shift = rwd_t - evt_t

            start = bs_s[trial_idx] + shift
            end   = be_s[trial_idx] + shift

            # now clip to your plotting window:
            win_start = -pre_window
            win_end   = +post_window

            plot_start = max(start, win_start)
            plot_end   = min(end,   win_end)

            # only draw if there’s actually something to show
            if plot_end > plot_start:
                rect = Rectangle(
                    (plot_start, row_idx - 0.4),
                    width  = (plot_end - plot_start),
                    height = 0.8,
                    color  = 'red',
                    alpha  = 0.3
                )
                ax_raster.add_patch(rect)

    # 7) PSTH
    bins   = np.arange(-pre_window,
                       post_window+bin_size_psth,
                       bin_size_psth)

    if multiple_tastant:
        # split aligned trials
        alt_al = [aligned[i] for i, ti in enumerate(trial_map) if ti in alt_idx]
        suc_al = [aligned[i] for i, ti in enumerate(trial_map) if ti in suc_idx]

        all_alt = np.hstack(alt_al) if alt_al else np.array([])
        all_suc = np.hstack(suc_al) if suc_al else np.array([])

        cnt_alt, _ = np.histogram(all_alt, bins=bins)
        cnt_suc, _ = np.histogram(all_suc, bins=bins)

        # normalization
        if normalize_histo:
            cnt_alt = cnt_alt.astype(float) / max(cnt_alt.max(), 1)
            cnt_suc = cnt_suc.astype(float) / max(cnt_suc.max(), 1)
        else:
            cnt_alt = cnt_alt / (len(alt_al) * bin_size_psth) if alt_al else cnt_alt
            cnt_suc = cnt_suc / (len(suc_al) * bin_size_psth) if suc_al else cnt_suc

        # smoothing
        if smooth_psth:
            cnt_alt = gaussian_filter1d(cnt_alt, sigma=psth_smoothing_sigma)
            cnt_suc = gaussian_filter1d(cnt_suc, sigma=psth_smoothing_sigma)

        # plot overlaid bars
        ax_psth.bar(bins[:-1], cnt_alt,
                    width=bin_size_psth,
                    align='edge',
                    color='green', alpha=0.5,
                    label='Alt tastant')
        ax_psth.bar(bins[:-1], cnt_suc,
                    width=bin_size_psth,
                    align='edge',
                    color='blue', alpha=0.15,
                    label='Sucrose')
        ax_psth.legend()

    else:
        # single‐color PSTH
        all_sp = np.hstack(aligned)
        cnts, _ = np.histogram(all_sp, bins=bins)

        if normalize_histo:
            vals = cnts.astype(float) / max(cnts.max(),1)
        else:
            vals = cnts / (n_trials * bin_size_psth)

        if smooth_psth:
            vals = gaussian_filter1d(vals, sigma=psth_smoothing_sigma)

        ax_psth.bar(bins[:-1], vals,
                    width=bin_size_psth,
                    align='edge',
                    edgecolor='none',
                    color='black')

    # 8) finalize x-axis
    ax_raster.set_xlim(-pre_window, post_window)
    xt = np.arange(-pre_window, post_window+1, 1.0)
    ax_psth.set_xticks(xt)

    plt.tight_layout()
    return fig, ax_raster, ax_psth


def label_meals(epocs, min_pellets=5, meal_duration=600):
    ipi = [(epocs[i+1]-epocs[i]) for i in range(len(epocs)-1)]
    output = []
    meal_no = 1 
    c = 0 

    while c < len(ipi):
        pellets = ipi[c+1:c+min_pellets] #pellets = ipi[c+1:c+min_pellets]
        if len(pellets) == 0 and c == len(ipi) - 1: #if last pellet
            if ipi[c] >= meal_duration: 
                print('appending None')
                output.append(meal_no if min_pellets == 1 else None)
            break
        if all(p < meal_duration for p in pellets): 
            output.append(meal_no)
            while c < len(ipi) - 1: 
                if ipi[c+1] < meal_duration: 
                    output.append(meal_no)
                    c += 1 
                else: 
                    c += 1 
                    break
            meal_no += 1 
        else: 
            output.append(None)
            c += 1 
    t = [1 if ((epocs[1]-epocs[0] < meal_duration) and (output[0] == 1)) else None]
    t += output
    output = t

    if len(epocs) == len(output)+1:
        output.append(None)
        print('appending another final None')

    if any(isinstance(item, int) for item in output[-min_pellets:]):
        inspect = output[-min_pellets:]
        if all(isinstance(item, int) for item in inspect):
            if inspect.count(inspect[0]) == len(inspect): #if all values are the same in the arr
                pass #constitutes a full meal 
            else: #replace new meal_no's with Nones, since not enough pellets to constitute meal 
                first_mealno = inspect[0] 
                idxs = [idx for idx, val in enumerate(inspect) if val != first_mealno] 
                replace_len = len(inspect[idxs[0]:])
                inspect[idxs[0]:] = [None] * replace_len
                output[-min_pellets:] = inspect 

        else: #if None in last 5 indices, replace all to right of None with None 
            idxs = [idx for idx, val in enumerate(inspect) if val == None] #get vals to the right of Nones 
            replace_len = len(inspect[idxs[0]:])
            inspect[idxs[0]:] = [None] * replace_len
            output[-min_pellets:] = inspect 

    
    df = pd.DataFrame.from_dict({'meal_no': output, 'epoc_time':epocs}).dropna().reset_index()
    meal_onsets = []
    meal_offsets = []
    arr = range(len(df.index))
    for i in arr:
        if i == 0:
            meal_onsets.append(df['epoc_time'][i])
        elif i != 0 and i != arr[-1]:
            if df['meal_no'][i] != df['meal_no'][i-1]:
                meal_onsets.append(df['epoc_time'][i])
            if df['meal_no'][i] != df['meal_no'][i+1]:
                meal_offsets.append(df['epoc_time'][i])
        elif i == arr[-1]:
            meal_offsets.append(df['epoc_time'][i])
        
    return meal_onsets, meal_offsets


def plot_avg_lick_histo(
    mat_files: list,   
    align_to: str = 'reward',
    pre_window: float = 5,
    post_window: float = 10,
    bin_size_psth: float = 0.01,
    smooth_psth: bool = True,
    psth_smoothing_sigma: float = 1.0,
    xlabel: str = 'Time (s)',
    ylabel_raster: str = 'Trial #',
    ylabel_psth: str = 'Licks/sec',
    fps: int = 15,
    plot_bouts: bool = True,
    multiple_tastant: bool = False,
    filter_pre_licknum: int = 5, 
    artifact_thresh: float = 0.100,
    normalize_histo: bool = False,
    lw = 1.0
):
    from scipy.io import loadmat
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from matplotlib.patches import Rectangle

    all_aligned = []
    all_trials = 0
    tastant_colors = []  # track 'alt' or 'suc' trial types
    trial_origin = []
    alt_trials = []
    suc_trials = []

    for f_idx, mat_file in enumerate(mat_files):
        beh       = loadmat(mat_file)
        lick_ts   = np.squeeze(beh['licks'])   / 1000.0
        cues      = np.squeeze(beh['cues'])    / 1000.0
        cues      = cues[cues > 0]
        rwd_ts    = np.squeeze(beh['fxdpumps'])/ 1000.0

        if multiple_tastant:
            alt_ts = np.squeeze(beh['cuesminus']) / 1000.0
            alt_idx, suc_idx = [], []
            for idx, t in enumerate(cues):
                (alt_idx if t in alt_ts else suc_idx).append(idx)

        # alignment
        a = align_to.lower()
        if a == 'cue':
            event_ts  = cues
            trial_map = np.arange(len(cues))
        elif a == 'reward':
            event_ts  = rwd_ts
            trial_map = np.arange(len(rwd_ts))
        elif a in ('firstlick', 'lick'):
            *_, firstlick_ms, _lat, _rate, _bl, _all = og_lickprocessing(mat_file, fps, return_more=True, filter_pre_licknum=filter_pre_licknum)
            fl_s_all = firstlick_ms / 1000.0
            valid    = ~np.isnan(fl_s_all)
            event_ts = fl_s_all[valid]
            trial_map = np.where(valid)[0]
        else:
            raise ValueError("align_to must be 'cue', 'reward', or 'firstlick'")

        for row, t0 in enumerate(event_ts, start=0):
            mask = (lick_ts >= t0 - pre_window) & (lick_ts <= t0 + post_window)
            rwd_time = rwd_ts[ trial_map[row] ]
            mask &= ~((lick_ts >= rwd_time) & (lick_ts < rwd_time + artifact_thresh))
            rel = lick_ts[mask] - t0
            all_aligned.append(rel)
            trial_origin.append(f_idx)

            # Save tastant identity per trial (for coloring and PSTH splitting)
            if multiple_tastant:
                true_idx = trial_map[row]
                if true_idx in alt_idx:
                    tastant_colors.append('green')
                    alt_trials.append(rel)
                else:
                    tastant_colors.append('blue')
                    suc_trials.append(rel)

        all_trials += len(event_ts)

        # 4) Figure setup
    fig, (ax_raster, ax_psth) = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
        figsize=(12, 8)
    )
    # 5) Raster plot
    for row, times in enumerate(all_aligned, start=1):
        if multiple_tastant:
            color = tastant_colors[row - 1]
            ax_raster.vlines(times, row - 0.4, row + 0.4, color=color, lw=lw)
            ax_raster.vlines(0, row - 0.4, row + 0.4, linestyle='--', color=color, lw=lw)
        else:
            ax_raster.vlines(times, row - 0.4, row + 0.4, color='black', lw=lw)
    if not multiple_tastant:
        ax_raster.axvline(0, linestyle='--', color='blue', lw=lw)

    ax_raster.set_ylabel(ylabel_raster)
    ax_raster.set_ylim(0.5, all_trials + 0.5)
    ax_raster.invert_yaxis()
    ax_raster.spines['top'].set_visible(False)
    ax_raster.spines['right'].set_visible(False)

    # 6) PSTH
    bins = np.arange(-pre_window, post_window + bin_size_psth, bin_size_psth)

    if multiple_tastant:
        all_alt = np.hstack(alt_trials) if alt_trials else np.array([])
        all_suc = np.hstack(suc_trials) if suc_trials else np.array([])

        cnt_alt, _ = np.histogram(all_alt, bins=bins)
        cnt_suc, _ = np.histogram(all_suc, bins=bins)

        if normalize_histo:
            cnt_alt = cnt_alt.astype(float) / max(cnt_alt.max(), 1)
            cnt_suc = cnt_suc.astype(float) / max(cnt_suc.max(), 1)
        else:
            cnt_alt = cnt_alt / (len(alt_trials) * bin_size_psth) if alt_trials else cnt_alt
            cnt_suc = cnt_suc / (len(suc_trials) * bin_size_psth) if suc_trials else cnt_suc

        if smooth_psth:
            cnt_alt = gaussian_filter1d(cnt_alt, sigma=psth_smoothing_sigma)
            cnt_suc = gaussian_filter1d(cnt_suc, sigma=psth_smoothing_sigma)
        centers = bins[:-1] + bin_size_psth / 2

        # Convert alt/suc trials into 2D binned arrays
        def binned_array(trials, bins):
            return np.array([
                np.histogram(trial, bins=bins)[0] for trial in trials
            ])

        alt_mat = binned_array(alt_trials, bins)
        suc_mat = binned_array(suc_trials, bins)

        # Normalize to licks/sec
        if not normalize_histo:
            alt_mat = alt_mat / bin_size_psth
            suc_mat = suc_mat / bin_size_psth

        # Compute mean and SEM
        alt_mean = alt_mat.mean(axis=0) if len(alt_mat) > 0 else np.zeros(len(bins) - 1)
        alt_sem  = alt_mat.std(axis=0, ddof=1) / np.sqrt(len(alt_mat)) if len(alt_mat) > 1 else np.zeros(len(bins) - 1)

        suc_mean = suc_mat.mean(axis=0) if len(suc_mat) > 0 else np.zeros(len(bins) - 1)
        suc_sem  = suc_mat.std(axis=0, ddof=1) / np.sqrt(len(suc_mat)) if len(suc_mat) > 1 else np.zeros(len(bins) - 1)

        # Optional smoothing
        if smooth_psth:
            alt_mean = gaussian_filter1d(alt_mean, sigma=psth_smoothing_sigma)
            alt_sem  = gaussian_filter1d(alt_sem, sigma=psth_smoothing_sigma)
            suc_mean = gaussian_filter1d(suc_mean, sigma=psth_smoothing_sigma)
            suc_sem  = gaussian_filter1d(suc_sem, sigma=psth_smoothing_sigma)

        # Plot mean ± SEM
        ax_psth.plot(centers, alt_mean, color='#316dc1', lw=2, label='Alt tastant')
        ax_psth.fill_between(centers, alt_mean - alt_sem, alt_mean + alt_sem,
                            color='#316dc1', alpha=0.3)

        ax_psth.plot(centers, suc_mean, color='#424141', lw=2, label='Sucrose')
        ax_psth.fill_between(centers, suc_mean - suc_sem, suc_mean + suc_sem,
                            color='#424141', alpha=0.2)

        ax_psth.legend()

    else:
        all_sp = np.hstack(all_aligned)
        cnts, _ = np.histogram(all_sp, bins=bins)
        if normalize_histo:
            vals = cnts.astype(float) / max(cnts.max(), 1)
        else:
            vals = cnts / (all_trials * bin_size_psth)
        if smooth_psth:
            vals = gaussian_filter1d(vals, sigma=psth_smoothing_sigma)
        ax_psth.bar(bins[:-1], vals,
                    width=bin_size_psth,
                    align='edge',
                    edgecolor='none',
                    color='black')

    ax_psth.set_ylabel(ylabel_psth)
    ax_psth.set_xticks(np.arange(-pre_window, post_window + 1, 1.0))
    ax_psth.set_xlabel(xlabel)
    ax_raster.set_xlim(-pre_window, post_window)

    plt.tight_layout()
    return fig, ax_raster, ax_psth
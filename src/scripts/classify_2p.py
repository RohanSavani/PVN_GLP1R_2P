import numpy as np
from scipy.stats import ranksums
import scipy.stats as stats


# Function to perform a circular shift
def circular_shift(Y, N, s):
    return np.concatenate((Y[s:], Y[:s]))

# Function to wrap data
def wrap_data(original_data, t):
    if t == 0:
        return original_data
    else:
        # break data around index t and swap pre-t and post-t chunks around 
        return np.concatenate((original_data[t:], original_data[:t]))

def constructWnull(Y_in, pre_event, post_event, wrapping_period=None, B=1000):
    # dimensions [neurons, trials, timepoints])
    
    # Wrap on all the data if not specified
    if wrapping_period is None:
        wrapping_period = range(Y_in.shape[2])

    # Focus on relevant data
    Y = Y_in[:, :, wrapping_period]

    n_c, n_tr, timepoints = Y.shape # cells x trials x time 

    Wib_tilde = np.zeros(B) # store H0 statistic from all permutations 
    for b in range(B): # for each permutation 
        stat_sum = 0
        i_star = np.random.randint(0, n_c) # take a random cell 
        for j in range(n_tr): # for each trial... 
            wrap_q = np.random.randint(0, timepoints) # random int from [0, time)
            Yij_tilde = wrap_data(Y[i_star, j, :], wrap_q) # shift array 
            Yij_tilde_pre = Yij_tilde[pre_event] # get baseline data
            Yij_tilde_post = Yij_tilde[post_event] # get stimulus data 
            stat_sum += ranksums(Yij_tilde_post, Yij_tilde_pre).statistic
            Wib_tilde[b] = stat_sum / n_tr  # normalize

    WSRT_randomization = Wib_tilde 

    return WSRT_randomization


def constructWobs(aligned_data, baseline_data, pre_event=slice(75, 105), post_event=slice(75, 105)):
    # Calculate observed test statistics
    # Take in aligned data (lick onset at 75)
    all_W = [] 
    for i, Y in enumerate(aligned_data): 
        n_c, n_tr, timepoints = Y.shape
        baseline = baseline_data[i]
        WSRT_obs = np.zeros(n_c) 
        for i in range(n_c):
            Wi_tilde = 0
            for j in range(n_tr):
                Yij_pre = baseline[i, j, pre_event]
                Yij_post = Y[i, j, post_event]
                Wi_tilde += ranksums(Yij_post, Yij_pre).statistic
            WSRT_obs[i] = Wi_tilde / n_tr  # 
        all_W.append(WSRT_obs)

    return all_W 


# Function to get p-values based on the Wilcoxon statistics
def alg1_pvals(Wi_obs, Wi_rand):
    n_c = len(Wi_obs)
    p_vals = np.zeros(n_c)
    resp_type = np.full(n_c, 'X')

    for i in range(n_c):
        p_val_1 = (np.sum(Wi_rand >= Wi_obs[i]) + 1) / (len(Wi_rand) + 1)
        p_val_2 = (np.sum(Wi_rand <= Wi_obs[i]) + 1) / (len(Wi_rand) + 1)

        p_vals[i] = 2 * min(p_val_1, p_val_2)
        resp_type[i] = '+' if p_val_1 < p_val_2 else '-'

    return p_vals, resp_type


def circ_shift(good_data, aligned_data, alpha=0.05, base_pre = slice(75, 105), base_post = slice(150, 180),
              stim_pre = slice(75, 105), stim_post = slice(75, 105)): 
    null_stats = []
    for rec_data in good_data:
        null_stats.append(constructWnull(rec_data, base_pre, base_post, B = 1000))
    null_stats = np.concatenate(np.asarray(null_stats), axis = 0)
    Wobs = constructWobs(aligned_data, good_data, pre_event=stim_pre, post_event=stim_post)
    Wobs = np.asarray([x for y in Wobs for x in y])
    pvals, response_types = alg1_pvals(Wobs, null_stats)
    print(f"Responsive: {sum(pvals <= alpha)}")
    print(f"Unresponsive: {sum(pvals > alpha)}")
    print(f"Total: {len(pvals)}")
    print(f"Inhibited: {sum(response_types[pvals <= alpha] == '-')}")
    print(f"Activated: {sum(response_types[pvals <= alpha] == '+')}")
    sucrose_activated_pvals = np.ones(pvals.shape)
    sucrose_inhibited_pvals = np.ones(pvals.shape)
    for idx, val in enumerate(pvals): 
        if response_types[idx] == '+':
            sucrose_activated_pvals[idx] = val
        elif response_types[idx] == '-':
            sucrose_inhibited_pvals[idx] = val
    return sucrose_activated_pvals, sucrose_inhibited_pvals, pvals, null_stats, Wobs

def calc_responsive_cells(all_good_data, all_aligned_data, pre_baseline_time=30, post_baseline_time=105, pre_response_time=75, post_response_time=150, alpha=0.05, method='t', explicit=False):
    all_baseline_data = [] 
    num_recordings = len(all_good_data)
    if len(all_aligned_data) != len(all_good_data):
        print("Error: aligned data and good data must be the same length")

    for i in range(num_recordings): 
        doi = all_good_data[i]
        baseline_data = doi[:, :, pre_baseline_time:post_baseline_time] # get 5 second baseline data
        all_baseline_data.append(baseline_data)

    all_response_data = []
    for i in range(num_recordings): 
        doi = all_aligned_data[i]
        response_data = doi[:, :, pre_response_time:post_response_time] #get 5 seconds after lick initiation 
        all_response_data.append(response_data)

    all_sucrose_activated_pvals = [] 
    all_sucrose_inhibited_pvals = [] 
    all_sucrose_responsive_pvals = [] 

    for i in range(num_recordings):
        baselines = all_baseline_data[i]
        responses = all_response_data[i]
        
        for u in range(responses.shape[0]): #for each cell 
            baseline_pertrial_mean = np.nanmean(baselines[u], axis = 1)
            post_pertrial_mean = np.nanmean(responses[u], axis = 1)  

            if method == 't':
                all_sucrose_inhibited_pvals.append(
                    stats.ttest_rel(baseline_pertrial_mean, post_pertrial_mean, alternative = "greater")[1])
                all_sucrose_activated_pvals.append(
                    stats.ttest_rel(baseline_pertrial_mean, post_pertrial_mean, alternative = "less")[1])
                all_sucrose_responsive_pvals.append(
                    stats.ttest_rel(baseline_pertrial_mean, post_pertrial_mean, alternative = "two-sided")[1]) 

            if method == 'wilcoxon':
                all_sucrose_inhibited_pvals.append(
                    stats.wilcoxon(baseline_pertrial_mean, post_pertrial_mean, alternative = "greater")[1])
                all_sucrose_activated_pvals.append(
                    stats.wilcoxon(baseline_pertrial_mean, post_pertrial_mean, alternative = "less")[1])
                all_sucrose_responsive_pvals.append(
                    stats.wilcoxon(baseline_pertrial_mean, post_pertrial_mean, alternative = "two-sided")[1]) 
            if method == "validate":
                if stats.shapiro(baseline_pertrial_mean - post_pertrial_mean)[1] < 0.05:
                    all_sucrose_inhibited_pvals.append(
                        stats.wilcoxon(baseline_pertrial_mean, post_pertrial_mean, alternative = "greater")[1])
                    all_sucrose_activated_pvals.append(
                        stats.wilcoxon(baseline_pertrial_mean, post_pertrial_mean, alternative = "less")[1])
                    all_sucrose_responsive_pvals.append(
                        stats.wilcoxon(baseline_pertrial_mean, post_pertrial_mean, alternative = "two-sided")[1]) 
                else:
                    all_sucrose_inhibited_pvals.append(
                        stats.ttest_rel(baseline_pertrial_mean, post_pertrial_mean, alternative = "greater")[1])
                    all_sucrose_activated_pvals.append(
                        stats.ttest_rel(baseline_pertrial_mean, post_pertrial_mean, alternative = "less")[1])
                    all_sucrose_responsive_pvals.append(
                        stats.ttest_rel(baseline_pertrial_mean, post_pertrial_mean, alternative = "two-sided")[1]) 
                if explicit:
                    print(stats.shapiro(baseline_pertrial_mean - post_pertrial_mean)[1])

    significant_sucrose_activated = [val for val in all_sucrose_activated_pvals if val < alpha]
    significant_sucrose_inhibited = [val for val in all_sucrose_inhibited_pvals if val < alpha]
    print(f"{len(significant_sucrose_activated)} cells significantly activated ({len(significant_sucrose_activated) / len(all_sucrose_activated_pvals) * 100 : .3f}%)")
    print(f"{len(significant_sucrose_inhibited)} cells significantly inhibited ({len(significant_sucrose_inhibited) / len(all_sucrose_activated_pvals) * 100 : .3f}%)")
    print(f"{len(significant_sucrose_activated) + len(significant_sucrose_inhibited)} cells significantly responsive  ({(len(significant_sucrose_activated) + len(significant_sucrose_inhibited)) / len(all_sucrose_activated_pvals) * 100 : .3f}%)")
    print(f"{len(all_sucrose_activated_pvals) - (len(significant_sucrose_inhibited) + len(significant_sucrose_activated))} cells not responsive  ({(len(all_sucrose_activated_pvals) - (len(significant_sucrose_inhibited) + len(significant_sucrose_activated))) / len(all_sucrose_activated_pvals) * 100 : .3f}%)")
    print(f"{len(all_sucrose_activated_pvals)} total cells analyzed")

    return all_sucrose_activated_pvals, all_sucrose_inhibited_pvals, all_sucrose_responsive_pvals


def get_activated_inhibited_cells(all_good_data, all_aligned_data , all_sucrose_activated_pvals, 
                                  all_sucrose_inhibited_pvals, alpha=0.05):
    #Get sucrose inhibited or activated cells, aligned to fist lick 
    all_doi = []
    all_doi2 = []
    num_recordings = len(all_good_data)
    if len(all_aligned_data) != len(all_good_data):
        print("Error: aligned data and good data must be the same length")
    for i in range(num_recordings):
        doi = all_aligned_data[i]
        for j in range(doi.shape[0]):
            all_doi.append(doi[j, :, :])
        doi2 = all_good_data[i]
        for j in range(doi2.shape[0]):
            all_doi2.append(doi2[j, :, :])
        
    sig_suc_activated = []
    sig_suc_inhibited = []
    sig_suc_activated_baseline = []
    sig_suc_inhibited_baseline = []
    for i in range(len(all_sucrose_activated_pvals)):
        if all_sucrose_activated_pvals[i] < alpha:
            sig_suc_activated.append(all_doi[i])
            sig_suc_activated_baseline.append(all_doi2[i])
        if all_sucrose_inhibited_pvals[i] < alpha:
            sig_suc_inhibited.append(all_doi[i])
            sig_suc_inhibited_baseline.append(all_doi2[i])

    for i in range(len(sig_suc_activated)): #get trial-averaged dF/F per responsive cell 
        sig_suc_activated[i] = np.nanmean(sig_suc_activated[i], axis = 0)
        sig_suc_activated_baseline[i] = np.nanmean(sig_suc_activated_baseline[i], axis = 0)

    for i in range(len(sig_suc_inhibited)): #get trial-averaged dF/F per responsive cell 
        sig_suc_inhibited[i] = np.nanmean(sig_suc_inhibited[i], axis = 0)
        sig_suc_inhibited_baseline[i] = np.nanmean(sig_suc_inhibited_baseline[i], axis = 0)

    if (len(sig_suc_activated) > 0):
        t = np.zeros((len(sig_suc_activated), sig_suc_activated[0].shape[0]))
        for i in range(len(sig_suc_activated)):
            t[i, :] = sig_suc_activated[i]
        sig_suc_activated1 = t
    
        t = np.zeros((len(sig_suc_activated_baseline), sig_suc_activated_baseline[0].shape[0]))
        for i in range(len(sig_suc_activated_baseline)):
            t[i, :] = sig_suc_activated_baseline[i]
        sig_suc_activated_baseline1 = t
    else: 
        sig_suc_activated1 = []
        sig_suc_activated_baseline1 = []

    if (len(sig_suc_inhibited) > 0):
        t = np.zeros((len(sig_suc_inhibited), sig_suc_inhibited[0].shape[0]))
        for i in range(len(sig_suc_inhibited)):
            t[i, :] = sig_suc_inhibited[i]
        sig_suc_inhibited1 = t 
    
        t = np.zeros((len(sig_suc_inhibited_baseline), sig_suc_inhibited_baseline[0].shape[0]))
        for i in range(len(sig_suc_inhibited_baseline)):
            t[i, :] = sig_suc_inhibited_baseline[i]
        sig_suc_inhibited_baseline1 = t 
    else:
        sig_suc_inhibited1 = []
        sig_suc_inhibited_baseline1 = []

    return sig_suc_activated1, sig_suc_activated_baseline1, sig_suc_inhibited1, sig_suc_inhibited_baseline1
import numpy as np
from scipy.stats import ranksums


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
        i_star = np.random.randint(0, n_c) # take a random cell 
        for j in range(n_tr): # for each trial... 
            wrap_q = np.random.randint(0, timepoints) # random int from [0, time)
            Yij_tilde = wrap_data(Y[i_star, j, :], wrap_q) # shift array 
            Yij_tilde_pre = Yij_tilde[pre_event] # get baseline data
            Yij_tilde_post = Yij_tilde[post_event] # get stimulus data 
            Wib_tilde[b] += ranksums(Yij_tilde_post, Yij_tilde_pre).statistic # calculate 

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
            WSRT_obs[i] = Wi_tilde
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


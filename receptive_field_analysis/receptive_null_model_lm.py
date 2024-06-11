import numpy as np
import pickle as pkl

# Load the data
with open('Data_50trials/lm_sums_trials.pkl', 'rb') as f:
    lm_all_trials = pkl.load(f)

with open('Data_50trials/lm_sums.pkl', 'rb') as f:
    lm_all = pkl.load(f)

n_neurons_lm = 7414
n_timesteps = 1000
n_trials = 50
n_rows = 13
n_cols = 11

# normalize v1 with the number of trials and time bins to get an average firing rate per neuron, multiply by 1000 to get Hz
lm_all = lm_all / (n_trials * n_timesteps) * 1000
lm_all_trials = lm_all_trials / (n_timesteps) * 1000

n_null_models = 1000
null_chi2_lm = np.zeros((n_neurons_lm, n_null_models))
real_chi2_lm = np.zeros((n_neurons_lm))

for neuron in range(n_neurons_lm):
    neuron_rf = lm_all[:,:,neuron] # select the neuron
    neuron_rf_trials = lm_all_trials[:,:,:,neuron] # select the neuron

    # set to median all the values below a th
    median = np.median(neuron_rf)
    th = median - 2*np.std(neuron_rf)
    neuron_rf[neuron_rf < th] = median

    # compute expected value
    expected_value = np.mean(neuron_rf)
    # compute residuals
    residuals = (neuron_rf - expected_value)**2 / expected_value

    # compute chi^2
    chi2 = np.sum(residuals)
    real_chi2_lm[neuron] = chi2

    # null model
    for k in range(n_null_models):
        # create a matrix to store the shuffled RFs
        shuffled_rf = np.zeros((n_rows, n_cols))

        for trial in range(n_trials):
            # shuffle the RF
            rf_flatten = neuron_rf_trials[:,:,trial].flatten()
            rf_shuffled = np.random.shuffle(rf_flatten)
            rf_shuffled = rf_flatten.reshape(n_rows, n_cols)

            # add the shuffled RF to the matrix
            shuffled_rf += rf_shuffled

        # average over trials
        shuffled_rf /= n_trials

        # set to the median all the values below a th
        median = np.median(shuffled_rf)
        th = median - 2*np.std(shuffled_rf)
        shuffled_rf[shuffled_rf < th] = median

        # compute the expected value
        expected_value = np.mean(shuffled_rf)
        # compute residuals
        residuals = (shuffled_rf - expected_value)**2 / expected_value

        # compute chi^2
        chi2_null = np.sum(residuals)
        null_chi2_lm[neuron, k] = chi2_null

# save the data
with open('Data_50trials/null_chi2_lm_trials.pkl', 'wb') as f:
    pkl.dump(null_chi2_lm, f)

with open('Data_50trials/real_chi2_lm_trials.pkl', 'wb') as f:
    pkl.dump(real_chi2_lm, f)
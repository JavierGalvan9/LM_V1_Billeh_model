import numpy as np
import pickle as pkl

# Load the data
with open('Data_50trials/v1_sums_trials.pkl', 'rb') as f:
    v1_sums_trials = pkl.load(f)

with open('Data_50trials/v1_sums.pkl', 'rb') as f:
    v1_sums = pkl.load(f)

n_neurons_v1 = 51978
n_timesteps = 1000
n_trials = 50
n_rows = 13
n_cols = 11

# normalize v1 with the number of trials and time bins to get an average firing rate per neuron, multiply by 1000 to get Hz
v1_sums = v1_sums / (n_trials * n_timesteps) * 1000
v1_sums_trials = v1_sums_trials / (n_timesteps) * 1000

n_null_models = 1000
null_chi2_v1 = np.zeros((n_neurons_v1, n_null_models))
real_chi2_v1 = np.zeros((n_neurons_v1))

for neuron in range(n_neurons_v1):
    neuron_rf = v1_sums[:,:,neuron] # select the neuron
    neuron_rf_trials = v1_sums_trials[:,:,:,neuron] # select the neuron

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
    real_chi2_v1[neuron] = chi2

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
        null_chi2_v1[neuron, k] = chi2_null

# save the data
with open('Data_50trials/null_chi2_v1_trials.pkl', 'wb') as f:
    pkl.dump(null_chi2_v1, f)

with open('Data_50trials/real_chi2_v1_trials.pkl', 'wb') as f:
    pkl.dump(real_chi2_v1, f)
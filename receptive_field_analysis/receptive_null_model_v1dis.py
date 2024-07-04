import numpy as np
import pickle as pkl

# Load the data
with open('Data_disconnected/v1_sums.pkl', 'rb') as f:
    v1_sums_dis = pkl.load(f)

with open('Data_disconnected/v1_sums_trials.pkl', 'rb') as f:
    v1_sums_trials_dis = pkl.load(f)

n_neurons_v1 = 51978
n_timesteps = 1000
n_trials = 50
n_rows = 13
n_cols = 11

# normalize v1 with the number of trials and time bins to get an average firing rate per neuron, multiply by 1000 to get Hz
v1_sums_dis = v1_sums_dis / (n_trials * n_timesteps) * 1000
v1_sums_trials_dis = v1_sums_trials_dis / (n_timesteps) * 1000

n_shuffles = 1000
# Now for V1 disconnected
null_chi2_v1_dis = np.zeros((n_neurons_v1, n_shuffles))
real_chi2_v1_dis = np.zeros((n_neurons_v1))

for neuron in range(n_neurons_v1):
    neuron_rf = np.copy(v1_sums_dis[:,:,neuron]) # select the neuron
    neuron_rf_trials = v1_sums_trials_dis[:,:,:,neuron] # select the neuron

    percentile_5 = np.percentile(neuron_rf, 2.5)
    percentile_95 = np.percentile(neuron_rf, 97.5)
    percentile_median = np.median(neuron_rf[(neuron_rf > percentile_5) & (neuron_rf < percentile_95)])
    # set 0.0 values to percentile median
    neuron_rf[np.round(neuron_rf) == 0.0] = percentile_median

    # compute expected value
    expected_value = np.median(neuron_rf)
    th = expected_value - 5*np.std(neuron_rf)
    neuron_rf[neuron_rf < th] = expected_value

    # compute residuals
    residuals = (neuron_rf - expected_value)**2 / expected_value
    max_residuals = np.max(residuals)

    # compute chi^2
    real_chi2_v1_dis[neuron] = np.sum(residuals)

    for i in range(n_shuffles):
        # create a matrix to store the shuffled RFs
        shuffled_rf = np.zeros((n_rows, n_cols))

        flattened_arr = neuron_rf_trials.flatten()
        bootstrap_sample = np.random.choice(flattened_arr, size=flattened_arr.size, replace=True)
        bootstrap_arr = bootstrap_sample.reshape(neuron_rf_trials.shape)

        shuffled_rf = np.mean(bootstrap_arr, axis = 2)

        # set to the median all the values below a th
        percentile_5 = np.percentile(shuffled_rf, 2.5)
        percentile_95 = np.percentile(shuffled_rf, 97.5)
        percentile_median = np.median(shuffled_rf[(shuffled_rf > percentile_5) & (shuffled_rf < percentile_95)])
        # set 0.0 values to percentile median
        shuffled_rf[np.round(shuffled_rf) == 0.0] = percentile_median

        # compute expected value
        expected_value = np.median(shuffled_rf)
        th = expected_value - 5*np.std(shuffled_rf)
        shuffled_rf[shuffled_rf < th] = expected_value

        # compute residuals
        residuals = (shuffled_rf - expected_value)**2 / expected_value

        # compute chi^2
        null_chi2_v1_dis[neuron, i] = np.sum(residuals)

# save the data
with open('Data_disconnected/null_chi2_v1_dis_trials_bootstrap.pkl', 'wb') as f:
    pkl.dump(null_chi2_v1_dis, f)

with open('Data_disconnected/real_chi2_v1_dis_trials_bootstrap.pkl', 'wb') as f:
    pkl.dump(real_chi2_v1_dis, f)
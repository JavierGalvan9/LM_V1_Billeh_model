import numpy as np
import pickle as pkl 
import skimage
from skimage.measure import label, regionprops

# load the data
with open('Data_connected/lm_sums_trials.pkl', 'rb') as f:
    lm_sums_trials = pkl.load(f)

with open('Data_connected/lm_sums.pkl', 'rb') as f:
    lm_sums = pkl.load(f)

n_neurons_lm = 7414
n_timesteps = 1000
n_trials = 50
n_rows = 13
n_cols = 11

# normalize lm with the number of trials and time bins to get an average firing rate per neuron, multiply by 1000 to get Hz
lm_sums = lm_sums / (n_trials * n_timesteps) * 1000
lm_sums_trials = lm_sums_trials / (n_timesteps) * 1000

n_shuffles = 1000
real_chi2_lm = np.zeros((n_neurons_lm))
null_chi2_lm = np.zeros((n_neurons_lm, n_shuffles))

for neuron in range(n_neurons_lm):
    neuron_rf = np.copy(lm_sums[:,:,neuron]) # select the neuron
    neuron_rf_trials = lm_sums_trials[:,:,:,neuron] # select the neuron

    # set 0.0 values to percentile median
    expected_value = np.median(neuron_rf)
    neuron_rf[np.round(neuron_rf) == 0.0] = expected_value

    expected_value = np.median(neuron_rf)
    mad = np.median(np.abs(neuron_rf - expected_value))

    # obtain list of positions above and below the std
    above_std = neuron_rf[neuron_rf > expected_value + mad*3.5]
    below_std = neuron_rf[neuron_rf < expected_value - mad*3.5]
        
    above_std_map_binary = np.zeros(neuron_rf.shape)
    below_std_map_binary = np.zeros(neuron_rf.shape)
    # merge
    above_below_std = np.concatenate((above_std, below_std))

    if len(above_std) > 0:
        for i in range(len(above_std)):
            idx = np.where(neuron_rf == above_std[i])
            above_std_map_binary[idx] = 1

        # labelling connected components
        above_std_map = skimage.measure.label(above_std_map_binary)
        # get the largest connected component of each map
        above_std_map = (above_std_map == np.argmax(np.bincount(above_std_map.flat)[1:]) + 1)
        # get the list of positions
        above_std_positions_largest = np.where(above_std_map == 1)

    else:
        above_std_positions_largest = [[]]  

    if len(below_std) > 0:
        for i in range(len(below_std)):
            idx = np.where(neuron_rf == below_std[i])
            below_std_map_binary[idx] = 1

        below_std_map = skimage.measure.label(below_std_map_binary)
        below_std_map = (below_std_map == np.argmax(np.bincount(below_std_map.flat)[1:]) + 1)
        below_std_positions_largest = np.where(below_std_map == 1)

    else:
        below_std_positions_largest = [[]]

    if len(below_std_positions_largest[0]) > len(above_std_positions_largest[0]):
        inverse = True
    else:
        inverse = False

    if inverse:
        # obtain lists of the positions of all the connected components
        connected_components = label(below_std_map_binary)

        # Identify the largest connected component
        largest_cc_label = max(regionprops(connected_components), key=lambda r: r.area).label

        # Create a mask for the largest connected component and set it to 0
        below_std_map_binary[connected_components == largest_cc_label] = 0

        mask = (below_std_map_binary == 1)
        outlier_values = neuron_rf[mask]

        if len(outlier_values) > 0:
            threshold = np.max(outlier_values)
        else:
            threshold = 0

        # set values below or equal to threshold to the median
        positions_to_change = np.where(neuron_rf <= threshold)
        neuron_rf[neuron_rf <= threshold] = expected_value

    else:
        connected_components = label(below_std_map_binary)

        mask = (below_std_map_binary == 1)
        outlier_values = neuron_rf[mask]

        if len(outlier_values)>0:
            threshold = np.max(outlier_values)
        else:
            threshold = 0

        # set values below or equal to threshold to the median
        positions_to_change = np.where(neuron_rf <= threshold)
        neuron_rf[neuron_rf <= threshold] = expected_value

    # compute residuals
    residuals = (neuron_rf - expected_value)**2 / expected_value
    real_chi2_lm[neuron] = np.sum(residuals)

    # set positions_to_change to expected_value of each trial in neuron_rf_trials across all the trials
    # if there are positions_to_change
    if len(positions_to_change[0]) > 0:
        expected_values_trials = np.median(neuron_rf_trials, axis=(0,1))
        for i in range(neuron_rf_trials.shape[2]):
            # set 0.0 values to expected value
            neuron_rf_trials[neuron_rf_trials[:,:,i] == 0.0, i] = expected_values_trials[i]
            # recompute expected value
            expected_value = np.median(neuron_rf_trials[:,:,i])
            neuron_rf_trials[positions_to_change[0], positions_to_change[1], i] = expected_value

    for i in range(n_shuffles):
        # create a matrix to store the shuffled RFs
        shuffled_rf = np.zeros((n_rows, n_cols))

        flattened_arr = neuron_rf_trials.flatten()
        bootstrap_sample = np.random.choice(flattened_arr, size=flattened_arr.size, replace=True)
        bootstrap_arr = bootstrap_sample.reshape(neuron_rf_trials.shape)

        shuffled_rf = np.mean(bootstrap_arr, axis = 2)

        # set 0.0 values to percentile median
        expected_value = np.median(shuffled_rf)
        shuffled_rf[np.round(shuffled_rf) == 0.0] = expected_value

        expected_value = np.median(shuffled_rf)

        # compute residuals
        residuals = (shuffled_rf - expected_value)**2 / expected_value

        # compute chi^2
        null_chi2_lm[neuron, i] = np.sum(residuals)

# save the data
with open('Data_connected/null_chi2_lm_bootstrap.pkl', 'wb') as f:
    pkl.dump(null_chi2_lm, f)

with open('Data_connected/real_chi2_lm_bootstrap.pkl', 'wb') as f:
    pkl.dump(real_chi2_lm, f)
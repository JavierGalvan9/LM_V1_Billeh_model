import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import pandas as pd
import pickle as pkl
from math import pi
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "Model_utils"))
import other_billeh_utils


class IndividualStiffRegularizer(Layer):
    def __init__(self, strength, network, penalize_relative_change=False, recurrent_weights=True, source_area='lm', 
                 initial_values=None, dtype=tf.float32):
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype
        self._penalize_relative_change = penalize_relative_change

        if initial_values is not None:
            initial_value = np.copy(initial_values).astype(np.float32)
        else:
            # Compute voltage scale
            voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
            # Get the initial weights and properly scale them down
            if recurrent_weights:
                indices = network["synapses"]["indices"]
                initial_value = np.array(network["synapses"]["weights"], dtype=np.float32)
            else:
                indices = network['interarea_synapses'][source_area]["indices"]
                initial_value = np.array(network['interarea_synapses'][source_area]["weights"], dtype=np.float32)

            # Scale initial values by the voltage scale of the node IDs
            voltage_scale_node_ids = voltage_scale[network['node_type_ids'][indices[:, 0]]]
            initial_value /= voltage_scale_node_ids

        if self._penalize_relative_change:
            epsilon = np.float32(1e-2)
            denominator = np.maximum(np.abs(initial_value), epsilon)
            self._denominator = tf.constant(denominator, dtype=tf.float32)

        self._target_mean_weights = tf.constant(initial_value, dtype=tf.float32)
    
    @tf.function(jit_compile=True)
    def __call__(self, x):

        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        if self._penalize_relative_change:
            # return self._strength * tf.reduce_mean(tf.abs(x - self._initial_value))
            relative_deviation = (x - self._target_mean_weights) / self._denominator
            # Penalize the relative deviation
            reg_loss = tf.sqrt(tf.reduce_mean(tf.square(relative_deviation)))
        else:
            reg_loss = tf.reduce_mean(tf.square(x - self._target_mean_weights))
        
        return tf.cast(reg_loss, dtype=self._dtype) * self._strength
    

class StiffRegularizer(Layer):
    def __init__(self, strength, network, penalize_relative_change=False, recurrent_weights=True, source_area='lm', 
                 initial_values=None, dtype=tf.float32):
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype
        self._penalize_relative_change = penalize_relative_change
        # Compute voltage scale
        voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
        # Get the initial weights and properly scale them down
        if recurrent_weights:
            indices = network["synapses"]["indices"]
            initial_value = np.array(network["synapses"]["weights"], dtype=np.float32)
            edge_type_ids = network['synapses']['edge_type_ids']
        else:
            indices = network['interarea_synapses'][source_area]["indices"]
            initial_value = np.array(network['interarea_synapses'][source_area]["weights"], dtype=np.float32)
            edge_type_ids = network['interarea_synapses'][source_area]["edge_type_ids"]

        # Scale initial values by the voltage scale of the node IDs
        initial_value /= voltage_scale[network['node_type_ids'][indices[:, 0]]]

        if initial_values is not None:
            initial_value = np.copy(initial_values).astype(np.float32)

        # Find unique values and their first occurrence indices
        unique_edge_types, self.idx = np.unique(edge_type_ids, return_inverse=True)
        # Sort first_occurrence_indices to maintain the order of first appearances
        self.num_unique = unique_edge_types.shape[0]
        sum_weights = np.bincount(self.idx, weights=initial_value, minlength=self.num_unique)
        count_weights = np.bincount(self.idx, minlength=self.num_unique)
        initial_mean_weights = sum_weights / count_weights
        # Determine target mean weights
        if self._penalize_relative_change:
            epsilon = np.float32(1e-4)
            denominator = np.maximum(np.abs(initial_mean_weights), epsilon)
            self._denominator = tf.constant(denominator, dtype=tf.float32)

        self.idx = tf.constant(self.idx, dtype=tf.int32)
        self.num_unique = tf.constant(self.num_unique, dtype=tf.int32)
        self._target_mean_weights = tf.constant(initial_mean_weights, dtype=tf.float32)
    
    @tf.function(jit_compile=True)
    def __call__(self, x):

        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        mean_edge_type_weights = tf.math.unsorted_segment_mean(x, self.idx, self.num_unique)
        if self._penalize_relative_change:
            # return self._strength * tf.reduce_mean(tf.abs(x - self._initial_value))
            relative_deviation = (mean_edge_type_weights - self._target_mean_weights) / self._denominator
            # Penalize the relative deviation
            reg_loss = tf.sqrt(tf.reduce_mean(tf.square(relative_deviation)))
        else:
            reg_loss = tf.reduce_mean(tf.square(mean_edge_type_weights - self._target_mean_weights))
        
        return tf.cast(reg_loss, dtype=self._dtype) * self._strength


class L2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, network, penalize_relative_change=False, recurrent_weights=True, source_area='lm', 
                 initial_values=None, dtype=tf.float32):
        super().__init__()
        self._strength = strength
        self._dtype = dtype
        
        if penalize_relative_change:
            # Compute voltage scale
            voltage_scale = (network['node_params']['V_th'] - network['node_params']['E_L']).astype(np.float32)
            # Get the initial weights and properly scale them down
            if recurrent_weights:
                indices = network["synapses"]["indices"]
                initial_value = np.array(network["synapses"]["weights"], dtype=np.float32)
            else:
                indices = network['interarea_synapses'][source_area]["indices"]
                initial_value = np.array(network['interarea_synapses'][source_area]["weights"], dtype=np.float32)

            # Scale initial values by the voltage scale of the node IDs
            voltage_scale_node_ids = voltage_scale[network['node_type_ids'][indices[:, 0]]]
            initial_value /= voltage_scale_node_ids

            # using the edge_type ids group calculate the mean weight of each type of edge in the network and then create a constant with same shape as weights and with each value corresponding to the populations mean
            # Calculate mean weights for each edge type
            if recurrent_weights:
                edge_type_ids = np.array(network['synapses']['edge_type_ids'])
            else:
                edge_type_ids = np.array(network['interarea_synapses'][source_area]["edge_type_ids"])

            if initial_values is not None:
                initial_value = np.copy(initial_values).astype(np.float32)

            unique_edge_type_ids, inverse_indices = np.unique(edge_type_ids, return_inverse=True)
            mean_weights = np.array([np.mean(initial_value[edge_type_ids == edge_type_id]) for edge_type_id in unique_edge_type_ids])
            # Create target mean weights array based on the edge type indices
            self._target_mean_weights = tf.constant(mean_weights[inverse_indices], dtype=tf.float32)
            epsilon = tf.constant(1e-4, dtype=tf.float32)  # A small constant to avoid division by zero
            self._target_mean_weights = tf.maximum(tf.abs(self._target_mean_weights), epsilon)
        else:
            self._target_mean_weights = None

    @tf.function(jit_compile=True)
    def __call__(self, x):

        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)
            
        if self._target_mean_weights is None:
            return tf.cast(self._strength * tf.reduce_mean(tf.square(x)), dtype=self._dtype)
        else:
            relative_deviation = x / self._target_mean_weights
            mse = self._strength * tf.reduce_mean(tf.square(relative_deviation))
            return tf.cast(mse, dtype=self._dtype)


def spike_trimming(spikes, pre_delay=50, post_delay=50, trim=True):
    pre = pre_delay or 0
    if trim:
        post = -post_delay if post_delay else None
        spikes = spikes[:, pre:post, :]
    else:
        spikes = spikes[:, pre:, :]
    return spikes

def process_neuropixels_data(area='v1', path=''):
    # Load data
    neuropixels_data_path = f'Neuropixels_data/cortical_metrics_1.4.csv'
    df_all = pd.read_csv(neuropixels_data_path, sep=",")
    billeh_to_neuropixels_area_mapping = {'v1':'VISp', 'lm':'VISl'}
    # Exc and PV have sufficient number of cells, so we'll filter out non-V1 Exc and PV.
    # SST and VIP are small populations, so let's keep also non-V1 neurons
    exclude = (df_all["cell_type"].isnull() | df_all["cell_type"].str.contains("EXC") | df_all["cell_type"].str.contains("PV")) \
            & (df_all["ecephys_structure_acronym"] != billeh_to_neuropixels_area_mapping[area])
    df = df_all[~exclude]
    print(f"Original: {df_all.shape[0]} cells,   filtered: {df.shape[0]} cells")
    # Some cells have very large values of RF. They are likely not-good fits, so ignore.
    df.loc[(df["width_rf"] > 100), "width_rf"] = np.nan
    df.loc[(df["height_rf"] > 100), "height_rf"] = np.nan
    # Save the processed table
    df.to_csv(f'Neuropixels_data/{area}_OSI_DSI_DF.csv', sep=" ", index=False)

def neuropixels_cell_type_to_cell_type(pop_name):
    # Convert pop_name in the neuropixels cell type to cell types. E.g, 'EXC_L23' -> 'L2/3 Exc', 'PV_L5' -> 'L5 PV'
    layer = pop_name.split('_')[1]
    class_name = pop_name.split('_')[0]
    if "2" in layer:
        layer = "L2/3"
    elif layer == "L1":
        return "L1 Htr3a"  # special case
    if class_name == "EXC":
        class_name = "Exc"
    if class_name == 'VIP':
        class_name = 'Htr3a'

    return f"{layer} {class_name}"

def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
    # Sort the original firing rates
    sorted_firing_rates = np.sort(firing_rates)
    # Calculate the empirical cumulative distribution function (CDF)
    percentiles = np.linspace(0, 1, sorted_firing_rates.size)
    # Generate random uniform values from 0 to 1
    rate_rd = np.random.RandomState(seed=rnd_seed)
    x_rand = rate_rd.uniform(low=0, high=1, size=n_neurons)
    # Use inverse transform sampling: interpolate the uniform values to find the firing rates
    target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))
    # target_firing_rates = np.interp(x_rand, percentiles, sorted_firing_rates)
    return target_firing_rates

def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
    tau = tf.cast(tau, dtype)
    abs_u = tf.abs(u)
    num = tf.abs(tau - tf.cast(u <= 0, dtype))
    branch_1 = num / (2 * kappa) * tf.square(u)
    branch_2 = num * (abs_u - 0.5 * kappa)
    return tf.where(abs_u <= kappa, branch_1, branch_2)

def compute_spike_rate_distribution_loss(_rates, target_rate, dtype=tf.float32):
    # Firstly we shuffle the current model rates to avoid bias towardsa particular tuning angles (inherited from neurons ordering in the network)
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rates = tf.gather(_rates, rand_ind)

    # _rates = tf.random.shuffle(_rates) # for some reason this shuffling is not enough for the loss to propagate correctly

    sorted_rate = tf.sort(_rates)
    u = sorted_rate - target_rate
    tau = (tf.range(target_rate.shape[0]) + 1) / target_rate.shape[0]
    loss = huber_quantile_loss(u, tau, 0.002, dtype=dtype)
    # loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype) # this kappa value is usually too large and makes the training unstable
    return loss


### To calculate the loss of firing rates between neuron types
def compute_spike_rate_target_loss(rates, target_rates, dtype=tf.float32):
    # TODO: define this function
    # target_rates is a dictionary that contains all the cell types.
    # I should iterate on them, and add the cost for each one at the end.
    # spikes will have a shape of (batch_size, n_steps, n_neurons)
    total_loss = tf.constant(0.0, dtype=dtype)
    num_neurons = tf.constant(0, dtype=tf.int32)
    for key, value in target_rates.items():
        neuron_ids = value["neuron_ids"]
        if len(neuron_ids) != 0:
            _rate_type = tf.gather(rates, neuron_ids)
            target_rate = value["sorted_target_rates"]
            loss_type = compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=dtype)
            total_loss += tf.reduce_sum(loss_type)
            num_neurons += tf.size(neuron_ids)

    total_loss /= tf.cast(num_neurons, dtype=dtype)  

    return total_loss
    

class SpikeRateDistributionTarget:
    """ Instead of regularization, treat it as a target.
        The main difference is that this class will calculate the loss
        for each subtypes of the neurons."""
    def __init__(self, network, spontaneous_fr=False, natural_images=False, rate_cost=.5, pre_delay=None, post_delay=None, data_dir='GLIF_network', area='v1', 
                core_mask=None, rates_dampening=1.0, seed=42, dtype=tf.float32):
        self._network = network
        self._rate_cost = rate_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._area = area
        self._rates_dampening = rates_dampening
        self._core_mask = core_mask
        if self._core_mask is not None:
            self._np_core_mask = self._core_mask.numpy()
        self._data_dir = data_dir
        self._dtype = dtype
        self._seed = seed
        # Check that only one option is selected
        if (spontaneous_fr and natural_images):
            raise ValueError("Only one of 'spontaneous_fr' or 'natural_images' can be selected at a time.")

        if spontaneous_fr:
            self.neuropixels_feature = 'firing_rate_sp'
        elif natural_images:
            self.neuropixels_feature = 'firing_rate_ns'
        else:
            self.neuropixels_feature = 'Ave_Rate(Hz)'
        self._target_rates = self.get_neuropixels_firing_rates()

    def get_neuropixels_firing_rates(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load required data
        neuropixels_data_path = f'Neuropixels_data/{self._area}_OSI_DSI_DF.csv'
        if not os.path.exists(neuropixels_data_path):
            process_neuropixels_data(area=self._area, path=neuropixels_data_path)
        features_to_load = ['ecephys_unit_id', 'cell_type', 'firing_rate_ns', 'firing_rate_sp', 'Ave_Rate(Hz)']
        np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
        area_node_types = pd.read_csv(os.path.join(self._data_dir, f'network/{self._area}_node_types.csv'), sep=" ")

        # Define population queries
        query_mapping = {
            "i1H": 'ALL_L1',
            "e23": 'EXC_L23',
            "i23P": 'PV_L23',
            "i23S": 'SST_L23',
            "i23H": 'VIP_L23',
            "e4": 'EXC_L4',
            "i4P": 'PV_L4',
            "i4S": 'SST_L4',
            "i4H": 'VIP_L4',
            "e5": 'EXC_L5',
            "i5P": 'PV_L5',
            "i5S": 'SST_L5',
            "i5H": 'VIP_L5',
            "e6": 'EXC_L6',
            "i6P": 'PV_L6',
            "i6S": 'SST_L6',
            "i6H": 'VIP_L6'
        }

        # define the reverse mapping
        reversed_query_mapping = {v:k for k, v in query_mapping.items()}
        # Process rates
        type_rates_dict = {
                            reversed_query_mapping[cell_type]: np.append(subdf[self.neuropixels_feature].dropna().values / 1000, 0)
                            # reversed_query_mapping[cell_type]: np.sort(np.append(subdf[self.neuropixels_feature].dropna().values / 1000, 0)) # the rates are sorted again later so is redundant
                            for cell_type, subdf in np_df.groupby("cell_type")
                        }
        # Identify node_type_ids for each population query
        pop_ids = {query: area_node_types[area_node_types.pop_name.str.contains(query)].index.values for query in query_mapping.keys()}
        # Create a dictionary with rates and IDs
        target_firing_rates = {pop_query: {'rates': type_rates_dict[pop_query], 'ids': pop_ids[pop_query]} for pop_query in pop_ids.keys()}
        
        for key, value in target_firing_rates.items():
            # identify tne ids that are included in value["ids"]
            neuron_ids = np.where(np.isin(self._network["node_type_ids"], value["ids"]))[0]
            if self._core_mask is not None:
                # if core_mask is not None, use only neurons in the core
                # core_node_type_ids = self._network['node_type_ids'][self._np_core_mask]
                # neuron_ids = np.where(np.isin(core_node_type_ids, value["ids"]))[0]
                neuron_ids = neuron_ids[self._np_core_mask[neuron_ids]]
            # else:
            #     neuron_ids = np.where(np.isin(self._network["node_type_ids"], value["ids"]))[0]
            
            type_n_neurons = len(neuron_ids)
            # target_firing_rates[key]['neuron_ids'] = tf.convert_to_tensor(neuron_ids, dtype=tf.int32)
            target_firing_rates[key]['neuron_ids'] = tf.convert_to_tensor(neuron_ids, dtype=tf.int32) #tf.convert_to_tensor(neuron_ids, dtype=tf.int32)
            sorted_target_rates = self._rates_dampening * sample_firing_rates(value["rates"], type_n_neurons, self._seed)
            target_firing_rates[key]['sorted_target_rates'] = tf.convert_to_tensor(sorted_target_rates, dtype=self._dtype)
            # target_firing_rates[key]['tuning_angles'] = tf.cast(tf.gather(self._network['tuning_angle'], neuron_ids), dtype=self._dtype)

        return target_firing_rates

    def __call__(self, spikes, trim=True):

        # if spikes.dtype != self._dtype:
        #     spikes = tf.cast(spikes, self._dtype)

        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)
        rates = tf.reduce_mean(spikes, (0, 1)) # calculate the mean firing rate over time and batch
        
        reg_loss = compute_spike_rate_target_loss(rates, self._target_rates, dtype=self._dtype) 

        return reg_loss * self._rate_cost

        # rates = tf.reduce_mean(spikes, axis=1) # rates.shape = (batch_size, num_neurons)
        # reg_loss_per_batch = tf.map_fn(lambda rate: compute_spike_rate_target_loss(rate, self._target_rates), rates, dtype=self._dtype)
        # reg_loss_per_batch = tf.cast(reg_loss_per_batch, dtype=self._dtype)
        # # reg_loss = compute_spike_rate_distribution_loss(rates, self._target_rates) * self._rate_cost
        # # reg_loss = tf.cast(tf.reduce_mean(reg_loss), dtype=self._dtype)

        # return reg_loss_per_batch * self._rate_cost


class GarretRateDistributionTarget:
    def __init__(self, network, rate_cost=.5, pre_delay=None, post_delay=None, data_dir='GLIF_network', 
                rates_dampening=1.0, seed=42, dtype=tf.float32):
        self._network = network
        self._rate_cost = rate_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._rates_dampening = rates_dampening
        self._data_dir = data_dir
        self._dtype = dtype
        self._seed = seed
        self._target_rates = self.get_garret_firing_rates()
    
    def get_garret_firing_rates(self):
        path = '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/garrett_firing_rates.pkl'
        with open(path, 'rb') as f:
            garrett_firing_rates = pkl.load(f)

        sorted_target_rates = self._rates_dampening * sample_firing_rates(garrett_firing_rates, self._network['n_nodes'], self._seed)

        return sorted_target_rates

    def __call__(self, spikes, trim=True):
        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        rates = tf.reduce_mean(spikes, axis=1) # rates.shape = (batch_size, num_neurons)
        reg_loss_per_batch = tf.map_fn(compute_spike_rate_distribution_loss_per_batch, rates, dtype=self._dtype)
        reg_loss_per_batch = tf.cast(reg_loss_per_batch, dtype=self._dtype)
        # reg_loss = compute_spike_rate_distribution_loss(rates, self._target_rates) * self._rate_cost
        # reg_loss = tf.cast(tf.reduce_mean(reg_loss), dtype=self._dtype)

        return reg_loss_per_batch * self._rate_cost


class SynchronizationLoss(Layer):
    def __init__(self, network, sync_cost=10., t_start=None, t_end=None, n_samples=50, data_dir='Synchronization_data', 
                 area='v1', session='evoked', dtype=tf.float32, core_mask=None, seed=42, **kwargs):
        super(SynchronizationLoss, self).__init__(dtype=dtype, **kwargs)
        self._sync_cost = sync_cost
        self._t_start = t_start
        self._t_end = t_end
        self._t_start_seconds = int(t_start * 1000)
        self._t_end_seconds = int(t_end * 1000)
        self._area = area
        self._core_mask = core_mask
        self._data_dir = data_dir
        self._dtype = dtype
        self._n_samples = n_samples
        self._base_seed = seed

        pop_names = other_billeh_utils.pop_names(network)
        if self._core_mask is not None:
            pop_names = pop_names[core_mask]
        node_ei = np.array([pop_name[0] for pop_name in pop_names])
        node_id = np.arange(len(node_ei))
        # Get the IDs for excitatory neurons
        node_id_e = node_id[node_ei == 'e']
        self.node_id_e = tf.constant(node_id_e, dtype=tf.int32)
        # Pre-define bin sizes (same as experimental data)
        bin_sizes = np.logspace(-3, 0, 20)
        # using the simulation length, limit bin_sizes to define at least 2 bins
        bin_sizes_mask = bin_sizes < (self._t_end - self._t_start)/2
        self.bin_sizes = bin_sizes[bin_sizes_mask]
        self.epsilon = 1e-7  # Small constant to avoid division by zero

        # Load the experimental data
        duration = str(int((t_end - t_start) * 1000))
        experimental_data_path = os.path.join(data_dir, f'Fano_factor_{self._area}', f'{self._area}_fano_running_{duration}ms_{session}.npy')
        assert os.path.exists(experimental_data_path), f'File not found: {experimental_data_path}'
        experimental_fanos = np.load(experimental_data_path, allow_pickle=True)
        # Calculate mean, standard deviation, and SEM of the Fano factors
        experimental_fanos_mean = np.nanmean(experimental_fanos[:, bin_sizes_mask], axis=0)
        self.experimental_fanos_mean = tf.constant(experimental_fanos_mean, dtype=self._dtype)

    def pop_fano_tf(self, spikes, bin_sizes):
        spikes = tf.expand_dims(spikes, axis=-1)
        fanos = tf.TensorArray(dtype=self._dtype, size=len(bin_sizes))
        for i, bin_width in enumerate(bin_sizes):
            bin_size = int(np.round(bin_width * 1000))
            # Use convolution for efficient binning
            kernel = tf.ones((bin_size, 1, 1), dtype=self._dtype)
            convolved = tf.nn.conv1d(spikes, kernel, stride=bin_size, padding="VALID")
            sp_counts = tf.squeeze(convolved, axis=-1)  # Shape: (60, new_width)
            # Compute mean and variance of spike counts
            mean_count = tf.reduce_mean(sp_counts, axis=1)
            var_count = tf.math.reduce_variance(sp_counts, axis=1)
            mean_count = tf.maximum(mean_count, self.epsilon)
            # fanos.append(tf.reduce_mean(var_count / mean_count))
            fano_per_sample = var_count / mean_count  # => [n_samples]
            fano = tf.reduce_mean(fano_per_sample)
            fanos = fanos.write(i, fano)

        return fanos.stack()

    def __call__(self, spikes, trim=True):

        # if spikes.dtype != self._dtype:
        #     spikes = tf.cast(spikes, self._dtype)
        
        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        if trim:
            spikes = spikes[:, self._t_start_seconds:self._t_end_seconds, :]
            bin_sizes = self.bin_sizes
            experimental_fanos_mean = self.experimental_fanos_mean
        else:
            t_start = 0
            t_end = spikes.shape[1] / 1000
            # using the simulation length, limit bin_sizes to define at least 2 bins
            bin_sizes_mask = self.bin_sizes < (t_end - t_start)/2
            bin_sizes = self.bin_sizes[bin_sizes_mask]
            experimental_fanos_mean = self.experimental_fanos_mean[bin_sizes_mask]

        spikes = tf.cast(spikes, self._dtype)  
        # choose random trials to sample from (usually we only have 1 trial to sample from)
        n_trials = tf.shape(spikes)[0]
        # increase the base seed to avoid the same random neurons to be selected in every instantiation of the class
        self._base_seed += 1
        sample_trials = tf.random.uniform([self._n_samples], minval=0, maxval=n_trials, dtype=tf.int32, seed=self._base_seed)
        # Generate sample counts with a normal distribution
        sample_size = 70 if self._area == 'v1' else 33
        sample_std = 30 if self._area == 'v1' else 14
        
        sample_counts = tf.cast(tf.random.normal([self._n_samples], mean=sample_size, stddev=sample_std, seed=self._base_seed), tf.int32)
        sample_counts = tf.clip_by_value(sample_counts, clip_value_min=15, clip_value_max=tf.shape(self.node_id_e)[0]) # lower cap to 15 to avoid small samples
        # Randomize the neuron ids
        shuffled_e_ids = tf.random.shuffle(self.node_id_e, seed=self._base_seed)
        selected_spikes_sample = tf.TensorArray(self._dtype, size=self._n_samples)
        previous_id = tf.constant(0, dtype=tf.int32)
        for i in tf.range(self._n_samples):
            sample_num = sample_counts[i] # 40 #68
            sample_trial = sample_trials[i] # 0
            ## randomly choose sample_num ids from self.node_id_e with replacement
            ## sample_ids = tf.random.shuffle(self.node_id_e)[:sample_num]
            ## randomly choose sample_num ids from shuffled_ids without replacement
            if previous_id + sample_num > tf.size(shuffled_e_ids):
                # shuffled_e_ids = tf.random.shuffle(self.node_id_e, seed=self._base_seed)
                shuffled_e_ids = tf.random.shuffle(shuffled_e_ids, seed=self._base_seed)
                previous_id = tf.constant(0, dtype=tf.int32)
            sample_ids = shuffled_e_ids[previous_id:previous_id+sample_num]
            previous_id += sample_num
            
            selected_spikes = tf.reduce_sum(tf.gather(spikes[sample_trial], sample_ids, axis=1), axis=-1)
            selected_spikes_sample = selected_spikes_sample.write(i, selected_spikes)

        selected_spikes_sample = selected_spikes_sample.stack()
        fanos_mean = self.pop_fano_tf(selected_spikes_sample, bin_sizes=bin_sizes)
        # # Calculate MSE between the experimental and calculated Fano factors
        # mse_loss = tf.sqrt(tf.reduce_mean(tf.square(experimental_fanos_mean - fanos_mean)))
        mse_loss = tf.reduce_mean(tf.square(experimental_fanos_mean - fanos_mean))
        # mse_loss = tf.reduce_sum(tf.square(experimental_fanos_mean - fanos_mean))
        # # Calculate the synchronization loss
        sync_loss = self._sync_cost * mse_loss

        return sync_loss
    

class SynchronizationRegularization(Layer):
    # This loss function penalizes the synchronization of the neurons by 
    # penalizing the bins with largest number of neurons spiking
    def __init__(self, sync_cost=10., target_sync=0.1, pre_delay=None, post_delay=None, data_dir='GLIF_network', 
                 area='v1', dtype=tf.float32, core_mask=None, **kwargs):
        super(SynchronizationLoss, self).__init__(dtype=dtype, **kwargs)
        self._sync_cost = sync_cost
        self._target_sync = tf.constant(target_sync, dtype=dtype)
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._area = area
        self._core_mask = core_mask
        self._data_dir = data_dir
        self._dtype = dtype

    def compute_synchronization_proxy(self, spikes, bin_size_ms=20, time_resolution_ms=1):
        spikes_shape = tf.shape(spikes)
        num_time_points = spikes_shape[1]
        num_neurons = spikes_shape[2]
        bin_size_points = bin_size_ms // time_resolution_ms
        num_bins = tf.math.floordiv(num_time_points, bin_size_points)
        # Just focus on the first batch
        if len(spikes_shape) == 3:
            spikes = spikes[0]
        else:
            pass
        # Reshape the spikes tensor to [num_bins, bin_size_points, num_neurons]
        spikes = tf.reshape(spikes[:num_bins * bin_size_points], [num_bins, bin_size_points, num_neurons])
        # Sum spike counts within each bin
        spike_counts_per_bin = tf.reduce_sum(spikes, axis=1)
        # Determine active neurons in each bin
        active_neurons_per_bin = tf.math.count_nonzero(spike_counts_per_bin, axis=1, dtype=self._dtype)
        # Compute the fraction of active neurons for each bin
        fraction_active_per_bin = active_neurons_per_bin / tf.cast(num_neurons, self._dtype)
        # Find the maximum fraction of active neurons
        max_fraction_active = tf.reduce_max(fraction_active_per_bin)

        return max_fraction_active
    
    def compute_mean_pairwise_correlation(self, spikes, bin_size=20): ### TOO MEMORY DEMANDING ###
        """
        Computes the mean pairwise correlation for neuron spikes.
        :param spike_matrix: Tensor of shape (batch_size, seq_len, n_neurons)
        :return: Scalar mean pairwise correlation
        """
        batch_size, seq_len, n_neurons = tf.shape(spikes)[0], tf.shape(spikes)[1], tf.shape(spikes)[2]        
        new_seq_len = seq_len // bin_size
        # Truncate the spike_matrix to be evenly divisible by bin_size
        truncated_len = new_seq_len * bin_size
        spikes = spikes[:, :truncated_len, :]
        # Reshape and sum within bins
        spikes = tf.reshape(spikes, (batch_size, new_seq_len, bin_size, n_neurons))
        spikes = tf.reduce_sum(spikes, axis=2)
        # Transpose to shape (batch_size, n_neurons, seq_len)
        spikes = tf.transpose(spikes, perm=[0, 2, 1])
        mean_pairwise_correlations = tf.constant(0, dtype=self._dtype)
        for batch in range(batch_size):
            # Calculate mean spike rate for each neuron
            mean_spikes = tf.reduce_mean(spikes[batch], axis=1, keepdims=True)
            centered_spikes = spikes[batch] - mean_spikes
            # Calculate the standard deviation of each neuron
            std_spikes = tf.math.reduce_std(spikes[batch], axis=1)
            # Remove neurons with zero standard deviation
            valid_neurons = std_spikes > 0
            centered_spikes = tf.boolean_mask(centered_spikes, valid_neurons, axis=0)
            # if tf.shape(centered_spikes)[0] > 1:  # Ensure there are at least two neurons to correlate
                # Calculate pairwise correlations
            correlation_matrix = tf.linalg.matmul(centered_spikes, centered_spikes, transpose_b=True) / tf.cast(new_seq_len, self._dtype)
            # Normalize the correlation matrices to get correlation coefficients
            norms = tf.sqrt(tf.linalg.diag_part(correlation_matrix))
            correlation_matrix = correlation_matrix / (tf.expand_dims(norms, 1) * tf.expand_dims(norms, 0))
            # Take the absolute value of the correlation coefficients
            # correlation_matrix = tf.abs(correlation_matrix)
            # Mask the diagonal (self-correlations)
            mask = tf.eye(tf.shape(correlation_matrix)[0], dtype=tf.bool)
            correlation_matrix = tf.where(mask, tf.zeros_like(correlation_matrix), correlation_matrix)
            # Calculate the mean of the upper triangle part of the correlation matrix (excluding diagonal)
            upper_triangle_indices = tf.where(tf.linalg.band_part(tf.ones_like(correlation_matrix), 0, -1) - tf.linalg.band_part(tf.ones_like(correlation_matrix), 0, 0))
            mean_correlation = tf.reduce_mean(tf.gather_nd(correlation_matrix, upper_triangle_indices))
            # mean_pairwise_correlations.append(mean_correlation)
            mean_pairwise_correlations = mean_pairwise_correlations + mean_correlation
        mean_pairwise_correlations = mean_pairwise_correlations / tf.cast(batch_size, self._dtype)

        return mean_pairwise_correlations
 
    def __call__(self, spikes, trim=True):

        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        # Compute the synchronization proxy
        sync_proxy  = self.compute_synchronization_proxy(spikes)
        # Compute the mean pairwise correlation
        # sync_proxy = self.compute_mean_pairwise_correlation(spikes)
        
        # Calculate the synchronization loss
        sync_loss = self._sync_cost * tf.abs(sync_proxy - self._target_sync)
        # sync_loss = self._sync_cost * sync_proxy

        return sync_loss


class VoltageRegularization:
    def __init__(self, cell, area='v1', voltage_cost=1e-5, dtype=tf.float32, core_mask=None):
        self._voltage_cost = voltage_cost
        self._cell = cell
        self._dtype = dtype
        self._core_mask = core_mask
        # self._voltage_offset = tf.cast(self._cell.voltage_offset, dtype)
        # self._voltage_scale = tf.cast(self._cell.voltage_scale, dtype)
        # if core_mask is not None:
        #     self._voltage_offset = tf.boolean_mask(self._voltage_offset, core_mask)
        #     self._voltage_scale = tf.boolean_mask(self._voltage_scale, core_mask)

    def __call__(self, voltages):
        if voltages.dtype != self._dtype:
            voltages = tf.cast(voltages, self._dtype)

        if self._core_mask is not None:
            voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)

        # voltages = (voltages - self._voltage_offset) / self._voltage_scale
        # v_tot = tf.square(tf.nn.relu(voltages - 1.0) + tf.nn.relu(-voltages + 1.0))
        # More efficient computation (the loss function tries to keep the voltage close to the threshold)
        v_tot = tf.square(voltages - 1.0)
        voltage_loss = tf.reduce_mean(v_tot)
        # voltage_loss = tf.reduce_mean(tf.reduce_mean(v_tot, -1))
        # voltage_loss = tf.reduce_mean(v_pos + v_neg, axis=[1, 2]) * self._voltage_cost 

        # return voltage_loss * self._voltage_cost
        return voltage_loss * self._voltage_cost


class CustomMeanLayer(Layer):
    def call(self, inputs):
        spike_rates, mask = inputs
        masked_data = tf.boolean_mask(spike_rates, mask)
        return tf.reduce_mean(masked_data)


class OrientationSelectivityLoss:
    def __init__(self, network=None, osi_cost=1e-5, area='v1', pre_delay=None, post_delay=None, dtype=tf.float32, core_mask=None,
                 method="crowd_osi", subtraction_ratio=1.0, layer_info=None):

        self._network = network
        self._osi_cost = osi_cost
        self._area = area
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._dtype = dtype
        self._core_mask = core_mask
        self._method = method
        self._subtraction_ratio = subtraction_ratio # only for crowd_spikes method
        self._tf_pi = tf.constant(np.pi, dtype=dtype)
        if (self._core_mask is not None) and (self._method == "crowd_spikes" or self._method == "crowd_osi"):
            # tuning_angles = network['tuning_angle']
            self.np_core_mask = self._core_mask.numpy()
            core_tuning_angles = network['tuning_angle'][self.np_core_mask]
            self._tuning_angles = tf.constant(core_tuning_angles, dtype=dtype)
        else:
            self._tuning_angles = tf.constant(network['tuning_angle'], dtype=dtype) 
        
        if self._method == "neuropixels_fr":
            self._layer_info = layer_info  # needed for neuropixels_fr method
            # the layer_info should be a dictionary that contains
            # the cell id of the corresponding layer.
            # the keys should be something like "EXC_L23" or "PV_L5"   

        elif self._method == "crowd_osi":
            # Get the target OSI
            self._target_osi_dsi = self.get_neuropixels_osi_dsi()
            self._min_rates_threshold = tf.constant(0.0005, dtype=self._dtype)
            # sum the core_mask
            n_nodes = len(self._tuning_angles)
            # self.node_type_ids = tf.zeros(n_nodes, dtype=tf.int32)
            node_type_ids = np.zeros(n_nodes, dtype=np.int32)
            osi_target_values = []
            dsi_target_values = []
            cell_type_count = []
            for node_type_id, (key, value) in enumerate(self._target_osi_dsi.items()):
                node_ids = value['ids']
                osi_target_values.append(value['OSI'])
                dsi_target_values.append(value['DSI'])
                cell_type_count.append(len(node_ids))
                # update the ndoe_type_ids tensor in positions node_ids with the node_type_id
                # self.node_type_ids = tf.tensor_scatter_nd_update(self.node_type_ids, indices=tf.expand_dims(node_ids, axis=1), updates=tf.fill(tf.shape(node_ids), node_type_id))
                node_type_ids[node_ids] = node_type_id

            self.osi_target_values = tf.constant(osi_target_values, dtype=self._dtype)
            self.dsi_target_values = tf.constant(dsi_target_values, dtype=self._dtype)
            self.cell_type_count = tf.constant(cell_type_count, dtype=self._dtype)
            self.node_type_ids = tf.constant(node_type_ids, dtype=tf.int32)
            self._n_node_types = len(self._target_osi_dsi)

    def calculate_delta_angle(self, stim_angle, tuning_angle):
        # angle unit is degrees.
        # this function calculates the difference between stim_angle and tuning_angle,
        # but it is fine to have the opposite direction.
        # so, delta angle is always between -90 and 90.
        # they are both vector, so dimension matche is needed.
        # stim_angle is a length of batch size
        # tuning_angle is a length of n_neurons

        # delta_angle = stim_angle - tuning_angle
        delta_angle = tf.expand_dims(stim_angle, axis=1) - tuning_angle
        delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
        delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)
        # # do it twice to make sure
        delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
        delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)

        return delta_angle
    
    def get_neuropixels_osi_dsi(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        neuropixels_data_path = f'Neuropixels_data/{self._area}_OSI_DSI_DF.csv'
        if not os.path.exists(neuropixels_data_path):
            process_neuropixels_data(area=self._area, path=neuropixels_data_path)
        features_to_load = ['ecephys_unit_id', 'cell_type', 'OSI', 'DSI', "Ave_Rate(Hz)", "max_mean_rate(Hz)"]
        osi_dsi_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')

        nonresponding = osi_dsi_df["max_mean_rate(Hz)"] < 0.5
        osi_dsi_df.loc[nonresponding, "OSI"] = np.nan
        osi_dsi_df.loc[nonresponding, "DSI"] = np.nan
        osi_dsi_df = osi_dsi_df[osi_dsi_df["Ave_Rate(Hz)"] != 0]
        osi_dsi_df.dropna(inplace=True)
        osi_dsi_df["cell_type"] = osi_dsi_df["cell_type"].apply(neuropixels_cell_type_to_cell_type)
        osi_target = osi_dsi_df.groupby("cell_type")['OSI'].mean()
        dsi_target = osi_dsi_df.groupby("cell_type")['DSI'].mean()

        original_pop_names = other_billeh_utils.pop_names(self._network)
        if self._core_mask is not None:
            original_pop_names = original_pop_names[self.np_core_mask] 

        cell_types = np.array([other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in original_pop_names])
        node_ids = np.arange(len(cell_types))
        cell_ids = {key: node_ids[cell_types == key] for key in set(osi_dsi_df['cell_type'])}
        
        osi_dsi_exp_dict = {key: {'OSI': val, 'DSI': dsi_target[key], 'ids': cell_ids[key]} for key, val in osi_target.to_dict().items()}

        return osi_dsi_exp_dict
    
    def vonmises_model_fr(self, structure, population):
        from scipy.stats import vonmises
        paramdic = self._von_mises_params
        _params = paramdic[structure][population]
        if len(_params) == 4:
            mu, kappa, a, b = _params
        vonmises_pdf = vonmises(kappa, loc=mu).pdf

        angles = np.deg2rad(np.arange(-85, 86, 10)) * 2  # *2 needed to make it proper model
        model_fr = a + b * vonmises_pdf(angles)

        return model_fr
    
    def neuropixels_fr_loss(self, spikes, angle):
        # if the trget fr is not set, construct them
        if not hasattr(self, "_target_frs"):

            # self._von_mises_params = np.load("GLIF_network/param_dict_orientation.npy")
            # pickle instead
            with open("GLIF_network/param_dict_orientation.pkl", 'rb') as f:
                self._von_mises_params = pkl.load(f)
            # get the model values with 10 degree increments 
            structure = "VISp"
            self._target_frs = {}
            for key in self._layer_info.keys():
                self._target_frs[key] = self.vonmises_model_fr(structure, key)
                # TODO: convert it to tensor if needed.

        # assuming 1 ms bins
        spike_rates = tf.reduce_mean(spikes, axis=[0, 1]) / spikes.shape[1] * 1000
        angle_bins = tf.constant(np.arange(-90, 91, 10), dtype=self._dtype)
        nbins = angle_bins.shape[0] - 1
        # now, process each layer
        # losses = tf.TensorArray(tf.float32, size=len(self._layer_info))
        losses = []
        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        custom_mean_layer = CustomMeanLayer()

        for key, value in self._layer_info.items():
            # first, calculate delta_angle

            # rates = tf.TensorArray(tf.float32, size=nbins)
            rates_list = []
            for i in range(nbins):
                mask = (delta_angle >= angle_bins[i]) & (delta_angle < angle_bins[i+1])
                # take the intersection with core mask
                mask = tf.logical_and(mask, self._core_mask)
                mask = tf.logical_and(mask, value)
                # mask = mask.flatten()
                # doesn't work.
                mask = tf.reshape(mask, [-1])
                mean_val = custom_mean_layer([spike_rates, mask])
                # rates_ = rates.write(i, mean_val)
                rates_list.append(mean_val)
                # rates = rates.write(i, tf.reduce_mean(tf.boolean_mask(spike_rates, mask)))

            # calculate the loss
            # rates = rates.stack()
            rates = tf.stack(rates_list)
            loss = tf.reduce_mean(tf.square(rates - self._target_frs[key]))
            # if key == "EXC_L6":
                # print the results!
                # tf.print("Layer6: ", rates)
                # tf.print("target: ", self._target_frs[key])
            # losses = losses.write(i, loss)
            losses.append(loss)

        # final_loss = tf.reduce_sum(losses.stack()) * self._osi_cost
        final_loss = tf.reduce_mean(tf.stack(losses)) * self._osi_cost

        return final_loss

    def crowd_spikes_loss(self, spikes, angle):
        # I need to access the tuning angle. of all the neurons.
        angle = tf.cast(angle, self._dtype)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        # sum spikes in _z, and multiply with delta_angle.
        mean_spikes = tf.reduce_mean(spikes, axis=[1]) 
        mean_angle = mean_spikes * delta_angle
        # Here, the expected value with random firing to subtract
        # (this prevents the osi loss to drive the firing rates to go to zero.)
        expected_sum_angle = tf.reduce_mean(mean_spikes) * 45

        angle_loss = tf.reduce_mean(tf.abs(mean_angle)) - expected_sum_angle * self._subtraction_ratio

        return angle_loss * self._osi_cost

    def crowd_osi_loss(self, spikes, angle, normalizer=None):  
        # Ensure angle is [batch_size] and cast to correct dtype
        angle = tf.cast(tf.reshape(angle, [-1]), self._dtype)  # [batch_size]
        # Compute delta_angle with broadcasting
        delta_angle = angle[:, tf.newaxis] - self._tuning_angles[tf.newaxis, :]  # [batch_size, n_neurons_core]
        radians_delta_angle = delta_angle * (self._tf_pi / 180)
            
        # Compute rates over time dimension
        rates = tf.reduce_mean(spikes, axis=1)  # [batch_size, n_neurons]
        if self._core_mask is not None:
            rates = tf.boolean_mask(rates, self._core_mask, axis=1)

        if normalizer is not None:
            if self._core_mask is not None:
                normalizer = tf.boolean_mask(normalizer, self._core_mask, axis=0)
            # Use tf.maximum to ensure each element of normalizer does not fall below min_normalizer_value
            normalizer = tf.maximum(normalizer, self._min_rates_threshold)
            rates = rates / normalizer

        # Instead of complex numbers, use cosine and sine separately
        weighted_osi_cos_responses = rates * tf.math.cos(2.0 * radians_delta_angle)
        weighted_dsi_cos_responses = rates * tf.math.cos(radians_delta_angle)

        batch_size = tf.shape(rates)[0]
        # Adjust segment_ids for batch dimension
        batch_offsets = tf.range(batch_size, dtype=self.node_type_ids.dtype) * self._n_node_types  # [batch_size]
        batch_offsets_expanded = batch_offsets[:, tf.newaxis]  # [batch_size, 1]

        segment_ids = self.node_type_ids[tf.newaxis, :]  # [1, n_neurons_core]
        segment_ids = tf.tile(segment_ids, [batch_size, 1])  # [batch_size, n_neurons_core]
        segment_ids = segment_ids + batch_offsets_expanded  # [batch_size, n_neurons_core]

        # Flatten data and segment_ids
        data_flat_rates = tf.reshape(rates, [-1])  # [batch_size * n_neurons_core]
        data_flat_weighted_osi = tf.reshape(weighted_osi_cos_responses, [-1])
        data_flat_weighted_dsi = tf.reshape(weighted_dsi_cos_responses, [-1])
        segment_ids_flat = tf.reshape(segment_ids, [-1])

        num_segments = batch_size * self._n_node_types

        # Compute denominators and numerators
        approximated_denominator = tf.math.unsorted_segment_mean(data_flat_rates, segment_ids_flat, num_segments=num_segments)
        approximated_denominator = tf.reshape(approximated_denominator, [batch_size, self._n_node_types])
        approximated_denominator = tf.maximum(approximated_denominator, self._min_rates_threshold)

        osi_numerator = tf.math.unsorted_segment_mean(data_flat_weighted_osi, segment_ids_flat, num_segments=num_segments)
        osi_numerator = tf.reshape(osi_numerator, [batch_size, self._n_node_types])

        dsi_numerator = tf.math.unsorted_segment_mean(data_flat_weighted_dsi, segment_ids_flat, num_segments=num_segments)
        dsi_numerator = tf.reshape(dsi_numerator, [batch_size, self._n_node_types])

        # Compute approximations
        osi_approx_type = osi_numerator / approximated_denominator  # [batch_size, n_node_types]
        dsi_approx_type = dsi_numerator / approximated_denominator

        # Average over batch size
        osi_approx_type = tf.reduce_mean(osi_approx_type, axis=0) # [n_node_types]
        dsi_approx_type = tf.reduce_mean(dsi_approx_type, axis=0)

        # Compute losses    
        # osi_target_values = self.osi_target_values[tf.newaxis, :]  # [1, n_node_types]
        # dsi_target_values = self.dsi_target_values[tf.newaxis, :]  # [1, n_node_types]
        osi_loss_type = tf.math.square(osi_approx_type - self.osi_target_values)  # [n_node_types]
        dsi_loss_type = tf.math.square(dsi_approx_type - self.dsi_target_values)

        # cell_type_count = self.cell_type_count[tf.newaxis, :]  # [1, n_node_types]
        numerator = tf.reduce_sum((osi_loss_type + dsi_loss_type) * self.cell_type_count)  # Scalar
        denominator = tf.reduce_sum(self.cell_type_count)  # Scalar

        # total_loss_per_batch = numerator / denominator  # [batch_size]
        # total_loss = tf.reduce_mean(total_loss_per_batch) * self._osi_cost

        total_loss = (numerator / denominator) * self._osi_cost

        return total_loss

    def __call__(self, spikes, angle, trim, normalizer=None):

        # if spikes.dtype != self._dtype:
        #     spikes = tf.cast(spikes, self._dtype)

        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        if self._method == "crowd_osi":
            return self.crowd_osi_loss(spikes, angle, normalizer=normalizer)
        elif self._method == "crowd_spikes":
            return self.crowd_spikes_loss(spikes, angle)
        elif self._method == "neuropixels_fr":
            return self.neuropixels_fr_loss(spikes, angle)

            
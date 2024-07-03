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


# class StiffRegularizer(Layer):
# # class StiffRegularizer(tf.keras.regularizers.Regularizer):
#     def __init__(self, strength, initial_value, edge_type_ids=None):
#         # super().__init__()
#         self._strength = strength
#         # self._initial_value = tf.Variable(initial_value, trainable=False)
#         self._initial_value = tf.constant(initial_value, dtype=tf.float32)
#         if edge_type_ids is not None:
#             # using the edge_type ids group calculate the mean weight of each type of edge in the network and then create a constant with same shape as weights and with each value corresponding to the populations mean
#             initial_weights = initial_value.numpy()
#             # Calculate mean weights for each edge type
#             unique_edge_type_ids, inverse_indices = np.unique(edge_type_ids, return_inverse=True)
#             mean_weights = np.array([np.mean(initial_value[edge_type_ids == edge_type_id]) for edge_type_id in unique_edge_type_ids])
#             # Create target mean weights array based on the edge type indices
#             self._target_mean_weights = tf.constant(mean_weights[inverse_indices], dtype=tf.float32)
#         else:
#             self._target_mean_weights = None

#     def __call__(self, x):
#         if self._target_mean_weights is None:
#             return self._strength * tf.reduce_mean(tf.square(x - self._initial_value))
#         else:
#             # return self._strength * tf.reduce_mean(tf.square(x - self._initial_value))
#             relative_deviation = (x - self._target_mean_weights) / self._target_mean_weights
#             # Penalize the relative deviation
#             return self._strength * tf.reduce_mean(tf.abs(relative_deviation))


class StiffRegularizer(Layer):
# class StiffRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, network, penalize_relative_change=False, recurrent_weights=True, source_area='lm', dtype=tf.float32):
        # super().__init__()
        self._strength = tf.cast(strength, dtype)
        self._dtype = dtype
        self._penalize_relative_change = penalize_relative_change
        voltage_scale = tf.constant(network['node_params']['V_th'] - network['node_params']['E_L'], dtype=dtype)
        _node_type_ids = tf.constant(network['node_type_ids'], dtype=tf.int32)
        # Get the initial weights and properly scale them down
        if recurrent_weights:
            indices = tf.constant(network["synapses"]["indices"], dtype=tf.int32)
            initial_value = tf.constant(network["synapses"]["weights"], dtype=dtype)
            edge_type_ids = tf.constant(network['synapses']['edge_type_ids'], dtype=tf.int16)
        else:
            indices = tf.constant(network['interarea_synapses'][source_area]["indices"], dtype=tf.int32)
            initial_value = tf.constant(network['interarea_synapses'][source_area]["weights"], dtype=dtype)
            edge_type_ids = tf.constant(network['interarea_synapses'][source_area]["edge_type_ids"], dtype=tf.int16)

        # initial_value /= voltage_scale[network['node_type_ids'][indices[:, 0]]] * flags.lr_scale
        # initial_value = tf.constant(initial_value, dtype=tf.float16)
        voltage_scale_node_ids = tf.gather(voltage_scale, tf.gather(_node_type_ids, indices[:, 0]))
        initial_value /= voltage_scale_node_ids
        
        # using the edge_type ids group calculate the mean weight of each type of edge in the network and then create a constant with same shape as weights and with each value corresponding to the populations mean
        # Calculate mean weights for each edge type
        unique_edge_types, self.idx = tf.unique(edge_type_ids)
        del indices, edge_type_ids
        self.num_unique = tf.shape(unique_edge_types)[0]
        # Calculate the initial mean weight for each edge type
        initial_value = tf.math.unsorted_segment_mean(initial_value, self.idx, self.num_unique)

        if penalize_relative_change:
            epsilon = tf.constant(1e-3, dtype=dtype)
            self._target_mean_weights = tf.maximum(tf.abs(initial_value), epsilon)
        else:
            self._target_mean_weights = initial_value

    # @tf.function
    def __call__(self, x):
        # only do tf.cast if the dtype is not the same
        if x.dtype != self._dtype:
            x = tf.cast(x, self._dtype)

        mean_edge_type_weights = tf.math.unsorted_segment_mean(x, self.idx, self.num_unique)
        if self._penalize_relative_change:
            # return self._strength * tf.reduce_mean(tf.abs(x - self._initial_value))
            relative_deviation = (mean_edge_type_weights - self._target_mean_weights) / self._target_mean_weights
            # Penalize the relative deviation
            reg_loss = tf.sqrt(tf.reduce_mean(tf.square(relative_deviation)))
        else:
            reg_loss = tf.reduce_mean(tf.square(mean_edge_type_weights - self._target_mean_weights))
        
        return tf.cast(reg_loss * self._strength, tf.float32) 

        
# class StiffRegularizer(Layer):
# # class StiffRegularizer(tf.keras.regularizers.Regularizer):
#     def __init__(self, strength, network, flags, penalize_relative_change=False, recurrent_weights=True, source_area='lm'):
#         # super().__init__()
#         self._strength = strength
#         # Get the initial weights and properly scale them down
#         if recurrent_weights:
#             indices = np.array(network["synapses"]["indices"])
#             weights = np.array(network["synapses"]["weights"])
#         else:
#             indices = np.array(network['interarea_synapses'][source_area]["indices"])
#             weights = np.array(network['interarea_synapses'][source_area]["weights"])
#         _params = dict(network['node_params'])
#         voltage_scale = _params['V_th'] - _params['E_L']
#         _node_type_ids = network['node_type_ids']
#         initial_value = (weights/voltage_scale[_node_type_ids[indices[:, 0]]]) / flags.lr_scale
#         self._initial_value = tf.constant(initial_value, dtype=tf.float32)

#         if penalize_relative_change:
#             # using the edge_type ids group calculate the mean weight of each type of edge in the network and then create a constant with same shape as weights and with each value corresponding to the populations mean
#             # Calculate mean weights for each edge type
#             if recurrent_weights:
#                 edge_type_ids = np.array(network['synapses']['edge_type_ids'])
#             else:
#                 edge_type_ids = np.array(network['interarea_synapses'][source_area]["edge_type_ids"])
#             unique_edge_type_ids, inverse_indices = np.unique(edge_type_ids, return_inverse=True)
#             mean_weights = np.array([np.mean(initial_value[edge_type_ids == edge_type_id]) for edge_type_id in unique_edge_type_ids])
#             # Create target mean weights array based on the edge type indices
#             self._target_mean_weights = tf.constant(mean_weights[inverse_indices], dtype=tf.float32)
#             epsilon = tf.constant(1e-3, dtype=tf.float32)  # Small constant to avoid division by zero
#             self._target_mean_weights_clipped = tf.maximum(tf.abs(self._target_mean_weights), epsilon)
#         else:
#             self._target_mean_weights = None

#     def __call__(self, x):
#         if self._target_mean_weights is None:
#             return self._strength * tf.reduce_mean(tf.square(x - self._initial_value))
#         else:
#             # return self._strength * tf.reduce_mean(tf.abs(x - self._initial_value))
#             relative_deviation = (x - self._target_mean_weights) / self._target_mean_weights_clipped
#             mse = self._strength * tf.reduce_mean(tf.square(relative_deviation))
#             # Penalize the relative deviation
#             return tf.sqrt(mse)


class L2Regularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, network, flags, penalize_relative_change=False, recurrent_weights=True, source_area='lm'):
        super().__init__()
        self._strength = strength
        # Get the initial weights and properly scale them down
        if recurrent_weights:
            indices = np.array(network["synapses"]["indices"])
            weights = np.array(network["synapses"]["weights"])
        else:
            indices = np.array(network['interarea_synapses'][source_area]["indices"])
            weights = np.array(network['interarea_synapses'][source_area]["weights"])

        _params = dict(network['node_params'])
        voltage_scale = _params['V_th'] - _params['E_L']
        _node_type_ids = network['node_type_ids']
        initial_value = (weights/voltage_scale[_node_type_ids[indices[:, 0]]]) / flags.lr_scale
        self._initial_value = tf.constant(initial_value, dtype=tf.float32)
        if penalize_relative_change:
            # using the edge_type ids group calculate the mean weight of each type of edge in the network and then create a constant with same shape as weights and with each value corresponding to the populations mean
            # Calculate mean weights for each edge type
            if recurrent_weights:
                edge_type_ids = np.array(network['synapses']['edge_type_ids'])
            else:
                edge_type_ids = np.array(network['interarea_synapses'][source_area]["edge_type_ids"])
            unique_edge_type_ids, inverse_indices = np.unique(edge_type_ids, return_inverse=True)
            mean_weights = np.array([np.mean(initial_value[edge_type_ids == edge_type_id]) for edge_type_id in unique_edge_type_ids])
            # Create target mean weights array based on the edge type indices
            self._target_mean_weights = tf.constant(mean_weights[inverse_indices], dtype=tf.float32)
            epsilon = tf.constant(1e-2, dtype=tf.float32)  # A small constant to avoid division by zero
            self._target_mean_weights = tf.maximum(tf.abs(self._target_mean_weights), epsilon)
        else:
            self._target_mean_weights = None

    def __call__(self, x):
        if self._target_mean_weights is None:
            return self._strength * tf.reduce_mean(tf.square(x))
        else:
            relative_deviation = x / self._target_mean_weights
            mse = self._strength * tf.reduce_mean(tf.square(relative_deviation))
            # return tf.sqrt(mse)
            return mse

        
# def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
#     # Sort the original firing rates
#     sorted_firing_rates = np.sort(firing_rates)
#     # Calculate the empirical cumulative distribution function (CDF)
#     # percentiles = (np.arange(firing_rates.shape[-1])).astype(np.float32) / (firing_rates.shape[-1] - 1)
#     percentiles = np.linspace(0, 1, sorted_firing_rates.size)
#     # Generate random uniform values from 0 to 1
#     rate_rd = np.random.RandomState(seed=rnd_seed)
#     x_rand = rate_rd.uniform(low=0, high=1, size=n_neurons)
#     # Use inverse transform sampling: interpolate the uniform values to find the firing rates
#     target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))
#     # target_firing_rates = np.interp(x_rand, percentiles, sorted_firing_rates)
#     return target_firing_rates
    
# def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
#     tau = tf.cast(tau, dtype)
#     num = tf.abs(tau - tf.cast(u <= 0, dtype))

#     branch_1 = num / (2 * kappa) * tf.square(u)
#     branch_2 = num * (tf.abs(u) - 0.5 * kappa)
#     return tf.where(tf.abs(u) <= kappa, branch_1, branch_2)

# ### To calculate the loss of firing rates between neuron types
# def compute_spike_rate_target_loss(_spikes, target_rates, dtype=tf.float32):
#     # TODO: define this function
#     # target_rates is a dictionary that contains all the cell types.
#     # I should iterate on them, and add the cost for each one at the end.
#     # spikes will have a shape of (batch_size, n_steps, n_neurons)
#     total_loss = tf.constant(0.0, dtype=dtype)
#     rates = tf.reduce_mean(_spikes, (0, 1))
#     # if core_mask is not None:
#     #     core_neurons_ids = np.where(core_mask)[0]

#     for key, value in target_rates.items():
#         if tf.size(value["neuron_ids"]) != 0:
#             _rate_type = tf.gather(rates, value["neuron_ids"])
#             target_rate = value["sorted_target_rates"]
#             # if core_mask is not None:
#             #     key_core_mask = np.isin(value["neuron_ids"], core_neurons_ids)
#             #     neuron_ids =  np.where(key_core_mask)[0]
#             #     _rate_type = tf.gather(rates, neuron_ids)
#             #     target_rate = value["sorted_target_rates"][key_core_mask]
#             # else:
#             #     _rate_type = tf.gather(rates, value["neuron_ids"])
#             #     target_rate = value["sorted_target_rates"]
#             loss_type = compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=dtype)
#             mean_loss_type = tf.reduce_mean(loss_type)
#         else:
#             mean_loss_type = tf.constant(0, dtype=dtype)
#         # losses.append(mean_loss_type)
#         total_loss += mean_loss_type
#     # total_loss = tf.reduce_sum(losses, axis=0)
#     return total_loss


# def compute_spike_rate_distribution_loss(_rates, target_rate, dtype=tf.float32):
#     # Firstly we shuffle the current model rates to avoid bias towardsa particular tuning angles (inherited from neurons ordering in the network)
#     ind = tf.range(target_rate.shape[0])
#     rand_ind = tf.random.shuffle(ind)
#     _rates = tf.gather(_rates, rand_ind)
#     sorted_rate = tf.sort(_rates)
#     # u = target_rate - sorted_rate
#     u = sorted_rate - target_rate
#     tau = (tf.range(target_rate.shape[0]) + 1) / target_rate.shape[0]
#     loss = huber_quantile_loss(u, tau, 0.002, dtype=dtype)
#     # loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype) # this kappa value is usually too large and makes the training unstable

#     return loss

# @tf.function
def spike_trimming(spikes, pre_delay=50, post_delay=50, trim=True):
    # remove pre and post delays
    if trim:
        if pre_delay is not None:
            spikes = spikes[:, pre_delay:, :]
        if post_delay is not None and post_delay != 0:
            spikes = spikes[:, :-post_delay, :]
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
    # return df

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


class SpikeRateDistributionTarget(Layer):
    """ Instead of regularization, treat it as a target.
        The main difference is that this class will calculate the loss
        for each subtypes of the neurons."""
    def __init__(self, network, spontaneous_fr=False, uniform_distribution_constraint=False, rate_cost=.5, pre_delay=None, post_delay=None, data_dir='GLIF_network', area='v1', 
                core_mask=None, seed=42, dtype=tf.float32, **kwargs):
        super(SpikeRateDistributionTarget, self).__init__(dtype=dtype, **kwargs)
        self._network = network
        self._rate_cost = rate_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._area = area
        self._core_mask = core_mask
        # if self._core_mask is not None:
        #     self._np_core_mask = self._core_mask.numpy()
        self._data_dir = data_dir
        self._dtype = dtype
        self._seed = seed
        self._uniform_distribution_constraint = uniform_distribution_constraint
        if spontaneous_fr:
            self.neuropixels_feature = 'firing_rate_sp'
        else:
            self.neuropixels_feature = 'Ave_Rate(Hz)'
        # self._target_rates = self.get_neuropixels_firing_rates()
        target_rates = self.get_neuropixels_firing_rates()
        # Build the loss layers
        if self._uniform_distribution_constraint:
            self.uniform_firing_rate_loss_layer = UniformFiringRateLossLayer(target_rates, dtype=self._dtype)
        self.compute_spike_rate_target_loss_layer = FiringRateTargetLossLayer(target_rates, dtype=self._dtype)

    @staticmethod
    def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
        # Sort the original firing rates
        sorted_firing_rates = np.sort(firing_rates)
        # Calculate the empirical cumulative distribution function (CDF)
        # percentiles = (np.arange(firing_rates.shape[-1])).astype(np.float32) / (firing_rates.shape[-1] - 1)
        percentiles = np.linspace(0, 1, sorted_firing_rates.size)
        # Generate random uniform values from 0 to 1
        rate_rd = np.random.RandomState(seed=rnd_seed)
        x_rand = rate_rd.uniform(low=0, high=1, size=n_neurons)
        # Use inverse transform sampling: interpolate the uniform values to find the firing rates
        target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))
        # target_firing_rates = np.interp(x_rand, percentiles, sorted_firing_rates)
        return target_firing_rates

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
        
        features_to_load = ['ecephys_unit_id', 'cell_type', 'firing_rate_sp', 'Ave_Rate(Hz)']
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
                # neuron_ids = neuron_ids[self._np_core_mask[neuron_ids]]
                neuron_ids = tf.boolean_mask(neuron_ids, tf.gather(self._core_mask, neuron_ids))

            # neuron_ids = tf.cast(neuron_ids, dtype=tf.int32)
            target_firing_rates[key]['neuron_ids'] = neuron_ids
            type_n_neurons = len(neuron_ids)
            sorted_target_rates = self.sample_firing_rates(value["rates"], type_n_neurons, self._seed)
            target_firing_rates[key]['sorted_target_rates'] = tf.cast(sorted_target_rates, dtype=self._dtype) 
            target_firing_rates[key]['tuning_angles'] = tf.cast(tf.gather(self._network['tuning_angle'], neuron_ids), dtype=self._dtype)

        return target_firing_rates
    
    def __call__(self, spikes, trim=True):

        # if trim:
        #     if self._pre_delay is not None:
        #         spikes = spikes[:, self._pre_delay:, :]
        #     if self._post_delay is not None and self._post_delay != 0:
        #         spikes = spikes[:, :-self._post_delay, :]

        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        # if self._core_mask is not None:
        #     spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        rates_loss = self.compute_spike_rate_target_loss_layer(spikes)

        if self._uniform_distribution_constraint:
            uniform_firing_rate_loss = self.uniform_firing_rate_loss_layer(spikes)
            rates_loss = tf.add([rates_loss, uniform_firing_rate_loss])

        return rates_loss * self._rate_cost


# Layer to compute the loss term for uniform firing rates across tuning angles
class UniformFiringRateLossLayer(Layer):
    def __init__(self, target_rates, dtype=tf.float32, **kwargs):
        super(UniformFiringRateLossLayer, self).__init__(dtype=dtype, **kwargs)
        self._target_rates = target_rates
        self._dtype = dtype

    def __call__(self, _spikes):
        uniform_distribution_loss = tf.constant(0.0, dtype=self._dtype)
        rates = tf.reduce_mean(_spikes, (0, 1))
        # Define main angles and number of bins
        main_angles = tf.constant([0, 45, 90, 135, 180, 225, 270, 315], dtype=self._dtype)
        num_bins = tf.cast(tf.size(main_angles), dtype=self._dtype)

        for key, value in self._target_rates.items():
            if tf.size(value["neuron_ids"]) != 0:
                # Compute differences and find closest main angles
                diff = tf.abs(tf.expand_dims(value["tuning_angles"], axis=1) - main_angles)
                min_indices = tf.argmin(diff, axis=1)
                _closest_main_angles_type = tf.gather(main_angles, min_indices)
                _rate_type = tf.gather(rates, value["neuron_ids"])
                # Find the average firing rate of each chunk for the current neuron type
                target_avg_rate = tf.reduce_mean(_rate_type)

                ### NOT SURE IF THIS NEW PIECE OF CODE WORKS
                # Use unsorted_segment_mean to calculate average firing rate per angle
                segment_ids = tf.argsort(_closest_main_angles_type)
                sorted_angles = tf.gather(_closest_main_angles_type, segment_ids)
                sorted_rates = tf.gather(_rate_type, segment_ids)

                unique_angles, idx = tf.unique(sorted_angles)
                avg_firing_rates = tf.math.unsorted_segment_mean(sorted_rates, idx, tf.size(unique_angles))

                # Calculate the loss
                loss_per_angle = tf.square(avg_firing_rates - target_avg_rate)
                neuron_type_uniform_distribution_loss = tf.reduce_sum(loss_per_angle) / num_bins

            #     neuron_type_uniform_distribution_loss = tf.constant(0.0, dtype=self._dtype)
            #     for main_angle in main_angles:
            #         indices = tf.where(_closest_main_angles_type == main_angle)
            #         avg_firing_rate = tf.cond(
            #             tf.size(indices) > 0,
            #             lambda: tf.reduce_mean(tf.gather(_rate_type, tf.squeeze(indices, axis=1))),
            #             lambda: tf.constant(0.0, dtype=self._dtype)
            #         )
            #         neuron_type_uniform_distribution_loss += tf.square(avg_firing_rate - target_avg_rate) / num_bins
            # else:
            #     neuron_type_uniform_distribution_loss = tf.constant(0.0, dtype=self._dtype)

            uniform_distribution_loss += neuron_type_uniform_distribution_loss

        return uniform_distribution_loss


class FiringRateTargetLossLayer(Layer):
    def __init__(self, target_rates, dtype=tf.float32, **kwargs):
        super(FiringRateTargetLossLayer, self).__init__(dtype=dtype, **kwargs)
        self._target_rates = target_rates
        self._dtype = dtype

    # @tf.function
    # def compute_spike_rate_distribution_loss(self, _rates, target_rate, dtype=tf.float32):
    #     # Firstly we shuffle the current model rates to avoid bias towardsa particular tuning angles (inherited from neurons ordering in the network)
    #     ind = tf.range(target_rate.shape[0])
    #     rand_ind = tf.random.shuffle(ind)
    #     _rates = tf.gather(_rates, rand_ind)
    #     sorted_rate = tf.sort(_rates)
    #     u = sorted_rate - target_rate
    #     tau = (tf.range(target_rate.shape[0]) + 1) / target_rate.shape[0]
    #     loss = self.huber_quantile_loss(u, tau, 0.002, dtype=dtype)
    #     # loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype) # this kappa value is usually too large and makes the training unstable

    #     return loss

    # @staticmethod
    # def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
    #     tau = tf.cast(tau, dtype)
    #     num = tf.abs(tau - tf.cast(u <= 0, dtype))

    #     branch_1 = num / (2 * kappa) * tf.square(u)
    #     branch_2 = num * (tf.abs(u) - 0.5 * kappa)
    #     return tf.where(tf.abs(u) <= kappa, branch_1, branch_2)

    def compute_spike_rate_distribution_loss(self, _rates, target_rate, dtype=tf.float32):
        # Firstly we shuffle the current model rates to avoid bias towardsa particular tuning angles (inherited from neurons ordering in the network)
        rand_ind = tf.random.shuffle(tf.range(target_rate.shape[0]))
        sorted_rate = tf.sort(tf.gather(_rates, rand_ind))
        u = sorted_rate - target_rate
        tau = (tf.range(target_rate.shape[0]) + 1) / target_rate.shape[0]
        # loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype) # this kappa value is usually too large and makes the training unstable

        return self.huber_quantile_loss(u, tau, 0.002, dtype=dtype)

    @staticmethod
    def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
        tau = tf.cast(tau, dtype)
        abs_u = tf.abs(u)
        num = tf.abs(tau - tf.cast(u <= 0, dtype))

        loss = tf.where(
                        abs_u <= kappa,
                        num / (2 * kappa) * tf.square(u),
                        num * (abs_u - 0.5 * kappa)
                    )
        return loss

    # def __call__(self, _spikes):
    #     total_loss = tf.constant(0.0, dtype=self._dtype)
    #     rates = tf.reduce_mean(_spikes, (0, 1))

    #     for key, value in self._target_rates.items():
    #         if tf.size(value["neuron_ids"]) != 0:
    #             _rate_type = tf.gather(rates, value["neuron_ids"])
    #             target_rate = value["sorted_target_rates"]
    #             loss_type = self.compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=self._dtype)
    #             mean_loss_type = tf.reduce_mean(loss_type)
    #         else:
    #             mean_loss_type = tf.constant(0, dtype=self._dtype)

    #         total_loss += mean_loss_type

    #     return total_loss

    # def __call__(self, _spikes):
    #     total_loss = tf.constant(0.0, dtype=self._dtype)
    #     rates = tf.reduce_mean(_spikes, (0, 1))
    #     cell_count = tf.constant(0, dtype=self._dtype)

    #     for key, value in self._target_rates.items():
    #         cell_count += tf.cast(tf.size(value["neuron_ids"]), dtype=self._dtype)

    #         def true_fn():
    #             _rate_type = tf.gather(rates, value["neuron_ids"])
    #             target_rate = value["sorted_target_rates"]
    #             loss_type = self.compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=self._dtype)
    #             mean_loss_type = tf.reduce_sum(loss_type)
    #             # mean_loss_type = tf.reduce_mean(loss_type)
    #             return mean_loss_type
    #         def false_fn():
    #             return tf.constant(0, dtype=self._dtype)

    #         mean_loss_type = tf.cond(tf.size(value["neuron_ids"]) != 0, 
    #                                  true_fn, 
    #                                  false_fn)

    #         total_loss += mean_loss_type

    #     return total_loss / cell_count

    def __call__(self, _spikes):
        total_loss = tf.constant(0.0, dtype=self._dtype)
        rates = tf.reduce_mean(_spikes, (0, 1))
        num_neurons = tf.constant(0, dtype=self._dtype)

        for key, value in self._target_rates.items():
            neuron_ids = value["neuron_ids"]
            num_neurons_type = tf.cast(tf.size(neuron_ids), dtype=self._dtype)

            if num_neurons_type != 0:
                _rate_type = tf.gather(rates, neuron_ids)
                target_rate = value["sorted_target_rates"]
                loss_type = self.compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=self._dtype)
                total_loss += tf.reduce_sum(loss_type)
                num_neurons += num_neurons_type
            else:
                pass

        total_loss /= num_neurons
        
        return total_loss



class SynchronizationLoss(Layer):
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

    # @tf.function
    # def compute_synchronization_proxy(self, spikes, bin_size_ms=20, time_resolution_ms=1):
    #     # Determine the number of bins
    #     spikes_shape = tf.shape(spikes)
    #     batch_size = spikes_shape[0]
    #     num_time_points = spikes_shape[1]
    #     num_neurons = spikes_shape[2]

    #     bin_size_points = bin_size_ms // time_resolution_ms
    #     num_bins = tf.math.floordiv(num_time_points, bin_size_points)
    #     # just focus on the first batch
    #     spikes = spikes[0]

    #     max_fraction_active = tf.constant(0.0, dtype=self._dtype)
    #     for i in tf.range(num_bins):
    #         start_idx = i * bin_size_points
    #         end_idx = start_idx + bin_size_points
    #         # Extract the bin
    #         bin_data = spikes[start_idx:end_idx, :]
    #         # Determine active neurons in the bin
    #         spike_counts = tf.reduce_sum(bin_data, axis=0)
    #         active_neurons = tf.math.count_nonzero(spike_counts, dtype=tf.float32)
    #         # Compute the fraction of active neurons
    #         fraction_active = active_neurons / tf.cast(num_neurons, tf.float32)
    #         # Update the maximum fraction
    #         max_fraction_active = tf.maximum(max_fraction_active, fraction_active)

    #     return max_fraction_active

    # @tf.function
    def compute_synchronization_proxy(self, spikes, bin_size_ms=20, time_resolution_ms=1):
        spikes_shape = tf.shape(spikes)
        batch_size = spikes_shape[0]
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
            correlation_matrix = tf.linalg.matmul(centered_spikes, centered_spikes, transpose_b=True) / tf.cast(new_seq_len, tf.float32)
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
        mean_pairwise_correlations = mean_pairwise_correlations / tf.cast(batch_size, tf.float32)

        return mean_pairwise_correlations
 

    def __call__(self, spikes, trim=True):

        # spikes = self.spike_trimming(spikes, trim)
        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)
        # spikes = tf.cast(spikes, tf.uint8)

        # if trim:
        #     if self._pre_delay is not None:
        #         spikes = spikes[:, self._pre_delay:, :]
        #     if self._post_delay is not None and self._post_delay != 0:
        #         spikes = spikes[:, :-self._post_delay, :]

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        # # Compute the synchronization proxy
        sync_proxy  = self.compute_synchronization_proxy(spikes)
        # Calculate the synchronization loss
        sync_loss = tf.abs(sync_proxy - self._target_sync)

        # sync_proxy = self.compute_mean_pairwise_correlation(spikes)
        # sync_loss = self._sync_cost * sync_proxy

        return sync_loss * self._sync_cost
    

# class SpikeRateDistributionRegularization:
#     def __init__(self, target_rates, rate_cost=0.5, dtype=tf.float32):
#         self._rate_cost = rate_cost
#         self._target_rates = target_rates
#         self._dtype = dtype

#     def __call__(self, spikes):
#         reg_loss = (
#             compute_spike_rate_distribution_loss(spikes, self._target_rates, dtype=self._dtype)
#             * self._rate_cost
#         )
#         reg_loss = tf.reduce_sum(reg_loss)

#         return reg_loss


class VoltageRegularization:
    def __init__(self, cell, area='v1', voltage_cost=1e-5, dtype=tf.float32, core_mask=None):
        self._voltage_cost = voltage_cost
        self._cell = cell
        self._dtype = dtype
        self._core_mask = core_mask
        self._voltage_offset = tf.cast(self._cell.voltage_offset, dtype)
        self._voltage_scale = tf.cast(self._cell.voltage_scale, dtype)
        if core_mask is not None:
            self._voltage_offset = tf.boolean_mask(self._voltage_offset, core_mask)
            self._voltage_scale = tf.boolean_mask(self._voltage_scale, core_mask)

    # @tf.function
    def __call__(self, voltages):
        if self._core_mask is not None:
            voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)

        voltages -= self._voltage_offset
        voltages /= self._voltage_scale
            
        # voltages = (voltages - self._voltage_offset) / self._voltage_scale
        v_pos = tf.square(tf.nn.relu(voltages - 1.0))
        v_neg = tf.square(tf.nn.relu(-voltages + 1.0))
        # voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
        # voltage_loss = tf.reduce_mean(tf.reduce_mean(v_pos + v_neg, -1)) * self._voltage_cost
        voltage_loss = tf.reduce_mean(v_pos + v_neg) 

        return voltage_loss * self._voltage_cost

    # @tf.function
    # def __call__(self, voltages):
    #     voltages = tf.cast(voltages, self._dtype)
    #     if self._core_mask is not None:
    #         voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)
            
    #     voltages = (voltages - self._voltage_offset) / self._voltage_scale
    #     v_pos = tf.square(tf.nn.relu(voltages - 1.0))
    #     v_neg = tf.square(tf.nn.relu(-voltages + 1.0))
    #     # voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
    #     # voltage_loss = tf.reduce_mean(tf.reduce_mean(v_pos + v_neg, -1)) * self._voltage_cost
    #     voltage_loss = tf.reduce_mean(v_pos + v_neg) 

        
    #     return tf.cast(voltage_loss, tf.float32) * self._voltage_cost


class CustomMeanLayer(Layer):
    def call(self, inputs):
        spike_rates, mask = inputs
        masked_data = tf.boolean_mask(spike_rates, mask)
        return tf.reduce_mean(masked_data)


class OrientationSelectivityLoss:
    def __init__(self, network=None, osi_cost=1e-5, area='v1', pre_delay=None, post_delay=None, dtype=tf.float32, core_mask=None,
                 method="crowd_osi", subtraction_ratio=1.0, layer_info=None):

        self._tuning_angles = tf.constant(network['tuning_angle'], dtype=dtype) 
        self._network = network
        self._osi_cost = osi_cost
        self._area = area
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._dtype = dtype
        self._core_mask = core_mask
        self._method = method
        self._subtraction_ratio = subtraction_ratio # only for crowd_spikes method
        if (self._core_mask is not None) and (self._method == "crowd_spikes" or self._method == "crowd_osi"):
            self._tuning_angles = tf.boolean_mask(self._tuning_angles, self._core_mask)
        
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
            self.node_type_ids = tf.zeros(n_nodes, dtype=tf.int32)
            osi_target_values = []
            dsi_target_values = []
            cell_count_type = []
            for node_type_id, (key, value) in enumerate(self._target_osi_dsi.items()):
                node_ids = value['ids']
                osi_target_values.append(value['OSI'])
                dsi_target_values.append(value['DSI'])
                cell_count_type.append(len(node_ids))
                # update the ndoe_type_ids tensor in positions node_ids with the node_type_id
                self.node_type_ids = tf.tensor_scatter_nd_update(self.node_type_ids, indices=tf.expand_dims(node_ids, axis=1), updates=tf.fill(tf.shape(node_ids), node_type_id))

            self.osi_target_values = tf.constant(osi_target_values, dtype=self._dtype)
            self.dsi_target_values = tf.constant(dsi_target_values, dtype=self._dtype)
            self.cell_count_type = tf.constant(cell_count_type, dtype=self._dtype)
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
        # osi_dsi_df.groupby("cell_type")['OSI'].mean()
        # osi_dsi_df.groupby("cell_type")['OSI'].median()

        osi_target = osi_dsi_df.groupby("cell_type")['OSI'].mean()
        dsi_target = osi_dsi_df.groupby("cell_type")['DSI'].mean()
        # osi_target = osi_dsi_df.groupby("cell_type")['OSI'].median()

        original_pop_names = other_billeh_utils.pop_names(self._network)
        if self._core_mask is not None:
            original_pop_names = original_pop_names[self._core_mask] 
        cell_types = np.array([other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in original_pop_names])
        node_ids = np.arange(len(cell_types))
        cell_ids = {key: node_ids[cell_types == key] for key in set(osi_dsi_df['cell_type'])}

        # osi_target = osi_dsi_df.groupby("cell_type")['OSI'].mean()
        # osi_target = osi_dsi_df.groupby("cell_type")['OSI'].median()
        # osi_dsi_df.groupby("cell_type")['OSI'].median()
        # convert to dict
        # osi_exp_dict = {key: {'OSI': val, 'ids': cell_ids[key]} for key, val in osi_target.to_dict().items()}
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
    
    def neuropixels_fr_loss(self, spikes, angle, trim=True):
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

        # spikes = self.spike_trimming(spikes, trim)
        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)
        # assuming 1 ms bins
        spike_rates = tf.reduce_mean(spikes, axis=[0, 1]) / spikes.shape[1] * 1000
        angle_bins = tf.constant(np.arange(-90, 91, 10), dtype=tf.float32)
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

    def crowd_spikes_loss(self, spikes, angle, trim=True):
        # I need to access the tuning angle. of all the neurons.
        angle = tf.cast(angle, self._dtype)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        # spikes = self.spike_trimming(spikes, trim)
        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        # sum spikes in _z, and multiply with delta_angle.
        mean_spikes = tf.reduce_mean(spikes, axis=[1]) 
        mean_angle = mean_spikes * delta_angle
        # Here, the expected value with random firing to subtract
        # (this prevents the osi loss to drive the firing rates to go to zero.)
        expected_sum_angle = tf.reduce_mean(mean_spikes) * 45

        angle_loss = tf.reduce_mean(tf.abs(mean_angle)) - expected_sum_angle * self._subtraction_ratio

        return angle_loss * self._osi_cost

    # @tf.function
    # def crowd_osi_loss(self, spikes, angle, trim=True, normalizer=None):  
        
    #     # Calculate the angle deltas between current angle and tuning angle
    #     angle = tf.cast(angle[0][0], self._dtype) 
    #     delta_angle = tf.expand_dims(angle, axis=0) - self._tuning_angles
    #     # i want the delta_angle to be within 0-360
    #     # delta_angle = tf.math.floormod(delta_angle, 360)
    #     radians_delta_angle = delta_angle * (pi / 180)

    #     if self._core_mask is not None:
    #         spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
    #     # spikes = self.spike_trimming(spikes, trim)
    #     spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)
    #     # sum spikes in _z, and multiply with delta_angle.
    #     rates = tf.reduce_mean(spikes, axis=[0, 1])

    #     if normalizer is not None:
    #         if self._core_mask is not None:
    #             normalizer = tf.boolean_mask(normalizer, self._core_mask, axis=0)
    #         # Minimum threshold for each element of the normalizer
    #         min_normalizer_value = 0.0005
    #         # Use tf.maximum to ensure each element of normalizer does not fall below min_normalizer_value
    #         normalizer = tf.maximum(normalizer, min_normalizer_value)
    #         rates = rates / normalizer

    #     # Instead of complex numbers, use cosine and sine separately
    #     weighted_osi_cos_responses = rates * tf.math.cos(2.0 * radians_delta_angle)
    #     weighted_dsi_cos_responses = rates * tf.math.cos(radians_delta_angle)
    #     # weighted_osi_sin_responses = rates * tf.math.sin(2.0 * radians_delta_angle)
    #     # weighted_dsi_sin_responses = rates * tf.math.sin(radians_delta_angle)

    #     # For weighted responses, we separately consider the contributions from cosine and sine
    #     # weighted_osi_cos_responses = rates * osi_cos_component
    #     # weighted_dsi_cos_responses = rates * dsi_cos_component
    #     # weighted_osi_sin_responses = rates * osi_sin_component
    #     # weighted_dsi_sin_responses = rates * dsi_sin_component

    #     # Define small epsilon values to avoid differentiability issues when 0 spikes are recorded within the population
    #     total_osi_loss = tf.constant(0.0, dtype=self._dtype)
    #     total_dsi_loss = tf.constant(0.0, dtype=self._dtype)
    #     # penalization_terms = tf.constant(0.0, dtype=self._dtype)
    #     # individual_osi_loss = {}
    #     # individual_dsi_loss = {}
    #     # individual_penalization_loss = {}
    #     # cell_count = tf.constant(0, dtype=self._dtype)
    #     cell_count = tf.cast(tf.size(rates), dtype=self._dtype)

    #     for key, value in self._target_osi_dsi.items():
    #         neuron_type_ids = value['ids']
    #         if tf.size(neuron_type_ids) != 0:
    #             _rates_type = tf.gather(rates, neuron_type_ids)
    #             _weighted_osi_cos_responses_type = tf.gather(weighted_osi_cos_responses, neuron_type_ids)
    #             _weighted_dsi_cos_responses_type = tf.gather(weighted_dsi_cos_responses, neuron_type_ids)
    #             # _weighted_osi_sin_responses_type = tf.gather(weighted_osi_sin_responses, neuron_type_ids)
    #             # _weighted_dsi_sin_responses_type = tf.gather(weighted_dsi_sin_responses, neuron_type_ids)
                
    #             # # Calculate the approximated OSI for the population
    #             # approximated_numerator = tf.sqrt(tf.maximum(tf.square(tf.reduce_sum(_weighted_cos_responses_type)) +
    #             #                                             tf.square(tf.reduce_sum(_weighted_sin_responses_type))
    #             #                                             , epsilon1))
    #             # approximated_denominator = tf.maximum(tf.reduce_sum(_rates_type), epsilon2)

    #             # approximated_osi_numerator = tf.reduce_mean(_weighted_osi_cos_responses_type)
    #             # approximated_dsi_numerator = tf.reduce_mean(_weighted_dsi_cos_responses_type)
    #             # approximated_denominator = tf.maximum(tf.reduce_sum(tf.abs(_weighted_cos_responses_type)), epsilon2)
    #             approximated_denominator = tf.maximum(tf.reduce_mean(_rates_type), self._min_rates_threshold)
    #             osi_approx_type = tf.reduce_mean(_weighted_osi_cos_responses_type) / approximated_denominator
    #             dsi_approx_type = tf.reduce_mean(_weighted_dsi_cos_responses_type) / approximated_denominator
    #             # osi_penalization = tf.math.square(tf.reduce_mean(_weighted_osi_sin_responses_type) / approximated_denominator)
    #             # dsi_penalization = tf.math.square(tf.reduce_mean(_weighted_dsi_sin_responses_type) / approximated_denominator)

    #             # Calculate the OSI loss
    #             osi_loss_type = tf.math.square(osi_approx_type - value['OSI'])
    #             dsi_loss_type = tf.math.square(dsi_approx_type - value['DSI'])

    #             cell_count_type = tf.cast(tf.size(neuron_type_ids), dtype=self._dtype)
    #             total_osi_loss += osi_loss_type * cell_count_type
    #             total_dsi_loss += dsi_loss_type * cell_count_type
    #             # penalization_terms += (osi_penalization + dsi_penalization) #* cell_count_type
    #             # cell_count += cell_count_type
    #             # individual_osi_loss[key] = osi_loss_type * cell_count_type
    #             # individual_dsi_loss[key] = dsi_loss_type * cell_count_type
    #             # individual_penalization_loss[key] = osi_penalization + dsi_penalization

    #         else:
    #             # individual_osi_loss[key] = 0.0
    #             # individual_dsi_loss[key] = 0.0
    #             # individual_penalization_loss[key] = 0.0
    #             pass

    #     total_osi_loss = tf.cond(cell_count > 0, lambda: total_osi_loss / cell_count, lambda: tf.constant(0.0, dtype=self._dtype))
    #     total_dsi_loss = tf.cond(cell_count > 0, lambda: total_dsi_loss / cell_count, lambda: tf.constant(0.0, dtype=self._dtype))

    #     # penalization_terms = penalization_terms / cell_count
            
    #     # return (total_osi_loss + total_dsi_loss + penalization_terms) * self._osi_cost, individual_osi_loss, individual_dsi_loss, individual_penalization_loss
    #     return (total_osi_loss + total_dsi_loss) * self._osi_cost#, individual_osi_loss, individual_dsi_loss


    # @tf.function
    def crowd_osi_loss(self, spikes, angle, trim=True, normalizer=None):  
        # Calculate the angle deltas between current angle and tuning angle
        angle = tf.cast(angle[0][0], self._dtype) 
        delta_angle = tf.expand_dims(angle, axis=0) - self._tuning_angles
        # i want the delta_angle to be within 0-360
        # delta_angle = tf.math.floormod(delta_angle, 360)
        radians_delta_angle = delta_angle * (pi / 180)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
        # spikes = self.spike_trimming(spikes, trim)
        spikes = spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)
        # sum spikes in _z, and multiply with delta_angle.
        rates = tf.reduce_mean(spikes, axis=[0, 1])

        if normalizer is not None:
            if self._core_mask is not None:
                normalizer = tf.boolean_mask(normalizer, self._core_mask, axis=0)
            # Minimum threshold for each element of the normalizer
            min_normalizer_value = 0.0005
            # Use tf.maximum to ensure each element of normalizer does not fall below min_normalizer_value
            normalizer = tf.maximum(normalizer, min_normalizer_value)
            rates = rates / normalizer

        # Instead of complex numbers, use cosine and sine separately
        weighted_osi_cos_responses = rates * tf.math.cos(2.0 * radians_delta_angle)
        weighted_dsi_cos_responses = rates * tf.math.cos(radians_delta_angle)

        approximated_denominator = tf.math.unsorted_segment_mean(rates, self.node_type_ids, self._n_node_types)
        approximated_denominator = tf.maximum(approximated_denominator, self._min_rates_threshold)
        osi_approx_type = tf.math.unsorted_segment_mean(weighted_osi_cos_responses, self.node_type_ids, self._n_node_types) / approximated_denominator
        dsi_approx_type = tf.math.unsorted_segment_mean(weighted_dsi_cos_responses, self.node_type_ids, self._n_node_types) / approximated_denominator

        # Calculate the OSI loss
        osi_loss_type = tf.math.square(osi_approx_type - self.osi_target_values)
        dsi_loss_type = tf.math.square(dsi_approx_type - self.dsi_target_values)

        total_osi_loss = tf.reduce_sum(osi_loss_type * self.cell_count_type) / tf.reduce_sum(self.cell_count_type)
        total_dsi_loss = tf.reduce_sum(dsi_loss_type * self.cell_count_type) / tf.reduce_sum(self.cell_count_type)

        return (total_osi_loss + total_dsi_loss) * self._osi_cost

    
    def __call__(self, spikes, angle, trim, normalizer=None):
        if self._method == "crowd_osi":
            return self.crowd_osi_loss(spikes, angle, trim, normalizer=normalizer)
        elif self._method == "crowd_spikes":
            return self.crowd_spikes_loss(spikes, angle, trim)
        elif self._method == "neuropixels_fr":
            return self.neuropixels_fr_loss(spikes, angle, trim)
        
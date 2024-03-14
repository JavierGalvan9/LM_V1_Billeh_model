import numpy as np
import tensorflow as tf
import pandas as pd
import os


class StiffRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, strength, initial_value):
        super().__init__()
        self._strength = strength
        self._initial_value = tf.Variable(initial_value, trainable=False)

    def __call__(self, x):
        return self._strength * tf.reduce_sum(tf.square(x - self._initial_value))

        
def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
    sorted_firing_rates = np.sort(firing_rates)
    percentiles = (np.arange(firing_rates.shape[-1])).astype(np.float32) / (firing_rates.shape[-1] - 1)
    rate_rd = np.random.RandomState(seed=rnd_seed)
    x_rand = rate_rd.uniform(size=n_neurons)
    target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))
    
    return target_firing_rates


def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
    tau = tf.cast(tau, dtype)
    num = tf.abs(tau - tf.cast(u <= 0, dtype))

    branch_1 = num / (2 * kappa) * tf.square(u)
    branch_2 = num * (tf.abs(u) - 0.5 * kappa)
    return tf.where(tf.abs(u) <= kappa, branch_1, branch_2)

### To calculate the loss of firing rates between neuron types
def compute_spike_rate_target_loss(_spikes, target_rates, dtype=tf.float32):
    # TODO: define this function
    # target_rates is a dictionary that contains all the cell types.
    # I should iterate on them, and add the cost for each one at the end.
    # spikes will have a shape of (batch_size, n_steps, n_neurons)
    # losses = []
    total_loss = tf.constant(0.0, dtype=dtype)
    rates = tf.reduce_mean(_spikes, (0, 1))
    # if core_mask is not None:
    #     core_neurons_ids = np.where(core_mask)[0]

    for i, (key, value) in enumerate(target_rates.items()):
        if tf.size(value["neuron_ids"]) != 0:
            _rate_type = tf.gather(rates, value["neuron_ids"])
            target_rate = value["sorted_target_rates"]
            # if core_mask is not None:
            #     key_core_mask = np.isin(value["neuron_ids"], core_neurons_ids)
            #     neuron_ids =  np.where(key_core_mask)[0]
            #     _rate_type = tf.gather(rates, neuron_ids)
            #     target_rate = value["sorted_target_rates"][key_core_mask]
            # else:
            #     _rate_type = tf.gather(rates, value["neuron_ids"])
            #     target_rate = value["sorted_target_rates"]

            loss_type = compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=dtype)
            mean_loss_type = tf.reduce_mean(loss_type)
        else:
            mean_loss_type = tf.constant(0, dtype=dtype)

        # losses.append(mean_loss_type)
        total_loss += mean_loss_type

    # total_loss = tf.reduce_sum(losses, axis=0)
    return total_loss

def compute_spike_rate_distribution_loss(_rates, target_rate, dtype=tf.float32):
    # ind = tf.range(target_rate.shape[0])
    # rand_ind = tf.random.shuffle(ind)
    # _rate = tf.gather(_rates, rand_ind)
    sorted_rate = tf.sort(_rates)
    u = target_rate - sorted_rate
    # tau = (tf.range(target_rate.shape[0]), dtype) + 1) / target_rate.shape[0]
    tau = (tf.range(target_rate.shape[0]) + 1) / target_rate.shape[0]
    loss = huber_quantile_loss(u, tau, 0.002, dtype=dtype)

    return loss


class SpikeRateDistributionTarget:
    """ Instead of regularization, treat it as a target.
        The main difference is that this class will calculate the loss
        for each subtypes of the neurons."""
    def __init__(self, network, rate_cost=.5, pre_delay=None, post_delay=None, data_dir='GLIF_network', area='v1', 
                core_mask=None, seed=0, dtype=tf.float32):
        self._network = network
        self._rate_cost = rate_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._area = area
        self._core_mask = core_mask
        if self._core_mask is not None:
            self._np_core_mask = self._core_mask.numpy()
        self._data_dir = data_dir
        self._dtype = dtype
        self._seed = seed
        self._target_rates = self.get_neuropixels_firing_rates()

    def process_neuropixels_data(self, path=''):
        # Load data
        neuropixels_data_path = f'Neuropixels_data/cortical_metrics_1.4.csv'
        df_all = pd.read_csv(neuropixels_data_path, sep=" ")
        billeh_to_neuropixels_area_mapping = {'v1':'VISp', 'lm':'VISl'}
        # Exc and PV have sufficient number of cells, so we'll filter out non-V1 Exc and PV.
        # SST and VIP are small populations, so let's keep also non-V1 neurons
        exclude = (df_all["cell_type"].isnull() | df_all["cell_type"].str.contains("EXC") | df_all["cell_type"].str.contains("PV")) \
                & (df_all["ecephys_structure_acronym"] != billeh_to_neuropixels_area_mapping[self._area])
        df = df_all[~exclude]
        print(f"Original: {df_all.shape[0]} cells,   filtered: {df.shape[0]} cells")

        # Some cells have very large values of RF. They are likely not-good fits, so ignore.
        df.loc[(df["width_rf"] > 100), "width_rf"] = np.nan
        df.loc[(df["height_rf"] > 100), "height_rf"] = np.nan

        # Save the processed table
        df.to_csv(f'Neuropixels_data/{self._area}_OSI_DSI_DF.csv', sep=" ", index=False)
        return df

    def get_neuropixels_firing_rates(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        neuropixels_data_path = f'Neuropixels_data/{self._area}_OSI_DSI_DF.csv'
        if not os.path.exists(neuropixels_data_path):
            np_df = self.process_neuropixels_data(path=neuropixels_data_path)
        else:
            np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ")

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
                            reversed_query_mapping[cell_type]: np.sort(np.append(subdf["Ave_Rate(Hz)"].dropna().values / 1000, 0))
                            for cell_type, subdf in np_df.groupby("cell_type")
                        }

        # Identify node_type_ids for each population query
        pop_ids = {query: area_node_types[area_node_types.pop_name.str.contains(query)].index.values for query in query_mapping.keys()}

        # Create a dictionary with rates and IDs
        target_firing_rates = {pop_query: {'rates': type_rates_dict[pop_query], 'ids': pop_ids[pop_query]} for pop_query in pop_ids.keys()}
        
        for i, (key, value) in enumerate(target_firing_rates.items()):
            # identify tne ids that are included in value["ids"]
            neuron_ids = np.where(np.isin(self._network["node_type_ids"], value["ids"]))[0]
            if self._core_mask is not None:
                # if core_mask is not None, use only neurons in the core
                neuron_ids = neuron_ids[self._np_core_mask[neuron_ids]]

            neuron_ids = tf.cast(neuron_ids, dtype=tf.int32)
            target_firing_rates[key]['neuron_ids'] = neuron_ids
            type_n_neurons = len(neuron_ids)
            sorted_target_rates = sample_firing_rates(value["rates"], type_n_neurons, self._seed)
            target_firing_rates[key]['sorted_target_rates'] = tf.cast(sorted_target_rates, dtype=tf.float32) 

        return target_firing_rates


    def __call__(self, spikes):
        if self._pre_delay is not None:
            spikes = spikes[:, self._pre_delay:, :]
        if self._post_delay is not None and self._post_delay != 0:
            spikes = spikes[:, :-self._post_delay, :]

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        reg_loss = compute_spike_rate_target_loss(spikes, self._target_rates, dtype=self._dtype) 
        
        return reg_loss * self._rate_cost


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
        if core_mask is None:
            self._voltage_offset = self._cell.voltage_offset
            self._voltage_scale = self._cell.voltage_scale
        else:
            self._voltage_offset = tf.boolean_mask(self._cell.voltage_offset, core_mask)
            self._voltage_scale = tf.boolean_mask(self._cell.voltage_scale, core_mask)

    def __call__(self, voltages):
        if self._core_mask is not None:
            voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)
            
        voltage_32 = (voltages - self._voltage_offset) / self._voltage_scale
        v_pos = tf.square(tf.nn.relu(voltage_32 - 1.0))
        v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.0))
        voltage_loss = tf.reduce_mean(tf.reduce_sum(v_pos + v_neg, -1)) * self._voltage_cost
        return voltage_loss


class OrientationSelectivityLoss:
    def __init__(self, tuning_angles, osi_cost=1e-5, area='v1', pre_delay=None, post_delay=None, dtype=tf.float32, core_mask=None):
        self._tuning_angles = tuning_angles
        self._osi_cost = osi_cost
        self._area = area
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._dtype = dtype
        self._core_mask = core_mask
        if self._core_mask is not None:
            self._tuning_angles = tf.boolean_mask(self._tuning_angles, self._core_mask)

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

    def shinya_version(self, spikes, angle):
        # I need to access the tuning angle. of all the neurons.
        angle = tf.cast(angle, self._dtype)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
        if self._pre_delay is not None:
            spikes = spikes[:, self._pre_delay:, :]
        if self._post_delay is not None and self._post_delay != 0:
            spikes = spikes[:, :-self._post_delay, :]

        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        # sum spikes in _z, and multiply with delta_angle.
        mean_spikes = tf.reduce_mean(spikes, axis=[1]) 
        mean_angle = mean_spikes * delta_angle
        # Here, the expected value with random firing to subtract
        # (this prevents the osi loss to drive the firing rates to go to zero.)
        expected_sum_angle = tf.reduce_mean(mean_spikes) * 45
        
        angle_loss = tf.reduce_mean(tf.abs(mean_angle)) - expected_sum_angle

        return tf.abs(angle_loss) * self._osi_cost
        # return angle_loss * self._osi_cost

    def javi_version(self, spikes, angle):
        angle = tf.cast(angle, self._dtype) 
        delta_angle = tf.expand_dims(angle, axis=1) -  self._tuning_angles
        # delta_angle = self._tuning_angles - tf.expand_dims(angle, axis=1)
        # i want the delta_angle to be within 0-360
        delta_angle = tf.math.floormod(delta_angle, tf.constant(360, dtype=self._dtype))

        delta_angle = delta_angle * (np.pi / 180)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
        if self._pre_delay is not None:
            spikes = spikes[:, self._pre_delay:, :]
        if self._post_delay is not None and self._post_delay != 0:
            spikes = spikes[:, :-self._post_delay, :]

        # sum spikes in _z, and multiply with delta_angle.
        mean_spikes = tf.reduce_mean(spikes, axis=[1]) 

        # Convert mean_spikes to a complex tensor with zero imaginary part
        mean_spikes = tf.cast(mean_spikes, tf.complex64)

        # Calculate weighted responses for OSI numerator
        # Adjust for preferred orientation by incorporating e^(2i(theta - theta_pref))
        weighted_responses_numerator = mean_spikes * tf.exp(tf.complex(0.0, 2.0 * delta_angle))
        approximated_numerator = tf.reduce_sum(weighted_responses_numerator)
        
        # Calculate denominator as the sum of mean_spikes
        approximated_denominator = tf.reduce_sum(mean_spikes)
        
        # Calculate OSI approximation
        osi_approx = tf.abs(approximated_numerator / tf.cast(approximated_denominator, tf.complex64))

        return tf.abs(osi_approx - 1) * self._osi_cost

    def original_version(self, spikes, angle):
        # I need to access the tuning angle. of all the neurons.
        angle = tf.cast(angle, self._dtype)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)

        if self._pre_delay is not None:
            spikes = spikes[:, self._pre_delay:, :]
        if self._post_delay is not None and self._post_delay != 0:
            spikes = spikes[:, :-self._post_delay, :]

        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        # sum spikes in _z, and multiply with delta_angle.
        sum_angle = tf.reduce_mean(spikes, axis=[1]) * delta_angle
        # make a huber loss for this.
        # angle_loss = tf.keras.losses.Huber(delta=1, reduction=tf.keras.losses.Reduction.SUM)(sum_angle, tf.zeros_like(sum_angle))
        angle_loss = tf.reduce_mean(tf.abs(sum_angle))
        # it might be nice to add regularization of weights
        # rec_weight_loss = rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)
        return angle_loss * self._osi_cost
        
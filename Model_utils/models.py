import numpy as np
import tensorflow as tf
from numba import njit

# Define a custom gradient for the spike function.
# Diverse functions can be used to define the gradient.
# Here we provide variations of this functions depending on
# the gradient type and the precision of the input tensor.
def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude

def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)

# def slayer_pseudo(v_scaled, sigma, amplitude):
#     return tf.math.exp(-sigma * tf.abs(v_scaled)) * amplitude


@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad

@tf.custom_gradient
def spike_gauss_16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad

@tf.custom_gradient
def spike_gauss_b16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(sigma), tf.zeros_like(amplitude)]

    return tf.identity(z_, name='spike_gauss'), grad

# @tf.custom_gradient
# def spike_slayer(v_scaled, sigma, amplitude):
#     z_ = tf.greater(v_scaled, 0.)
#     z_ = tf.cast(z_, tf.float32)

#     def grad(dy):
#         de_dz = dy
#         dz_dv_scaled = slayer_pseudo(v_scaled, sigma, amplitude)

#         de_dv_scaled = de_dz * dz_dv_scaled

#         return [de_dv_scaled,
#                 tf.zeros_like(sigma), tf.zeros_like(amplitude)]

#     return tf.identity(z_, name='spike_slayer'), grad


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy # ?? we can get de_dz directely from here?
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def spike_function_16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def spike_function_b16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


def exp_convolve(tensor, decay=.8, reverse=False, initializer=None, axis=0):
    rank = len(tensor.get_shape())
    perm = np.arange(rank)
    perm[0], perm[axis] = perm[axis], perm[0]
    tensor = tf.transpose(tensor, perm)

    if initializer is None:
        initializer = tf.zeros_like(tensor[0])

    def scan_fun(_acc, _t):
        return _acc * decay + _t

    filtered = tf.scan(scan_fun, tensor, reverse=reverse,
                       initializer=initializer)

    filtered = tf.transpose(filtered, perm)
    return filtered

@njit
def process_receptors(postsynaptic_indices, receptor_ids, _n_neurons):
    total_receptors = receptor_ids.max() + 1  # Assuming receptor IDs start from 0
    receptors_per_neuron = np.zeros((_n_neurons, total_receptors), dtype=np.bool_)
    # Populate the receptors_per_neuron array
    for i in range(len(postsynaptic_indices)):
        neuron_id = postsynaptic_indices[i]
        receptor_id = receptor_ids[i]
        receptors_per_neuron[neuron_id, receptor_id] = True
    # Find the maximum number of receptors per neuron
    max_receptors_per_neuron = np.sum(receptors_per_neuron, axis=1).max()

    # Initialize mapping arrays
    new_receptor_id_to_old_receptor_id = np.full((_n_neurons, max_receptors_per_neuron), -1, dtype=np.int32)
    old_receptor_id_to_new_receptor_id = np.full((_n_neurons, total_receptors), -1, dtype=np.int32)
    # Populate the mapping arrays
    for neuron_id in range(_n_neurons):
        idx = 0
        for receptor_id in range(total_receptors):
            if receptors_per_neuron[neuron_id, receptor_id]:
                new_receptor_id_to_old_receptor_id[neuron_id, idx] = receptor_id
                old_receptor_id_to_new_receptor_id[neuron_id, receptor_id] = idx
                idx += 1

    return max_receptors_per_neuron, new_receptor_id_to_old_receptor_id, old_receptor_id_to_new_receptor_id

def make_pre_ind_table(indices, n_source_neurons=197613):
    """
    This function creates a table that maps presynaptic indices to 
    the indices of the recurrent_indices tensor using a RaggedTensor.
    This approach ensures that every presynaptic neuron, even those with no
    postsynaptic connections, has an entry in the RaggedTensor.
    """
    pre_inds = indices[:, 1]
    n_syn = pre_inds.shape[0]
    # Since pre_inds may not be sorted, we sort them along with synapse_indices
    sorted_pre_inds, sorted_synapse_indices = tf.math.top_k(-pre_inds, k=n_syn)
    sorted_pre_inds = -sorted_pre_inds  # Undo the negation to get the sorted pre_inds
    # Count occurrences (out-degrees) for each presynaptic neuron using bincount
    counts = tf.math.bincount(sorted_pre_inds, minlength=n_source_neurons)
    # Create row_splits that covers all presynaptic neurons (0 to n_source_neurons)
    row_splits = tf.concat([[0], tf.cumsum(counts)], axis=0)
    # Create the RaggedTensor with empty rows for missing neurons
    rt = tf.RaggedTensor.from_row_splits(sorted_synapse_indices, row_splits)

    return rt

def get_new_inds_table(indices, non_zero_cols, pre_ind_table):
    """Optimized function that prepares new sparse indices tensor."""
    # Gather the rows corresponding to the non_zero_cols
    selected_rows = tf.gather(pre_ind_table, non_zero_cols)
    # Flatten the selected rows to get all_inds
    all_inds = selected_rows.flat_values
    # Gather from indices using all_inds
    new_indices = tf.gather(indices, all_inds)

    return new_indices, all_inds


class BackgroundNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, cell, batch_size, seq_len,
                 bkg_firing_rate=1000, n_bkg_units=1, 
                 n_bkg_connections = 1, 
                  **kwargs):
        super().__init__(**kwargs)
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._bkg_firing_rate = bkg_firing_rate
        self._n_bkg_units = n_bkg_units
        self._n_bkg_connections = n_bkg_connections

        # Initialize weights and indices
        self._bkg_weights = {'v1': None, 'lm': None}
        self._bkg_indices = {'v1': None, 'lm': None}
        self._dense_shape = {'v1': None, 'lm': None}
        self._initialize_background_inputs(cell)

    def _initialize_background_inputs(self, cell):
        # Create connectivity and assign weights for 'v1' and 'lm' areas with new pool of 100 Poisson sources
        for column in ['v1', 'lm']:
            original_weights = cell.__getattribute__(column).bkg_input_weights
            original_indices = cell.__getattribute__(column).bkg_input_indices
            original_dense_shape = cell.__getattribute__(column).bkg_input_dense_shape

            if self._n_bkg_connections == 1:
                indices = original_indices
                weights = original_weights
            else:
                # Generate random connections
                new_bkg_indices = tf.random.uniform(shape=(original_indices.shape[0], self._n_bkg_connections), minval=0, maxval=self._n_bkg_units, dtype=tf.int64)
                indices = tf.reshape(tf.stack([tf.repeat(original_indices[:, 0], self._n_bkg_connections), tf.reshape(new_bkg_indices, [-1])], axis=1), [-1, 2])
                # this implementation allows a neuron to establish more than one connection to a single BKG unit
                # Repeat weights for each connection
                weights = tf.repeat(original_weights, self._n_bkg_connections)
            
            # Create a new constraint based on the new weights
            new_bkg_input_weight_positive = tf.Variable(weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
            new_constraint = SignedConstraint(new_bkg_input_weight_positive)
            cell.__getattribute__(column).bkg_input_weights = tf.Variable(weights, 
                                                                        name=column+'_rest_of_brain_weights', 
                                                                        constraint=new_constraint,
                                                                        dtype=original_weights.dtype,
                                                                        trainable=original_weights.trainable)

            self._bkg_weights[column] = cell.__getattribute__(column).bkg_input_weights
            self._bkg_indices[column] = tf.Variable(indices, trainable=False, dtype=tf.int64)
            self._dense_shape[column] = (original_dense_shape[0],  self._n_bkg_units)

    def calculate_bkg_i_in(self, inputs, column='v1'):
        # Define the sparse weight matrix (need to be defined here for the gradient to work)
        sparse_w_in = tf.sparse.SparseTensor(
            self._bkg_indices[column],
            self._bkg_weights[column], 
            self._dense_shape[column]
        )
        i_in = tf.sparse.sparse_dense_matmul(
                                            sparse_w_in, 
                                            inputs, 
                                            adjoint_b=True
                                            )
        # Optionally cast the output back to float16
        if i_in.dtype != self.compute_dtype:
            i_in = tf.cast(i_in, dtype=self.compute_dtype)

        return i_in

    def call(self, inp):
        seq_len = tf.shape(inp)[1]

        rest_of_brain = tf.random.poisson(shape=(self._batch_size * seq_len, self._n_bkg_units), 
                                          lam=self._bkg_firing_rate/1000, 
                                          dtype=self.variable_dtype) # this implementation is slower but allows to produce proper Poisson values (not just 0's and 1's)
        # rest_of_brain = tf.cast(tf.random.uniform(
        #         (self._batch_size * seq_len, self._n_bkg_units)) < self._bkg_firing_rate * .001, 
        #         tf.float32) # (1, 600, 100)
        # rest_of_brain = tf.reshape(rest_of_brain, (self._batch_size * seq_len, self._n_bkg_units))

        noise_inputs = {'v1': None, 'lm': None}
        for column in noise_inputs.keys():
            noise_input = self.calculate_bkg_i_in(rest_of_brain, column=column)
            noise_input = tf.transpose(noise_input)
            noise_inputs[column] = tf.reshape(noise_input, (self._batch_size, seq_len, -1))
        
        cat_noise_inputs = tf.concat([noise_inputs['v1'], noise_inputs['lm']], axis=-1)

        return cat_noise_inputs


# class LGNInputLayer(tf.keras.layers.Layer):
#     def __init__(self, indices, weights, dense_shape, dtype=tf.float32, **kwargs):
#         super().__init__(**kwargs)
#         self._indices = indices
#         self._input_weights = weights
#         self._dense_shape = dense_shape
#         self._dtype = dtype
#         # Define a threshold that determines whether to compute the sparse
#         # matrix multiplication directly or split it into smaller batches in a GPU.
#         # The value is calculated to ensure that output.shape[1] * nnz(a) > 2^31, 
#         # where output.shape[1] is the time_length, and nnz(a) is the number of non-zero elements in the sparse matrix.
#         nnz_sparse_matrix = self._indices.shape[0]
#         self._max_batch = int(2**31 / nnz_sparse_matrix)
        
#     def calculate_i_in(self, inputs):
#         # Define the sparse weight matrix (need to be defined here for the gradient to work)
#         sparse_w_in = tf.sparse.SparseTensor(
#             indices=self._indices,
#             values=self._input_weights,
#             dense_shape=self._dense_shape,
#         )

#         i_in = tf.sparse.sparse_dense_matmul(
#                                                 sparse_w_in,
#                                                 inputs,
#                                                 adjoint_b=True
#                                             )
#         return i_in

#     def call(self, inp, verbose=False):
#         # replace any None values in the shape of inp with the actual values obtained from the input tensor at runtime (tf.shape(inp)).
#         # This is necessary because the SparseTensor multiplication operation requires a fully defined shape.
#         inp_shape = inp.get_shape().as_list() # [None, 600, 17400]
#         shp = [dim if dim is not None else tf.shape(inp)[i] for i, dim in enumerate(inp_shape)]
#         batch_size = shp[0]
#         seq_len = shp[1]
#         input_dim = shp[2]
#         total_batch_size = batch_size * seq_len
        
#         # Reshape inp to (total_batch_size, input_dim)
#         inp = tf.reshape(inp, (total_batch_size, input_dim))

#         if total_batch_size < self._max_batch:
#             # Process the input directly
#             input_current = self.calculate_i_in(inp)  # shape: (output_dim, total_batch_size)
#             input_current = tf.transpose(input_current)  # shape: (total_batch_size, output_dim)
#         else: 
#             # Need to process in chunks
#             chunk_size = self._max_batch
#             num_chunks = (total_batch_size + chunk_size - 1) // chunk_size
#             result_array = tf.TensorArray(dtype=self.compute_dtype, size=num_chunks, infer_shape=False)
#             for i in range(num_chunks):
#                 start_idx = i * chunk_size
#                 end_idx = tf.minimum((i + 1)* chunk_size, total_batch_size)
#                 chunk = inp[start_idx:end_idx, :]
#                 partial_input_current = self.calculate_i_in(chunk)  # shape: (output_dim, chunk_size)
#                 partial_input_current = tf.transpose(partial_input_current)  # shape: (chunk_size, output_dim)
#                 result_array = result_array.write(i, partial_input_current)
#             # Concatenate the partial results
#             input_current = result_array.concat()  # shape: (total_batch_size, output_dim)

#         # Reshape properly the input current
#         input_current = tf.reshape(input_current, (batch_size, seq_len, -1)) # New shape (1, 3000, 333170)

#         return input_current


class LGNInputLayerCell(tf.keras.layers.Layer):
    # This implementation is slightly slower, but if there were more connections to the LGN, it would be faster
    # as it happens with the new V1 model
    def __init__(self, indices, weights, dense_shape, **kwargs):
        super().__init__(**kwargs)
        self._indices = indices
        self._input_weights = weights
        self._dense_shape = dense_shape
        # Precompute the synapses table
        self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=dense_shape[1])

    @property
    def state_size(self):
        # No states are maintained in this cell
        return []

    # def build(self, input_shape):
    #     # If you have any trainable variables, initialize them here
    #     pass

    # @tf.function
    def call(self, inputs_t, states):
        # inputs_t: Shape [batch_size, input_dim]
        # batch_size = tf.shape(inputs_t)[0]
        # Compute the input current for the timestep
        non_zero_cols = tf.where(inputs_t > 0)[:, 1]
        new_indices, inds = get_new_inds_table(self._indices, non_zero_cols, self.pre_ind_table)
        # Sort the segment IDs and corresponding data
        sorted_indices = tf.argsort(new_indices[:, 0])
        sorted_segment_ids = tf.gather(new_indices[:, 0], sorted_indices)
        sorted_inds = tf.gather(inds, sorted_indices)
        # Get the weights for each active synapse
        sorted_data = tf.gather(self._input_weights, sorted_inds, axis=0)
        # Calculate the total LGN input current received by each neuron
        i_in = tf.math.unsorted_segment_sum(
            sorted_data,
            sorted_segment_ids,
            num_segments=self._dense_shape[0]
        )

        # Optionally cast the output back to float16
        if i_in.dtype != self.compute_dtype:
            i_in = tf.cast(i_in, dtype=self.compute_dtype)

        # Add batch dimension
        i_in = tf.expand_dims(i_in, axis=0)  # Shape: [1, n_post_neurons, n_syn_basis]
        # i_in = tf.reshape(i_in, [batch_size, -1])
                
        # Since no states are maintained, return empty state
        return i_in, []


class LGNInputLayer(tf.keras.layers.Layer):
    """
    Calculates input currents from the LGN by processing one timestep at a time using a custom RNN cell.
    """
    def __init__(self, indices, weights, dense_shape,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_cell = LGNInputLayerCell(
            indices, weights, dense_shape,
            **kwargs
        )
        # Create the input RNN layer with the custom cell to recursively process all the inputs by timesteps
        self.input_rnn = tf.keras.layers.RNN(self.input_cell, return_sequences=True, return_state=False, name='lgn_rsnn')

    def call(self, inputs, **kwargs):
        # inputs: Shape [batch_size, seq_len, input_dim]
        input_current = self.input_rnn(inputs, **kwargs)  # Outputs: [batch_size, seq_len, n_postsynaptic_neurons]
        return input_current


class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        self._positive = positive

    def __call__(self, w):
        sign_corrected_w = tf.where(
            self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
        return sign_corrected_w


class SparseSignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask, positive):
        self._mask = mask
        self._positive = positive

    def __call__(self, w):
        sign_corrected_w = tf.where(
            self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
        return tf.where(self._mask, sign_corrected_w, tf.zeros_like(sign_corrected_w))
   

class BillehColumn(tf.keras.layers.Layer):

    def __init__(
        self, 
        network, 
        lgn_input, 
        bkg_input,
        dt=1., 
        gauss_std=.5, 
        dampening_factor=.3,
        recurrent_dampening_factor=.5,
        input_weight_scale=1., 
        recurrent_weight_scale=1., 
        interarea_weight_scale=1.,
        lr_scale=1., 
        max_delay=5, 
        batch_size=12, 
        pseudo_gauss=False, 
        spike_gradient=False,
        train_recurrent=True, 
        train_input=True, 
        train_noise=False,
        train_interarea=True, 
        hard_reset=False, 
        connected_areas=True,
        connected_recurrent_connections=True, 
        connected_noise=True,
        **kwargs
        ):

        super().__init__(**kwargs)

        print(f'###### COLUMN {self.name} ######')

        _params = dict(network['node_params'])
        # Rescale the voltages to have them near 0, as we wanted the effective step size 
        # for the weights to be normalized when learning (weights are scaled similarly)
        voltage_scale = _params['V_th'] - _params['E_L']
        voltage_offset = _params['E_L']
        _params['V_th'] = (_params['V_th'] - voltage_offset) / voltage_scale
        _params['E_L'] = (_params['E_L'] - voltage_offset) / voltage_scale
        _params['V_reset'] = (_params['V_reset'] - voltage_offset) / voltage_scale
        _params['asc_amps'] = _params['asc_amps'] / voltage_scale[..., None]   # _params['asc_amps'] has shape (111, 2)
        # Define the other model variables
        self._node_type_ids = np.array(network['node_type_ids'])
        self._dt = tf.constant(dt, self.compute_dtype)
        self._recurrent_dampening = tf.constant(recurrent_dampening_factor, self.compute_dtype)
        self._dampening_factor = tf.constant(dampening_factor, self.compute_dtype)
        self._pseudo_gauss = pseudo_gauss
        self._lr_scale = tf.constant(lr_scale, dtype=self.compute_dtype)
        self._spike_gradient = spike_gradient
        self._hard_reset = hard_reset
        self._connected_areas = connected_areas
        self._connected_recurrent_connections = connected_recurrent_connections
        self._connected_noise = connected_noise
        self._n_neurons = int(network['n_nodes'])
        self._gauss_std = tf.constant(gauss_std, self.compute_dtype)
        # determine the membrane time decay constant
        tau = _params['C_m'] / _params['g'] 
        membrane_decay = np.exp(-dt / tau)
        current_factor = 1 / _params['C_m'] * (1 - membrane_decay) * tau
        
        # create a new variable with all the postsynatpic indices concatenated: network["synapses"]["indices"], lgn_input["indices"],...
        all_interarea_postsynaptic_indices = np.concatenate([network['interarea_synapses'][order]['indices'][:, 0] for order in network['interarea_synapses'].keys()])
        all_interarea_receptor_ids = np.concatenate([network['interarea_synapses'][order]['receptor_ids'] for order in network['interarea_synapses'].keys()])
        if lgn_input is not None:
            all_postsynaptic_indices = np.concatenate([network["synapses"]["indices"][:, 0], lgn_input["indices"][:, 0], bkg_input["indices"][:, 0], all_interarea_postsynaptic_indices], axis=0)
            all_receptor_ids = np.concatenate([network["synapses"]["receptor_ids"], lgn_input["receptor_ids"], bkg_input["receptor_ids"], all_interarea_receptor_ids], axis=0)
        else:
            all_postsynaptic_indices = np.concatenate([network["synapses"]["indices"][:, 0], bkg_input["indices"][:, 0], all_interarea_postsynaptic_indices], axis=0)
            all_receptor_ids = np.concatenate([network["synapses"]["receptor_ids"], bkg_input["receptor_ids"], all_interarea_receptor_ids], axis=0)

        self._n_receptors, new_receptor_id_to_old_receptor_id, old_receptor_id_to_new_receptor_id = process_receptors(all_postsynaptic_indices, all_receptor_ids, self._n_neurons)
        del all_postsynaptic_indices, all_receptor_ids

        # Determine the synaptic dynamic parameters for each of the 4 receptors
        tau_syns = np.array([5.5, 8.5, 2.8, 5.8])
        syn_decay = np.exp(-dt / tau_syns)
        syn_decay = tf.constant(syn_decay, dtype=self.compute_dtype)
        syn_decay = tf.gather(syn_decay, new_receptor_id_to_old_receptor_id, axis=0)
        self.syn_decay = tf.reshape(syn_decay, [1, -1])
        psc_initial = np.e / tau_syns
        psc_initial = tf.constant(psc_initial, dtype=self.compute_dtype)
        psc_initial = tf.gather(psc_initial, new_receptor_id_to_old_receptor_id, axis=0)
        self.psc_initial = tf.reshape(psc_initial, [1, -1])

        # Find the maximum delay in the network
        self.max_delay = int(np.round(np.min([np.max(network['synapses']['delays']), max_delay])))
       
        def _gather(prop):
            return tf.gather(prop, self._node_type_ids)
    
        def _f(_v, trainable=False):
            return tf.Variable(tf.cast(_gather(_v), self.compute_dtype), trainable=trainable)

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        def custom_val(_v, trainable=False):
            _v = tf.Variable(tf.cast(inv_sigmoid(_gather(
                _v)), self.compute_dtype), trainable=trainable)

            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        # Gather the neuron parameters for every neuron
        self.t_ref = _f(_params['t_ref'])
        self.v_reset = _f(_params['V_reset'])
        self.asc_amps = _f(_params['asc_amps'], trainable=False)
        _k = tf.cast(_params['k'], self.compute_dtype)
        # inverse sigmoid of the adaptation rate constant (1/ms)
        param_k, param_k_read = custom_val(_k, trainable=False) # ?? what is this doing?
        k = param_k_read()
        self.exp_dt_k = tf.exp(-self._dt * k)

        self.v_th = _f(_params["V_th"])
        self.v_gap = self.v_reset - self.v_th
        e_l = _f(_params["E_L"])
        self.normalizer = self.v_th - e_l
        param_g = _f(_params["g"])
        self.gathered_g = param_g * e_l

        self.decay = _f(membrane_decay)
        self.current_factor = _f(current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)

        ### Network recurrent connectivity ###
        indices = np.array(network["synapses"]["indices"])
        weights = np.array(network["synapses"]["weights"])
        dense_shape = np.array(network["synapses"]["dense_shape"])
        receptor_ids = np.array(network["synapses"]["receptor_ids"])
        delays = np.array(network["synapses"]["delays"])

        # Scale down the recurrent weights
        weights = (weights/voltage_scale[self._node_type_ids[indices[:, 0]]]) # scale down the weights based on the postsynaptic neuron
        
        # Use the maximum delay to clip the synaptic delays
        delays = np.round(np.clip(delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the presynaptic neuron indices
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)
                
        # Introduce the receptor ids in the postsynaptic neuron indices
        new_receptor_ids = old_receptor_id_to_new_receptor_id[indices[:, 0], receptor_ids]
        indices[:, 0] = indices[:, 0] * self._n_receptors + new_receptor_ids

        self.recurrent_dense_shape = self._n_receptors * dense_shape[0], self.max_delay * dense_shape[1] 
        #the first column (presynaptic neuron) has size n_neurons*n_max_receptors and the second column (postsynaptic neuron) has size max_delay*n_neurons

        # Define the Tensorflow variables
        self.recurrent_indices = tf.Variable(indices, dtype=tf.int64, trainable=False)
        self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=dense_shape[1])

        # Set the sign of the connections (exc or inh)
        recurrent_weight_positive = tf.Variable(
            weights >= 0., name='recurrent_weights_sign', trainable=False)

        # Scale the weights
        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale,
            name=self.name+'_sparse_recurrent_weights',
            constraint=SignedConstraint(recurrent_weight_positive),
            trainable=train_recurrent,
            dtype=self.variable_dtype
        ) # shape = (n_synapses,)

        print(f"    > # Recurrent synapses: {len(indices)}")
        del indices, weights, dense_shape, receptor_ids, delays

        ### LGN input connectivity ###
        if lgn_input is not None:
            self.lgn_input_dense_shape = (self._n_neurons * self._n_receptors, lgn_input["n_inputs"],)
            input_indices = np.array(lgn_input["indices"])
            input_weights = np.array(lgn_input["weights"])
            input_receptor_ids = np.array(lgn_input["receptor_ids"])
            input_delays = np.array(lgn_input["delays"])

            # Scale down the input weights
            input_weights = (input_weights/ voltage_scale[self._node_type_ids[input_indices[:, 0]]])
            input_delays = np.round(np.clip(input_delays, dt, self.max_delay)/dt).astype(np.int32)
            # Introduce the delays in the postsynaptic neuron indices
            # input_indices[:, 1] = input_indices[:, 1] + self._n_neurons * (input_delays - 1)

            #the first column (presynaptic neuron) has size n_neurons and the second column (postsynaptic neuron) has size max_delay*n_neurons
            new_input_receptor_ids = old_receptor_id_to_new_receptor_id[input_indices[:, 0], input_receptor_ids]
            input_indices[:, 0] = input_indices[:, 0] * self._n_receptors + new_input_receptor_ids
            self.input_indices = tf.Variable(input_indices, trainable=False, dtype=tf.int64)

            input_weight_positive = tf.Variable(
                input_weights >= 0., name=self.name+'_input_weights_sign', trainable=False)
            
            self.input_weight_values = tf.Variable(
                input_weights * input_weight_scale / lr_scale, 
                name=self.name+'_sparse_input_weights',
                constraint=SignedConstraint(input_weight_positive),
                trainable=train_input,
                dtype=self.variable_dtype)

            print(f"    > # LGN input synapses {len(input_indices)}")
            del input_indices, input_weights, input_receptor_ids, input_delays

        else:
            self.input_indices, self.input_weight_values, self.input_dense_shape = None, None, None
            print(f'    > # LGN to {self.name} input synapses 0')
        
        ### BKG input connectivity ###
        self.bkg_input_dense_shape = (self._n_neurons * self._n_receptors, bkg_input["n_inputs"],)
        bkg_input_indices = np.array(bkg_input['indices'])
        bkg_input_weights = np.array(bkg_input['weights'])
        bkg_input_receptor_ids = np.array(bkg_input['receptor_ids'])
        bkg_input_delays = np.array(bkg_input['delays'])

        bkg_input_weights = (bkg_input_weights/voltage_scale[self._node_type_ids[bkg_input_indices[:, 0]]])
        bkg_input_delays = np.round(np.clip(bkg_input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # bkg_input_indices[:, 1] = bkg_input_indices[:, 1] + self._n_neurons * (bkg_input_delays - 1)
        new_bkg_input_receptor_ids = old_receptor_id_to_new_receptor_id[bkg_input_indices[:, 0], bkg_input_receptor_ids]
        bkg_input_indices[:, 0] = bkg_input_indices[:, 0] * self._n_receptors + new_bkg_input_receptor_ids
        self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int64)

        # Define Tensorflow variables
        bkg_input_weight_positive = tf.Variable(
            bkg_input_weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
        self.bkg_input_weights = tf.Variable(
            bkg_input_weights * input_weight_scale / lr_scale, 
            name=self.name+'_rest_of_brain_weights', 
            constraint=SignedConstraint(bkg_input_weight_positive),
            trainable=train_noise,
            dtype=self.variable_dtype
        )

        print(f"    > # BKG input synapses {len(bkg_input_indices)}")
        del bkg_input_indices, bkg_input_weights, bkg_input_receptor_ids, bkg_input_delays

        # # inter-area connectivity
        if self.name == 'v1':
            self.source_column_order = 'lm'        
        elif self.name == 'lm':
            self.source_column_order = 'v1'
        else:
            raise ValueError('Unknown source column')

        interarea_indices = np.array(network['interarea_synapses'][self.source_column_order]['indices'])
        interarea_weights = np.array(network['interarea_synapses'][self.source_column_order]['weights'])
        interarea_receptor_ids = np.array(network['interarea_synapses'][self.source_column_order]['receptor_ids'])
        interarea_delays = np.array(network['interarea_synapses'][self.source_column_order]['delays'])
        interarea_dense_shape = np.array(network['interarea_synapses'][self.source_column_order]['dense_shape'])
                
        if interarea_indices is not None:
            _n_neurons_source_column = interarea_dense_shape[1]
            _n_interarea_synapses = len(interarea_indices)

            interarea_weights = interarea_weights / voltage_scale[self._node_type_ids[interarea_indices[:, 0]]] # indices[:,0] target, [:,1] source
            interarea_delays = np.round(np.clip(interarea_delays, dt, self.max_delay) / dt).astype(np.int32)

            interarea_indices[:, 1] = interarea_indices[:, 1] + _n_neurons_source_column * (interarea_delays - 1) # here, the _n_neurons should be the #neurons in source column 
            new_receptor_ids = old_receptor_id_to_new_receptor_id[interarea_indices[:, 0], interarea_receptor_ids]
            interarea_indices[:, 0] = interarea_indices[:, 0] * self._n_receptors + new_receptor_ids

            interarea_dense_shape = self._n_receptors * interarea_dense_shape[0], self.max_delay * _n_neurons_source_column 
            pre_interarea_ind_table = make_pre_ind_table(interarea_indices, n_source_neurons=interarea_dense_shape[1])
            interarea_indices = tf.Variable(interarea_indices, dtype=tf.int64, trainable=False)
            
            interarea_weight_positive = tf.Variable(
                interarea_weights >= 0., name=self.name + '_interarea_weights_sign_'+self.source_column_order, trainable=False)
            
            interarea_weight_values = tf.Variable(
                interarea_weights * interarea_weight_scale / lr_scale, 
                name=self.name+'_sparse_interarea_weights_'+self.source_column_order,
                constraint=SignedConstraint(interarea_weight_positive),
                dtype=self.variable_dtype,
                trainable=train_interarea)
                        
            # check legal indices
            max_tgt_ind , max_src_ind = interarea_indices.numpy().max(axis=0)            
            assert  max_tgt_ind // self._n_receptors <= interarea_dense_shape[0], 'wrong inter-area indices from target!'
            assert  max_src_ind <= interarea_dense_shape[1], 'wrong inter-area indices from source!'

            print(f'> {self.source_column_order} to {self.name} interarea synapses {_n_interarea_synapses}')

        else:
            interarea_weight_values, interarea_dense_shape, interarea_weight_positive, interarea_indices = None, None, None, None

        self.interarea_weight_values = {self.source_column_order: interarea_weight_values}
        self.interarea_dense_shapes = {self.source_column_order: interarea_dense_shape}
        self.interarea_indices = {self.source_column_order: interarea_indices}
        self.pre_interarea_ind_table = {self.source_column_order: pre_interarea_ind_table}

        print('Init finished!!')
    
    # @tf.function
    def calculate_i_rec(self, rec_z_buf):
        # This implementation works only for batch size 1
        non_zero_cols = tf.where(rec_z_buf > 0)[:, 1]
        new_indices, inds = get_new_inds_table(self.recurrent_indices, non_zero_cols, self.pre_ind_table)
        # Sort the segment IDs and corresponding data
        sorted_indices = tf.argsort(new_indices[:, 0])
        sorted_segment_ids = tf.gather(new_indices[:, 0], sorted_indices)
        sorted_inds = tf.gather(inds, sorted_indices)
        sorted_weights = tf.gather(self.recurrent_weight_values, sorted_inds)
        # Calculate the recurrent currents
        i_rec = tf.math.unsorted_segment_sum(
            sorted_weights,
            sorted_segment_ids,
            num_segments=self.recurrent_dense_shape[0]
        )
        # i_rec = tf.cast(i_rec, dtype=self.compute_dtype)
        if i_rec.dtype != self.compute_dtype:
            i_rec = tf.cast(i_rec, dtype=self.compute_dtype)
        # # Add batch dimension
        # i_rec = tf.expand_dims(i_rec, axis=0)
        return i_rec
    
    # @tf.function
    def calculate_i_inter(self, interarea_z_bufs, column_order):
        # This implementation works only for batch size 1
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        # This is a new faster implementation that uses the pre_ind_table 
        # Memory consumption and processing time depends on the number of spiking neurons
        # this faster method uses sparseness of the rec_z_buf.
        # it identifies the non_zero rows of rec_z_buf and only computes the
        # sparse matrix multiplication for those rows.
        if self.interarea_indices[column_order] is None:
            return tf.zeros((self._n_receptors * self._n_neurons), dtype=self.compute_dtype)
        # find the non-zero rows of rec_z_buf
        non_zero_cols = tf.where(interarea_z_bufs > 0)[:, 1]
        new_indices, inds = get_new_inds_table(self.interarea_indices[column_order], non_zero_cols, self.pre_interarea_ind_table[column_order])        
        # Sort the segment IDs and corresponding data
        sorted_indices = tf.argsort(new_indices[:, 0])
        sorted_segment_ids = tf.gather(new_indices[:, 0], sorted_indices)
        sorted_inds = tf.gather(inds, sorted_indices)
        sorted_weights = tf.gather(self.interarea_weight_values[column_order], sorted_inds)
        # Calculate the contribution of the interarea currents
        i_interarea = tf.math.unsorted_segment_sum(
            sorted_weights,
            sorted_segment_ids,
            num_segments=self.interarea_dense_shapes[column_order][0]
        )
        # Add batch dimension
        # i_interarea = tf.cast(i_interarea, dtype=self.compute_dtype)  # convert float32 to float16 if needed
        if i_interarea.dtype != self.compute_dtype:
            i_interarea = tf.cast(i_interarea, dtype=self.compute_dtype)
        # # Add batch dimension
        # i_interarea = tf.expand_dims(i_interarea, axis=0)
        return i_interarea
            
    def update_psc(self, psc, psc_rise, rec_inputs):
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        return new_psc, new_psc_rise
    
    @property
    def state_size(self):
        # Define the state size of the network
        state_size = (
            self._n_neurons * self.max_delay,                # z buffer
            self._n_neurons,                                 # v
            self._n_neurons,                                 # r
            self._n_neurons * 2,                                 # asc 
            self._n_neurons * self._n_receptors,                   # psc rise
            self._n_neurons * self._n_receptors,                   # psc
        )   
        return state_size

    def zero_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * \
                tf.cast(self.v_th * .0 + 1. * self.v_reset, dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc = tf.zeros((batch_size, self._n_neurons * 2), dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_receptors), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_receptors), dtype)

        return z0_buf, v0, r0, asc, psc_rise0, psc0 

    # @tf.function
    def call(self, inputs, state, constants=None):
        # Get all the model inputs
        external_current, bkg_noise, interarea_z_bufs, state_input = inputs
        batch_size = tf.shape(bkg_noise)[0]
        
        if self._spike_gradient:
            state_input = tf.zeros((1,), dtype=self.compute_dtype)
        else:
            state_input = tf.zeros((4,), dtype=self.compute_dtype)           
        
        # Extract the network variables from the state
        z_buf, v, r, asc, psc_rise, psc = state
        # Get previous spikes
        prev_z = z_buf[:, :self._n_neurons]  # Shape: [batch_size, n_neurons]

        ### Calculate the recurrent input current ###
        if self._connected_recurrent_connections:
            dampened_z_buf = z_buf * self._recurrent_dampening  # dampened version of z_buf # no entiendo muy bien la utilidad de esto
            # Now we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained
            rec_z_buf = (tf.stop_gradient(z_buf - dampened_z_buf) + dampened_z_buf)  
            i_rec = self.calculate_i_rec(rec_z_buf)
        else:
            i_rec = tf.zeros((self._n_neurons * self._n_receptors), dtype=self.compute_dtype)

        # ### Calculate the interarea input current ###
        if self._connected_areas:
            interarea_z_bufs = interarea_z_bufs[0]
            dampened_interarea_z_bufs = interarea_z_bufs * self._recurrent_dampening  # dampened version of z_buf # no entiendo muy bien la utilidad de esto
            # Now we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained
            interarea_z_buf = (tf.stop_gradient(interarea_z_bufs - dampened_interarea_z_bufs) + dampened_interarea_z_bufs)  
            i_interarea = tf.zeros((self._n_receptors * self._n_neurons), dtype=self.compute_dtype)
            for column_order in [self.source_column_order]:
                i_interarea += self.calculate_i_inter(interarea_z_buf, column_order)
        else:
            i_interarea = tf.zeros((self._n_neurons * self._n_receptors), dtype=self.compute_dtype)

        ### Compute the network internal currents
        i_network = tf.expand_dims(i_rec + i_interarea, axis=0)

        ### Background noise
        if self._connected_noise:
            i_noise = bkg_noise
        else:
            i_noise = tf.zeros((batch_size, self._n_neurons, self._n_receptors), dtype=self.compute_dtype)
        
        # Add all the current sources
        rec_inputs = i_network + i_noise
        if external_current is not None and self.name == 'v1': # only V1 area can receive external input
            rec_inputs = rec_inputs + external_current
            
        rec_inputs = rec_inputs * self._lr_scale
        if constants is not None and not self._spike_gradient:
            rec_inputs = rec_inputs + state_input * self._lr_scale #(1, n_neurons, _n_receptors)

        # Calculate the new psc variables
        new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)
        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)  # =max(r + prev_z * self.t_ref - self._dt, 0)
        # Calculate the ASC
        asc = tf.reshape(asc, (batch_size, self._n_neurons, 2))
        new_asc = self.exp_dt_k * asc + tf.expand_dims(prev_z, axis=-1) * self.asc_amps
        new_asc = tf.reshape(new_asc, (batch_size, self._n_neurons * 2))

        input_current = tf.reshape(psc, (batch_size, self._n_neurons, self._n_receptors))
        input_current = tf.reduce_sum(input_current, -1) # sum over receptors
        if constants is not None and self._spike_gradient:
            input_current += state_input

        # Add all the postsynaptic current sources
        c1 = input_current + tf.reduce_sum(asc, axis=-1) + self.gathered_g

        # Calculate the new voltage values
        decayed_v = self.decay * v
        reset_current = prev_z * self.v_gap
        new_v = decayed_v + self.current_factor * c1 + reset_current

        # Update the voltage according to the LIF equation and the refractory period
        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            new_v = tf.where(new_r > 0.0, self.v_reset, new_v)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period

        # Generate the network spikes
        v_sc = (new_v - self.v_th) / self.normalizer
        if self._pseudo_gauss:
            if self.compute_dtype == tf.bfloat16:
                new_z = spike_gauss_b16(v_sc, self._dampening_factor)
            elif self.compute_dtype == tf.float16:
                new_z = spike_gauss_16(v_sc, self._gauss_std, self._dampening_factor)
            else:
                new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)
        else:
            if self.compute_dtype == tf.bfloat16:
                new_z = spike_function_b16(v_sc, self._dampening_factor)
            elif self.compute_dtype == tf.float16:
                new_z = spike_function_16(v_sc, self._dampening_factor)
            else:
                new_z = spike_function(v_sc, self._dampening_factor)

        # Generate the new spikes if the refractory period is concluded
        new_z = tf.where(new_r > 0.0, tf.zeros_like(new_z), new_z)
        # Add current spikes to the buffer
        new_z_buf = tf.concat([new_z, z_buf[:, :-self._n_neurons]], axis=1)  # Shift buffer

        outputs = (
            new_z, 
            new_v * self.voltage_scale + self.voltage_offset,
            # (input_current + new_asc_1 + new_asc_2) * self.voltage_scale
            )
        
        new_state = (new_z_buf, 
                    new_v, 
                    new_r, 
                    new_asc,
                    new_psc_rise, 
                    new_psc)

        return outputs, new_state


class MultiAreaModel(tf.keras.layers.Layer):
    def __init__(self, 
                networks, 
                lgn_inputs, 
                bkg_inputs, 
                gauss_std, 
                dampening_factor, 
                recurrent_dampening_factor,
                input_weight_scale, 
                interarea_weight_scale, 
                lr_scale,
                spike_gradient, 
                batch_size,
                max_delay, 
                pseudo_gauss, 
                hard_reset, 
                connected_recurrent_connections,
                connected_areas,
                connected_noise,
                train_recurrent_v1, 
                train_recurrent_lm, 
                train_input, 
                train_interarea_lm_v1,
                train_interarea_v1_lm,
                train_noise):
        
        super().__init__()

        self._n_areas = len(networks) #2
        self._batch_size  = batch_size #1

        self.v1 = BillehColumn(networks['v1'], lgn_inputs['v1'], bkg_inputs['v1'],
                               gauss_std=gauss_std, dampening_factor=dampening_factor, recurrent_dampening_factor=recurrent_dampening_factor,
                               input_weight_scale=input_weight_scale, interarea_weight_scale=interarea_weight_scale,
                               lr_scale=lr_scale, max_delay=max_delay, batch_size=batch_size,
                               pseudo_gauss=pseudo_gauss, spike_gradient=spike_gradient, train_recurrent=train_recurrent_v1, 
                               train_input=train_input, train_interarea=train_interarea_v1_lm, train_noise=train_noise, 
                               name='v1', hard_reset=hard_reset, connected_areas=connected_areas,
                               connected_recurrent_connections=connected_recurrent_connections, connected_noise=connected_noise)

        self.lm = BillehColumn(networks['lm'], None, bkg_inputs['lm'],
                               gauss_std=gauss_std, dampening_factor=dampening_factor, recurrent_dampening_factor=recurrent_dampening_factor,
                               input_weight_scale=input_weight_scale, interarea_weight_scale=interarea_weight_scale,
                               lr_scale=lr_scale, max_delay=max_delay, batch_size=batch_size,
                               pseudo_gauss=pseudo_gauss, spike_gradient=spike_gradient, train_recurrent=train_recurrent_lm, 
                               train_input=train_input, train_interarea=train_interarea_lm_v1, train_noise=train_noise, 
                               name='lm', hard_reset=hard_reset, connected_areas=connected_areas,
                               connected_recurrent_connections=connected_recurrent_connections, connected_noise=connected_noise)

        self._n_neurons = self.v1._n_neurons + self.lm._n_neurons
        self.num_elements_in_one_state = int(len(self.state_size)/self._n_areas)
    
    @property
    def state_size(self):
        # Define the state size of the network
        state_size = (
            self.v1._n_neurons * self.v1.max_delay,   # z buffer
            self.v1._n_neurons,                        # v
            self.v1._n_neurons,                        # r
            self.v1._n_neurons*2,                        # asc 
            self.v1._n_receptors * self.v1._n_neurons, # psc rise, double exponential synapses
            self.v1._n_receptors * self.v1._n_neurons, # psc
            self.lm._n_neurons * self.lm.max_delay,   # z buffer
            self.lm._n_neurons,                        # v
            self.lm._n_neurons,                        # r
            self.lm._n_neurons*2,                        # asc
            self.lm._n_receptors * self.lm._n_neurons, # psc rise, double exponential synapses
            self.lm._n_receptors * self.lm._n_neurons, # psc
        )
        return state_size
    
    def zero_state_multi_areas(self, batch_size, dtype=tf.float32):
        multi_zero_state =  self.v1.zero_state(batch_size, dtype) + self.lm.zero_state(batch_size, dtype) # add the two states together, one after the other
        return multi_zero_state

    def call(self, inputs, state, constants=None):   
        external_current = inputs[:, :self.v1._n_neurons*self.v1._n_receptors] # first part is real external input
        bkg_input = inputs[:, self.v1._n_neurons*self.v1._n_receptors:-self._n_neurons] # background noise
        state_input = inputs[:, -self._n_neurons:] # dummy zeros
        
        v1_state, lm_state = state[:self.num_elements_in_one_state], state[-self.num_elements_in_one_state:]
        interarea_z_bufs_to_v1 = (lm_state[0],)
        inputs_to_v1 = (external_current, bkg_input[:,:self.v1._n_neurons*self.v1._n_receptors], interarea_z_bufs_to_v1, state_input[:,:self.v1._n_neurons])
        outputs_v1, new_v1_state = self.v1(inputs_to_v1, v1_state)

        interarea_z_bufs_to_lm = (v1_state[0],)
        inputs_to_lm = (None, bkg_input[:,self.v1._n_neurons*self.v1._n_receptors:], interarea_z_bufs_to_lm, state_input[:,self.v1._n_neurons:])
        outputs_lm, new_lm_state = self.lm(inputs_to_lm, lm_state)
        
        outputs = outputs_v1 + outputs_lm
        new_state = new_v1_state + new_lm_state # merge tuple, because nested tuple as state cannot give a legal state_size

        return outputs, new_state


def create_model(networks, 
                lgn_inputs, 
                bkg_inputs, 
                seq_len=200, 
                n_input=10, 
                n_output=2,
                dtype=tf.float32,
                input_weight_scale=1., 
                interarea_weight_scale=1., 
                gauss_std=.5,
                dampening_factor=.2, 
                recurrent_dampening_factor=.5,
                lr_scale=800., 
                train_recurrent_v1=True, 
                train_recurrent_lm=False, 
                train_interarea_lm_v1=True, 
                train_interarea_v1_lm=True, 
                train_input=False, 
                train_noise=True,
                neuron_output=True, 
                use_state_input=True, 
                return_state=True,
                return_sequences=True, 
                down_sample=50,
                cue_duration=20,
                add_rate_metric=True, 
                max_delay=5,
                batch_size=None,
                pseudo_gauss=False,
                hard_reset=False,
                connected_recurrent_connections=True,
                connected_areas=True,
                connected_noise=True,
                output_completed_valid_from_time=120, 
                output_abstract_valid_from_time=100,
                ):

    # Create the input of the model
    # x = tf.keras.layers.Input(shape=(seq_len, n_input,)) # this shape (None, seq_len, n_input) dose not contain batch size, batch_size can be assigned as another argument
    x = tf.keras.layers.Input(shape=(None, n_input,))
    neurons =  networks['v1']['n_nodes'] + networks['lm']['n_nodes']
    
     # Create an input layer for the initial state of the RNN
    # state_input_holder = tf.keras.layers.Input(shape=(seq_len, neurons))
    state_input_holder = tf.keras.layers.Input(shape=(None, neurons))
    state_input = tf.cast(tf.identity(state_input_holder), dtype) # (None, 2500, 1000+150)
    # The state_input is initialized with zeros and its purpose is to provide a way to pass additional input to the model if needed

    # if use_state_input # for rollout the network, may be needed later
    if batch_size is None:
        batch_size = tf.shape(x)[0] # if batch_size is None just update after each timestep (1 ms in our case)
    else:
        batch_size = batch_size
                
    # Create the model columns
    cell = MultiAreaModel(
                            networks, 
                            lgn_inputs, 
                            bkg_inputs, 
                            gauss_std=gauss_std, 
                            dampening_factor=dampening_factor,
                            recurrent_dampening_factor=recurrent_dampening_factor,
                            input_weight_scale=input_weight_scale, 
                            interarea_weight_scale=interarea_weight_scale,
                            lr_scale=lr_scale, 
                            spike_gradient=True, 
                            batch_size=batch_size, 
                            max_delay=max_delay,
                            pseudo_gauss=pseudo_gauss, 
                            hard_reset=hard_reset,
                            connected_recurrent_connections=connected_recurrent_connections,
                            connected_areas=connected_areas,
                            connected_noise=connected_noise,
                            train_recurrent_v1=train_recurrent_v1, 
                            train_recurrent_lm=train_recurrent_lm, 
                            train_input=train_input, 
                            train_interarea_lm_v1=train_interarea_lm_v1, 
                            train_interarea_v1_lm=train_interarea_v1_lm, 
                            train_noise=train_noise)
        
    zero_state = cell.zero_state_multi_areas(batch_size, dtype)

    if use_state_input:
        initial_state_holder = tf.nest.map_structure(lambda _x: tf.keras.layers.Input(shape=_x.shape[1:], dtype=_x.dtype), zero_state) # get the shape of zero_state without the batch_size # (None, 5000)
        rnn_initial_state = tf.nest.map_structure(tf.identity, initial_state_holder) # copy of initial_state_holder (None, 5000)
        constants = tf.zeros_like(rnn_initial_state[0][:, 0], dtype) # (None,)
    else:
        rnn_initial_state = zero_state
        constants = tf.zeros((batch_size,))
    
    # Calculate the LGN input
    rnn_inputs = LGNInputLayer(indices=cell.v1.input_indices, 
                            weights=cell.v1.input_weight_values, 
                            dense_shape=cell.v1.lgn_input_dense_shape, 
                            name='input_layer'
                            )(x)

    # Calculate the background noise input
    bkg_inputs = BackgroundNoiseLayer(cell, 
                                      batch_size=batch_size,
                                      seq_len=seq_len,
                                      bkg_firing_rate=250, 
                                      n_bkg_units=100,
                                      n_bkg_connections=4
                                    )(x) # (None, 2500, 4000+600)

    # Concatenate all the inputs together    
    full_inputs = tf.concat((rnn_inputs, bkg_inputs, state_input), -1) # (None, 2500, 120(LGN)+200(BKG)+50(dummy_zeros))

    # Create the RNN layer
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state, name='rsnn') # return_sequences=True return the full sequence. 
    
    # Run the model over the inputs
    out = rnn(full_inputs, initial_state=rnn_initial_state, constants=constants)
    
    # Extract the model output and its new state
    if return_state:
        hidden = out[0] #4
        # new_state = out[1:] #(14)
    else:
        hidden = out

    # assume all areas have similar firing rates
    spikes_v1, voltage_v1, spikes_lm, voltage_lm = hidden  ## these are results in all time steps? Yes, if return_sequences=True
    rate_v1 = tf.reduce_mean(spikes_v1, (1, 2))
    rate_lm = tf.reduce_mean(spikes_lm, (1, 2))
    
    # if neuron_output:
    #     output_spikes = 1 / dampening_factor * spikes_v1 + \
    #         (1 - 1 / dampening_factor) * tf.stop_gradient(spikes_v1)
    #     output = tf.gather(output_spikes, networks['v1']['readout_neuron_ids'], axis=2)
    #     output = tf.reduce_mean(output, -1)
    #     scale = (1 + tf.nn.softplus(tf.keras.layers.Dense(1)(tf.zeros_like(output[:1, :1]))))
    #     thresh = tf.keras.layers.Dense(1)(tf.zeros_like(output))
    #     output = tf.stack([thresh[..., 0], output[..., -1]], -1) * scale
    # else:
    #     output = tf.keras.layers.Dense(n_output, name='projection', trainable=True)(spikes_v1)
    
    # output = tf.keras.layers.Dense(
    #             n_output, name='projection', trainable=True)(spikes_v1)

    # # Finally, a prediction layer is created
    # output = tf.keras.layers.Lambda(lambda _a: _a, name='prediction')(output)
        
    abstract_output = tf.reduce_mean(tf.gather(spikes_lm[
                    :,#batch dim
                    output_abstract_valid_from_time:,# time dim
                    :#neuron dim
                    ], networks['lm']['laminar_indices']['L5_e'], axis=2), axis=1)
    abstract_output = tf.keras.layers.Lambda(lambda _a: _a, name='abstract_output')(abstract_output)

    completed_output = tf.reduce_mean(tf.gather(spikes_v1[
                    :,#batch dim
                    output_completed_valid_from_time:,# time dim
                    :#neuron dim
                    ], networks['v1']['laminar_indices']['L5_e'], axis=2), axis=1)
    completed_output = tf.keras.layers.Lambda(lambda _a: _a, name='completed_output')(completed_output)

    # If return_sequences is True, then the mean_output tensor is computed by
    # averaging over sequences of length down_sample in the output tensor.
    # Otherwise, mean_output is simply the mean of the last cue_duration time steps
    # of the output tensor.

    # if return_sequences:
    #     mean_output = tf.reshape(output, (-1, int(seq_len / down_sample), down_sample, n_output))
    #     mean_output = tf.reduce_mean(mean_output, 2)
    #     mean_output = tf.nn.softmax(mean_output, axis=-1)
    # else:
    #     mean_output = tf.reduce_mean(output[:, -cue_duration:], 1)
    #     mean_output = tf.nn.softmax(mean_output)

    if use_state_input:
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder, initial_state_holder], outputs=[abstract_output, completed_output, bkg_inputs])
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder, initial_state_holder], 
        #     outputs=mean_output
        # )
    else:
        # many_input_model = tf.keras.Model(
        #     inputs=[x, state_input_holder], 
        #     outputs=mean_output
        # )
        many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder], outputs=[abstract_output, completed_output])
    
    # many_input_model = tf.keras.Model(inputs=[x, state_input_holder], outputs=[abstract_output, completed_output])

    if add_rate_metric:
        many_input_model.add_metric(rate_v1, name='v1_area_rate')
        many_input_model.add_metric(rate_lm, name='lm_area_rate')

    return many_input_model

import numpy as np
import tensorflow as tf
import psutil
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


# class BackgroundNoiseLayer(tf.keras.layers.Layer):
#     """
#     This class calculates the input currents from the BKG noise by processing all timesteps at once."
#     For that reason is unfeasible if the user wants to train the LGN -> V1 weights.
#     Each call takes 0.03 seconds for 600 ms of simulation.

#     Returns:
#         _type_: input_currents (self._compute_dtype)
#     """
#     def __init__(self, cell, batch_size, seq_len,
#                  bkg_units_firing_rate=1000, n_bkg_units=1, 
#                  dtype=tf.float32, **kwargs):
#         super().__init__(**kwargs)
#         self._dtype = dtype
#         self._batch_size = batch_size
#         self._seq_len = seq_len
#         self._bkg_units_firing_rate = bkg_units_firing_rate
#         self._n_bkg_units = n_bkg_units

#         self._bkg_weights = {'v1': None, 'lm': None}
#         self._bkg_weights['v1'] = cell.v1.bkg_input_weights #(4000,)
#         self._bkg_weights['lm'] = cell.lm.bkg_input_weights #* 1.2 #(600,)

#         self._bkg_indices = {'v1': None, 'lm': None}
#         self._bkg_indices['v1'] = cell.v1.bkg_input_indices
#         self._bkg_indices['lm'] = cell.lm.bkg_input_indices

#         self._dense_shape = {'v1': None, 'lm': None}
#         self._dense_shape['v1'] = cell.v1.bkg_input_dense_shape
#         self._dense_shape['lm'] = cell.lm.bkg_input_dense_shape

#     def calculate_bkg_i_in(self, inputs, column='v1'):
#         # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
#         # i_in = tf.TensorArray(dtype=self._compute_dtype, size=self._n_max_receptors)
#         sparse_w_in = tf.sparse.SparseTensor(
#                 self._bkg_indices[column],
#                 self._bkg_weights[column], 
#                 self._dense_shape[column],
#             )
        
#         i_in = tf.sparse.sparse_dense_matmul(
#                                             sparse_w_in,
#                                             inputs,
#                                             adjoint_b=True
#                                             )
#         return i_in

#     def call(self, inp): # inp only provides the shape
#         seq_len = tf.shape(inp)[1]

#         # Generate the background spikes
#         # rest_of_brain = tf.random.poisson(shape=(self._batch_size, self._seq_len, self._n_bkg_units), 
#         #                                 lam=self._bkg_units_firing_rate/1000, 
#         #                                 dtype=self._compute_dtype) # (1, 3000, 1)
#         rest_of_brain = tf.random.poisson(shape=(self._batch_size, seq_len, self._n_bkg_units), 
#                                         lam=self._bkg_units_firing_rate/1000, 
#                                         dtype=self._compute_dtype) # (1, 3000, 1
#         # rest_of_brain = tf.reshape(rest_of_brain, (self._batch_size * self._seq_len, self._n_bkg_units)) # (3000, 1) # (batch_size*sequence_length, input_dim)
#         rest_of_brain = tf.reshape(rest_of_brain, (self._batch_size * seq_len, self._n_bkg_units))

#         noise_inputs = {'v1': None, 'lm': None}
#         for column in noise_inputs.keys():
#             noise_input = self.calculate_bkg_i_in(rest_of_brain, column=column) # (200000, 3000)
#             noise_input = tf.transpose(noise_input) # (3000, 200000) # New shape (3000, 66634, 5)
#             # noise_inputs[column] = tf.reshape(noise_input, (self._batch_size, self._seq_len, -1)) # (1, 3000, 200000) # (1, 3000, 333170)
#             noise_inputs[column] = tf.reshape(noise_input, (self._batch_size, seq_len, -1))
#         cat_noise_inputs = tf.concat([noise_inputs['v1'], noise_inputs['lm']], axis=-1)

#         return cat_noise_inputs



class BackgroundNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, cell, batch_size, seq_len,
                 bkg_units_firing_rate=1000, n_bkg_units=1, 
                 n_bkg_connections = 1, 
                 dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._bkg_units_firing_rate = bkg_units_firing_rate
        self._n_bkg_units = n_bkg_units
        self._n_bkg_connections = n_bkg_connections

        # Initialize weights and indices
        self._bkg_weights = {'v1': None, 'lm': None}
        self._bkg_indices = {'v1': None, 'lm': None}
        self._dense_shape = {'v1': None, 'lm': None}
        self._initialize_background_inputs(cell)

    def _initialize_background_inputs(self, cell):
        # Create connectivity and assign weights for 'v1' and 'lm' areas
        for column in ['v1', 'lm']:
            num_neurons = cell.__getattribute__(column).bkg_input_weights.shape[0]
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

            # tf.print(column)
            # tf.print(cell.__getattribute__(column).bkg_input_weights)
            # cell.__getattribute__(column).bkg_input_weights = tf.Variable(weights, trainable=original_weights.trainable)
            # tf.print(weights)
            # tf.print(cell.__getattribute__(column).bkg_input_weights.shape)
            # self._bkg_weights[column] = cell.__getattribute__(column).bkg_input_weights #tf.Variable(weights, trainable=original_weights.trainable)
            
             # Create a new constraint based on the new weights
            new_bkg_input_weight_positive = tf.Variable(weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
            new_constraint = SignedConstraint(new_bkg_input_weight_positive)


            cell.__getattribute__(column).bkg_input_weights = tf.Variable(weights, 
                                                                        name=column+'_rest_of_brain_weights', 
                                                                        constraint=new_constraint,
                                                                        dtype=original_weights.dtype,
                                                                        trainable=original_weights.trainable)

            # cell.__getattribute__(column).bkg_input_weights.constraint = new_constraint

            # self._bkg_weights[column] = tf.Variable(weights, 
            #                                         name=column+'_rest_of_brain_weights', 
            #                                         constraint=SignedConstraint(bkg_input_weight_positive),
            #                                         dtype=original_weights.dtype,
            #                                         trainable=original_weights.trainable)

            self._bkg_weights[column] = cell.__getattribute__(column).bkg_input_weights
            self._bkg_indices[column] = tf.Variable(indices, trainable=False, dtype=tf.int64)
            self._dense_shape[column] = (original_dense_shape[0],  self._n_bkg_units)

    def calculate_bkg_i_in(self, inputs, column='v1'):
        sparse_w_in = tf.sparse.SparseTensor(
            self._bkg_indices[column],
            self._bkg_weights[column], 
            self._dense_shape[column]
        )
        i_in = tf.sparse.sparse_dense_matmul(sparse_w_in, inputs, adjoint_b=True)
        return i_in

    def call(self, inp):
        seq_len = tf.shape(inp)[1]
        rest_of_brain = tf.random.poisson(shape=(self._batch_size, seq_len, self._n_bkg_units), 
                                          lam=self._bkg_units_firing_rate/1000, 
                                          dtype=self._dtype)
        rest_of_brain = tf.reshape(rest_of_brain, (self._batch_size * seq_len, self._n_bkg_units))

        noise_inputs = {'v1': None, 'lm': None}
        for column in noise_inputs.keys():
            noise_input = self.calculate_bkg_i_in(rest_of_brain, column=column)
            noise_input = tf.transpose(noise_input)
            noise_inputs[column] = tf.reshape(noise_input, (self._batch_size, seq_len, -1))
        
        cat_noise_inputs = tf.concat([noise_inputs['v1'], noise_inputs['lm']], axis=-1)
        return cat_noise_inputs



    

class SparseLayer(tf.keras.layers.Layer):
    def __init__(self, indices, weights, dense_shape, dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self._indices = indices
        self._input_weights = weights
        self._dense_shape = dense_shape
        self._dtype = dtype
        # Define a threshold that determines whether to compute the sparse
        # matrix multiplication directly or split it into smaller batches in a GPU.
        # The value is calculated to ensure that output.shape[1] * nnz(a) > 2^31, 
        # where output.shape[1] is the time_length, and nnz(a) is the number of non-zero elements in the sparse matrix.
        nnz_sparse_matrix = self._indices.shape[0]
        self._max_batch = int(2**31 / nnz_sparse_matrix)

    def calculate_i_in(self, inputs):
        sparse_w_in = tf.sparse.SparseTensor(
                self._indices,
                self._input_weights, 
                self._dense_shape,
            )
        i_in = tf.sparse.sparse_dense_matmul(
                                                sparse_w_in,
                                                inputs,
                                                adjoint_b=True
                                            )
        return i_in

    # @profile
    def call(self, inp, verbose=False):
        # replace any None values in the shape of inp with the actual values obtained from the input tensor at runtime (tf.shape(inp)).
        # This is necessary because the SparseTensor multiplication operation requires a fully defined shape.
        inp_shape = inp.get_shape().as_list() # [None, 600, 17400]
        shp = [dim if dim is not None else tf.shape(inp)[i] for i, dim in enumerate(inp_shape)]
        batch_size = shp[0] * shp[1]
        if verbose:
            tf.print(f"The ratio of the current input batch size to the maximum batch size is {batch_size}/{self._max_batch}")
        
        inp = tf.cast(inp, self._compute_dtype)
        inp = tf.reshape(inp, (batch_size, shp[2])) # (batch_size*sequence_length, input_dim)
        if shp[0] * shp[1] < self._max_batch:
            # the sparse tensor multiplication can be directly performed
            if verbose:
                tf.print('Processing input tensor directly.')
            input_current = self.calculate_i_in(inp)  #(461848, 3000) # (5, 1000, 600)
            input_current = tf.transpose(input_current) #(3000, 461848) # (600, 1000, 5)
        else: 
            # Define the current batch size and calculate the number of chunks
            batch_size = tf.shape(inp)[0]
            num_chunks = int(batch_size / self._max_batch)
            num_pad_elements = 0
            if batch_size % self._max_batch != 0:
                num_chunks += 1 # add 1 chunk if the quotient is not an integer
                # Padd the input with 0's to ensure all chunks have the same size for the matrix multiplication
                num_pad_elements += num_chunks * self._max_batch - batch_size
                inp = tf.pad(inp, [(0, num_pad_elements), (0, 0)])
            if verbose:
                tf.print(f'Chunking input tensor into {num_chunks} batches.')
            
            # Initialize a tensor array to hold the partial results of every chunk
            result_array = tf.TensorArray(dtype=self._compute_dtype, size=num_chunks)
            # Iterate over the chunks
            for i in range(num_chunks):
                start_idx = int(i * self._max_batch)
                end_idx = int((i + 1) * self._max_batch)
                chunk = inp[start_idx:end_idx, :]
                chunk = tf.reshape(chunk, (self._max_batch, -1)) # new_shape (612, 17400)
                partial_input_current = self.calculate_i_in(chunk)  #(461848, 612) # ( 5, 66634, 68) 
                # Store the partial result in the tensor array     
                result_array = result_array.write(i, partial_input_current)
            
            # Concatenate the partial results to get the final result
            input_current = result_array.stack() # ( 9, 5, 66634, 68)
            input_current = tf.transpose(input_current, perm=[1, 0, 2])
            input_current = tf.reshape(input_current, (-1, num_chunks * self._max_batch)) # (461848, 3060) # (5, 66634, 612)
            input_current = tf.transpose(input_current) # (3060, 461848)
            
            if num_pad_elements > 0: # Remove the padded 0's
                input_current = input_current[:-num_pad_elements, :] # (3000, 461848) # New shape (600, 66634, 5)

        # Reshape properly the input current
        input_current = tf.reshape(input_current, (shp[0], shp[1], -1)) # New shape (1, 3000, 333170)

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
   

def process_receptors(postsynaptic_indices, receptor_ids):
    # Create a dictionary for every neuron that associates its receptor types to a new set of receptor ids
    neuron_dict = {}
    for neuron_id, receptor_id in zip(postsynaptic_indices, receptor_ids):
        if neuron_id in neuron_dict:
            neuron_dict[neuron_id].add(receptor_id)
        else:
            neuron_dict[neuron_id] = {receptor_id}

    # find the maximum number of receptors for any neuron
    max_receptors = max(len(receptors) for receptors in neuron_dict.values())
    neuron_mappings = {neuron_id: {} for neuron_id in neuron_dict.keys()}
    original_receptor_ids = []

    for neuron_id, receptors in neuron_dict.items():
        sorted_receptors = sorted(receptors)
        for i, rec_id in enumerate(sorted_receptors):
            neuron_mappings[neuron_id][rec_id] = i
            original_receptor_ids.append(rec_id)
        # append 0 until it reaches the max_receptors
        if len(sorted_receptors) < max_receptors:
            for i in range(max_receptors - len(sorted_receptors)):
                original_receptor_ids.append(0)

    original_receptor_ids = np.array(original_receptor_ids, dtype=np.int32)
    
    return max_receptors, neuron_mappings, original_receptor_ids


class BillehColumn(tf.keras.layers.Layer):
    # @profile
    def __init__(
        self, 
        network, 
        lgn_input, 
        bkg_input,
        dt=1., 
        gauss_std=.5, 
        dampening_factor=.3,
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
        **kwargs
        ):

        super().__init__(**kwargs)

        print(f'###### COLUMN {self.name} ######')

        self._params = network['node_params']
        # Rescale the voltages to have them near 0, as we wanted the effective step size 
        # for the weights to be normalized when learning (weights are scaled similarly)
        voltage_scale = self._params['V_th'] - self._params['E_L']
        voltage_offset = self._params['E_L']
        self._params['V_th'] = (self._params['V_th'] - voltage_offset) / voltage_scale
        self._params['E_L'] = (self._params['E_L'] - voltage_offset) / voltage_scale
        self._params['V_reset'] = (self._params['V_reset'] - voltage_offset) / voltage_scale
        self._params['asc_amps'] = self._params['asc_amps'] / voltage_scale[..., None]   # _params['asc_amps'] has shape (111, 2)
        # Define the other model variables
        self._node_type_ids = network['node_type_ids']
        self._dt = dt
        self._dampening_factor = tf.cast(dampening_factor, self._compute_dtype)
        self._pseudo_gauss = pseudo_gauss
        self._lr_scale = lr_scale
        self._spike_gradient = spike_gradient
        self._hard_reset = hard_reset
        self._connected_areas = connected_areas
        self._n_neurons = network['n_nodes']
        self._gauss_std = tf.cast(gauss_std, self._compute_dtype)
        # determine the membrane time decay constant
        tau = self._params['C_m'] / self._params['g'] 
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / self._params['C_m'] * (1 - self._decay) * tau
        # Determine the synaptic dynamic parameters for each of the 5 basis receptors
        tau_syns = np.array([5.5, 8.5, 2.8, 5.8])
        syn_decay = np.exp(-dt / tau_syns)
        syn_decay = tf.constant(syn_decay, dtype=self._compute_dtype)
        psc_initial = np.e / tau_syns
        psc_initial = tf.constant(psc_initial, dtype=self._compute_dtype)

        # self._n_receptors = 4 # network['node_params']['tau_syn'].shape[1] # we have 4 receptor compartments (soma, dendrites, etc) for each neuron
        # create a new variable with all the postsynatpic indices concatenated: network["synapses"]["indices"], lgn_input["indices"],...
        all_interarea_postsynaptic_indices = np.concatenate([network['interarea_synapses'][order]['indices'][:, 0] for order in network['interarea_synapses'].keys()])
        all_interarea_receptor_ids = np.concatenate([network['interarea_synapses'][order]['receptor_ids'] for order in network['interarea_synapses'].keys()])
        if lgn_input is not None:
            all_postsynaptic_indices = np.concatenate([network["synapses"]["indices"][:, 0], lgn_input["indices"][:, 0], bkg_input["indices"][:, 0], all_interarea_postsynaptic_indices], axis=0)
            all_receptor_ids = np.concatenate([network["synapses"]["receptor_ids"], lgn_input["receptor_ids"], bkg_input["receptor_ids"], all_interarea_receptor_ids], axis=0)
        else:
            all_postsynaptic_indices = np.concatenate([network["synapses"]["indices"][:, 0], bkg_input["indices"][:, 0], all_interarea_postsynaptic_indices], axis=0)
            all_receptor_ids = np.concatenate([network["synapses"]["receptor_ids"], bkg_input["receptor_ids"], all_interarea_receptor_ids], axis=0)
        
        self._n_max_receptors, neuron_mappings, original_receptor_ids = process_receptors(all_postsynaptic_indices, all_receptor_ids)
        # create a repetion of the range(0, _n_max_receptors) for every neuron
        self.syn_decay = tf.gather(syn_decay, original_receptor_ids, axis=0)
        self.psc_initial = tf.gather(psc_initial, original_receptor_ids, axis=0)

        self.syn_decay = tf.reshape(self.syn_decay, (self._n_neurons, self._n_max_receptors))
        self.psc_initial = tf.reshape(self.psc_initial, (self._n_neurons, self._n_max_receptors))

        # this are the axonal delays
        self.max_delay = int(np.round(np.min([np.max(network['synapses']['delays']), max_delay])))
        # self.batch_size = batch_size

        # Define the state size of the network
        self.state_size = (
            self._n_neurons * self.max_delay,                # z buffer
            self._n_neurons,                                 # v
            self._n_neurons,                                 # r
            self._n_neurons,                                 # asc 1
            self._n_neurons,                                 # asc 2
            self._n_neurons * self._n_max_receptors,                   # psc rise
            self._n_neurons * self._n_max_receptors,                   # psc
        )   
        

        def _f(_v, trainable=False):
            return tf.Variable(tf.cast(self._gather(_v), self._compute_dtype), trainable=trainable)

        def inv_sigmoid(_x):
            return tf.math.log(_x / (1 - _x))

        def custom_val(_v, trainable=False):
            _v = tf.Variable(tf.cast(inv_sigmoid(self._gather(
                _v)), self._compute_dtype), trainable=trainable)

            def _g():
                return tf.nn.sigmoid(_v.read_value())

            return _v, _g

        # put neuron type features to every neuron
        self.t_ref = _f(self._params['t_ref'])
        self.v_reset = _f(self._params['V_reset'])
        self.asc_amps = _f(self._params['asc_amps'], trainable=False)
        _k = self._params['k']
        # inverse sigmoid of the adaptation rate constant (1/ms)
        self.param_k, self.param_k_read = custom_val(_k, trainable=False) # ?? what is this doing?
        self.k = self.param_k_read()
        self.exp_dt_k_1 = tf.exp(-self._dt * self.k[:, 0])
        self.exp_dt_k_2 = tf.exp(-self._dt * self.k[:, 1])
        self.v_th = _f(self._params["V_th"])
        self.v_gap = self.v_reset - self.v_th
        self.e_l = _f(self._params["E_L"])
        self.normalizer = self.v_th - self.e_l
        self.param_g = _f(self._params["g"])
        self.gathered_g = self.param_g * self.e_l
        self.decay = _f(self._decay)
        self.current_factor = _f(self._current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)

        # self.syn_decay = _f(self._syn_decay)
        # self.psc_initial = _f(self._psc_initial)

        ### Network recurrent connectivity ###
        indices, weights, dense_shape, receptor_ids, delays = (
            network["synapses"]["indices"],
            network["synapses"]["weights"],
            network["synapses"]["dense_shape"],
            network["synapses"]["receptor_ids"],
            network["synapses"]["delays"]
        )
        # Scale down the recurrent weights
        weights = (weights/voltage_scale[self._node_type_ids[indices[:, 0]]])     
        # weights = weights / \
        #     voltage_scale[self._node_type_ids[indices[:, 0] // #indices[:,0] target, [:,1] source
        #                                      self._n_receptors]]  # scale down the weights
        
        # Use the maximum delay to clip the synaptic delays
        delays = np.round(np.clip(delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the presynaptic neuron indices
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1)
        #the first column (presynaptic neuron) has size n_neurons and the second column (postsynaptic neuron) has size max_delay*n_neurons
        # dense_shape = dense_shape[0], self.max_delay * dense_shape[1]  # (target*receptors, source*delays)
        
        new_receptor_ids = np.array([neuron_mappings[i][r] for i, r in zip(indices[:, 0], receptor_ids)], dtype=np.int32)
        indices[:, 0] = indices[:, 0] * self._n_max_receptors + new_receptor_ids

        self.recurrent_dense_shape = self._n_max_receptors * dense_shape[0], self.max_delay * dense_shape[1] 
        # Define the Tensorflow variables
        self.recurrent_indices = tf.Variable(indices, dtype=tf.int64, trainable=False)
        self.pre_ind_table = self.make_pre_ind_table(indices, n_source_neurons=dense_shape[1])

        # Set the sign of the connections (exc or inh)
        self.recurrent_weight_positive = tf.Variable(
            weights >= 0., name='recurrent_weights_sign', trainable=False)

        # Scale the weights
        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale,
            name=self.name+'_sparse_recurrent_weights',
            constraint=SignedConstraint(self.recurrent_weight_positive),
            trainable=train_recurrent,
            dtype=self._compute_dtype
        ) # shape = (n_synapses,)

        print(f"    > # Recurrent synapses: {len(indices)}")
        del indices, weights, dense_shape, receptor_ids, delays

        ### LGN input connectivity ###
        if lgn_input is not None:
            self.lgn_input_dense_shape = (self._n_neurons * self._n_max_receptors, lgn_input["n_inputs"],)
            input_indices, input_weights, input_receptor_ids, input_delays = (
                lgn_input["indices"],
                lgn_input["weights"],
                lgn_input["receptor_ids"],
                lgn_input["delays"]
            )

            # Scale down the input weights
            input_weights = (input_weights/ voltage_scale[self._node_type_ids[input_indices[:, 0]]])
            input_delays = np.round(np.clip(input_delays, dt, self.max_delay)/dt).astype(np.int32)
            # Introduce the delays in the postsynaptic neuron indices
            # input_indices[:, 1] = input_indices[:, 1] + self._n_neurons * (input_delays - 1)

            #the first column (presynaptic neuron) has size n_neurons and the second column (postsynaptic neuron) has size max_delay*n_neurons
            # dense_shape = dense_shape[0], self.max_delay * dense_shape[1]  # (target*receptors, source*delays)
            new_input_receptor_ids = np.array([neuron_mappings[i][r] for i, r in zip(input_indices[:, 0], input_receptor_ids)], dtype=np.int32)
            input_indices[:, 0] = input_indices[:, 0] * self._n_max_receptors + new_input_receptor_ids
            self.input_indices = tf.Variable(input_indices, trainable=False, dtype=tf.int64)

            self.input_weight_positive = tf.Variable(
                input_weights >= 0., name=self.name+'_input_weights_sign', trainable=False)
            self.input_weight_values = tf.Variable(
                input_weights * input_weight_scale / lr_scale, name=self.name+'_sparse_input_weights',
                constraint=SignedConstraint(self.input_weight_positive),
                trainable=train_input,
                dtype=self._compute_dtype)

            print(f"    > # LGN input synapses {len(input_indices)}")
            del input_indices, input_weights, input_receptor_ids, input_delays

        else:
            self.input_indices, self.input_weight_values, self.input_dense_shape = None, None, None
            print(f'    > # LGN to {self.name} input synapses 0')
        
        ### BKG input connectivity ###
        self.bkg_input_dense_shape = (self._n_neurons * self._n_max_receptors, bkg_input["n_inputs"],)
        bkg_input_indices, bkg_input_weights, bkg_input_receptor_ids, bkg_input_delays = (
            bkg_input["indices"],
            bkg_input["weights"],
            bkg_input["receptor_ids"],
            bkg_input["delays"]
        )
        # print('Lokk:')
        # print(bkg_input_weights)
        # print(voltage_scale)
        # print(self._node_type_ids)
        # print(bkg_input_indices[:, 0])
        bkg_input_weights = (bkg_input_weights/voltage_scale[self._node_type_ids[bkg_input_indices[:, 0]]])
        # print(bkg_input_weights)
        bkg_input_delays = np.round(np.clip(bkg_input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # Introduce the delays in the postsynaptic neuron indices
        # bkg_input_indices[:, 1] = bkg_input_indices[:, 1] + self._n_neurons * (bkg_input_delays - 1)
        new_bkg_input_receptor_ids = np.array([neuron_mappings[i][r] for i, r in zip(bkg_input_indices[:, 0], bkg_input_receptor_ids)], dtype=np.int32)
        bkg_input_indices[:, 0] = bkg_input_indices[:, 0] * self._n_max_receptors + new_bkg_input_receptor_ids
        self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int64)

        # Define Tensorflow variables
        self.bkg_input_weight_positive = tf.Variable(
            bkg_input_weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
        self.bkg_input_weights = tf.Variable(
            bkg_input_weights * input_weight_scale / lr_scale, 
            name=self.name+'_rest_of_brain_weights', 
            constraint=SignedConstraint(self.bkg_input_weight_positive),
            trainable=train_noise,
            dtype=self._compute_dtype
        )
        # print(self.bkg_input_weights)

        print(f"    > # BKG input synapses {len(bkg_input_indices)}")
        del bkg_input_indices, bkg_input_weights, bkg_input_receptor_ids, bkg_input_delays

        # # background noise connection
        # bkg_weights = bkg_weights / \
        #     np.repeat(voltage_scale[self._node_type_ids], self._n_receptors)
        # self.bkg_weights = tf.Variable(
        #     bkg_weights * 10., name=self.name+'_rest_of_brain_weights', trainable=train_input)

        # # inter-area connectivity
        if self.name == 'v1':
            self.source_column_order = 'lm'        
        elif self.name == 'lm':
            self.source_column_order = 'v1'
        else:
            raise ValueError('Unknown source column')

        interarea_indices, interarea_weights, interarea_dense_shape, interarea_receptor_ids, interarea_delays = \
            network['interarea_synapses'][self.source_column_order]['indices'], network['interarea_synapses'][self.source_column_order]['weights'], \
                network['interarea_synapses'][self.source_column_order]['dense_shape'], network['interarea_synapses'][self.source_column_order]['receptor_ids'], \
                    network['interarea_synapses'][self.source_column_order]['delays']
                
        if interarea_indices is not None:
            _n_neurons_source_column = interarea_dense_shape[1]
            _n_interarea_synapses = len(interarea_indices)

            interarea_weights = interarea_weights / voltage_scale[self._node_type_ids[interarea_indices[:, 0]]] # indices[:,0] target, [:,1] source
            interarea_delays = np.round(np.clip(interarea_delays, dt, self.max_delay) / dt).astype(np.int32)

            interarea_indices[:, 1] = interarea_indices[:, 1] + _n_neurons_source_column * (interarea_delays - 1) # here, the _n_neurons should be the #neurons in source column 
            new_receptor_ids = np.array([neuron_mappings[i][r] for i, r in zip(interarea_indices[:, 0], interarea_receptor_ids)], dtype=np.int32)
            interarea_indices[:, 0] = interarea_indices[:, 0] * self._n_max_receptors + new_receptor_ids

            interarea_dense_shape = self._n_max_receptors * interarea_dense_shape[0], self.max_delay * _n_neurons_source_column 
            pre_interarea_ind_table = self.make_pre_ind_table(interarea_indices, n_source_neurons=_n_neurons_source_column)
            interarea_indices = tf.Variable(interarea_indices, dtype=tf.int64, trainable=False)
            
            interarea_weight_positive = tf.Variable(
                interarea_weights >= 0., name=self.name + '_interarea_weights_sign_'+self.source_column_order, trainable=False)
            
            interarea_weight_values = tf.Variable(
                interarea_weights * interarea_weight_scale / lr_scale, 
                name=self.name+'_sparse_interarea_weights_'+self.source_column_order,
                constraint=SignedConstraint(interarea_weight_positive),
                dtype=self._compute_dtype,
                trainable=train_interarea)
                        
            # check legal indices
            max_tgt_ind , max_src_ind = interarea_indices.numpy().max(axis=0)            
            assert  max_tgt_ind // self._n_max_receptors <= interarea_dense_shape[0], 'wrong inter-area indices from target!'
            assert  max_src_ind <= interarea_dense_shape[1], 'wrong inter-area indices from source!'

            print(f'> {self.source_column_order} to {self.name} interarea synapses {_n_interarea_synapses}')

        else:
            interarea_weight_values, interarea_dense_shape, interarea_weight_positive, interarea_indices = None, None, None, None

        
        # interarea_weight_values_lower, interarea_dense_shape_lower, interarea_indices_lower = set_interarea_connections('lower')
        # interarea_weight_values_higher, interarea_dense_shape_higher, interarea_indices_higher = set_interarea_connections('higher')
        # self.interarea_weight_values = (interarea_weight_values_lower,interarea_weight_values_higher)
        # self.interarea_dense_shapes = (interarea_dense_shape_lower,interarea_dense_shape_higher)
        # self.interarea_indices = (interarea_indices_lower,interarea_indices_higher)
        
        self.interarea_weight_positive = {self.source_column_order: interarea_weight_positive}
        self.interarea_weight_values = {self.source_column_order: interarea_weight_values}
        self.interarea_dense_shapes = {self.source_column_order: interarea_dense_shape}
        self.interarea_indices = {self.source_column_order: interarea_indices}
        self.pre_interarea_ind_table = {self.source_column_order: pre_interarea_ind_table}

        print('Init finished!!')

    def make_pre_ind_table(self, indices, n_source_neurons=5000):
        """ This function creates a table that maps the presynaptyc index to 
        the indices of the recurrent_indices tensor. it takes a dimension of
        (number_of_neurons * max_delay) x (largest out-degree)
        
        If this causes address overflow, Try using TensorArray instead.
        
        """
        pre_inds = indices[:, 1]
        uni, counts = np.unique(pre_inds, return_counts=True)
        max_elem = np.max(counts)
        n_elem = n_source_neurons * self.max_delay
        n_syn = pre_inds.shape[0]
        
        # checking the possibility of address overflow
        if n_elem * max_elem > 2**31:
            # with my observation, this never happens with the current model.
            # with all 296991 neurons, the largest out-degree is 1548.
            # this results in 1,838,968,272, which is barely below 2**31 (~2.1 billion)
            print("n_elem: ", n_elem)
            print("max_elem: ", max_elem)
            print("n_elem * max_elem: ", n_elem * max_elem)
            print("n_elem * max_elem > 2**31")
            raise ValueError("It will cause address overflow. Time to think about a different approach.")
        
        @njit
        def make_table(pre_inds, n_elem, max_elem, n_syn):
            # first, make a big array to allocate memory
            arr = np.full((n_elem, max_elem), -1, dtype=np.int32)
            arr_inds = np.zeros(n_elem, dtype=np.int32)
            for i in range(n_syn):
                arr[pre_inds[i], arr_inds[pre_inds[i]]] = i
                arr_inds[pre_inds[i]] += 1
            return arr
        
        table = make_table(pre_inds, n_elem, max_elem, n_syn)
        # exit with int64 for faster processing on a GPU (don't know why...)
        table = tf.constant(table, dtype=tf.int64)
        return table

    def get_new_inds_table(self, indices, non_zero_cols, pre_ind_table):
        """ a new function that prepares new sparse indices tensor.
        This effectively does 'gather' operation for the sparse tensor.
        It utilizes the pre_ind_table to find the indices of the recurrent_indices

        """
        pre_inds = indices[:, 1]
        post_inds = indices[:, 0]
        all_inds = tf.gather(pre_ind_table, non_zero_cols)
        all_inds = tf.reshape(all_inds, [-1])  # flatten the tensor
        # remove unecessary -1's
        all_inds = tf.boolean_mask(all_inds, all_inds != -1)
        # if all_inds is empty, then return empty tensors
        if tf.size(all_inds) == 0:
            return tf.zeros((0, 2), dtype=tf.int64), tf.zeros((0,), dtype=tf.int64)
        else:
            # sort to make it compatible with sparse tensor creation
            inds = tf.sort(all_inds)
            remaining_pre = tf.gather(pre_inds, inds)
            _, idx = tf.unique(remaining_pre, out_idx=tf.int64)
            # if tf.size(inds) == 0:
            #     tf.print('OJITO: ', tf.size(inds))
            new_pre = tf.gather(idx, tf.range(tf.size(inds)))
            new_post = tf.gather(post_inds, inds)
            new_indices = tf.stack((new_post, new_pre), axis=1)

            return new_indices, inds

    def calculate_i_rec(self, rec_z_buf):
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        # This is a new faster implementation that uses the pre_ind_table 
        # Memory consumption and processing time depends on the number of spiking neurons
        # this faster method uses sparseness of the rec_z_buf.
        # it identifies the non_zero rows of rec_z_buf and only computes the
        # sparse matrix multiplication for those rows.
        rec_z_buf = tf.cast(rec_z_buf, self._compute_dtype)
        
        # find the non-zero rows of rec_z_buf
        non_zero_cols = tf.where(rec_z_buf)[:, 1]
        nnz = tf.size(non_zero_cols) # number of non zero
        if nnz == 0: # nothing is firing
            i_rec = tf.zeros((self._n_max_receptors * self._n_neurons, 1), dtype=self._compute_dtype)
        else:
            sliced_rec_z_buf = tf.gather(rec_z_buf, non_zero_cols, axis=1)
            # sliced_rec_z_buf = tf.cast(sliced_rec_z_buf, self._compute_dtype)
            # let's make sparse arrays for multiplication
            # new_indices will be a version of indices that only contains the non-zero columns
            # in the non_zero_cols, and changes the indices accordingly.
            new_indices, inds = self.get_new_inds_table(self.recurrent_indices, non_zero_cols, self.pre_ind_table)
            # print(inds.shape, inds)
            # print(tf.shape(inds)[0])
            if tf.size(inds) == 0:  # if firing cells do not have any outputs
                i_rec = tf.zeros((self._n_max_receptors * self._n_neurons, 1), dtype=self._compute_dtype)
            else:
                picked_weights = tf.gather(self.recurrent_weight_values, inds)
                sliced_sparse = tf.sparse.SparseTensor(
                        new_indices,
                        picked_weights,
                        [self.recurrent_dense_shape[0], nnz]
                    )
                i_rec = tf.sparse.sparse_dense_matmul(
                                                            sliced_sparse,
                                                            sliced_rec_z_buf,
                                                            adjoint_b=True
                                                        )
        return i_rec
    
    def calculate_i_inter(self, interarea_z_bufs, column_order):
        # This function performs the tensor multiplication to calculate the recurrent currents at each timestep
        # This is a new faster implementation that uses the pre_ind_table 
        # Memory consumption and processing time depends on the number of spiking neurons
        # this faster method uses sparseness of the rec_z_buf.
        # it identifies the non_zero rows of rec_z_buf and only computes the
        # sparse matrix multiplication for those rows.
        if self.interarea_indices[column_order] is not None:
            interarea_z_bufs = tf.cast(interarea_z_bufs, self._compute_dtype)
            # find the non-zero rows of rec_z_buf
            non_zero_cols = tf.where(interarea_z_bufs)[:, 1]
            nnz = tf.size(non_zero_cols)  # number of non zero
            if nnz == 0:
                i_interarea = tf.zeros((self._n_max_receptors * self._n_neurons, 1), dtype=self._compute_dtype)
            else:
                sliced_interarea_z_buf = tf.gather(interarea_z_bufs, non_zero_cols, axis=1)
                # sliced_interarea_z_buf = tf.cast(sliced_interarea_z_buf, self._compute_dtype)
                # let's make sparse arrays for multiplication
                # new_indices will be a version of indices that only contains the non-zero columns
                # in the non_zero_cols, and changes the indices accordingly.
                new_indices, inds = self.get_new_inds_table(self.interarea_indices[column_order], non_zero_cols, self.pre_interarea_ind_table[column_order])
                if tf.size(inds) == 0:  # if firing cells do not have any outputs
                    i_interarea = tf.zeros((self._n_max_receptors * self._n_neurons, 1), dtype=self._compute_dtype)
                else:
                    picked_weights = tf.gather(self.interarea_weight_values[column_order], inds)
                    sliced_sparse = tf.sparse.SparseTensor(
                            new_indices,
                            picked_weights,
                            [self.interarea_dense_shapes[column_order][0], nnz]
                        )
                    i_interarea = tf.sparse.sparse_dense_matmul(
                                                                sliced_sparse,
                                                                sliced_interarea_z_buf,
                                                                adjoint_b=True
                                                            )
        else:
            i_interarea = tf.zeros((self._n_max_receptors * self._n_neurons, 1), dtype=self._compute_dtype)
       
        return i_interarea
    
    def update_psc(self, psc, psc_rise, rec_inputs):
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        return new_psc, new_psc_rise

    def update_asc(self, asc_1, asc_2, prev_z):
        new_asc_1 = self.exp_dt_k_1 * asc_1 + prev_z * self.asc_amps[:, 0]
        new_asc_2 = self.exp_dt_k_2 * asc_2 + prev_z * self.asc_amps[:, 1]
        return new_asc_1, new_asc_2

    def zero_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * \
                tf.cast(self.v_th * .0 + 1. * self.v_reset, dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_20 = tf.zeros((batch_size, self._n_neurons), dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_max_receptors), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_max_receptors), dtype)
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    def _gather(self, prop):
        return tf.gather(prop, self._node_type_ids)

    def reshape_recurrent_currents(self, i_rec, batch_size):
        recurrent_currents_shape = (batch_size, self._n_neurons, self._n_max_receptors)
        return tf.reshape(i_rec, recurrent_currents_shape)

    # @profile
    def call(self, inputs, state, constants=None):
       
        # Get all the model inputs
        external_current = inputs[0]
        bkg_noise = inputs[1]
        interarea_z_bufs = inputs[2] # interarea_z_bufs[0] relative to lower area, interarea_z_bufs relative to higher area.
        state_input = inputs[3]

        batch_size = tf.shape(bkg_noise)[0]
        
        if self._spike_gradient:
            state_input = tf.zeros((1,))
        else:
            state_input = tf.zeros((4,))

        if constants is not None and not self._spike_gradient:
            state_input = self.reshape_recurrent_currents(state_input, batch_size)
        
        # Extract the network variables from the state
        z_buf, v, r, asc_1, asc_2, psc_rise, psc = state

        # Define the previous max_delay spike matrix
        shaped_z_buf = tf.reshape(z_buf, (-1, self.max_delay, self._n_neurons)) #shape (batch, delay, neurons)
        prev_z = shaped_z_buf[:, 0] # previous spikes with shape (neurons)
        dampened_z_buf = z_buf * self._dampening_factor  # dampened version of z_buf # no entiendo muy bien la utilidad de esto
        # Now we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained
        rec_z_buf = (tf.stop_gradient(z_buf - dampened_z_buf) + dampened_z_buf)  
        # print('rec_z_buf: ', rec_z_buf.shape)

        # Reshape the psc variables
        psc_rise = self.reshape_recurrent_currents(psc_rise, batch_size)
        psc = self.reshape_recurrent_currents(psc, batch_size)

        ### Calculate the recurrent input current ###
        i_rec = self.calculate_i_rec(rec_z_buf)
        i_rec = tf.transpose(i_rec)

        # ### Calculate the interarea input current ###
        if self._connected_areas:
            interarea_z_bufs = interarea_z_bufs[0]
            dampened_interarea_z_bufs = interarea_z_bufs * self._dampening_factor  # dampened version of z_buf # no entiendo muy bien la utilidad de esto
            # Now we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained
            interarea_z_buf = (tf.stop_gradient(interarea_z_bufs - dampened_interarea_z_bufs) + dampened_interarea_z_bufs)  
            i_interarea = tf.zeros((self._n_max_receptors * self._n_neurons, 1), dtype=self._compute_dtype)
            for column_order in [self.source_column_order]:
                i_interarea += self.calculate_i_inter(interarea_z_buf, column_order)
            i_interarea = tf.transpose(i_interarea) # shape (batch, neurons*receptors)
        else:
            i_interarea = tf.zeros((batch_size, self._n_neurons * self._n_max_receptors), dtype=self._compute_dtype)
        
        # Add all the current sources
        rec_inputs = self.reshape_recurrent_currents(i_rec + i_interarea + bkg_noise, batch_size)
        # rec_inputs = self.reshape_recurrent_currents(i_rec + bkg_noise, batch_size)
        if external_current is not None and self.name == 'v1': # only V1 area can receive external input
            external_current = self.reshape_recurrent_currents(external_current, batch_size)
            rec_inputs = rec_inputs + external_current
            
        rec_inputs = rec_inputs * self._lr_scale
        if constants is not None and not self._spike_gradient:
            rec_inputs = rec_inputs + state_input * self._lr_scale #(1, n_neurons, n_receptors)

        # Calculate the new psc variables
        new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)

        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)  # =max(r + prev_z * self.t_ref - self._dt, 0)

        # Calculate the ASC
        new_asc_1, new_asc_2 = self.update_asc(asc_1, asc_2, prev_z)

        input_current = tf.reduce_sum(psc, -1)
        if constants is not None and self._spike_gradient:
            input_current += state_input

        # Add all the postsynaptic current sources
        c1 = input_current + asc_1 + asc_2 + self.gathered_g

        # Calculate the new voltage values
        decayed_v = self.decay * v
        # Update the voltage according to the LIF equation and the refractory period
        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            new_v = tf.where(new_r > 0.0, self.v_reset, decayed_v + self.current_factor * c1)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period
        else:
            reset_current = prev_z * self.v_gap
            new_v = decayed_v + self.current_factor * c1 + reset_current

        # Generate the network spikes
        v_sc = (new_v - self.v_th) / self.normalizer
        if self._pseudo_gauss:
            if self._compute_dtype == tf.bfloat16:
                new_z = spike_function_b16(v_sc, self._dampening_factor)
            elif self._compute_dtype == tf.float16:
                new_z = spike_gauss_16(v_sc, self._gauss_std, self._dampening_factor)
            else:
                new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)
        else:
            if self._compute_dtype == tf.float16:
                new_z = spike_function_16(v_sc, self._dampening_factor)
            else:
                new_z = spike_function(v_sc, self._dampening_factor)

        # Generate the new spikes if the refractory period is concluded
        new_z = tf.where(new_r > 0.0, tf.zeros_like(new_z), new_z)

        # Reshape the network variables
        new_psc = tf.reshape(new_psc, (batch_size, self._n_neurons * self._n_max_receptors))
        new_psc_rise = tf.reshape(new_psc_rise, (batch_size, self._n_neurons * self._n_max_receptors))

        # Add current spikes to the buffer
        new_shaped_z_buf = tf.concat((new_z[:, None], shaped_z_buf[:, :-1]), 1)
        new_z_buf = tf.reshape(new_shaped_z_buf, (-1, self._n_neurons * self.max_delay))
        
        outputs = (
            new_z, 
            new_v * self.voltage_scale + self.voltage_offset,
            # (input_current + new_asc_1 + new_asc_2) * self.voltage_scale
            )

        new_state = (new_z_buf, 
                    new_v, 
                    new_r, 
                    new_asc_1,
                    new_asc_2, 
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
                 input_weight_scale, 
                 interarea_weight_scale, 
                 lr_scale,
                 spike_gradient, 
                 batch_size,
                 max_delay, 
                 pseudo_gauss, 
                 hard_reset, 
                 connected_areas,
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
                               gauss_std=gauss_std, dampening_factor=dampening_factor, 
                               input_weight_scale=input_weight_scale, interarea_weight_scale=interarea_weight_scale,
                               lr_scale=lr_scale, max_delay=max_delay, batch_size=batch_size,
                               pseudo_gauss=pseudo_gauss, spike_gradient=spike_gradient, train_recurrent=train_recurrent_v1, 
                               train_input=train_input, train_interarea=train_interarea_v1_lm, train_noise=train_noise, 
                               name='v1', hard_reset=hard_reset, connected_areas=connected_areas)

        self.lm = BillehColumn(networks['lm'], None, bkg_inputs['lm'],
                               gauss_std=gauss_std, dampening_factor=dampening_factor,
                               input_weight_scale=input_weight_scale, interarea_weight_scale=interarea_weight_scale,
                               lr_scale=lr_scale, max_delay=max_delay, batch_size=batch_size,
                               pseudo_gauss=pseudo_gauss, spike_gradient=spike_gradient, train_recurrent=train_recurrent_lm, 
                               train_input=train_input, train_interarea=train_interarea_lm_v1, train_noise=train_noise, 
                               name='lm', hard_reset=hard_reset, connected_areas=connected_areas)

        self._n_neurons = self.v1._n_neurons + self.lm._n_neurons

        self.state_size = (
            self.v1._n_neurons * self.v1.max_delay,   # z buffer
            self.v1._n_neurons,                        # v
            self.v1._n_neurons,                        # r
            self.v1._n_neurons,                        # asc 1
            self.v1._n_neurons,                        # asc 2
            self.v1._n_max_receptors * self.v1._n_neurons, # psc rise, double exponential synapses
            self.v1._n_max_receptors * self.v1._n_neurons, # psc
            self.lm._n_neurons * self.lm.max_delay,   # z buffer
            self.lm._n_neurons,                        # v
            self.lm._n_neurons,                        # r
            self.lm._n_neurons,                        # asc 1
            self.lm._n_neurons,                        # asc 2
            self.lm._n_max_receptors * self.lm._n_neurons, # psc rise, double exponential synapses
            self.lm._n_max_receptors * self.lm._n_neurons, # psc
        )
        self.num_elements_in_one_state = int(len(self.state_size)/self._n_areas)
    
    def zero_state_multi_areas(self, batch_size, dtype=tf.float32):
        multi_zero_state =  self.v1.zero_state(batch_size, dtype) + self.lm.zero_state(batch_size, dtype) # add the two states together, one after the other
        return multi_zero_state

    # @profile
    def call(self, inputs, state, constants=None):   
        external_current = inputs[:, :self.v1._n_neurons*self.v1._n_max_receptors] # first part is real external input
        bkg_input = inputs[:, self.v1._n_neurons*self.v1._n_max_receptors:-self._n_neurons] # background noise
        state_input = inputs[:, -self._n_neurons:] # dummy zeros
        
        v1_state, lm_state = state[:self.num_elements_in_one_state], state[-self.num_elements_in_one_state:]
        # breakpoint()
        interarea_z_bufs_to_v1 = (lm_state[0],)
        # print('interarea_z_bufs_to_v1, ', interarea_z_bufs_to_v1)
        inputs_to_v1 = (external_current, bkg_input[:,:self.v1._n_neurons*self.v1._n_max_receptors], interarea_z_bufs_to_v1, state_input[:,:self.v1._n_neurons])
        outputs_v1, new_v1_state = self.v1(inputs_to_v1, v1_state)

        interarea_z_bufs_to_lm = (v1_state[0],)
        # print('interarea_z_bufs_to_lm, ', interarea_z_bufs_to_lm)
        inputs_to_lm = (None, bkg_input[:,self.v1._n_neurons*self.v1._n_max_receptors:], interarea_z_bufs_to_lm, state_input[:,self.v1._n_neurons:])
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
                connected_areas=True,
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
                            input_weight_scale=input_weight_scale, 
                            interarea_weight_scale=interarea_weight_scale,
                            lr_scale=lr_scale, 
                            spike_gradient=True, 
                            batch_size=batch_size, 
                            max_delay=max_delay,
                            pseudo_gauss=pseudo_gauss, 
                            hard_reset=hard_reset,
                            connected_areas=connected_areas,
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
    rnn_inputs = SparseLayer(indices=cell.v1.input_indices, 
                            weights=cell.v1.input_weight_values, 
                            dense_shape=cell.v1.lgn_input_dense_shape, 
                            dtype=dtype, 
                            name='input_layer'
                            )(x)

    # Calculate the background noise input
    bkg_inputs = BackgroundNoiseLayer(cell, 
                                      batch_size=batch_size,
                                      seq_len=seq_len,
                                      dtype=dtype,
                                      bkg_units_firing_rate=250, 
                                      n_bkg_units=100,
                                      n_bkg_connections=4
                                    )(x) # (None, 2500, 4000+600)

    # Concatenate all the inputs together    
    full_inputs = tf.cast(tf.concat((rnn_inputs, bkg_inputs, state_input), -1), dtype) # (None, 2500, 120(LGN)+200(BKG)+50(dummy_zeros))

    # Create the RNN layer
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state, name='rsnn') # return_sequences=True return the full sequence. 
    
    # Run the model over the inputs
    out = rnn(full_inputs, initial_state=rnn_initial_state, constants=constants)
    
    # Extract the model output and its new state
    if return_state:
        hidden = out[0] #4
        new_state = out[1:] #(14)
    else:
        hidden = out

    # assume all areas have similar firing rates
    spikes_v1, voltage_v1, spikes_lm, voltage_lm = hidden  ## these are results in all time steps? Yes, if return_sequences=True

    rate_v1 = tf.cast(tf.reduce_mean(spikes_v1, (1, 2)), tf.float32)
    rate_lm = tf.cast(tf.reduce_mean(spikes_lm, (1, 2)), tf.float32)
    
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

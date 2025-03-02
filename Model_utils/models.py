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

        return [de_dv_scaled, None, None]

    return tf.identity(z_, name='spike_gauss'), grad

@tf.custom_gradient
def spike_gauss_16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None, None]

    return tf.identity(z_, name='spike_gauss'), grad

@tf.custom_gradient
def spike_gauss_b16(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None, None]

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

        return [de_dv_scaled, None]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def spike_function_16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def spike_function_b16(v_scaled, dampening_factor):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.bfloat16)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None]

    return tf.identity(z_, name="spike_function"), grad

@tf.custom_gradient
def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values, dense_shape, pre_ind_table):
    # Get the batch size and number of neurons
    batch_size = tf.cast(tf.shape(rec_z_buf)[0], dtype=tf.int64)
    n_post_neurons = dense_shape[0] # Assuming square weight matrix
    # Find the indices of non-zero spikes in rec_z_buf
    # non_zero_indices: [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
    non_zero_indices = tf.where(rec_z_buf > 0)
    batch_indices = non_zero_indices[:, 0]         # Shape: [num_non_zero_spikes]
    pre_neuron_indices = non_zero_indices[:, 1]    # Shape: [num_non_zero_spikes]
    # Get the indices into self.synapse_indices for each pre_neuron_index
    # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in synapse_indices
    new_indices, new_weights, post_in_degree, all_synaptic_inds = get_new_inds_table(synapse_indices, weight_values, pre_neuron_indices, pre_ind_table)
    # Expand batch_indices to match the length of inds_flat
    batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
    # batch_indices_per_connection: Shape: [total_num_connections]
    post_neuron_indices = new_indices[:, 0]  # Indices of post-synaptic neurons
    # Compute segment_ids for unsorted_segment_sum
    # We need to combine batch_indices and post_neuron_indices to create unique segment IDs
    segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
    num_segments = batch_size * n_post_neurons
    # Compute the recurrent currents using unsorted_segment_sum
    i_rec_flat = tf.math.unsorted_segment_sum(
        new_weights,
        segment_ids,
        num_segments=num_segments
    )
    
    def grad(dy): #, variables=None): # variables: List of variables used in forward pass (weight_values)
        # Compute gradient w.r.t rec_z_buf
        dy_reshaped = tf.reshape(dy, [batch_size, -1])
        # Recompute sparse_w_rec
        sparse_w_rec = tf.sparse.SparseTensor(synapse_indices, weight_values, dense_shape)
        de_dv = tf.sparse.sparse_dense_matmul(dy_reshaped, sparse_w_rec, adjoint_a=False)
        de_dv = tf.cast(de_dv, dtype=rec_z_buf.dtype)
        
        # Compute gradient w.r.t weight_values
        de_dnew_weights = tf.gather(dy, segment_ids)
        # Scatter gradients back to weight_values
        # de_dweight_values = tf.tensor_scatter_nd_add(
        #     tf.zeros_like(weight_values),
        #     indices=tf.expand_dims(all_synaptic_inds, axis=1),
        #     updates=de_dnew_weights
        # )
        de_dweight_values = tf.math.unsorted_segment_sum(
            data=de_dnew_weights,
            segment_ids=all_synaptic_inds,
            num_segments=tf.shape(weight_values)[0]
        )
        # Return gradients w.r.t inputs and variables
        return [
            de_dv,              # Gradient w.r.t rec_z_buf
            None,               # synapse_indices (constant)
            de_dweight_values,  # weight_values (trainable)
            None,               # dense_shape[0] (constant)
            None,               # dense_shape[1] (constant) 
            None,               # pre_ind_table (constant)
        ]
    
    return i_rec_flat, grad  # Return forward pass result and grad function


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
        # Replace -1 with valid receptor IDs different from the ones assigned
        assigned_receptors = new_receptor_id_to_old_receptor_id[neuron_id, :idx].astype(np.int32)
        replacement_id = 0
        for j in range(idx, max_receptors_per_neuron):
            while replacement_id in assigned_receptors:
                replacement_id += 1
            new_receptor_id_to_old_receptor_id[neuron_id, j] = replacement_id
            # append the replacement id to the assigned receptors array
            assigned_receptors = np.append(assigned_receptors, np.int32(replacement_id))          
            replacement_id += 1

    return max_receptors_per_neuron, new_receptor_id_to_old_receptor_id, old_receptor_id_to_new_receptor_id

def make_pre_ind_table(indices, n_source_neurons=197613):
    """
    This function creates a table that maps presynaptic indices to 
    the indices of the recurrent_indices tensor using a RaggedTensor.
    This approach ensures that every presynaptic neuron, even those with no
    postsynaptic connections, has an entry in the RaggedTensor.
    """
    # Extract presynaptic IDs
    pre_ids = indices[:, 1]  # shape: [num_synapses]
    # Sort the synapses by presynaptic ID
    sort_idx = tf.argsort(pre_ids, axis=0)
    sorted_pre = tf.gather(pre_ids, sort_idx)
    # Count how many synapses belong to each presynaptic neuron
    # (We cast to int32 for tf.math.bincount.)
    counts = tf.math.bincount(tf.cast(sorted_pre, tf.int32), minlength=n_source_neurons)
    # Build row_splits to define a RaggedTensor from these sorted indices
    row_splits = tf.concat([[0], tf.cumsum(counts)], axis=0)
    # The values of the RaggedTensor are the original synapse-array row indices,
    # but sorted by presyn neuron
    rt = tf.RaggedTensor.from_row_splits(sort_idx, row_splits, validate=False)

    return rt

def get_new_inds_table(indices, weights, non_zero_cols, pre_ind_table):
    """Optimized function that prepares new sparse indices tensor."""
    # Gather the rows corresponding to the non_zero_cols
    selected_rows = tf.gather(pre_ind_table, non_zero_cols)
    # Flatten the selected rows to get all_inds
    all_synaptic_inds = selected_rows.flat_values
    # get the number of postsynaptic neurons 
    post_in_degree = selected_rows.row_lengths()
    # Gather from indices using all_inds
    new_indices = tf.gather(indices, all_synaptic_inds)
    # Gather from weights using all_inds
    new_weights = tf.gather(weights, all_synaptic_inds)

    return new_indices, new_weights, post_in_degree, all_synaptic_inds


# class BackgroundNoiseLayer(tf.keras.layers.Layer):
#     def __init__(self, cell, batch_size,
#                  bkg_firing_rate=1000, 
#                  n_bkg_units=1,# n_bkg_connections = 1, 
#                   **kwargs):
#         super().__init__(**kwargs)
#         self._batch_size = batch_size
#         # self._seq_len = seq_len
#         self._bkg_firing_rate = bkg_firing_rate
#         self._n_bkg_units = n_bkg_units
#         # self._n_bkg_connections = n_bkg_connections

#         # Initialize weights and indices
#         self._bkg_weights = {'v1': None, 'lm': None}
#         self._bkg_indices = {'v1': None, 'lm': None}
#         self._dense_shape = {'v1': None, 'lm': None}
#         # self._initialize_background_inputs(cell)
#         # Create connectivity and assign weights for 'v1' and 'lm' areas with new pool of 100 Poisson sources
#         for column in ['v1', 'lm']:
#             weights = cell.__getattribute__(column).bkg_input_weights
#             indices = cell.__getattribute__(column).bkg_input_indices
#             dense_shape = cell.__getattribute__(column).bkg_input_dense_shape
#             self._bkg_weights[column] = weights
#             self._bkg_indices[column] = indices
#             self._dense_shape[column] = dense_shape

#     # def _initialize_background_inputs(self, cell):
#     #     # Create connectivity and assign weights for 'v1' and 'lm' areas with new pool of 100 Poisson sources
#     #     for column in ['v1', 'lm']:
#     #         original_weights = cell.__getattribute__(column).bkg_input_weights
#     #         original_indices = cell.__getattribute__(column).bkg_input_indices
#     #         original_dense_shape = cell.__getattribute__(column).bkg_input_dense_shape

#     #         if self._n_bkg_connections == 1:
#     #             indices = original_indices
#     #             weights = original_weights
#     #         else:
#     #             # Generate random connections
#     #             new_bkg_indices = tf.random.uniform(shape=(original_indices.shape[0], self._n_bkg_connections), minval=0, maxval=self._n_bkg_units, dtype=tf.int64)
#     #             indices = tf.reshape(tf.stack([tf.repeat(original_indices[:, 0], self._n_bkg_connections), tf.reshape(new_bkg_indices, [-1])], axis=1), [-1, 2])
#     #             # this implementation allows a neuron to establish more than one connection to a single BKG unit
#     #             # Repeat weights for each connection
#     #             weights = tf.repeat(original_weights, self._n_bkg_connections)
            
#     #         # Create a new constraint based on the new weights
#     #         new_bkg_input_weight_positive = tf.Variable(weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
#     #         new_constraint = SignedConstraint(new_bkg_input_weight_positive)
#     #         cell.__getattribute__(column).bkg_input_weights = tf.Variable(weights, 
#     #                                                                     name=column+'_rest_of_brain_weights', 
#     #                                                                     constraint=new_constraint,
#     #                                                                     dtype=original_weights.dtype,
#     #                                                                     trainable=original_weights.trainable)

#     #         self._bkg_weights[column] = cell.__getattribute__(column).bkg_input_weights
#     #         self._bkg_indices[column] = tf.Variable(indices, trainable=False, dtype=tf.int64)
#     #         self._dense_shape[column] = (original_dense_shape[0],  self._n_bkg_units)

#     def calculate_bkg_i_in(self, inputs, column='v1'):
#         # Define the sparse weight matrix (need to be defined here for the gradient to work)
#         sparse_w_in = tf.sparse.SparseTensor(
#             self._bkg_indices[column],
#             self._bkg_weights[column], 
#             self._dense_shape[column]
#         )
#         i_in = tf.sparse.sparse_dense_matmul(
#                                             sparse_w_in, 
#                                             inputs, 
#                                             adjoint_b=True
#                                             )
#         # Optionally cast the output back to float16
#         if i_in.dtype != self.compute_dtype:
#             i_in = tf.cast(i_in, dtype=self.compute_dtype)

#         return i_in

#     @tf.function
#     def call(self, inp):
#         seq_len = tf.shape(inp)[1]

#         rest_of_brain = tf.random.poisson(shape=(self._batch_size * seq_len, self._n_bkg_units), 
#                                           lam=self._bkg_firing_rate/1000, 
#                                           dtype=self.variable_dtype) # this implementation is slower but allows to produce proper Poisson values (not just 0's and 1's)
#         # rest_of_brain = tf.cast(tf.random.uniform(
#         #         (self._batch_size * seq_len, self._n_bkg_units)) < self._bkg_firing_rate * .001, 
#         #         tf.float32) # (1, 600, 100)
#         # rest_of_brain = tf.reshape(rest_of_brain, (self._batch_size * seq_len, self._n_bkg_units))

#         noise_inputs = {'v1': None, 'lm': None}
#         for column in noise_inputs.keys():
#             noise_input = self.calculate_bkg_i_in(rest_of_brain, column=column)
#             noise_input = tf.transpose(noise_input)
#             noise_inputs[column] = tf.reshape(noise_input, (self._batch_size, seq_len, -1))
        
#         cat_noise_inputs = tf.concat([noise_inputs['v1'], noise_inputs['lm']], axis=-1)

#         return cat_noise_inputs

# class BKGInputLayerCell(tf.keras.layers.Layer):
#     def __init__(self, cell, n_bkg_units, n_bkg_connections, **kwargs):
#         super().__init__(**kwargs)
#         self._n_bkg_units = n_bkg_units
#         self._n_bkg_connections = n_bkg_connections
#         # Initialize weights and indices
#         self._bkg_weights = {'v1': None, 'lm': None}
#         self._bkg_indices = {'v1': None, 'lm': None}
#         self._dense_shape = {'v1': None, 'lm': None}
#         self._pre_ind_table = {'v1': None, 'lm': None}
#         self._initialize_background_inputs(cell)

#     def _initialize_background_inputs(self, cell):
#         # Create connectivity and assign weights for 'v1' and 'lm' areas with new pool of Poisson sources
#         for column in ['v1', 'lm']:
#             original_weights = cell.__getattribute__(column).bkg_input_weights
#             original_indices = cell.__getattribute__(column).bkg_input_indices
#             original_dense_shape = cell.__getattribute__(column).bkg_input_dense_shape

#             if self._n_bkg_connections == 1:
#                 indices = original_indices
#                 weights = original_weights
#             else:
#                 # Generate random connections
#                 new_bkg_indices = tf.random.uniform(
#                     shape=(tf.shape(original_indices)[0], self._n_bkg_connections),
#                     minval=0,
#                     maxval=self._n_bkg_units,
#                     dtype=tf.int64,
#                     seed=42
#                 )
#                 # Combine post-synaptic neuron indices with new pre-synaptic neuron indices
#                 indices = tf.stack([
#                     tf.repeat(original_indices[:, 0], self._n_bkg_connections),
#                     tf.reshape(new_bkg_indices, [-1])
#                 ], axis=1)
#                 # Repeat weights for each connection
#                 weights = tf.repeat(original_weights, self._n_bkg_connections)

#             # Create a new constraint based on the new weights
#             new_bkg_input_weight_positive = tf.Variable(weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
#             new_constraint = SignedConstraint(new_bkg_input_weight_positive)
#             cell.__getattribute__(column).bkg_input_weights = tf.Variable(
#                 weights,
#                 name=column+'_rest_of_brain_weights',
#                 constraint=new_constraint,
#                 dtype=original_weights.dtype,
#                 trainable=original_weights.trainable
#             )

#             self._bkg_weights[column] = cell.__getattribute__(column).bkg_input_weights
#             self._bkg_indices[column] = tf.Variable(indices, trainable=False, dtype=tf.int64)
#             self._dense_shape[column] = (original_dense_shape[0], self._n_bkg_units)

#             # Precompute the synapses table
#             self._pre_ind_table[column] = make_pre_ind_table(
#                 indices,
#                 n_source_neurons=self._n_bkg_units
#             )

#     @property
#     def state_size(self):
#         # No states are maintained in this cell
#         return []

    # def call(self, inputs_t, states):
    #     # inputs_t: Shape [batch_size, input_dim]
    #     batch_size = tf.shape(inputs_t)[0]
    #     n_bkg_units = tf.shape(inputs_t)[1]  # Number of background units

    #     # Find the indices of non-zero spikes in inputs_t
    #     # non_zero_indices: Shape [num_non_zero_inputs, 2], columns are [batch_index, pre_neuron_index]
    #     non_zero_indices = tf.where(inputs_t > 0)
    #     batch_indices = non_zero_indices[:, 0]         # Shape: [num_non_zero_inputs]
    #     pre_neuron_indices = non_zero_indices[:, 1]    # Shape: [num_non_zero_inputs]

    #     noise_inputs = {}
    #     for column in ['v1', 'lm']:
    #         # Get the synapse indices for each active pre-synaptic neuron
    #         inds = tf.gather(self._pre_ind_table[column], pre_neuron_indices)
    #         # inds: RaggedTensor of shape [num_non_zero_inputs, None]
    #         # Flatten inds to get a 1D tensor of synapse indices
    #         inds_flat = inds.flat_values  # Shape: [total_num_connections]
    #         # Expand batch_indices to align with inds_flat
    #         row_lengths = inds.row_lengths()  # Shape: [num_non_zero_inputs]
    #         batch_indices_per_connection = tf.repeat(batch_indices, row_lengths)
    #         # batch_indices_per_connection: Shape: [total_num_connections]
    #         # Get the synapse data
    #         synapse_indices = tf.gather(self._bkg_indices[column], inds_flat)  # Shape: [total_num_connections, 2]
    #         post_neuron_indices = synapse_indices[:, 0]  # Shape: [total_num_connections]
    #         pre_neuron_indices_syn = synapse_indices[:, 1]  # Shape: [total_num_connections]
    #         # Gather n_pre_spikes for each connection
    #         n_pre_spikes = tf.gather_nd(
    #             inputs_t,
    #             tf.stack([batch_indices_per_connection, pre_neuron_indices_syn], axis=1)
    #         )
    #         n_pre_spikes = tf.cast(n_pre_spikes, dtype=self.variable_dtype)
    #         # Gather the weights
    #         weights_flat = tf.gather(self._bkg_weights[column], inds_flat)  # Shape: [total_num_connections]
    #         # Multiply weights by n_pre_spikes
    #         weighted_inputs = weights_flat * n_pre_spikes  # Shape: [total_num_connections]
    #         # Compute segment_ids for unsorted_segment_sum
    #         num_post_neurons = self._dense_shape[column][0]
    #         segment_ids = batch_indices_per_connection * num_post_neurons + post_neuron_indices
    #         num_segments = batch_size * num_post_neurons
    #         # Calculate the total input current received by each neuron
    #         i_in_flat = tf.math.unsorted_segment_sum(
    #             weighted_inputs,
    #             segment_ids,
    #             num_segments=num_segments
    #         )  # Shape: [num_segments]

    #         # Optionally cast the output back to the compute dtype
    #         if i_in_flat.dtype != self.compute_dtype:
    #             i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)
    #         # Reshape i_in_flat back to [batch_size, num_post_neurons]
    #         i_in = tf.reshape(i_in_flat, [batch_size, num_post_neurons])

    #         noise_inputs[column] = i_in  # Shape: [batch_size, num_post_neurons]

    #     # Concatenate the noise inputs from 'v1' and 'lm'
    #     cat_noise_inputs = tf.concat([noise_inputs['v1'], noise_inputs['lm']], axis=-1)  # Shape: [batch_size, total_post_neurons]

    #     # Since no states are maintained, return empty state
    #     return cat_noise_inputs, []

# class BKGInputLayer(tf.keras.layers.Layer):
#     """
#     Calculates input currents from the background (BKG) by processing one timestep at a time using a custom RNN cell.
#     """
#     def __init__(self, cell, batch_size, bkg_firing_rate=250, n_bkg_units=100, n_bkg_connections=4, **kwargs):
#         super().__init__(**kwargs)
#         self.input_cell = BKGInputLayerCell(
#             cell, n_bkg_units, n_bkg_connections, **kwargs
#         )
#         self._batch_size = batch_size
#         self._bkg_firing_rate = bkg_firing_rate
#         self._n_bkg_units = n_bkg_units
#         # Create the input RNN layer with the custom cell to recursively process all the inputs by timesteps
#         self.input_rnn = tf.keras.layers.RNN(
#             self.input_cell,
#             return_sequences=True,
#             return_state=False,
#             name='noise_rsnn'
#         )

#     @tf.function
#     def call(self, inputs, **kwargs):
#         # inputs: Shape [batch_size, seq_len, input_dim]
#         seq_len = tf.shape(inputs)[1]
#         # # Alternatively, generate Poisson spike trains if inputs are not provided
#         rest_of_brain = tf.random.poisson(
#             shape=(self._batch_size, seq_len, self._n_bkg_units),
#             lam=self._bkg_firing_rate * 0.001,
#             dtype=self.variable_dtype
#         )
#         input_current = self.input_rnn(rest_of_brain, **kwargs)  # Outputs: [batch_size, seq_len, total_post_neurons]

#         return input_current
    

class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        self._positive = positive

    def __call__(self, w):
        condition = tf.greater(self._positive, 0)  # yields bool
        sign_corrected_w = tf.where(condition, tf.nn.relu(w), -tf.nn.relu(-w))
        return sign_corrected_w


class SparseSignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, mask, positive):
        self._mask = mask
        self._positive = positive

    def __call__(self, w):
        condition = tf.greater(self._positive, 0)  # yields bool
        sign_corrected_w = tf.where(condition, tf.nn.relu(w), -tf.nn.relu(-w))
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
        batch_size=1,
        bkg_firing_rate=250,
        pseudo_gauss=False, 
        spike_gradient=False,
        train_recurrent=True, 
        train_input=False, 
        train_noise=True,
        train_interarea=True, 
        hard_reset=False, 
        connected_areas=True,
        connected_recurrent_connections=True, 
        connected_noise=True,
        current_input=False,
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
        self._batch_size = batch_size
        self._gauss_std = tf.constant(gauss_std, self.compute_dtype)
        self._bkg_firing_rate = bkg_firing_rate
        self._current_input = current_input
        # determine the membrane time decay constant
        tau = _params['C_m'] / _params['g'] 
        membrane_decay = np.exp(-dt / tau)
        current_factor = 1 / _params['C_m'] * (1 - membrane_decay) * tau
        
        # # create a new variable with all the postsynatpic indices concatenated: network["synapses"]["indices"], lgn_input["indices"],...
        all_postsynaptic_indices = np.concatenate([network["synapses"]["indices"][:, 0], bkg_input["indices"][:, 0]], axis=0)
        all_receptor_ids = np.concatenate([network["synapses"]["receptor_ids"], bkg_input["receptor_ids"]], axis=0)
        if lgn_input is not None:
            all_postsynaptic_indices = np.concatenate([all_postsynaptic_indices, lgn_input["indices"][:, 0]], axis=0)
            all_receptor_ids = np.concatenate([all_receptor_ids, lgn_input["receptor_ids"]], axis=0)
        for order in network['interarea_synapses'].keys():
            if network['interarea_synapses'][order]['indices'] is not None:
                all_postsynaptic_indices = np.concatenate([all_postsynaptic_indices, network['interarea_synapses'][order]['indices'][:, 0]], axis=0)
                all_receptor_ids = np.concatenate([all_receptor_ids, network['interarea_synapses'][order]['receptor_ids']], axis=0)

        # all_interarea_postsynaptic_indices = np.concatenate([network['interarea_synapses'][order]['indices'][:, 0] for order in network['interarea_synapses'].keys()])
        # all_interarea_receptor_ids = np.concatenate([network['interarea_synapses'][order]['receptor_ids'] for order in network['interarea_synapses'].keys()])
        # if lgn_input is not None:
        #     all_postsynaptic_indices = np.concatenate([network["synapses"]["indices"][:, 0], lgn_input["indices"][:, 0], bkg_input["indices"][:, 0], all_interarea_postsynaptic_indices], axis=0)
        #     all_receptor_ids = np.concatenate([network["synapses"]["receptor_ids"], lgn_input["receptor_ids"], bkg_input["receptor_ids"], all_interarea_receptor_ids], axis=0)
        # else:
        #     all_postsynaptic_indices = np.concatenate([network["synapses"]["indices"][:, 0], bkg_input["indices"][:, 0], all_interarea_postsynaptic_indices], axis=0)
        #     all_receptor_ids = np.concatenate([network["synapses"]["receptor_ids"], bkg_input["receptor_ids"], all_interarea_receptor_ids], axis=0)

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
            _v = tf.Variable(
                tf.cast(inv_sigmoid(_gather(_v)), self.compute_dtype), 
                trainable=trainable,
                )

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
        # self.recurrent_indices = tf.constant(indices, dtype=tf.int64)
        self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=self.recurrent_dense_shape[1])
        # Set the sign of the connections (exc or inh)
        # recurrent_weight_positive = tf.Variable(weights >= 0., name='recurrent_weights_sign', trainable=False)
        recurrent_weight_positive = tf.cast(weights >= 0, dtype=tf.int8)
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
        self.input_dim = lgn_input["n_inputs"] if lgn_input is not None else 0
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
            if not self._current_input:
                self.pre_input_ind_table  = make_pre_ind_table(input_indices, n_source_neurons=self.lgn_input_dense_shape[1])
            # input_weight_positive = tf.Variable(input_weights >= 0., name=self.name+'_input_weights_sign', trainable=False)
            input_weight_positive = tf.cast(input_weights >= 0, dtype=tf.int8)
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
        new_bkg_input_receptor_ids = old_receptor_id_to_new_receptor_id[bkg_input_indices[:, 0], bkg_input_receptor_ids]
        bkg_input_indices[:, 0] = bkg_input_indices[:, 0] * self._n_receptors + new_bkg_input_receptor_ids
        self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int64)
        # self.pre_bkg_ind_table  = make_pre_ind_table(bkg_input_indices, n_source_neurons=self.bkg_input_dense_shape[1])
        # Define Tensorflow variables
        # bkg_input_weight_positive = tf.Variable(bkg_input_weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
        bkg_input_weight_positive = tf.constant(bkg_input_weights >= 0, dtype=tf.int8)
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
            
            # interarea_weight_positive = tf.Variable(interarea_weights >= 0., name=self.name + '_interarea_weights_sign_'+self.source_column_order, trainable=False)
            interarea_weight_positive = tf.cast(interarea_weights >= 0, dtype=tf.int8)
            
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
            self.interarea_weight_values = {self.source_column_order: interarea_weight_values}
            self.interarea_dense_shapes = {self.source_column_order: interarea_dense_shape}
            self.interarea_indices = {self.source_column_order: interarea_indices}
            self.pre_interarea_ind_table = {self.source_column_order: pre_interarea_ind_table}

        else:
            self.interarea_weight_values, self.interarea_dense_shape, self.interarea_weight_positive, self.interarea_indices = None, None, None, None

    def calculate_input_current_from_firing_probabilities(self, x_t):
        """
        Calculate the input current from the LGN neurons .
        """
        sparse_w_in = tf.sparse.SparseTensor(
            self.input_indices,
            self.input_weight_values, 
            self.lgn_input_dense_shape
        )
        i_in = tf.sparse.sparse_dense_matmul(
                                            sparse_w_in, 
                                            tf.cast(x_t, dtype=self.variable_dtype), 
                                            adjoint_b=True
                                            )
        # Optionally cast the output back to float16
        if i_in.dtype != self.compute_dtype:
            i_in = tf.cast(i_in, dtype=self.compute_dtype)

        # flat the output
        i_in = tf.transpose(i_in)
        i_in_flat = tf.reshape(i_in, [-1])

        return i_in_flat

    def calculate_input_current_from_spikes(self, x_t):
        # x_t: Shape [batch_size, input_dim]
        # batch_size = tf.shape(x_t)[0]
        n_post_neurons = self.lgn_input_dense_shape[0]
        # if self.pre_input_ind_table is None:
        #     return tf.zeros((self._batch_size, n_post_neurons), dtype=self.compute_dtype)
        # Find the indices of non-zero inputs
        non_zero_indices = tf.where(x_t > 0)
        batch_indices = non_zero_indices[:, 0]
        pre_neuron_indices = non_zero_indices[:, 1]
        # Get the indices into self.recurrent_indices for each pre_neuron_index
        # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        new_indices, new_weights, post_in_degree, _ = get_new_inds_table(self.input_indices, self.input_weight_values, pre_neuron_indices, self.pre_input_ind_table)
        # Expand batch_indices to match the length of inds_flat
        # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
        batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
        # Get post-synaptic neuron indices
        post_neuron_indices = new_indices[:, 0]        
        # Compute segment IDs
        segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        num_segments = self._batch_size * n_post_neurons  
        # Calculate input currents
        i_in_flat = tf.math.unsorted_segment_sum(
            new_weights,
            segment_ids,
            num_segments=num_segments
        )
        # Cast i_rec to the compute dtype if necessary
        if i_in_flat.dtype != self.compute_dtype:
            i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        return i_in_flat

    def calculate_noise_current(self, rest_of_brain):
        sparse_w_bkg = tf.sparse.SparseTensor(
            self.bkg_input_indices,
            self.bkg_input_weights, 
            self.bkg_input_dense_shape
        )
        i_in = tf.sparse.sparse_dense_matmul(
                                            sparse_w_bkg, 
                                            tf.cast(rest_of_brain, dtype=self.variable_dtype), 
                                            adjoint_b=True
                                            )
        # Optionally cast the output back to float16
        if i_in.dtype != self.compute_dtype:
            i_in = tf.cast(i_in, dtype=self.compute_dtype)

        # flat the output
        i_in = tf.transpose(i_in)
        i_in_flat = tf.reshape(i_in, [-1])

        return i_in_flat

    # def calculate_noise_current(self, rest_of_brain):
    #     # x_t: Shape [batch_size, input_dim]
    #     batch_size = tf.shape(rest_of_brain)[0]
    #     n_post_neurons = self.bkg_input_dense_shape[0]
    #     if self.pre_bkg_ind_table is None:
    #         return tf.zeros((batch_size, n_post_neurons), dtype=self.compute_dtype)
    #     # Find the indices of non-zero inputs
    #     non_zero_indices = tf.where(rest_of_brain > 0)
    #     batch_indices = non_zero_indices[:, 0]
    #     pre_neuron_indices = non_zero_indices[:, 1]
    #     # Get the indices into self.recurrent_indices for each pre_neuron_index
    #     # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
    #     new_indices, new_weights, post_in_degree, _ = get_new_inds_table(self.bkg_input_indices, self.bkg_input_weights, pre_neuron_indices, self.pre_bkg_ind_table)
    #     # Expand batch_indices to match the length of inds_flat
    #     # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
    #     batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
    #     # Get post-synaptic neuron indices
    #     post_neuron_indices = new_indices[:, 0]        
    #     # Compute segment IDs
    #     segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
    #     num_segments = batch_size * n_post_neurons  
    #     # Get the number of presynaptic spikes
    #     presynaptic_indices = tf.stack([batch_indices_per_connection, new_indices[:, 1]], axis=1)
    #     n_pre_spikes = tf.cast(tf.gather_nd(rest_of_brain, presynaptic_indices), dtype=self.variable_dtype)
    #     new_weights = new_weights * n_pre_spikes
    #     # Calculate input currents
    #     i_in_flat = tf.math.unsorted_segment_sum(
    #         new_weights,
    #         segment_ids,
    #         num_segments=num_segments
    #     )
    #     # Cast i_rec to the compute dtype if necessary
    #     if i_in_flat.dtype != self.compute_dtype:
    #         i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

    #     return i_in_flat
    
    # def calculate_i_rec(self, rec_z_buf):
    #     # Get the batch size and number of neurons
    #     batch_size = tf.shape(rec_z_buf)[0]
    #     n_post_neurons = self.recurrent_dense_shape[0]
    #     # Find the indices of non-zero spikes in rec_z_buf
    #     # non_zero_indices: [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
    #     non_zero_indices = tf.where(rec_z_buf > 0)
    #     batch_indices = non_zero_indices[:, 0]         # Shape: [num_non_zero_spikes]
    #     pre_neuron_indices = non_zero_indices[:, 1]    # Shape: [num_non_zero_spikes]
    #     # Get the indices into self.recurrent_indices for each pre_neuron_index
    #     # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
    #     new_indices, new_weights, post_in_degree, _ = get_new_inds_table(self.recurrent_indices, self.recurrent_weight_values, pre_neuron_indices, self.pre_ind_table)
    #     # Expand batch_indices to match the length of inds_flat
    #     # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
    #     batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
    #     # batch_indices_per_connection: Shape: [total_num_connections]
    #     post_neuron_indices = new_indices[:, 0]  # Indices of post-synaptic neurons
    #     # Compute segment_ids for unsorted_segment_sum
    #     # We need to combine batch_indices and post_neuron_indices to create unique segment IDs
    #     segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
    #     num_segments = batch_size * n_post_neurons
    #     # Compute the recurrent currents using unsorted_segment_sum
    #     i_rec_flat = tf.math.unsorted_segment_sum(
    #         new_weights,
    #         segment_ids,
    #         num_segments=num_segments
    #     )
    #     # Cast i_rec to the compute dtype if necessary
    #     if i_rec_flat.dtype != self.compute_dtype:
    #         i_rec_flat = tf.cast(i_rec_flat, dtype=self.compute_dtype)

    #     return i_rec_flat

    # def calculate_i_rec(self, rec_z_buf):
    #     sparse_w_rec = tf.sparse.SparseTensor(
    #         self.recurrent_indices, self.recurrent_weight_values, self.recurrent_dense_shape)
    #     i_rec = tf.sparse.sparse_dense_matmul(
    #         sparse_w_rec, tf.cast(rec_z_buf, dtype=self.variable_dtype), adjoint_b=True)
    #     # Optionally cast the output back to the compute dtype
    #     if i_rec.dtype != self.compute_dtype:
    #         i_rec = tf.cast(i_rec, dtype=self.compute_dtype)

    #     i_rec = tf.transpose(i_rec)
    #     i_rec = tf.reshape(i_rec, [-1])
    #     return i_rec

    def calculate_i_rec_with_custom_grad(self, rec_z_buf):          
        i_rec_flat = calculate_synaptic_currents(rec_z_buf, self.recurrent_indices, self.recurrent_weight_values, self.recurrent_dense_shape, self.pre_ind_table)
        # # Cast i_rec to the compute dtype if necessary
        if i_rec_flat.dtype != self.compute_dtype:
            i_rec_flat = tf.cast(i_rec_flat, dtype=self.compute_dtype)

        return i_rec_flat

    def calculate_i_inter_with_custom_grad(self, interarea_z_bufs, column_order):
        
        if self.interarea_indices[column_order] is None:
            batch_size = tf.shape(interarea_z_bufs)[0]
            n_post_neurons = self.interarea_dense_shapes[column_order][0]
            return tf.zeros((batch_size, n_post_neurons), dtype=self.compute_dtype)
        
        i_inter_flat = calculate_synaptic_currents(interarea_z_bufs, self.interarea_indices[column_order], self.interarea_weight_values[column_order], self.interarea_dense_shapes[column_order], self.pre_interarea_ind_table[column_order])
         # Cast to the compute dtype if necessary
        if i_inter_flat.dtype != self.compute_dtype:
            i_inter_flat = tf.cast(i_inter_flat, dtype=self.compute_dtype)

        return i_inter_flat
        
    # def calculate_i_inter(self, interarea_z_bufs, column_order):
        # interarea_z_bufs: Shape [batch_size, n_pre_neurons]
        # batch_size = tf.shape(interarea_z_bufs)[0]
        # n_post_neurons = self.interarea_dense_shapes[column_order][0]

        # if self.interarea_indices[column_order] is None:
        #     return tf.zeros((batch_size, n_post_neurons), dtype=self.compute_dtype)
        # # Find the indices of non-zero spikes in interarea_z_bufs
        # # non_zero_indices: Shape [num_non_zero_spikes, 2], columns are [batch_index, pre_neuron_index]
        # non_zero_indices = tf.where(interarea_z_bufs > 0)
        # batch_indices = non_zero_indices[:, 0]         # Shape: [num_non_zero_spikes]
        # pre_neuron_indices = non_zero_indices[:, 1]    # Shape: [num_non_zero_spikes]
        # # Get the indices into self.recurrent_indices for each pre_neuron_index
        # # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        # new_indices, new_weights, post_in_degree, _ = get_new_inds_table(self.interarea_indices[column_order], self.interarea_weight_values[column_order], pre_neuron_indices, self.pre_interarea_ind_table[column_order])
        # # Expand batch_indices to match the length of inds_flat
        # # row_lengths = selected_rows.row_lengths()  # Shape: [num_non_zero_spikes]
        # batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
        # # batch_indices_per_connection: Shape: [total_num_connections]
        # # Get the post-synaptic neuron indices from interarea_indices using inds_flat
        # post_neuron_indices = new_indices[:, 0]  # Shape: [total_num_connections]
        # # Compute segment IDs by combining batch indices and post-neuron indices
        # segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        # num_segments = batch_size * n_post_neurons
        # # Calculate the interarea currents using unsorted_segment_sum
        # i_interarea_flat = tf.math.unsorted_segment_sum(
        #     new_weights,
        #     segment_ids,
        #     num_segments=num_segments
        # )  # Shape: [num_segments]
        # # Cast to the compute dtype if necessary
        # if i_interarea_flat.dtype != self.compute_dtype:
        #     i_interarea_flat = tf.cast(i_interarea_flat, dtype=self.compute_dtype)

        # return i_interarea_flat
       
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

    def call(self, inputs, state, constants=None):
        # Get all the model inputs
        # lgn_spikes, bkg_noise, interarea_z_bufs, state_input = inputs
        lgn_spikes, interarea_z_bufs, state_input = inputs
        # batch_size = tf.shape(lgn_spikes)[0]
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
            i_rec = self.calculate_i_rec_with_custom_grad(rec_z_buf)
            # i_rec = self.calculate_i_rec(rec_z_buf)
        else:
            i_rec = tf.zeros((self._batch_size * self._n_neurons * self._n_receptors), dtype=self.compute_dtype)

        # ### Calculate the interarea input current ###
        if self._connected_areas and self.interarea_indices is not None:
            interarea_z_bufs = interarea_z_bufs[0]
            dampened_interarea_z_bufs = interarea_z_bufs * self._recurrent_dampening  # dampened version of z_buf # no entiendo muy bien la utilidad de esto
            # Now we use tf.stop_gradient to prevent the term (z_buf - dampened_z_buf) to be trained
            interarea_z_buf = (tf.stop_gradient(interarea_z_bufs - dampened_interarea_z_bufs) + dampened_interarea_z_bufs)  
            i_interarea = tf.zeros((self._batch_size * self._n_receptors * self._n_neurons), dtype=self.compute_dtype)
            for column_order in [self.source_column_order]:
                i_interarea += self.calculate_i_inter_with_custom_grad(interarea_z_buf, column_order)
                # i_interarea += self.calculate_i_inter(interarea_z_buf, column_order)
        else:
            i_interarea = tf.zeros((self._batch_size * self._n_neurons * self._n_receptors), dtype=self.compute_dtype)

        # Background noise
        if self._connected_noise:
            bkg_input = tf.random.poisson(shape=(self._batch_size, self.bkg_input_dense_shape[1]), 
                                    lam=self._bkg_firing_rate*.001, 
                                    dtype=self.variable_dtype) # this implementation is slower
            i_noise = self.calculate_noise_current(bkg_input)
        else:
            i_noise = tf.zeros((self._batch_size, self._n_neurons * self._n_receptors), dtype=self.compute_dtype)

        # Add all the current sources
        if self.input_indices is not None: # only V1 area can receive external input
            if self._current_input:
                external_current = self.calculate_input_current_from_firing_probabilities(lgn_spikes)
            else:
                external_current = self.calculate_input_current_from_spikes(lgn_spikes)
        else:
            external_current = tf.zeros((self._batch_size * self._n_neurons * self._n_receptors), dtype=self.compute_dtype)

        rec_inputs = i_rec + i_interarea + external_current + i_noise
        # Reshape i_rec_flat back to [batch_size, num_neurons]
        rec_inputs = tf.reshape(rec_inputs, [self._batch_size, self._n_neurons * self._n_receptors])
        rec_inputs = rec_inputs * self._lr_scale
        if constants is not None and not self._spike_gradient:
            rec_inputs = rec_inputs + state_input * self._lr_scale #(1, n_neurons, _n_receptors)

        # Calculate the new psc variables
        new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)
        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)  # =max(r + prev_z * self.t_ref - self._dt, 0)
        # Calculate the ASC
        asc = tf.reshape(asc, (self._batch_size, self._n_neurons, 2))
        new_asc = self.exp_dt_k * asc + tf.expand_dims(prev_z, axis=-1) * self.asc_amps
        new_asc = tf.reshape(new_asc, (self._batch_size, self._n_neurons * 2))
        # Calculate the postsynaptic current 
        input_current = tf.reshape(psc, (self._batch_size, self._n_neurons, self._n_receptors))
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
            new_v = tf.where(tf.greater(new_r, 0.0), self.v_reset, new_v)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period

        # Generate the network spikes
        v_sc = (new_v - self.v_th) / self.normalizer
        if self._pseudo_gauss:
            if self.compute_dtype == tf.bfloat16:
                new_z = spike_gauss_b16(v_sc, self._gauss_std, self._dampening_factor)
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
        new_z = tf.where(tf.greater(new_r, 0.0), tf.zeros_like(new_z), new_z)
        # Add current spikes to the buffer
        new_z_buf = tf.concat([new_z, z_buf[:, :-self._n_neurons]], axis=1)  # Shift buffer

        outputs = (
            new_z, 
            new_v #* self.voltage_scale + self.voltage_offset,
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
                 train_noise,
                 current_input):
        
        super().__init__()

        self._n_areas = len(networks)  # 2
        self._batch_size  = batch_size  # Batch size

        # Initialize the V1 column
        self.v1 = BillehColumn(
            networks['v1'], lgn_inputs['v1'], bkg_inputs['v1'],
            gauss_std=gauss_std, dampening_factor=dampening_factor, recurrent_dampening_factor=recurrent_dampening_factor,
            input_weight_scale=input_weight_scale, interarea_weight_scale=interarea_weight_scale,
            lr_scale=lr_scale, max_delay=max_delay, batch_size=batch_size,
            pseudo_gauss=pseudo_gauss, spike_gradient=spike_gradient, train_recurrent=train_recurrent_v1, 
            train_input=train_input, train_interarea=train_interarea_v1_lm, train_noise=train_noise, 
            name='v1', hard_reset=hard_reset, connected_areas=connected_areas,
            connected_recurrent_connections=connected_recurrent_connections, connected_noise=connected_noise, current_input=current_input)

        # Initialize the LM column
        # if networks['lm']['n_nodes'] > 0:
        self.lm = BillehColumn(
            networks['lm'], lgn_inputs.get('lm', None), bkg_inputs['lm'],
            gauss_std=gauss_std, dampening_factor=dampening_factor, recurrent_dampening_factor=recurrent_dampening_factor,
            input_weight_scale=input_weight_scale, interarea_weight_scale=interarea_weight_scale,
            lr_scale=lr_scale, max_delay=max_delay, batch_size=batch_size,
            pseudo_gauss=pseudo_gauss, spike_gradient=spike_gradient, train_recurrent=train_recurrent_lm, 
            train_input=train_input, train_interarea=train_interarea_lm_v1, train_noise=train_noise, 
            name='lm', hard_reset=hard_reset, connected_areas=connected_areas,
            connected_recurrent_connections=connected_recurrent_connections, connected_noise=connected_noise, current_input=current_input)
        # Number of neurons in each area
        self._n_neurons = self.v1._n_neurons + self.lm._n_neurons
        # Number of elements in one area's state
        self.num_elements_in_one_state = int(len(self.state_size) / self._n_areas)

    @property
    def state_size(self):
        # Define the state size of the network
        state_size = (
            self.v1._n_neurons * self.v1.max_delay,   # z buffer
            self.v1._n_neurons,                        # v
            self.v1._n_neurons,                        # r
            self.v1._n_neurons * 2,                    # asc 
            self.v1._n_receptors * self.v1._n_neurons, # psc rise
            self.v1._n_receptors * self.v1._n_neurons, # psc
            self.lm._n_neurons * self.lm.max_delay,    # z buffer
            self.lm._n_neurons,                        # v
            self.lm._n_neurons,                        # r
            self.lm._n_neurons * 2,                    # asc
            self.lm._n_receptors * self.lm._n_neurons, # psc rise
            self.lm._n_receptors * self.lm._n_neurons, # psc
        )
        
        return state_size

    def zero_state_multi_areas(self, batch_size, dtype=tf.float32):
        # Combine zero states from both areas
        multi_zero_state = self.v1.zero_state(batch_size, dtype) + self.lm.zero_state(batch_size, dtype)

        return multi_zero_state

    def call(self, inputs, state, constants=None):   
        # The inputs are concatenated: [x, bkg_inputs, state_input]
        # We need to split the inputs accordingly for V1 and LM
        # Split the inputs for V1 and LM
        lgn_input = inputs[:, :self.v1.input_dim]
        # bkg_input = inputs[:, self.v1.input_dim:-self._n_neurons]
        state_input = inputs[:, -self._n_neurons:]

        # Extract the states for V1 and LM
        v1_state = state[:self.num_elements_in_one_state]
        lm_state = state[self.num_elements_in_one_state:]
        # Prepare inter-area spike buffers
        interarea_z_bufs_to_v1 = (lm_state[0],)
        interarea_z_bufs_to_lm = (v1_state[0],)
        # Call the BillehColumn for V1
        # inputs_to_v1 = (lgn_input, bkg_input[:,:self.v1._n_neurons*self.v1._n_receptors], interarea_z_bufs_to_v1, state_input[:,:self.v1._n_neurons])
        inputs_to_v1 = (lgn_input, interarea_z_bufs_to_v1, state_input[:,:self.v1._n_neurons])
        outputs_v1, new_v1_state = self.v1(inputs_to_v1, v1_state)
        # Call the BillehColumn for LM
        # inputs_to_lm = (None, bkg_input[:,self.v1._n_neurons*self.v1._n_receptors:], interarea_z_bufs_to_lm, state_input[:,self.v1._n_neurons:])
        inputs_to_lm = (None, interarea_z_bufs_to_lm, state_input[:,self.v1._n_neurons:])

        outputs_lm, new_lm_state = self.lm(inputs_to_lm, lm_state)
        # Outputs can be concatenated or combined as needed
        outputs = outputs_v1 + outputs_lm
        # Combine new states
        new_state = new_v1_state + new_lm_state

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
                current_input=False
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
        batch_size = tf.shape(x)[0] # if batch_size is None infer it from the input shape
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
                            train_noise=train_noise,
                            current_input=current_input)
        
    zero_state = cell.zero_state_multi_areas(batch_size, dtype)

    if use_state_input:
        initial_state_holder = tf.nest.map_structure(lambda _x: tf.keras.layers.Input(shape=_x.shape[1:], dtype=_x.dtype), zero_state) # get the shape of zero_state without the batch_size # (None, 5000)
        rnn_initial_state = tf.nest.map_structure(tf.identity, initial_state_holder) # copy of initial_state_holder (None, 5000)
        constants = tf.zeros_like(rnn_initial_state[0][:, 0], dtype) # (None,)
    else:
        rnn_initial_state = zero_state
        constants = tf.zeros((batch_size,))

    # # Calculate the background noise input
    # bkg_inputs = BackgroundNoiseLayer(cell, 
    #                                   batch_size=batch_size,
    #                                 #   seq_len=seq_len,
    #                                   bkg_firing_rate=250, 
    #                                   n_bkg_units=100,
    #                                 #   n_bkg_connections=4
    #                                 )(x) # (None, 2500, 4000+600)

    # Concatenate all the inputs together    
    # full_inputs = tf.concat((rnn_inputs, bkg_inputs, state_input), -1) # (None, 2500, 120(LGN)+200(BKG)+50(dummy_zeros))
    # full_inputs = tf.concat((tf.cast(x, dtype), bkg_inputs, state_input), -1) # (None, 2500, 120(LGN)+200(BKG)+50(dummy_zeros))
    full_inputs = tf.concat((tf.cast(x, dtype), state_input), -1)

    # Create the RNN layer
    rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=return_state, name='rsnn') # return_sequences=True return the full sequence. 
    # Run the model over the inputs
    rsnn_out = rnn(full_inputs, initial_state=rnn_initial_state, constants=constants)
    
    # Extract the model output and its new state
    if return_state:
        hidden = rsnn_out[0] #4
        # new_state = out[1:] #(14)
    else:
        hidden = rsnn_out

    # calculate the model outputs
    spikes = dict()
    # voltages = dict()
    outputs = dict()
    # spikes['v1'], voltages['v1'], spikes['lm'], voltages['ml'] = hidden  ## these are results in all time steps? Yes, if return_sequences=True
    spikes['v1'], _, spikes['lm'], _ = hidden  ## these are results in all time steps? Yes, if return_sequences=True

    rate_v1 = tf.reduce_mean(spikes['v1'])
    rate_lm = tf.reduce_mean(spikes['lm'])
    # scale = (1 + tf.nn.softplus(tf.keras.layers.Dense(1)(tf.zeros((1, 1))))) 
    for area in ['v1', 'lm']:
        if neuron_output:
            output_spikes = 1 / dampening_factor * spikes[area] + \
                (1 - 1 / dampening_factor) * tf.stop_gradient(spikes[area])
            # scale = (1 + tf.nn.softplus(tf.keras.layers.Dense(1)(tf.zeros((1, 1))))) #scale parameter (which is trainable)
            scale = 1 + tf.nn.softplus(tf.Variable(0.0, trainable=True, dtype=tf.float32))
            outputs_10 = []
            for i in range(n_output):
                t_output = tf.gather(output_spikes, networks[area][f'readout_neuron_ids_{i}'], axis=2)
                t_output = tf.cast(t_output, tf.float32)
                t_output = tf.reduce_mean(t_output, -1)
                outputs_10.append(t_output)
                # thresh = tf.keras.layers.Dense(1)(tf.zeros_like(output))
                # output = tf.stack([thresh[..., 0], output[..., -1]], -1) * scale
            output = tf.concat(outputs_10, -1) * scale
        else:
            out_pop_spikes = spikes[area]
            # output = tf.keras.layers.Dense(n_output, name=f'projection_{area}', trainable=True)(out_pop_spikes)
            output = tf.keras.layers.Dense(n_output, name=f'projection_{area}', trainable=False)(out_pop_spikes)
            
        
        mean_output = tf.reshape(output, (-1, int(seq_len / down_sample), down_sample, n_output)) #TensorShape([None, 4, 50, 10])
        mean_output = tf.reduce_mean(mean_output, axis=2) # TensorShape([None, 4, 10])
        mean_output = tf.nn.softmax(mean_output, axis=-1) # TensorShape([None, 4, 10])
        outputs[area] = mean_output

    many_input_model = tf.keras.Model(
            inputs=[x, state_input_holder, initial_state_holder], outputs=[rsnn_out, outputs['v1']])

    # # Finally, a prediction layer is created
    # output = tf.keras.layers.Lambda(lambda _a: _a, name='prediction')(output)
        
    # abstract_output = tf.reduce_mean(tf.gather(spikes_lm[
    #                 :,#batch dim
    #                 output_abstract_valid_from_time:,# time dim
    #                 :#neuron dim
    #                 ], networks['lm']['laminar_indices']['L5_e'], axis=2), axis=1)
    # abstract_output = tf.keras.layers.Lambda(lambda _a: _a, name='abstract_output')(abstract_output)

    # completed_output = tf.reduce_mean(tf.gather(spikes_v1[
    #                 :,#batch dim
    #                 output_completed_valid_from_time:,# time dim
    #                 :#neuron dim
    #                 ], networks['v1']['laminar_indices']['L5_e'], axis=2), axis=1)
    # completed_output = tf.keras.layers.Lambda(lambda _a: _a, name='completed_output')(completed_output)

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

    # if use_state_input:
    #     many_input_model = tf.keras.Model(
    #         inputs=[x, state_input_holder, initial_state_holder], outputs=[abstract_output, completed_output, bkg_inputs])
    #     # many_input_model = tf.keras.Model(
    #     #     inputs=[x, state_input_holder, initial_state_holder], 
    #     #     outputs=mean_output
    #     # )
    # else:
    #     # many_input_model = tf.keras.Model(
    #     #     inputs=[x, state_input_holder], 
    #     #     outputs=mean_output
    #     # )
    #     many_input_model = tf.keras.Model(
    #         inputs=[x, state_input_holder], outputs=[abstract_output, completed_output])
    
    # many_input_model = tf.keras.Model(inputs=[x, state_input_holder], outputs=[abstract_output, completed_output])

    # Add metrics to the model if specified.
    if add_rate_metric:
        many_input_model.add_metric(rate_v1, name='v1_area_rate')
        many_input_model.add_metric(rate_lm, name='lm_area_rate')

    return many_input_model

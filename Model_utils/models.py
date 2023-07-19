import numpy as np
import tensorflow as tf
import psutil


def gauss_pseudo(v_scaled, sigma, amplitude):
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude

def pseudo_derivative(v_scaled, dampening_factor):
    return dampening_factor * tf.maximum(1 - tf.abs(v_scaled), 0)

# def slayer_pseudo(v_scaled, sigma, amplitude):
#     return tf.math.exp(-sigma * tf.abs(v_scaled)) * amplitude


# @tf.custom_gradient
# def spike_gauss_16(v_scaled, sigma, amplitude):
#     z_ = tf.greater(v_scaled, 0.)
#     z_ = tf.cast(z_, tf.float16)

#     def grad(dy):
#         de_dz = dy
#         dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)

#         de_dv_scaled = de_dz * dz_dv_scaled

#         return [de_dv_scaled,
#                 tf.zeros_like(sigma), tf.zeros_like(amplitude)]

#     return tf.identity(z_, name='spike_gauss'), grad


@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, tf.float32)

    def grad(dy):
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled,
                tf.zeros_like(sigma), tf.zeros_like(amplitude)]

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

        return [de_dv_scaled,
                tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="spike_function"), grad


# @tf.custom_gradient
# def spike_function_16(v_scaled, dampening_factor):
#     z_ = tf.greater(v_scaled, 0.)
#     z_ = tf.cast(z_, tf.float16)

#     def grad(dy):
#         de_dz = dy
#         dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

#         de_dv_scaled = de_dz * dz_dv_scaled

#         return [de_dv_scaled,
#                 tf.zeros_like(dampening_factor)]

#     return tf.identity(z_, name="spike_function"), grad


# @tf.custom_gradient
# def spike_function_b16(v_scaled, dampening_factor):
#     z_ = tf.greater(v_scaled, 0.)
#     z_ = tf.cast(z_, tf.bfloat16)

#     def grad(dy):
#         de_dz = dy
#         dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)

#         de_dv_scaled = de_dz * dz_dv_scaled

#         return [de_dv_scaled,
#                 tf.zeros_like(dampening_factor)]

#     return tf.identity(z_, name="spike_function"), grad


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


class SpikeInputLayer(tf.keras.layers.Layer):
    def __init__(self, indices, weights, dense_shape, lr_scale=1., dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self._indices = indices
        self._weights = weights
        self._dense_shape = dense_shape
        self._max_batch = int(2**31 / weights.shape[0])
        print('The maximum batch size is: ', self._max_batch)
        self._dtype = dtype
        self._lr_scale = lr_scale
        self._sparse_w_in = tf.sparse.SparseTensor(
            indices, tf.cast(weights, self._dtype), dense_shape
        )

    # def call(self, inp): ## what is going on here? calculate input current  
    #     tf_shp = tf.unstack(tf.shape(inp))
    #     shp = inp.shape.as_list()  # what do the inp's dimensions represent? (batch, sequence, n_input_dim)
    #     for i, a in enumerate(shp):
    #         if a is None:
    #             shp[i] = tf_shp[i]

    #     sparse_w_in = tf.sparse.SparseTensor(
    #         self._indices, self._weights, self._dense_shape)
    #     inp = tf.reshape(inp, (shp[0] * shp[1], shp[2]))
    #     tf.print(shp[0] * shp[1])
    #     # if shp[0] * shp[1] < self._max_batch:  # ?? when it can exceed the threshold? I found it is fixed.
    #     if True: #shp[0] * shp[1] < self._max_batch:  # ?? when it can exceed the threshold? I found it is fixed.
    #         input_current = tf.sparse.sparse_dense_matmul(
    #             sparse_w_in, tf.cast(inp, tf.float32), adjoint_b=True)
    #         input_current = tf.transpose(input_current)
    #         input_current = tf.cast(input_current, self._dtype)
    #     else:
    #         _batch_size = shp[0] * shp[1]
    #         _n_iter = tf.ones((), tf.int32)

    #         # while _batch_size > self._max_batch:
    #         for _i in tf.range(int(np.log(self._max_batch) * 2)):
    #             _sel = tf.greater(_batch_size, self._max_batch)
    #             _batch_size = tf.where(_sel, tf.cast(
    #                 _batch_size / 2, tf.int32), _batch_size)
    #             _n_iter = tf.where(_sel, _n_iter + 1, _n_iter)

    #         results = tf.TensorArray(self._dtype, size=_n_iter)
    #         for _i in tf.range(_n_iter):
    #             partial_input_current = tf.sparse.sparse_dense_matmul(
    #                 sparse_w_in, tf.cast(inp[_i * _batch_size:(_i + 1) * _batch_size], tf.float32), adjoint_b=True)
    #             partial_input_current = tf.cast(
    #                 tf.transpose(partial_input_current), self._dtype)
    #             results = results.write(_i, partial_input_current)
    #             _i += 1
    #         input_current = results.stack()

    #     input_current = tf.reshape(
    #         input_current, (shp[0], shp[1], -1))
    #     return input_current

    def call(self, inp):
        # replace any None values in the shape of inp with the actual values obtained
        # from the input tensor at runtime (tf.shape(inp)).
        # This is necessary because the SparseTensor multiplication operation requires
        # a fully defined shape.
        inp_shape = inp.get_shape().as_list()
        shp = [dim if dim is not None else tf.shape(inp)[i] for i, dim in enumerate(inp_shape)]

        # cast the weights to dtype
        # self._weights = tf.cast(self._weights, self._dtype)
        # sparse_w_in = tf.sparse.SparseTensor(
        #     self._indices, self._weights, self._dense_shape
        # )  # (923696, 17400)
        inp = tf.cast(inp, self._compute_dtype)
        inp = tf.reshape(inp, (shp[0] * shp[1], shp[2])) # (batch_size*sequence_length, input_dim)
        # By setting self._max_batch to this value, the code ensures that the input
        # tensor is processed in smaller batches when its shape exceeds the maximum
        # number of elements in a tensor.
        # the sparse tensor multiplication can be directly performed

        # check if the code is running in a GPU environment and if the input tensor is smaller than the max_batch size
        if not(tf.test.is_gpu_available()) and shp[0] * shp[1] < self._max_batch:
            tf.print('Processing input tensor in one go.')
            input_current = tf.sparse.sparse_dense_matmul(
                self._sparse_w_in,
                inp,
                adjoint_b=True
            )
            input_current = tf.transpose(input_current)
        else:
            tf.print('Chunking input tensor into smaller batches.')
            num_chunks = tf.cast(tf.math.ceil(
                tf.shape(inp)[0] / self._max_batch), tf.int32)
            num_pad_elements = num_chunks * self._max_batch - tf.shape(inp)[0]
            padded_input = tf.pad(inp, [(0, num_pad_elements), (0, 0)])
            # Initialize a tensor array to hold the partial results
            result_array = tf.TensorArray(
                dtype=self._compute_dtype, size=num_chunks)
            for i in tf.range(num_chunks):
                tf.print('Processing chunk:', i, 'of', num_chunks, '.')
                if tf.config.list_physical_devices('GPU'):
                    print(tf.config.experimental.get_memory_usage('GPU:0'))
                # print the memory consumption at this point
                process = psutil.Process()
                mem = process.memory_info().rss / (1024**3)  # in GB
                tf.print("Memory consumption in GB:", round(mem, 2))

                start_idx = i * self._max_batch
                end_idx = (i + 1) * self._max_batch
                chunk = padded_input[start_idx:end_idx]
                chunk = tf.cast(chunk, self._compute_dtype)
                partial_input_current = tf.sparse.sparse_dense_matmul(
                    self._sparse_w_in, chunk, adjoint_b=True)
                partial_input_current = tf.transpose(partial_input_current)
                # Store the partial result in the tensor array
                # print the shape of result_array and partial_input_current
                result_array = result_array.write(i, partial_input_current)

            # Concatenate the partial results to get the final result
            result_array = result_array.concat()[:-num_pad_elements, :]

            input_current = tf.cast(result_array, self._compute_dtype)

        input_current = tf.reshape(input_current, (shp[0], shp[1], -1))
        return input_current


class BackgroundNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, cell, lr_scale=1., dtype=tf.float32, **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype
        self._bkg_weights = dict(V1=0,LM=0)
        self._bkg_weights['V1'] = cell.V1.bkg_weights #(4000,)
        self._bkg_weights['LM'] = cell.LM.bkg_weights #(600,)
        self._lr_scale = lr_scale

    def call(self, inp): # inp only provides the shape   
        tf_shp = tf.unstack(tf.shape(inp))
        shp = inp.shape.as_list() # what do the inp's dimensions represent? (batch, sequence, n_input_dim)
        for i, a in enumerate(shp):
            if a is None:
                shp[i] = tf_shp[i]

        inp_shape = inp.get_shape().as_list()
        shp = [dim if dim is not None else tf.shape(inp)[i] for i, dim in enumerate(inp_shape)]

        rest_of_brain = tf.reduce_sum(tf.cast(
                tf.random.uniform((shp[0], shp[1], 10)) < .1, self._compute_dtype), -1) # (1, 2500)

        noise_inputs = dict()
        for area in ['V1','LM']:
            noise_inputs[area] = tf.cast(
                self._bkg_weights[area][None, None], self._compute_dtype) * rest_of_brain[..., None] / 10.         
        cat_noise_inputs = tf.concat([noise_inputs['V1'], noise_inputs['LM']], axis=-1)

        return cat_noise_inputs


class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        self._positive = positive

    def __call__(self, w):
        sign_corrected_w = tf.where(
            self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
        return sign_corrected_w


# class SparseSignedConstraint(tf.keras.constraints.Constraint):
#     def __init__(self, mask, positive):
#         self._mask = mask
#         self._positive = positive

#     def __call__(self, w):
#         sign_corrected_w = tf.where(
#             self._positive, tf.nn.relu(w), -tf.nn.relu(-w))
#         return tf.where(self._mask, sign_corrected_w, tf.zeros_like(sign_corrected_w))


# class StiffRegularizer(tf.keras.regularizers.Regularizer):
#     def __init__(self, strength, initial_value):
#         super().__init__()
#         self._strength = strength
#         self._initial_value = tf.Variable(initial_value, trainable=False)

#     def __call__(self, x):
#         return self._strength * tf.reduce_sum(tf.square(x - self._initial_value))


class BillehColumn(tf.keras.layers.Layer):
    def __init__(self, network, input_population, bkg_weights,
                 dt=1., gauss_std=.5, dampening_factor=.3,
                 input_weight_scale=1., recurrent_weight_scale=1., interarea_weight_scale=1.,
                 lr_scale=1., max_delay=5, batch_size=12, pseudo_gauss=False, spike_gradient=False,
                 train_recurrent=True, train_input=True, train_interarea=True, hard_reset=False, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        
        self._params = network['node_params']
        self._laminar_indices = network['laminar_indices']
        
        # Rescale the voltages to have them near 0, as we wanted the effective step size 
        # for the weights to be normalized when learning (weights are scaled similarly)
        voltage_scale = self._params['V_th'] - self._params['E_L']
        voltage_offset = self._params['E_L']
        self._params['V_th'] = (self._params['V_th'] -
                                voltage_offset) / voltage_scale
        self._params['E_L'] = (self._params['E_L'] -
                               voltage_offset) / voltage_scale
        self._params['V_reset'] = (
            self._params['V_reset'] - voltage_offset) / voltage_scale
        self._params['asc_amps'] = self._params['asc_amps'] / \
            voltage_scale[..., None]   # _params['asc_amps'] has shape (111, 2)

        self._node_type_ids = network['node_type_ids']
        self._dt = dt
        self._pseudo_gauss = pseudo_gauss

        self._lr_scale = lr_scale
        
        self._spike_gradient = spike_gradient
        self._hard_reset = hard_reset

        n_receptors = network['node_params']['tau_syn'].shape[1] # we have 4 receptor compartments (soma, dendrites, etc) for each neuron
        self._n_receptors = n_receptors
        self._n_neurons = network['n_nodes']
        self._dampening_factor = tf.cast(dampening_factor, self._compute_dtype)
        self._gauss_std = tf.cast(gauss_std, self._compute_dtype)

        tau = self._params['C_m'] / self._params['g'] # determine the time decay constant
        self._decay = np.exp(-dt / tau)
        self._current_factor = 1 / \
            self._params['C_m'] * (1 - self._decay) * tau
        self._syn_decay = np.exp(-dt / np.array(self._params['tau_syn']))
        self._psc_initial = np.e / np.array(self._params['tau_syn'])

        # synapses: target_ids, source_ids, weights, delays
        # this are the axonal delays
        self.max_delay = int(
            np.round(np.min([np.max(network['synapses']['delays']), max_delay])))
        # self.max_delay = max_delay
        
        self.state_size = (
            self._n_neurons * self.max_delay,                # z buffer
            self._n_neurons,                                 # v
            self._n_neurons,                                 # r
            self._n_neurons,                                 # asc 1
            self._n_neurons,                                 # asc 2
            self._n_neurons * n_receptors,                   # psc rise
            self._n_neurons * n_receptors,                   # psc
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
        self.v_reset = _f(self._params['V_reset'])
        self.syn_decay = _f(self._syn_decay)
        self.psc_initial = _f(self._psc_initial)
        self.t_ref = _f(self._params['t_ref'])
        self.asc_amps = _f(self._params['asc_amps'], trainable=False)
        _k = self._params['k']
        self.param_k, self.param_k_read = custom_val(_k, trainable=False) # ?? what is this doing?
        self.v_th = _f(self._params['V_th'])
        self.e_l = _f(self._params['E_L'])
        self.param_g = _f(self._params['g'])
        self.decay = _f(self._decay)
        self.current_factor = _f(self._current_factor)
        self.voltage_scale = _f(voltage_scale)
        self.voltage_offset = _f(voltage_offset)
        self.recurrent_weights = None

        indices, weights, dense_shape = \
            network['synapses']['indices'], network['synapses']['weights'], network['synapses']['dense_shape']
        weights = weights / \
            voltage_scale[self._node_type_ids[indices[:, 0] // #indices[:,0] target, [:,1] source
                                             self._n_receptors]]  # scale down the weights
        delays = np.round(np.clip(
            network['synapses']['delays'], dt, self.max_delay) / dt).astype(np.int32) # (1, 2, 3, 4) possible delays
        dense_shape = dense_shape[0], self.max_delay * dense_shape[1]  # (target*receptors, source*delays)

        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1) 
        
        weights = weights.astype(np.float32)
        print(f'> {self.name} recurrent connections {len(indices)}')
        
        self.recurrent_weight_positive = tf.Variable(
            weights >= 0., name='recurrent_weights_sign', trainable=False)
        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale, name=self.name+'_sparse_recurrent_weights',
            constraint=SignedConstraint(self.recurrent_weight_positive), # keep the sign of the synapses
            trainable=train_recurrent)
        self.recurrent_indices = tf.Variable(indices, trainable=False)
        self.recurrent_dense_shape = dense_shape
        
        # input external stimulus connection
        # if self.name == 'Low': Particular for Guozhang program
        #     template = np.zeros((self.batch_size, self._n_neurons, self._n_receptors))
        #     template[:, self._laminar_indices['L4e'], 0] = 1 # assume the external current only project to one receptor
        #     receive_neuorn_indices = np.nonzero(template)
        #     self.receive_neuron_indices_tf = np.stack(receive_neuorn_indices,axis=1)
        
        if input_population is not None:
            input_weights = input_population['weights'].astype(np.float32)
            input_indices = input_population['indices']
            input_weights = input_weights / \
                voltage_scale[self._node_type_ids[input_indices[:, 0] // self._n_receptors]]
            print(f'> {self.name} input connections {len(input_indices)}')
            input_dense_shape = (self._n_receptors * self._n_neurons, input_population['n_inputs'])
            self.input_weight_positive = tf.Variable(
                input_weights >= 0., name=self.name+'_input_weights_sign', trainable=False)
            self.input_weight_values = tf.Variable(
                input_weights * input_weight_scale / lr_scale, name=self.name+'_sparse_input_weights',
                constraint=SignedConstraint(self.input_weight_positive),
                trainable=train_input)
            self.input_indices = tf.Variable(input_indices, trainable=False)
            self.input_dense_shape = input_dense_shape
        else:
            self.input_indices, self.input_weight_values, self.input_dense_shape = None, None, None
            print(f'> {self.name} input synapses 0')
        
        # background noise connection
        bkg_weights = bkg_weights / \
            np.repeat(voltage_scale[self._node_type_ids], self._n_receptors)
        self.bkg_weights = tf.Variable(
            bkg_weights * 10., name=self.name+'_rest_of_brain_weights', trainable=train_input)

        # inter-area connections
        def set_interarea_connections(source_column_order):
            interarea_indices, interarea_weights, interarea_dense_shape = \
                network['interarea_synapses'][source_column_order]['indices'], network['interarea_synapses'][source_column_order]['weights'], network['interarea_synapses'][source_column_order]['dense_shape']
            if interarea_indices is not None:
                _n_neurons_source_column = interarea_dense_shape[1]
                interarea_weights = interarea_weights / voltage_scale[self._node_type_ids[interarea_indices[:, 0] // self._n_receptors]] # indices[:,0] target, [:,1] source
                interarea_delays = np.round(np.clip(network['interarea_synapses'][source_column_order]['delays'], dt, self.max_delay) / dt).astype(np.int32)
                interarea_dense_shape = interarea_dense_shape[0], self.max_delay * _n_neurons_source_column 
                interarea_indices[:, 1] = interarea_indices[:, 1] + _n_neurons_source_column * (interarea_delays - 1) # here, the _n_neurons should be the #neurons in source column 
                interarea_weights = interarea_weights.astype(np.float32)
                print(f'> {source_column_order} to {self.name} interarea synapses {len(interarea_indices)}')
                
                interarea_weight_positive = tf.Variable(
                    interarea_weights >= 0., name=self.name + '_interarea_weights_sign_'+source_column_order, trainable=False)
                interarea_weight_values = tf.Variable(
                    interarea_weights * interarea_weight_scale / lr_scale, name=self.name+'_sparse_interarea_weights_'+source_column_order,
                    constraint=SignedConstraint(interarea_weight_positive),
                    trainable=train_interarea)
                interarea_indices = tf.Variable(interarea_indices, trainable=False)  

                # check legal indices
                max_tgt_ind , max_src_ind = interarea_indices.numpy().max(axis=0)            
                assert  max_tgt_ind <= interarea_dense_shape[0], 'wrong inter-area indices from target!'
                assert  max_src_ind <= interarea_dense_shape[1], 'wrong inter-area indices from source!'
            else:
                interarea_weight_values, interarea_dense_shape, interarea_indices = None, None, None

            return interarea_weight_values, interarea_dense_shape, interarea_indices
        
        if self.name == 'V1':
            interarea_weight_values, interarea_dense_shape, interarea_indices = set_interarea_connections('LM')
        
        elif self.name == 'LM':
            interarea_weight_values, interarea_dense_shape, interarea_indices = set_interarea_connections('V1')

        else:
            interarea_weight_values, interarea_dense_shape, interarea_indices = None, None, None
        
        # interarea_weight_values_lower, interarea_dense_shape_lower, interarea_indices_lower = set_interarea_connections('lower')
        # interarea_weight_values_higher, interarea_dense_shape_higher, interarea_indices_higher = set_interarea_connections('higher')

        # self.interarea_weight_values = (interarea_weight_values_lower,interarea_weight_values_higher)
        # self.interarea_dense_shapes = (interarea_dense_shape_lower,interarea_dense_shape_higher)
        # self.interarea_indices = (interarea_indices_lower,interarea_indices_higher)
        
        self.interarea_weight_values = (interarea_weight_values,)
        self.interarea_dense_shapes = (interarea_dense_shape,)
        self.interarea_indices = (interarea_indices,)

    def zero_state(self, batch_size, dtype=tf.float32):
        z0_buf = tf.zeros(
            (batch_size, self._n_neurons * self.max_delay), dtype)
        v0 = tf.ones((batch_size, self._n_neurons), dtype) * \
                tf.cast(self.v_th * .0 + 1. * self.v_reset, dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_10 = tf.zeros((batch_size, self._n_neurons), dtype)
        asc_20 = tf.zeros((batch_size, self._n_neurons), dtype)
        psc_rise0 = tf.zeros(
            (batch_size, self._n_neurons * self._n_receptors), dtype)
        psc0 = tf.zeros(
            (batch_size, self._n_neurons * self._n_receptors), dtype)
        return z0_buf, v0, r0, asc_10, asc_20, psc_rise0, psc0

    def _gather(self, prop):
        return tf.gather(prop, self._node_type_ids)
    
    def get_interarea_current(self, interarea_z_bufs, column_order):
        # breakpoint()
        if self.interarea_indices[column_order] is not None:
            sparse_w_interarea = tf.sparse.SparseTensor(
                self.interarea_indices[column_order], self.interarea_weight_values[column_order], self.interarea_dense_shapes[column_order])
            i_interarea = tf.sparse.sparse_dense_matmul(sparse_w_interarea, tf.cast(interarea_z_bufs[column_order], tf.float32), adjoint_b=True) 
            i_interarea = tf.transpose(i_interarea)
            i_interarea = tf.cast(i_interarea, self._compute_dtype) 
        else:
            i_interarea = 0
        return i_interarea

    def call(self, inputs, state, constants=None):
        
        external_current = inputs[0]
        bkg_noise = inputs[1]
        interarea_z_bufs = inputs[2] # interarea_z_bufs[0] relative to lower area, interarea_z_bufs relative to higher area.
        dummy_input = inputs[3]
        
        if self._spike_gradient:
            state_input = tf.zeros((1,))
        else:
            state_input = tf.zeros((4,))
        
        # inter-area current
        source_column_order = [0] # if more areas should connect to the target area, their indices should be included here
        
        i_interarea = 0
        for column_order in source_column_order:
            i_interarea += self.get_interarea_current(interarea_z_bufs,column_order) #(1, 120)
        
        # recurrent current
        z_buf, v, r, asc_1, asc_2, psc_rise, psc = state

        shaped_z_buf = tf.reshape(z_buf, (-1, self.max_delay, self._n_neurons)) #shape (batch, delay, neurons)
        prev_z = shaped_z_buf[:, 0] # previous spikes with shape (neurons)

        psc_rise = tf.reshape(psc_rise, (self.batch_size, self._n_neurons, self._n_receptors))
        psc = tf.reshape(psc, (self.batch_size, self._n_neurons, self._n_receptors))

        sparse_w_rec = tf.sparse.SparseTensor(
            self.recurrent_indices, self.recurrent_weight_values, self.recurrent_dense_shape)

        i_rec = tf.sparse.sparse_dense_matmul(sparse_w_rec, tf.cast(z_buf, tf.float32), adjoint_b=True)
        i_rec = tf.transpose(i_rec)
        i_rec= tf.cast(i_rec, self._compute_dtype) #(batch_size, n_neurons*receptors)
        
        rec_inputs = tf.reshape(
            i_rec + i_interarea + bkg_noise, (self.batch_size, self._n_neurons, self._n_receptors))
        
        if external_current is not None and self.name == 'V1': # only V1 area can receive external input
            # rec_inputs = tf.tensor_scatter_nd_add(tensor=rec_inputs, indices=self.receive_neuron_indices_tf, updates=tf.reshape(external_current,[-1]))
            external_current = tf.reshape(external_current, 
                                          (self.batch_size, self._n_neurons, self._n_receptors))
            rec_inputs = rec_inputs + external_current
            
        rec_inputs = rec_inputs * self._lr_scale
        if constants is not None and not self._spike_gradient:
            rec_inputs = rec_inputs + state_input * self._lr_scale #(1, n_neurons, n_receptors)
                
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise

        new_r = tf.nn.relu(r + prev_z * self.t_ref - self._dt)

        k = self.param_k_read()
        asc_amps = self.asc_amps
        new_asc_1 = tf.exp(-self._dt * k[:, 0]) * \
            asc_1 + prev_z * asc_amps[:, 0]
        new_asc_2 = tf.exp(-self._dt * k[:, 1]) * \
            asc_2 + prev_z * asc_amps[:, 1]

        if constants is not None and self._spike_gradient:
            input_current = tf.reduce_sum(psc, -1) + dummy_input # dummy zeros can help one with getting dE/dV
        else:
            input_current = tf.reduce_sum(psc, -1)

        decayed_v = self.decay * v
        gathered_g = self.param_g * self.e_l
        c1 = input_current + asc_1 + asc_2 + gathered_g

         # Update the voltage according to the LIF equation and the refractory period
        if self._hard_reset: # hard reset : directly changing the membrane voltage to the reset value , if used you can not train the model (not differentiable)
            # Here we keep the voltage at the reset value during the refractory period
            new_v = tf.where(
                new_r > 0.0, self.v_reset, decayed_v + self.current_factor * c1
            )
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period
            # new_v = tf.where(prev_z > 0., self.v_reset, decayed_v + self.current_factor * c1)
        else:
            reset_current = prev_z * (self.v_reset - self.v_th) # soft reset
            new_v = decayed_v + self.current_factor * c1 + reset_current

        normalizer = self.v_th - self.e_l
        v_sc = (new_v - self.v_th) / normalizer

        if self._pseudo_gauss: # what is this for? different types of pseudo-derivative
            new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)
        else:
            new_z = spike_function(v_sc, self._dampening_factor)

        new_z = tf.where(new_r > 0., tf.zeros_like(new_z), new_z)

        new_psc = tf.reshape(new_psc, (self.batch_size, self._n_neurons * self._n_receptors))
        new_psc_rise = tf.reshape(new_psc_rise, (self.batch_size, self._n_neurons * self._n_receptors))
        new_shaped_z_buf = tf.concat((new_z[:, None], shaped_z_buf[:, :-1]), 1)
        new_z_buf = tf.reshape(new_shaped_z_buf, (-1, self._n_neurons * self.max_delay))
        
        outputs = (new_z, new_v * self.voltage_scale + self.voltage_offset)
        new_state = (new_z_buf, new_v, new_r, new_asc_1,
                     new_asc_2, new_psc_rise, new_psc)

        return outputs, new_state


class MultiAreaModel(tf.keras.layers.Layer):
    def __init__(self, networks, input_population, bkg_weights, gauss_std, dampening_factor, 
            input_weight_scale, interarea_weight_scale, lr_scale, batch_size,
            max_delay, pseudo_gauss, spike_gradient, train_recurrent_V1, train_recurrent_LM, train_input, train_interarea, hard_reset):
        super().__init__()

        self._n_areas = len(networks) #2
        self.batch_size  = batch_size #1
        
        self.V1 = BillehColumn(networks['V1'], input_population, bkg_weights['V1'],
                               gauss_std=gauss_std, dampening_factor=dampening_factor, 
                               input_weight_scale=input_weight_scale, interarea_weight_scale=interarea_weight_scale,
                               lr_scale=lr_scale, max_delay=max_delay, batch_size=batch_size,
                               pseudo_gauss=pseudo_gauss, spike_gradient=False, train_recurrent=train_recurrent_V1, 
                               train_input=train_input, train_interarea=train_interarea, name='V1', hard_reset=hard_reset)
        self.LM = BillehColumn(networks['LM'], None, bkg_weights['LM'],
                               gauss_std=gauss_std, dampening_factor=dampening_factor,
                               input_weight_scale=input_weight_scale, interarea_weight_scale=interarea_weight_scale,
                               lr_scale=lr_scale, max_delay=max_delay, batch_size=batch_size,
                               pseudo_gauss=pseudo_gauss, spike_gradient=False, train_recurrent=train_recurrent_LM, 
                               train_input=train_input, train_interarea=train_interarea, name='LM', hard_reset=hard_reset)
        
        self._n_neurons = self.V1._n_neurons + self.LM._n_neurons

        self.state_size = (
            self.V1._n_neurons * self.V1.max_delay,   # z buffer
            self.V1._n_neurons,                        # v
            self.V1._n_neurons,                        # r
            self.V1._n_neurons,                        # asc 1
            self.V1._n_neurons,                        # asc 2
            self.V1._n_receptors * self.V1._n_neurons, # psc rise, double exponential synapses
            self.V1._n_receptors * self.V1._n_neurons, # psc
            self.LM._n_neurons * self.LM.max_delay,   # z buffer
            self.LM._n_neurons,                        # v
            self.LM._n_neurons,                        # r
            self.LM._n_neurons,                        # asc 1
            self.LM._n_neurons,                        # asc 2
            self.LM._n_receptors * self.LM._n_neurons, # psc rise, double exponential synapses
            self.LM._n_receptors * self.LM._n_neurons, # psc
        )
        self.num_elements_in_one_state = int(len(self.state_size)/self._n_areas)
    
    def zero_state_multi_areas(self, batch_size, dtype=tf.float32):
        multi_zero_state =  self.V1.zero_state(batch_size, dtype) + self.LM.zero_state(batch_size, dtype) # add the two states together, one after the other
        return multi_zero_state

    def call(self, inputs, state, constants=None):   
        external_current = inputs[:, :self.V1._n_neurons*self.V1._n_receptors] # first part is real external input
        bkg_input = inputs[:, self.V1._n_neurons*self.V1._n_receptors:-self._n_neurons] # background noise
        dummy_input = inputs[:, -self._n_neurons:] # dummy zeros
        
        V1_state, LM_state = state[:self.num_elements_in_one_state], state[-self.num_elements_in_one_state:]
        # breakpoint()
        interarea_z_bufs_to_V1 = (LM_state[0],)
        inputs_to_V1 = (external_current, bkg_input[:,:self.V1._n_neurons*self.V1._n_receptors], interarea_z_bufs_to_V1, dummy_input[:,:self.V1._n_neurons])
        outputs_V1, new_V1_state = self.V1(inputs_to_V1, V1_state)

        interarea_z_bufs_to_LM = (V1_state[0],)
        inputs_to_LM = (None, bkg_input[:,self.V1._n_neurons*self.V1._n_receptors:], interarea_z_bufs_to_LM, dummy_input[:,self.V1._n_neurons:])
        outputs_LM, new_LM_state = self.LM(inputs_to_LM, LM_state)
        
        outputs = outputs_V1 + outputs_LM
        new_state = new_V1_state + new_LM_state # merge tuple, because nested tuple as state cannot give a legal state_size

        return outputs, new_state

def huber_quantile_loss(u, tau, kappa): # characterize the distance between two distributions
    branch_1 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) / \
        (2 * kappa) * tf.square(u)
    branch_2 = tf.abs(tau - tf.cast(u <= 0, tf.float32)) * \
        (tf.abs(u) - .5 * kappa)
    return tf.where(tf.abs(u) <= kappa, branch_1, branch_2)


def compute_spike_rate_distribution_loss(_spikes, target_rate):
    _rate = tf.reduce_mean(_spikes, (0, 1))
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rate = tf.gather(_rate, rand_ind)
    sorted_rate = tf.sort(_rate) # what are these three lines for? when using sort, the entries with the same value could have a fixed sequence, using shuffle to break them. The results are sensitive to the shuffle op
    u = target_rate - sorted_rate
    tau = (
        tf.cast(tf.range(target_rate.shape[0]), tf.float32) + 1) / target_rate.shape[0]
    loss = huber_quantile_loss(u, tau, .002)

    return loss


class SpikeRateDistributionRegularization:
    def __init__(self, target_rates, rate_cost=.5):
        self._rate_cost = rate_cost
        self._target_rates = target_rates

    def __call__(self, spikes):
        reg_loss = compute_spike_rate_distribution_loss(
            spikes, self._target_rates) * self._rate_cost
        reg_loss = tf.reduce_sum(reg_loss)

        return reg_loss


class VoltageRegularization:
    def __init__(self, cell, voltage_cost=1e-5):
        self._voltage_cost = voltage_cost
        self._cell = cell
        self.voltage_offset = tf.concat([cell.V1.voltage_offset, cell.LM.voltage_offset,],axis=-1)
        self.voltage_scale = tf.concat([cell.V1.voltage_scale, cell.LM.voltage_scale],axis=-1)

    def __call__(self, voltages):
        voltage_32 = (tf.cast(voltages, tf.float32) -
                      self.voltage_offset) / self.voltage_scale
        v_pos = tf.square(tf.nn.relu(voltage_32 - 1.))
        v_neg = tf.square(tf.nn.relu(-voltage_32 + 1.))
        voltage_loss = tf.reduce_mean(tf.reduce_sum(
            v_pos + v_neg, -1)) * self._voltage_cost
        return voltage_loss


def create_model(networks, input_population, bkg_weights, inputIsspike=True, 
                 output_completed_valid_from_time=120, output_abstract_valid_from_time=100,
                 seq_len=200, n_input=10, dtype=tf.float32, input_weight_scale=1., 
                 interarea_weight_scale=1., gauss_std=.5, dampening_factor=.2, lr_scale=800., 
                 train_recurrent_V1=True, train_recurrent_LM=True, train_input=True, train_interarea=True, 
                 use_state_input=True, return_state=True, add_rate_metric=True, max_delay=5, pseudo_gauss=False,
                 neuron_output=True, return_sequences=True, batch_size=None, hard_reset=False):

    # Create the input of the model
    x = tf.keras.layers.Input(shape=(seq_len, n_input,)) # this shape (None, seq_len, n_input) dose not contain batch size, batch_size can be assigned as another argument
    neurons =  networks['V1']['n_nodes'] + networks['LM']['n_nodes']
    dummy_input_holder = tf.keras.layers.Input(shape=(seq_len, neurons))
    dummy_input = tf.cast(tf.identity(dummy_input_holder), dtype) # (None, 2500, 1000+150)
    # The dummy_input is initialized with zeros and its purpose is to provide a way to pass additional input to the model if needed

    # if use_state_input # for rollout the network, may be needed later
    if batch_size is None:
        batch_size = tf.shape(x)[0] # if batch_size is None just update after each timestep (1 ms in our case)
    else:
        batch_size = batch_size
                
    cell = MultiAreaModel(networks, input_population, bkg_weights, 
                            gauss_std=gauss_std, dampening_factor=dampening_factor,
                            input_weight_scale=input_weight_scale, interarea_weight_scale=interarea_weight_scale,
                            lr_scale=lr_scale, spike_gradient=True, batch_size=batch_size, max_delay=max_delay,
                            pseudo_gauss=pseudo_gauss, hard_reset=hard_reset,
                            train_recurrent_V1=train_recurrent_V1, train_recurrent_LM=train_recurrent_LM, train_input=train_input, train_interarea=train_interarea)
        
    zero_state = cell.zero_state_multi_areas(batch_size, dtype)

    if use_state_input:
        initial_state_holder = tf.nest.map_structure(lambda _x: tf.keras.layers.Input(shape=_x.shape[1:]), zero_state) # get the shape of zero_state without the batch_size # (None, 5000)
        rnn_initial_state = tf.nest.map_structure(tf.identity, initial_state_holder) # copy of initial_state_holder (None, 5000)
        constants = tf.zeros_like(rnn_initial_state[0][:, 0], dtype) # (None,)
    else:
        rnn_initial_state = zero_state
        constants = tf.zeros((batch_size,))

    # Calculate the background noise input
    bkg_inputs = BackgroundNoiseLayer(cell, lr_scale=1., dtype=tf.float32)(x) # (None, 2500, 4000+600)
    
    # Calculate the LGN input
    if inputIsspike: # if the data input from LGN to V1 are spikes calculate the currents
        rnn_inputs = SpikeInputLayer(indices=cell.V1.input_indices, weights=cell.V1.input_weight_values, dense_shape=cell.V1.input_dense_shape, 
                                     lr_scale=1., dtype=tf.float32, name='input_layer')(x)
    else:
        rnn_inputs = x # if not, the data is current based which can be added directly

    # Concatenate all the inputs together    
    full_inputs = tf.cast(tf.concat((rnn_inputs, bkg_inputs, dummy_input), -1), dtype) # (None, 2500, 120(LGN)+200(BKG)+50(dummy_zeros))

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
    spikes_V1, voltage_V1, spikes_LM, voltage_LM = hidden  ## these are results in all time steps? Yes, if return_sequences=True

    rate_V1 = tf.cast(tf.reduce_mean(spikes_V1, (1, 2)), tf.float32)
    rate_LM = tf.cast(tf.reduce_mean(spikes_LM, (1, 2)), tf.float32)
    
    

    # if neuron_output:
    #     output_spikes = 1 / dampening_factor * spikes + \
    #         (1 - 1 / dampening_factor) * tf.stop_gradient(spikes)
    #     output = tf.gather(
    #         output_spikes, network['readout_neuron_ids'], axis=2)
    #     output = tf.reduce_mean(output, -1)
    #     scale = (1 + tf.nn.softplus(tf.keras.layers.Dense(1)
    #                                 (tf.zeros_like(output[:1, :1]))))
    #     thresh = tf.keras.layers.Dense(1)(tf.zeros_like(output))
    #     output = tf.stack([thresh[..., 0], output[..., -1]], -1) * scale
    # else:
        # output = tf.keras.layers.Dense(
        #     n_output, name='projection', trainable=True)(spikes)
    
    # n_output = 4

    # output = tf.keras.layers.Dense(
    #             n_output, name='projection', trainable=True)(spikes_V1)

    # output = tf.keras.layers.Lambda(lambda _a: _a, name='prediction')(output)
        

    abstract_output = tf.reduce_mean(tf.gather(spikes_LM[
                    :,#batch dim
                    output_abstract_valid_from_time:,# time dim
                    :#neuron dim
                    ], networks['LM']['laminar_indices']['L5_e'], axis=2), axis=1)
    abstract_output = tf.keras.layers.Lambda(lambda _a: _a, name='abstract_output')(abstract_output)

    completed_output = tf.reduce_mean(tf.gather(spikes_V1[
                    :,#batch dim
                    output_completed_valid_from_time:,# time dim
                    :#neuron dim
                    ], networks['V1']['laminar_indices']['L5_e'], axis=2), axis=1)
    completed_output = tf.keras.layers.Lambda(lambda _a: _a, name='completed_output')(completed_output)

    if use_state_input:
        many_input_model = tf.keras.Model(
            inputs=[x, dummy_input_holder, initial_state_holder], outputs=[abstract_output, completed_output])
    else:
        many_input_model = tf.keras.Model(
            inputs=[x, dummy_input_holder], outputs=[abstract_output, completed_output])
    
    # many_input_model = tf.keras.Model(inputs=[x, dummy_input_holder], outputs=[abstract_output, completed_output])

    # down_sample = 50
    # cue_duration=20
    # if return_sequences:
    #     mean_output = tf.reshape(
    #         output, (-1, int(seq_len / down_sample), down_sample, n_output))
    #     mean_output = tf.reduce_mean(mean_output, 2)
    #     mean_output = tf.nn.softmax(mean_output, axis=-1)
    # else:
    #     mean_output = tf.reduce_mean(output[:, -cue_duration:], 1)
    #     mean_output = tf.nn.softmax(mean_output)
        
        
        
    if add_rate_metric:
        many_input_model.add_metric(rate_V1, name='V1_area_rate')
        many_input_model.add_metric(rate_LM, name='LM_area_rate')

    return many_input_model

import os

# Define the environment variables for optimal GPU performance
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private' # only use this if training with a single gpu
os.environ['TF_GPU_THREAD_MODE'] = 'global'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' # before import tensorflow
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import absl
import numpy as np
import tensorflow as tf
import pickle as pkl
from packaging import version

if version.parse(tf.__version__) < version.parse("2.4.0"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision

from Model_utils import load_sparse, models, other_billeh_utils, stim_dataset, toolkit
import Model_utils.loss_functions as losses
from Model_utils.callbacks import ClassificationCallbacks, printgpu
from Model_utils.optimizers import ExponentiatedAdam

from time import time
import ctypes.util
import random
import logging
tf.get_logger().setLevel(logging.INFO)

print("--- CUDA version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("--- CUDNN version: ", tf.sysconfig.get_build_info()["cudnn_version"])
print("--- TensorFlow version: ", tf.__version__)

# For CUDA Runtime API
lib_path = ctypes.util.find_library('cudart')
print("--- CUDA Library path: ", lib_path)

def main(_):
    # Allow for memory growth (also to observe memory consumption)
    physical_devices = tf.config.list_physical_devices("GPU")
    for dev in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(dev, True)
            print(f"Memory growth enabled for device {dev}")
        except:
            print(f"Invalid device {dev} or cannot modify virtual devices once initialized.")
            pass
    print("- Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), '\n')

    flags = absl.app.flags.FLAGS

    # Set the seeds for reproducibility
    np.random.seed(flags.seed)
    tf.random.set_seed(flags.seed)
    random.seed(flags.seed)

    if flags.realistic_neurons_ratio:
        # Select the connectivity rules in the network
        v1_to_lm_neurons_ratio = 7.010391285652859
        n_neurons = {'v1': flags.v1_neurons,
                     'lm': int(flags.v1_neurons/v1_to_lm_neurons_ratio)}
    else:
        n_neurons = {'v1': flags.v1_neurons,
                     'lm': flags.lm_neurons}

    # Get the neurons of each column of the network
    v1_neurons = n_neurons['v1']
    lm_neurons = n_neurons['lm']

    logdir = flags.ckpt_dir
    if logdir == '':
        flag_str = f'v1_{v1_neurons}_lm_{lm_neurons}'
        for name, value in flags.flag_values_dict().items():
            if value != flags[name].default and name in ['n_input', 'core_only', 'connected_selection', 'interarea_weight_distribution', 'E4_weight_factor', 'random_weights']:
                flag_str += f'_{name}_{value}'
        # Define flag string as the second part of results_path
        results_dir = f'{flags.results_dir}/{flag_str}'
        os.makedirs(results_dir, exist_ok=True)
        print('Simulation results path: ', results_dir)
        # Generate a ticker for the current simulation
        sim_name = toolkit.get_random_identifier('b_')
        logdir = os.path.join(results_dir, sim_name)
        print(f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')
    else:
        flag_str = logdir.split(os.path.sep)[-2]

    # Can be used to try half precision training
    if flags.dtype=='float16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            mixed_precision.set_global_policy('mixed_float16')
        dtype = tf.float16
        print('Mixed precision (float16) enabled!')
    elif flags.dtype=='bfloat16':
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = mixed_precision.Policy("mixed_bfloat16")
            mixed_precision.set_policy(policy)
        else:
            mixed_precision.set_global_policy('mixed_bfloat16')
        dtype = tf.bfloat16
        print('Mixed precision (bfloat16) enabled!')
    else:
        dtype = tf.float32

    # n_workers, n_gpus_per_worker = 1, 1
    # task_name = flags.task_name
    # tasks = ['garrett', 'evidence', 'vcd_grating', 'ori_diff', '10class']

    # model is being run on multiple GPUs or CPUs, and the results are being reduced to a single CPU device. 
    # device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    # strategy = tf.distribute.OneDeviceStrategy(device=device)
    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0"))

    if flags.test_only:
        per_replica_batch_size = 28
        val_steps = int(10000 / per_replica_batch_size)
    else:
        per_replica_batch_size = flags.batch_size
        val_steps = flags.val_steps

    num_replicas = strategy.num_replicas_in_sync
    global_batch_size = per_replica_batch_size * num_replicas
    print(f'Per replica batch size: {per_replica_batch_size}')
    print(f'Global batch size: {global_batch_size}\n')
    print(f'Training with current input: {flags.current_input}')
    print(f'Pseudo derivative gaussian: {flags.pseudo_gauss}')

    # Load data of Billeh et al. (2020) and select appropriate number of neurons and inputs
    # Create the v1-lm model
    t0 = time()
    if flags.caching:
        load_fn = load_sparse.cached_load_billeh
    else:
        load_fn = load_sparse.load_billeh
    networks, lgn_inputs, bkg_inputs = load_fn(flags, n_neurons, flag_str=flag_str)
    print(f"Model files loading: {time()-t0:.2f} seconds\n")

    delays = [int(a) for a in flags.delays.split(',') if a != '']

    # Define the scope in which the model training will be executed
    with strategy.scope():
        t0 = time()
        # Build the model
        model = models.create_model(
            networks, 
            lgn_inputs, 
            bkg_inputs, 
            seq_len=flags.seq_len-delays[0],
            n_input=flags.n_input, 
            dtype=dtype, 
            input_weight_scale=flags.input_weight_scale,
            recurrent_dampening_factor=flags.recurrent_dampening_factor,
            dampening_factor=flags.dampening_factor, 
            gauss_std=flags.gauss_std, 
            lr_scale=flags.lr_scale,
            train_recurrent_v1=flags.train_recurrent_v1, 
            train_recurrent_lm=flags.train_recurrent_lm, 
            train_input=flags.train_input, 
            train_interarea_lm_v1=flags.train_interarea_lm_v1,
            train_interarea_v1_lm=flags.train_interarea_v1_lm,
            train_noise=flags.train_noise,
            batch_size=per_replica_batch_size, 
            pseudo_gauss=flags.pseudo_gauss, 
            use_state_input=True, 
            return_state=True,
            hard_reset=flags.hard_reset,
            connected_recurrent_connections=flags.connected_recurrent_connections,
            connected_areas=flags.connected_areas,
            connected_noise=flags.connected_noise,
            add_rate_metric=False, 
            max_delay=5, 
            n_output=flags.n_output,
            neuron_output=flags.neuron_output,
            # output_completed_valid_from_time=120, 
            # output_abstract_valid_from_time=100,
            current_input=flags.current_input
            )

        model.build((per_replica_batch_size, flags.seq_len-delays[0], flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")

        # Store the initial model variables that are going to be trained
        model_variables_dict = {'Initial': {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}}

        # Define the optimizer
        # optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11, clipnorm=0.001)  
        if flags.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)
        elif flags.optimizer == 'exp_adam':
            optimizer = ExponentiatedAdam(flags.learning_rate, epsilon=1e-11)
        elif flags.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
        else:
            print(f"Invalid optimizer: {flags.optimizer}")
            raise ValueError
          
        # Build the optimizer
        optimizer.build(model.trainable_variables) # the optimizer needs to be built before restoring from the checkpoint

        #Enable loss scaling for training float16 model. This needs to be done before restoring from the checkpoint
        if flags.dtype == 'float16':
            optimizer = mixed_precision.LossScaleOptimizer(optimizer) # to prevent suffering from underflow gradients when using tf.float16

        # Option to resume the training from a checkpoint from a previous training session
        if flags.restore_from != '' and os.path.exists(flags.restore_from):
            checkpoint_directory = tf.train.latest_checkpoint(flags.restore_from)
            print(f'Restoring checkpoint from {checkpoint_directory}...')
            optimizer_continuing = other_billeh_utils.optimizers_match(optimizer, checkpoint_directory)            
            if not optimizer_continuing:
                # Define the optimizer
                if flags.optimizer == 'adam':
                    optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)
                elif flags.optimizer == 'exp_adam':
                    optimizer = ExponentiatedAdam(flags.learning_rate, epsilon=1e-11)
                elif flags.optimizer == 'sgd':
                    optimizer = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
                else:
                    print(f"Invalid optimizer: {flags.optimizer}")
                    raise ValueError
                if flags.dtype == 'float16':
                    optimizer = mixed_precision.LossScaleOptimizer(optimizer) # to prevent suffering from underflow gradients when using tf.float16

                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
            else:
                try:
                    # Restore the model
                    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                    checkpoint.restore(checkpoint_directory).assert_consumed()
                except:
                    print("Failed to restore the optimizer. Starting from scratch...")
                    # Define the optimizer
                    if flags.optimizer == 'adam':
                        optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)
                    elif flags.optimizer == 'exp_adam':
                        optimizer = ExponentiatedAdam(flags.learning_rate, epsilon=1e-11)
                    elif flags.optimizer == 'sgd':
                        optimizer = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
                    else:
                        print(f"Invalid optimizer: {flags.optimizer}")
                        raise ValueError
                    if flags.dtype == 'float16':
                        optimizer = mixed_precision.LossScaleOptimizer(optimizer) # to prevent suffering from underflow gradients when using tf.float16

                    # Restore the model
                    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                    checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()

        # Restore model and optimizer from an intermediate checkpoint if it exists
        elif flags.ckpt_dir != '' and os.path.exists(os.path.join(flags.ckpt_dir, "Intermediate_checkpoints")):
            checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, "Intermediate_checkpoints"))
            print(f'Restoring checkpoint from {checkpoint_directory}...')
            optimizer_continuing = other_billeh_utils.optimizers_match(optimizer, checkpoint_directory)            
            if not optimizer_continuing:
                if flags.optimizer == 'adam':
                    optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)  
                elif flags.optimizer == 'sgd':
                    optimizer = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
                else:
                    print(f"Invalid optimizer: {flags.optimizer}")
                    raise ValueError
                if flags.dtype == 'float16':
                    optimizer = mixed_precision.LossScaleOptimizer(optimizer) # to prevent suffering from underflow gradients when using tf.float16

                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
            else:
                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.restore(checkpoint_directory).assert_consumed()
        else:
            print(f"No checkpoint found in {flags.ckpt_dir} or {flags.restore_from}. Starting from scratch...\n")
            checkpoint = None

        model_variables_dict['Best'] =  {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}
        print(f"Model variables stored in dictionary\n")

        ### BUILD THE LOSS AND REGULARIZER FUNCTIONS ###
        # Create rate and voltage regularizers
        if flags.core_loss and v1_neurons > 51978:
            v1_core_mask = other_billeh_utils.isolate_core_neurons(networks['v1'], n_selected_neurons=51978, data_dir=flags.data_dir)
            v1_core_mask = tf.constant(v1_core_mask, dtype=tf.bool)
            lm_core_mask = other_billeh_utils.isolate_core_neurons(networks['lm'], n_selected_neurons=7414, data_dir=flags.data_dir)
            lm_core_mask = tf.constant(lm_core_mask, dtype=tf.bool)
        else:
            v1_core_mask = None
            lm_core_mask = None

        # Extract outputs of intermediate keras layers to get access to spikes and membrane voltages of the model
        rsnn_layer = model.get_layer('rsnn')
        
        # ### RECURRENT REGULARIZERS ###
        # v1_recurrent_regularizer = losses.IndividualStiffRegularizer(flags.recurrent_weight_regularization, networks['v1'], penalize_relative_change=False, 
        #                                                    initial_values=rsnn_layer.cell.v1.recurrent_weight_values, dtype=tf.float32)
        # model.add_loss(lambda: v1_recurrent_regularizer(rsnn_layer.cell.v1.recurrent_weight_values))
        # lm_recurrent_regularizer = losses.IndividualStiffRegularizer(flags.recurrent_weight_regularization, networks['lm'], penalize_relative_change=False, 
        #                                                    initial_values=rsnn_layer.cell.lm.recurrent_weight_values, dtype=tf.float32)
        # model.add_loss(lambda: lm_recurrent_regularizer(rsnn_layer.cell.lm.recurrent_weight_values))

        # ### INTERAREA REGULARIZERS ###
        # v1_lm_regularizer = losses.IndividualStiffRegularizer(flags.interarea_weight_regularization, networks['v1'], penalize_relative_change=False, recurrent_weights=False, source_area='lm', 
        #                                             initial_values=rsnn_layer.cell.v1.interarea_weight_values['lm'], dtype=tf.float32)
        # model.add_loss(lambda: v1_lm_regularizer(rsnn_layer.cell.v1.interarea_weight_values['lm'])) 
        # lm_v1_regularizer = losses.IndividualStiffRegularizer(flags.interarea_weight_regularization, networks['lm'], penalize_relative_change=False, recurrent_weights=False, source_area='v1', 
        #                                             initial_values=rsnn_layer.cell.lm.interarea_weight_values['v1'], dtype=tf.float32)
        # model.add_loss(lambda: lm_v1_regularizer(rsnn_layer.cell.lm.interarea_weight_values['v1']))

        ### SPONTANEOUS RATES REGULARIZERS ###
        v1_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], natural_images=True, rate_cost=flags.rate_cost, pre_delay=0, post_delay=0, 
                                                                            data_dir=flags.data_dir, area='v1', core_mask=None, seed=flags.seed, dtype=tf.float32)
        model.add_loss(lambda: v1_evoked_rate_regularizer(rsnn_layer.output[0][0]))
        lm_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], natural_images=True, rate_cost=flags.rate_cost, pre_delay=0, post_delay=0, 
                                                                            data_dir=flags.data_dir, area='lm', core_mask=None, seed=flags.seed, dtype=tf.float32)
        model.add_loss(lambda: lm_evoked_rate_regularizer(rsnn_layer.output[0][2]))

        ### VOLTAGE REGULARIZERS ###
        v1_voltage_regularizer = losses.VoltageRegularization(rsnn_layer.cell.v1, area='v1', voltage_cost=flags.voltage_cost, dtype=tf.float32) #, core_mask=v1_core_mask)
        lm_voltage_regularizer = losses.VoltageRegularization(rsnn_layer.cell.lm, area='lm', voltage_cost=flags.voltage_cost, dtype=tf.float32) #, core_mask=lm_core_mask)
        model.add_loss(lambda: v1_voltage_regularizer(rsnn_layer.output[0][1]) + lm_voltage_regularizer(rsnn_layer.output[0][3]))

        ### CLASIFICATION LOSS ###
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(_l, _p, _w):
            per_example_loss = loss_object(_l, _p, sample_weight=_w)
            per_example_loss = per_example_loss / tf.reduce_sum(_w, axis=1, keepdims=True)
            per_example_loss = tf.reduce_sum(per_example_loss, axis=-1) #sum over chunks (0 all chunks except the last one)
            class_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
            
            return class_loss
        
        # prediction_layer = model.get_layer('prediction')
        # abstract_layer = model.get_layer('abstract_output')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=model.output)
        # create a keras model for just running 
        feedforward_model = tf.keras.Model(inputs=model.inputs,
                                        outputs=rsnn_layer.output)
        
        # These "dummy" zeros are injected to the models membrane voltage
        # Provides the opportunity to compute gradients wrt. membrane voltages at all time steps
        # Not important for general use
        n_total_neurons = v1_neurons + lm_neurons
        zero_state = rsnn_layer.cell.zero_state_multi_areas(per_replica_batch_size, dtype=dtype)
        # state_variables = tf.nest.map_structure(lambda a: tf.Variable(
        #     a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        # ), zero_state)
        # dummy_zeros = tf.zeros((per_replica_batch_size, flags.seq_len-delays[0], n_total_neurons), dtype)

        # Add other metrics and losses
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()
        train_classification_loss = tf.keras.metrics.Mean()
        val_classification_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        train_firing_rate = tf.keras.metrics.Mean()
        val_firing_rate = tf.keras.metrics.Mean()
        train_rate_loss = tf.keras.metrics.Mean()
        val_rate_loss = tf.keras.metrics.Mean()
        train_voltage_loss = tf.keras.metrics.Mean()
        val_voltage_loss = tf.keras.metrics.Mean()
        train_regularizer_loss = tf.keras.metrics.Mean()
        val_regularizer_loss = tf.keras.metrics.Mean()

        def reset_train_metrics():
            train_loss.reset_states(), train_classification_loss.reset_states(),
            train_accuracy.reset_states(), train_firing_rate.reset_states()
            train_rate_loss.reset_states(), train_voltage_loss.reset_states(), train_regularizer_loss.reset_states(), 

        def reset_validation_metrics():
            val_loss.reset_states(), val_classification_loss.reset_states(),
            val_accuracy.reset_states(), val_firing_rate.reset_states(), 
            val_rate_loss.reset_states(), val_voltage_loss.reset_states(), val_regularizer_loss.reset_states(), 
    
        # Precompute spontaneous LGN firing rates once
        def compute_spontaneous_lgn_firing_rates():
            cache_dir = "lgn_model/.cache_lgn"
            cache_file = os.path.join(cache_dir, f"spontaneous_lgn_probabilities_n_input_{flags.n_input}_seqlen_{flags.seq_len-delays[0]}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    spontaneous_prob = pkl.load(f)
                print("Loaded cached spontaneous LGN firing rates.")
            else:
                # Compute and cache the spontaneous firing rates
                spontaneous_lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
                    seq_len=flags.seq_len-delays[0],
                    pre_delay=flags.seq_len-delays[0],
                    post_delay=0,
                    n_input=flags.n_input,
                    rotation=flags.rotation,
                    billeh_phase=True,
                    return_firing_rates=True,
                    dtype=dtype
                )
                spontaneous_lgn_firing_rates = next(iter(spontaneous_lgn_firing_rates))
                spontaneous_prob = 1 - tf.exp(-tf.cast(spontaneous_lgn_firing_rates, dtype) / 1000.)
                # Save to cache
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, "wb") as f:
                    pkl.dump(spontaneous_prob, f)
                print("Computed and cached spontaneous LGN firing rates.")
            
            # repeat the spontaneous firing rates with shape (seqlen, n_input) to match the batch size 
            spontaneous_prob = tf.tile(tf.expand_dims(spontaneous_prob, axis=0), [per_replica_batch_size, 1, 1])

            return tf.cast(spontaneous_prob, dtype=dtype)

        # Load the spontaneous probabilities once
        spontaneous_prob = compute_spontaneous_lgn_firing_rates()

        # num_grad_accumulates = tf.constant(flags.num_grad_accumulates, dtype=tf.float32)

    def roll_out(_x, _y, _w, _state_variables, spontaneous=False):
        # _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
        # Access initial state values directly
        _initial_state = _state_variables
        seq_len = tf.shape(_x)[1]
        dummy_zeros = tf.zeros((per_replica_batch_size, seq_len, n_total_neurons), dtype)

        if flags.gradient_checkpointing:
            @tf.recompute_grad
            def roll_out_with_gradient_checkpointing(x, dummy_zeros, state_vars):
                # Call extractor model without storing intermediate state variables
                # dummy_zeros = tf.zeros((per_replica_batch_size, seq_len, n_total_neurons), dtype)
                _out, _p = extractor_model((x, dummy_zeros, state_vars))
                return _out, _p
            _out, _p = roll_out_with_gradient_checkpointing(_x, dummy_zeros, _initial_state)
        else:
            # dummy_zeros = tf.zeros((per_replica_batch_size, seq_len, n_total_neurons), dtype)
            _out, _p = extractor_model((_x, dummy_zeros, _initial_state))

        # _out, _p = extractor_model((_x, dummy_zeros, _initial_state))
        _v1_z, _v1_v, _lm_z, _lm_v = _out[0]

        if flags.dtype != 'float32':
            _v1_z = tf.cast(_v1_z, tf.float32)
            _v1_v = tf.cast(_v1_v, tf.float32)
            _lm_z = tf.cast(_lm_z, tf.float32)
            _lm_v = tf.cast(_lm_v, tf.float32)

        # update state_variables with the new model state
        # new_state = tuple(_out[1:])
        # tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)
        
        if spontaneous:
            return _out
        else:
            # Compute the final term only after the first three terms have been computed
            v1_rate_loss = v1_evoked_rate_regularizer(_v1_z, False)
            lm_rate_loss = lm_evoked_rate_regularizer(_lm_z, False)
            rate_loss = v1_rate_loss + lm_rate_loss

            v1_voltage_loss = v1_voltage_regularizer(_v1_v) # trim is irrelevant for this
            lm_voltage_loss = lm_voltage_regularizer(_lm_v) # trim is irrelevant for this
            voltage_loss = v1_voltage_loss + lm_voltage_loss

            # v1_recurrent_stiff_regularizer = v1_recurrent_regularizer(rsnn_layer.cell.v1.recurrent_weight_values)
            # lm_recurrent_stiff_regularizer = lm_recurrent_regularizer(rsnn_layer.cell.lm.recurrent_weight_values)
            # # v1_lm_weights_l2_regularizer = v1_lm_regularizer(rsnn_layer.cell.v1.interarea_weight_values['lm'])
            # # lm_v1_weights_l2_regularizer = lm_v1_regularizer(rsnn_layer.cell.lm.interarea_weight_values['v1'])
            # regularizers_loss = v1_recurrent_stiff_regularizer + lm_recurrent_stiff_regularizer #+ v1_lm_weights_l2_regularizer + lm_v1_weights_l2_regularizer
            regularizers_loss = tf.constant(0., tf.float32)
            
            classification_loss = 10*compute_loss(_y, _p, _w)
            # Scale the losses since the optimizer will aggregate the gradients across replicas automatically before applying them
            _loss = tf.nn.scale_regularization_loss(rate_loss + voltage_loss + regularizers_loss) + classification_loss
            
            _aux = dict(rate_loss=rate_loss, voltage_loss=voltage_loss, regularizer_loss=regularizers_loss, classification_loss=classification_loss * num_replicas)
            # tf.print('Loss: ', _loss, rate_loss, voltage_loss, classification_loss)

            return _out, _p, _loss, _aux
    
    def roll_out_without_losses(_x, _state_variables, seq_len, return_spikes=False):
        """Process initial delay period without computing gradients"""
        _initial_state = _state_variables
        dummy_zeros = tf.zeros((per_replica_batch_size, seq_len, n_total_neurons), dtype)
        _out = feedforward_model((_x, dummy_zeros, _initial_state))
        new_state = _out[1:] #(14)

        if return_spikes:
            hidden_variables = _out[0]
            return hidden_variables, new_state  # Return spikes and state variables
        else:
            return new_state  # Return only the state variables
    
    def train_step(_x, _y, _w, state_variables):
        # Split input into initial delay and training periods inside the replica
        _x = tf.cast(_x, dtype)
        initial_period = _x[:, :delays[0], :]  # First delays[0] timesteps
        training_period = _x[:, delays[0]:, :]  # Remaining timesteps
        # remove first chunk from y and w
        _y = _y[:, 1:]
        _w = _w[:, 1:]
        
        # Process initial period without gradients
        state_after_delay = roll_out_without_losses(initial_period, state_variables, delays[0])

        ### Forward propagation of the model
        with tf.GradientTape() as tape:
            _out, _p, _loss, _aux = roll_out(training_period, _y, _w, state_after_delay)
            # Scale the loss for float16         
            if flags.dtype == 'float16':
                _scaled_loss = optimizer.get_scaled_loss(_loss)
                loss = _scaled_loss
            else:
                loss = _loss

        # # Print average gradients
        grad = tape.gradient(loss, model.trainable_variables)
        if flags.dtype == 'float16':
            grad = optimizer.get_unscaled_gradients(grad)

        # for g, v in zip(grad, model.trainable_variables):
        #     tf.print(f"Gradient for {v.name}: ", g)
            # optimizer.apply_gradients([(g, v)])

        # The optimizer will aggregate the gradients across replicas automatically before applying them by default,
        # so the losses have to be properly scaled to account for the number of replicas
        # https://www.tensorflow.org/tutorials/distribute/custom_training
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/optimizer_v2.py#L741
        optimizer.apply_gradients(zip(grad, model.trainable_variables)) #, experimental_aggregate_gradients=False)

        # Update the model metrics
        train_accuracy.update_state(_y, _p, sample_weight=_w)
        train_loss.update_state(_loss * num_replicas)
        rate = tf.reduce_mean(tf.concat([_out[0][0], _out[0][2]], axis=-1))
        train_firing_rate.update_state(rate)
        train_rate_loss.update_state(_aux['rate_loss'])
        train_classification_loss.update_state(_aux['classification_loss'])
        train_voltage_loss.update_state(_aux['voltage_loss'])
        train_regularizer_loss.update_state(_aux['regularizer_loss'])

        rate_loss = train_rate_loss.result()
        voltage_loss = train_voltage_loss.result()
        regularizers_loss = train_regularizer_loss.result()
        classification_loss = train_classification_loss.result()
        _loss = train_loss.result()
        rate = train_firing_rate.result()
        accuracy = train_accuracy.result()

        return state_after_delay, [accuracy, _loss, rate, rate_loss, voltage_loss, regularizers_loss, classification_loss]

    @tf.function
    def distributed_train_step(x, y, weights, state_variables):
        gray_state, step_values = strategy.run(train_step, args=(x, y, weights, state_variables))
        # step_values = [
        #     strategy.reduce(tf.distribute.ReduceOp.MEAN if i in [0, 2] else tf.distribute.ReduceOp.SUM, value, axis=None)
        #     for i, value in enumerate(step_values)
        # ]

        return gray_state, step_values
    
    def validation_step(_x, _y, _w, state_variables, output_spikes=False):
        # _out, _p, _loss, _aux = roll_out(_x, _y, _w, state_variables)

        _x = tf.cast(_x, dtype)
        initial_period = _x[:, :delays[0], :]  # First delays[0] timesteps
        testing_period = _x[:, delays[0]:, :]  # Remaining timesteps
        # remove first chunk from y and w
        _y = _y[:, 1:]
        _w = _w[:, 1:]
        
        # # Process initial period without gradients
        _initial_out, state_after_delay = roll_out_without_losses(initial_period, state_variables, delays[0], return_spikes=True)
        # ### Forward propagation of the model
        _out, _p, _loss, _aux = roll_out(testing_period, _y, _w, state_after_delay)

        val_accuracy.update_state(_y, _p, sample_weight=_w)
        # Update the validation loss
        val_loss.update_state(_loss * num_replicas)
        # Compute the rate
        _rate = tf.reduce_mean(tf.concat([_out[0][0], _out[0][2]], axis=-1))
        # Update the validation metrics
        val_firing_rate.update_state(_rate)
        val_rate_loss.update_state(_aux['rate_loss'])
        val_classification_loss.update_state(_aux['classification_loss'])
        val_voltage_loss.update_state(_aux['voltage_loss'])
        val_regularizer_loss.update_state(_aux['regularizer_loss'])                        
        # tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])

        if output_spikes:
            # _v1_z0, _lm_z0 = _initial_out[0], _initial_out[2]
            # _v1_z, _lm_z = _out[0][0], _out[0][2]
            # concatenate the spikes of two periods
            _v1_z = tf.concat([_initial_out[0], _out[0][0]], axis=1)
            _lm_z = tf.concat([_initial_out[2], _out[0][2]], axis=1)

            return _v1_z, _lm_z   

    @tf.function
    def distributed_validation_step(x, y, weights, state_variables, output_spikes=True):
        if output_spikes:
            return strategy.run(validation_step, args=(x, y, weights, state_variables, output_spikes))
        else:
            strategy.run(validation_step, args=(x, y, weights, state_variables))

    ### LGN INPUT ###
    # Define the MNIST dataset
    def get_dataset_fn(is_test=False):
        def _f(input_context):
            batch_size = input_context.get_per_replica_batch_size(global_batch_size)
            n_examples = 10000 if is_test else 60000 #int(50000/64)

            _data_set = (stim_dataset.generate_pure_classification_data_set_from_generator(
                data_usage=int(is_test), contrast=1, im_slice=100, std=flags.mnist_noise_std,
                pre_delay=delays[0], post_delay=delays[1], current_input=flags.current_input, n_input=flags.n_input,
                dataset='mnist', pre_chunks=3, resp_chunks=1, post_chunks=0, from_lgn=True,
                )
            .take(n_examples) # Limit the number of examples
            .shard(input_context.num_input_pipelines, input_context.input_pipeline_id) # Distribute dataset across replicas
            .shuffle(buffer_size=100, reshuffle_each_iteration=True)  # Shuffle the dataset
            .batch(batch_size, drop_remainder=True) # Batch the dataset and drop the remainder to ensure consistent batch size
            .prefetch(tf.data.AUTOTUNE) # Prefetch for performance
            #.shard(8, input_context.input_pipeline_id - 32).prefetch(1) # task_id = input_context.input_pipeline_id, [16,23] is 10class
                # post_chunks=flags.post_chunks).take(n_examples).batch(per_replica_batch_size).shard(8, input_context.input_pipeline_id//3).prefetch(8) # 49984=int(50000/64); 8 nodes for each task, so divide it to 8 parts; total 3 tasks, so every 3 task_ids will choose the correct part
            )
            return _data_set
        return _f

    # Generate spontaneous spikes efficiently
    @tf.function
    def generate_spontaneous_spikes(spontaneous_prob):
        random_uniform = tf.random.uniform(tf.shape(spontaneous_prob), dtype=dtype)
        return tf.less(random_uniform, spontaneous_prob)
    
    train_data_set = strategy.distribute_datasets_from_function(get_dataset_fn(False))
    test_data_set = strategy.distribute_datasets_from_function(get_dataset_fn(True))
    
    def generate_gray_state(spontaneous_prob):
        # Generate LGN spikes
        x = generate_spontaneous_spikes(spontaneous_prob)
        # Simulate the network with a gray screen   
        gray_state = roll_out_without_losses(x, zero_state, flags.seq_len-delays[0])

        return gray_state
    
    @tf.function
    def distributed_generate_gray_state(spontaneous_prob):
        # Run generate_gray_state on each replica
        return strategy.run(generate_gray_state, args=(spontaneous_prob,))
    
    def process_output_spikes(v1_spikes, networks, seq_len=200, down_sample=50, n_output=10, area='v1'):
        outputs_10 = []
        for i in range(n_output):
            t_output = tf.gather(v1_spikes, networks[area][f'readout_neuron_ids_{i}'], axis=2)
            t_output = tf.cast(t_output, tf.float32)
            t_output = tf.reduce_mean(t_output, -1)
            outputs_10.append(t_output)
        output = tf.concat(outputs_10, axis=-1)

        mean_output = tf.reshape(output, (-1, int(seq_len / down_sample), down_sample, n_output)) # TensorShape([None, 4, 50, 10])
        mean_output = tf.reduce_mean(mean_output, axis=2) # TensorShape([None, 4, 10])
        # mean_output = mean_output[:, -1, :]
        
        return mean_output

    ############################ TRAINING #############################
    stop = False
    # Initialize your callbacks
    metric_keys = ['train_accuracy', 'train_loss', 'train_firing_rate', 'train_rate_loss', 'train_voltage_loss', 'train_regularizer_loss', 'train_classification_loss',
                   'val_accuracy', 'val_loss', 'val_firing_rate', 'val_rate_loss', 'val_voltage_loss', 'val_regularizer_loss', 'val_classification_loss']

    callbacks = ClassificationCallbacks(networks, lgn_inputs, bkg_inputs, model, optimizer, flags, logdir, strategy, metric_keys, 
                                        pre_delay=delays[0], post_delay=delays[1], model_variables_init=model_variables_dict,
                                        checkpoint=checkpoint)

    callbacks.on_train_begin()
    n_prev_epochs = flags.run_session * flags.n_epochs

    ############################ ONLY TESTING #############################
    if flags.test_only:
        print(f'Testing...')
        callbacks.on_epoch_start()  
        # Reset the model state to the gray state  
        gray_state = distributed_generate_gray_state(spontaneous_prob)
        # Load the dataset iterator
        test_it = iter(test_data_set)
        val_t0 = time()  
        # accumulated_p_v1 = []
        # accumulated_p_lm = []
        accumulated_p = []
        accumulated_y = []
        for step in range(val_steps): # total samples = batch_size * val_steps
            t0 = time()
            # Generate LGN spikes
            x, y, image_id, w = next(test_it)
            v1_spikes, lm_spikes = distributed_validation_step(x, y, w, gray_state, True)
            ### Process the output spikes
            mean_output = process_output_spikes(v1_spikes, networks, seq_len=flags.seq_len, down_sample=10, 
                                                n_output=flags.n_output, area='v1')
            
            # mean_output_v1 = tf.reduce_sum(v1_spikes[:, -50:, :], axis=1)
            # mean_output_lm = tf.reduce_sum(lm_spikes[:, -50:, :], axis=1)

            accumulated_y.append(y[:, -1])
            accumulated_p.append(mean_output.numpy().astype(np.float16))

            # accumulated_p_v1.append(mean_output_v1.numpy().astype(np.float16))
            # accumulated_p_lm.append(mean_output_lm.numpy().astype(np.float16))

            print(f'\nStep {step}/{val_steps} running time: {time() - t0:.2f}s')
            for gpu_id in range(len(strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)

        # Convert accumulated lists to numpy arrays if needed
        # accumulated_p_v1 = np.array(accumulated_p_v1)
        # accumulated_p_lm = np.array(accumulated_p_lm)
        accumulated_y = np.array(accumulated_y)
        accumulated_p = np.array(accumulated_p)
        # save the accumulated predictions and labels
        # with open(f'{logdir}/accumulated_predictions.pkl', 'wb') as f:
        #     pkl.dump({'predictions': accumulated_p, 'labels': accumulated_y}, f)
        with open(f'accumulated_predictions_lm_TD_{flags.connected_areas}_2.pkl', 'wb') as f:
            pkl.dump({'predictions': accumulated_p, 'labels': accumulated_y}, f)
            # pkl.dump({'v1_spikes': accumulated_p_v1, 'lm_spikes': accumulated_p_lm, 'labels': accumulated_y}, f)
            
        print(f'\nValidation running time: {time() - val_t0:.2f}s')

        val_values = [a.result().numpy() for a in [val_accuracy, val_loss, val_firing_rate, val_rate_loss,
                                                    val_voltage_loss, val_regularizer_loss, val_classification_loss]]        
        metric_values = val_values + val_values
        
        # select only the values for the last replica in case there are multiple replicas
        if strategy.num_replicas_in_sync > 1:
            x = strategy.experimental_local_results(x)[-1]
            v1_spikes = strategy.experimental_local_results(v1_spikes)[-1]
            lm_spikes = strategy.experimental_local_results(lm_spikes)[-1]
            y = strategy.experimental_local_results(y)[-1]
            image_id = strategy.experimental_local_results(image_id)[-1]

        # save pkl file with x[0], v1_spikes[0], lm_spikes[0], y[0]
        with open('x_v1_lm_spikes.pkl', 'wb') as f:
            pkl.dump({'x': x[0], 'v1_spikes': v1_spikes[0], 'lm_spikes': lm_spikes[0], 'y': y[0]}, f)
        
        callbacks.on_testing_end(x, v1_spikes, lm_spikes, y, image_id, metric_values, verbose=True)    
        # Reset the metrics for the next epoch
        reset_validation_metrics()

    # import datetime
    # profiler_logdir = f"{logdir}/logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # # Set steps to profile
    # profile_start_step = 1
    # profile_end_step = 4
    else:
        for epoch in range(n_prev_epochs, n_prev_epochs + flags.n_epochs):
            callbacks.on_epoch_start()  
            
            # Reset the model state to the gray state  
            gray_state = distributed_generate_gray_state(spontaneous_prob)
            
            # Load the dataset iterator - this must be done inside the epoch loop
            it = iter(train_data_set)
            test_it = iter(test_data_set)
            
            for step in range(flags.steps_per_epoch):
                callbacks.on_step_start()
                # # # Start profiler at specified step
                # if step == profile_start_step:
                #     tf.profiler.experimental.start(logdir=logdir)

                # # try resetting every iteration or every 25 steps to prevent drifting away from the gray state
                # if flags.reset_every_step or step % 25 == 0:
                #     print("Resetting gray state!")
                #     gray_state = distributed_generate_gray_state(spontaneous_prob)

                # Generate LGN spikes
                x, y, _, w = next(it) # x dtype tf.bool
                # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
                try:
                    gray_state, step_values = distributed_train_step(x, y, w, gray_state)
                    # step_gradients, step_values = distributed_train_step(x, y, w, gray_state)
                except tf.errors.ResourceExhaustedError as e:
                    print("OOM error occurred!")

                # # # Stop profiler after profiling steps
                # if step == profile_end_step:
                #     tf.profiler.experimental.stop()
                callbacks.on_step_end(step_values, verbose=True)

            ## VALIDATION AFTER EACH EPOCH          
            gray_state = distributed_generate_gray_state(spontaneous_prob)
            val_t0 = time()  
            for step in range(val_steps): # total samples = batch_size * val_steps
                # Generate LGN spikes
                x, y, image_id, w = next(test_it)
                v1_spikes, lm_spikes = distributed_validation_step(x, y, w, gray_state, output_spikes=True)

            print(f'\nValidation running time: {time() - val_t0:.2f}s')
            
            train_values = [a.result().numpy() for a in [train_accuracy, train_loss, train_firing_rate, train_rate_loss,
                                                        train_voltage_loss, train_regularizer_loss, train_classification_loss]]

            val_values = [a.result().numpy() for a in [val_accuracy, val_loss, val_firing_rate, val_rate_loss,
                                                        val_voltage_loss, val_regularizer_loss, val_classification_loss]]
            metric_values = train_values + val_values

            # select only the values for the last replica in case there are multiple replicas
            if num_replicas > 1:
                x = strategy.experimental_local_results(x)[-1]
                v1_spikes = strategy.experimental_local_results(v1_spikes)[-1]
                lm_spikes = strategy.experimental_local_results(lm_spikes)[-1]
                y = strategy.experimental_local_results(y)[-1]
                image_id = strategy.experimental_local_results(image_id)[-1]

            # # save pkl file with x[0], v1_spikes[0], lm_spikes[0], y[0]
            # with open('x_v1_lm_spikes.pkl', 'wb') as f:
            #     pkl.dump({'x': x[0], 'v1_spikes': v1_spikes[0], 'lm_spikes': lm_spikes[0], 'y': y[0]}, f)

            stop = callbacks.on_epoch_end(x, v1_spikes, lm_spikes, y, image_id, metric_values, verbose=True)    
            
            if stop:
                break
            
            # Reset the metrics for the next epoch
            reset_train_metrics()
            reset_validation_metrics()

        callbacks.on_train_end(metric_values)


if __name__ == '__main__':
    _data_dir = 'GLIF_network'
    _results_dir = 'Simulation_results'

    absl.app.flags.DEFINE_string('task_name', '10class' , '')
    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('ckpt_dir', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('interarea_weight_distribution', 'billeh_weights', '')
    # absl.app.flags.DEFINE_string('interarea_weight_distribution', 'zero_weights', '')
    absl.app.flags.DEFINE_string('delays', '50,50', '')
    absl.app.flags.DEFINE_string('optimizer', 'adam', '')
    absl.app.flags.DEFINE_string('dtype', 'float32', '')
    absl.app.flags.DEFINE_string('osi_loss_method', 'crowd_osi', '')
    absl.app.flags.DEFINE_string("rotation", "ccw", "")

    absl.app.flags.DEFINE_float('learning_rate', .001, '')
    absl.app.flags.DEFINE_float('rate_cost', 100., '')
    absl.app.flags.DEFINE_float('voltage_cost', 1., '')
    absl.app.flags.DEFINE_float('sync_cost', 1., '')
    absl.app.flags.DEFINE_float('osi_cost', 1., '')
    absl.app.flags.DEFINE_float('osi_loss_subtraction_ratio', 1., '')
    absl.app.flags.DEFINE_float('dampening_factor', .5, '')
    absl.app.flags.DEFINE_float('recurrent_dampening_factor', .5, '')
    absl.app.flags.DEFINE_float('input_weight_scale', 1., '')
    absl.app.flags.DEFINE_float('gauss_std', .3, '')
    absl.app.flags.DEFINE_float('recurrent_weight_regularization', 0., '')
    absl.app.flags.DEFINE_float('interarea_weight_regularization', 0., '')
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    absl.app.flags.DEFINE_float('input_f0', 0.2, '')
    absl.app.flags.DEFINE_float('E4_weight_factor', 1., '')
    absl.app.flags.DEFINE_float('temporal_f', 2., '')
    absl.app.flags.DEFINE_float('max_time', -1, '')
    absl.app.flags.DEFINE_float('mnist_noise_std', 0, '')

    absl.app.flags.DEFINE_integer('n_runs', 1, '')
    absl.app.flags.DEFINE_integer('run_session', 0, '')
    absl.app.flags.DEFINE_integer('n_epochs', 20, '')
    absl.app.flags.DEFINE_integer('osi_dsi_eval_period', 50, '') # number of epochs for osi/dsi evaluation if n_runs = 1
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('v1_neurons', 10, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('lm_neurons', 10, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('steps_per_epoch', 10, '')# EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('val_steps', 10, '')# EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    # number of LGN filters in visual space (input population)
    absl.app.flags.DEFINE_integer('n_input', 17400, '')
    absl.app.flags.DEFINE_integer('seq_len', 200, '')
    absl.app.flags.DEFINE_integer('n_cues', 3, '')
    absl.app.flags.DEFINE_integer('recall_duration', 40, '')
    absl.app.flags.DEFINE_integer('cue_duration', 40, '')
    absl.app.flags.DEFINE_integer('interval_duration', 40, '')
    absl.app.flags.DEFINE_integer('examples_in_epoch', 32, '')
    absl.app.flags.DEFINE_integer('validation_examples', 16, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_integer('n_output', 10, '')
    absl.app.flags.DEFINE_integer('neurons_per_output', 30, '')
    absl.app.flags.DEFINE_integer('n_trials_per_angle', 10, '')
    absl.app.flags.DEFINE_integer('num_grad_accumulates', 4, '')

    # absl.app.flags.DEFINE_boolean('float16', False, '')
    absl.app.flags.DEFINE_boolean('caching', True, '')
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('core_loss', False, '')
    absl.app.flags.DEFINE_boolean('hard_reset', False, '')
    absl.app.flags.DEFINE_boolean('disconnect_lm_L6_inhibition', False, '')
    absl.app.flags.DEFINE_boolean('disconnect_v1_lm_L6_excitatory_projections', False, '')
    absl.app.flags.DEFINE_boolean('random_weights', False, '')
    absl.app.flags.DEFINE_boolean('realistic_neurons_ratio', False, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_v1', False, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_lm', False, '')
    absl.app.flags.DEFINE_boolean('train_input', False, '')
    absl.app.flags.DEFINE_boolean('train_interarea_lm_v1', False, '')
    absl.app.flags.DEFINE_boolean('train_interarea_v1_lm', False, '')
    absl.app.flags.DEFINE_boolean('train_noise', False, '')
    # absl.app.flags.DEFINE_boolean('train_recurrent_per_type', False, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', True, '')
    # absl.app.flags.DEFINE_boolean('hard_only', False, '')
    absl.app.flags.DEFINE_boolean('visualize_test', False, '')
    absl.app.flags.DEFINE_boolean('pseudo_gauss', False, '')
    absl.app.flags.DEFINE_boolean("bmtk_compat_lgn", True, "")
    absl.app.flags.DEFINE_boolean("reset_every_step", False, "")
    absl.app.flags.DEFINE_boolean("spontaneous_training", False, "")
    absl.app.flags.DEFINE_boolean("spontaneous_uniform_distribution_constraint", False, "")
    absl.app.flags.DEFINE_boolean("current_input", False, "")
    absl.app.flags.DEFINE_boolean("test_only", False, "")
    absl.app.flags.DEFINE_boolean("connected_areas", True, "")
    absl.app.flags.DEFINE_boolean("connected_recurrent_connections", True, "")
    absl.app.flags.DEFINE_boolean("connected_noise", True, "")
    absl.app.flags.DEFINE_boolean("gradient_checkpointing", True, "")

    absl.app.run(main)

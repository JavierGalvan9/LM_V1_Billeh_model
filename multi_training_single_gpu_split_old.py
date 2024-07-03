import os

# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # before import tensorflow
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
from Model_utils.callbacks import Callbacks
# from general_utils import file_management

from time import time
import ctypes.util


print("--- CUDA version: ", tf.sysconfig.get_build_info()["cuda_version"])
print("--- CUDNN version: ", tf.sysconfig.get_build_info()["cudnn_version"])
print("--- TensorFlow version: ", tf.__version__)

# For CUDA Runtime API
lib_path = ctypes.util.find_library('cudart')
print("--- CUDA Library path: ", lib_path)

debug = False


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
            if value != flags[name].default and name in ['n_input', 'core_only', 'connected_selection', 'interarea_weight_distribution', 'E4_weight_factor']:
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
    if flags.float16:
        # policy = mixed_precision.Policy("mixed_float16")
        # mixed_precision.set_policy(policy)
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            mixed_precision.set_global_policy('mixed_float16')
        dtype = tf.float16
    else:
        dtype = tf.float32

    n_workers, n_gpus_per_worker = 1, 1
    # model is being run on multiple GPUs or CPUs, and the results are being reduced to a single CPU device. 
    # In this case, the reduce_to_device argument is set to "cpu:0", which means that the results are being reduced to the first CPU device.
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0")) 
    device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device=device)
    # strategy = tf.distribute.MirroredStrategy()

    per_replica_batch_size = flags.batch_size
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    print(f'Global batch size: {strategy.num_replicas_in_sync}, {global_batch_size}\n')

    # Define 2 outputs that correspond to having more cues top or bottom
    # Note that two different output conventions can be used:
    # 1) Linear readouts from all neurons in the model (softmax)
    # 2) Selecting a population of neurons that report a binary decision
    # with high firing rate (flag --neuron_output)
    # n_output = 2

    # Load data of Billeh et al. (2020) and select appropriate number of neurons and inputs
    # Create the v1-lm model
    # input_population, networks, bkg_weights, n_input, n_abstract_output, n_completed_output = load_fn(
    #     flags, interarea_connectivity, n_neurons)
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
            seq_len=flags.seq_len,
            n_input=flags.n_input, 
            dtype=dtype, 
            input_weight_scale=flags.input_weight_scale,
            interarea_weight_scale=1., 
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
            batch_size=flags.batch_size, 
            pseudo_gauss=flags.pseudo_gauss, 
            use_state_input=True, 
            return_state=True,
            hard_reset=flags.hard_reset,
            connected_areas=True,
            add_rate_metric=True, 
            max_delay=5, 
            # output_completed_valid_from_time=120, 
            # output_abstract_valid_from_time=100,
            )

        model.build((flags.batch_size, flags.seq_len, flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")

        # Store the initial model variables that are going to be trained
        model_variables_dict = {'Initial': {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}}

        # Define the optimizer
        # optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11, clipnorm=0.001)  
        if flags.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)  
        elif flags.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(flags.learning_rate, momentum=0.0, nesterov=False)
        else:
            print(f"Invalid optimizer: {flags.optimizer}")
            raise ValueError
        
        optimizer.build(model.trainable_variables)

        # Restore model and optimizer from a checkpoint if it exists
        if flags.ckpt_dir != '' and os.path.exists(os.path.join(flags.ckpt_dir, "OSI_DSI_checkpoints")):
            checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, "OSI_DSI_checkpoints"))
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
                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
                optimizer.build(model.trainable_variables)
            else:
                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.restore(checkpoint_directory).assert_consumed()
        # Option to resume the training from a checkpoint from a previous training session
        elif flags.restore_from != '' and os.path.exists(flags.restore_from):
            checkpoint_directory = tf.train.latest_checkpoint(flags.restore_from)
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
                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.restore(checkpoint_directory).expect_partial()#.assert_consumed()
                optimizer.build(model.trainable_variables)
            else:
                # Restore the model
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
                checkpoint.restore(checkpoint_directory).assert_consumed()
        else:
            print(f"No checkpoint found in {flags.ckpt_dir} or {flags.restore_from}. Starting from scratch...\n")
            checkpoint = None

        model_variables_dict['Best'] =  {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}
        print(f"Model variables stored in dictionary\n")
        # print(model_variables_dict)

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

        rsnn_layer = model.get_layer('rsnn')
        
        ### RECURRENT REGULARIZERS ###
        # v1_recurrent_regularizer = losses.L2Regularizer(flags.recurrent_weight_regularization, networks['v1'], flags, penalize_relative_change=True)
        v1_recurrent_regularizer = losses.StiffRegularizer(flags.recurrent_weight_regularization, rsnn_layer.cell.v1.recurrent_weight_values)
        model.add_loss(lambda: v1_recurrent_regularizer(rsnn_layer.cell.v1.recurrent_weight_values))
        # lm_recurrent_regularizer = losses.L2Regularizer(flags.recurrent_weight_regularization, networks['lm'], flags, penalize_relative_change=True)
        lm_recurrent_regularizer = losses.StiffRegularizer(flags.recurrent_weight_regularization, rsnn_layer.cell.lm.recurrent_weight_values)
        model.add_loss(lambda: lm_recurrent_regularizer(rsnn_layer.cell.lm.recurrent_weight_values))

        # # ### INTERAREA REGULARIZERS ###
        # v1_lm_regularizer = losses.L2Regularizer(flags.interarea_weight_regularization, networks['v1'], flags, penalize_relative_change=True, recurrent_weights=False, source_area='lm')
        # model.add_loss(lambda: v1_lm_regularizer(rsnn_layer.cell.v1.interarea_weight_values['lm']))
        # lm_v1_regularizer = losses.L2Regularizer(flags.interarea_weight_regularization, networks['lm'], flags, penalize_relative_change=True, recurrent_weights=False, source_area='v1')
        # model.add_loss(lambda: lm_v1_regularizer(rsnn_layer.cell.lm.interarea_weight_values['v1']))

        ### EVOKED RATES REGULARIZERS ###
        v1_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], spontaneous_fr=False, rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1], 
                                                                            data_dir=flags.data_dir, area='v1', core_mask=v1_core_mask, seed=flags.seed, dtype=dtype)
        model.add_loss(lambda: v1_evoked_rate_regularizer(rsnn_layer.output[0][0]))
        lm_evoked_rate_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], spontaneous_fr=False, rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1], 
                                                                            data_dir=flags.data_dir, area='lm', core_mask=lm_core_mask, seed=flags.seed, dtype=dtype)
        model.add_loss(lambda: lm_evoked_rate_regularizer(rsnn_layer.output[0][2]))
        
        ### SPONTANEOUS RATES REGULARIZERS ###
        v1_spont_rate_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], spontaneous_fr=True, rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1], 
                                                                            data_dir=flags.data_dir, area='v1', core_mask=v1_core_mask, seed=flags.seed, dtype=dtype)
        model.add_loss(lambda: v1_spont_rate_regularizer(rsnn_layer.output[0][0]))
        lm_spont_rate_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], spontaneous_fr=True, rate_cost=flags.rate_cost, pre_delay=delays[0], post_delay=delays[1], 
                                                                            data_dir=flags.data_dir, area='lm', core_mask=lm_core_mask, seed=flags.seed, dtype=dtype)
        model.add_loss(lambda: lm_spont_rate_regularizer(rsnn_layer.output[0][2]))

        # ### ANNULUS RATES REGULARIZERS ###
        # if flags.core_loss and v1_neurons > 51978:
        #     # Add rate regularizer for the annulus
        #     v1_annulus_mask = ~v1_core_mask
        #     v1_annulus_mask = tf.constant(v1_annulus_mask, dtype=tf.bool)
        #     v1_annulus_rate_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], spontaneous_fr=True, rate_cost=0.1*flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
        #                                                                     data_dir=flags.data_dir, area='v1', core_mask=v1_annulus_mask, seed=flags.seed, dtype=dtype)
        #     model.add_loss(lambda: v1_annulus_rate_regularizer(rsnn_layer.output[0][0]))
        #     # Add rate regularizer for the annulus
        #     lm_annulus_mask = ~lm_core_mask
        #     lm_annulus_mask = tf.constant(lm_annulus_mask, dtype=tf.bool)
        #     lm_annulus_rate_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], spontaneous_fr=True, rate_cost=0.1*flags.rate_cost, pre_delay=delays[0], post_delay=delays[1],
        #                                                                     data_dir=flags.data_dir, area='lm', core_mask=lm_annulus_mask, seed=flags.seed, dtype=dtype)
        #     model.add_loss(lambda: lm_annulus_rate_regularizer(rsnn_layer.output[0][2]))

        ### VOLTAGE REGULARIZERS ###
        v1_voltage_regularizer = losses.VoltageRegularization(rsnn_layer.cell.v1, area='v1', voltage_cost=flags.voltage_cost, dtype=dtype) #, core_mask=v1_core_mask)
        lm_voltage_regularizer = losses.VoltageRegularization(rsnn_layer.cell.lm, area='lm', voltage_cost=flags.voltage_cost, dtype=dtype) #, core_mask=lm_core_mask)
        model.add_loss(lambda: v1_voltage_regularizer(rsnn_layer.output[0][1]) + lm_voltage_regularizer(rsnn_layer.output[0][3]))

        # # ### SYNCHRONIZATION REGULARIZERS ###
        # v1_sync_regularizer = losses.SynchronizationLoss(sync_cost=1., target_sync=0.1, area='v1', core_mask=v1_core_mask, pre_delay=delays[0], post_delay=delays[1], dtype=dtype)
        # lm_sync_regularizer = losses.SynchronizationLoss(sync_cost=1., target_sync=0.1, area='lm', core_mask=lm_core_mask, pre_delay=delays[0], post_delay=delays[1], dtype=dtype)
        # model.add_loss(lambda: v1_sync_regularizer(rsnn_layer.output[0][0]) + lm_sync_regularizer(rsnn_layer.output[0][2]))

        # Create an ExponentialMovingAverage object
        # Define the decay factor for the exponential moving average
        ema_decay = 0.95
        # Initialize exponential moving averages for V1 and LM firing rates
        if os.path.exists(os.path.join(logdir, 'train_end_data.pkl')):
            with open(os.path.join(logdir, 'train_end_data.pkl'), 'rb') as f:
                data_loaded = pkl.load(f)
                v1_ema = tf.Variable(data_loaded['v1_ema'], trainable=False, name='V1_EMA')
                lm_ema = tf.Variable(data_loaded['lm_ema'], trainable=False, name='LM_EMA')
        else:
            v1_ema = tf.Variable(tf.constant(0.01, shape=(v1_neurons,)), trainable=False, name='V1_EMA')
            lm_ema = tf.Variable(tf.constant(0.01, shape=(lm_neurons,)), trainable=False, name='LM_EMA')

        # if training for spontaneous firing rates set the osi loss to 0
        if flags.spontaneous_training:
            osi_cost = 0
        else:
            osi_cost = flags.osi_cost

        # here we need information of the layer mask for the OSI loss
        if flags.osi_loss_method == 'neuropixels_fr':
            v1_layer_info = other_billeh_utils.get_layer_info(networks['v1'])
            lm_layer_info = other_billeh_utils.get_layer_info(networks['lm'])
        else:
            v1_layer_info = None
            lm_layer_info = None
        
        ### OSI / DSI LOSSES ###
        v1_OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=networks['v1'], osi_cost=osi_cost, area='v1',
                                                        pre_delay=delays[0], post_delay=delays[1], 
                                                        dtype=dtype, core_mask=v1_core_mask,
                                                        method=flags.osi_loss_method,
                                                        subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                        layer_info=v1_layer_info)
        lm_OSI_DSI_Loss = losses.OrientationSelectivityLoss(network=networks['lm'], osi_cost=osi_cost, area='lm',
                                                        pre_delay=delays[0], post_delay=delays[1], 
                                                        dtype=dtype, core_mask=lm_core_mask,
                                                        method=flags.osi_loss_method,
                                                        subtraction_ratio=flags.osi_loss_subtraction_ratio,
                                                        layer_info=lm_layer_info)
        placeholder_angle = tf.constant(0, dtype=tf.float32, shape=(1, 1))
        model.add_loss(lambda: v1_OSI_DSI_Loss(rsnn_layer.output[0][0], placeholder_angle, trim=True, normalizer=v1_ema) \
                            + lm_OSI_DSI_Loss(rsnn_layer.output[0][2], placeholder_angle, trim=True, normalizer=lm_ema))


        # prediction_layer = model.get_layer('prediction')
        # abstract_layer = model.get_layer('abstract_output')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=[rsnn_layer.output, model.output[0], model.output[1], model.output[2]])

        # Loss from Guozhang classification task (unused in our case)
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        # def compute_loss(_l, _p, _w):
        #     per_example_loss = loss_object(_l, _p, sample_weight=_w) * strategy.num_replicas_in_sync / tf.reduce_sum(_w)
        #     rec_weight_loss = rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)
        #     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size) + rec_weight_loss
        
        # These "dummy" zeros are injected to the models membrane voltage
        # Provides the opportunity to compute gradients wrt. membrane voltages at all time steps
        # Not important for general use
        n_total_neurons = v1_neurons + lm_neurons
        zero_state = rsnn_layer.cell.zero_state_multi_areas(flags.batch_size, dtype=dtype)
        state_variables = tf.nest.map_structure(lambda a: tf.Variable(
            a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        ), zero_state)

        # Add other metrics and losses
        train_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()
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
        train_osi_dsi_loss = tf.keras.metrics.Mean()
        val_osi_dsi_loss = tf.keras.metrics.Mean()
        train_sync_loss = tf.keras.metrics.Mean()
        val_sync_loss = tf.keras.metrics.Mean()

        def reset_train_metrics():
            train_loss.reset_states(), train_accuracy.reset_states(), train_firing_rate.reset_states()
            train_rate_loss.reset_states(), train_voltage_loss.reset_states(), train_regularizer_loss.reset_states(), 
            train_osi_dsi_loss.reset_states(), train_sync_loss.reset_states()

        def reset_validation_metrics():
            val_loss.reset_states(), val_accuracy.reset_states(), val_firing_rate.reset_states(), 
            val_rate_loss.reset_states(), val_voltage_loss.reset_states(), val_regularizer_loss.reset_states(), 
            val_osi_dsi_loss.reset_states(), val_sync_loss.reset_states()

    def roll_out(_x, _y, _w, spontaneous=False, trim=True):
        _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
        # Access initial state values directly
        # _initial_state = [var.read_value() for var in state_variables]

        seq_len = tf.shape(_x)[1]
        dummy_zeros = tf.zeros((flags.batch_size, seq_len, n_total_neurons), dtype)
        _out, _p, _, _bkg_noise = extractor_model((_x, dummy_zeros, _initial_state))
        _v1_z, _v1_v, _lm_z, _lm_v = _out[0]
        # update state_variables with the new model state
        new_state = tuple(_out[1:])
        tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

        # update the exponential moving average of the firing rates
        v1_rates = tf.reduce_mean(_v1_z, (0, 1))
        lm_rates = tf.reduce_mean(_lm_z, (0, 1))
        # Update the EMAs
        v1_ema.assign(ema_decay * v1_ema + (1 - ema_decay) * v1_rates)
        lm_ema.assign(ema_decay * lm_ema + (1 - ema_decay) * lm_rates)
        tf.print('V1_ema: ', tf.reduce_mean(v1_ema), tf.reduce_mean(v1_rates), v1_ema)
        tf.print('lm_ema: ', tf.reduce_mean(lm_ema), tf.reduce_mean(lm_rates), lm_ema)

        v1_voltage_loss = v1_voltage_regularizer(_v1_v) # trim is irrelevant for this
        lm_voltage_loss = lm_voltage_regularizer(_lm_v) # trim is irrelevant for this
        voltage_loss = v1_voltage_loss + lm_voltage_loss

        # v1_sync_loss = v1_sync_regularizer(_v1_z, trim)
        # lm_sync_loss = lm_sync_regularizer(_lm_z, trim)
        # sync_loss = v1_sync_loss + lm_sync_loss
        sync_loss = tf.constant(0.0, dtype=dtype)

        if spontaneous:
            v1_rate_loss = v1_spont_rate_regularizer(_v1_z, trim=trim) #+ v1_spont_rate_regularizer(_v1_z, trim, uniform_distribution_constraint=True)
            lm_rate_loss = lm_spont_rate_regularizer(_lm_z, trim=trim) #+ lm_spont_rate_regularizer(_lm_z, trim, uniform_distribution_constraint=True)
            # v1_rate_loss = tf.constant(0.0, dtype=dtype)
            # lm_rate_loss = tf.constant(0.0, dtype=dtype)
            osi_dsi_loss = tf.constant(0.0, dtype=dtype)
            # lm_to_v1_weights_l2_regularizer = lm_to_v1_weight_regularizer(rsnn_layer.cell.v1.interarea_weight_values['lm'])
            # v1_to_lm_weights_l2_regularizer = v1_to_lm_weight_regularizer(rsnn_layer.cell.lm.interarea_weight_values['v1'])
            v1_recurrent_stiff_regularizer = v1_recurrent_regularizer(rsnn_layer.cell.v1.recurrent_weight_values)
            lm_recurrent_stiff_regularizer = lm_recurrent_regularizer(rsnn_layer.cell.lm.recurrent_weight_values)
            # v1_lm_weights_l2_regularizer = v1_lm_regularizer(rsnn_layer.cell.v1.interarea_weight_values['lm'])
            # lm_v1_weights_l2_regularizer = lm_v1_regularizer(rsnn_layer.cell.lm.interarea_weight_values['v1'])
            regularizers_loss = v1_recurrent_stiff_regularizer + lm_recurrent_stiff_regularizer # + v1_lm_weights_l2_regularizer + lm_v1_weights_l2_regularizer
            # tf.print('Regularizers: ', regularizers_loss, v1_recurrent_stiff_regularizer, lm_recurrent_stiff_regularizer) #, v1_lm_weights_l2_regularizer, lm_v1_weights_l2_regularizer)
            # regularizers_loss = tf.constant(0.0, dtype=dtype)
        else:
            # Compute the final term only after the first three terms have been computed
            v1_rate_loss = v1_evoked_rate_regularizer(_v1_z, trim=trim)
            lm_rate_loss = lm_evoked_rate_regularizer(_lm_z, trim=trim)
            v1_osi_dsi_loss = v1_OSI_DSI_Loss(_v1_z, _y, trim=trim, normalizer=v1_ema)
            lm_osi_dsi_loss = lm_OSI_DSI_Loss(_lm_z, _y, trim=trim, normalizer=lm_ema)
            osi_dsi_loss = v1_osi_dsi_loss + lm_osi_dsi_loss
            # osi_dsi_loss = tf.constant(0.0, dtype=dtype)
            regularizers_loss = tf.constant(0.0, dtype=dtype)

        # # compute the annulus rate loss
        # if flags.core_loss and v1_neurons > 51978:
        #     v1_annulus_rate_loss = v1_annulus_rate_regularizer(_v1_z, trim)
        #     lm_annulus_rate_loss = lm_annulus_rate_regularizer(_lm_z, trim)
        #     rate_loss = v1_rate_loss + lm_rate_loss + v1_annulus_rate_loss + lm_annulus_rate_loss
        #     tf.print('Annulus rate loss: ', v1_annulus_rate_loss, lm_annulus_rate_loss)
        #     tf.print('Core rate loss: ', v1_rate_loss, lm_rate_loss)
        # else:
        #     rate_loss = v1_rate_loss + lm_rate_loss

        rate_loss = v1_rate_loss + lm_rate_loss

        _aux = dict(rate_loss=rate_loss, voltage_loss=voltage_loss, regularizer_loss=regularizers_loss, osi_dsi_loss=osi_dsi_loss, sync_loss=sync_loss) 
        # _loss = osi_dsi_loss + rate_loss + voltage_loss + regularizers_loss + sync_loss
        _loss = rate_loss + voltage_loss
        tf.print(osi_dsi_loss, rate_loss, voltage_loss, sync_loss) #, interarea_weights_l2_regularizer*100, recurrent_weights_regularizer/100)
        return _out, _p, _loss, _aux, _bkg_noise
    
    @tf.function
    def distributed_roll_out(x, y, w, spontaneous=False, trim=True):
        _out, _p, _loss, _aux, _bkg_noise = strategy.run(roll_out, args=(x, y, w, trim, spontaneous))
        # if output_spikes:
        #     _v1_z, _lm_z = _out[0][0], _out[0][2]
        #     return _v1_z, _lm_z, _bkg_noise
        # else:
        return _out, _p, _loss, _aux, _bkg_noise

    def train_step(_x, _y, _w, spontaneous, trim=True):
        ### Forward propagation of the model
        with tf.GradientTape() as tape:
            _out, _p, _loss, _aux, _ = roll_out(_x, _y, _w, trim=trim, spontaneous=spontaneous)

        grad = tape.gradient(_loss, model.trainable_variables)
        # for g, v in zip(grad, model.trainable_variables):
        #     tf.print(f'Spontaneous {v.name}: ', 'Loss, total_gradients : ', _loss, tf.reduce_sum(tf.math.abs(g)))
        #     _op = optimizer.apply_gradients([(g, v)])

        return _loss, _aux, _out, _p, grad
            
    @tf.function
    def distributed_train_step(x, y, weights, spontaneous, trim):
        # _loss, _aux, _out, _p = strategy.run(train_step, args=(x, y, weights, spontaneous, trim))
        _loss, _aux, _out, _p, grad = train_step(x, y, weights, spontaneous, trim)
        return _loss, _aux, _out, _p, grad

    def split_train_step(_x, _y, _w, _x_spontaneous, trim=True):
        # Run the training step for the spontaneous condition
        _loss_spontaneous, _aux_spontaneous, _out_spontaneous, _, grad_spontaneous = distributed_train_step(_x_spontaneous, _y, _w, True, trim)
        # Run the training step for the non-spontaneous condition
        _loss_gratings, _aux_gratings, _out_gratings, _p, grad_gratings = distributed_train_step(_x, _y, _w, False, trim)
        _loss = _loss_gratings + _loss_spontaneous

        # Accumulate gradients
        average_gradients = [tf.add(g1, g2) / 2.0 for g1, g2 in zip(grad_spontaneous, grad_gratings)]
        # average_gradients = [g / 2.0 for g in accumulated_gradients]
        # print total loss and the total gradients for each variable:
        for g, v in zip(average_gradients, model.trainable_variables):
            tf.print(f'Gratings {v.name}: ', 'Loss, average_gradient : ', _loss, tf.reduce_mean(tf.math.abs(g)))
        # Apply average gradients
        optimizer.apply_gradients(zip(average_gradients, model.trainable_variables))

        ### Backpropagation of the model
        _op = train_accuracy.update_state(_y, _p, sample_weight=_w)
        with tf.control_dependencies([_op]):
            _op = train_loss.update_state(_loss)

        _rate = tf.reduce_mean(tf.concat([_out_gratings[0][0], _out_gratings[0][2], _out_spontaneous[0][0], _out_spontaneous[0][2]], axis=-1))
        with tf.control_dependencies([_op]):                   
            _op = train_firing_rate.update_state(_rate)
        with tf.control_dependencies([_op]):
            _op = train_rate_loss.update_state(_aux_gratings['rate_loss'] + _aux_spontaneous['rate_loss'])
        with tf.control_dependencies([_op]):
            _op = train_voltage_loss.update_state(_aux_gratings['voltage_loss'] + _aux_spontaneous['voltage_loss'])
        with tf.control_dependencies([_op]):
            _op = train_regularizer_loss.update_state(_aux_spontaneous['regularizer_loss'])
        with tf.control_dependencies([_op]):
            _op = train_osi_dsi_loss.update_state(_aux_gratings['osi_dsi_loss'])
        with tf.control_dependencies([_op]):
            _op = train_sync_loss.update_state(_aux_gratings['sync_loss'] + _aux_spontaneous['sync_loss'])    

        tf.print("Train Loss:", train_loss.result(), _loss, ' - ', _y, tf.reduce_sum(tf.cast(_x, tf.float32)))  
        tf.print(model.trainable_variables)

        v1_spikes = _out_gratings[0][0]
        lm_spikes = _out_gratings[0][2]
        v1_spikes_spont = _out_spontaneous[0][0]
        lm_spikes_spont = _out_spontaneous[0][2]

        model_spikes = (v1_spikes, lm_spikes, v1_spikes_spont, lm_spikes_spont)	
        rate_loss = _aux_gratings['rate_loss'] + _aux_spontaneous['rate_loss']
        voltage_loss = _aux_gratings['voltage_loss'] + _aux_spontaneous['voltage_loss']
        regularizers_loss = _aux_gratings['regularizer_loss'] + _aux_spontaneous['regularizer_loss']
        osi_dsi_loss = _aux_gratings['osi_dsi_loss'] + _aux_spontaneous['osi_dsi_loss']
        sync_loss = _aux_gratings['sync_loss'] + _aux_spontaneous['sync_loss']

        return model_spikes, [0., _loss, _rate, rate_loss, voltage_loss, regularizers_loss, osi_dsi_loss, sync_loss]

    @tf.function
    def distributed_split_train_step(x, y, weights, x_spontaneous, trim):
        _out = strategy.run(split_train_step, args=(x, y, weights, x_spontaneous, trim))
        return _out
    
    # @tf.function
    # def distributed_train_step(dist_inputs):
    #     per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
    #     return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
    #                             axis=None)

    def validation_step(_x, _y, _w, output_spikes=True):
        _out, _p, _loss, _aux, _bkg_noise = roll_out(_x, _y, _w)
        tf.print("Validation loss:", _loss)
        _op = val_accuracy.update_state(_y, _p, sample_weight=_w)
        with tf.control_dependencies([_op]):
            _op = val_loss.update_state(_loss)
        _rate = tf.reduce_mean(tf.concat([_out[0][0], _out[0][2]], axis=-1))
        with tf.control_dependencies([_op]):
            _op = val_firing_rate.update_state(_rate)
        with tf.control_dependencies([_op]):
            _op =    val_rate_loss.update_state(_aux['rate_loss'])
        with tf.control_dependencies([_op]):
            _op = val_voltage_loss.update_state(_aux['voltage_loss'])
        with tf.control_dependencies([_op]):
            _op = val_regularizer_loss.update_state(_aux['regularizer_loss'])
        with tf.control_dependencies([_op]):
            _op = val_osi_dsi_loss.update_state(_aux['osi_dsi_loss'])
        with tf.control_dependencies([_op]):
            _op = val_sync_loss.update_state(_aux['sync_loss'])
                        
        # tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])

        if output_spikes:
            _v1_z, _lm_z = _out[0][0], _out[0][2]
            return _v1_z, _lm_z, _bkg_noise

    @tf.function
    def distributed_validation_step(x, y, weights, output_spikes=True):
        if output_spikes:
            return strategy.run(validation_step, args=(x, y, weights, output_spikes))
        else:
            strategy.run(validation_step, args=(x, y, weights))

    ### LGN INPUT ###
    # Define the function that generates the dataset for our task
    def get_dataset_fn(regular=False):
        def _f(input_context):
            _data_set = stim_dataset.generate_drifting_grating_tuning(
                seq_len=flags.seq_len,
                pre_delay=delays[0],
                post_delay=delays[1],
                n_input=flags.n_input,
                regular=regular,
                bmtk_compat=flags.bmtk_compat_lgn,
                rotation=flags.rotation,
                billeh_phase=True,
            ).batch(per_replica_batch_size)
                        
            return _data_set
        return _f

    def get_gray_dataset_fn():
        def _f(input_context):
            _lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
                seq_len=flags.seq_len,
                pre_delay=flags.seq_len,
                post_delay=0,
                n_input=flags.n_input,
                rotation=flags.rotation,
                billeh_phase=True,
                return_firing_rates=True,
            ).batch(per_replica_batch_size)
                        
            return _lgn_firing_rates
        return _f

    # We define the dataset generates function under the strategy scope for a randomly selected orientation       
    # test_data_set = strategy.distribute_datasets_from_function(get_dataset_fn(regular=True))   
    gray_data_set = strategy.distribute_datasets_from_function(get_gray_dataset_fn())
    gray_it = iter(gray_data_set)
    y_spontaneous = tf.constant(0, dtype=tf.float32, shape=(1,1)) 
    w_spontaneous = tf.constant(flags.seq_len, dtype=tf.float32, shape=(1,1))
    spontaneous_lgn_firing_rates = next(iter(gray_data_set))   
    spontaneous_lgn_firing_rates = tf.constant(spontaneous_lgn_firing_rates, dtype=dtype)
    # load LGN spontaneous firing rates 
    spontaneous_prob = 1 - tf.exp(-spontaneous_lgn_firing_rates / 1000.)

    @tf.function(jit_compile=True)
    def generate_spontaneous_spikes(spontaneous_prob):
        x_spontaneous = tf.random.uniform(tf.shape(spontaneous_prob)) < spontaneous_prob
        return x_spontaneous
    
    del gray_data_set, gray_it, spontaneous_lgn_firing_rates

    if flags.spontaneous_training:
        train_data_set = strategy.distribute_datasets_from_function(get_gray_dataset_fn()) 
    else:
        train_data_set = strategy.distribute_datasets_from_function(get_dataset_fn())   

    def reset_state(reset_type='zero', new_state=None):
        if reset_type == 'zero':
            tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)
        elif reset_type == 'gray':
            # Run a gray simulation to get the model state
            tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)
        elif reset_type == 'continue':
            # Continue on the previous model state
            # No action needed, as the state_variables will not be modified
            pass
        else:
            raise ValueError(f"Invalid reset_type: {reset_type}")

    @tf.function
    def distributed_reset_state(reset_type, gray_state=None):
        if reset_type == 'gray':
            if gray_state is None:
                # Generate LGN spikes
                x = generate_spontaneous_spikes(spontaneous_prob)
                tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)    
                _out, _, _, _, _ = distributed_roll_out(x, y_spontaneous, w_spontaneous)
                gray_state = tuple(_out[1:])
                strategy.run(reset_state, args=(reset_type, gray_state))
                return gray_state
            else:
                strategy.run(reset_state, args=(reset_type, gray_state))
        else:
            strategy.run(reset_state, args=(reset_type, zero_state))

    def get_next_chunknum(chunknum, seq_len, direction='up'):
        # get the next chunk number (diviser) for seq_len.
        if direction == 'up':
            chunknum += 1
            # check if it is a valid diviser
            while seq_len % chunknum != 0:
                chunknum += 1
                if chunknum >= seq_len:
                    print('Chunk number reached seq_len')
                    return seq_len
        elif direction == 'down':
            chunknum -= 1
            while seq_len % chunknum != 0:
                chunknum -= 1
                if chunknum <= 1:
                    print('Chunk number reached 1')
                    return 1
        else:
            raise ValueError(f"Invalid direction: {direction}")
        return chunknum

    ############################ TRAINING #############################
    stop = False
    # Initialize your callbacks
    metric_keys = ['train_accuracy', 'train_loss', 'train_firing_rate', 'train_rate_loss', 'train_voltage_loss',
            'train_regularizer_loss', 'train_osi_dsi_loss', 'train_sync_loss', 'val_accuracy', 'val_loss',
            'val_firing_rate', 'val_rate_loss', 'val_voltage_loss', 'val_regularizer_loss', 'val_osi_dsi_loss', 'val_sync_loss']

    callbacks = Callbacks(networks, lgn_inputs, bkg_inputs, model, optimizer, distributed_roll_out, flags, logdir, strategy, 
                        metric_keys, pre_delay=delays[0], post_delay=delays[1], model_variables_init=model_variables_dict,
                        checkpoint=checkpoint, spontaneous_training=flags.spontaneous_training)

    callbacks.on_train_begin()
    chunknum = 1
    max_working_fr = {}   # defined for each chunknum
    n_prev_epochs = flags.run_session * flags.n_epochs
    for epoch in range(n_prev_epochs, n_prev_epochs + flags.n_epochs):
        callbacks.on_epoch_start()  
        # Reset the model state to the gray state  
        gray_state = distributed_reset_state('gray')  
        
        # Load the dataset iterator - this must be done inside the epoch loop
        it = iter(train_data_set)
        # gray_it = iter(gray_data_set)
        
        for step in range(flags.steps_per_epoch):
            # if step > 2:
            #     tf.profiler.experimental.start(logdir=logdir)
            # with tf.profiler.experimental.Trace("Train", step_num=step):
            callbacks.on_step_start()
            # try resetting every iteration
            if flags.reset_every_step:
                distributed_reset_state('gray')
            else:
                distributed_reset_state('gray', gray_state=gray_state)

            x, y, _, w = next(it) # x dtype tf.bool
            # Generate LGN spikes
            x_spontaneous = generate_spontaneous_spikes(spontaneous_prob)
            
            # # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            while True:
                try:
                    x_chunks = tf.split(x, chunknum, axis=1)
                    x_spont_chunks = tf.split(x_spontaneous, chunknum, axis=1)
                    seq_len_local = x.shape[1] // chunknum
                    print('Chunking finished!')
                    for j in range(chunknum):
                        print(j, seq_len_local)
                        x_chunk = x_chunks[j]
                        x_spont_chunk = x_spont_chunks[j]
                        # step_values = distributed_train_step(x_chunk, y, w, x_spont_chunk, trim=chunknum==1, output_metrics=True)
                        model_spikes, step_values = distributed_split_train_step(x_chunk, y, w, x_spont_chunk, trim=chunknum==1)
                        print(model_spikes[0].shape)
                        # step_values = distributed_train_step(x, y, w, trim=chunknum==1)
                    break
                except tf.errors.ResourceExhaustedError as e:
                    print("OOM error occurred!")
                    import gc
                    gc.collect()
                    # increase the chunknum
                    chunknum = get_next_chunknum(chunknum, flags.seq_len, direction='up')
                    tf.config.experimental.reset_memory_stats('GPU:0')
                    print("Increasing chunknum to: ", chunknum)
                    print("BPTT truncation: ", flags.seq_len / chunknum)

            # update max working fr for the chunk num
            current_fr = step_values[2].numpy()
            if chunknum not in max_working_fr:
                max_working_fr[chunknum] = current_fr
            else:
                max_working_fr[chunknum] = max(max_working_fr[chunknum], current_fr)
            # determine if the chunknum should be decreased
            if chunknum > 1:
                chunknum_down = get_next_chunknum(chunknum, flags.seq_len, direction='down')
                if chunknum_down in max_working_fr:
                    if current_fr < max_working_fr[chunknum_down]:
                        chunknum = chunknum_down
                        print("Decreasing chunknum to: ", chunknum)
                        print(current_fr, max_working_fr)
                        print(max_working_fr)
                else:  # data not available, estimate from the current one.
                    fr_ratio = current_fr / max_working_fr[chunknum]
                    chunknum_ratio = chunknum_down / chunknum
                    print(current_fr, max_working_fr, fr_ratio, chunknum_ratio)
                    if fr_ratio < chunknum_ratio:  # potentially good to decrease
                        chunknum = chunknum_down
                        print("Tentatively decreasing chunknum to: ", chunknum)

            callbacks.on_step_end(step_values, y, verbose=True)
            # print('Max_rates: ', chunknum, current_fr, max_working_fr)

        # tf.profiler.experimental.stop() 

        # ## VALIDATION AFTER EACH EPOCH 
        # test_it = iter(test_data_set)           
        # test_it = it
        # for step in range(flags.val_steps):
        #     x, y, _, w = next(it)
        #     # Generate LGN spikes
        #     x_spontaneous = generate_spontaneous_spikes(spontaneous_prob)

        #     gray_state = distributed_reset_state('gray')  
        #     distributed_reset_state('gray', gray_state=gray_state)

        #     # v1_spikes_spont, lm_spikes_spont, bkg_noise = distributed_validation_step(x_spontaneous, y, w, output_spikes=True) 
        #     # v1_spikes, lm_spikes, _ = distributed_validation_step(x, y, w, output_spikes=True)
        #     _out, _, _, _, bkg_noise = distributed_roll_out(x_spontaneous, y_spontaneous, w_spontaneous)
        #     v1_spikes_spont, lm_spikes_spont = _out[0][0], _out[0][2]
        #     _out, _, _, _, _ = distributed_roll_out(x, y, w)
        #     v1_spikes, lm_spikes = _out[0][0], _out[0][2]
        
        train_values = [a.result().numpy() for a in [train_accuracy, train_loss, train_firing_rate, 
                                                    train_rate_loss, train_voltage_loss, train_regularizer_loss, 
                                                    train_osi_dsi_loss, train_sync_loss]]
        val_values = train_values # for our case, training set and testing set are undistinguishible
        # val_values = [a.result().numpy() for a in [val_accuracy, val_loss, val_firing_rate, 
        #                                         val_rate_loss, val_voltage_loss, val_regularizer_loss, 
                                                # val_osi_dsi_loss, val_sync_loss]]
        metric_values = train_values + val_values

        # if the model train loss is minimal, save the model.
        v1_spikes = model_spikes[0]
        lm_spikes = model_spikes[1]
        v1_spikes_spont = model_spikes[2]
        lm_spikes_spont = model_spikes[3]

        bkg_noise = None

        stop = callbacks.on_epoch_end(x, v1_spikes, lm_spikes, y, metric_values, bkg_noise=bkg_noise, verbose=True, 
                                      x_spont=x_spontaneous, v1_spikes_spont=v1_spikes_spont, lm_spikes_spont=lm_spikes_spont)    
        
        if stop:
            break
        
        # Reset the metrics for the next epoch
        reset_train_metrics()
        reset_validation_metrics()

    # # print the traces of the different graph functions
    # print(distributed_reset_state.pretty_printed_concrete_signatures())
    # print(distributed_train_step.pretty_printed_concrete_signatures())
    # print(distributed_split_train_step.pretty_printed_concrete_signatures())
    # print(distributed_roll_out.pretty_printed_concrete_signatures())
    # print(generate_spontaneous_spikes.pretty_printed_concrete_signatures())

    normalizers = {'v1_ema': v1_ema.numpy(), 'lm_ema': lm_ema.numpy()}
    callbacks.on_train_end(metric_values, normalizers=normalizers)


if __name__ == '__main__':
    _data_dir = 'GLIF_network'
    _results_dir = 'Simulation_results'

    absl.app.flags.DEFINE_string('task_name', 'drifting_gratings_firing_rates_distr' , '')
    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('interarea_weight_distribution', 'billeh_weights', '')
    # absl.app.flags.DEFINE_string('interarea_weight_distribution', 'zero_weights', '')
    absl.app.flags.DEFINE_string('delays', '100,0', '')
    absl.app.flags.DEFINE_string('optimizer', 'adam', '')

    absl.app.flags.DEFINE_float('learning_rate', .001, '')
    absl.app.flags.DEFINE_float('rate_cost', 100., '')
    absl.app.flags.DEFINE_float('voltage_cost', 1., '')
    absl.app.flags.DEFINE_float('osi_cost', 1., '')
    absl.app.flags.DEFINE_string('osi_loss_method', 'crowd_osi', '')
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

    absl.app.flags.DEFINE_integer('n_runs', 1, '')
    absl.app.flags.DEFINE_integer('run_session', 0, '')
    absl.app.flags.DEFINE_integer('n_epochs', 20, '')
    absl.app.flags.DEFINE_integer('osi_dsi_eval_period', 50, '') # number of epochs for osi/dsi evaluation if n_runs = 1
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('v1_neurons', 10, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('lm_neurons', 10, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('steps_per_epoch', 10, '')# EA and garret dose not need this many but pure classification needs 781 = int(50000/64)
    absl.app.flags.DEFINE_integer('val_steps', 1, '')# EA and garret dose not need this many but pure classification needs 156 = int(10000/64)
    # number of LGN filters in visual space (input population)
    absl.app.flags.DEFINE_integer('n_input', 17400, '')
    absl.app.flags.DEFINE_integer('seq_len', 600, '')
    absl.app.flags.DEFINE_integer('n_cues', 3, '')
    absl.app.flags.DEFINE_integer('recall_duration', 40, '')
    absl.app.flags.DEFINE_integer('cue_duration', 40, '')
    absl.app.flags.DEFINE_integer('interval_duration', 40, '')
    absl.app.flags.DEFINE_integer('examples_in_epoch', 32, '')
    absl.app.flags.DEFINE_integer('validation_examples', 16, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_integer('neurons_per_output', 16, '')

    absl.app.flags.DEFINE_boolean('float16', False, '')
    absl.app.flags.DEFINE_boolean('caching', True, '')
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('core_loss', False, '')
    absl.app.flags.DEFINE_boolean('hard_reset', False, '')
    absl.app.flags.DEFINE_boolean('disconnect_lm_L6_inhibition', False, '')
    absl.app.flags.DEFINE_boolean('disconnect_v1_lm_L6_excitatory_projections', False, '')
    absl.app.flags.DEFINE_boolean('realistic_neurons_ratio', True, '')
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
    absl.app.flags.DEFINE_string("rotation", "ccw", "")

    absl.app.flags.DEFINE_string('ckpt_dir', '', '')

    absl.app.run(main)

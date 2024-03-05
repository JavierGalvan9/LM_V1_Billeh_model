import os
import sys
import absl
import json
import numpy as np
import tensorflow as tf
import pickle as pkl
import matplotlib.pyplot as plt
from packaging import version

if version.parse(tf.__version__) < version.parse("2.4.0"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision

from Model_utils import load_sparse, models, other_billeh_utils, stim_dataset, toolkit
from Model_utils.plotting_utils import InputActivityFigure, RasterPlot, LaminarPlot, LGN_sample_plot
from general_utils import file_management
import Model_utils.loss_functions as losses
from Model_utils.callbacks import Callbacks

from time import time
import ctypes.util


# Define the environment variables for optimal GPU performance
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# TF_GPU_ALLOCATOR=cuda_malloc_async

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
    np.random.seed(flags.seed)
    tf.random.set_seed(flags.seed)

    # Select the connectivity rules in the network
    v1_to_lm_neurons_ratio = 7.010391285652859

    if flags.realistic_neurons_ratio and flags.lm_neurons is None:
        n_neurons = {'v1': flags.v1_neurons,
                     'lm': int(flags.v1_neurons/v1_to_lm_neurons_ratio)}
    else:
        n_neurons = {'v1': flags.v1_neurons,
                     'lm': flags.lm_neurons}

    # Get the neurons of each column of the network
    v1_neurons = n_neurons['v1']
    lm_neurons = n_neurons['lm']

    # Save the configuration of the model
    flag_str = f'v1_{v1_neurons}_lm_{lm_neurons}'
    for name, value in flags.flag_values_dict().items():
        if value != flags[name].default and name in ['learning_rate', 'rate_cost', 'voltage_cost', 'osi_cost', 'temporal_f', 'n_input', 'seq_len']:
            print(name, value, flags[name].default)
            flag_str += f'_{name}_{value}'
    # Define flag string as the second part of results_path
    results_dir = f'{flags.results_dir}/{flag_str}'
    os.makedirs(results_dir, exist_ok=True)
    print('Simulation results path: ', results_dir)
    # Save the flags configuration in a JSON file
    with open(os.path.join(results_dir, 'flags_config.json'), 'w') as fp:
        json.dump(flags.flag_values_dict(), fp)

    # Generate a ticker for the current simulation
    sim_name = toolkit.get_random_identifier('b_')
    logdir = os.path.join(results_dir, sim_name)
    print(f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')

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

    per_replica_batch_size = flags.batch_size
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    print(f'Global batch size: {global_batch_size}\n')

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
    networks, lgn_inputs, bkg_inputs = load_fn(flags, n_neurons)
    print(f"Model files loading: {time()-t0:.2f} seconds\n")

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
            dtype=tf.float32, 
            input_weight_scale=flags.input_weight_scale,
            interarea_weight_scale=1., 
            dampening_factor=flags.dampening_factor, 
            gauss_std=flags.gauss_std, 
            lr_scale=flags.lr_scale,
            train_recurrent_v1=flags.train_recurrent_v1, 
            train_recurrent_lm=flags.train_recurrent_lm, 
            train_input=flags.train_input, 
            train_interarea=flags.train_interarea,
            train_noise=flags.train_noise,
            batch_size=flags.batch_size, 
            pseudo_gauss=flags.pseudo_gauss, 
            use_state_input=True, 
            return_state=True,
            hard_reset=flags.hard_reset,
            add_rate_metric=True, 
            max_delay=5, 
            # output_completed_valid_from_time=120, 
            # output_abstract_valid_from_time=100,
            )

        del lgn_inputs, bkg_inputs

        model.build((flags.batch_size, flags.seq_len, flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")

        ### BUILD THE LOSS AND REGULARIZER FUNCTIONS ###
        # Create rate and voltage regularizers
        delays = [int(a) for a in flags.delays.split(',') if a != '']
        if flags.core_loss:
            v1_core_radius = 200
            v1_core_mask = other_billeh_utils.isolate_core_neurons(networks['v1'], radius=v1_core_radius, data_dir=flags.data_dir)
            v1_core_mask = tf.constant(v1_core_mask, dtype=tf.bool)
        else:
            v1_core_mask = None

        rsnn_layer = model.get_layer('rsnn')
        lm_to_v1_weight_regularizer = losses.StiffRegularizer(flags.recurrent_weight_regularization,
                                                            rsnn_layer.cell.v1.interarea_weight_values['lm'])
        v1_to_lm_weight_regularizer = losses.StiffRegularizer(flags.recurrent_weight_regularization,
                                                            rsnn_layer.cell.lm.interarea_weight_values['v1'])

        v1_rate_distribution_regularizer = losses.SpikeRateDistributionTarget(networks['v1'], flags.rate_cost, pre_delay=delays[0], post_delay=delays[1], 
                                                                            data_dir=flags.data_dir, area='v1', core_mask=None, seed=flags.seed, dtype=dtype)
        lm_rate_distribution_regularizer = losses.SpikeRateDistributionTarget(networks['lm'], flags.rate_cost, pre_delay=delays[0], post_delay=delays[1], 
                                                                            data_dir=flags.data_dir, area='lm', core_mask=None, seed=flags.seed, dtype=dtype)
        rate_loss = v1_rate_distribution_regularizer(rsnn_layer.output[0][0]) + lm_rate_distribution_regularizer(rsnn_layer.output[0][2])

        v1_voltage_regularizer = losses.VoltageRegularization(rsnn_layer.cell.v1, area='v1', voltage_cost=flags.voltage_cost, dtype=dtype, core_mask=None)
        lm_voltage_regularizer = losses.VoltageRegularization(rsnn_layer.cell.lm, area='lm', voltage_cost=flags.voltage_cost, dtype=dtype, core_mask=None)
        voltage_loss = v1_voltage_regularizer(rsnn_layer.output[0][1]) + lm_voltage_regularizer(rsnn_layer.output[0][3])

        v1_tuning_angles = tf.constant(networks['v1']['tuning_angle'], dtype=dtype)
        lm_tuning_angles = tf.constant(networks['lm']['tuning_angle'], dtype=dtype)
        v1_OSI_Loss = losses.OrientationSelectivityLoss(v1_tuning_angles, osi_cost=flags.osi_cost, area='v1',
                                                        pre_delay=delays[0], post_delay=delays[1], 
                                                        dtype=dtype, core_mask=None)
        lm_OSI_Loss = losses.OrientationSelectivityLoss(lm_tuning_angles, osi_cost=flags.osi_cost, area='lm',
                                                        pre_delay=delays[0], post_delay=delays[1], 
                                                        dtype=dtype, core_mask=None)
        osi_loss = v1_OSI_Loss(rsnn_layer.output[0][0], tf.constant(0, dtype=tf.float32, shape=(1,))) \
                + lm_OSI_Loss(rsnn_layer.output[0][2], tf.constant(0, dtype=tf.float32, shape=(1,))) # this is just a placeholder


        # Load the firing rates distribution as a regularizer that we have and generate target firing rates for every neuron type
        # with open(os.path.join(flags.data_dir, 'np_gratings_firing_rates.pkl'), 'rb') as f:
        #     target_firing_rates = pkl.load(f) # they are in Hz and divided by 1000 to make it in kHz and match the dt = 1 ms

        # for i, (key, value) in enumerate(target_firing_rates.items()):
        #     # identify tne ids that are included in value["ids"]
        #     neuron_ids = np.where(np.isin(networks['V1']["node_type_ids"], value["ids"]))[0]
        #     neuron_ids = tf.cast(neuron_ids, dtype=tf.int32)
        #     type_n_neurons = len(neuron_ids)
        #     sorted_target_rates = models.sample_firing_rates(value["rates"], type_n_neurons, flags.seed)
        #     target_firing_rates[key]['sorted_target_rates'] = tf.cast(sorted_target_rates, dtype=tf.float32) 

        model.add_loss(rate_loss)
        model.add_loss(voltage_loss)
        model.add_loss(osi_loss)
        model.add_metric(rate_loss, name='rate_loss')
        model.add_metric(voltage_loss, name='voltage_loss')
        model.add_metric(osi_loss, name='osi_loss')

        # prediction_layer = model.get_layer('prediction')
        abstract_layer = model.get_layer('abstract_output')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=[rsnn_layer.output, model.output[0], model.output[1]])

        # Loss from Guozhang classification task (unused in our case)
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        #     from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        # def compute_loss(_l, _p, _w):
        #     per_example_loss = loss_object(_l, _p, sample_weight=_w) * strategy.num_replicas_in_sync / tf.reduce_sum(_w)
        #     rec_weight_loss = rec_weight_regularizer(rsnn_layer.cell.recurrent_weight_values)
        #     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size) + rec_weight_loss
        
        optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)  
        # These "dummy" zeros are injected to the models membrane voltage
        # Provides the opportunity to compute gradients wrt. membrane voltages at all time steps
        # Not important for general use
        n_total_neurons = v1_neurons + lm_neurons
        zero_state = rsnn_layer.cell.zero_state_multi_areas(flags.batch_size, np.float32)
        state_variables = tf.nest.map_structure(lambda a: tf.Variable(
            a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        ), zero_state)
        optimizer.build(model.trainable_variables)  # Add this line to build the optimizer with the trainable variables

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
        train_osi_loss = tf.keras.metrics.Mean()
        val_osi_loss = tf.keras.metrics.Mean()

        def reset_train_metrics():
            train_loss.reset_states(), train_accuracy.reset_states(), train_firing_rate.reset_states()
            train_rate_loss.reset_states(), train_voltage_loss.reset_states(), train_osi_loss.reset_states()

        def reset_validation_metrics():
            val_loss.reset_states(), val_accuracy.reset_states(), val_firing_rate.reset_states()
            val_rate_loss.reset_states(), val_voltage_loss.reset_states(), val_osi_loss.reset_states()

        def roll_out(_x, _y, _w, output_spikes=False):
            _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
            dummy_zeros = tf.zeros((flags.batch_size, flags.seq_len, n_total_neurons), dtype)
            _out, _p, _ = extractor_model((_x, dummy_zeros, _initial_state))

            _v1_z, _v1_v, _lm_z, _lm_v = _out[0]
            # update state_variables with the new model state
            new_state = tuple(_out[1:])
            tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

            v1_voltage_loss = v1_voltage_regularizer(_v1_v)
            lm_voltage_loss = lm_voltage_regularizer(_lm_v)
            voltage_loss = v1_voltage_loss + lm_voltage_loss

            v1_rate_loss = v1_rate_distribution_regularizer(_v1_z)
            lm_rate_loss = lm_rate_distribution_regularizer(_lm_z)
            rate_loss = v1_rate_loss + lm_rate_loss

            v1_osi_loss = v1_OSI_Loss(_v1_z, _y)
            lm_osi_loss = lm_OSI_Loss(_lm_z, _y)
            osi_loss = v1_osi_loss + lm_osi_loss

            lm_to_v1_weights_l2_regularizer = lm_to_v1_weight_regularizer(rsnn_layer.cell.v1.interarea_weight_values['lm'])
            v1_to_lm_weights_l2_regularizer = v1_to_lm_weight_regularizer(rsnn_layer.cell.lm.interarea_weight_values['v1'])
            weights_l2_regularizer = lm_to_v1_weights_l2_regularizer + v1_to_lm_weights_l2_regularizer

            _aux = dict(rate_loss=rate_loss, voltage_loss=voltage_loss, osi_loss=osi_loss)
            _loss = osi_loss + rate_loss + voltage_loss + weights_l2_regularizer

            return _out, _p, _loss, _aux

        @tf.function
        def distributed_roll_out(x, y, w, output_spikes=True):
            _out, _p, _loss, _aux = strategy.run(roll_out, args=(x, y, w))
            if output_spikes:
                _v1_z, _lm_z = _out[0][0], _out[0][2]
                return _v1_z, _lm_z
            else:
                return _out, _p, _loss, _aux     


        def train_step(_x, _y, _w):
            ### Forward propagation of the model
            with tf.GradientTape() as tape:
                _out, _p, _loss, _aux = roll_out(_x, _y, _w)

            ### Backpropagation of the model
            _op = train_accuracy.update_state(_y, _p, sample_weight=_w)
            with tf.control_dependencies([_op]):
                _op = train_loss.update_state(_loss)
                _rate = tf.reduce_mean(_out[0][0])
                _op = train_firing_rate.update_state(_rate)
                _op = train_rate_loss.update_state(_aux['rate_loss'])
                _op = train_voltage_loss.update_state(_aux['voltage_loss'])
                _op = train_osi_loss.update_state(_aux['osi_loss'])
            
                
            # with tf.control_dependencies([_op]):
            #     _op = train_voltage_loss.update_state(_aux['voltage_loss'])

            grad = tape.gradient(_loss, model.trainable_variables)
            for g, v in zip(grad, model.trainable_variables):
                # tf.print(f'{v.name} optimization')
                # tf.print('Loss, total_gradients : ', _loss, tf.reduce_sum(tf.math.abs(g)))
                with tf.control_dependencies([_op]):
                    _op = optimizer.apply_gradients([(g, v)])
                    
        @tf.function
        def distributed_train_step(x, y, weights):
            strategy.run(train_step, args=(x, y, weights))

        # @tf.function
        # def distributed_train_step(dist_inputs):
        #     per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
        #     return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
        #                             axis=None)

        def validation_step(_x, _y, _w, output_spikes=True):
            _out, _p, _loss, _aux = roll_out(_x, _y, _w)
            _op = val_accuracy.update_state(_y, _p, sample_weight=_w)
            with tf.control_dependencies([_op]):
                _op = val_loss.update_state(_loss)
                _rate = tf.reduce_mean(_out[0][0])
                _op = val_firing_rate.update_state(_rate)
                _op = val_rate_loss.update_state(_aux['rate_loss'])
                _op = val_voltage_loss.update_state(_aux['voltage_loss'])
                _op = val_osi_loss.update_state(_aux['osi_loss'])
                
            # tf.nest.map_structure(lambda _a, _b: _a.assign(_b), list(state_variables), _out[1:])

            if output_spikes:
                _v1_z, _lm_z = _out[0][0], _out[0][2]
                return _v1_z, _lm_z


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
                ).batch(per_replica_batch_size)
                            
                return _data_set
            return _f

        def get_gray_dataset_fn():
            def _f(input_context):
                _gray_data_set = stim_dataset.generate_drifting_grating_tuning(
                    seq_len=flags.seq_len,
                    pre_delay=flags.seq_len,
                    post_delay=0,
                    n_input=flags.n_input,
                ).batch(per_replica_batch_size)
                            
                return _gray_data_set
            return _f
    
        # We define the dataset generates function under the strategy scope for a randomly selected orientation       
        # test_data_set = strategy.distribute_datasets_from_function(get_dataset_fn(regular=True))
        train_data_set = strategy.distribute_datasets_from_function(get_dataset_fn())      
        gray_data_set = strategy.distribute_datasets_from_function(get_gray_dataset_fn())

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

        def distributed_reset_state(reset_type, gray_state=None):
            if reset_type == 'gray':
                if gray_state is None:
                    gray_it = next(iter(gray_data_set))
                    x, y, _, w = gray_it
                    tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)    
                    _out, _p, _loss, _aux = distributed_roll_out(x, y, w, output_spikes=False)
                    gray_state = tuple(_out[1:])
                    strategy.run(reset_state, args=(reset_type, gray_state))
                    return gray_state
                else:
                    strategy.run(reset_state, args=(reset_type, gray_state))

            else:
                strategy.run(reset_state, args=(reset_type, zero_state))

        
        ############################ TRAINING #############################
        stop = False
        # Initialize your callbacks
        metric_keys = ['train_accuracy', 'train_loss', 'train_firing_rate', 'train_rate_loss',
                'train_voltage_loss', 'train_osi_loss', 'val_accuracy', 'val_loss',
                'val_firing_rate', 'val_rate_loss', 'val_voltage_loss', 'val_osi_loss']
        delays = [int(a) for a in flags.delays.split(',') if a != '']
        callbacks = Callbacks(model, optimizer, distributed_roll_out, networks, flags, logdir, strategy, 
                            metric_keys, pre_delay=delays[0], post_delay=delays[1])

        callbacks.on_train_begin()
        for epoch in range(flags.n_epochs):
            callbacks.on_epoch_start()  
            # Reset the model state to the gray state    
            gray_state = distributed_reset_state('gray')  
            
            # Load the dataset iterator - this must be done inside the epoch loop
            it = iter(train_data_set)

            # tf.profiler.experimental.start(logdir=logdir)
            for step in range(flags.steps_per_epoch):
                callbacks.on_step_start()
                distributed_reset_state('gray', gray_state=gray_state)
                x, y, _, w = next(it) # x dtype tf.bool
                # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
                distributed_train_step(x, y, w)
                train_values = [a.result().numpy() for a in [train_accuracy, train_loss, train_firing_rate, 
                                                            train_rate_loss, train_voltage_loss, train_osi_loss]]

                callbacks.on_step_end(train_values, y, verbose=False)

            # tf.profiler.experimental.stop() 

            # ## VALIDATION AFTER EACH EPOCH
                
            # test_it = iter(test_data_set)
            test_it = it
            for step in range(flags.val_steps):
                x, y, _, w = next(test_it)
                distributed_reset_state('gray', gray_state=gray_state)
                v1_spikes, lm_spikes = distributed_validation_step(x, y, w, output_spikes=True) 

            val_values = [a.result().numpy() for a in [val_accuracy, val_loss, val_firing_rate, 
                                                    val_rate_loss, val_voltage_loss, val_osi_loss]]
            metric_values = train_values + val_values

            # saving model
            print('Saving model...')
            model.save('my_model.keras', save_format='h5')

            # if the model train loss is minimal, save the model.
            stop = callbacks.on_epoch_end(x, v1_spikes, lm_spikes, y, metric_values)

            

            if stop:
                break
            
            # Reset the metrics for the next epoch
            reset_train_metrics()
            reset_validation_metrics()

        callbacks.on_train_end(metric_values)


if __name__ == '__main__':
    _data_dir = 'GLIF_network'
    _results_dir = 'Simulation_results'

    absl.app.flags.DEFINE_string('task_name', 'drifting_gratings_firing_rates_distr' , '')
    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('interarea_weight_distribution', 'billeh_like', '')
    absl.app.flags.DEFINE_string('delays', '100,0', '')

    absl.app.flags.DEFINE_float('learning_rate', .01, '')
    absl.app.flags.DEFINE_float('rate_cost', 100., '')
    absl.app.flags.DEFINE_float('voltage_cost', .00001, '')
    absl.app.flags.DEFINE_float('osi_cost', 1., '')
    absl.app.flags.DEFINE_float('dampening_factor', .1, '')
    absl.app.flags.DEFINE_float('recurrent_dampening_factor', .5, '')
    absl.app.flags.DEFINE_float('input_weight_scale', 1., '')
    absl.app.flags.DEFINE_float('gauss_std', .3, '')
    absl.app.flags.DEFINE_float('recurrent_weight_regularization', 0., '')
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    absl.app.flags.DEFINE_float('input_f0', 0.2, '')
    absl.app.flags.DEFINE_float('E4_weight_factor', 1., '')
    absl.app.flags.DEFINE_float('temporal_f', 2., '')
    absl.app.flags.DEFINE_float('max_time', -1, '')

    absl.app.flags.DEFINE_integer('n_epochs', 20, '')
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('v1_neurons', 10, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('lm_neurons', None, '')  # -1 to take all neurons
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
    absl.app.flags.DEFINE_boolean('train_interarea', True, '')
    absl.app.flags.DEFINE_boolean('train_noise', False, '')
    # absl.app.flags.DEFINE_boolean('train_recurrent_per_type', False, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', True, '')
    # absl.app.flags.DEFINE_boolean('hard_only', False, '')
    absl.app.flags.DEFINE_boolean('visualize_test', False, '')
    absl.app.flags.DEFINE_boolean('pseudo_gauss', False, '')
    absl.app.flags.DEFINE_boolean("bmtk_compat_lgn", True, "")
    absl.app.flags.DEFINE_boolean("average_grad_for_cell_type", False, "")

    absl.app.run(main)

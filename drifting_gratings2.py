import os
import sys
import absl
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from packaging import version

if version.parse(tf.__version__) < version.parse("2.4.0"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision

from Model_utils import load_sparse, models, other_billeh_utils, stim_dataset
from Model_utils.plotting_utils import InputActivityFigure, RasterPlot, LaminarPlot, LGN_sample_plot
from general_utils import file_management
from Model_utils.model_metrics_analysis import DirtyAnalysis
from time import time
import ctypes.util


# Define the environment variables for optimal GPU performance
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    # simulation_results_path = f'{flags.save_dir}/v1_{v1_neurons}_lm_{lm_neurons}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}_interarea_weight_distribution_{flags.interarea_weight_distribution}'
    flag_str = f'v1_{v1_neurons}_lm_{lm_neurons}'
    for name, value in flags.flag_values_dict().items():
        if value != flags[name].default and name not in ['save_dir', 'v', 'verbosity', 'n_simulations', 'caching', 'v1_neurons', 'lm_neurons', 'gratings_orientation', 'gratings_frequency']:
            print(name, value, flags[name].default)
            flag_str += f'_{name}_{value}'
    # Define flag string as the second part of results_path
    results_path = f'{flags.save_dir}/{flag_str}'
    simulation_results_path = os.path.join(results_path, f'orien_{flags.gratings_orientation}_freq_{flags.gratings_frequency}')
    print('Simulation results path: ', simulation_results_path)
   
    # simul ation_results_path = f'{flags.save_dir}/v1_{v1_neurons}_lm_{lm_neurons}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}_L4_weight_factor_{flags.L4_weight_factor}'
    # simulation_results_path = os.path.join(simulation_results_path, f'orien_{str(flags.gratings_orientation)}_freq_{str(flags.gratings_frequency)}')
    os.makedirs(simulation_results_path, exist_ok=True)
    with open(os.path.join(simulation_results_path, 'flags_config.json'), 'w') as fp:
        json.dump(flags.flag_values_dict(), fp)

    # Can be used to try half precision training
    if flags.float16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        dtype = tf.float16
    else:
        dtype = tf.float32

    # Define 2 outputs that correspond to having more cues top or bottom
    # Note that two different output conventions can be used:
    # 1) Linear readouts from all neurons in the model (softmax)
    # 2) Selecting a population of neurons that report a binary decision
    # with high firing rate (flag --neuron_output)
    n_output = 2

    # Load data of Billeh et al. (2020) and select appropriate number of neurons and inputs
    # Create the v1-lm model

    # Choose only the flags that 

    # input_population, networks, bkg_weights, n_input, n_abstract_output, n_completed_output = load_fn(
    #     flags, interarea_connectivity, n_neurons)
    if flags.caching:
        load_fn = load_sparse.cached_load_billeh
    else:
        load_fn = load_sparse.load_billeh

    networks, lgn_inputs, bkg_inputs = load_fn(flags, n_neurons)

    input_weight_scale = 1.

    ### LGN INPUT ###
    
    # lgn_firing_rates_filename = f"orientation_{str(flags.gratings_orientation)}&TF_{str(float(flags.gratings_frequency))}&SF_0.04&reverse_False&init_screen_dur_1.0&visual_flow_dur_1.0&end_screen_dur_1.0&min_value_-1&max_value_1&contrast_0.8&dt_0.001&height_120&width_240&init_gray_screen_False&end_gray_screen_False.lzma"
    lgn_firing_rates_filename = f"orientation_{str(flags.gratings_orientation)}&TF_{str(float(flags.gratings_frequency))}&SF_0.04&reverse_False&init_screen_dur_0.5&visual_flow_dur_2.5&end_screen_dur_0.0&min_value_-1&max_value_1&contrast_0.8&dt_0.001&height_120&width_240&init_gray_screen_True&end_gray_screen_False.lzma"
    with open(os.path.join(flags.data_dir, "input", "Drifting_gratings", lgn_firing_rates_filename), "rb") as f:
        firing_rates = file_management.load_lzma(f)
    # firing_rates are the probability of spikes/seconds so we convert that to spike/ms
    # then, if random number in 0-1 is lower that firing_rate in spike/ms, there is a
    # neuron spike at that ms
    # firing_rates = firing_rates[None, 500: flags.seq_len + 500]  # (1,2500,17400)
    firing_rates = firing_rates[None, :]

    # Create data_path to save simulation data 
    data_path = os.path.join(simulation_results_path, 'Data')
    os.makedirs(data_path, exist_ok=True)

    # Build the model
    model = models.create_model(
        networks, 
        lgn_inputs, 
        bkg_inputs, 
        seq_len=flags.seq_len,
        n_input=flags.n_input, 
        dtype=tf.float32, 
        input_weight_scale=input_weight_scale,
        interarea_weight_scale=1., 
        dampening_factor=flags.dampening_factor, 
        gauss_std=flags.gauss_std, 
        lr_scale=flags.lr_scale,
        train_recurrent_v1=flags.train_recurrent_v1, 
        train_recurrent_lm=flags.train_recurrent_lm, 
        train_input=flags.train_input, 
        train_interarea=flags.train_interarea,
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

    model.build((flags.batch_size, flags.seq_len, flags.n_input))

    # Extract outputs of intermediate keras layers to get access to
    # spikes and membrane voltages of the model
    rsnn_layer = model.get_layer('rsnn')
    # prediction_layer = model.get_layer('prediction')
    abstract_layer = model.get_layer('abstract_output')

    # Create a model that returns the variables of interest as first output, and the abstract output and completed output in second place
    extractor_model = tf.keras.Model(inputs=model.inputs,
                                     outputs=[rsnn_layer.output, model.output[0], model.output[1]])

    # Loss used for training (evidence accumulation is a classification task)
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
    #                                                             reduction=tf.keras.losses.Reduction.SUM)

    # These "dummy" zeros are injected to the models membrane voltage
    # Provides the opportunity to compute gradients wrt. membrane voltages at all time steps
    # Not important for general use
    zero_state = rsnn_layer.cell.zero_state_multi_areas(flags.batch_size, np.float32)
    n_total_neurons = n_neurons['v1'] + n_neurons['lm']
    dummy_zeros = tf.zeros((flags.batch_size, flags.seq_len, n_total_neurons), dtype)
    inputs = tf.zeros((flags.batch_size, flags.seq_len, flags.n_input), dtype)
    state = zero_state

    keys = ["z", "v", "z_lgn"] # "input_current", "recurrent_current", "bottom_up_current"
    
    # Select the dtype of the data saved
    if flags.save_float16:
        save_dtype = np.float16
    else:
        save_dtype = np.float32
    data_path = os.path.join(simulation_results_path, "Data")

    SimulationDataHDF5 = other_billeh_utils.SaveSimDataHDF5(
        flags, keys, data_path, networks, n_neurons, v1_to_lm_neurons_ratio, save_core_only=True, dtype=save_dtype
    )

    time_per_sim = 0
    time_per_save = 0
    for trial in range(0, flags.n_simulations):
        print('{trial}:new trial'.format(trial=trial))
        inputs = (np.random.uniform(size=inputs.shape,
                                    low=0., high=1.) < firing_rates * .001).astype(np.float32)
        # Simulate the model for one stimulus sequence
        t0 = time()
        out = extractor_model((inputs, dummy_zeros, state))
        time_per_sim += time() - t0
        print('Simulation time: ', time() - t0)

        # Extract spikes and membrane voltages
        t0 = time()
        network_outputs = out[0][0]
        v1_spikes, v1_voltage, lm_spikes, lm_voltage = network_outputs

        # Save simulation data
        simulation_data = {
            "v1": {
                "z": v1_spikes,
                # "v": v1_voltage
            },
            "lm": {
                "z": lm_spikes,
                # "v": lm_voltage
            },
            "LGN": {
                "z_lgn": inputs
            }
        }
        
        SimulationDataHDF5(simulation_data, trial)
        time_per_save += time() - t0

        # Reset the model state to the last state of the previous simulation
        state = out[0][1:]

    # Save the simulation metadata  
    time_per_sim /= flags.n_simulations
    time_per_save /= flags.n_simulations
    metadata_path = os.path.join(data_path, 'Simulation stats')
    with open(metadata_path, 'w') as out_file:
        out_file.write(f'Consumed time per simulation: {time_per_sim}\n')
        out_file.write(f'Consumed time saving: {time_per_save}\n')

    # Plot last trial raster plot
    raster_filename = 'Raster_plot.png'
    image_path = os.path.join(simulation_results_path, 'Images general')
    os.makedirs(image_path, exist_ok=True)
    graph = InputActivityFigure(networks, flags.data_dir, image_path,
                                filename=raster_filename, frequency=flags.gratings_frequency,
                                stimuli_init_time=500, stimuli_end_time=3000)
    graph(inputs, v1_spikes, lm_spikes)

    # # Plot last trial LGN sample plot
    # LGN_units = LGN_sample_plot(firing_rates, inputs, stimuli_init_time=0.5,
    #                             stimuli_end_time=flags.seq_len/1000, images_dir=image_path, n_samples=2)
    # LGN_units()

    # # Dirty boxplot the firing rates
    # v1_dirty_analysis = DirtyAnalysis(flags, networks['v1'], simulation_results_path, area='v1', 
    #                                   drifting_gratings_init=500, drifting_gratings_end=3000, 
    #                                   path=results_path, skip_first_simulation=True)
    # v1_dirty_analysis.plot_tuning_curves()
    # v1_dirty_analysis.plot_boxplots()

    # lm_dirty_analysis = DirtyAnalysis(flags, networks['lm'], simulation_results_path, area='lm', 
    #                                   drifting_gratings_init=500, drifting_gratings_end=3000, 
    #                                   path=results_path, skip_first_simulation=True)
    # lm_dirty_analysis.plot_tuning_curves()
    # lm_dirty_analysis.plot_boxplots()
                        

    ### TRAINING ###

    # # Load a typical distribution of firing rates to which the model is regularized to
    # # during training
    # with open(os.path.join(flags.data_dir, 'garrett_firing_rates.pkl'), 'rb') as f:
    #     firing_rates = pkl.load(f)
    # sorted_firing_rates = np.sort(firing_rates)
    # percentiles = (np.arange(
    #     firing_rates.shape[-1]) + 1).astype(np.float32) / firing_rates.shape[-1]
    # rate_rd = np.random.RandomState(seed=3000)
    # x_rand = rate_rd.uniform(size=flags.neurons)
    # target_firing_rates = np.sort(
    #     np.interp(x_rand, percentiles, sorted_firing_rates))

    # # ---
    # # Training disrupts the firing properties of the model
    # # To counteract, two types of regularizations are used
    # # 1) Firing rate regularization keeps the distribution of the firing rates close
    # # to the previously loaded distribution of firing rates
    # rate_distribution_regularizer = models.SpikeRateDistributionRegularization(
    #     target_firing_rates, flags.rate_cost)
    # # 2) Voltage regularization penalizes membrane voltages that are below resting potential or above threshold
    # voltage_regularizer = models.VoltageRegularization(
    #     rsnn_layer.cell, flags.voltage_cost)

    # rate_loss = rate_distribution_regularizer(rsnn_layer.output[0])
    # voltage_loss = voltage_regularizer(rsnn_layer.output[1])
    # model.add_loss(rate_loss)
    # model.add_loss(voltage_loss)
    # model.add_metric(rate_loss, name='rate_loss')
    # model.add_metric(voltage_loss, name='voltage_loss')

    # def compute_loss(_target, _pred):
    #     return loss_object(_target, _pred) / global_batch_size

    # # Adaptive learning rates
    # optimizer = tf.keras.optimizers.Adam(flags.learning_rate)
    # model.compile(optimizer, compute_loss, metrics=['accuracy'])

    # # Restore weights from a checkpoint if desired
    # if flags.restore_from != '':
    #     model.load_weights(flags.restore_from)
    #     print(f'> Model successfully restored from {flags.restore_from}')

    # def get_dataset_fn(_n, _n_cues):
    #     def _f(_):
    #         _data_set = data_sets.create_evidence_accumulation(
    #             batch_size=global_batch_size, n_input=flags.n_input, seq_len=flags.seq_len,
    #             recall_duration=flags.recall_duration, examples_in_epoch=_n, n_cues=_n_cues,
    #             hard_only=flags.hard_only, t_cue=flags.cue_duration, t_interval=flags.interval_duration,
    #             input_f0=flags.input_f0
    #         ).repeat().map(_expand)
    #         return _data_set
    #     return _f

    # test_data_set = get_dataset_fn(
    #     flags.validation_examples, flags.n_cues)(None)

    # # Bookkeeping of simulations
    # sim_name = toolkit.get_random_identifier('b_')
    # results_dir = os.path.join(
    #     flags.results_dir, 'drifting_gratings_simulation')
    # print(f'> Results will be stored in {os.path.join(results_dir, sim_name)}')
    # os.makedirs(results_dir, exist_ok=True)

    # cm = simmanager.SimManager(sim_name, results_dir, write_protect_dirs=False, tee_stdx_to='output.log')

    # with cm:
    #     # Save the settings with which the script was invoked
    #     with open(os.path.join(cm.paths.data_path, 'flags.json'), 'w') as f:
    #         json.dump(flags.flag_values_dict(), f, indent=4)

    #     # Apply a learning curriculum using iteratively more cues up to the desired number
    #     for n_cues in range(1, flags.n_cues + 1, 2):
    #         train_data_set = get_dataset_fn(flags.examples_in_epoch, n_cues)(None)

    #         vis_data = test_data_set if flags.visualize_test else train_data_set
    #         # Define callbacks that are used for visualizing network activity (see above),
    #         # for stopping the training if the task is solved, and for saving the model
    #         plot_callback = PlotCallback(vis_data, extractor_model, network, flags.data_dir,
    #                                       path=cm.paths.results_path, prefix=f'cue_{n_cues}_')
    #         fit_callbacks = [
    #             plot_callback,
    #             callbacks.StopAt('accuracy', .99),
    #             tf.keras.callbacks.ModelCheckpoint(
    #                 filepath=os.path.join(cm.paths.results_path, 'model'),
    #                 monitor='val_accuracy', save_weights_only=True),
    #             tf.keras.callbacks.TensorBoard(log_dir=cm.paths.results_path)
    #         ]

    #         # Perform training
    #         model.fit(train_data_set, steps_per_epoch=flags.examples_in_epoch, epochs=flags.n_epochs,
    #                   validation_data=test_data_set, validation_steps=flags.validation_examples,
    #                   callbacks=fit_callbacks)


if __name__ == '__main__':
    _data_dir = 'GLIF_network'
    _results_dir = 'GLIF_network/results'
    _save_dir = 'Simulation_results'
    _images_dir = 'Images_general'

    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('save_dir', _save_dir, '')
    absl.app.flags.DEFINE_string('images_dir', _images_dir, '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('interarea_weight_distribution', 'billeh_like', '')
    absl.app.flags.DEFINE_integer('gratings_orientation', 0, '')
    absl.app.flags.DEFINE_integer('gratings_frequency', 2, '')
    absl.app.flags.DEFINE_integer('n_simulations', 20, '')

    absl.app.flags.DEFINE_float('learning_rate', .001, '')
    absl.app.flags.DEFINE_float('rate_cost', 0., '')
    absl.app.flags.DEFINE_float('voltage_cost', .001, '')
    absl.app.flags.DEFINE_float('dampening_factor', .1, '')
    absl.app.flags.DEFINE_float('recurrent_dampening_factor', .5, '')
    absl.app.flags.DEFINE_float('gauss_std', .3, '')
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    absl.app.flags.DEFINE_float('input_f0', 0.2, '')
    absl.app.flags.DEFINE_float('E4_weight_factor', 1., '')

    absl.app.flags.DEFINE_integer('n_epochs', 20, '')
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('v1_neurons', 10, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('lm_neurons', None, '')  # -1 to take all neurons
    # number of LGN filters in visual space (input population)
    absl.app.flags.DEFINE_integer('n_input', 17400, '')
    absl.app.flags.DEFINE_integer('seq_len', 3000, '')
    absl.app.flags.DEFINE_integer('n_cues', 3, '')
    absl.app.flags.DEFINE_integer('recall_duration', 40, '')
    absl.app.flags.DEFINE_integer('cue_duration', 40, '')
    absl.app.flags.DEFINE_integer('interval_duration', 40, '')
    absl.app.flags.DEFINE_integer('examples_in_epoch', 32, '')
    absl.app.flags.DEFINE_integer('validation_examples', 16, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_integer('neurons_per_output', 16, '')

    absl.app.flags.DEFINE_boolean('float16', False, '')
    absl.app.flags.DEFINE_boolean("save_float16", True, "")
    absl.app.flags.DEFINE_boolean('caching', True, '')
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('hard_reset', False, '')
    absl.app.flags.DEFINE_boolean('disconnect_lm_L6_inhibition', False, '')
    absl.app.flags.DEFINE_boolean('disconnect_v1_lm_L6_excitatory_projections', False, '')
    absl.app.flags.DEFINE_boolean('realistic_neurons_ratio', True, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_v1', True, '')
    absl.app.flags.DEFINE_boolean('train_recurrent_lm', True, '')
    absl.app.flags.DEFINE_boolean('train_input', True, '')
    absl.app.flags.DEFINE_boolean('train_interarea', True, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', True, '')
    absl.app.flags.DEFINE_boolean('hard_only', False, '')
    absl.app.flags.DEFINE_boolean('visualize_test', False, '')
    absl.app.flags.DEFINE_boolean('pseudo_gauss', False, '')
    absl.app.flags.DEFINE_boolean("bmtk_compat_lgn", True, "")

    absl.app.run(main)

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
from Model_utils.plotting_utils import InputActivityFigure
from general_utils import file_management
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

def printgpu(verbose=0):
    if tf.config.list_physical_devices('GPU'):
        meminfo = tf.config.experimental.get_memory_info('GPU:0')
        current = meminfo['current'] / 1024**3
        peak = meminfo['peak'] / 1024**3
        if verbose == 0:
            print(f"GPU memory use: {current:.2f} GB / Peak: {peak:.2f} GB")
        if verbose == 1:
            return current, peak


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

    # Get the neurons of each column of the network
    n_neurons = {'v1': flags.v1_neurons, 'lm': flags.lm_neurons}

    # Define flag string as the second part of results_path
    flag_str = f'v1_{n_neurons["v1"]}_lm_{n_neurons["lm"]}'
    
    # Create the results path
    results_dir = flags.results_dir
    os.makedirs(results_dir, exist_ok=True)
    print(f'> Results will be stored in:\n {results_dir} \n')

    # Can be used to try half precision training
    if flags.float16:
        if version.parse(tf.__version__) < version.parse("2.4.0"):
            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)
        else:
            mixed_precision.set_global_policy('mixed_float16')
        dtype = tf.float16
    else:
        dtype = tf.float32

    device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device=device)

    # Load LM-V1 column data
    t0 = time()
    if flags.caching:
        load_fn = load_sparse.cached_load_billeh
    else:
        load_fn = load_sparse.load_billeh

    networks, lgn_inputs, bkg_inputs = load_fn(flags, n_neurons, flag_str=flag_str)
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

        # Define the optimizer
        optimizer = tf.keras.optimizers.Adam(flags.learning_rate, epsilon=1e-11)  
        optimizer.build(model.trainable_variables)

        # Restore weights from a checkpoint if desired
        # Restore model and optimizer from a checkpoint if it exists
        print(model.trainable_variables)
        if flags.ckpt_dir != '' and os.path.exists(flags.ckpt_dir):
            print(f'Restoring checkpoint from {flags.ckpt_dir}...')
            checkpoint_directory = tf.train.latest_checkpoint(flags.ckpt_dir)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            checkpoint.restore(checkpoint_directory).assert_consumed()
            print('Checkpoint restored!')

        print(model.trainable_variables)

        # Build the model layers
        rsnn_layer = model.get_layer('rsnn')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=[rsnn_layer.output, model.output[0], model.output[1]])

        n_total_neurons = n_neurons['v1'] + n_neurons['lm']
        zero_state = rsnn_layer.cell.zero_state_multi_areas(flags.batch_size, np.float32)
        state_variables = tf.nest.map_structure(lambda a: tf.Variable(
            a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        ), zero_state)

        def roll_out(_x, _y, _w):
            _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
            dummy_zeros = tf.zeros((flags.batch_size, flags.seq_len, n_total_neurons), dtype)
            _out, _p, _ = extractor_model((_x, dummy_zeros, _initial_state))

            _v1_z, _lm_z = _out[0][0], _out[0][2]
            # update state_variables with the new model state
            new_state = tuple(_out[1:])
            tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

            return _v1_z, _lm_z

        @tf.function
        def distributed_roll_out(x, y, w):
            _v1_z, _lm_z = strategy.run(roll_out, args=(x, y, w))
            return _v1_z, _lm_z

        # Define OSI/DSI dataset
        def get_dataset_fn(regular=False):
            def _f(input_context):
                post_delay = flags.seq_len - (2500 % flags.seq_len)
                _data_set = stim_dataset.generate_drifting_grating_tuning(
                    seq_len=2500+post_delay,
                    pre_delay=500,
                    post_delay = post_delay,
                    n_input=flags.n_input,
                    regular=regular
                ).batch(1)
                            
                return _data_set
            return _f
        
        data_set = strategy.distribute_datasets_from_function(get_dataset_fn())

        print('Starting to plot OSI and DSI...')
        sim_duration = (2500//flags.seq_len + 1) * flags.seq_len
        v1_spikes = np.zeros((8, sim_duration, networks['v1']['n_nodes']), dtype=float)
        lm_spikes = np.zeros((8, sim_duration, networks['lm']['n_nodes']), dtype=float)
        DG_angles = np.arange(0, 360, 45)
        for trial_id in range(flags.n_trials):
            data_it = iter(data_set)
            for angle_id, angle in enumerate(range(0, 360, 45)):
                t0 = time()
                x, y, _, w = next(data_it)
                chunk_size = flags.seq_len
                num_chunks = (2500//chunk_size + 1)
                for i in range(num_chunks):
                    chunk = x[:, i * chunk_size : (i + 1) * chunk_size, :]
                    v1_z_chunk, lm_z_chunk = distributed_roll_out(chunk, y, w)
                    v1_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += v1_z_chunk.numpy()[0, :, :].astype(float)
                    lm_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += lm_z_chunk.numpy()[0, :, :].astype(float)
                    
                if trial_id == 0 and angle_id == 0:
                    # Raster plot for 0 degree orientation
                    lgn_spikes = x[:, :2500, :].numpy()
                    z_v1 = v1_spikes[:, :2500, :]
                    z_lm = lm_spikes[:, :2500, :]
                    images_dir = os.path.join(flags.ckpt_dir, "OSI_DSI_checkpoints", 'Raster_plots_OSI_DSI')
                    os.makedirs(images_dir, exist_ok=True)
                    graph = InputActivityFigure(
                                                networks,
                                                flags.data_dir,
                                                images_dir,
                                                filename=f'Epoch_0_orientation',
                                                frequency=flags.temporal_f,
                                                stimuli_init_time=500,
                                                stimuli_end_time=2500,
                                                reverse=False,
                                                plot_core_only=True,
                                                )
                    graph(lgn_spikes, z_v1, z_lm)

                print(f'Trial {trial_id+1}/{n_trials_per_angle} - Angle {angle} done.')
                print(f'    Trial running time: {time() - t0:.2f}s')
                mem_data = printgpu(verbose=1)
                print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB\n')

        # Average the spikes over the number of trials
        v1_spikes = v1_spikes/n_trials_per_angle
        v1_spikes = v1_spikes[:, :2500, :]
        lm_spikes = lm_spikes/n_trials_per_angle
        lm_spikes = lm_spikes[:, :2500, :]

    keys = ["z", "v", "z_lgn"] # "input_current", "recurrent_current", "bottom_up_current"
    
    # # Select the dtype of the data saved
    # if flags.save_float16:
    #     save_dtype = np.float16
    # else:
    #     save_dtype = np.float32
    # data_path = os.path.join(simulation_results_path, "Data")

    # SimulationDataHDF5 = other_billeh_utils.SaveSimDataHDF5(
    #     flags, keys, data_path, networks, n_neurons, v1_to_lm_neurons_ratio, save_core_only=True, dtype=save_dtype
    # )

    # time_per_sim = 0
    # time_per_save = 0
    # for trial in range(0, flags.n_simulations):
    #     print('{trial}:new trial'.format(trial=trial))
    #     inputs = (np.random.uniform(size=inputs.shape,
    #                                 low=0., high=1.) < firing_rates * .001).astype(np.float32)
    #     # Simulate the model for one stimulus sequence
    #     t0 = time()
    #     out = extractor_model((inputs, dummy_zeros, state))
    #     time_per_sim += time() - t0
    #     print('Simulation time: ', time() - t0)

    #     # Extract spikes and membrane voltages
    #     t0 = time()
    #     network_outputs = out[0][0]
    #     v1_spikes, v1_voltage, lm_spikes, lm_voltage = network_outputs

    #     # Save simulation data
    #     simulation_data = {
    #         "v1": {
    #             "z": v1_spikes,
    #             # "v": v1_voltage
    #         },
    #         "lm": {
    #             "z": lm_spikes,
    #             # "v": lm_voltage
    #         },
    #         "LGN": {
    #             "z_lgn": inputs
    #         }
    #     }
        
    #     SimulationDataHDF5(simulation_data, trial)
    #     time_per_save += time() - t0

    #     # Reset the model state to the last state of the previous simulation
    #     state = out[0][1:]

    # # Save the simulation metadata  
    # time_per_sim /= flags.n_simulations
    # time_per_save /= flags.n_simulations
    # metadata_path = os.path.join(data_path, 'Simulation stats')
    # with open(metadata_path, 'w') as out_file:
    #     out_file.write(f'Consumed time per simulation: {time_per_sim}\n')
    #     out_file.write(f'Consumed time saving: {time_per_save}\n')


if __name__ == '__main__':

    # Define the directory to save the results
    _results_dir = 'Time_to_first_spike_analysis'
    _checkpoint_dir = 'Current_model'

    # Load the best model configuration and set it as default
    config_path = 'Current_model/config.json'
    with open(config_path, 'r') as f:
        flags_dict = json.load(f)

    # Define particular task flags
    absl.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.flags.DEFINE_string('ckpt_dir', _checkpoint_dir, '')
    absl.flags.DEFINE_integer('n_trials', 1, '')


    # Define the flags using the loaded values as defaults
    absl.flags.DEFINE_string('data_dir', flags_dict.get('data_dir', 'GLIF_network'), '')
    absl.flags.DEFINE_string('restore_from', flags_dict.get('restore_from', ''), '')
    absl.flags.DEFINE_string('comment', flags_dict.get('comment', ''), '')
    absl.flags.DEFINE_string('interarea_weight_distribution', flags_dict.get('interarea_weight_distribution', 'billeh_like'), '')
    absl.flags.DEFINE_integer('gratings_orientation', flags_dict.get('gratings_orientation', 0), '')
    absl.flags.DEFINE_integer('gratings_frequency', flags_dict.get('gratings_frequency', 2), '')
    absl.flags.DEFINE_integer('n_simulations', flags_dict.get('n_simulations', 20), '')
    absl.flags.DEFINE_float('learning_rate', flags_dict.get('learning_rate', .01), '')
    absl.flags.DEFINE_float('rate_cost', flags_dict.get('rate_cost', 0.), '')
    absl.flags.DEFINE_float('voltage_cost', flags_dict.get('voltage_cost', .001), '')
    absl.flags.DEFINE_float('dampening_factor', flags_dict.get('dampening_factor', .1), '')
    absl.flags.DEFINE_float('recurrent_dampening_factor', flags_dict.get('recurrent_dampening_factor', .5), '')
    absl.flags.DEFINE_float('gauss_std', flags_dict.get('gauss_std', .3), '')
    absl.flags.DEFINE_float('lr_scale', flags_dict.get('lr_scale', 1.), '')
    absl.flags.DEFINE_float('input_f0', flags_dict.get('input_f0', 0.2), '')
    absl.flags.DEFINE_float('E4_weight_factor', flags_dict.get('E4_weight_factor', 1.), '')
    absl.flags.DEFINE_float('input_weight_scale', flags_dict.get('input_weight_scale', 1.), '')
    absl.flags.DEFINE_integer('n_epochs', flags_dict.get('n_epochs', 20), '')
    absl.flags.DEFINE_integer('batch_size', flags_dict.get('batch_size', 1), '')
    absl.flags.DEFINE_integer('v1_neurons', flags_dict.get('v1_neurons', 10), '')
    absl.flags.DEFINE_integer('lm_neurons', flags_dict.get('lm_neurons', None), '')
    absl.flags.DEFINE_integer('n_input', flags_dict.get('n_input', 17400), '')
    absl.flags.DEFINE_integer('seq_len', flags_dict.get('seq_len', 3000), '')
    absl.flags.DEFINE_integer('n_cues', flags_dict.get('n_cues', 3), '')
    absl.flags.DEFINE_integer('recall_duration', flags_dict.get('recall_duration', 40), '')
    absl.flags.DEFINE_integer('cue_duration', flags_dict.get('cue_duration', 40), '')
    absl.flags.DEFINE_integer('interval_duration', flags_dict.get('interval_duration', 40), '')
    absl.flags.DEFINE_integer('examples_in_epoch', flags_dict.get('examples_in_epoch', 32), '')
    absl.flags.DEFINE_integer('validation_examples', flags_dict.get('validation_examples', 16), '')
    absl.flags.DEFINE_integer('seed', flags_dict.get('seed', 3000), '')
    absl.flags.DEFINE_integer('neurons_per_output', flags_dict.get('neurons_per_output', 16), '')
    absl.flags.DEFINE_boolean('float16', flags_dict.get('float16', False), '')
    absl.flags.DEFINE_boolean('save_float16', flags_dict.get('save_float16', True), '')
    absl.flags.DEFINE_boolean('caching', flags_dict.get('caching', True), '')
    absl.flags.DEFINE_boolean('core_only', flags_dict.get('core_only', False), '')
    absl.flags.DEFINE_boolean('hard_reset', flags_dict.get('hard_reset', False), '')
    absl.flags.DEFINE_boolean('disconnect_lm_L6_inhibition', flags_dict.get('disconnect_lm_L6_inhibition', False), '')
    absl.flags.DEFINE_boolean('disconnect_v1_lm_L6_excitatory_projections', flags_dict.get('disconnect_v1_lm_L6_excitatory_projections', False), '')
    absl.flags.DEFINE_boolean('realistic_neurons_ratio', flags_dict.get('realistic_neurons_ratio', True), '')
    absl.flags.DEFINE_boolean('train_recurrent_v1', flags_dict.get('train_recurrent_v1', False), '')
    absl.flags.DEFINE_boolean('train_recurrent_lm', flags_dict.get('train_recurrent_lm', False), '')
    absl.flags.DEFINE_boolean('train_input', flags_dict.get('train_input', False), '')
    absl.flags.DEFINE_boolean('train_interarea', flags_dict.get('train_interarea', False), '')
    absl.flags.DEFINE_boolean('train_noise', flags_dict.get('train_noise', False), '')
    absl.flags.DEFINE_boolean('connected_selection', flags_dict.get('connected_selection', True), '')
    absl.flags.DEFINE_boolean('neuron_output', flags_dict.get('neuron_output', True), '')
    absl.flags.DEFINE_boolean('hard_only', flags_dict.get('hard_only', False), '')
    absl.flags.DEFINE_boolean('visualize_test', flags_dict.get('visualize_test', False), '')
    absl.flags.DEFINE_boolean('pseudo_gauss', flags_dict.get('pseudo_gauss', False), '')
    absl.flags.DEFINE_boolean('bmtk_compat_lgn', flags_dict.get('bmtk_compat_lgn', True), '')

    absl.app.run(main)

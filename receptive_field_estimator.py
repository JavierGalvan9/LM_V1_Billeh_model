import os
# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'global'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import absl
import numpy as np
import pandas as pd
import tensorflow as tf
from packaging import version

if version.parse(tf.__version__) < version.parse("2.4.0"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision

import json
import datetime as dt
from Model_utils import load_sparse, models, other_billeh_utils, stim_dataset, toolkit
from Model_utils.plotting_utils import InputActivityFigure
from Model_utils.callbacks import printgpu
from time import time
import ctypes.util
import random
# import itertools

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
    sim_seed = flags.seed
    # # set the seeds to ensure randomness across different runs
    # # Get the current time in milliseconds and use it as a seed with range 0 to 2^32 - 1
    # sim_seed = int(time() * 1000) % (2**32 - 1)
    np.random.seed(sim_seed)
    tf.random.set_seed(sim_seed)
    random.seed(sim_seed)
    print("Current seed:", sim_seed)

    # Get the neurons of each column of the network
    n_neurons = {'v1': flags.v1_neurons, 'lm': flags.lm_neurons}
    n_total_neurons = n_neurons['v1'] + n_neurons['lm']
    
    total_seq_len = flags.n_angles * flags.seq_len
    
    logdir = flags.ckpt_dir
    if logdir == '':
        flag_str = f'v1_{n_neurons["v1"]}_lm_{n_neurons["lm"]}'
        for name, value in flags.flag_values_dict().items():
            if value != flags[name].default and name in ['n_input', 'core_only', 'connected_selection', 'interarea_weight_distribution', 'E4_weight_factor', 'random_weights']:
                flag_str += f'_{name}_{value}'
        # Define flag string as the second part of results_path
        results_dir = f'Simulation_results/{flag_str}'
        os.makedirs(results_dir, exist_ok=True)
        print('Simulation results path: ', results_dir)
        # Generate a ticker for the current simulation
        sim_name = toolkit.get_random_identifier('b_')
        logdir = os.path.join(results_dir, sim_name)
        print(f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')
        # current_epoch = 0
    else:
        flag_str = logdir.split(os.path.sep)[-2]
        # current_epoch = (flags.run_session + 1) * flags.n_epochs

    # # Create the results path
    # os.makedirs(flags.results_dir, exist_ok=True)
    # print(f'> Results will be stored in:\n {flags.results_dir} \n')

    # Can be used to try half precision training
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

    device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    strategy = tf.distribute.OneDeviceStrategy(device=device)

    per_replica_batch_size = flags.batch_size
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    print(f'Per replica batch size: {per_replica_batch_size}')
    print(f'Global batch size: {global_batch_size}')
    print(f'Number of trials: {flags.n_trials}\n')

    # Load LM-V1 column data
    t0 = time()
    if flags.caching:
        load_fn = load_sparse.cached_load_billeh
    else:
        load_fn = load_sparse.load_billeh

    networks, lgn_inputs, bkg_inputs = load_fn(flags, n_neurons, flag_str=flag_str)
    print(f"Model files loading: {time()-t0:.2f} seconds\n")

    # Load the LGN units data for find the extreme possible values for the RF locations
    path = '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/lgn_model/data/lgn_full_col_cells_240x120.csv'
    lgn_data = pd.read_csv(path, sep=' ')
    # Get the extreme locations of the lgn connected neurons
    lgn_ids = np.array(list(set(lgn_inputs['v1']['indices'][:, 1])))
    x, y = lgn_data['x'][lgn_ids].values, lgn_data['y'][lgn_ids].values
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    # using the min_x and max_x, calculate the real initial and final row for iteration
    initial_row = int((min_y - flags.radius_circle) / flags.radius_circle)
    final_row = int(max_y / flags.radius_circle) + 1
    # same for the columns
    initial_col = int((min_x - flags.radius_circle) / flags.radius_circle)
    final_col = int(max_x / flags.radius_circle) + 1

    # Define the scope in which the model training will be executed
    with strategy.scope():
        t0 = time()
        # Build the model
        model = models.create_model(
            networks, 
            lgn_inputs, 
            bkg_inputs, 
            seq_len=total_seq_len,
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
            batch_size=per_replica_batch_size, 
            pseudo_gauss=flags.pseudo_gauss, 
            use_state_input= True, 
            return_state= True,
            hard_reset=flags.hard_reset,
            connected_recurrent_connections=flags.connected_recurrent_connections,
            connected_areas=flags.connected_areas,
            connected_noise=flags.connected_noise,
            add_rate_metric=False, 
            max_delay=5, 
            n_output=flags.n_output,
            neuron_output=flags.neuron_output,
            current_input=flags.current_input
            # output_completed_valid_from_time=120, 
            # output_abstract_valid_from_time=100,
            )
        
        #  Define and build the optimizer
        model.build((per_replica_batch_size, total_seq_len, flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")

        # Restore model and optimizer from a checkpoint if it exists
        if flags.ckpt_dir != '' and os.path.exists(os.path.join(flags.ckpt_dir, flags.restore_from)):
            print(f'Restoring checkpoint from {os.path.join(flags.ckpt_dir, flags.restore_from)}...')
            # checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, "OSI_DSI_checkpoints"))
            checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, flags.restore_from))
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_directory).expect_partial() #.assert_consumed()
            if flags.restore_from == "Best_model":
                logdir = checkpoint_directory + "_results"
            print('Checkpoint restored!')
        elif flags.restore_from != '' and os.path.exists(flags.restore_from):
            checkpoint_directory = tf.train.latest_checkpoint(flags.restore_from)
            print(f'Restoring checkpoint from {checkpoint_directory}...')
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_directory).expect_partial() #.assert_consumed()
            print('Checkpoint restored!')
        else:
            checkpoint_directory = None
            print(f"No checkpoint found in {flags.ckpt_dir} or {flags.restore_from}. Starting from scratch...\n")

        setup_dir = f'connected_areas_{flags.connected_areas}_conn_rec_{flags.connected_recurrent_connections}_conn_noise_{flags.connected_noise} '
        if checkpoint_directory is not None:
            results_dir = os.path.join(checkpoint_directory+'_results', setup_dir, f'Receptive_field_Gabor_radius_{flags.radius_circle} ')
        else:
            results_dir = os.path.join(logdir, setup_dir, f'Receptive_field_Gabor_radius_{flags.radius_circle}_from_scratch')
        os.makedirs(results_dir, exist_ok=True)
        print(f'Receptive field analysis will be saved in: {results_dir}\n')

        # Build the model layers
        rsnn_layer = model.get_layer('rsnn')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=model.output)
        zero_state = rsnn_layer.cell.zero_state_multi_areas(per_replica_batch_size, dtype=dtype)
        # state_variables = tf.nest.map_structure(lambda a: tf.Variable(
        #     a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        # ), zero_state)

        # Define the functions to forward pass the inputs through the model
        def roll_out(_x, _state_variables):
            # _initial_state = tf.nest.map_structure(lambda _a: _a.read_value(), state_variables)
            _initial_state = _state_variables
            seq_len = tf.shape(_x)[1]
            dummy_zeros = tf.zeros((per_replica_batch_size, seq_len, n_total_neurons), dtype)
            _out, _ = extractor_model((_x, dummy_zeros, _initial_state))
            _v1_z, _, _lm_z, _ = _out[0]
            # update state_variables with the new model state
            new_state = tuple(_out[1:])
            # tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)

            return _v1_z, _lm_z, new_state

        @tf.function
        def distributed_roll_out(x, state_variables):
            _v1_z, _lm_z, new_state = strategy.run(roll_out, args=(x, state_variables))
            return _v1_z, _lm_z, new_state
        
        # Generate spontaneous spikes efficiently
        @tf.function
        def generate_spontaneous_spikes(spontaneous_prob):
            random_uniform = tf.random.uniform(tf.shape(spontaneous_prob), dtype=dtype)
            return tf.less(random_uniform, spontaneous_prob)

        # Define dataset
        delays = [int(a) for a in flags.delays.split(',') if a != '']
        DG_angles = [0, 45, 90]
        col_size = 240
        row_size = 120
        n_cols =  int(np.ceil((col_size - flags.radius_circle) / flags.radius_circle))
        n_rows = int(np.ceil((row_size - flags.radius_circle) / flags.radius_circle))
        # n_rows = 13
        # n_cols = 11
        # Select the dtype of the data saved
        data_path = os.path.join(results_dir, "Data")
        os.makedirs(data_path, exist_ok=True)
        SimulationDataHDF5 = other_billeh_utils.SaveGaborSimDataHDF5(flags, data_path, networks, initial_row=initial_row, initial_col=initial_col,
                                                                    final_row=final_row, final_col=final_col,
                                                                    save_core_only=True)

        # circle_rows = np.arange(0, n_rows)
        # circle_columns = np.arange(0, n_cols)
        # # # get all the possible combinations using itertools
        # import itertools
        # all_combinations = list(itertools.product(circle_rows, circle_columns))

        print("\n---------- Simulations started at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')

        col = flags.circle_column
        for row in range(initial_row, final_row):
        # for row, col in [(flags.circle_row, flags.circle_column)]:
        # for row, col in all_combinations:
            continuing_state = zero_state
            # print gabor position
            print(f'\nGabor position: row {row}, column {col}')
            # lgn_firing_rates_complete = np.zeros((total_seq_len, flags.n_input), dtype=np.float32)
            t0 = time()
            # convert row and col to tensors to avoid retracing
            col, row = tf.convert_to_tensor(col, dtype=tf.int64), tf.convert_to_tensor(row, dtype=tf.int64)
            # for i in range(len(DG_angles)):
            lgn_firing_rates_generator = stim_dataset.generate_gabor_patches_tuning(
                                seq_len = flags.seq_len,
                                pre_delay = delays[0],
                                post_delay = delays[1],
                                n_input = flags.n_input,
                                # orientation=DG_angles[i],
                                regular = True, 
                                x0 = col,
                                y0 = row,
                                r = flags.radius_circle,
                                temporal_f = 2,
                                cpd = 0.04,
                                contrast = 1,
                                moving_flag = True, 
                                inverse = False,
                                rotation=flags.rotation,
                                billeh_phase=True,
                                dtype=dtype,
                                return_firing_rates=True,
                            )
                # lgn_firing_rates = next(iter(lgn_firing_rates_generator))
                # lgn_firing_rates_complete[i*flags.seq_len:(i+1)*flags.seq_len, :] = lgn_firing_rates.numpy()

            lgn_firing_rates_complete = []
            lgn_firing_rates_it = iter(lgn_firing_rates_generator)
            for _ in range(len(DG_angles)):
                lgn_firing_rates = next(lgn_firing_rates_it)
                lgn_firing_rates_complete.append(lgn_firing_rates.numpy())
                # lgn_firing_rates_complete[i*flags.seq_len:(i+1)*flags.seq_len, :] = lgn_firing_rates.numpy()

            print(f'Gabor patches generation time: {time() - t0:.2f}s\n')	

            # # Compute LGN firing probabilities
            # lgn_firing_rates_complete = tf.constant(lgn_firing_rates_complete, dtype=dtype)
            # lgn_firing_prob = 1 - tf.exp(-tf.cast(lgn_firing_rates_complete, dtype) / 1000.)
            # lgn_firing_prob = tf.tile(tf.expand_dims(lgn_firing_prob, axis=0), [per_replica_batch_size, 1, 1])
            
            # Initialize arrays to store sums over valid time ranges
            lgn_sums = np.zeros((flags.n_trials, flags.n_input), dtype=np.int32)
            v1_sums = np.zeros((flags.n_trials, n_neurons['v1']), dtype=np.int32)
            lm_sums = np.zeros((flags.n_trials, n_neurons['lm']), dtype=np.int32)

            n_iterations = int(np.ceil(flags.n_trials / per_replica_batch_size))       
            slot_indices = [
                (angle_id * flags.seq_len + delays[0], (angle_id + 1) * flags.seq_len - delays[1])
                for angle_id in range(flags.n_angles)
            ]
            time_per_sim = 0
            time_per_save = 0
            for iter_id in range(n_iterations):
                t0 = time()
                # Randomize the order of orientations
                randomized_indices = random.sample(range(len(lgn_firing_rates_complete)), len(lgn_firing_rates_complete))
                # Concatenate the randomized firing rates for the current iteration
                randomized_firing_rates = [lgn_firing_rates_complete[idx] for idx in randomized_indices]
                # Flatten the randomized firing rates into the complete firing rate array
                randomized_firing_rates_complete = np.concatenate(randomized_firing_rates, axis=0)
                # Convert to TensorFlow tensor
                lgn_firing_prob = tf.constant(randomized_firing_rates_complete, dtype=dtype)
                lgn_firing_prob = 1 - tf.exp(-lgn_firing_prob / 1000.)
                lgn_firing_prob = tf.tile(tf.expand_dims(lgn_firing_prob, axis=0), [per_replica_batch_size, 1, 1])
                # Concatenate the lgn spikes for the different orientations
                start_idx = iter_id * per_replica_batch_size
                end_idx = min((iter_id + 1) * per_replica_batch_size, flags.n_trials)
                iteration_length = end_idx - start_idx
                # Slice the available portion of lgn_spikes
                x = generate_spontaneous_spikes(lgn_firing_prob)
                v1_spikes, lm_spikes, continuing_state = distributed_roll_out(x, continuing_state)
                # Directly accumulate sums for the relevant slots without storing intermediate spike arrays
                for slot_start, slot_end in slot_indices:
                    # Convert to numpy and accumulate for the batch slice
                    lgn_sums[start_idx:end_idx, :] += np.sum(x[:iteration_length, slot_start:slot_end, :].numpy(), axis=1).astype(np.int32)
                    v1_sums[start_idx:end_idx, :] += np.sum(v1_spikes[:iteration_length, slot_start:slot_end, :].numpy(), axis=1).astype(np.int32)
                    lm_sums[start_idx:end_idx, :] += np.sum(lm_spikes[:iteration_length, slot_start:slot_end, :].numpy(), axis=1).astype(np.int32)
                
                print(f'Trial {iter_id+1}/{n_iterations} running time: {time() - t0:.2f}s')
                printgpu()
                
            # Save simulation data
            simulation_data = {
                "v1": {"z": v1_sums},
                "lm": {"z": lm_sums},
                "LGN": {"z": lgn_sums}
            }
            SimulationDataHDF5(simulation_data, row, col, flags.radius_circle)
            time_per_save += time() - t0
                
            # Save the simulation metadata  
            time_per_sim /= flags.n_trials
            time_per_save /= flags.n_trials
            metadata_path = os.path.join(data_path, 'Simulation stats')
            with open(metadata_path, 'w') as out_file:
                out_file.write(f'Consumed time per simulation: {time_per_sim}\n')
                out_file.write(f'Consumed time saving: {time_per_save}\n')
                out_file.write(f'Simulation finished at {dt.datetime.now().strftime("%d-%m-%Y %H:%M")}\n')

            # empty the memory from the gpu
            # tf.keras.backend.clear_session()
            printgpu()
            print('Simulation finished successfully.')
            
        ### RASTER PLOT ###
        images_dir = os.path.join(results_dir, 'Raster_plots')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    networks,
                                    flags.data_dir,
                                    images_dir,
                                    filename=f'Rasterplot_0',
                                    frequency=flags.temporal_f,
                                    stimuli_init_time=delays[0],
                                    stimuli_end_time=total_seq_len-delays[1],
                                    reverse=False,
                                    plot_core_only=True,
                                    )
        graph(x.numpy(), v1_spikes, lm_spikes)

        ### READ THE DATA FILES ###
        # full_data_path = 'Time_to_first_spike_analysis/Data/simulation_data.hdf5'
        # data, flags_dict, n_trials = other_billeh_utils.load_simulation_results_hdf5(full_data_path)


if __name__ == '__main__':

    # Define the directory to save the results
    _results_dir = 'Receptive_field'
    _checkpoint_dir = 'Benchmark_models/v1_100000_lm_30000'
    # _checkpoint_dir = '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_drbc'
    # _checkpoint_dir = '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_3514_mnist'
    _restore_from = 'Best_model'
    # Define particular task flags
    absl.app.flags.DEFINE_string('task_name', 'receptive_fields_estimation' , '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('ckpt_dir', _checkpoint_dir, '')
    absl.app.flags.DEFINE_string('restore_from', _restore_from, '')
    absl.app.flags.DEFINE_integer('n_trials', 10, '')
    absl.app.flags.DEFINE_integer('seq_len', 3000, '')

    absl.app.flags.DEFINE_string('delays', '500,500', '')

    # flags for gabor patches
    absl.app.flags.DEFINE_integer('orientation', 0, '')
    absl.app.flags.DEFINE_integer('row_size', 120, '')
    absl.app.flags.DEFINE_integer('col_size', 240, '')
    absl.app.flags.DEFINE_integer('circle_row', 6, '')
    absl.app.flags.DEFINE_integer('circle_column', 5, '')
    absl.app.flags.DEFINE_integer('radius_circle', 10, '')
    absl.app.flags.DEFINE_integer('batch_size', 1, '')
    absl.app.flags.DEFINE_integer('n_angles', 3, '')
    

    # Load the best model configuration and set it as default
    config_path = os.path.join(_checkpoint_dir, 'config.json')
    with open(config_path, 'r') as f:
        flags_dict = json.load(f)

    # Define the flags using the loaded values as defaults
    absl.app.flags.DEFINE_string('data_dir', flags_dict.get('data_dir', 'GLIF_network'), '')
    absl.app.flags.DEFINE_string('comment', flags_dict.get('comment', ''), '')
    absl.app.flags.DEFINE_string('interarea_weight_distribution', flags_dict.get('interarea_weight_distribution', 'billeh_like'), '')
    absl.app.flags.DEFINE_string('dtype', flags_dict.get('dtype', 'float32'), '')
    absl.app.flags.DEFINE_string('rotation', flags_dict.get('rotation', 'ccw'), '')

    absl.app.flags.DEFINE_float('learning_rate', flags_dict.get('learning_rate', .01), '')
    absl.app.flags.DEFINE_float('dampening_factor', flags_dict.get('dampening_factor', .1), '')
    absl.app.flags.DEFINE_float('recurrent_dampening_factor', flags_dict.get('recurrent_dampening_factor', .5), '')
    absl.app.flags.DEFINE_float('input_weight_scale', flags_dict.get('input_weight_scale', 1.), '')
    absl.app.flags.DEFINE_float('gauss_std', flags_dict.get('gauss_std', .3), '')
    absl.app.flags.DEFINE_float('lr_scale', flags_dict.get('lr_scale', 1.), '')
    absl.app.flags.DEFINE_float('input_f0', flags_dict.get('input_f0', 0.2), '')
    absl.app.flags.DEFINE_float('E4_weight_factor', flags_dict.get('E4_weight_factor', 1.), '')
    absl.app.flags.DEFINE_float('temporal_f', flags_dict.get('temporal_f', 2.), '')

    absl.app.flags.DEFINE_integer('n_epochs', flags_dict.get('n_epochs', 20), '')
    absl.app.flags.DEFINE_integer('v1_neurons', flags_dict.get('v1_neurons', 10), '')
    absl.app.flags.DEFINE_integer('lm_neurons', flags_dict.get('lm_neurons', None), '')
    absl.app.flags.DEFINE_integer('steps_per_epoch', flags_dict.get('steps_per_epoch', 10), '')
    absl.app.flags.DEFINE_integer('n_input', flags_dict.get('n_input', 17400), '')
    absl.app.flags.DEFINE_integer('n_cues', flags_dict.get('n_cues', 3), '')
    absl.app.flags.DEFINE_integer('recall_duration', flags_dict.get('recall_duration', 40), '')
    absl.app.flags.DEFINE_integer('cue_duration', flags_dict.get('cue_duration', 40), '')
    absl.app.flags.DEFINE_integer('interval_duration', flags_dict.get('interval_duration', 40), '')
    absl.app.flags.DEFINE_integer('examples_in_epoch', flags_dict.get('examples_in_epoch', 32), '')
    absl.app.flags.DEFINE_integer('validation_examples', flags_dict.get('validation_examples', 16), '')
    absl.app.flags.DEFINE_integer('seed', flags_dict.get('seed', 3000), '')
    absl.app.flags.DEFINE_integer('neurons_per_output', flags_dict.get('neurons_per_output', 16), '')
    absl.app.flags.DEFINE_integer('n_output', flags_dict.get('n_output', 10), '')
    
    absl.app.flags.DEFINE_boolean('caching', flags_dict.get('caching', True), '')
    absl.app.flags.DEFINE_boolean('core_only', flags_dict.get('core_only', False), '')
    absl.app.flags.DEFINE_boolean('hard_reset', flags_dict.get('hard_reset', False), '')
    absl.app.flags.DEFINE_boolean('disconnect_lm_L6_inhibition', flags_dict.get('disconnect_lm_L6_inhibition', False), '')
    absl.app.flags.DEFINE_boolean('disconnect_v1_lm_L6_excitatory_projections', flags_dict.get('disconnect_v1_lm_L6_excitatory_projections', False), '')
    absl.app.flags.DEFINE_boolean('realistic_neurons_ratio', flags_dict.get('realistic_neurons_ratio', True), '')
    absl.app.flags.DEFINE_boolean('train_recurrent_v1', flags_dict.get('train_recurrent_v1', False), '')
    absl.app.flags.DEFINE_boolean('train_recurrent_lm', flags_dict.get('train_recurrent_lm', False), '')
    absl.app.flags.DEFINE_boolean('train_input', flags_dict.get('train_input', False), '')
    absl.app.flags.DEFINE_boolean('train_interarea_lm_v1', flags_dict.get('train_interarea_lm_v1', False), '')
    absl.app.flags.DEFINE_boolean('train_interarea_v1_lm', flags_dict.get('train_interarea_v1_lm', False), '')
    absl.app.flags.DEFINE_boolean('train_noise', flags_dict.get('train_noise', False), '')
    absl.app.flags.DEFINE_boolean('connected_selection', flags_dict.get('connected_selection', True), '')
    absl.app.flags.DEFINE_boolean('neuron_output', flags_dict.get('neuron_output', True), '')
    # absl.app.flags.DEFINE_boolean('hard_only', flags_dict.get('hard_only', False), '')
    absl.app.flags.DEFINE_boolean('visualize_test', flags_dict.get('visualize_test', False), '')
    absl.app.flags.DEFINE_boolean('pseudo_gauss', flags_dict.get('pseudo_gauss', False), '')
    absl.app.flags.DEFINE_boolean('bmtk_compat_lgn', flags_dict.get('bmtk_compat_lgn', True), '')
    absl.app.flags.DEFINE_boolean("connected_areas", flags_dict.get("connected_areas", True), "")
    absl.app.flags.DEFINE_boolean("connected_recurrent_connections", flags_dict.get("connected_recurrent_connections", True), "")
    absl.app.flags.DEFINE_boolean("connected_noise", flags_dict.get("connected_noise", True), "")
    absl.app.flags.DEFINE_boolean("current_input", flags_dict.get("current_input", False), "")
    absl.app.flags.DEFINE_boolean("random_weights", flags_dict.get("random_weights", False), "")

    absl.app.run(main)

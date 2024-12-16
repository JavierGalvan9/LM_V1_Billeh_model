import os
# Define the environment variables for optimal GPU performance
os.environ['TF_GPU_THREAD_MODE'] = 'global'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import absl
import numpy as np
import tensorflow as tf
import pickle as pkl
from packaging import version
# from tensorflow.python.client import device_lib

if version.parse(tf.__version__) < version.parse("2.4.0"):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
else:
    from tensorflow.keras import mixed_precision

from Model_utils import load_sparse, models, stim_dataset, toolkit
from Model_utils.callbacks import OsiDsiCallbacks, printgpu
from time import time
import ctypes.util
import random


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

    # Select the connectivity rules in the network
    if flags.realistic_neurons_ratio:
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
        current_epoch = 0
    else:
        flag_str = logdir.split(os.path.sep)[-2]
        current_epoch = (flags.run_session + 1) * flags.n_epochs
        
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
    # model is being run on multiple GPUs or CPUs, and the results are being reduced to a single CPU device. 
    # In this case, the reduce_to_device argument is set to "cpu:0", which means that the results are being reduced to the first CPU device.
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice(reduce_to_device="cpu:0")) 
    # device = "/gpu:0" if tf.config.list_physical_devices("GPU") else "/cpu:0"
    # print(tf.config.list_physical_devices("GPU"))
    # strategy = tf.distribute.OneDeviceStrategy(device=device)
    # def get_available_devices():
    #     local_device_protos = device_lib.list_local_devices()
    #     return [x.name for x in local_device_protos]
    # print(get_available_devices())

    strategy = tf.distribute.MirroredStrategy()

    per_replica_batch_size = flags.n_trials_per_angle #flags.batch_size
    global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
    print(f'Per replica batch size: {per_replica_batch_size}')
    print(f'Global batch size: {global_batch_size}\n')

    # Load data of Billeh et al. (2020) and select appropriate number of neurons and inputs
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
            current_input=flags.current_input
            # output_completed_valid_from_time=120, 
            # output_abstract_valid_from_time=100,
            )

        model.build((per_replica_batch_size, flags.seq_len, flags.n_input))
        print(f"Model built in {time()-t0:.2f} s\n")

        # Store the initial model variables that are going to be trained
        model_variables_dict = {'Initial': {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}}
        
        # Restore model (not the optimizer) from a checkpoint if it exists
        if flags.ckpt_dir != '' and os.path.exists(os.path.join(flags.ckpt_dir, flags.restore_from)):
            # checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, "OSI_DSI_checkpoints"))
            checkpoint_directory = tf.train.latest_checkpoint(os.path.join(flags.ckpt_dir, flags.restore_from))
            print(f'Restoring checkpoint from {checkpoint_directory}...')
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_directory).expect_partial() #.assert_consumed()
            if flags.restore_from == "Best_model":
                logdir = checkpoint_directory + "_results"
            print('Checkpoint restored!')
            print(f'OSI/DSI results for epoch {current_epoch} will be saved in: {logdir}\n')
        elif flags.restore_from != '' and os.path.exists(flags.restore_from):
            checkpoint_directory = tf.train.latest_checkpoint(flags.restore_from)
            print(f'Restoring checkpoint from {checkpoint_directory}...')
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.restore(checkpoint_directory).expect_partial() #.assert_consumed()
            print('Checkpoint restored!')
            print(f'OSI/DSI results for epoch {current_epoch} will be saved in: {logdir}\n')
        else:
            print(f"No checkpoint found in {flags.ckpt_dir} or {flags.restore_from}. Starting from scratch...\n")

        model_variables_dict['Best'] =  {var.name: var.numpy().astype(np.float16) for var in model.trainable_variables}
        print(f"Model variables stored in dictionary\n")

        # Build the model layers
        rsnn_layer = model.get_layer('rsnn')
        # prediction_layer = model.get_layer('prediction')
        # abstract_layer = model.get_layer('abstract_output')
        extractor_model = tf.keras.Model(inputs=model.inputs,
                                         outputs=model.output)

        n_total_neurons = v1_neurons + lm_neurons
        zero_state = rsnn_layer.cell.zero_state_multi_areas(per_replica_batch_size, dtype=dtype)
        # state_variables = tf.nest.map_structure(lambda a: tf.Variable(
        #     a, trainable=False, synchronization=tf.VariableSynchronization.ON_READ
        # ), zero_state)

        # Precompute spontaneous LGN firing rates once
        def compute_spontaneous_lgn_firing_rates():
            cache_dir = "lgn_model/.cache_lgn"
            cache_file = os.path.join(cache_dir, f"spontaneous_lgn_probabilities_n_input_{flags.n_input}_seqlen_{flags.seq_len}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    spontaneous_prob = pkl.load(f)
                print("Loaded cached spontaneous LGN firing rates.")
            else:
                # Compute and cache the spontaneous firing rates
                spontaneous_lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
                    seq_len=flags.seq_len,
                    pre_delay=flags.seq_len,
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

        # LGN firing rates to the different angles
        DG_angles = np.arange(0, 360, 45)
        # Precompute the OSI/DSI LGN firing rates dataset
        def compute_osi_dsi_lgn_firing_rates():
            cache_dir = "lgn_model/.cache_lgn"
            post_delay = flags.seq_len - (2500 % flags.seq_len) if flags.seq_len < 2500 else 0
            osi_seq_len = 2500 + post_delay
            osi_dataset_path = os.path.join(cache_dir, f"osi_dsi_lgn_probabilities__n_input_{flags.n_input}_seqlen_{osi_seq_len}.pkl")

            if os.path.exists(osi_dataset_path):
                # Load cached OSI/DSI LGN firing rates dataset
                with open(osi_dataset_path, "rb") as f:
                    lgn_firing_probabilities_dict = pkl.load(f)
                print("Loaded cached OSI/DSI LGN firing rates dataset.")
            else:
                # Create and cache the OSI/DSI LGN firing rates dataset
                print("Creating OSI/DSI dataset...")
                lgn_firing_rates = stim_dataset.generate_drifting_grating_tuning(
                    seq_len=osi_seq_len,
                    pre_delay=500,
                    post_delay=post_delay,
                    n_input=flags.n_input,
                    regular=True,
                    return_firing_rates=True,
                    rotation=flags.rotation,
                    billeh_phase=True,
                    dtype=dtype
                    )

                # Distribute and generate the dataset
                osi_dsi_data_set = iter(lgn_firing_rates)
                lgn_firing_probabilities_dict = {}
                # Compute and store LGN firing rates for each angle
                for angle in DG_angles:
                    t0 = time()
                    angle_lgn_firing_rates = next(osi_dsi_data_set)
                    lgn_prob = 1 - tf.exp(-tf.cast(angle_lgn_firing_rates, dtype) / 1000.)
                    lgn_firing_probabilities_dict[angle] = lgn_prob
                    print(f"Angle {angle} done.")
                    print(f"    LGN running time: {time() - t0:.2f}s")
                    for gpu_id in range(len(strategy.extended.worker_devices)):
                        printgpu(gpu_id=gpu_id)

                # Save the OSI/DSI dataset to cache
                os.makedirs(cache_dir, exist_ok=True)
                with open(osi_dataset_path, "wb") as f:
                    pkl.dump(lgn_firing_probabilities_dict, f)
                print("OSI/DSI dataset created and cached successfully!")

            return lgn_firing_probabilities_dict

        # Load the OSI/DSI LGN firing probabilites dataset once
        lgn_firing_probabilities_dict = compute_osi_dsi_lgn_firing_rates()

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
    def distributed_roll_out(x,state_variables):
        _v1_z, _lm_z, new_state = strategy.run(roll_out, args=(x,state_variables))
        return _v1_z, _lm_z, new_state
    
    # Generate spontaneous spikes efficiently
    @tf.function
    def generate_spontaneous_spikes(spontaneous_prob):
        random_uniform = tf.random.uniform(tf.shape(spontaneous_prob), dtype=dtype)
        return tf.less(random_uniform, spontaneous_prob)
            
    def generate_gray_state(spontaneous_prob):
        # Generate LGN spikes
        x = generate_spontaneous_spikes(spontaneous_prob)
        # Simulate the network with a gray screen   
        _, _, new_state = roll_out(x, zero_state)
        return new_state
    
    @tf.function
    def distributed_generate_gray_state(spontaneous_prob):
        # Run generate_gray_state on each replica
        return strategy.run(generate_gray_state, args=(spontaneous_prob,))
    
    # Generate the gray state
    gray_state = distributed_generate_gray_state(spontaneous_prob)
    continuing_state = gray_state

    # def reset_state(reset_type='zero', new_state=None):
    #     if reset_type == 'zero':
    #         tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, zero_state)
    #     elif reset_type == 'gray':
    #         # Run a gray simulation to get the model state
    #         tf.nest.map_structure(lambda a, b: a.assign(b), state_variables, new_state)
    #     elif reset_type == 'continue':
    #         # Continue on the previous model state
    #         # No action needed, as the state_variables will not be modified
    #         pass
    #     else:
    #         raise ValueError(f"Invalid reset_type: {reset_type}")

    # # @tf.function
    # def distributed_reset_state(reset_type, gray_state=None):
    #     if reset_type == 'gray':
    #         strategy.run(reset_state, args=(reset_type, gray_state))
    #     else:
    #         strategy.run(reset_state, args=(reset_type, zero_state))
    
    # Define the callbacks for the OSI/DSI analysis
    # sim_duration = (2500//flags.seq_len + 1) * flags.seq_len
    sim_duration = int(np.ceil(2500/flags.seq_len)) * flags.seq_len
    post_delay = sim_duration - 2500
    callbacks = OsiDsiCallbacks(networks, lgn_inputs, bkg_inputs, flags, logdir, current_epoch=current_epoch,
                                pre_delay=500, post_delay=post_delay, model_variables_init=model_variables_dict)
    # v1_spikes = np.zeros((8, sim_duration, networks['v1']['n_nodes']), dtype=float)
    # lm_spikes = np.zeros((8, sim_duration, networks['lm']['n_nodes']), dtype=float)
    v1_spikes = np.zeros((flags.n_trials_per_angle, len(DG_angles), sim_duration, networks['v1']['n_nodes']), dtype=np.uint8)
    lm_spikes = np.zeros((flags.n_trials_per_angle, len(DG_angles), sim_duration, networks['lm']['n_nodes']), dtype=np.uint8)

    n_iterations = int(np.ceil(flags.n_trials_per_angle / per_replica_batch_size))
    chunk_size = flags.seq_len
    # num_chunks = (2500//chunk_size + 1)
    num_chunks = int(np.ceil(2500/chunk_size))
    for angle_id, angle in enumerate(DG_angles):
        print(f'Running angle {angle} ...')
        t0 = time()
        # load LGN firign rates for the given angle and calculate spiking probability
        lgn_prob = lgn_firing_probabilities_dict[angle]
        lgn_prob = tf.tile(tf.expand_dims(lgn_prob, axis=0), [per_replica_batch_size, 1, 1])
        for iter_id in range(n_iterations):
            start_idx = iter_id * per_replica_batch_size
            end_idx = min((iter_id + 1) * per_replica_batch_size, flags.n_trials_per_angle)
            iteration_length = end_idx - start_idx

            lgn_spikes = generate_spontaneous_spikes(lgn_prob)
            # Reset the memory stats
            # tf.config.experimental.reset_memory_stats('GPU:0')
            for i in range(num_chunks):
                chunk = lgn_spikes[:, i * chunk_size : (i + 1) * chunk_size, :]
                v1_z_chunk, lm_z_chunk, continuing_state = distributed_roll_out(chunk, continuing_state)
                # v1_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += v1_z_chunk.numpy()[0, :, :].astype(float)
                # lm_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += lm_z_chunk.numpy()[0, :, :].astype(float)
                v1_spikes[start_idx:end_idx, angle_id, i * chunk_size : (i + 1) * chunk_size, :] += v1_z_chunk.numpy()[:iteration_length, :, :].astype(np.uint8)
                lm_spikes[start_idx:end_idx, angle_id, i * chunk_size : (i + 1) * chunk_size, :] += lm_z_chunk.numpy()[:iteration_length, :, :].astype(np.uint8)
                        
        if angle_id == 0:
            # Raster plot for 0 degree orientation
            callbacks.single_trial_callbacks(lgn_spikes.numpy(), v1_spikes[0], lm_spikes[0], y=angle)

        # print(f'Angle {angle} done.')
        print(f'    Angle processing time: {time() - t0:.2f}s')
        for gpu_id in range(len(strategy.extended.worker_devices)):
            printgpu(gpu_id=gpu_id)

        if not flags.calculate_osi_dsi:
            break

    # save lm spikes as a pkl file
    # with open(os.path.join(logdir, 'lm_spikes.pkl'), 'wb') as f:
    #     pkl.dump(lm_spikes, f)

    if flags.calculate_osi_dsi:
        callbacks.osi_dsi_analysis(v1_spikes, lm_spikes, DG_angles)
    #     for trial_id in range(flags.n_trials_per_angle):
    #         t0 = time()
    #         # Reset the memory stats
    #         # tf.config.experimental.reset_memory_stats('GPU:0')
    #         chunk_size = flags.seq_len
    #         num_chunks = (2500//chunk_size + 1)
    #         for i in range(num_chunks):
    #             chunk = x[:, i * chunk_size : (i + 1) * chunk_size, :]
    #             v1_z_chunk, lm_z_chunk, _ = distributed_roll_out(chunk)
    #             # v1_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += v1_z_chunk.numpy()[0, :, :].astype(float)
    #             # lm_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += lm_z_chunk.numpy()[0, :, :].astype(float)
    #             v1_spikes[trial_id, angle_id, i * chunk_size : (i + 1) * chunk_size, :] += v1_z_chunk.numpy()[0, :, :].astype(np.uint8)
    #             lm_spikes[trial_id, angle_id, i * chunk_size : (i + 1) * chunk_size, :] += lm_z_chunk.numpy()[0, :, :].astype(np.uint8)
                
    #         if trial_id == 0 and angle_id == 0:
    #             # Raster plot for 0 degree orientation
    #             callbacks.single_trial_callbacks(x.numpy(), v1_spikes[0], lm_spikes[0], y=angle)
    
    #         print(f'Trial {trial_id+1}/{flags.n_trials_per_angle} - Angle {angle} done.')
    #         print(f'    Trial running time: {time() - t0:.2f}s')
    #         for gpu_id in range(len(strategy.extended.worker_devices)):
    #             printgpu(gpu_id=gpu_id)

    #         if not flags.calculate_osi_dsi:
    #             break

    # # save lm spikes as a pkl file
    # # with open(os.path.join(logdir, 'lm_spikes.pkl'), 'wb') as f:
    # #     pkl.dump(lm_spikes, f)

    # if flags.calculate_osi_dsi:
    #     callbacks.osi_dsi_analysis(v1_spikes, lm_spikes, DG_angles)


if __name__ == '__main__':
    _data_dir = 'GLIF_network'
    _results_dir = 'Simulation_results'

    absl.app.flags.DEFINE_string('task_name', 'osi_dsi_estimation' , '')
    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('restore_from', 'Intermediate_checkpoints', '')
    absl.app.flags.DEFINE_string('comment', '', '')
    absl.app.flags.DEFINE_string('interarea_weight_distribution', 'billeh_weights', '')
    absl.app.flags.DEFINE_string('delays', '100,0', '')
    absl.app.flags.DEFINE_string('optimizer', 'adam', '')
    absl.app.flags.DEFINE_string('dtype', 'float32', '')

    absl.app.flags.DEFINE_float('learning_rate', .001, '')
    absl.app.flags.DEFINE_float('rate_cost', 100., '')
    absl.app.flags.DEFINE_float('voltage_cost', 1., '')
    absl.app.flags.DEFINE_float('sync_cost', 1., '')
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
    absl.app.flags.DEFINE_integer('n_output', 10, '')
    absl.app.flags.DEFINE_integer('n_trials_per_angle', 10, '')

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
    absl.app.flags.DEFINE_boolean("average_grad_for_cell_type", False, "")
    absl.app.flags.DEFINE_boolean("reset_every_step", False, "")
    absl.app.flags.DEFINE_boolean("spontaneous_training", False, "")
    absl.app.flags.DEFINE_boolean("spontaneous_uniform_distribution_constraint", False, "")
    absl.app.flags.DEFINE_boolean("connected_areas", True, "")
    absl.app.flags.DEFINE_boolean("connected_recurrent_connections", True, "")
    absl.app.flags.DEFINE_boolean("connected_noise", True, "")
    absl.app.flags.DEFINE_boolean("calculate_osi_dsi", True, "")
    absl.app.flags.DEFINE_boolean("current_input", False, "")

    absl.app.flags.DEFINE_string("rotation", "ccw", "")

    absl.app.flags.DEFINE_string('ckpt_dir', '', '')

    absl.app.run(main)

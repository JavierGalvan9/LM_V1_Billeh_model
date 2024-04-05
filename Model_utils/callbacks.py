import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
from time import time
import pickle as pkl
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from Model_utils import stim_dataset, other_billeh_utils, load_sparse
from Model_utils.plotting_utils import InputActivityFigure, PopulationActivity
from Model_utils.model_metrics_analysis import ModelMetricsAnalysis, OneShotTuningAnalysis


def printgpu(verbose=0):
    if tf.config.list_physical_devices('GPU'):
        meminfo = tf.config.experimental.get_memory_info('GPU:0')
        current = meminfo['current'] / 1024**3
        peak = meminfo['peak'] / 1024**3
        if verbose == 0:
            print(f"GPU memory use: {current:.2f} GB / Peak: {peak:.2f} GB")
        if verbose == 1:
            return current, peak

def compose_str(metrics_values):
        _acc, _loss, _rate, _rate_loss, _voltage_loss, _osi_loss = metrics_values
        _s = f'Loss {_loss:.4f}, '
        _s += f'RLoss {_rate_loss:.4f}, '
        _s += f'VLoss {_voltage_loss:.4f}, '
        _s += f'OLoss {_osi_loss:.4f}, '
        _s += f'Accuracy {_acc:.4f}, '
        _s += f'Rate {_rate:.4f}'
        return _s


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


class Callbacks:
    def __init__(self, model, optimizer, distributed_roll_out, flags, logdir, flag_str, strategy, 
                metrics_keys, pre_delay=50, post_delay=50, checkpoint=None, model_variables_init=None, 
                save_optimizer=True, spontaneous_fr=False):
        parts = flag_str.split('_')
        n_neurons = {'v1': int(parts[1]), 'lm': int(parts[3])}
        self.n_neurons = n_neurons

        if flags.caching:
            load_fn = load_sparse.cached_load_billeh
        else:
            load_fn = load_sparse.load_billeh
        self.networks, self.lgn_inputs, self.bkg_inputs = load_fn(flags, n_neurons, flag_str=flag_str)      
        if spontaneous_fr:
            self.neuropixels_feature = 'Spontaneous rate (Hz)'
        else:
            self.neuropixels_feature = 'Ave_Rate(Hz)'  
        self.model = model
        self.flags = flags
        self.logdir = logdir
        self.strategy = strategy
        self.distributed_roll_out = distributed_roll_out
        self.metrics_keys = metrics_keys
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.step = 0
        self.total_epochs = flags.n_runs * flags.n_epochs
        self.step_running_time = []
        self.model_variables_dict = model_variables_init
        self.initial_metric_values = None
        self.summary_writer = tf.summary.create_file_writer(self.logdir)
        with open(os.path.join(self.logdir, 'config.json'), 'w') as f:
            json.dump(flags.flag_values_dict(), f, indent=4)
    
        if checkpoint is None:
            if save_optimizer:
                checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
            else:
                checkpoint = tf.train.Checkpoint(model=model)
            self.min_val_loss = float('inf')
            self.no_improve_epochs = 0
            # create a dictionary to save the values of the metric keys after each epoch
            self.epoch_metric_values = {key: [] for key in self.metrics_keys}
        else:
            # Load epoch_metric_values and min_val_loss from the file
            with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'rb') as f:
                data_loaded = pkl.load(f)
            self.epoch_metric_values = data_loaded['epoch_metric_values']
            self.min_val_loss = data_loaded['min_val_loss']
            self.no_improve_epochs = data_loaded['no_improve_epochs']

        # Manager for the best model
        self.best_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir, max_to_keep=1
        )
        # Manager for osi/dsi checkpoints 
        self.epoch_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/OSI_DSI_checkpoints', max_to_keep=None
        )


    def on_train_begin(self):
        print("---------- Training started at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        self.train_start_time = time()
        self.epoch = self.flags.run_session * self.flags.n_epochs

    def on_train_end(self, metric_values):
        self.train_end_time = time()
        self.final_metric_values = metric_values
        print("\n ---------- Training ended at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        print(f"Total time spent: {self.train_end_time - self.train_start_time:.2f} seconds")
        print(f"Average step time: {np.mean(self.step_running_time):.2f} seconds\n")
        # Determine the maximum key length for formatting the table
        max_key_length = max(len(key) for key in self.metrics_keys)

        # Start of the Markdown table
        print(f"| {'Metric':<{max_key_length}} | {'Initial Value':<{max_key_length}} | {'Final Value':<{max_key_length}} |")
        print(f"|{'-' * (max_key_length + 2)}|{'-' * (max_key_length + 2)}|{'-' * (max_key_length + 2)}|")

        n_metrics = len(self.initial_metric_values)//2
        for initial, final, key in zip(self.initial_metric_values[n_metrics:], self.final_metric_values[n_metrics:], self.metrics_keys[n_metrics:]):
            print(f"| {key:<{max_key_length}} | {initial:<{max_key_length}.3f} | {final:<{max_key_length}.3f} |")

        # Save epoch_metric_values and min_val_loss to a file
        data_to_save = {
            'epoch_metric_values': self.epoch_metric_values,
            'min_val_loss': self.min_val_loss,
            'no_improve_epochs': self.no_improve_epochs
        }
        with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'wb') as f:
            pkl.dump(data_to_save, f)

        if self.flags.n_runs > 1:
            self.plot_osi_dsi(parallel=True)

    def on_epoch_start(self):
        self.epoch += 1
        # self.step_counter.assign_add(1)
        self.epoch_init_time = time()
        date_str = dt.datetime.now().strftime('%d-%m-%Y %H:%M')
        print(f'Epoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')
        tf.print(f'\nEpoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')

    def on_epoch_end(self, x, v1_spikes, lm_spikes, y, metric_values, bkg_noise=None, verbose=True):
        self.step = 0
        if self.initial_metric_values is None:
            self.initial_metric_values = metric_values
        
        if verbose:
            print_str = '  Validation: \n' 
            val_values = metric_values[len(metric_values)//2:]
            print_str += '    ' + compose_str(val_values) 
            print(print_str)
            mem_data = printgpu(verbose=1)
            print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB'+ '\n')

        self.epoch_metric_values = {key: value + [metric_values[i]] for i, (key, value) in enumerate(self.epoch_metric_values.items())}

        if 'val_loss' in self.metrics_keys:
            val_loss_index = self.metrics_keys.index('val_loss')
            val_loss_value = metric_values[val_loss_index]
        else:
            val_loss_index = self.metrics_keys.index('train_loss')
            val_loss_value = metric_values[val_loss_index]

        self.plot_losses_curves()

        if val_loss_value < self.min_val_loss:
            self.min_val_loss = val_loss_value
            self.no_improve_epochs = 0
            # self.plot_bkg_noise(bkg_noise)
            self.noise_psc(bkg_noise)
            self.bkg_correlation_analysis(v1_spikes, lm_spikes, bkg_noise)
            self.plot_lgn_activity(x)
            self.save_best_model()
            self.plot_raster(x, v1_spikes, lm_spikes, y)
            self.plot_mean_firing_rate_boxplot(v1_spikes, lm_spikes, y)
            self.plot_tuning_analysis(v1_spikes, lm_spikes, y)
            self.plot_populations_activity(v1_spikes, lm_spikes)

            self.model_variables_dict['Best'] = {var.name: var.numpy() for var in self.model.trainable_variables}
            for var in self.model_variables_dict['Best'].keys():
                t0 = time()
                self.variable_change_analysis(var)
                print(f'Time spent in {var}: {time()-t0}')
        else:
            self.no_improve_epochs += 1
           
        # Plot osi_dsi if only 1 run and the osi/dsi period is reached
        # if self.flags.n_runs == 1 and (self.epoch % self.flags.osi_dsi_eval_period == 0 or self.epoch==1):
        #     self.plot_osi_dsi(parallel=False)

        with self.summary_writer.as_default():
            for k, v in zip(self.metrics_keys, metric_values):
                tf.summary.scalar(k, v, step=self.epoch)

        # EARLY STOPPING CONDITIONS
        if (0 < self.flags.max_time < (time() - self.epoch_init_time) / 3600):
            print(f'[ Maximum optimization time of {self.flags.max_time:.2f}h reached ]')
            stop = True
        elif self.no_improve_epochs >= 200:
            print("Early stopping: Validation loss has not improved for 50 epochs.")
            stop = True  
        else:
            stop = False

        return stop

    def on_step_start(self):
        self.step += 1
        self.step_init_time = time()

    def on_step_end(self, train_values, y, verbose=True):
        self.step_running_time.append(time() - self.step_init_time)
        if verbose:
            print_str = f'  Step {self.step:2d}/{self.flags.steps_per_epoch} - Angle: {y[0][0]:.2f}\n'
            print_str += '    ' + compose_str(train_values)
            print(print_str)
            print(f'    Step running time: {time() - self.step_init_time:.2f}s')
            mem_data = printgpu(verbose=1)
            print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB')
        
    def save_best_model(self):
        # self.step_counter.assign_add(1)
        print(f'[ Saving the model at epoch {self.epoch} ]')
        try:
            p = self.best_manager.save(checkpoint_number=self.epoch)
            print(f'Model saved in {p}\n')
        except:
            print("Saving failed. Maybe next time?")

    def plot_losses_curves(self):
        # plotting_metrics = ['val_loss', 'val_firing_rate', 'val_rate_loss', 'val_voltage_loss']
        plotting_metrics = ['val_loss', 'val_osi_loss', 'val_rate_loss', 'val_voltage_loss']
        images_dir = os.path.join(self.logdir, 'Loss_curves')
        os.makedirs(images_dir, exist_ok=True)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        for i, metric_key in enumerate(plotting_metrics):
            ax = axs[i // 2, i % 2]
            ax.plot(range(1, self.epoch + 1), self.epoch_metric_values[metric_key])
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_key)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f'losses_curves_epoch.png'), dpi=300, transparent=False)
        plt.close()
    
    def plot_raster(self, x, v1_spikes, lm_spikes, y):
        x = x.numpy()
        v1_spikes = v1_spikes.numpy()
        lm_spikes = lm_spikes.numpy()
        images_dir = os.path.join(self.logdir, 'Raster_plots')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.networks,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=self.pre_delay,
                                    stimuli_end_time=self.flags.seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=True,
                                    )
        graph(x, v1_spikes, lm_spikes)

    def plot_bkg_noise(self, x, bin_width_ms=10):
        x = x.numpy()[0, :, :]
        x_mean = np.mean(x, axis=1)
        
        # Calculate number of bins
        num_bins = int(x_mean.shape[0] / (bin_width_ms * 0.001))  # Convert bin width to seconds
        
        # Reshape into bins
        x_mean_reshaped = [np.mean(x_mean[i:i + bin_width_ms]) for i in range(0, len(x_mean), bin_width_ms)]
        # x_mean_reshaped = np.mean(x_mean[:int(num_bins * (bin_width_ms * 0.001))], axis=0)
        
        # Generate time axis
        time_axis = np.arange(0, len(x_mean), bin_width_ms)
        
        plt.figure(figsize=(10, 5))
        plt.plot(time_axis, x_mean_reshaped)
        plt.title('Mean BKG input')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean BKG weight')
        os.makedirs(os.path.join(self.logdir, 'Populations activity'), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, 'Populations activity', f'BKG_activity_epoch_{self.epoch}.png'))

    def plot_lgn_activity(self, x):
        x = x.numpy()[0, :, :]
        x_mean = np.mean(x, axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(x_mean)
        plt.title('Mean input activity')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean input activity')
        os.makedirs(os.path.join(self.logdir, 'Populations activity'), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, 'Populations activity', f'LGN_population_activity_epoch_{self.epoch}.png'))

    def plot_populations_activity(self, v1_spikes, lm_spikes):
        z = [v1_spikes.numpy(), lm_spikes.numpy()]

        # Plot the mean firing rate of the population of neurons
        for area_id, area in enumerate(['v1', 'lm']):
            neurons = self.networks[area]['n_nodes']
            filename = f'{area}_Epoch_{self.epoch}'
            Population_activity = PopulationActivity(n_neurons=neurons, network=self.networks[area], 
                                                    stimuli_init_time=self.pre_delay, stimuli_end_time=self.flags.seq_len-self.post_delay, 
                                                    image_path=self.logdir, filename=filename, data_dir=self.flags.data_dir)
            Population_activity(z[area_id], area=area, plot_core_only=True, bin_size=10)

    def bkg_correlation_analysis(self, v1_spikes, lm_spikes, bkg_noise):
        z = v1_spikes.numpy()
        z = np.mean(z[0,:,:], axis=1)
        noise = self.noise_psc(bkg_noise)
        noise = np.mean(noise, axis=1)
        # Binning data into 10 ms bins
        bin_size = 10
        binned_noise = [np.mean(noise[i:i + bin_size]) for i in range(0, len(noise), bin_size)]
        # print(min(binned_noise), max(binned_noise))
        binned_spikes = [np.mean(z[i:i + bin_size]) for i in range(0, len(z), bin_size)]

        # Setting the range of lags to explore (for example, ±50 time steps)
        max_lag = 50
        lags = np.arange(-max_lag, max_lag + 1)
        # Computing the cross-correlation
        z_centered = z - np.mean(z)
        noise_centered = noise - np.mean(noise)
        # The 'full' mode returns the cross-correlation at each possible lag
        cross_corr_raw = correlate(z_centered, noise_centered, mode='full')
        # Normalization factor (standard approach for signals of different lengths)
        n = len(z)
        std_z = np.std(z)
        std_noise = np.std(noise)
        norm_factor = std_z * std_noise * n
        cross_corr_normalized = cross_corr_raw / norm_factor
        
        central_index = len(cross_corr_normalized) // 2
        cross_corr = cross_corr_normalized[central_index - max_lag: central_index + max_lag + 1]
        # Finding the lag with the maximum correlation
        max_corr_lag = lags[np.argmax(cross_corr)]

        # Shuffle noise and compute cross-correlation
        shuffled_noise = np.random.permutation(noise_centered)
        shuffled_cross_corr = correlate(z_centered, shuffled_noise, mode='full')
        # Normalize the shuffled cross-correlation
        normalized_shuffled_cross_corr = shuffled_cross_corr / norm_factor
        shuffled_cross_corr = normalized_shuffled_cross_corr[central_index - max_lag: central_index + max_lag + 1]

        # Plotting the binned data and correlation
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Binned BKG Noise PSC', color=color)
        ax1.plot(binned_noise, color=color)
        ax1.set_ylim([0, 4])
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Binned V1 Neuron Spikes', color=color)  # we already handled the x-label with ax1
        ax2.plot(binned_spikes, color=color)
        ax2.set_ylim([0, 0.03])
        ax2.tick_params(axis='y', labelcolor=color)

        # Second subplot
        ax3.plot(lags, cross_corr, color='purple')
        ax3.plot(lags, shuffled_cross_corr, color='gray', label='Shuffled')
        ax3.axvline(x=max_corr_lag, color='red', linestyle='--', label=f'Max Correlation at Lag = {max_corr_lag}')
        ax3.set_xlabel('Lag [ms]')
        ax3.set_ylabel('Normalized Cross-correlation')
        ax3.set_ylim(-1, 1)
        ax3.set_title('Cross-correlation between Background Noise and V1 Neuron Spikes')
        ax3.legend()
        ax3.grid(True)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        os.makedirs(os.path.join(self.logdir, 'Populations activity'), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, 'Populations activity', f'bkg_correlation_analysis_epoch_{self.epoch}.png'))

    def noise_psc(self, bkg_noise):
        bkg_noise = bkg_noise.numpy()[0, :, :2*self.n_neurons['v1']]
        bkg_noise = tf.constant(bkg_noise, dtype=tf.float32)
        network = self.networks['v1']
        lgn_input = self.lgn_inputs['v1']
        bkg_input = self.bkg_inputs['v1']
        dt=1
        _compute_dtype = tf.float32
        _params = network['node_params']
        _n_neurons = network['n_nodes']
        # Determine the synaptic dynamic parameters for each of the 5 basis receptors
        tau_syns = np.array([5.5, 8.5, 2.8, 5.8])
        syn_decay = np.exp(-dt / tau_syns)
        syn_decay = tf.constant(syn_decay, dtype=_compute_dtype)
        psc_initial = np.e / tau_syns
        psc_initial = tf.constant(psc_initial, dtype=_compute_dtype)

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

        _n_max_receptors, neuron_mappings, original_receptor_ids = process_receptors(all_postsynaptic_indices, all_receptor_ids)
        # create a repetion of the range(0, _n_max_receptors) for every neuron
        syn_decay = tf.gather(syn_decay, original_receptor_ids, axis=0)
        psc_initial = tf.gather(psc_initial, original_receptor_ids, axis=0)

        syn_decay = tf.reshape(syn_decay, (_n_neurons, _n_max_receptors))
        psc_initial = tf.reshape(psc_initial, (_n_neurons, _n_max_receptors))

        def update_psc(psc, psc_rise, rec_inputs):
            new_psc_rise = psc_rise * syn_decay + rec_inputs * psc_initial
            new_psc = psc * syn_decay + dt * syn_decay * psc_rise
            return new_psc, new_psc_rise

        noise_current = []
        dtype = tf.float32
        batch_size = 1

        psc_rise = tf.zeros((batch_size, _n_neurons, _n_max_receptors), dtype)
        psc = tf.zeros((batch_size, _n_neurons, _n_max_receptors), dtype)

        for step in np.arange(bkg_noise.shape[0]):
            bkg_noise_step = tf.reshape(bkg_noise[step,:], (batch_size, _n_neurons, _n_max_receptors))
            noise_current.append(tf.reduce_sum(psc[0], -1))
            psc, psc_rise = update_psc(psc, psc_rise, bkg_noise_step)

        noise_current = np.array(noise_current)
        
        return noise_current


    def plot_mean_firing_rate_boxplot(self, v1_spikes, lm_spikes, y):
        v1_spikes = v1_spikes.numpy()
        lm_spikes = lm_spikes.numpy()
        y = y.numpy()
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{self.neuropixels_feature}')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        for axis_id, spikes, area in zip([0, 1], [v1_spikes, lm_spikes], ['v1', 'lm']):
            metrics_analysis = ModelMetricsAnalysis(self.networks[area], neuropixels_feature=self.neuropixels_feature, data_dir=self.flags.data_dir, n_trials=1,
                                                    analyze_core_only=True, drifting_gratings_init=self.pre_delay, drifting_gratings_end=self.flags.seq_len-self.post_delay, 
                                                    area=area, directory=boxplots_dir, filename=f'{area}_epoch_{self.epoch}')
            metrics_analysis(spikes, y, axis=axs[axis_id])     

        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'epoch_{self.epoch}.png'), dpi=300, transparent=False)
        plt.close()

    def plot_tuning_analysis(self, v1_spikes, lm_spikes, y):
        v1_spikes = v1_spikes.numpy()
        lm_spikes = lm_spikes.numpy()
        y = y.numpy()
        images_dir = os.path.join(self.logdir, 'OneShotTuningAnalysis')
        os.makedirs(images_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        for axis_id, spikes, area in zip([0, 1], [v1_spikes, lm_spikes], ['v1', 'lm']):
            tuning_analyzer = OneShotTuningAnalysis(self.networks[area], data_dir=self.flags.data_dir, area=area, 
                                                    directory=images_dir, analyze_core_only=True,
                                                    drifting_gratings_init=self.pre_delay, drifting_gratings_end=self.flags.seq_len-self.post_delay,
                                                    )
            tuning_analyzer(spikes, y)
            tuning_analyzer.plot_tuning_curves(self.epoch, remove_zero_rate_neurons=True)
            tuning_analyzer.plot_max_rate_boxplots(self.epoch, remove_zero_rate_neurons=True, axis=axs[axis_id])
        
        images_dir = os.path.join(images_dir, 'Max_rate_boxplot')
        os.makedirs(images_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f'epoch_{self.epoch}.png'), dpi=300, transparent=False)
        plt.close()

    def variable_change_analysis(self, variable):
        if 'rest_of_brain_weights' in variable or 'sparse_input_weights' in variable:
            area = variable.split('_')[0]
            self.node_to_pop_weights_analysis(variable=variable, area=area)
        elif 'sparse_recurrent_weights' in variable:
            area = variable.split('_')[0] 
            self.pop_to_pop_weights_analysis(self.networks[area]['synapses']['indices'], variable=variable, 
            source_area=area, target_area=area)
        elif 'sparse_interarea_weights' in variable:
            source_area = variable.split('_')[-1][:2] 
            target_area = variable.split('_')[0] 
            self.pop_to_pop_weights_analysis(self.networks[target_area]['interarea_synapses'][source_area]['indices'], variable=variable, 
            source_area=source_area, target_area=target_area)
    
    def node_to_pop_weights_analysis(self, variable='', area=''):
        pop_names = other_billeh_utils.pop_names(self.networks[area])
        pop_names = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
        # Create DataFrame with all the necessary data
        df = pd.DataFrame({
            'Cell type': pop_names * 2,  # Duplicate node names for initial and final weights
            'Weight': self.model_variables_dict['Initial'][variable].tolist() + self.model_variables_dict['Best'][variable].tolist(),  # Combine initial and final weights
            'State': ['Initial'] * len(self.model_variables_dict['Initial'][variable]) + ['Final'] * len(self.model_variables_dict['Best'][variable])  # Distinguish between initial and final weights
        })

        # Sort the dataframe by Node Name and then by Type to ensure consistent order
        df = df.sort_values(['Cell type', 'State'])

        # Plotting
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))

        fig = plt.figure(figsize=(12, 6))
        hue_order = ['Initial', 'Final']
        # sns.boxplot(x='Node Name', y='Weight Change', data=df)
        sns.barplot(x='Cell type', y='Weight', hue='State', hue_order=hue_order, data=df)
        # sns.boxplot(x='Node Name', y='Weight', hue='Type', hue_order=hue_order, data=df)
        # plt.axhline(0, color='black', linewidth=1)  # include a horizontal black line at 0
        plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
        plt.title(f'{variable}')
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_analysis(self, indices, variable='', source_area='', target_area=''):
        source_pop_names = other_billeh_utils.pop_names(self.networks[source_area])
        source_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_billeh_utils.pop_names(self.networks[target_area])
        target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        ### Initial Weight ###
        weight_changes = self.model_variables_dict['Best'][variable] - self.model_variables_dict['Initial'][variable]
        df = pd.DataFrame({'Post Name': post_cell_types, 'Pre_names':pre_cell_types, 
                            'Initial weight': self.model_variables_dict['Initial'][variable], 'Final weight': self.model_variables_dict['Best'][variable], 
                            'Weight Change': weight_changes})
        
        # Calculate global min and max for color normalization
        global_grouped_df = df.groupby(['Pre_names', 'Post Name'])[['Initial weight', 'Final weight']].mean().reset_index()
        global_min = global_grouped_df[['Initial weight', 'Final weight']].min().min()
        global_max = global_grouped_df[['Initial weight', 'Final weight']].max().max()
        # global_min = df[['Initial weight', 'Final weight']].min().min()
        # global_max = df[['Initial weight', 'Final weight']].max().max()

        # Plot for Initial Weight
        grouped_df = df.groupby(['Pre_names', 'Post Name'])['Initial weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post Name', values='Initial weight')
        # Plot heatmap
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig = plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        plt.xlabel(f'{target_area}')
        plt.ylabel(f'{source_area}')
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Initial Weight')
        plt.savefig(os.path.join(boxplots_dir, f'Initial_weight.png'), dpi=300, transparent=False)
        plt.close()

        ### Final Weight ###
        grouped_df = df.groupby(['Pre_names', 'Post Name'])['Final weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post Name', values='Final weight')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        plt.xlabel(f'{target_area}')
        plt.ylabel(f'{source_area}')
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Final Weight')
        plt.savefig(os.path.join(boxplots_dir, f'Final_weight.png'), dpi=300, transparent=False)
        plt.close()

        ### Weight change ###
        grouped_df = df.groupby(['Pre_names', 'Post Name'])['Weight Change'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post Name', values='Weight Change')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0)
        plt.xlabel(f'{target_area}')
        plt.ylabel(f'{source_area}')
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Weight Change')
        plt.savefig(os.path.join(boxplots_dir, f'Weight_change.png'), dpi=300, transparent=False)
        plt.close()

    def get_osi_dsi_dataset_fn(self, regular=False):
        def _f(input_context):
            post_delay = self.flags.seq_len - (2500 % self.flags.seq_len)
            _data_set = stim_dataset.generate_drifting_grating_tuning(
                seq_len=2500+post_delay,
                pre_delay=500,
                post_delay = post_delay,
                n_input=self.flags.n_input,
                regular=regular
            ).batch(1)
                        
            return _data_set
        return _f
    
    def plot_osi_dsi(self, parallel=False):
        print('Starting to plot OSI and DSI...')
        # Save the checkpoint to reload weights in the osi_dsi_estimator
        if parallel:
            p = self.epoch_manager.save(checkpoint_number=self.epoch)
            print(f'Checkpoint model saved in {p}\n')
        else:              
            osi_dsi_data_set = self.strategy.distribute_datasets_from_function(self.get_osi_dsi_dataset_fn(regular=True))
            
            sim_duration = (2500//self.flags.seq_len + 1) * self.flags.seq_len
            n_trials_per_angle = 10
            v1_spikes = np.zeros((8, sim_duration, self.networks['v1']['n_nodes']), dtype=float)
            lm_spikes = np.zeros((8, sim_duration, self.networks['lm']['n_nodes']), dtype=float)
            DG_angles = np.arange(0, 360, 45)
            for trial_id in range(n_trials_per_angle):
                test_it = iter(osi_dsi_data_set)
                for angle_id, angle in enumerate(range(0, 360, 45)):
                    t0 = time()
                    x, y, _, w = next(test_it)
                    chunk_size = self.flags.seq_len
                    num_chunks = (2500//chunk_size + 1)
                    for i in range(num_chunks):
                        chunk = x[:, i * chunk_size : (i + 1) * chunk_size, :]
                        v1_z_chunk, lm_z_chunk = self.distributed_roll_out(chunk, y, w, output_spikes=True)
                        v1_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += v1_z_chunk.numpy()[0, :, :].astype(float)
                        lm_spikes[angle_id, i * chunk_size : (i + 1) * chunk_size, :] += lm_z_chunk.numpy()[0, :, :].astype(float)

                    if trial_id == 0 and angle_id == 0:
                        # Raster plot for 0 degree orientation
                        lgn_spikes = x[:, :2500, :].numpy()
                        z_v1 = v1_spikes[:, :2500, :]
                        z_lm = lm_spikes[:, :2500, :]
                        images_dir = os.path.join(self.logdir, 'Raster_plots_OSI_DSI')
                        os.makedirs(images_dir, exist_ok=True)
                        graph = InputActivityFigure(
                                                    self.networks,
                                                    self.flags.data_dir,
                                                    images_dir,
                                                    filename=f'Epoch_{self.epoch}',
                                                    frequency=self.flags.temporal_f,
                                                    stimuli_init_time=500,
                                                    stimuli_end_time=2500,
                                                    reverse=False,
                                                    plot_core_only=True,
                                                    )
                        graph(lgn_spikes, z_v1, z_lm)

                    print(f'Trial {trial_id}/{n_trials_per_angle} - Angle {angle} done.')
                    print(f'    Trial running time: {time() - t0:.2f}s')
                    mem_data = printgpu(verbose=1)
                    print(f'    Memory consumption (current - peak): {mem_data[0]:.2f} GB - {mem_data[1]:.2f} GB\n')
        
            # Average the spikes over the number of trials
            v1_spikes = v1_spikes/n_trials_per_angle
            v1_spikes = v1_spikes[:, :2500, :]
            lm_spikes = lm_spikes/n_trials_per_angle
            lm_spikes = lm_spikes[:, :2500, :]

            # Do the OSI/DSI analysis 
            boxplots_dir = os.path.join(self.logdir, 'Boxplots_OSI_DSI')
            os.makedirs(boxplots_dir, exist_ok=True)
            for spikes, area in zip([v1_spikes, lm_spikes], ['v1', 'lm']):
                metrics_analysis = ModelMetricsAnalysis(self.networks[area], data_dir=self.flags.data_dir,
                                                        drifting_gratings_init=500, drifting_gratings_end=2500,
                                                        area=area, analyze_core_only=True, directory=boxplots_dir, filename=f'Epoch_{self.epoch}')
                metrics_analysis(spikes, DG_angles)
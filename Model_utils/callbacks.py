import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
import subprocess
from time import time
import pickle as pkl
from numba import njit
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy.signal import correlate, welch
from scipy.stats import ks_2samp
from Model_utils import stim_dataset, other_billeh_utils
from Model_utils.models import process_receptors
from Model_utils.plotting_utils import InputActivityFigure, PopulationActivity
from Model_utils.model_metrics_analysis import ModelMetricsAnalysis, OneShotTuningAnalysis
from Model_utils.model_metrics_analysis import calculate_Firing_Rate, get_borders, draw_borders

# sns.set(style="ticks")
# plt.rcParams['text.usetex'] = True

def printgpu(gpu_id=0):
    if tf.config.list_physical_devices('GPU'):
        # Check TensorFlow memory info
        meminfo = tf.config.experimental.get_memory_info(f'GPU:{gpu_id}')
        current = meminfo['current'] / 1024**3
        peak = meminfo['peak'] / 1024**3
        print(f'    TensorFlow GPU {gpu_id} Memory Usage: {current:.2f} GiB, Peak Usage: {peak:.2f} GiB')
        # Check GPU memory using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.free,memory.total', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, encoding='utf-8'
        )  # MiB
        # Split output into lines for each GPU
        gpu_memory_info = result.stdout.strip().split('\n')
        if gpu_id is not None:
            # Display memory info for a specific GPU
            if gpu_id < len(gpu_memory_info):
                used, free, total = [float(x)/1024 for x in gpu_memory_info[gpu_id].split(',')]
                print(f"    Total GPU {gpu_id} Memory Usage: Used: {used:.2f} GiB, Free: {free:.2f} GiB, Total: {total:.2f} GiB")
            else:
                print(f"    Invalid GPU ID: {gpu_id}. Available GPUs: {len(gpu_memory_info)}")
        else:
            # Display memory info for all GPUs
            for i, info in enumerate(gpu_memory_info):
                used, free, total = [float(x)/1024 for x in info.split(',')]
                print(f"    Total GPU {gpu_id} Memory Usage: Used: {used:.2f} GiB, Free: {free:.2f} GiB, Total: {total:.2f} GiB")
                
def compose_str(metrics_values, mode='classification'):
        if mode == 'classification':
            _acc, _loss, _rate, _rate_loss, _voltage_loss, _regularizer_loss, _classification_loss = metrics_values
            _s = f'Loss {_loss:.4f}, '
            _s += f'RLoss {_rate_loss:.4f}, '
            _s += f'VLoss {_voltage_loss:.4f}, '
            _s += f'RegLoss {_regularizer_loss:.4f}, '
            _s += f'CLoss {_classification_loss:.4f}, '
            _s += f'Accuracy {_acc:.4f}, '
            _s += f'Rate {_rate:.4f}'
        else:
            _loss, _rate, _rate_loss, _voltage_loss, _regularizer_loss, _osi_dsi_loss, _sync_loss = metrics_values
            _s = f'Loss {_loss:.4f}, '
            _s += f'RLoss {_rate_loss:.4f}, '
            _s += f'VLoss {_voltage_loss:.4f}, '
            _s += f'RegLoss {_regularizer_loss:.4f}, '
            _s += f'OLoss {_osi_dsi_loss:.4f}, '
            _s += f'SLoss {_sync_loss:.4f}, '
            _s += f'Rate {_rate:.4f}'
            
        return _s

def compute_ks_statistics(df, metric='Weight', min_n_sample=15):
    """
    Compute the Kolmogorov-Smirnov statistic and similarity scores for each cell type in the dataframe.
    Parameters:
    - df: pd.DataFrame, contains data with columns 'data_type' and 'Ave_Rate(Hz)', and indexed by cell type.
    Returns:
    - mean_similarity_score: float, the mean of the similarity scores computed across all cell types.
    """
    # Get unique cell types
    # cell_types = df.index.unique()
    cell_types = df['Post_names'].unique()
    # Initialize a dictionary to store the results
    ks_results = {}
    similarity_scores = {}
    # Iterate over cell types
    for cell_type in cell_types:
        # Filter data for current cell type from two different data types
        # df1 = df.loc[(df.index == cell_type) & (df['data_type'] == 'V1/LM GLIF model'), metric]
        # df2 = df.loc[(df.index == cell_type) & (df['data_type'] == 'Neuropixels'), metric]
        df1 = df.loc[(df['Post_names'] == cell_type) & (df['Weight Type'] == 'Initial weight'), metric]
        df2 = df.loc[(df['Post_names'] == cell_type) & (df['Weight Type'] == 'Final weight'), metric]
        # Drop NA values
        df1.dropna(inplace=True)
        df2.dropna(inplace=True)
        # Calculate the Kolmogorov-Smirnov statistic
        if len(df1) >= min_n_sample and len(df2) >= min_n_sample:
            ks_stat, p_value = ks_2samp(df1, df2)
            ks_results[cell_type] = (ks_stat, p_value)
            similarity_scores[cell_type] = 1 - ks_stat

    # Calculate the mean of the similarity scores and return it
    mean_similarity_score = np.mean(list(similarity_scores.values()))
    return mean_similarity_score

# Define a function to compute the exponential decay of a spike train
def exponential_decay_filter(spike_train, tau=20):
    decay_factor = np.exp(-1/tau)
    continuous_signal = np.zeros_like(spike_train, dtype=float)
    continuous_signal[0] = spike_train[0]
    for i in range(1, len(spike_train)):
        continuous_signal[i] = decay_factor * continuous_signal[i-1] + spike_train[i]
    return continuous_signal

# Define a function to calculate the power spectrum
def calculate_power_spectrum(signal, fs=1000):
    f, Pxx = welch(signal, fs, nperseg=100)
    return f, Pxx

@njit
def pop_fano(spikes, bin_sizes):
    fanos = np.zeros(len(bin_sizes))
    for i, bin_width in enumerate(bin_sizes):
        bin_size = int(np.round(bin_width * 1000))
        max_index = spikes.shape[0] // bin_size * bin_size
        # drop the last bin if it is not complete
        # sum over neurons to get the spike counts
        # trimmed_spikes = np.sum(spikes[:max_index, :], axis=1) 
        trimmed_spikes = spikes[:max_index]
        trimmed_spikes = np.reshape(trimmed_spikes, (max_index // bin_size, bin_size))
        # sum over the bins
        sp_counts = np.sum(trimmed_spikes, axis=1)
        # Calculate the mean of the spike counts
        mean_count = np.mean(sp_counts)
        if mean_count > 0:
            # Calculate the Fano Factor
            fanos[i] = np.var(sp_counts) / mean_count
                 
    return fanos

# def pop_fano_tf(spikes, bin_sizes, epsilon=1e-7):
#     # transpose the spikes tensor to have the shape (seq_len, samples)
#     all_spikes_transposed = tf.transpose(spikes)
#     # Initialize the Fano factors tensor
#     fanos = tf.TensorArray(dtype=tf.float32, size=len(bin_sizes))
#     for i, bin_width in enumerate(bin_sizes):
#         # drop the last entry to avoid last bin smaller size effect
#         bin_size = int(np.round(bin_width * 1000))
#         max_index = all_spikes_transposed.shape[0] // bin_size * bin_size
#         trimmed_spikes = all_spikes_transposed[:max_index, :]
#         # Reshape the spikes tensor 
#         trimmed_spikes = tf.reshape(trimmed_spikes, [max_index // bin_size, bin_size, -1])
#         # Calculate the number of spikes in each bin
#         sp_counts = tf.reduce_sum(trimmed_spikes, axis=1)
#         # Calculate the mean and variance of spike counts
#         mean_count = tf.reduce_mean(sp_counts, axis=0)
#         mean_count = tf.maximum(mean_count, epsilon)
#         var_count = tf.math.reduce_variance(sp_counts, axis=0)
#         fano = var_count / mean_count
#         fanos = fanos.write(i, fano)

#     return fanos.stack()

# create a class for callbacks in other training sessions (e.g. validation, testing)
class OsiDsiCallbacks:
    def __init__(self, networks, lgn_inputs, bkg_inputs, flags, logdir, current_epoch=0,
                pre_delay=50, post_delay=50, model_variables_init=None):
        self.n_neurons = {'v1': networks['v1']['n_nodes'], 'lm': networks['lm']['n_nodes']}
        self.networks = networks
        self.lgn_inputs = lgn_inputs
        self.bkg_inputs = bkg_inputs
        self.flags = flags
        self.logdir = logdir
        # if self.flags.connected_areas and self.flags.connected_recurrent_connections and self.flags.connected_noise:
        #     self.images_dir = self.logdir
        # else:
        #     self.images_dir = os.path.join(self.logdir, f'connected_areas_{self.flags.connected_areas}_conn_rec_{self.flags.connected_recurrent_connections}_conn_noise_{self.flags.connected_noise}')
        self.images_dir = os.path.join(self.logdir, f'connected_areas_{self.flags.connected_areas}_conn_rec_{self.flags.connected_recurrent_connections}_conn_noise_{self.flags.connected_noise}')
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.current_epoch = current_epoch
        self.model_variables_dict = model_variables_init
        # Analize changes in trainable variables.
        if self.model_variables_dict is not None:
            for var in self.model_variables_dict['Best'].keys():
                t0 = time()
                self.trainable_variable_change_heatmaps_and_distributions(var)
                print(f'Time spent in {var}: {time()-t0}')

    def trainable_variable_change_heatmaps_and_distributions(self, variable):
        area = variable.split('_')[0]
        node_types_voltage_scale = (self.networks[area]['node_params']['V_th'] - self.networks[area]['node_params']['E_L']).astype(np.float16)
        node_type_ids = self.networks[area]['node_type_ids']
        
        if 'rest_of_brain_weights' in variable:
            voltage_scale = node_types_voltage_scale[node_type_ids[self.bkg_inputs[area]['indices'][:, 0]]]
            self.node_to_pop_weights_analysis(self.bkg_inputs[area]['indices'], variable=variable, area=area, voltage_scale=voltage_scale)
        elif'sparse_input_weights' in variable:
            voltage_scale = node_types_voltage_scale[node_type_ids[self.lgn_inputs[area]['indices'][:, 0]]]
            self.node_to_pop_weights_analysis(self.lgn_inputs[area]['indices'], variable=variable, area=area, voltage_scale=voltage_scale)
        elif 'sparse_recurrent_weights' in variable:
            indices = self.networks[area]['synapses']['indices']
            voltage_scale = node_types_voltage_scale[node_type_ids[indices[:, 0]]]
            self.pop_to_pop_weights_analysis(indices, variable=variable, 
                                             source_area=area, target_area=area, voltage_scale=voltage_scale)
            self.pop_to_pop_weights_distribution(indices, variable=variable, 
                                                source_area=area, target_area=area, voltage_scale=voltage_scale)
        elif 'sparse_interarea_weights' in variable:
            source_area = variable.split('_')[-1][:2] 
            indices = self.networks[area]['interarea_synapses'][source_area]['indices']
            voltage_scale = node_types_voltage_scale[node_type_ids[indices[:, 0]]]
            self.pop_to_pop_weights_analysis(indices, variable=variable, 
                                            source_area=source_area, target_area=area, voltage_scale=voltage_scale)
            self.pop_to_pop_weights_distribution(indices, variable=variable, 
                                                source_area=source_area, target_area=area, voltage_scale=voltage_scale)

    def node_to_pop_weights_analysis(self, indices, variable='', area='', voltage_scale=None):
        pop_names = other_billeh_utils.pop_names(self.networks[area])
        target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
        # print(variable, indices[:, 0].shape, voltage_scale.shape)
        # if 'rest_of_brain_weights' in variable:
        #     post_indices =  np.repeat(indices[:, 0], 4)
        #     voltage_scale = np.repeat(voltage_scale, 4)
        # else:
        #     post_indices = indices[:, 0]
        post_indices = indices[:, 0]
        post_cell_types = [target_cell_types[i] for i in post_indices]

        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        # Create DataFrame with all the necessary data
        df = pd.DataFrame({
            'Post_names': post_cell_types * 2,  # Duplicate node names for initial and final weights
            'Weight': initial_weights.tolist() + final_weights.tolist(),  # Combine initial and final weights
            'Weight Type': ['Initial weight'] * len(initial_weights) + ['Final weight'] * len(final_weights)  # Distinguish between initial and final weights
        })

        # Count the number of cell_types fro each type
        # cell_type_counts = df['Post_names'].value_counts()

        # Sort the dataframe by Node Name and then by Type to ensure consistent order
        df = df.sort_values(['Post_names', 'Weight Type'])

        # Plotting
        boxplots_dir = os.path.join(self.images_dir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        # fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        fig = plt.figure(figsize=(12, 6))
        #get the axis of the figure
        ax = fig.gca()
        similarity_score = compute_ks_statistics(df, metric='Weight', min_n_sample=15)
        hue_order = ['Initial weight', 'Final weight']
        # sns.boxplot(x='Node Name', y='Weight Change', data=df)
        # sns.barplot(x='Post_names', y='Weight', hue='State', hue_order=hue_order, data=df)
        sns.boxplot(x='Post_names', y='Weight', hue='Weight Type', hue_order=hue_order, data=df, ax=ax, width=0.7, fliersize=1.)
        # plt.axhline(0, color='black', linewidth=1)  # include a horizontal black line at 0
        ax.set_yscale('log')
        # ax.set_ylim(bottom=0)  # Set bottom limit of y-axis to 0
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_title(f'{variable}')
        ax.legend(loc='upper right')
        if similarity_score is not None:
            ax.text(0.9, 0.1, f'S: {similarity_score:.2f}', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_distribution(self, indices, variable='', source_area='v1', target_area='v1', voltage_scale=None):
        source_pop_names = other_billeh_utils.pop_names(self.networks[source_area])
        source_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_billeh_utils.pop_names(self.networks[target_area])
        target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        df = pd.DataFrame({
            'Post_names': post_cell_types,
            'Pre_names': pre_cell_types,
            'Initial weight': initial_weights,
            'Final weight': final_weights,
        })

        # Melt DataFrame to long format
        df_melted = df.melt(id_vars=['Post_names', 'Pre_names'], value_vars=['Initial weight', 'Final weight'], 
                            var_name='Weight Type', value_name='Weight')
        df_melted['Weight'] = np.abs(df_melted['Weight'])
        # Create directory for saving plots
        boxplots_dir = os.path.join(self.images_dir, f'Boxplots/{variable}_distribution')
        os.makedirs(boxplots_dir, exist_ok=True)
        # Establish the order of the neuron types in the boxplots
        cell_type_order = np.sort(df['Pre_names'].unique())
        target_type_order = np.sort(df['Post_names'].unique())

        # Define the palette
        palette = {"Initial weight": "#87CEEB", "Final weight": "#FFA500"}
        # Create subplots
        num_pre_names = len(cell_type_order)

        num_columns = 4
        num_rows = (num_pre_names + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(24, 6 * num_rows))
        # Flatten the axes array and handle the first row separately
        axes = axes.flatten()
        for i, pre_name in enumerate(cell_type_order):
            if num_pre_names == 17 and i == 0:
                i = 1
            elif num_pre_names == 17 and i != 0:
                i += 3
            
            ax = axes[i] 
            subset_df = df_melted[df_melted['Pre_names'] == pre_name]
            similarity_score = compute_ks_statistics(subset_df, metric='Weight', min_n_sample=15)
            # subset_cell_type_order = np.sort(subset_df['Post_names'].unique())
            # Create boxplot for Initial and Final weights
            sns.boxplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', order=target_type_order, ax=ax, palette=palette, 
                        width=0.7, fliersize=1.)
            # sns.violinplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', order=subset_cell_type_order, ax=ax, palette=palette, width=0.7,
            #                split=True, inner="quart", gap=0.2)
            ax.set_title(f'Source Cell Type: {pre_name}')
            ax.set_ylabel(r'$\vert$ Synaptic Weight (pA)$\vert$', fontsize=12)
            # ax.set_yscale('symlog', linthresh=0.001)
            ax.set_yscale('log')
            if i % num_columns == 0 or i == 1:  # First column
                ax.set_ylabel(r'$\vert$ Synaptic Weight (pA)$\vert$', fontsize=12)
            else:
                ax.set_ylabel('')
            if i >= (num_rows - 1) * num_columns:
                ax.set_xlabel('Target Cell Type')
            else:
                ax.set_xlabel('')
            ax.tick_params(axis="x", labelrotation=90)
            # Apply shadings to each layer
            xticklabel = ax.get_xticklabels()
            borders = get_borders(xticklabel)
            # change y limit
            # if 'E' in pre_name:
            #     bottom_limit = 0
            #     upper_limit = 1000
            # else:
            #     bottom_limit = -500
            #     upper_limit = 0
            bottom_limit = 0.001
            upper_limit = 1000
            ax.set_ylim(bottom=bottom_limit, top=upper_limit)
            # get the current ylim
            ylim = ax.get_ylim()
            draw_borders(ax, borders, ylim)
            # ax.legend(loc='best')
            if i == 1 and num_pre_names == 17:
                ax.legend(loc='upper left')
            else:
                ax.get_legend().remove()

            if similarity_score is not None:
                ax.text(0.82, 0.95, f'S: {similarity_score:.2f}', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Remove any empty subplots
        if num_pre_names == 17:
            for j in [0, 2, 3]:
                fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_analysis(self, indices, variable='', source_area='', target_area='', voltage_scale=None):
        source_pop_names = other_billeh_utils.pop_names(self.networks[source_area])
        source_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_billeh_utils.pop_names(self.networks[target_area])
        target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        ### Initial Weight ###
        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        weight_changes = final_weights - initial_weights
        df = pd.DataFrame({'Post_names': post_cell_types, 
                            'Pre_names':pre_cell_types, 
                            'Initial weight': initial_weights, 
                            'Final weight': final_weights, 
                            'Weight Change': weight_changes})
        
        # Calculate global min and max for color normalization
        # global_grouped_df = df.groupby(['Pre_names', 'Post_names'])[['Initial weight', 'Final weight']].mean().reset_index()
        # global_min = global_grouped_df[['Initial weight', 'Final weight']].min().min()
        # global_max = global_grouped_df[['Initial weight', 'Final weight']].max().max()
        # global_min = df[['Initial weight', 'Final weight']].min().min()
        # global_max = df[['Initial weight', 'Final weight']].max().max()

        boxplots_dir = os.path.join(self.images_dir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)

        # Plot for Initial Weight
        if not os.path.exists(os.path.join(boxplots_dir, 'Initial_weight.png')):
            grouped_df = df.groupby(['Pre_names', 'Post_names'])['Initial weight'].mean().reset_index()
            # Create a pivot table to reshape the data for the heatmap
            pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Initial weight')
            # Plot heatmap
            fig = plt.figure(figsize=(12, 6))
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
            heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
            plt.xlabel(f'{target_area}')
            plt.ylabel(f'{source_area}')
            plt.xticks(rotation=90)
            plt.gca().set_aspect('equal')
            plt.title(f'{variable}')
            # Create a separate color bar axis
            cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
            # Plot color bar
            cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
            cbar.set_label('Initial Weight (pA)')
            plt.savefig(os.path.join(boxplots_dir, f'Initial_weight.png'), dpi=300, transparent=False)
            plt.close()

        ### Final Weight ###
        grouped_df = df.groupby(['Pre_names', 'Post_names'])['Final weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Final weight')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
        plt.xlabel(f'{target_area}')
        plt.ylabel(f'{source_area}')
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Final Weight (pA)')
        plt.savefig(os.path.join(boxplots_dir, f'Final_weight.png'), dpi=300, transparent=False)
        plt.close()

        ### Weight change ###
        grouped_df = df.groupby(['Pre_names', 'Post_names'])['Weight Change'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        try:
            pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Weight Change')
            # Plot heatmap
            plt.figure(figsize=(12, 6))
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0)
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
            heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
            plt.xlabel(f'{target_area}')
            plt.ylabel(f'{source_area}')
            plt.xticks(rotation=90)
            plt.gca().set_aspect('equal')
            plt.title(f'{variable}')
            # Create a separate color bar axis
            cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
            # Plot color bar
            cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
            cbar.set_label('Weight Change (pA)')
            plt.savefig(os.path.join(boxplots_dir, f'Weight_change.png'), dpi=300, transparent=False)
            plt.close()
        except:
            print('Skipping the plot for the weight change heatmap...')
            # raise the actual error
            print(grouped_df)
        
    def fano_factor(self, spikes, area='', t_start=0.7, t_end=2.5, n_samples=100, analyze_core_only=True):
        
        # Trim spikes to the desired time window
        t_start_idx = int(t_start * 1000)
        t_end_idx = int(t_end * 1000)
        spikes = spikes[:, :, t_start_idx:t_end_idx, :]

        if analyze_core_only:
            # Isolate the core neurons
            n_core_neurons = 51978 if area == 'v1' else 7414 
            core_mask = other_billeh_utils.isolate_core_neurons(self.networks[area], n_selected_neurons=n_core_neurons, data_dir=self.flags.data_dir)
        else:
            n_core_neurons = spikes.shape[-1]
            core_mask = np.ones(n_core_neurons, dtype=bool)
            
        # Calculate the Fano Factor for the spikes
        pop_names = other_billeh_utils.pop_names(self.networks[area])
        node_ei = np.array([pop_name[0] for pop_name in pop_names])
        # Get the IDs for excitatory neurons
        total_mask = np.logical_and(node_ei == 'e', core_mask)
        n_e_neurons = np.sum(total_mask)
        spikes = spikes[:, :, :, total_mask]
        node_id_e = np.arange(0, n_e_neurons)
        # Reshape the spikes tensor to have the shape (n_trials, seq_len, n_neurons)
        spikes = np.reshape(spikes, [spikes.shape[0]*spikes.shape[1], spikes.shape[2], spikes.shape[3]])
        n_trials = spikes.shape[0]
        # Pre-define bin sizes
        bin_sizes = np.logspace(-3, 0, 20)
        # using the simulation length, limit bin_sizes to define at least 2 bins
        # bin_sizes_mask = bin_sizes < (t_end - t_start)/2
        bin_sizes_mask = bin_sizes < (t_end - t_start)/2
        bin_sizes = bin_sizes[bin_sizes_mask]
        # Vectorize the sampling process
        if area == 'v1':
            sample_size = 70
            sample_std = 30
        else:
            sample_size = 33
            sample_std = 14
        sample_counts = np.random.normal(sample_size, sample_std, n_samples).astype(int)
        # ensure that the sample counts are at least 15 and less than the number of neurons
        sample_counts = np.clip(sample_counts, 15, n_e_neurons)
        # trial_ids =np.random.choice(np.arange(n_trials), n_samples, replace=False)
        trial_ids = np.random.randint(n_trials, size=n_samples)

        # Generate Fano factors across random samples
        fanos = []
        for i in range(n_samples):
            random_trial_id = trial_ids[i]
            sample_num = sample_counts[i]
            sample_ids = np.random.choice(node_id_e, sample_num, replace=False)
            # selected_spikes = np.concatenate([spikes_timestamps[random_trial_id][np.isin(node_id, sample_ids), :]])
            # selected_spikes = selected_spikes[~np.isnan(selected_spikes)]
            # selected_spikes = spikes[random_trial_id][:, np.isin(node_id, sample_ids)]
            selected_spikes = spikes[random_trial_id][:, sample_ids]
            selected_spikes = np.sum(selected_spikes, axis=1)
            # if there are spikes use pop_fano
            if np.sum(selected_spikes) > 0:
                fano = pop_fano(selected_spikes, bin_sizes)
                fanos.append(fano)

        fanos = np.array(fanos)
        return fanos, bin_sizes
        
    def fanos_figure(self, spikes, area='', n_samples=100, fano_duration=300, analyze_core_only=True, data_dir='Synchronization_data'):
        # Calculate fano factors for both sessions
        fano_duration_seconds = fano_duration / 1000
        t_start_seconds = self.pre_delay / 1000 + 0.2
        t_end_seconds = t_start_seconds + fano_duration_seconds
        # evoked_fanos, evoked_bin_sizes = self.fano_factor(spikes, area=area, t_start=0.7, t_end=2.5, n_samples=n_samples, analyze_core_only=analyze_core_only)
        evoked_fanos, evoked_bin_sizes = self.fano_factor(spikes, area=area, t_start=t_start_seconds, t_end=t_end_seconds, n_samples=n_samples, analyze_core_only=analyze_core_only)
        spontaneous_fanos, spont_bin_sizes = self.fano_factor(spikes, area=area, t_start=0.2, t_end=self.pre_delay/1000, n_samples=n_samples, analyze_core_only=analyze_core_only)

        # Calculate mean, standard deviation, and SEM of the Fano factors
        evoked_fanos_mean = np.nanmean(evoked_fanos, axis=0)
        evoked_fanos_std = np.nanstd(evoked_fanos, axis=0)
        evoked_fanos_sem = evoked_fanos_std / np.sqrt(n_samples)
        spontaneous_fanos_mean = np.nanmean(spontaneous_fanos, axis=0)
        spontaneous_fanos_std = np.nanstd(spontaneous_fanos, axis=0)
        spontaneous_fanos_sem = spontaneous_fanos_std / np.sqrt(n_samples)
        # Find max fanos and corresponding frequency
        evoked_max_fano = np.nanmax(evoked_fanos_mean)
        evoked_max_fano_freq = 1/(2*evoked_bin_sizes[np.nanargmax(evoked_fanos_mean)])
        spontaneous_max_fano = np.nanmax(spontaneous_fanos_mean)
        spontaneous_max_fano_freq = 1/(2*spont_bin_sizes[np.nanargmax(spontaneous_fanos_mean)])
        # Load experimental data
        evoked_exp_data_path = os.path.join(data_dir, f'Fano_factor_{area}', f'{area}_fano_running_{fano_duration}ms_evoked.npy')
        evoked_exp_fanos = np.load(evoked_exp_data_path, allow_pickle=True)
        spont_exp_data_path = os.path.join(data_dir, f'Fano_factor_{area}', f'{area}_fano_running_{fano_duration}ms_spont.npy')
        spont_exp_fanos = np.load(spont_exp_data_path, allow_pickle=True)
        # Mask experimental data to match bin sizes
        # Assuming experimental data was computed with the same initial bin_sizes array
        # We'll just take the first len(evoked_bin_sizes) and len(spont_bin_sizes)
        evoked_exp_fanos_mean = np.nanmean(evoked_exp_fanos, axis=0)[:len(evoked_bin_sizes)]
        evoked_exp_fanos_std = np.nanstd(evoked_exp_fanos, axis=0)[:len(evoked_bin_sizes)]
        evoked_exp_fanos_sem = evoked_exp_fanos_std / np.sqrt(evoked_exp_fanos.shape[0])
        spont_exp_fanos_mean = np.nanmean(spont_exp_fanos, axis=0)[:len(spont_bin_sizes)]
        spont_exp_fanos_std = np.nanstd(spont_exp_fanos, axis=0)[:len(spont_bin_sizes)]
        spont_exp_fanos_sem = spont_exp_fanos_std / np.sqrt(spont_exp_fanos.shape[0])
        
        # Plot the results
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        # Plot all experimental evoked traces with low alpha
        for row in range(evoked_exp_fanos.shape[0]):
            axs[0].plot(evoked_bin_sizes, evoked_exp_fanos[row, :len(evoked_bin_sizes)], color='gray', alpha=0.3)

        axs[0].errorbar(evoked_bin_sizes, evoked_fanos_mean, yerr=evoked_fanos_sem, fmt='o-', label='Evoked Model', color='blue')
        axs[0].errorbar(evoked_bin_sizes, evoked_exp_fanos_mean, yerr=evoked_exp_fanos_sem, fmt='o-', label='Evoked Experimental', color='k')
        axs[0].set_xscale("log")
        axs[0].set_title(f'{area} - Max: {evoked_max_fano:.2f}, Freq: {evoked_max_fano_freq:.1f} Hz', fontsize=16)
        axs[0].set_xlabel('Bin Size (s)', fontsize=14)
        axs[0].set_ylabel('Fano Factor', fontsize=14)
        axs[0].legend(fontsize=14)

        # Plot all experimental evoked traces with low alpha
        for row in range(spont_bin_sizes.shape[0]):
            axs[1].plot(spont_bin_sizes, spont_exp_fanos[row, :len(spont_bin_sizes)], color='gray', alpha=0.3)

        axs[1].errorbar(spont_bin_sizes, spontaneous_fanos_mean, yerr=spontaneous_fanos_sem, fmt='o-', label='Spontaneous Model', color='orange')
        axs[1].errorbar(spont_bin_sizes, spont_exp_fanos_mean, yerr=spont_exp_fanos_sem, fmt='o-', label='Spontaneous Experimental', color='k')
        axs[1].set_xscale("log")
        axs[1].set_title(f'{area} - Max: {spontaneous_max_fano:.2f}, Freq: {spontaneous_max_fano_freq:.1f} Hz', fontsize=16)
        axs[1].set_xlabel('Bin Size (s)', fontsize=14)
        axs[1].legend(fontsize=14)

        plt.tight_layout()
        os.makedirs(os.path.join(self.images_dir, 'Fano_Factor'), exist_ok=True)
        plt.savefig(os.path.join(self.images_dir, 'Fano_Factor', f'{area}_epoch_{self.current_epoch}.png'), dpi=300, transparent=False)
        plt.close()
    
    # def fano_factor_tf(self, spikes, area='', t_start=0.7, t_end=2.5, n_samples=100, 
    #                     analyze_core_only=True, seed=42):
    #     """
    #     Compute the Fano factors for given spikes using TensorFlow ops and GPU acceleration.
    #     spikes: TF tensor [n_trials, seq_len, n_neurons], spike counts
    #     """
    #     # Trim spikes to the desired time window
    #     t_start_idx = int(t_start * 1000)
    #     t_end_idx = int(t_end * 1000)
    #     spikes = spikes[:, :, t_start_idx:t_end_idx, :]

    #     if analyze_core_only:
    #         # Isolate the core neurons
    #         n_core_neurons = 51978 if area == 'v1' else 7414 
    #         core_mask = other_billeh_utils.isolate_core_neurons(self.networks[area], n_selected_neurons=n_core_neurons, data_dir='GLIF_network')
    #     else:
    #         n_core_neurons = spikes.shape[-1]
    #         core_mask = np.ones(n_core_neurons, dtype=bool)

    #     # Calculate the Fano Factor for the spikes
    #     pop_names = other_billeh_utils.pop_names(self.networks[area])
    #     node_ei = np.array([pop_name[0] for pop_name in pop_names])
    #     # Get the IDs for excitatory neurons
    #     total_mask = np.logical_and(node_ei == 'e', core_mask)
    #     n_e_neurons = np.sum(total_mask)
    #     spikes = spikes[:, :, :, total_mask]
    #     node_id_e = tf.range(0, n_e_neurons)
    #     # Flatten trials if needed
    #     spikes = np.reshape(spikes, [spikes.shape[0]*spikes.shape[1], spikes.shape[2], spikes.shape[3]])
    #     n_trials = tf.shape(spikes)[0]
    #     # bin sizes
    #     bin_sizes = np.logspace(-3, 0, 20)
    #     # bin_sizes_mask = bin_sizes < ((t_end - t_start) / 2.0)
    #     bin_sizes_mask = bin_sizes < ((t_end - t_start) / 5.0)
    #     bin_sizes = bin_sizes[bin_sizes_mask]
    #     # Determine sample size distribution
    #     if area == 'v1':
    #         sample_size = 70
    #         sample_std = 30
    #     else:
    #         sample_size = 33
    #         sample_std = 14
    #     base_seed = seed
    #     sample_counts = tf.cast(tf.random.normal([n_samples], mean=sample_size, stddev=sample_std, seed=base_seed), tf.int32)
    #     sample_counts = tf.clip_by_value(sample_counts, 15, n_e_neurons) 
    #     # random trials
    #     base_seed += 1
    #     sample_trials = tf.random.uniform([n_samples], minval=0, maxval=n_trials, dtype=tf.int32, seed=base_seed)
    #     # Shuffle excitatory IDs
    #     base_seed += 1
    #     shuffled_e_ids = tf.random.shuffle(node_id_e, seed=base_seed)
    #     previous_id = tf.constant(0, dtype=tf.int32)
    #     # Collect samples
    #     selected_samples = tf.TensorArray(tf.float32, size=n_samples)
    #     for i in tf.range(n_samples):
    #         sample_num = sample_counts[i]
    #         sample_trial = sample_trials[i]
    #         # If we don't have enough left in shuffled_e_ids, reshuffle
    #         if previous_id + sample_num > tf.size(shuffled_e_ids):
    #             shuffled_e_ids = tf.random.shuffle(node_id_e, seed=base_seed)
    #             previous_id = tf.constant(0, dtype=tf.int32)
    #         sample_ids = shuffled_e_ids[previous_id:previous_id+sample_num]
    #         previous_id += sample_num
    #         # Sum spikes over selected neurons
    #         trial_spikes = tf.constant(spikes[sample_trial, :, :], tf.float32)
    #         selected_spikes = tf.reduce_sum(tf.gather(trial_spikes, sample_ids, axis=1), axis=-1)
    #         selected_samples = selected_samples.write(i, selected_spikes)

    #     # stack into [n_samples, seq_len]
    #     selected_spikes_sample = selected_samples.stack()
    #     fanos = pop_fano_tf(selected_spikes_sample, bin_sizes=bin_sizes)

    #     return fanos, bin_sizes

    # def fanos_figure_tf(self, spikes, area='', n_samples=100, analyze_core_only=True, data_dir='Synchronization_data'):
    #     """
    #     Compute Fano factors for both evoked and spontaneous conditions and plot the results.
    #     """
    #     # Evoked condition
    #     evoked_fanos, evoked_bin_sizes = self.fano_factor_tf(spikes, area=area, t_start=0.7, t_end=2.5,
    #                                                     n_samples=n_samples, analyze_core_only=analyze_core_only)
    #     # Spontaneous condition
    #     spont_fanos, spont_bin_sizes = self.fano_factor_tf(spikes, area=area, t_start=0.2, t_end=0.5,
    #                                                 n_samples=n_samples, analyze_core_only=analyze_core_only)
    #     # Postprocess the data
    #     evoked_fanos_np = evoked_fanos.numpy()
    #     spont_fanos_np = spont_fanos.numpy()
    #     # Calculate mean and SEM
    #     evoked_fanos_mean = np.nanmean(evoked_fanos_np, axis=1)
    #     evoked_fanos_std = np.nanstd(evoked_fanos_np, axis=1)
    #     evoked_fanos_sem = evoked_fanos_std / np.sqrt(n_samples)
    #     spont_fanos_mean = np.nanmean(spont_fanos_np, axis=1)
    #     spont_fanos_std = np.nanstd(spont_fanos_np, axis=1)
    #     spont_fanos_sem = spont_fanos_std / np.sqrt(n_samples)
    #     # Find max fanos and corresponding frequency
    #     evoked_max_fano = np.nanmax(evoked_fanos_mean)
    #     evoked_max_fano_freq = 1/(2*evoked_bin_sizes[np.nanargmax(evoked_fanos_mean)])
    #     spont_max_fano = np.nanmax(spont_fanos_mean)
    #     spont_max_fano_freq = 1/(2*spont_bin_sizes[np.nanargmax(spont_fanos_mean)])
    #     # Load experimental data
    #     evoked_exp_data_path = os.path.join(data_dir, f'Fano_factor_{area}', f'{area}_fano_running_1800ms_evoked.npy')
    #     spont_exp_data_path = os.path.join(data_dir, f'Fano_factor_{area}', f'{area}_fano_running_300ms_spont.npy')
    #     evoked_exp_fanos = np.load(evoked_exp_data_path, allow_pickle=True)
    #     spont_exp_fanos = np.load(spont_exp_data_path, allow_pickle=True)
    #     # Mask experimental data to match bin sizes
    #     # Assuming experimental data was computed with the same initial bin_sizes array
    #     # We'll just take the first len(evoked_bin_sizes) and len(spont_bin_sizes)
    #     evoked_exp_fanos_mean = np.nanmean(evoked_exp_fanos, axis=0)[:len(evoked_bin_sizes)]
    #     evoked_exp_fanos_std = np.nanstd(evoked_exp_fanos, axis=0)[:len(evoked_bin_sizes)]
    #     evoked_exp_fanos_sem = evoked_exp_fanos_std / np.sqrt(evoked_exp_fanos.shape[0])
    #     spont_exp_fanos_mean = np.nanmean(spont_exp_fanos, axis=0)[:len(spont_bin_sizes)]
    #     spont_exp_fanos_std = np.nanstd(spont_exp_fanos, axis=0)[:len(spont_bin_sizes)]
    #     spont_exp_fanos_sem = spont_exp_fanos_std / np.sqrt(spont_exp_fanos.shape[0])

    #     # Plot the results
    #     fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    #     # Plot all experimental evoked traces with low alpha
    #     for row in range(evoked_exp_fanos.shape[0]):
    #         axs[0].plot(evoked_bin_sizes, evoked_exp_fanos[row, :len(evoked_bin_sizes)], color='gray', alpha=0.3)
            
    #     axs[0].errorbar(evoked_bin_sizes, evoked_fanos_mean, yerr=evoked_fanos_sem, fmt='o-', label='Evoked Model', color='blue')
    #     axs[0].errorbar(evoked_bin_sizes, evoked_exp_fanos_mean, yerr=evoked_exp_fanos_sem, fmt='o-', label='Evoked Experimental', color='k')
    #     axs[0].set_xscale("log")
    #     axs[0].set_title(f'{area} - Max: {evoked_max_fano:.2f}, Freq: {evoked_max_fano_freq:.1f} Hz', fontsize=16)
    #     axs[0].set_xlabel('Bin Size (s)', fontsize=14)
    #     axs[0].set_ylabel('Fano Factor', fontsize=14)
    #     axs[0].legend(fontsize=14)

    #     # Plot all experimental evoked traces with low alpha
    #     for row in range(spont_bin_sizes.shape[0]):
    #         axs[1].plot(spont_bin_sizes, spont_exp_fanos[row, :len(spont_bin_sizes)], color='gray', alpha=0.3)
            
    #     axs[1].errorbar(spont_bin_sizes, spont_fanos_mean, yerr=spont_fanos_sem, fmt='o-', label='Spontaneous Model', color='orange')
    #     axs[1].errorbar(spont_bin_sizes, spont_exp_fanos_mean, yerr=spont_exp_fanos_sem, fmt='o-', label='Spontaneous Experimental', color='k')
    #     axs[1].set_xscale("log")
    #     axs[1].set_title(f'{area} - Max: {spont_max_fano:.2f}, Freq: {spont_max_fano_freq:.1f} Hz', fontsize=16)
    #     axs[1].set_xlabel('Bin Size (s)', fontsize=14)
    #     axs[1].legend(fontsize=14)

    #     plt.tight_layout()
    #     os.makedirs(os.path.join(self.images_dir, 'Fano_Factor'), exist_ok=True)
    #     plt.savefig(os.path.join(self.images_dir, 'Fano_Factor', f'{area}_epoch_{self.current_epoch}_tf.png'), dpi=300, transparent=False)
    #     plt.close()


    def power_spectrum(self, v1_spikes, lm_spikes, v1_spikes_spont=None, lm_spikes_spont=None, fs=1000, directory=''):
        # Sum the spikes over the batch size and all neurons to get a single spiking activity signal for each area
        combined_spiking_activity_v1 = v1_spikes.mean(axis=-1)
        combined_spiking_activity_lm = lm_spikes.mean(axis=-1)
        v1_signal = exponential_decay_filter(combined_spiking_activity_v1)
        lm_signal = exponential_decay_filter(combined_spiking_activity_lm)
        # # Compute the power spectrum for the combined signal for each area
        # seq_len = combined_spiking_activity_v1.shape[0]
        # fs = 1000.0
        # frequencies = np.fft.fftfreq(seq_len, d=1/fs)
        # fft_values_v1 = np.fft.fft(combined_spiking_activity_v1)
        # fft_values_lm = np.fft.fft(combined_spiking_activity_lm)
        # power_spectrum_v1 = np.abs(fft_values_v1) ** 2 / seq_len
        # power_spectrum_lm = np.abs(fft_values_lm) ** 2 / seq_len

        # Calculate the power spectrum
        # Sampling frequency (1 kHz)
        f_v1, power_spectrum_v1 = calculate_power_spectrum(v1_signal, fs)
        f_lm, power_spectrum_lm = calculate_power_spectrum(lm_signal, fs)

        # Plot the power spectrum
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=f_v1, y=power_spectrum_v1, label='V1', color='blue')
        sns.lineplot(x=f_lm, y=power_spectrum_lm, label='LM', color='orange')

        if v1_spikes_spont is not None:
            combined_spiking_activity_v1_spont = v1_spikes_spont.mean(axis=1)
            combined_spiking_activity_lm_spont = lm_spikes_spont.mean(axis=1)
            v1_signal_spont = exponential_decay_filter(combined_spiking_activity_v1_spont)
            lm_signal_spont = exponential_decay_filter(combined_spiking_activity_lm_spont)
            f_v1_spont, power_spectrum_v1_spont = calculate_power_spectrum(v1_signal_spont, fs)
            f_lm_spont, power_spectrum_lm_spont = calculate_power_spectrum(lm_signal_spont, fs)
            
            sns.lineplot(x=f_v1_spont, y=power_spectrum_v1_spont, label='V1 Spontaneous', linestyle='--', color='blue')
            sns.lineplot(x=f_lm_spont, y=power_spectrum_lm_spont, label='LM Spontaneous', linestyle='--', color='orange')

        # Remove the 0 Hz component for plotting
        # positive_frequencies = frequencies[:seq_len // 2]
        # positive_power_spectrum_v1 = power_spectrum_v1[:seq_len // 2]
        # positive_power_spectrum_lm = power_spectrum_lm[:seq_len // 2]
        # # # Set up the Seaborn style
        # sns.set(style="ticks")
        # plt.semilogy()
        plt.xlim([0, 50])
        plt.title('Power Spectral Density of Neuronal Spiking Activity in V1 and LM', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Power Spectral Density [1/Hz]', fontsize=14)
        plt.legend()
        # plt.grid(True)
        plt.tight_layout()
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, f'epoch_{self.current_epoch}.png'), dpi=300, transparent=False)
        plt.close()

    def plot_populations_activity(self, v1_spikes, lm_spikes):
        z = [v1_spikes, lm_spikes]
        seq_len = v1_spikes.shape[1]
        # Plot the mean firing rate of the population of neurons
        for area_id, area in enumerate(['v1', 'lm']):
            neurons = self.networks[area]['n_nodes']
            filename = f'{area}_Epoch_{self.current_epoch}'
            Population_activity = PopulationActivity(n_neurons=neurons, network=self.networks[area], 
                                                    stimuli_init_time=self.pre_delay, stimuli_end_time=seq_len-self.post_delay, 
                                                    image_path=self.images_dir, filename=filename, data_dir=self.flags.data_dir)
            Population_activity(z[area_id], area=area, plot_core_only=True, bin_size=10)

    def plot_raster(self, x, v1_spikes, lm_spikes, angle=0):
        seq_len = v1_spikes.shape[1]
        images_dir = os.path.join(self.images_dir, 'Raster_plots_OSI_DSI')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.networks,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.current_epoch}_orientation_{angle}_degrees',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=self.pre_delay,
                                    stimuli_end_time=seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=True,
                                    )
        graph(x, v1_spikes, lm_spikes)

    def plot_population_firing_rates_vs_tuning_angle(self, area, spikes, DG_angles):
        # Save the spikes
        spikes_dir = os.path.join(self.images_dir, 'Spikes_OSI_DSI')
        os.makedirs(spikes_dir, exist_ok=True)

        # Isolate the core neurons
        n_core_neurons = 51978 if area == 'v1' else 7414 
        core_mask = other_billeh_utils.isolate_core_neurons(self.networks[area], n_selected_neurons=n_core_neurons, data_dir=self.flags.data_dir)
        spikes = spikes[:, :, :, core_mask]
        spikes = np.sum(spikes, axis=0).astype(np.float32)/self.flags.n_trials_per_angle
        seq_len = spikes.shape[1]

        tuning_angles = self.networks[area]['tuning_angle'][core_mask]
        for angle_id, angle in enumerate(DG_angles):
            firingRates = calculate_Firing_Rate(spikes[angle_id, :, :], stimulus_init=self.pre_delay, stimulus_end=seq_len-self.post_delay, temporal_axis=0)
            x = tuning_angles
            y = firingRates
            # Define bins for delta_angle
            bins = np.linspace(np.min(x), np.max(x), 50)
            # Compute average rates for each bin
            average_rates = []
            for i in range(len(bins)-1):
                mask = (x >= bins[i]) & (x < bins[i+1])
                average_rates.append(np.mean(y[mask]))
            # Create bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(bins[:-1], average_rates, width=np.diff(bins))
            plt.xlabel('Tuning Angle')
            plt.ylabel('Average Rates')
            plt.title(f'Gratings Angle: {angle}')
            plt.savefig(os.path.join(spikes_dir, f'{area}_spikes_angle_{angle}.png'))
            plt.close()

    def single_trial_callbacks(self, x, v1_spikes, lm_spikes, y, bkg_noise=None):
        # Separate the spontaneous and evoked spikes for the power spectrum
        # x_spont_spikes = x[0, :self.pre_delay, :]
        # x_evoked_spikes = x[0, self.pre_delay:-self.post_delay, :]
        seq_len = v1_spikes.shape[1]
        v1_spont_spikes = v1_spikes[0, :self.pre_delay, :].astype(np.float32)
        v1_evoked_spikes = v1_spikes[0, self.pre_delay:seq_len-self.post_delay, :].astype(np.float32)
        lm_spont_spikes = lm_spikes[0, :self.pre_delay, :].astype(np.float32)
        lm_evoked_spikes = lm_spikes[0, self.pre_delay:seq_len-self.post_delay, :].astype(np.float32)
        # Plot the power spectrum of the neuronal activity
        self.power_spectrum(v1_evoked_spikes, lm_evoked_spikes, v1_spikes_spont=v1_spont_spikes, lm_spikes_spont=lm_spont_spikes, 
                            directory=os.path.join(self.images_dir, 'Power_spectrum_OSI_DSI'))
        # Plot the population activity
        self.plot_populations_activity(v1_spikes, lm_spikes)
        # Plot the raster plot
        self.plot_raster(x, v1_spikes, lm_spikes, angle=y)

    def osi_dsi_analysis(self, v1_spikes, lm_spikes, DG_angles, fano_duration=300):
        # Do the OSI/DSI analysis       
        boxplots_dir = os.path.join(self.images_dir, 'Boxplots_OSI_DSI')
        os.makedirs(boxplots_dir, exist_ok=True)
        fr_boxplots_dir = os.path.join(self.images_dir, f'Boxplots_OSI_DSI/Ave_Rate(Hz)')
        os.makedirs(fr_boxplots_dir, exist_ok=True)
        spontaneous_boxplots_dir = os.path.join(self.images_dir, 'Boxplots_OSI_DSI/Spontaneous rate (Hz)')
        os.makedirs(spontaneous_boxplots_dir, exist_ok=True)
        # Define the figures
        fig1, axs1 = plt.subplots(2, 1, figsize=(12, 14))
        fig2, axs2 = plt.subplots(2, 1, figsize=(12, 14))
        
        for axis_id, spikes, area in zip([0, 1], [v1_spikes, lm_spikes], ['v1', 'lm']):
            # Fano factor analysis
            print('Fano factor analysis for area: ', area)
            t_fano0 = time()
            self.fanos_figure(spikes, area=area, n_samples=1000, fano_duration=fano_duration, analyze_core_only=True)
            print('Fanos figure saved in ', time() - t_fano0, ' seconds!\n')

            # print('Fano factor TF analysis for area: ', area)
            # t_fano0 = time()
            # self.fanos_figure_tf(spikes, area=area, n_samples=1000, analyze_core_only=True)
            # print('Fanos TF figure saved in ', time() - t_fano0, ' seconds!\n')

            # Plot the tuning angle analysis
            self.plot_population_firing_rates_vs_tuning_angle(area, spikes, DG_angles)
            # Estimate tuning parameters from the model neurons
            metrics_analysis = ModelMetricsAnalysis(spikes, DG_angles, self.networks[area], data_dir=self.flags.data_dir,
                                                    drifting_gratings_init=500, drifting_gratings_end=2500,
                                                    spontaneous_init=0, spontaneous_end=500,
                                                    area=area, analyze_core_only=True, df_directory=self.images_dir, save_df=True)
            # Figure for OSI/DSI boxplots
            metrics_analysis(metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"], directory=boxplots_dir, filename=f'Epoch_{self.current_epoch}')
            # Figure for Average firing rate boxplots
            metrics_analysis(metrics=["Ave_Rate(Hz)"], axis=axs1[axis_id], directory=fr_boxplots_dir, filename=f'{area}_epoch_{self.current_epoch}')   
            # Spontaneous rates figure
            metrics_analysis(metrics=['Spontaneous rate (Hz)'], axis=axs2[axis_id], directory=spontaneous_boxplots_dir, filename=f'{area}_epoch_{self.current_epoch}') 

        fig1.tight_layout()
        fig1.savefig(os.path.join(fr_boxplots_dir, f'epoch_{self.current_epoch}.png'), dpi=300, transparent=False)
        fig2.tight_layout()
        fig2.savefig(os.path.join(spontaneous_boxplots_dir, f'epoch_{self.current_epoch}.png'), dpi=300, transparent=False)
        plt.close()  
        

class ClassificationCallbacks:
    def __init__(self, networks, lgn_inputs, bkg_inputs, model, optimizer, flags, logdir, strategy, 
                metrics_keys, pre_delay=50, post_delay=50, checkpoint=None, model_variables_init=None, 
                save_optimizer=True):

        self.n_neurons = {'v1': networks['v1']['n_nodes'], 'lm': networks['lm']['n_nodes']}
        self.networks = networks
        self.lgn_inputs = lgn_inputs
        self.bkg_inputs = bkg_inputs
        self.model = model
        self.optimizer = optimizer
        self.flags = flags
        self.logdir = logdir
        self.strategy = strategy
        self.metrics_keys = metrics_keys
        self.neuropixels_feature = 'Natural scenes rates(Hz)' # natural images firing rates
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.step = 0
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
            self.checkpoint_epochs = 0
            # create a dictionary to save the values of the metric keys after each epoch
            self.epoch_metric_values = {key: [] for key in self.metrics_keys}
        else:
            # Load epoch_metric_values and min_val_loss from the file
            if os.path.exists(os.path.join(self.logdir, 'train_end_data.pkl')):
                with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'rb') as f:
                    data_loaded = pkl.load(f)
                self.epoch_metric_values = data_loaded['epoch_metric_values']
                self.min_val_loss = data_loaded['min_val_loss']
                self.no_improve_epochs = data_loaded['no_improve_epochs']
                self.checkpoint_epochs = len(data_loaded['epoch_metric_values']['train_loss'])
            elif os.path.exists(os.path.join(os.path.dirname(flags.restore_from), 'train_end_data.pkl')):
                with open(os.path.join(os.path.dirname(flags.restore_from), 'train_end_data.pkl'), 'rb') as f:
                    data_loaded = pkl.load(f)
                self.epoch_metric_values = {key: [] for key in self.metrics_keys}
                self.min_val_loss = float('inf')
                self.no_improve_epochs = 0
                self.checkpoint_epochs = 0
            else:
                print('No train_end_data.pkl file found. Initializing...')
                self.epoch_metric_values = {key: [] for key in self.metrics_keys}
                self.min_val_loss = float('inf')
                self.no_improve_epochs = 0
                self.checkpoint_epochs = 0

        # Total number of epochs to train
        if flags.run_session == 0:
            self.total_epochs = flags.n_runs * flags.n_epochs + self.checkpoint_epochs
        else:
            self.total_epochs = (flags.n_runs - flags.run_session) * flags.n_epochs + self.checkpoint_epochs
        # Manager for the best model
        self.best_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/Best_model', max_to_keep=1
        )
        # Manager for osi/dsi checkpoints 
        self.epoch_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/Intermediate_checkpoints', max_to_keep=3
        )

    def on_train_begin(self):
        print("---------- Training started at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        self.train_start_time = time()
        # self.epoch = self.flags.run_session * self.flags.n_epochs
        self.epoch = self.checkpoint_epochs
        # # Clear the session to reset the graph state
        # tf.keras.backend.clear_session()

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
            self.save_intermediate_checkpoint()

        self.model_variables_dict['Best'] = {var.name: var.numpy().astype(np.float16) for var in self.model.trainable_variables}
        # Analize changes in trainable variables.
        if self.model_variables_dict is not None:
            for var in self.model_variables_dict['Best'].keys():
                t0 = time()
                self.trainable_variable_change_heatmaps_and_distributions(var)
                print(f'Time spent in {var}: {time()-t0}')

    def on_epoch_start(self):
        self.epoch += 1
        # self.step_counter.assign_add(1)
        self.epoch_init_time = time()
        date_str = dt.datetime.now().strftime('%d-%m-%Y %H:%M')
        print(f'Epoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')
        tf.print(f'\nEpoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')

    def on_epoch_end(self, x, v1_spikes, lm_spikes, y, metric_values, verbose=True):
        
        if self.flags.dtype != 'float32':
            v1_spikes = v1_spikes.numpy().astype(np.float32)
            lm_spikes = lm_spikes.numpy().astype(np.float32)
            x = x.numpy().astype(np.float32)
            y = y.numpy().astype(np.float32)
        else:
            v1_spikes = v1_spikes.numpy()
            lm_spikes = lm_spikes.numpy()
            x = x.numpy()
            y = y.numpy()
        
        self.step = 0
        if self.initial_metric_values is None:
            self.initial_metric_values = metric_values
        
        if verbose:
            print_str = f'  Validation:\n' 
            val_values = metric_values[len(metric_values)//2:]
            print_str += '    ' + compose_str(val_values, mode='classification') 
            print(print_str)
            for gpu_id in range(len(self.strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)

        for i, (key, value) in enumerate(self.epoch_metric_values.items()):
            self.epoch_metric_values[key] = value + [metric_values[i]]

        # self.epoch_metric_values = {key: value + [metric_values[i]] for i, (key, value) in enumerate(self.epoch_metric_values.items())}

        if 'val_loss' in self.metrics_keys:
            val_loss_index = self.metrics_keys.index('val_loss')
            val_loss_value = metric_values[val_loss_index]
        else:
            val_loss_index = self.metrics_keys.index('train_loss')
            val_loss_value = metric_values[val_loss_index]

        self.plot_losses_curves()
        # self.plot_synchronization_evolution(v1_spikes, lm_spikes, v1_spikes_spont, lm_spikes_spont)

        if val_loss_value < self.min_val_loss:
            self.min_val_loss = val_loss_value
            self.no_improve_epochs = 0
            self.save_best_model()
            images_dir = self.logdir
            self.classification_boxplots(x, v1_spikes, lm_spikes, y, images_dir=images_dir)
            self.classification_rates(x, v1_spikes, lm_spikes, y, images_dir=images_dir)
            self.plot_mean_firing_rate_boxplot(v1_spikes, lm_spikes, y, images_dir=images_dir)
            self.plot_raster(x, v1_spikes, lm_spikes, y, images_dir=images_dir)
            self.model_variables_dict['Best'] = {var.name: var.numpy().astype(np.float16) for var in self.model.trainable_variables}
        else:
            self.no_improve_epochs += 1
           
        with self.summary_writer.as_default():
            for k, v in zip(self.metrics_keys, metric_values):
                tf.summary.scalar(k, v, step=self.epoch)

        # EARLY STOPPING CONDITIONS
        if (0 < self.flags.max_time < (time() - self.epoch_init_time) / 3600):
            print(f'[ Maximum optimization time of {self.flags.max_time:.2f}h reached ]')
            stop = True
        elif self.no_improve_epochs >= 500:
            print("Early stopping: Validation loss has not improved for 50 epochs.")
            stop = True  
        else:
            stop = False

        return stop

    def on_step_start(self):
        self.step += 1
        self.step_init_time = time()
        # reset the gpu memory stat
        # tf.config.experimental.reset_memory_stats('GPU:0')

    def on_step_end(self, train_values, verbose=True):
        self.step_running_time.append(time() - self.step_init_time)
        if verbose:
            print_str = f'  Step {self.step:2d}/{self.flags.steps_per_epoch}\n'
            print_str += '    ' + compose_str(train_values, mode='classification')
            print(print_str)
            tf.print('\n'+print_str)
            print(f'    Step running time: {time() - self.step_init_time:.2f}s')
            for gpu_id in range(len(self.strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)

    def on_testing_end(self, x, v1_spikes, lm_spikes, y, metric_values, verbose=True):
        
        if self.flags.dtype != 'float32':
            v1_spikes = v1_spikes.numpy().astype(np.float32)
            lm_spikes = lm_spikes.numpy().astype(np.float32)
            x = x.numpy().astype(np.float32)
            y = y.numpy().astype(np.float32)
        else:
            v1_spikes = v1_spikes.numpy()
            lm_spikes = lm_spikes.numpy()
            x = x.numpy()
            y = y.numpy()

        images_dir = os.path.join(self.logdir, f'connected_areas_{self.flags.connected_areas}_conn_rec_{self.flags.connected_recurrent_connections}_conn_noise_{self.flags.connected_noise}')
                
        if verbose:
            print_str = f'  Validation:\n' 
            val_values = metric_values[len(metric_values)//2:]
            print_str += '    ' + compose_str(val_values, mode='classification') 
            print(print_str)
            # Save print_str to a file in images_dir
            os.makedirs(images_dir, exist_ok=True)
            with open(os.path.join(images_dir, 'validation_output.txt'), 'w') as f:
                f.write(print_str)

            for gpu_id in range(len(self.strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)

        # Noise analysis
        os.makedirs(os.path.join(images_dir, 'Noise_analysis'), exist_ok=True)
        with open(f'{images_dir}/Noise_analysis/validation_results_std_{self.flags.mnist_noise_std}.pkl', 'wb') as f:
            pkl.dump({'validation_results': val_values}, f)

        self.classification_boxplots(x, v1_spikes, lm_spikes, y, images_dir=images_dir)
        self.classification_stripplot(v1_spikes, y, images_dir=images_dir)
        self.classification_rates(x, v1_spikes, lm_spikes, y, images_dir=images_dir)
        self.plot_mean_firing_rate_boxplot(v1_spikes, lm_spikes, y, images_dir=images_dir)
        self.plot_raster(x, v1_spikes, lm_spikes, y, images_dir=images_dir)

    def trainable_variable_change_heatmaps_and_distributions(self, variable):
        area = variable.split('_')[0]
        node_types_voltage_scale = (self.networks[area]['node_params']['V_th'] - self.networks[area]['node_params']['E_L']).astype(np.float16)
        node_type_ids = self.networks[area]['node_type_ids']
        
        if 'rest_of_brain_weights' in variable:
            voltage_scale = node_types_voltage_scale[node_type_ids[self.bkg_inputs[area]['indices'][:, 0]]]
            self.node_to_pop_weights_analysis(self.bkg_inputs[area]['indices'], variable=variable, area=area, voltage_scale=voltage_scale)
        elif'sparse_input_weights' in variable:
            voltage_scale = node_types_voltage_scale[node_type_ids[self.lgn_inputs[area]['indices'][:, 0]]]
            self.node_to_pop_weights_analysis(self.lgn_inputs[area]['indices'], variable=variable, area=area, voltage_scale=voltage_scale)
        elif 'sparse_recurrent_weights' in variable:
            indices = self.networks[area]['synapses']['indices']
            voltage_scale = node_types_voltage_scale[node_type_ids[indices[:, 0]]]
            self.pop_to_pop_weights_analysis(indices, variable=variable, 
                                             source_area=area, target_area=area, voltage_scale=voltage_scale)
            self.pop_to_pop_weights_distribution(indices, variable=variable, 
                                                source_area=area, target_area=area, voltage_scale=voltage_scale)
        elif 'sparse_interarea_weights' in variable:
            source_area = variable.split('_')[-1][:2] 
            indices = self.networks[area]['interarea_synapses'][source_area]['indices']
            voltage_scale = node_types_voltage_scale[node_type_ids[indices[:, 0]]]
            self.pop_to_pop_weights_analysis(indices, variable=variable, 
                                            source_area=source_area, target_area=area, voltage_scale=voltage_scale)
            self.pop_to_pop_weights_distribution(indices, variable=variable, 
                                                source_area=source_area, target_area=area, voltage_scale=voltage_scale)

    def node_to_pop_weights_analysis(self, indices, variable='', area='', voltage_scale=None):
        pop_names = other_billeh_utils.pop_names(self.networks[area])
        target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
        # print(variable, indices[:, 0].shape, voltage_scale.shape)
        # if 'rest_of_brain_weights' in variable:
        #     post_indices =  np.repeat(indices[:, 0], 4)
        #     voltage_scale = np.repeat(voltage_scale, 4)
        # else:
        #     post_indices = indices[:, 0]
        post_indices = indices[:, 0]
        post_cell_types = [target_cell_types[i] for i in post_indices]

        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        # Create DataFrame with all the necessary data
        df = pd.DataFrame({
            'Post_names': post_cell_types * 2,  # Duplicate node names for initial and final weights
            'Weight': initial_weights.tolist() + final_weights.tolist(),  # Combine initial and final weights
            'Weight Type': ['Initial weight'] * len(initial_weights) + ['Final weight'] * len(final_weights)  # Distinguish between initial and final weights
        })

        # Count the number of cell_types fro each type
        # cell_type_counts = df['Post_names'].value_counts()

        # Sort the dataframe by Node Name and then by Type to ensure consistent order
        df = df.sort_values(['Post_names', 'Weight Type'])

        # Plotting
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)
        # fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        fig = plt.figure(figsize=(12, 6))
        #get the axis of the figure
        ax = fig.gca()
        similarity_score = compute_ks_statistics(df, metric='Weight', min_n_sample=15)
        hue_order = ['Initial weight', 'Final weight']
        # sns.boxplot(x='Node Name', y='Weight Change', data=df)
        # sns.barplot(x='Post_names', y='Weight', hue='State', hue_order=hue_order, data=df)
        sns.boxplot(x='Post_names', y='Weight', hue='Weight Type', hue_order=hue_order, data=df, ax=ax, width=0.7, fliersize=1.)
        # plt.axhline(0, color='black', linewidth=1)  # include a horizontal black line at 0
        ax.set_yscale('log')
        # ax.set_ylim(bottom=0)  # Set bottom limit of y-axis to 0
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_title(f'{variable}')
        ax.legend(loc='upper right')
        if similarity_score is not None:
            ax.text(0.9, 0.1, f'S: {similarity_score:.2f}', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_distribution(self, indices, variable='', source_area='v1', target_area='v1', voltage_scale=None):
        source_pop_names = other_billeh_utils.pop_names(self.networks[source_area])
        source_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_billeh_utils.pop_names(self.networks[target_area])
        target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        df = pd.DataFrame({
            'Post_names': post_cell_types,
            'Pre_names': pre_cell_types,
            'Initial weight': initial_weights,
            'Final weight': final_weights,
        })

        # Melt DataFrame to long format
        df_melted = df.melt(id_vars=['Post_names', 'Pre_names'], value_vars=['Initial weight', 'Final weight'], 
                            var_name='Weight Type', value_name='Weight')
        df_melted['Weight'] = np.abs(df_melted['Weight'])
        # Create directory for saving plots
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}_distribution')
        os.makedirs(boxplots_dir, exist_ok=True)
        # Establish the order of the neuron types in the boxplots
        cell_type_order = np.sort(df['Pre_names'].unique())
        target_type_order = np.sort(df['Post_names'].unique())

        # Define the palette
        palette = {"Initial weight": "#87CEEB", "Final weight": "#FFA500"}
        # Create subplots
        num_pre_names = len(cell_type_order)

        num_columns = 4
        num_rows = (num_pre_names + num_columns - 1) // num_columns
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(24, 6 * num_rows))
        # Flatten the axes array and handle the first row separately
        axes = axes.flatten()
        for i, pre_name in enumerate(cell_type_order):
            if num_pre_names == 17 and i == 0:
                i = 1
            elif num_pre_names == 17 and i != 0:
                i += 3
            
            ax = axes[i] 
            subset_df = df_melted[df_melted['Pre_names'] == pre_name]
            similarity_score = compute_ks_statistics(subset_df, metric='Weight', min_n_sample=15)
            # subset_cell_type_order = np.sort(subset_df['Post_names'].unique())
            # Create boxplot for Initial and Final weights
            sns.boxplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', order=target_type_order, ax=ax, palette=palette, 
                        width=0.7, fliersize=1.)
            # sns.violinplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', order=subset_cell_type_order, ax=ax, palette=palette, width=0.7,
            #                split=True, inner="quart", gap=0.2)
            ax.set_title(f'Source Cell Type: {pre_name}')
            ax.set_ylabel(r'$\vert$ Synaptic Weight (pA)$\vert$', fontsize=12)
            # ax.set_yscale('symlog', linthresh=0.001)
            ax.set_yscale('log')
            if i % num_columns == 0 or i == 1:  # First column
                ax.set_ylabel(r'$\vert$ Synaptic Weight (pA)$\vert$', fontsize=12)
            else:
                ax.set_ylabel('')
            if i >= (num_rows - 1) * num_columns:
                ax.set_xlabel('Target Cell Type')
            else:
                ax.set_xlabel('')
            ax.tick_params(axis="x", labelrotation=90)
            # Apply shadings to each layer
            xticklabel = ax.get_xticklabels()
            borders = get_borders(xticklabel)
            # change y limit
            # if 'E' in pre_name:
            #     bottom_limit = 0
            #     upper_limit = 1000
            # else:
            #     bottom_limit = -500
            #     upper_limit = 0
            bottom_limit = 0.001
            upper_limit = 1000
            ax.set_ylim(bottom=bottom_limit, top=upper_limit)
            # get the current ylim
            ylim = ax.get_ylim()
            draw_borders(ax, borders, ylim)
            # ax.legend(loc='best')
            if i == 1 and num_pre_names == 17:
                ax.legend(loc='upper left')
            else:
                ax.get_legend().remove()

            if similarity_score is not None:
                ax.text(0.82, 0.95, f'S: {similarity_score:.2f}', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Remove any empty subplots
        if num_pre_names == 17:
            for j in [0, 2, 3]:
                fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
        plt.close()

    def pop_to_pop_weights_analysis(self, indices, variable='', source_area='', target_area='', voltage_scale=None):
        source_pop_names = other_billeh_utils.pop_names(self.networks[source_area])
        source_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
        target_pop_names = other_billeh_utils.pop_names(self.networks[target_area])
        target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
        post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
        pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

        ### Initial Weight ###
        if voltage_scale is not None:
            # Normalize the weights by the voltage scale
            initial_weights = self.model_variables_dict['Initial'][variable] * voltage_scale
            final_weights = self.model_variables_dict['Best'][variable] * voltage_scale
        else:
            initial_weights = self.model_variables_dict['Initial'][variable]
            final_weights = self.model_variables_dict['Best'][variable]

        weight_changes = final_weights - initial_weights
        df = pd.DataFrame({'Post_names': post_cell_types, 
                            'Pre_names':pre_cell_types, 
                            'Initial weight': initial_weights, 
                            'Final weight': final_weights, 
                            'Weight Change': weight_changes})
        
        # Calculate global min and max for color normalization
        # global_grouped_df = df.groupby(['Pre_names', 'Post_names'])[['Initial weight', 'Final weight']].mean().reset_index()
        # global_min = global_grouped_df[['Initial weight', 'Final weight']].min().min()
        # global_max = global_grouped_df[['Initial weight', 'Final weight']].max().max()
        # global_min = df[['Initial weight', 'Final weight']].min().min()
        # global_max = df[['Initial weight', 'Final weight']].max().max()

        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
        os.makedirs(boxplots_dir, exist_ok=True)

        # Plot for Initial Weight
        if not os.path.exists(os.path.join(boxplots_dir, 'Initial_weight.png')):
            grouped_df = df.groupby(['Pre_names', 'Post_names'])['Initial weight'].mean().reset_index()
            # Create a pivot table to reshape the data for the heatmap
            pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Initial weight')
            # Plot heatmap
            fig = plt.figure(figsize=(12, 6))
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
            heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
            plt.xlabel(f'{target_area}')
            plt.ylabel(f'{source_area}')
            plt.xticks(rotation=90)
            plt.gca().set_aspect('equal')
            plt.title(f'{variable}')
            # Create a separate color bar axis
            cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
            # Plot color bar
            cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
            cbar.set_label('Initial Weight (pA)')
            plt.savefig(os.path.join(boxplots_dir, f'Initial_weight.png'), dpi=300, transparent=False)
            plt.close()

        ### Final Weight ###
        grouped_df = df.groupby(['Pre_names', 'Post_names'])['Final weight'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Final weight')
        # Plot heatmap
        plt.figure(figsize=(12, 6))
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
        # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
        heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
        plt.xlabel(f'{target_area}')
        plt.ylabel(f'{source_area}')
        plt.xticks(rotation=90)
        plt.gca().set_aspect('equal')
        plt.title(f'{variable}')
        # Create a separate color bar axis
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
        # Plot color bar
        cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
        cbar.set_label('Final Weight (pA)')
        plt.savefig(os.path.join(boxplots_dir, f'Final_weight.png'), dpi=300, transparent=False)
        plt.close()

        ### Weight change ###
        grouped_df = df.groupby(['Pre_names', 'Post_names'])['Weight Change'].mean().reset_index()
        # Create a pivot table to reshape the data for the heatmap
        try:
            pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Weight Change')
            # Plot heatmap
            plt.figure(figsize=(12, 6))
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0)
            # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
            heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-20, vmax=20)
            plt.xlabel(f'{target_area}')
            plt.ylabel(f'{source_area}')
            plt.xticks(rotation=90)
            plt.gca().set_aspect('equal')
            plt.title(f'{variable}')
            # Create a separate color bar axis
            cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
            # Plot color bar
            cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
            cbar.set_label('Weight Change (pA)')
            plt.savefig(os.path.join(boxplots_dir, f'Weight_change.png'), dpi=300, transparent=False)
            plt.close()
        except:
            print('Skipping the plot for the weight change heatmap...')
            # raise the actual error
            print(grouped_df)

    def save_best_model(self):
        # self.step_counter.assign_add(1)
        print(f'[ Saving the model at epoch {self.epoch} ]')
        try:
            p = self.best_manager.save(checkpoint_number=self.epoch)
            print(f'Model saved in {p}\n')
        except:
            print("Saving failed. Maybe next time?")

    def plot_losses_curves(self):
        plotting_metrics = ['val_loss', 'val_rate_loss', 'val_voltage_loss', 'val_regularizer_loss', 'val_classification_loss','val_accuracy']
        images_dir = os.path.join(self.logdir, 'Loss_curves')
        os.makedirs(images_dir, exist_ok=True)

        # start_epoch = 6 if self.epoch > 5 else 1
        start_epoch = 1
        epochs = range(start_epoch, self.epoch + 1)

        fig, axs = plt.subplots(3, 2, figsize=(12, 18))
        axs = axs.ravel()  # Flatten the array for easy indexing
        for i, metric_key in enumerate(plotting_metrics):
            ax = axs[i]
            if metric_key == 'val_loss':
                color = 'red'
            else:
                color = 'blue'
            ax.plot(epochs, self.epoch_metric_values[metric_key][start_epoch-1:], color=color)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_key)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f'classification_losses_curves_epoch.png'), dpi=300, transparent=False)
        plt.close()
    
    def plot_raster(self, x, v1_spikes, lm_spikes, y, images_dir=''):
        seq_len = v1_spikes.shape[1]
        images_dir = os.path.join(images_dir, 'Classification_Raster_plots')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.networks,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=self.pre_delay,
                                    stimuli_end_time=seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=True,
                                    )
        graph(x, v1_spikes, lm_spikes)

    def plot_mean_firing_rate_boxplot(self, v1_spikes, lm_spikes, y, images_dir=''):
        seq_len = v1_spikes.shape[1]
        DG_angles = y
        boxplots_dir = os.path.join(images_dir, f'Boxplots_classification/{self.neuropixels_feature}')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        for axis_id, spikes, area in zip([0, 1], [v1_spikes, lm_spikes], ['v1', 'lm']):
            if self.neuropixels_feature == "Ave_Rate(Hz)":
                metrics_analysis = ModelMetricsAnalysis(spikes, DG_angles, self.networks[area], data_dir=self.flags.data_dir, 
                                                        drifting_gratings_init=self.pre_delay, drifting_gratings_end=seq_len-self.post_delay,
                                                        area=area, analyze_core_only=True, df_directory=self.logdir, save_df=False) 
            elif self.neuropixels_feature == 'Spontaneous rate (Hz)':
                metrics_analysis = ModelMetricsAnalysis(spikes, DG_angles, self.networks[area], data_dir=self.flags.data_dir, 
                                                        spontaneous_init=self.pre_delay, spontaneous_end=seq_len-self.post_delay,
                                                        area=area, analyze_core_only=True, df_directory=self.logdir, save_df=False) 
            elif self.neuropixels_feature == 'Natural scenes rates(Hz)':
                metrics_analysis = ModelMetricsAnalysis(spikes, DG_angles, self.networks[area], data_dir=self.flags.data_dir, 
                                                        natural_init=self.pre_delay, natural_end=seq_len-self.post_delay,
                                                        area=area, analyze_core_only=True, df_directory=self.logdir, save_df=False)
            # Figure for Average firing rate boxplots
            metrics_analysis(metrics=[self.neuropixels_feature], axis=axs[axis_id], directory=boxplots_dir, filename=f'{area}_epoch_{self.epoch}')   
                       
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'epoch_{self.epoch}.png'), dpi=300, transparent=False)
        plt.close()

    def classification_boxplots(self, x_spikes, v1_spikes, lm_spikes, y, images_dir=''):
        total_trials = min(v1_spikes.shape[0], 5)
        for trial_id in range(total_trials):
            boxplots_dir = os.path.join(images_dir, f'Readout_Populations', f'Epoch {self.epoch}')
            os.makedirs(boxplots_dir, exist_ok=True)
            # fig, axs = plt.subplots(2, 1, figsize=(12, 14))
            fig, ax = plt.subplots(figsize=(12, 6))
            # Calculate mean firing rates for each population
            readout_sums = {}
            for i in range(10):
                neurons_ids = self.networks['v1'][f'readout_neuron_ids_{i}'][0]
                readout_spikes = v1_spikes[trial_id, -self.post_delay:, neurons_ids]  # Last 50 timesteps
                readout_sums[f'{i}'] = np.sum(readout_spikes, axis=0)
            # Convert to a format suitable for plotting
            populations = list(readout_sums.keys())
            data = [readout_sums[pop] for pop in populations]
            # Calculate the total sum of spikes for each population
            total_sum = [np.sum(data[i]) for i in range(len(data))]
            largest_population_index = np.argmax(total_sum)
            # largest_population = populations[largest_population_index]
            # Target readout population (`y[-1]`)
            target_population_index = int(y[trial_id][-1])
            # target_population = f'Readout {target_population_index}'
            # Plotting the barplot with summed spikes for each population
            data_for_plotting = pd.DataFrame({
                'Population': populations,
                'Total Spikes': total_sum
            })

            max_spikes = data_for_plotting['Total Spikes'].max()
            # get the ylim with 25% larger than max value
            ylim = max_spikes + max_spikes * 0.3
            # data_for_plotting.loc[data_for_plotting['Population']=='4', 'Total Spikes'] = 6
            # Seaborn barplot with suggested colors
            # plt.figure(figsize=(12, 6))
            sns.barplot(data=data_for_plotting, x='Population', y='Total Spikes', color='lightgray', edgecolor='black', ax=ax)
            # Highlighting specific populations with suggested colors
            # ax = plt.gca()
            bars = ax.patches
            # Apply colors to bars
            for i, bar in enumerate(bars):
                if i == largest_population_index:
                    bar.set_color('gold')  # Largest population
                    bar.set_edgecolor('black')
                elif i == target_population_index:
                    bar.set_color('teal')  # Target population
                    bar.set_edgecolor('black')
            # Customize the plot
            ax.set_ylabel('Readout Population Spikes', fontsize=20)
            ax.set_xlabel('Readout Populations', fontsize=20)
            ax.set_ylim(0, ylim)
            # plt.title('Summed Spikes for Each Readout Population')
            ax.legend(handles=[
                plt.Line2D([0], [0], color='gold', lw=4, label=f'Decision'),
                plt.Line2D([0], [0], color='teal', lw=4, label=f'Label')
            ], loc='upper left', fontsize=20, framealpha=0.6)

            # Add inset with MNIST image
            inset_ax = fig.add_axes([0.75, 0.7, 0.2, 0.3])  # [x, y, width, height] relative to figure
            # inset_ax.imshow(images[9].squeeze(), cmap='gray')
            data = '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/lgn_model/data/lgn_full_col_cells_240x120.csv'
            df = pd.read_csv(data, sep=' ')
            lgn_x = df['x']
            lgn_y = df['y']
            rates = np.sum(x_spikes[trial_id, self.pre_delay:-self.post_delay, :], axis=0)
            # Create a 2D grid for the rates
            grid_x, grid_y = np.meshgrid(np.arange(240), np.arange(120))
            rate_grid = np.zeros_like(grid_x, dtype=float)
            # Populate the rate grid using positions x and y
            for i in range(len(lgn_x)):
                x_grid = int(lgn_x[i])
                y_grid = int(lgn_y[i])
                rate_grid[y_grid, x_grid] += rates[i]

            inset_ax.imshow(rate_grid, cmap='hot', interpolation='nearest')
            inset_ax.axis('off')  # No axis for the image

            fig.tight_layout()
            fig.savefig(os.path.join(boxplots_dir, f'trial_{trial_id}.png'), dpi=300, transparent=False)
            plt.close()

    def classification_stripplot(self, v1_spikes, y, images_dir=''):
        total_trials = min(v1_spikes.shape[0], 5)  # Limit to 5 trials
        for trial_id in range(total_trials):
            # Create the directory for saving images
            boxplots_dir = os.path.join(images_dir, f'Readout_Populations', f'Epoch {self.epoch}')
            os.makedirs(boxplots_dir, exist_ok=True)
            # Calculate mean firing rates for each population
            readout_sums = {}
            for i in range(10):
                neurons_ids = self.networks['v1'][f'readout_neuron_ids_{i}'][0]
                readout_spikes = v1_spikes[trial_id, -self.post_delay:, neurons_ids]  # Last `post_delay` timesteps
                readout_sums[f'{i}'] = np.sum(readout_spikes, axis=0)
            # Convert to a format suitable for plotting
            populations = list(readout_sums.keys())
            data = [readout_sums[pop] for pop in populations]
            # Determine the population with the largest mean firing rate
            total_sum = [np.sum(data[i]) for i in range(len(data))]
            largest_population_index = np.argmax(total_sum)
            # largest_population = populations[largest_population_index]
            # Target readout population
            target_population_index = int(y[trial_id][-1])
            # target_population = f'Readout {target_population_index}'
            # Plotting the stripplot
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.stripplot(data=data, ax=ax)
            ax.set_xticks(range(len(populations)))
            ax.set_xticklabels(populations, rotation=0)
            # Set y-axis to integer values
            ax.yaxis.get_major_locator().set_params(integer=True)
            ax.set_ylabel('Number of spikes')
            ax.set_title(f'Summed Spikes for Each Readout Population - Trial {trial_id}')
            # Add rectangles to highlight specific populations
            rect = plt.Rectangle((largest_population_index - 0.4, ax.get_ylim()[0]),
                                 0.8, ax.get_ylim()[1] - ax.get_ylim()[0],
                                 color='red', alpha=0.2, label=f'Decision')
            ax.add_patch(rect)
            rect2 = plt.Rectangle((target_population_index - 0.4, ax.get_ylim()[0]),
                                  0.8, ax.get_ylim()[1] - ax.get_ylim()[0],
                                  color='blue', alpha=0.2, label=f'Label')
            ax.add_patch(rect2)
            # Ensure labels don't duplicate in legend
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
            ax.set_xlabel('Readout Populations')
            # Save the figure
            fig.tight_layout()
            fig.savefig(os.path.join(boxplots_dir, f'trial_{trial_id}_stripplot.png'), dpi=300, transparent=False)
            plt.close()
            
    def classification_rates(self, x_spikes, v1_spikes, lm_spikes, y, images_dir=''):
        total_trials = min(v1_spikes.shape[0], 5)
        for trial_id in range(total_trials):
            # Set up directories for saving plots
            image_dir = os.path.join(images_dir, f'Readout_Populations', f'Epoch {self.epoch}')
            os.makedirs(image_dir, exist_ok=True)
            # Create the main figure
            fig, ax = plt.subplots(figsize=(12, 6))
            # Calculate mean firing rates for each population with gaussian smoothing
            frs = other_billeh_utils.firing_rates_smoothing(v1_spikes.astype(np.float32), sampling_rate=1000, window_size=10)[0]
            
            for i in range(10):
                neurons_ids = self.networks['v1'][f'readout_neuron_ids_{i}'][0]
                readout_spikes = frs[trial_id][:, neurons_ids]
                readout_sum = np.mean(readout_spikes, axis=1)  # Mean firing rate in Hz
                ax.plot(np.arange(frs.shape[1]), readout_sum, label=f'{i}')
            
            # Highlight time periods
            ax.axvspan(0, 50, color='gray', alpha=0.3)
            ax.axvspan(50, 150, color='green', alpha=0.1)
            ax.axvspan(150, 200, color='gray', alpha=0.3)
            # Adjust legend
            ax.legend(title="Readout Populations", loc="upper left", ncol=2)
            # Customize labels and axis limits
            ax.set_xlabel("Time (ms)", fontsize=14)
            ax.set_ylabel("Average Instantaneous Firing Rate (Hz)", fontsize=14)
            ax.set_xlim(0, 200)
            # Add an inset with LGN activity
            inset_ax = fig.add_axes([0.4, 0.7, 0.25, 0.25])  # [x, y, width, height] relative to figure
            data = '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/lgn_model/data/lgn_full_col_cells_240x120.csv'
            df = pd.read_csv(data, sep=' ')
            lgn_x = df['x']
            lgn_y = df['y']
            rates = np.sum(x_spikes[trial_id, self.pre_delay:-self.post_delay, :], axis=0)
            # Create a 2D grid for the rates
            grid_x, grid_y = np.meshgrid(np.arange(240), np.arange(120))
            rate_grid = np.zeros_like(grid_x, dtype=float)
            for i in range(len(lgn_x)):
                x_grid = int(lgn_x[i])
                y_grid = int(lgn_y[i])
                rate_grid[y_grid, x_grid] += rates[i]
            
            inset_ax.imshow(rate_grid, cmap='hot', interpolation='nearest')
            inset_ax.axis('off')  # No axis for the image
            
            # Tight layout and save the figure
            fig.tight_layout()
            fig.savefig(os.path.join(image_dir, f'trial_{trial_id}_firing_rates.png'), dpi=300, transparent=False)
            plt.close()

    def save_intermediate_checkpoint(self):
        # Save the checkpoint to reload weights in the osi_dsi_estimator
        p = self.epoch_manager.save(checkpoint_number=self.epoch)
        print(f'Checkpoint model saved in {p}\n')
       
                    
class Callbacks:
    def __init__(self, networks, lgn_inputs, bkg_inputs, model, optimizer, flags, logdir, strategy, 
                metrics_keys, pre_delay=50, post_delay=50, checkpoint=None, model_variables_init=None, 
                save_optimizer=True, spontaneous_training=False):

        self.n_neurons = {'v1': networks['v1']['n_nodes'], 'lm': networks['lm']['n_nodes']}
        self.networks = networks
        self.lgn_inputs = lgn_inputs
        self.bkg_inputs = bkg_inputs
        if spontaneous_training:
            self.neuropixels_feature = 'Spontaneous rate (Hz)'
        else:
            self.neuropixels_feature = 'Ave_Rate(Hz)'  
        self.model = model
        self.optimizer = optimizer
        self.flags = flags
        self.logdir = logdir
        self.strategy = strategy
        self.metrics_keys = metrics_keys
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.step = 0
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
            self.checkpoint_epochs = 0
            # create a dictionary to save the values of the metric keys after each epoch
            self.epoch_metric_values = {key: [] for key in self.metrics_keys}
            self.epoch_metric_values['v1_sync'] = []
            self.epoch_metric_values['lm_sync'] = []
        else:
            # Load epoch_metric_values and min_val_loss from the file
            if os.path.exists(os.path.join(self.logdir, 'train_end_data.pkl')):
                with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'rb') as f:
                    data_loaded = pkl.load(f)
                self.epoch_metric_values = data_loaded['epoch_metric_values']
                self.min_val_loss = data_loaded['min_val_loss']
                self.no_improve_epochs = data_loaded['no_improve_epochs']
                self.checkpoint_epochs = len(data_loaded['epoch_metric_values']['train_loss'])
            elif os.path.exists(os.path.join(os.path.dirname(flags.restore_from), 'train_end_data.pkl')):
                with open(os.path.join(os.path.dirname(flags.restore_from), 'train_end_data.pkl'), 'rb') as f:
                    data_loaded = pkl.load(f)
                self.epoch_metric_values = data_loaded['epoch_metric_values']
                self.min_val_loss = data_loaded['min_val_loss']
                self.no_improve_epochs = data_loaded['no_improve_epochs']
                self.checkpoint_epochs = len(data_loaded['epoch_metric_values']['train_loss'])
            else:
                print('No train_end_data.pkl file found. Initializing...')
                self.epoch_metric_values = {key: [] for key in self.metrics_keys}
                self.epoch_metric_values['v1_sync'] = []
                self.epoch_metric_values['lm_sync'] = []
                self.min_val_loss = float('inf')
                self.no_improve_epochs = 0
                self.checkpoint_epochs = 0

        self.total_epochs = flags.n_runs * flags.n_epochs + self.checkpoint_epochs
        # Manager for the best model
        self.best_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/Best_model', max_to_keep=1
        )
        # Manager for osi/dsi checkpoints 
        self.epoch_manager = tf.train.CheckpointManager(
            checkpoint, directory=self.logdir + '/Intermediate_checkpoints', max_to_keep=5
        )

    def on_train_begin(self):
        print("---------- Training started at ", dt.datetime.now().strftime('%d-%m-%Y %H:%M'), ' ----------\n')
        self.train_start_time = time()
        # self.epoch = self.flags.run_session * self.flags.n_epochs
        self.epoch = self.checkpoint_epochs
        # # Clear the session to reset the graph state
        # tf.keras.backend.clear_session()

    def on_train_end(self, metric_values, normalizers=None):
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

        if normalizers is not None:
            data_to_save['v1_ema'] = normalizers['v1_ema']
            data_to_save['lm_ema'] = normalizers['lm_ema']

        with open(os.path.join(self.logdir, 'train_end_data.pkl'), 'wb') as f:
            pkl.dump(data_to_save, f)

        # # Analize changes in trainable variables.
        # for var in self.model_variables_dict['Best'].keys():
        #     t0 = time()
        #     self.trainable_variable_change_heatmaps(var)
        #     print(f'Time spent in {var}: {time()-t0}')

        if self.flags.n_runs > 1:
            self.save_intermediate_checkpoint()

    def on_epoch_start(self):
        self.epoch += 1
        # self.step_counter.assign_add(1)
        self.epoch_init_time = time()
        date_str = dt.datetime.now().strftime('%d-%m-%Y %H:%M')
        print(f'Epoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')
        tf.print(f'Epoch {self.epoch:2d}/{self.total_epochs} @ {date_str}')

    def on_epoch_end(self, x, v1_spikes, lm_spikes, y, metric_values, bkg_noise=None, verbose=True, x_spont=None, v1_spikes_spont=None, lm_spikes_spont=None):
        
        if self.flags.dtype != 'float32':
            v1_spikes = v1_spikes.numpy().astype(np.float32)
            lm_spikes = lm_spikes.numpy().astype(np.float32)
            x = x.numpy().astype(np.float32)
            y = y.numpy().astype(np.float32)
            if x_spont is not None:
                x_spont = x_spont.numpy().astype(np.float32)
                v1_spikes_spont = v1_spikes_spont.numpy().astype(np.float32)
                lm_spikes_spont = lm_spikes_spont.numpy().astype(np.float32)
        else:
            v1_spikes = v1_spikes.numpy()
            lm_spikes = lm_spikes.numpy()
            x = x.numpy()
            y = y.numpy()
            if x_spont is not None:
                x_spont = x_spont.numpy()
                v1_spikes_spont = v1_spikes_spont.numpy()
                lm_spikes_spont = lm_spikes_spont.numpy()
        
        self.step = 0
        if self.initial_metric_values is None:
            self.initial_metric_values = metric_values
        
        if verbose:
            print_str = f'  Validation:\n' 
            val_values = metric_values[len(metric_values)//2:]
            print_str += '    ' + compose_str(val_values, mode='functional_training') 
            print(print_str)
            for gpu_id in range(len(self.strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)

        for i, (key, value) in enumerate(self.epoch_metric_values.items()):
            if key not in ['v1_sync', 'lm_sync']:
                self.epoch_metric_values[key] = value + [metric_values[i]]

        # self.epoch_metric_values = {key: value + [metric_values[i]] for i, (key, value) in enumerate(self.epoch_metric_values.items())}

        if 'val_loss' in self.metrics_keys:
            val_loss_index = self.metrics_keys.index('val_loss')
            val_loss_value = metric_values[val_loss_index]
        else:
            val_loss_index = self.metrics_keys.index('train_loss')
            val_loss_value = metric_values[val_loss_index]

        self.plot_losses_curves()
        # self.plot_synchronization_evolution(v1_spikes, lm_spikes, v1_spikes_spont, lm_spikes_spont)

        if val_loss_value < self.min_val_loss:
        # if True:
            self.min_val_loss = val_loss_value
            self.no_improve_epochs = 0
            # self.plot_bkg_noise(bkg_noise)
            # self.noise_psc(bkg_noise)
            # self.bkg_correlation_analysis(v1_spikes, lm_spikes, bkg_noise)
            self.save_best_model()
            self.plot_mean_firing_rate_boxplot(v1_spikes, lm_spikes, y)
            # self.plot_mean_osi_boxplot(v1_spikes, lm_spikes, y)

            # self.power_spectrum(v1_spikes, lm_spikes, v1_spikes_spont=v1_spikes_spont, lm_spikes_spont=lm_spikes_spont, 
            #                     directory=os.path.join(self.logdir, 'Power_spectrum'))
            directory = os.path.join(self.logdir, 'OneShotTuningAnalysis')
            self.plot_tuning_analysis(v1_spikes, lm_spikes, y, directory=directory)

            if v1_spikes_spont is not None:
                # self.plot_lgn_activity(x, x_spont)
                self.composed_raster(x, v1_spikes, lm_spikes, x_spont, v1_spikes_spont, lm_spikes_spont, y)
                self.composed_raster(x, v1_spikes, lm_spikes, x_spont, v1_spikes_spont, lm_spikes_spont, y, plot_core_only=False)
                self.plot_spontaneous_boxplot(v1_spikes_spont, lm_spikes_spont, y)
                spontaneous_directory = os.path.join(self.logdir, 'OneShotTuningAnalysis/Spontaneous')
                self.plot_tuning_analysis(v1_spikes_spont, lm_spikes_spont, y, directory=spontaneous_directory)
                # self.plot_populations_activity(v1_spikes, lm_spikes, v1_spikes_spont, lm_spikes_spont)
                # self.power_spectrum(v1_spikes_spont, lm_spikes_spont, directory=os.path.join(self.logdir, 'Spontaneous power_spectrum'))
            else:
                self.plot_raster(x, v1_spikes, lm_spikes, y)
            
            self.model_variables_dict['Best'] = {var.name: var.numpy().astype(np.float16) for var in self.model.trainable_variables}

        else:
            self.no_improve_epochs += 1

        with self.summary_writer.as_default():
            for k, v in zip(self.metrics_keys, metric_values):
                tf.summary.scalar(k, v, step=self.epoch)

        # EARLY STOPPING CONDITIONS
        if (0 < self.flags.max_time < (time() - self.epoch_init_time) / 3600):
            print(f'[ Maximum optimization time of {self.flags.max_time:.2f}h reached ]')
            stop = True
        elif self.no_improve_epochs >= 500:
            print("Early stopping: Validation loss has not improved for 50 epochs.")
            stop = True  
        else:
            stop = False

        return stop

    def on_step_start(self):
        self.step += 1
        self.step_init_time = time()
        # reset the gpu memory stat
        # tf.config.experimental.reset_memory_stats('GPU:0')

    def on_step_end(self, train_values, y, verbose=True):
        self.step_running_time.append(time() - self.step_init_time)
        if verbose:
            print_str = f'  Step {self.step:2d}/{self.flags.steps_per_epoch}: \n'
            print_str += '    ' + compose_str(train_values, mode='functional_training')
            print(print_str)
            tf.print(print_str)
            print(f'    Step running time: {time() - self.step_init_time:.2f}s')
            for gpu_id in range(len(self.strategy.extended.worker_devices)):
                printgpu(gpu_id=gpu_id)
        
    def save_intermediate_checkpoint(self):
        # Save the checkpoint to reload weights in the osi_dsi_estimator
        p = self.epoch_manager.save(checkpoint_number=self.epoch)
        print(f'Checkpoint model saved in {p}\n')
        
    def save_best_model(self):
        # self.step_counter.assign_add(1)
        print(f'[ Saving the model at epoch {self.epoch} ]')
        try:
            p = self.best_manager.save(checkpoint_number=self.epoch)
            print(f'Model saved in {p}\n')
        except:
            print("Saving failed. Maybe next time?")

    def plot_losses_curves(self):
        plotting_metrics = ['val_loss', 'val_osi_dsi_loss', 'val_rate_loss', 'val_voltage_loss', 'val_regularizer_loss', 'val_sync_loss']
        images_dir = os.path.join(self.logdir, 'Loss_curves')
        os.makedirs(images_dir, exist_ok=True)

        # start_epoch = 6 if self.epoch > 5 else 1
        start_epoch = 1
        epochs = range(start_epoch, self.epoch + 1)

        fig, axs = plt.subplots(3, 2, figsize=(12, 18))
        axs = axs.ravel()  # Flatten the array for easy indexing
        for i, metric_key in enumerate(plotting_metrics):
            ax = axs[i]
            if metric_key == 'val_loss':
                color = 'red'
            else:
                color = 'blue'
            ax.plot(epochs, self.epoch_metric_values[metric_key][start_epoch-1:], color=color)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_key)
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, f'losses_curves_epoch.png'), dpi=300, transparent=False)
        plt.close()
    
    def plot_raster(self, x, v1_spikes, lm_spikes, y):
        seq_len = v1_spikes.shape[1]
        images_dir = os.path.join(self.logdir, 'Raster_plots')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.networks,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=self.pre_delay,
                                    stimuli_end_time=seq_len-self.post_delay,
                                    reverse=False,
                                    plot_core_only=True,
                                    )
        graph(x, v1_spikes, lm_spikes)

    def composed_raster(self, x, v1_spikes, lm_spikes, x_spont, v1_spikes_spont, lm_spikes_spont, y, plot_core_only=True):
        # concatenate the normal and spontaneous arrays
        x = np.concatenate((x_spont, x), axis=1)
        v1_spikes = np.concatenate((v1_spikes_spont, v1_spikes), axis=1)
        lm_spikes = np.concatenate((lm_spikes_spont, lm_spikes), axis=1)
        seq_len = v1_spikes.shape[1]

        images_dir = os.path.join(self.logdir, 'Raster_plots')
        if plot_core_only:
            images_dir = os.path.join(images_dir, 'Core_only')
        else:
            images_dir = os.path.join(images_dir, 'Full')
        os.makedirs(images_dir, exist_ok=True)
        graph = InputActivityFigure(
                                    self.networks,
                                    self.flags.data_dir,
                                    images_dir,
                                    filename=f'Epoch_{self.epoch}_complete',
                                    frequency=self.flags.temporal_f,
                                    stimuli_init_time=int(seq_len/2) + self.pre_delay,
                                    stimuli_end_time=seq_len - self.post_delay,
                                    reverse=False,
                                    plot_core_only=plot_core_only,
                                    )
        graph(x, v1_spikes, lm_spikes)

    def plot_bkg_noise(self, x, bin_width_ms=10):
        x = x[0, :, :]
        x_mean = np.mean(x, axis=1)
        
        # Calculate number of bins
        # num_bins = int(x_mean.shape[0] / (bin_width_ms * 0.001))  # Convert bin width to seconds
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
        plt.close()

    def plot_lgn_activity(self, x, x_spont):
        x = x[0, :, :]
        x_spont = x_spont[0, :, :]
        x = np.concatenate((x_spont, x), axis=0)
        x_mean = np.mean(x, axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(x_mean)
        plt.title('Mean input activity')
        plt.xlabel('Time (ms)')
        plt.ylabel('Mean input activity')
        os.makedirs(os.path.join(self.logdir, 'Populations activity'), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, 'Populations activity', f'LGN_population_activity_epoch_{self.epoch}.png'))
        plt.close()

    def compute_synchronization_proxy(self, spike_data, area='v1', analyze_core_only=True, bin_size_ms=20, time_resolution_ms=1):
        """
        Compute the synchronization proxy for the network based on spike data.
        
        Parameters:
        - spike_data (np.ndarray): 2D array with shape (num_neurons, num_time_points)
        - bin_size_ms (int): Size of each bin in milliseconds
        - time_resolution_ms (int): Time resolution of the spike data in milliseconds
        
        Returns:
        - float: The maximum fraction of active neurons in any time bin
        """
        if analyze_core_only:
            core_neurons = 51978 if area == 'v1' else 7414
            if self.n_neurons[area] > core_neurons:
                num_neurons = core_neurons
                self.core_mask = other_billeh_utils.isolate_core_neurons(self.networks[area], n_selected_neurons=num_neurons, data_dir=self.flags.data_dir)
            else:
                self.core_mask = np.full(self.n_neurons[area], True)
        else:
            self.core_mask = np.full(self.n_neurons[area], True)

        spike_data = spike_data[:, self.core_mask]

        # Determine the number of bins
        num_time_points, num_neurons = spike_data.shape
        bin_size_points = bin_size_ms // time_resolution_ms
        num_bins = num_time_points // bin_size_points

        # Reshape the spikes tensor to [num_bins, bin_size_points, num_neurons]
        bins = np.reshape(spikes[:num_bins * bin_size_points], [num_bins, bin_size_points, num_neurons])
        # Sum spike counts within each bin
        spike_counts_per_bin = np.sum(bins, axis=1)
        # Determine active neurons in each bin
        active_neurons_per_bin = np.count_nonzero(spike_counts_per_bin, axis=1).astype(np.float32)
        # Compute the fraction of active neurons for each bin
        fraction_active_per_bin = active_neurons_per_bin / num_neurons
        # Find the maximum fraction of active neurons
        max_fraction_active = np.max(fraction_active_per_bin)

        # max_fraction_active = 0
        # # Iterate over each bin
        # for i in range(num_bins):
        #     start_idx = i * bin_size_points
        #     end_idx = start_idx + bin_size_points
        #     # Extract the bin
        #     bin_data = spike_data[start_idx:end_idx, :]
        #     # Determine active neurons in the bin
        #     # active_neurons = np.any(bin_data, axis=1)
        #     spike_counts = np.sum(spike_data[start_idx:end_idx, :], axis=0)
        #     active_neurons = np.count_nonzero(spike_counts)
        #     # Compute the fraction of active neurons
        #     fraction_active = np.sum(active_neurons) / num_neurons
        #     # Update the maximum fraction
        #     if fraction_active > max_fraction_active:
        #         max_fraction_active = fraction_active
        
        return max_fraction_active
    
    def compute_mean_pairwise_correlation(self, spikes, area='v1', analyze_core_only=True, bin_size=20):
        """
        Computes the mean pairwise correlation for neuron spikes.
        :param spike_matrix: Tensor of shape (batch_size, seq_len, n_neurons)
        :return: Scalar mean pairwise correlation
        """
        if analyze_core_only:
            core_neurons = 51978 if area == 'v1' else 7414
            if self.n_neurons[area] > core_neurons:
                num_neurons = core_neurons
                self.core_mask = other_billeh_utils.isolate_core_neurons(self.networks[area], n_selected_neurons=num_neurons, data_dir=self.flags.data_dir)
            else:
                self.core_mask = np.full(self.n_neurons[area], True)
        else:
            self.core_mask = np.full(self.n_neurons[area], True)

        spikes = spikes[:, :, self.core_mask]
        batch_size, seq_len, n_neurons = spikes.shape
        new_seq_len = seq_len // bin_size
        # Truncate the spike_matrix to be evenly divisible by bin_size
        truncated_len = new_seq_len * bin_size
        spikes = spikes[:, :truncated_len, :]

        # # use the exponential decay filter to smooth the spikes 
        # for i in range(batch_size):
        #     for j in range(n_neurons):
        #         spikes[i, :, j] = self.exponential_decay_filter(spikes[i, :, j])

        # Reshape and sum within bins
        binned_spikes = spikes.reshape(batch_size, new_seq_len, bin_size, n_neurons).sum(axis=2)

        # Transpose to shape (batch_size, n_neurons, seq_len)
        binned_spikes = np.transpose(binned_spikes, (0, 2, 1))        
        mean_pairwise_correlations = []
        for batch in range(batch_size):
            # Calculate mean spike rate for each neuron
            mean_spikes = np.mean(binned_spikes[batch], axis=1, keepdims=True)
            centered_spikes = binned_spikes[batch] - mean_spikes
            # Calculate the standard deviation of each neuron
            std_spikes = np.std(binned_spikes[batch], axis=1)
            # Remove neurons with zero standard deviation
            valid_neurons = std_spikes > 0
            centered_spikes = centered_spikes[valid_neurons, :]

            if centered_spikes.shape[0] > 1:
                # Calculate pairwise correlations
                correlation_matrix = np.corrcoef(centered_spikes)
                # # Take the absolute value of the correlation coefficients
                # correlation_matrix = np.abs(correlation_matrix)
                # Mask the diagonal (self-correlations)
                np.fill_diagonal(correlation_matrix, 0)
                # Calculate the mean of the upper triangle part of the correlation matrix (excluding diagonal)
                upper_triangle_indices = np.triu_indices(correlation_matrix.shape[0], k=1)
                mean_correlation = np.mean(correlation_matrix[upper_triangle_indices])
                mean_pairwise_correlations.append(mean_correlation)
        
        return np.abs(np.mean(mean_pairwise_correlations))
    
    def plot_synchronization_evolution(self, v1_spikes, lm_spikes, v1_spikes_spont=None, lm_spikes_spont=None, pairwise_correlation=False):
        # join the spikes in each area
        if v1_spikes_spont is not None:
            v1_spikes = np.concatenate((v1_spikes_spont, v1_spikes), axis=1)
            lm_spikes = np.concatenate((lm_spikes_spont, lm_spikes), axis=1)
        else:
            v1_spikes = v1_spikes
            lm_spikes = lm_spikes

        if pairwise_correlation:
            v1_sync = self.compute_mean_pairwise_correlation(v1_spikes, area='v1')
            lm_sync = self.compute_mean_pairwise_correlation(lm_spikes, area='lm')
            ylabel = 'Mean pairwise correlation'
        else:
            v1_sync = self.compute_synchronization_proxy(v1_spikes[0], area='v1')
            lm_sync = self.compute_synchronization_proxy(lm_spikes[0], area='lm')
            ylabel = 'Maximum fraction of active neurons'
        
        # print(f'Time spent in synchronization analysis: {time()-t0:.2f}')
        # print(f'    V1 synchronization: {v1_sync}%')
        # print(f'    LM synchronization: {lm_sync}%')

        self.epoch_metric_values['v1_sync'] = self.epoch_metric_values['v1_sync'] + [v1_sync]
        self.epoch_metric_values['lm_sync'] = self.epoch_metric_values['lm_sync'] + [lm_sync]

        epochs = range(1, self.epoch + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.epoch_metric_values['v1_sync'], label='V1')
        plt.plot(epochs, self.epoch_metric_values['lm_sync'], label='LM')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        # plt.title('Evolution of synchronization')
        plt.legend()
        # plt.grid(True)
        os.makedirs(os.path.join(self.logdir, 'Synchronization analysis'), exist_ok=True)
        plt.savefig(os.path.join(self.logdir, 'Synchronization analysis', 'synchronization_evolution.png'), dpi=300, transparent=False)
        plt.close()

    def plot_populations_activity(self, v1_spikes, lm_spikes, v1_spikes_spont, lm_spikes_spont):
        # join the spikes in each area
        v1_spikes = np.concatenate((v1_spikes_spont, v1_spikes), axis=1)
        lm_spikes = np.concatenate((lm_spikes_spont, lm_spikes), axis=1)
        seq_len = v1_spikes.shape[1]
        # # save the spikes in a pickle file 
        # with open(f'v1_lm_spikes.pkl', 'wb') as f:
        #     pkl.dump({'v1_spikes': v1_spikes, 'lm_spikes': lm_spikes}, f)

        z = [v1_spikes, lm_spikes]

        # Plot the mean firing rate of the population of neurons
        for area_id, area in enumerate(['v1', 'lm']):
            neurons = self.networks[area]['n_nodes']
            filename = f'{area}_Epoch_{self.epoch}'
            Population_activity = PopulationActivity(n_neurons=neurons, network=self.networks[area], 
                                                    stimuli_init_time=self.pre_delay, stimuli_end_time=seq_len-self.post_delay, 
                                                    image_path=self.logdir, filename=filename, data_dir=self.flags.data_dir)
            Population_activity(z[area_id], area=area, plot_core_only=True, bin_size=10)

    def bkg_correlation_analysis(self, v1_spikes, lm_spikes, bkg_noise):
        z = v1_spikes
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
        seq_len = v1_spikes.shape[1]
        DG_angles = y
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/{self.neuropixels_feature}')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        for axis_id, spikes, area in zip([0, 1], [v1_spikes, lm_spikes], ['v1', 'lm']):
            if self.neuropixels_feature == "Ave_Rate(Hz)":
                metrics_analysis = ModelMetricsAnalysis(spikes, DG_angles, self.networks[area], data_dir=self.flags.data_dir, 
                                                        drifting_gratings_init=self.pre_delay, drifting_gratings_end=seq_len-self.post_delay,
                                                        area=area, analyze_core_only=True, df_directory=self.logdir, save_df=False) 
            elif self.neuropixels_feature == 'Spontaneous rate (Hz)':
                metrics_analysis = ModelMetricsAnalysis(spikes, DG_angles, self.networks[area], data_dir=self.flags.data_dir, 
                                                        spontaneous_init=self.pre_delay, spontaneous_end=seq_len-self.post_delay,
                                                        area=area, analyze_core_only=True, df_directory=self.logdir, save_df=False) 
            
            # Figure for Average firing rate boxplots
            metrics_analysis(metrics=[self.neuropixels_feature], axis=axs[axis_id], directory=boxplots_dir, filename=f'{area}_epoch_{self.epoch}')   
                       
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'epoch_{self.epoch}.png'), dpi=300, transparent=False)
        plt.close()

    # def plot_mean_osi_boxplot(self, v1_spikes, lm_spikes, y):
    #     v1_spikes = v1_spikes.numpy()
    #     lm_spikes = lm_spikes.numpy()
#       seq_len = v1_spikes.shape[1]
    #     y = y.numpy()
    #     boxplots_dir = os.path.join(self.logdir, f'Boxplots/OSI')
    #     os.makedirs(boxplots_dir, exist_ok=True)
    #     fig, axs = plt.subplots(2, 1, figsize=(12, 14))
    #     for axis_id, spikes, area in zip([0, 1], [v1_spikes, lm_spikes], ['v1', 'lm']):
    #         metrics_analysis = ModelMetricsAnalysis(self.networks[area], data_dir=self.flags.data_dir, n_trials=1,
    #                                                 analyze_core_only=True, drifting_gratings_init=self.pre_delay, drifting_gratings_end=seq_len-self.post_delay, 
    #                                                 area=area)
    #         metrics_analysis(spikes, y, metrics=["OSI"], axis=axs[axis_id], 
    #                          directory=boxplots_dir, filename=f'{area}_epoch_{self.epoch}', df_directory=self.logdir, save_df=False)     

    #     plt.tight_layout()
    #     plt.savefig(os.path.join(boxplots_dir, f'epoch_{self.epoch}.png'), dpi=300, transparent=False)
    #     plt.close()

    def plot_spontaneous_boxplot(self, v1_spikes, lm_spikes, y):
        seq_len = v1_spikes.shape[1]
        DG_angles = y
        boxplots_dir = os.path.join(self.logdir, f'Boxplots/Spontaneous')
        os.makedirs(boxplots_dir, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        for axis_id, spikes, area in zip([0, 1], [v1_spikes, lm_spikes], ['v1', 'lm']):
            metrics_analysis = ModelMetricsAnalysis(spikes, DG_angles, self.networks[area], data_dir=self.flags.data_dir, 
                                                    spontaneous_init=self.pre_delay, spontaneous_end=seq_len-self.post_delay,
                                                    area=area, analyze_core_only=True, df_directory=self.logdir, save_df=False) 
            # Figure for Average firing rate boxplots
            metrics_analysis(metrics=['Spontaneous rate (Hz)'], axis=axs[axis_id], directory=boxplots_dir, filename=f'{area}_epoch_{self.epoch}')   
                       
        plt.tight_layout()
        plt.savefig(os.path.join(boxplots_dir, f'epoch_{self.epoch}.png'), dpi=300, transparent=False)
        plt.close()

    def plot_tuning_analysis(self, v1_spikes, lm_spikes, y, directory=''):
        seq_len = v1_spikes.shape[1]
        os.makedirs(directory, exist_ok=True)
        fig, axs = plt.subplots(2, 1, figsize=(12, 14))
        for axis_id, spikes, area in zip([0, 1], [v1_spikes, lm_spikes], ['v1', 'lm']):
            tuning_analyzer = OneShotTuningAnalysis(self.networks[area], data_dir=self.flags.data_dir, area=area, 
                                                    directory=directory, analyze_core_only=True,
                                                    drifting_gratings_init=self.pre_delay, drifting_gratings_end=seq_len-self.post_delay,
                                                    )
            tuning_analyzer(spikes, y)
            tuning_analyzer.plot_tuning_curves(self.epoch, remove_zero_rate_neurons=True)
            # tuning_analyzer.plot_max_rate_boxplots(self.epoch, remove_zero_rate_neurons=True, axis=axs[axis_id])
        
        # images_dir = os.path.join(images_dir, 'Max_rate_boxplot')
        # os.makedirs(images_dir, exist_ok=True)
        # plt.tight_layout()
        # plt.savefig(os.path.join(images_dir, f'epoch_{self.epoch}.png'), dpi=300, transparent=False)
        # plt.close()

    def power_spectrum(self, v1_spikes, lm_spikes, v1_spikes_spont=None, lm_spikes_spont=None, fs=1000, directory=''):
        # Sum the spikes over the batch size and all neurons to get a single spiking activity signal for each area
        combined_spiking_activity_v1 = v1_spikes.mean(axis=(0, 2))
        combined_spiking_activity_lm = lm_spikes.mean(axis=(0, 2))
        v1_signal = exponential_decay_filter(combined_spiking_activity_v1)
        lm_signal = exponential_decay_filter(combined_spiking_activity_lm)
        # # Compute the power spectrum for the combined signal for each area
        # seq_len = combined_spiking_activity_v1.shape[0]
        # fs = 1000.0
        # frequencies = np.fft.fftfreq(seq_len, d=1/fs)
        # fft_values_v1 = np.fft.fft(combined_spiking_activity_v1)
        # fft_values_lm = np.fft.fft(combined_spiking_activity_lm)
        # power_spectrum_v1 = np.abs(fft_values_v1) ** 2 / seq_len
        # power_spectrum_lm = np.abs(fft_values_lm) ** 2 / seq_len

        # Calculate the power spectrum
        # Sampling frequency (1 kHz)
        f_v1, power_spectrum_v1 = calculate_power_spectrum(v1_signal, fs)
        f_lm, power_spectrum_lm = calculate_power_spectrum(lm_signal, fs)

        # Plot the power spectrum
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=f_v1, y=power_spectrum_v1, label='V1', color='blue')
        sns.lineplot(x=f_lm, y=power_spectrum_lm, label='LM', color='orange')

        if v1_spikes_spont is not None:
            combined_spiking_activity_v1_spont = v1_spikes_spont.mean(axis=(0, 2))
            combined_spiking_activity_lm_spont = lm_spikes_spont.mean(axis=(0, 2))
            v1_signal_spont = exponential_decay_filter(combined_spiking_activity_v1_spont)
            lm_signal_spont = exponential_decay_filter(combined_spiking_activity_lm_spont)
            f_v1_spont, power_spectrum_v1_spont = calculate_power_spectrum(v1_signal_spont, fs)
            f_lm_spont, power_spectrum_lm_spont = calculate_power_spectrum(lm_signal_spont, fs)
            
            sns.lineplot(x=f_v1_spont, y=power_spectrum_v1_spont, label='V1 Spontaneous', linestyle='--', color='blue')
            sns.lineplot(x=f_lm_spont, y=power_spectrum_lm_spont, label='LM Spontaneous', linestyle='--', color='orange')

        # Remove the 0 Hz component for plotting
        # positive_frequencies = frequencies[:seq_len // 2]
        # positive_power_spectrum_v1 = power_spectrum_v1[:seq_len // 2]
        # positive_power_spectrum_lm = power_spectrum_lm[:seq_len // 2]
        # # # Set up the Seaborn style
        # sns.set(style="ticks")

        # plt.semilogy()
        plt.xlim([0, 50])
        plt.title('Power Spectral Density of Neuronal Spiking Activity in V1 and LM', fontsize=16)
        plt.xlabel('Frequency (Hz)', fontsize=14)
        plt.ylabel('Power Spectral Density [1/Hz]', fontsize=14)
        plt.legend()
        # plt.grid(True)
        plt.tight_layout()
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, f'epoch_{self.epoch}.png'), dpi=300, transparent=False)
        plt.close()

    # def trainable_variable_change_heatmaps(self, variable):
    #     if 'rest_of_brain_weights' in variable:
    #         area = variable.split('_')[0]
    #         self.node_to_pop_weights_analysis(self.bkg_inputs[area]['indices'], variable=variable, area=area)
    #     elif'sparse_input_weights' in variable:
    #         area = variable.split('_')[0]
    #         self.node_to_pop_weights_analysis(self.lgn_inputs[area]['indices'], variable=variable, area=area)
    #     elif 'sparse_recurrent_weights' in variable:
    #         area = variable.split('_')[0] 
    #         self.pop_to_pop_weights_analysis(self.networks[area]['synapses']['indices'], variable=variable, 
    #                                          source_area=area, target_area=area)
    #         # self.pop_to_pop_weights_distribution(self.networks[area]['synapses']['indices'], variable=variable, 
    #         #                                     source_area=area, target_area=area)
    #     elif 'sparse_interarea_weights' in variable:
    #         source_area = variable.split('_')[-1][:2] 
    #         target_area = variable.split('_')[0] 
    #         self.pop_to_pop_weights_analysis(self.networks[target_area]['interarea_synapses'][source_area]['indices'], variable=variable, 
    #         source_area=source_area, target_area=target_area)
    
    # def node_to_pop_weights_analysis(self, indices, variable='', area=''):
    #     pop_names = other_billeh_utils.pop_names(self.networks[area])
    #     target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in pop_names]
    #     if 'rest_of_brain_weights' in variable:
    #         post_indices =  np.repeat(indices[:, 0], 4)
    #     else:
    #         post_indices = indices[:, 0]

    #     post_cell_types = [target_cell_types[i] for i in post_indices]
    #     # Create DataFrame with all the necessary data
    #     df = pd.DataFrame({
    #         'Cell type': post_cell_types * 2,  # Duplicate node names for initial and final weights
    #         'Weight': self.model_variables_dict['Initial'][variable].tolist() + self.model_variables_dict['Best'][variable].tolist(),  # Combine initial and final weights
    #         'State': ['Initial'] * len(self.model_variables_dict['Initial'][variable]) + ['Final'] * len(self.model_variables_dict['Best'][variable])  # Distinguish between initial and final weights
    #     })

    #     # Count the number of cell_types fro each type
    #     # cell_type_counts = df['Cell type'].value_counts()

    #     # Sort the dataframe by Node Name and then by Type to ensure consistent order
    #     df = df.sort_values(['Cell type', 'State'])

    #     # Plotting
    #     boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
    #     os.makedirs(boxplots_dir, exist_ok=True)
    #     fig, axs = plt.subplots(2, 1, figsize=(12, 14))

    #     fig = plt.figure(figsize=(12, 6))
    #     hue_order = ['Initial', 'Final']
    #     # sns.boxplot(x='Node Name', y='Weight Change', data=df)
    #     sns.barplot(x='Cell type', y='Weight', hue='State', hue_order=hue_order, data=df)
    #     # sns.boxplot(x='Node Name', y='Weight', hue='Type', hue_order=hue_order, data=df)
    #     # plt.axhline(0, color='black', linewidth=1)  # include a horizontal black line at 0
    #     plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    #     plt.title(f'{variable}')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
    #     plt.close()

    # # def pop_to_pop_weights_distribution(self, indices, variable='', source_area='v1', target_area='v1'):
    # #     source_pop_names = other_billeh_utils.pop_names(self.networks[source_area])
    # #     source_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
    # #     target_pop_names = other_billeh_utils.pop_names(self.networks[target_area])
    # #     target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
    # #     post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
    # #     pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

    # #     df = pd.DataFrame({
    # #         'Post_names': post_cell_types,
    # #         'Pre_names': pre_cell_types,
    # #         'Initial weight': self.model_variables_dict['Initial'][variable],
    # #         'Final weight': self.model_variables_dict['Best'][variable],
    # #     })
    # #     # Melt DataFrame to long format
    # #     df_melted = df.melt(id_vars=['Post_names', 'Pre_names'], value_vars=['Initial weight', 'Final weight'], 
    # #                         var_name='Weight Type', value_name='Weight')
    # #     # Create directory for saving plots
    # #     boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}_distribution')
    # #     os.makedirs(boxplots_dir, exist_ok=True)
    # #     # Establish the order of the neuron types in the boxplots
    # #     cell_type_order = np.sort(df['Post_names'].unique())
    # #     # Define the palette
    # #     palette = {"Initial weight": "#87CEEB", "Final weight": "#FFA500"}
    # #     # Create subplots
    # #     num_pre_names = len(cell_type_order)
    # #     num_columns = 4
    # #     num_rows = (num_pre_names + num_columns - 1) // num_columns
    # #     fig, axes = plt.subplots(num_rows, num_columns, figsize=(24, 6 * num_rows))
    # #     # Flatten the axes array and handle the first row separately
    # #     axes = axes.flatten()
    # #     for i, pre_name in enumerate(cell_type_order):
    # #         if i==0:
    # #             i = 1
    # #         else:
    # #             i += 3
    # #         ax = axes[i] 
    # #         subset_df = df_melted[df_melted['Pre_names'] == pre_name]
    # #         # subset_cell_type_order = np.sort(subset_df['Post_names'].unique())
    # #         # Create boxplot for Initial and Final weights
    # #         sns.boxplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', order=cell_type_order, ax=ax, palette=palette, 
    # #                     width=0.7, fliersize=1.)
    # #         # sns.violinplot(data=subset_df, x='Post_names', y='Weight', hue='Weight Type', order=subset_cell_type_order, ax=ax, palette=palette, width=0.7,
    # #         #                split=True, inner="quart", gap=0.2)
    # #         ax.set_title(f'Source Cell Type: {pre_name}')
    # #         ax.set_ylabel('Weight', fontsize=12)
    # #         if i % num_columns == 0:  # First column
    # #             ax.set_ylabel('Weight')
    # #         else:
    # #             ax.set_ylabel('')
    # #         if i >= (num_rows - 1) * num_columns:
    # #             ax.set_xlabel('Target Cell Type')
    # #         else:
    # #             ax.set_xlabel('')
    # #         ax.tick_params(axis="x", labelrotation=90)
    # #         # Apply shadings to each layer
    # #         xticklabel = ax.get_xticklabels()
    # #         borders = get_borders(xticklabel)
    # #         # change y limit
    # #         if 'E' in pre_name:
    # #             bottom_limit = 0
    # #             upper_limit = None
    # #         else:
    # #             bottom_limit = None
    # #             upper_limit = 0
    # #         ax.set_ylim(bottom=bottom_limit, top=upper_limit)
    # #         # get the current ylim
    # #         ylim = ax.get_ylim()
    # #         draw_borders(ax, borders, ylim)
    # #         ax.legend(loc='best')
    # #     # Remove any empty subplots
    # #     for j in [0, 2, 3]:
    # #         fig.delaxes(axes[j])

    # #     # Adjust layout
    # #     plt.tight_layout()
    # #     plt.savefig(os.path.join(boxplots_dir, f'{variable}.png'), dpi=300, transparent=False)
    # #     plt.close()


    # def pop_to_pop_weights_analysis(self, indices, variable='', source_area='', target_area=''):
    #     source_pop_names = other_billeh_utils.pop_names(self.networks[source_area])
    #     source_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in source_pop_names]
    #     target_pop_names = other_billeh_utils.pop_names(self.networks[target_area])
    #     target_cell_types = [other_billeh_utils.pop_name_to_cell_type(pop_name) for pop_name in target_pop_names]
    #     post_cell_types = [target_cell_types[i] for i in indices[:, 0]]
    #     pre_cell_types = [source_cell_types[i] for i in indices[:, 1]]

    #     ### Initial Weight ###
    #     weight_changes = self.model_variables_dict['Best'][variable] - self.model_variables_dict['Initial'][variable]
    #     df = pd.DataFrame({'Post_names': post_cell_types, 
    #                         'Pre_names':pre_cell_types, 
    #                         'Initial weight': self.model_variables_dict['Initial'][variable], 
    #                         'Final weight': self.model_variables_dict['Best'][variable], 
    #                         'Weight Change': weight_changes})
        
    #     # Calculate global min and max for color normalization
    #     # global_grouped_df = df.groupby(['Pre_names', 'Post_names'])[['Initial weight', 'Final weight']].mean().reset_index()
    #     # global_min = global_grouped_df[['Initial weight', 'Final weight']].min().min()
    #     # global_max = global_grouped_df[['Initial weight', 'Final weight']].max().max()
    #     # global_min = df[['Initial weight', 'Final weight']].min().min()
    #     # global_max = df[['Initial weight', 'Final weight']].max().max()

    #     # Plot for Initial Weight
    #     grouped_df = df.groupby(['Pre_names', 'Post_names'])['Initial weight'].mean().reset_index()
    #     # Create a pivot table to reshape the data for the heatmap
    #     pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Initial weight')
    #     # Plot heatmap
    #     boxplots_dir = os.path.join(self.logdir, f'Boxplots/{variable}')
    #     os.makedirs(boxplots_dir, exist_ok=True)
    #     fig = plt.figure(figsize=(12, 6))
    #     # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
    #     heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
    #     plt.xlabel(f'{target_area}')
    #     plt.ylabel(f'{source_area}')
    #     plt.xticks(rotation=90)
    #     plt.gca().set_aspect('equal')
    #     plt.title(f'{variable}')
    #     # Create a separate color bar axis
    #     cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
    #     # Plot color bar
    #     cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
    #     cbar.set_label('Initial Weight')
    #     plt.savefig(os.path.join(boxplots_dir, f'Initial_weight.png'), dpi=300, transparent=False)
    #     plt.close()

    #     ### Final Weight ###
    #     grouped_df = df.groupby(['Pre_names', 'Post_names'])['Final weight'].mean().reset_index()
    #     # Create a pivot table to reshape the data for the heatmap
    #     pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Final weight')
    #     # Plot heatmap
    #     plt.figure(figsize=(12, 6))
    #     # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=global_min, vmax=global_max)
    #     heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
    #     plt.xlabel(f'{target_area}')
    #     plt.ylabel(f'{source_area}')
    #     plt.xticks(rotation=90)
    #     plt.gca().set_aspect('equal')
    #     plt.title(f'{variable}')
    #     # Create a separate color bar axis
    #     cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
    #     # Plot color bar
    #     cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
    #     cbar.set_label('Final Weight')
    #     plt.savefig(os.path.join(boxplots_dir, f'Final_weight.png'), dpi=300, transparent=False)
    #     plt.close()

    #     ### Weight change ###
    #     grouped_df = df.groupby(['Pre_names', 'Post_names'])['Weight Change'].mean().reset_index()
    #     # Create a pivot table to reshape the data for the heatmap
    #     try: 
    #         pivot_df = grouped_df.pivot(index='Pre_names', columns='Post_names', values='Weight Change')
    #         # Plot heatmap
    #         plt.figure(figsize=(12, 6))
    #         # heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0)
    #         heatmap = sns.heatmap(pivot_df, cmap='RdBu_r', annot=False, cbar=False, center=0, vmin=-0.5, vmax=0.5)
    #         plt.xlabel(f'{target_area}')
    #         plt.ylabel(f'{source_area}')
    #         plt.xticks(rotation=90)
    #         plt.gca().set_aspect('equal')
    #         plt.title(f'{variable}')
    #         # Create a separate color bar axis
    #         cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.55])  # [left, bottom, width, height]
    #         # Plot color bar
    #         cbar = plt.colorbar(heatmap.collections[0], cax=cbar_ax)
    #         cbar.set_label('Weight Change')
    #         plt.savefig(os.path.join(boxplots_dir, f'Weight_change.png'), dpi=300, transparent=False)
    #         plt.close()

    #     except:
    #         print('Error in creating pivot table')
    #         print(grouped_df)
    #         print(df)
    #         print(source_area, target_area)
    #         print(self.model_variables_dict['Best'][variable])
    #         print(self.model_variables_dict['Initial'][variable])
    #         # save the df
    #         df.to_csv('error.csv')
        
                
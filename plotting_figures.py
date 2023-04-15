# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:03:12 2022

@author: javig
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

def spike_effect_correction_plot(v, z, neuron_id, tf_id, pre_spike_gap=1, post_spike_gap=5, path=''):
    n_simulations, simulation_length, n_neurons = v.shape
    v = v.reshape((n_simulations*simulation_length, n_neurons))
    z = z.reshape((n_simulations*simulation_length, n_neurons))
    v_corrected = np.copy(v)
    min_v = v_corrected[:,neuron_id].min()
    max_v = v_corrected[:,neuron_id].max()
    vs = v_corrected[:,neuron_id]
    zs = z[:,neuron_id].astype(dtype=bool)
    for t_idx, spike in enumerate(zs[:-post_spike_gap]):
        if spike and t_idx >= pre_spike_gap:
            prev_value = vs[t_idx-pre_spike_gap]
            post_value = vs[t_idx+post_spike_gap]
            xp = [t_idx-pre_spike_gap, t_idx+post_spike_gap]
            fp = [prev_value, post_value]
            new_values = np.interp(np.arange(t_idx-pre_spike_gap+1, t_idx+post_spike_gap, 1), xp, fp)
            # new_values = np.interp([t_idx-1, t_idx, t_idx+1, t_idx+2], xp, fp)
            v_corrected[t_idx-pre_spike_gap+1:t_idx+post_spike_gap, neuron_id] = new_values

    times = np.linspace(0, (simulation_length/1000), simulation_length)
        
    fig, axs = plt.subplots(3, sharex=True)
    axs[0].plot(times, v[:int(simulation_length), neuron_id], color='r', ms=1,
                 alpha=0.7, label='Original')
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel('V [mV]')
    axs[0].set_ylim(min_v-5, max_v+10)
    axs[1].plot(times, v_corrected[:int(simulation_length), neuron_id], color='r', ms=1,
                 alpha=0.7, label=f'pre_{pre_spike_gap}_post_{post_spike_gap}')
    axs[1].set_ylabel('V [mV]')
    axs[1].set_ylim(min_v-5, max_v+10)
    axs[1].legend(loc='upper right')
    axs[2].plot(times, z[:int(simulation_length), neuron_id], color='b',
                 ms=1, alpha=0.7, label='Membrane potential')
    axs[2].set_yticks([0, 1])
    axs[2].set_ylim(0, 1)
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Spikes')
    
    stimuli_init_time = 0.5
    stimuli_end_time = stimuli_init_time + 1
    # if voltage_threshold is not None:
    #     horizontal_line = axs[0].hlines(self.mean_potential_variations[neuron_id], stimuli_init_time, stimuli_end_time, linewidth=2, 
    #                                   color='b', linestyles='-', label='Mean stimulus response', zorder=1000)
    for subplot in range(3):
        axs[subplot].axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1)
        axs[subplot].axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1)

    path = os.path.join(path, 'Voltage corrected samples')
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, f'pre_{pre_spike_gap}_post_{post_spike_gap}_idx_{tf_id}'), dpi=300, transparent=True)
    plt.close(fig)
    
    
def plot_neuron_voltage_and_spikes_and_input_current(voltage, input_current, firing_rate, neuron_id, tf_id=None, 
                                                     mean_potential_variations=None, mean_input_current_variations=None,
                                                     baseline_potential=None, baseline_input_current=None,
                                                     voltage_threshold=None, input_current_threshold=None, 
                                                     fig_title='', path=''):
    
    n_simulations, simulation_length, n_neurons = voltage.shape
    fig, axs = plt.subplots(3)
    times = np.linspace(0, simulation_length/1000, simulation_length)
    axs[0].plot(times, np.mean(voltage[:,:, neuron_id], axis=0), color='r', ms=1,
                alpha=0.7, label='Membrane potential')
    
    axs[0].fill_between(times, np.mean(voltage[:,:, neuron_id], axis=0) 
                              + stats.sem(voltage[:,:, neuron_id], axis=0), 
                              np.mean(voltage[:,:, neuron_id], axis=0)
                              - stats.sem(voltage[:,:, neuron_id], axis=0), 
                              alpha=0.3, color='r')
    axs[0].set_ylabel(r'$V_m$ [mV]', fontsize=12)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    axs[0].tick_params(axis='y', which='both', labelsize=10) 
    axs[1].plot(times, np.mean(input_current[:,:, neuron_id], axis=0), color='r', ms=1,
                 alpha=0.7, label='input_current')
    axs[1].fill_between(times, np.mean(input_current[:,:, neuron_id], axis=0) 
                              + stats.sem(input_current[:,:, neuron_id], axis=0), 
                              np.mean(input_current[:,:, neuron_id], axis=0)
                              - stats.sem(input_current[:,:, neuron_id], axis=0), 
                              alpha=0.3, color='r')
    stats.sem(firing_rate[:,:, neuron_id], axis=0)
    axs[1].set_ylabel(r'$I_{syn}$ [pA]', fontsize=12)
    axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    axs[1].tick_params(axis='y', which='both', labelsize=10) 
    times = np.linspace(0, simulation_length/1000, firing_rate.shape[1])
    axs[2].plot(times, np.mean(firing_rate[:,:, neuron_id], axis=0), color='b',
                 ms=1, alpha=0.7, label='Spikes')
    axs[2].fill_between(times, np.mean(firing_rate[:,:, neuron_id], axis=0) 
                              + stats.sem(firing_rate[:,:, neuron_id], axis=0), 
                              np.mean(firing_rate[:,:, neuron_id], axis=0)
                              - stats.sem(firing_rate[:,:, neuron_id], axis=0), 
                              alpha=0.3, color='b')
    axs[2].set_xlabel('Time [s]', fontsize=12)
    axs[2].set_ylabel('Firing rate \n [Hz]', fontsize=12)
    axs[2].tick_params(axis='both', which='both', labelsize=10) 
    
    stimuli_init_time = 0.5
    stimuli_end_time = stimuli_init_time + 1
    # if voltage_threshold is not None:
    #     horizontal_line = axs[0].hlines(mean_potential_variations[neuron_id], 
    #                                     stimuli_init_time, stimuli_end_time, linewidth=2, 
    #                                     color='b', linestyles='-', label='Mean voltage', zorder=1000)  
    #     threshold = axs[0].axhline(-voltage_threshold, 0, (simulation_length/1000), linestyle='dotted', color='k', linewidth=2, zorder=1000, label='Classification threshold')
    #     axs[0].axhline(voltage_threshold, 0, (simulation_length/1000), linestyle='dotted', color='k', linewidth=2, zorder=1000)
    #     axs[0].legend(handles = [horizontal_line, threshold], loc='best')
    if input_current_threshold is not None:
        horizontal_line2 = axs[1].hlines(mean_input_current_variations[neuron_id], 
                                         stimuli_init_time, stimuli_end_time, linewidth=2, 
                                         color='b', linestyles='-', label='Mean input current', zorder=1000)
        threshold2 = axs[1].axhline(-input_current_threshold, 0, (simulation_length/1000), linestyle='dotted', color='k', linewidth=2, zorder=1000, label='Classification threshold')
        axs[1].axhline(input_current_threshold, 0, (simulation_length/1000), linestyle='dotted', color='k', linewidth=2, zorder=1000)
        # axs[1].legend(handles = [horizontal_line2, threshold2], loc='best')
    for subplot in range(3):
        axs[subplot].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5)
        axs[subplot].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5)
    # if baseline_potential is not None:
    #     axs[0].axhline(baseline_potential[neuron_id], linestyle='dotted', color='gray', linewidth=2, 
    #                 label='Mean pre-stimulus voltage: %.2f mV'% baseline_potential[neuron_id])
    #     axs[1].axhline(baseline_input_current[neuron_id], linestyle='dotted', color='gray', linewidth=2, 
    #                 label='Mean pre-stimulus input_current: %.2f pA'% baseline_input_current[neuron_id])
    
    fig.suptitle(fig_title)
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    if tf_id is not None:
        neuron_id = tf_id
    fig.savefig(os.path.join(path, 'neurons_{n_neurons}_index_{neuron_id}.png'.format(
        n_neurons=n_neurons, neuron_id=neuron_id)), dpi=300, transparent=True)
    plt.close(fig)
        
def plot_neuron_input_currents(input_current, recurrent_current, bottom_up_current, neuron_id, 
                               tf_id=None, fig_title='', path='', input_current_threshold=None):
    fig = plt.figure()
    n_simulations, simulation_length, n_neurons = input_current.shape
    asc = input_current - recurrent_current - bottom_up_current
    times = np.linspace(0, simulation_length/1000, simulation_length)
    plt.plot(times, np.mean(input_current[:,:, neuron_id], axis=0), color='r', ms=1,
                 alpha=0.7, label='Total input current')
    plt.plot(times, np.mean(recurrent_current[:,:, neuron_id], axis=0), color='olive', ms=1,
                 alpha=0.7, label='Recurrent input current')
    plt.plot(times, np.mean(bottom_up_current[:,:, neuron_id], axis=0), color='dodgerblue', ms=1,
                 alpha=0.7, label='Bottom up input current')
    # plt.plot(times, np.mean(asc[:,:, neuron_id], axis=0),
    #              alpha=0.7, label='ASC')
    
    # if input_current_threshold is not None:
    #     plt.axhline(-input_current_threshold, 0, (simulation_length/1000), linestyle='dotted', color='k', linewidth=2, zorder=1000, label='Classification threshold')
    #     plt.axhline(input_current_threshold, 0, (simulation_length/1000), linestyle='dotted', color='k', linewidth=2, zorder=1000)
    #plt.ylim(-5, 5)
    plt.ylabel('Input current [pA]', fontsize=12)
    plt.xlabel('Time [s]', fontsize=12)
    # plt.legend()
            
    stimuli_init_time = 0.5 
    stimuli_end_time = stimuli_init_time + 1
    plt.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5)
    plt.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5)
    plt.axhline(0, linestyle='-', color='k', linewidth=1)
    plt.tick_params(axis='both', labelsize=10)
    
    fig.suptitle(fig_title)
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    if tf_id is not None:
        neuron_id = tf_id
    fig.savefig(os.path.join(path, 'neurons_{n_neurons}_index_{neuron_id}_currents.png'.format(
        n_neurons=n_neurons, neuron_id=neuron_id)), dpi=300, transparent=True)
    plt.close(fig)
    
   
class DriftingGrating:
    def __init__(self, scale=2., frequency=2., simulation_length=2500, drifting_init_time=500, drifting_end_time=1500, reverse=False, marker_size=1., alpha=1, color='g'):
        self.marker_size = marker_size
        self.alpha = alpha
        self.color = color
        self.scale = scale
        self.simulation_length = simulation_length
        self.drifting_init_time = drifting_init_time
        self.drifting_end_time = drifting_end_time
        self.reverse = reverse
        self.frequency = frequency

    def __call__(self, ax, time_length):
        times = np.arange(time_length)
        stimuli_speed = np.zeros((time_length))
        if self.reverse:
            stimuli_speed[:self.drifting_init_time] = self.frequency
            stimuli_speed[self.drifting_end_time:] = self.frequency
        else:
            stimuli_speed[self.drifting_init_time:self.drifting_end_time] = self.frequency
        
        ax.plot(times, stimuli_speed, color=self.color,
                    ms=self.marker_size, alpha=self.alpha, linewidth=2*self.scale)
        ax.set_ylabel('Visual flow \n [Hz]', fontsize=10)
        ax.set_yticks([0, self.frequency])
        ax.set_yticklabels(['0', f'{self.frequency}'], fontsize=8)
        ax.set_xlim([0, time_length])
        ax.set_xticks(np.linspace(0, time_length, 6))
        ax.set_xticklabels(np.linspace(0, self.simulation_length/1000, 6), fontsize=8)
        ax.set_xlabel('Time [s]', fontsize=10)
        #ax.tick_params(axis='both', which='major', labelsize=18)
        
def firing_rate_heatmap(firing_rates, sampling_interval, frequency, drifting_init_time=500, drifting_end_time=1500, reverse=False, simulation_length=2500, path=''):   
    smoothed_firing_rates = np.mean(firing_rates, axis=0)
    firing_rates_non_zero = np.where(np.std(smoothed_firing_rates, axis=0)!=0)[0]
    normalised_smoothed_firing_rates = smoothed_firing_rates/np.max(smoothed_firing_rates, axis=0)
    firing_rates = normalised_smoothed_firing_rates[:, firing_rates_non_zero]
    mean_variations = np.mean(firing_rates[int(500/sampling_interval): int(1500/sampling_interval), :], axis=0)
    sorted_indices = np.arange(firing_rates.shape[1])
    sorted_indices = [x for _, x in sorted(zip(mean_variations, sorted_indices), key=lambda element: (element[0]), reverse=True)]   
    firing_rates_ordered = firing_rates[:,sorted_indices]
            
    fig = plt.figure()
    grid = fig.subplots(nrows=2, ncols=2,
                   gridspec_kw={'width_ratios':(1,0.05), 'height_ratios':(1,0.1)}, sharex='col')
    firing_rates_ax, cax1 = grid[0]
    drifting_grating_ax, cax2 = grid[1]
    cax2.set_visible(False)
    
    firing_rates_ax.clear()
    drifting_grating_ax.clear()
    time_length = firing_rates.shape[0]
    cs = firing_rates_ax.imshow(np.transpose(firing_rates_ordered),
                                interpolation='none', aspect='auto', cmap='coolwarm')
    cbar = fig.colorbar(cs, cax=cax1, extend='neither')
    cbar.set_label(label=r'Normalized firing rate (FR)', weight='bold', fontsize=8)
    firing_rates_ax.axvline(int(500/sampling_interval), linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    firing_rates_ax.axvline(int(1500/sampling_interval), linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    firing_rates_ax.set_ylabel('Sorted neuron')
    firing_rates_ax.tick_params(axis='x',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False) 
    drifting_grating_plot = DriftingGrating(frequency=frequency, drifting_init_time=int(drifting_init_time/sampling_interval), 
                                            drifting_end_time=int(drifting_end_time/sampling_interval), reverse=reverse)
    drifting_grating_plot(drifting_grating_ax, time_length)
    
    plt.subplots_adjust(wspace=0.07, hspace=0.07)
    #fig.canvas.draw()
    drifting_grating_ax.set_xlim(firing_rates_ax.get_xlim())
    
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'normalized_firing_rate_new.png'), dpi=300, transparent=True)#
    
def baseline_histogram(in_current, rec_current, bottom_current, asc, v, path):
    n_selected_neurons = len(in_current)
    fig = plt.figure()
    ax1 = plt.subplot(4, 2, 1)
    plt.hist(in_current, label='Total', color='r', alpha=0.7)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.ylim(0, n_selected_neurons/3)
    plt.legend(fontsize=8)
    plt.ylabel('# neurons', fontsize=9)
    ax2 = plt.subplot(4, 2, 3, sharex=ax1, sharey=ax1)
    plt.hist(rec_current, label='Recurrent', color='olive', alpha=0.7)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.legend(fontsize=8)
    plt.ylabel('# neurons', fontsize=9)
    ax3 = plt.subplot(4, 2, 5, sharex=ax1, sharey=ax1)
    plt.hist(bottom_current, label='Bottom up', color='dodgerblue', alpha=0.7)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.ylabel('# neurons', fontsize=9)
    plt.legend(fontsize=8)
    plt.subplot(4, 2, 7, sharex=ax1, sharey=ax1)
    plt.hist(asc, label='ASC', color='gray', alpha=0.7)
    plt.ylabel('# neurons', fontsize=9)
    plt.xlabel('Average input current [pA]', fontsize=9)
    plt.legend(fontsize=8)
    plt.subplot(1, 2, 2, sharey=ax1)
    plt.hist(v, color='b', alpha=0.7)
    plt.xlabel('Average membrane potential [mV]', fontsize=9)
    plt.tick_params(axis='both', labelsize=9)
    # fig.suptitle('Mean pre-stimulus analysis')
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'resting_distribution.png'), dpi=300, transparent=True)
    plt.close(fig) 
    
def firing_rate_histogram(drifting_fr, static_fr, fr_threshold, path=''):
    nan_inf_firing_rate_ratio =drifting_fr/static_fr
    remove_nans_and_infs = np.logical_not(np.logical_or(np.isnan(nan_inf_firing_rate_ratio), nan_inf_firing_rate_ratio == np.inf))
    fig = plt.figure()
    plt.hist(nan_inf_firing_rate_ratio[remove_nans_and_infs], bins=np.logspace(np.log10(0.01),np.log10(50), 50))
    plt.axvline(1/fr_threshold, color='k', linestyle='--', linewidth=2, zorder=1000, label='Class threshold')
    plt.axvline(fr_threshold, color='k', linestyle='--', linewidth=2, zorder=1000)
    plt.axvspan(0.05, 0.5, alpha=0.2, color='#F06233')
    plt.axvspan(0.5, 2, alpha=0.2, color='#9CB0AE')
    plt.axvspan(2, 50, alpha=0.2, color='#33ABA2')
    plt.gca().set_xscale("log")
    plt.figtext(0.2, 0.8, 'hVf', fontsize=12, color='#F06233', fontweight='bold')
    plt.figtext(0.45, 0.8, 'unc', fontsize=12, color='#9CB0AE', fontweight='bold')
    plt.figtext(0.75, 0.8, 'dVf', fontsize=12, color='#33ABA2', fontweight='bold')
    plt.xlabel('Firing rate ratio')
    plt.ylabel('# neurons')
    plt.xlim(0.05, 50)
    plt.legend()
    fig.suptitle("Firing rate ratio histogram")
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'firing_rate_ratio.png'), dpi=300, transparent=True)
    plt.close(fig)

    # Plot the most excited neuron
    max_firing_rate_value = np.max(nan_inf_firing_rate_ratio[remove_nans_and_infs])
    most_excited_neuron_id = np.where(nan_inf_firing_rate_ratio==max_firing_rate_value)[0][0]
    
    return most_excited_neuron_id
    
def modulation_index_histogram(drifting_fr, static_fr, path=''):
    nan_inf_mi = (drifting_fr - static_fr)/(drifting_fr + static_fr)
    remove_nans_and_infs = np.logical_not(np.logical_or(np.isnan(nan_inf_mi), nan_inf_mi == np.inf))
    fig = plt.figure()
    a = -np.logspace(np.log10(2), np.log10(0.1), 20)
    b = np.logspace(np.log10(0.1),np.log10(40), 30)
    log_array = np.concatenate((a, b))
    # plt.hist(self.mi[self.remove_nans_and_infs_mi], bins=np.logspace(np.log10(0.01),np.log10(50), 50))
    plt.hist(nan_inf_mi[remove_nans_and_infs], bins=log_array)
    plt.axvline(0, color='k', linestyle='--', linewidth=2, label='Class threshold') #threshold
    plt.axvspan(-1, 0, alpha=0.2, color='#F06233')
    plt.axvspan(0, 40, alpha=0.2, color='#33ABA2')
    # plt.gca().set_xscale("log")
    plt.xscale('symlog')
    plt.figtext(0.12, 0.8, 'hVf', fontsize=12, color='#F06233', fontweight='bold')
    plt.figtext(0.6, 0.8, 'dVf', fontsize=12, color='#33ABA2', fontweight='bold')
    plt.xlim(-1, 40)
    plt.xlabel('Modulation index')
    plt.ylabel('# neurons')
    plt.legend()
    fig.suptitle("Modulation index histogram")
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'modulation_index.png'), dpi=300, transparent=True)
    plt.close(fig)
        
def plot_average_classes(neuron_populations, neuron_variable='voltage_changes', 
                         voltage_threshold=1, input_current_threshold=0.2,
                         path=''):
    fig = plt.figure()
    for neurons_subset_label, subset_info in neuron_populations.items():
        n_simulations, simulation_length, n_neurons = subset_info[neuron_variable].shape
        mean_neurons_subset = np.mean(subset_info[neuron_variable], axis=(0, 2))
        sem_neurons_subset = (np.std(subset_info[neuron_variable], ddof=1, axis=(0, 2)) /
                        np.sqrt(n_neurons))
        times = np.linspace(0, (simulation_length/1000), simulation_length)   
        plt.plot(times, mean_neurons_subset, color=subset_info['color'],
                label=neurons_subset_label)
        plt.fill_between(times, 
                          mean_neurons_subset + sem_neurons_subset, 
                          mean_neurons_subset - sem_neurons_subset, 
                          alpha=0.3, color=subset_info['color'])
    
    stimuli_init_time = 0.5
    stimuli_end_time = stimuli_init_time + 1
    plt.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    plt.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    plt.hlines(0, 0, (simulation_length/1000), color='k', linewidths=1,zorder=10)
    
    if neuron_variable == 'voltage_changes':
        plt.hlines(voltage_threshold, 0, (simulation_length/1000), linestyles='dotted', color='k', linewidths=2,zorder=10, label='Classification threshold')
        plt.hlines(-voltage_threshold, 0, (simulation_length/1000), linestyles='dotted', color='k', linewidths=2,zorder=10)
        plt.ylabel('Voltage variations [mV]')
    elif neuron_variable == 'input_current_changes':
        plt.hlines(input_current_threshold, 0, (simulation_length/1000), linestyles='dotted', color='k', linewidths=2,zorder=10, label='Classification threshold')
        plt.hlines(-input_current_threshold, 0, (simulation_length/1000), linestyles='dotted', color='k', linewidths=2,zorder=10)
        plt.ylabel('Input current [pA]')
        plt.ylim(-2, 2)
    else:
        plt.ylabel('Input current [pA]')
        plt.ylim(-2, 2)
        
    plt.xlabel('Time [s]')
    plt.tick_params(axis='both', labelsize=12)
    leg = plt.legend()
    neurons_color = ['#F06233', '#33ABA2', '#9CB0AE']
    for text, color in zip(leg.get_texts(), neurons_color):
        plt.setp(text, color=color)
    leg.set_zorder(102)
    
    fig.suptitle(neuron_variable)
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, f'average_movement_{neuron_variable}_response_per_class.png'), dpi=300, transparent=True)

def subplot_currents_average_classes(neuron_populations, input_current_threshold=0.2, path=''):
    
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(9, 3), constrained_layout=True)
    neuron_variables = ['recurrent_current_changes', 'bottom_up_current_changes', 'input_current_changes']
    titles = ['Recurrent', 'Bottom up', 'Total']
    
    stimuli_init_time = 0.5
    stimuli_end_time = stimuli_init_time + 1
    for idx in range(3):
        for neurons_subset_label, subset_info in neuron_populations.items():
            neuron_variable = neuron_variables[idx]
            n_simulations, simulation_length, n_neurons = subset_info[neuron_variable].shape
            mean_neurons_subset = np.mean(subset_info[neuron_variable], axis=(0, 2))
            sem_neurons_subset = (np.std(subset_info[neuron_variable], ddof=1, axis=(0, 2)) /
                            np.sqrt(n_neurons))
            times = np.linspace(0, (simulation_length/1000), simulation_length)   
            axs[idx].plot(times, mean_neurons_subset, color=subset_info['color'],
                    label=neurons_subset_label, linewidth=1)
            axs[idx].fill_between(times, 
                              mean_neurons_subset 
                              + sem_neurons_subset, 
                              mean_neurons_subset 
                              - sem_neurons_subset, 
                              alpha=0.3, color=subset_info['color'])
        axs[idx].set_xlabel('Time [s]', fontsize=8)
        axs[idx].tick_params(axis='both', labelsize=8)
        axs[idx].axhline(0, color='k', linewidth=1)
        axs[idx].axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        axs[idx].axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)        
        axs[idx].set_title(titles[idx], fontsize=9)

    leg = axs[0].legend(fontsize=7)
    neurons_color = ['#F06233', '#33ABA2', '#9CB0AE']
    for text, color in zip(leg.get_texts(), neurons_color):
        plt.setp(text, color=color)
    leg.set_zorder(102)
    axs[0].set_ylabel('Input current [pA]', fontsize=8)
    threshold = axs[2].hlines(input_current_threshold, 0, (simulation_length/1000), 
                              linestyles='dotted', color='k', linewidths=1,zorder=10, label='Classification threshold')
    axs[2].hlines(-input_current_threshold, 0, (simulation_length/1000), 
                  linestyles='dotted', color='k', linewidths=1,zorder=10)
    axs[2].legend([threshold],['Class threshold'], fontsize=7)            
    # fig.suptitle('Input current averages')
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'currents_average_per_class_subplot.png'), dpi=300, transparent=True)

def plot_current_per_class(neuron_populations, input_current_threshold=0.2, path=''):
    currents_variables = ['input_current_changes', 'recurrent_current_changes', 'bottom_up_current_changes']#, 'asc_changes']
    current_labels = ['Total', 'Recurrent', 'Bottom up'] #, 'ASC']
    currents_colors = ['r', 'olive', 'dodgerblue'] #, 'gray']
    for neurons_subset_label, subset_info in neuron_populations.items():
        fig = plt.figure()
        for neuron_variable, label, current_color in zip(currents_variables, current_labels, currents_colors):
            n_simulations, simulation_length, n_neurons = subset_info[neuron_variable].shape
            mean_neurons_subset = np.mean(subset_info[neuron_variable], axis=(0, 2))
            sem_neurons_subset = (np.std(subset_info[neuron_variable], ddof=1, axis=(0, 2)) /
                            np.sqrt(n_neurons))
            times = np.linspace(0, (simulation_length/1000), simulation_length) 
            plt.plot(times, mean_neurons_subset, color=current_color,
                    label=label, alpha=0.7)
            plt.fill_between(times, 
                              mean_neurons_subset + sem_neurons_subset, 
                              mean_neurons_subset - sem_neurons_subset, 
                              alpha=0.3, color=current_color)

        stimuli_init_time = 0.5
        stimuli_end_time = stimuli_init_time + 1
        plt.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        plt.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        plt.hlines(0, 0, (simulation_length/1000), color='k', linewidths=1, zorder=10)
        plt.hlines(input_current_threshold, 0, (simulation_length/1000), linestyles='dotted', color='k', linewidths=2,zorder=10, label='Classification threshold')
        plt.hlines(-input_current_threshold, 0, (simulation_length/1000), linestyles='dotted', color='k', linewidths=2,zorder=10)
        plt.ylim(-2, 2)
        plt.ylabel('Input current [pA]')
        plt.xlabel('Time [s]')
        plt.tick_params(axis='both', labelsize=12)
        leg = plt.legend()
        for text, color in zip(leg.get_texts(), currents_colors):
            plt.setp(text, color=color)
        leg.set_zorder(102)
        
        fig.suptitle(f'Average {neurons_subset_label} neuron current response')
        os.makedirs(path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(path, f'average_{neurons_subset_label}_neuron_current_response.png'), dpi=300, transparent=True)

def subplot_current_per_class(neuron_populations, input_current_threshold=0.2, path=''): 
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(9, 3), constrained_layout=True)
    currents_variables = ['input_current_changes', 'recurrent_current_changes', 'bottom_up_current_changes']#, 'asc_changes']
    current_labels = ['Total', 'Recurrent', 'Bottom up']#, 'ASC']
    currents_colors = ['r', 'olive', 'dodgerblue']#, 'gray']
    for index, (neurons_subset_label, subset_info) in enumerate(neuron_populations.items()):
        for neuron_variable, label, current_color in zip(currents_variables, current_labels, currents_colors):
            n_simulations, simulation_length, n_neurons = subset_info[neuron_variable].shape
            mean_neurons_subset = np.mean(subset_info[neuron_variable], axis=(0, 2))
            sem_neurons_subset = (np.std(subset_info[neuron_variable], ddof=1, axis=(0, 2)) /
                            np.sqrt(n_neurons))
            times = np.linspace(0, (simulation_length/1000), simulation_length)
            axs[index].plot(times, mean_neurons_subset, color=current_color,
                    label=label, alpha=0.7, linewidth=1)
            axs[index].fill_between(times, 
                              mean_neurons_subset + sem_neurons_subset, 
                              mean_neurons_subset - sem_neurons_subset, 
                              alpha=0.3, color=current_color)
        
        stimuli_init_time = 0.5
        stimuli_end_time = stimuli_init_time + 1
        axs[index].axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        axs[index].axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        axs[index].axhline(0, color='k', linewidth=1, zorder=10)
        axs[index].axhline(input_current_threshold, linestyle='dotted', color='k', linewidth=1,zorder=10, label='Class threshold')
        axs[index].axhline(-input_current_threshold, linestyle='dotted', color='k', linewidth=1,zorder=10)
        axs[index].set_ylim(-2, 2)
        axs[index].set_xlabel('Time [s]', fontsize=8)
        axs[index].set_title(neurons_subset_label, fontsize=9)
        axs[index].tick_params(axis='both', labelsize=8)
        leg = plt.legend(fontsize=7)
        for text, color in zip(leg.get_texts(), currents_colors):
            plt.setp(text, color=color)
        leg.set_zorder(102)
        
    axs[0].set_ylabel('Input current [pA]', fontsize=8)
    # fig.suptitle(f'Average {neurons_subset_label} neuron current response')
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'average_subplot_neuron_current_response.png'), dpi=300, transparent=True)

def firing_rate_per_class(smoothed_firing_rate_selected, hVf_mask, dVf_mask, unc_mask, simulation_length=2500, path=''):
    smoothed_firing_rates = np.mean(smoothed_firing_rate_selected, axis=0)
    firing_rates_non_zero = np.where(np.std(smoothed_firing_rates, axis=0)!=0)[0]
    fr = smoothed_firing_rates[:, firing_rates_non_zero]
    fig = plt.figure()
    ax = plt.subplot(111)
    times = np.linspace(0, (simulation_length/1000), fr.shape[0])
    ax.plot(times, np.mean(fr[:, hVf_mask[firing_rates_non_zero]], axis=1), '#F06233', label=f'hVf:{fr[:, hVf_mask[firing_rates_non_zero]].shape[1]} - {np.sum(hVf_mask)}')
    ax.plot(times, np.mean(fr[:, dVf_mask[firing_rates_non_zero]], axis=1), '#33ABA2', label=f'dVf:{fr[:, dVf_mask[firing_rates_non_zero]].shape[1]} - {np.sum(dVf_mask)}')
    ax.plot(times, np.mean(fr[:, unc_mask[firing_rates_non_zero]], axis=1), '#9CB0AE', label=f'unc:{fr[:, unc_mask[firing_rates_non_zero]].shape[1]} - {np.sum(unc_mask)}')
    stimuli_init_time = 0.5
    stimuli_end_time = stimuli_init_time + 1
    ax.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    ax.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    ax.legend()
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Firing rate [Hz]')
    # ax.set_xticks(np.linspace(0, fr.shape[0], 5))
    # ax.set_xticklabels(np.linspace(0, 2500, 5))  
    fig.savefig(os.path.join(path, 'fr_per_class.png'), dpi=300, transparent=True)

def keller_heatmap(voltages, neuron_populations='', classes=True, path=''):
    simulation_length, n_neurons = voltages.shape
    fig = plt.figure()
    ax = plt.subplot(111)      
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=20)
    cs = plt.imshow(np.transpose(voltages), norm=norm,
                    interpolation='none', aspect='auto', cmap=cm)
    cbar = fig.colorbar(cs, extend='both')
    cbar.set_label(label=r'$V_m$ $response$ $[mV]$', weight='bold')
    ax.axvline(500, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    ax.axvline(1500, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Sorted neuron #')
    ax.set_xlabel('Time [s]')
    ax.set_xticks(np.arange(0, 3000, 500))
    ax.set_xticklabels(np.arange(0, 3, 0.5))
    
    if classes:
        fig.patches.extend([plt.Rectangle((0.075, 0.613), 0.05, 0.267,
                                              fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                              transform=fig.transFigure, figure=fig)])
        fig.patches.extend([plt.Rectangle((0.075, 0.113), 0.05, 0.24,
                                              fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                              transform=fig.transFigure, figure=fig)])
        fig.text(0.05, 0.82, 'hVf', rotation='vertical', color='#F06233', visible=True)
        fig.text(0.05, 0.13, 'dVf', rotation='vertical', color='#33ABA2', visible=True)
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, 'heatmap_voltage_response.png'), dpi=300, transparent=True)
        plt.close(fig)

        # Second figure
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(
            6.4, 12), sharex=True, constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0, h_pad=0.5, hspace=-0.13, wspace=0)
        
        cs = axs[0].imshow(np.transpose(voltages), norm=norm,
                        interpolation='none', aspect='auto', cmap=cm)
        cbar = fig.colorbar(cs, extend='both', ax=axs[0], location='right')
        cbar.set_label(label=r'$V_m$ $response$ $[mV]$', weight='bold', fontsize=18)
        
        axs[0].axvline(500, linestyle='dashed', color='k', linewidth=1.5, alpha=1)
        axs[0].axvline(1500, linestyle='dashed', color='k', linewidth=1.5, alpha=1)
        axs[0].set_ylabel('Sorted neuron #', fontsize=16)
        fig.patches.extend([plt.Rectangle((0.06, 0.81), 0.046, 0.147,
                                              fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                              transform=fig.transFigure, figure=fig)])
        fig.patches.extend([plt.Rectangle((0.06, 0.535), 0.046, 0.138,
                                              fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                              transform=fig.transFigure, figure=fig)])
        
        fig.text(0.025, 0.93, 'hVf', rotation='vertical',
                  color='#F06233', visible=True, fontsize=14)
        fig.text(0.025, 0.54, 'dVf', rotation='vertical',
                  color='#33ABA2', visible=True, fontsize=14)
            
        axs[0].tick_params(axis='both', which='major', labelsize=14)
        
        times = np.arange(0, 2500, 1)
        for neurons_subset_label, subset_info in neuron_populations.items():
            neuron_population = subset_info['voltage_changes']
            neuron_population = np.mean(neuron_population, axis=0) #average over realizations
            mean_neuron_population = np.mean(neuron_population, axis=1) #average over neurons
            sem_neuron_population = (np.std(neuron_population, ddof=1, axis=1) /
                            np.sqrt(neuron_population.shape[1]))
            axs[1].plot(times, mean_neuron_population, color=subset_info['color'], 
                        label=neurons_subset_label)
            axs[1].fill_between(times, mean_neuron_population + sem_neuron_population,
                            mean_neuron_population - sem_neuron_population, alpha=0.3, color=subset_info['color'])
        
        axs[1].hlines(0, 0, simulation_length, linestyles='dotted', color='k', linewidths=1, zorder=1000)
        axs[1].vlines(500, -10, 10, linestyle='dashed',
                      color='k', linewidth=1.5, alpha=1, zorder=1000)
        axs[1].vlines(1500, -10, 10, linestyle='dashed',
                      color='k', linewidth=1.5, alpha=1, zorder=1000)
        
        axs[1].set_ylim(-7.5, 7.5)
        axs[1].set_ylabel('Membrane potential [mV]', fontsize=16)
        axs[1].set_xlabel('Time [s]', fontsize=16)
        
        axs[1].set_xticks(np.arange(0, 3000, 500))
        axs[1].set_xticklabels(np.arange(0, 3, 0.5))
        axs[1].tick_params(axis='both', labelsize=16)
        
        leg = axs[1].legend(fontsize=16)
        for text, color in zip(leg.get_texts(), ['#F06233', '#33ABA2', '#9CB0AE']):
            plt.setp(text, color=color)
        
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, 'keller_figure.png'), dpi=300, transparent=True)

    else:
        os.makedirs(path, exist_ok=True)
        fig.savefig(os.path.join(path, 'heatmap_voltage_response.png'), dpi=300, transparent=True)
        plt.close(fig)
        
def currents_heatmap(current_variable, variable_label, selected_df, classes=True, path=''):
    selected_df = selected_df.sort_values(by=['heatmap_neurons_order'])
    sorted_indices = selected_df.loc[selected_df['heatmap_neurons_order']<30, 'heatmap_neurons_order'].index
    
    neurons_sample = current_variable[:, :, sorted_indices]
    neurons_sample = np.mean(neurons_sample, axis=0)
                
    fig = plt.figure()
    ax = plt.subplot(111)
    
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    cs = plt.imshow(np.transpose(neurons_sample), norm=norm,
                    interpolation='none', aspect='auto', cmap=cm)
    cbar = fig.colorbar(cs, extend='both')
    cbar.set_label(label=r'Input current response $[pA]$', weight='bold')
    ax.axvline(500, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    ax.axvline(1500, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Sorted neuron #')
    ax.set_xlabel('Time [s]')
    ax.set_xticks(np.arange(0, 3000, 500))
    ax.set_xticklabels(np.arange(0, 3, 0.5))
    
    if classes:
        pos1 = ax.get_position() # get the original position 
        sample_row_height = pos1.height/30  
        selection = selected_df.loc[selected_df['heatmap_neurons_order']<30, 'neuron class']
        n_dvf = list(selection.values).count('dVf')
        n_hvf = list(selection.values).count('hVf')
    
        fig.patches.extend([plt.Rectangle((pos1.x0-0.05, pos1.y1-n_hvf*sample_row_height), 0.05, n_hvf*sample_row_height,
                                          fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])
        fig.patches.extend([plt.Rectangle((pos1.x0-0.05, pos1.y0), 0.05, n_dvf*sample_row_height,
                                          fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                          transform=fig.transFigure, figure=fig)])
        fig.text(0.05, pos1.y1-2*sample_row_height, 'hVf', rotation='vertical', color='#F06233', visible=True)
        fig.text(0.05, pos1.y0, 'dVf', rotation='vertical', color='#33ABA2', visible=True)
            
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'heatmap_{variable_label}.png'), dpi=300, transparent=True)
    plt.close(fig)
 
           
def currents_figure(neuron_populations, input_current_variations, recurrent_current_variations, bottom_up_current_variations, selected_df, input_current_threshold=0.2, path=''):
    fig, grid = plt.subplots(nrows=2, ncols=4, figsize=(12,5),
                   gridspec_kw={'width_ratios':(1,1,1,0.05), 'height_ratios':(0.6,0.4)}, sharex='col')
    heatmaps_ax = grid[0]
    averages_ax = grid[1]
    averages_ax[3].set_visible(False)
    heatmaps_ax[0].sharey(heatmaps_ax[1])
    heatmaps_ax[1].sharey(heatmaps_ax[2])
    averages_ax[0].sharey(averages_ax[1])
    averages_ax[1].sharey(averages_ax[2])
    
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    current_variables = [recurrent_current_variations, bottom_up_current_variations, input_current_variations]
    selected_df = selected_df.sort_values(by=['heatmap_neurons_order'])
    sorted_indices = selected_df.loc[selected_df['heatmap_neurons_order']<30, 'heatmap_neurons_order'].index
    stimuli_init_time = 500
    stimuli_end_time = stimuli_init_time + 1000
    for idx, current_variable in enumerate(current_variables):
        neurons_sample = current_variable[:, :, sorted_indices]
        neurons_sample = np.mean(neurons_sample, axis=0)
        cs = heatmaps_ax[idx].imshow(np.transpose(neurons_sample), norm=norm,
                             interpolation='none', aspect='auto', cmap=cm)
        heatmaps_ax[idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
        heatmaps_ax[idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
        heatmaps_ax[idx].tick_params(axis='both', bottom=False, labelbottom=False, labelsize=12)
        if idx != 0:
            heatmaps_ax[idx].set_yticklabels([])
            heatmaps_ax[idx].set_yticks([])
    heatmaps_ax[0].set_ylabel('Sorted neuron #', fontsize=12)
    cbar = fig.colorbar(cs, cax=heatmaps_ax[3], extend='both')
    cbar.set_label(label='Input current [pA]', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels([-2, -1, 0, 1, 2])

    neuron_classes = ['hVf', 'dVf', 'unc']
    neuron_variables = ['recurrent_current_changes', 'bottom_up_current_changes', 'input_current_changes']    
    for idx in range(3):
        for index, (neurons_subset_label, subset_info) in enumerate(neuron_populations.items()):
            neuron_variable = neuron_variables[idx]
            n_simulations, simulation_length, n_neurons = subset_info[neuron_variable].shape
            mean_neurons_subset = np.mean(subset_info[neuron_variable], axis=(0, 2))
            sem_neurons_subset = (np.std(subset_info[neuron_variable], ddof=1, axis=(0, 2)) /
                            np.sqrt(n_neurons))
            times = np.linspace(0, simulation_length, simulation_length)   
            averages_ax[idx].plot(times, mean_neurons_subset, color=subset_info['color'],
                    label=neuron_classes[index], linewidth=1)
            averages_ax[idx].fill_between(times, 
                              mean_neurons_subset 
                              + sem_neurons_subset, 
                              mean_neurons_subset 
                              - sem_neurons_subset, 
                              alpha=0.3, color=subset_info['color'])
        averages_ax[idx].tick_params(axis='both', labelsize=12)
        averages_ax[idx].set_xticks(np.arange(0, 2500, 500))
        averages_ax[idx].set_xticklabels(np.arange(0, 2.5, 0.5))
        averages_ax[idx].axhline(0, color='k', linewidth=1)
        averages_ax[idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1, zorder=10)
        averages_ax[idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1, zorder=10)        
        averages_ax[idx].set_xlabel('Time [s]', fontsize=12)
        # averages_ax[idx].set_ylim(-1, 1)
        averages_ax[idx].set_xlim(0, 2500)
        if idx != 0:
            averages_ax[idx].set_yticklabels([])
            averages_ax[idx].set_yticks([])
    averages_ax[0].set_ylabel('Input current [pA]', fontsize=12)
    threshold = averages_ax[2].hlines(input_current_threshold, 0, (simulation_length/1000), 
                              linestyles='dotted', color='k', linewidths=1,zorder=10, label='Classification threshold')
    averages_ax[2].hlines(-input_current_threshold, 0, (simulation_length/1000), 
                  linestyles='dotted', color='k', linewidths=1,zorder=10)
    
    averages_ax[1].get_shared_y_axes().join(averages_ax[1], averages_ax[0])
    averages_ax[2].get_shared_y_axes().join(averages_ax[2], averages_ax[0])

    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    pos1 = heatmaps_ax[0].get_position() # get the original position 
    sample_row_height = pos1.height/30  
    selection = selected_df.loc[selected_df['heatmap_neurons_order']<30, 'neuron class']
    n_dvf = list(selection.values).count('dVf')
    n_hvf = list(selection.values).count('hVf')
        
    fig.patches.extend([plt.Rectangle((pos1.x0-0.025, pos1.y1-n_hvf*sample_row_height), 0.025, n_hvf*sample_row_height,
                                      fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.patches.extend([plt.Rectangle((pos1.x0-0.025, pos1.y0), 0.025, n_dvf*sample_row_height,
                                      fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.text(0.085, pos1.y1-2*sample_row_height, 'hVf', rotation='vertical', color='#F06233', visible=True, fontsize=10, weight='bold')
    fig.text(0.085, pos1.y0, 'dVf', rotation='vertical', color='#33ABA2', visible=True, fontsize=10, weight='bold')
    
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'figure_heatmap.png'), dpi=300, transparent=True)
    plt.close(fig)
    
    
def currents_figure_with_stimulus(current_variable, variable_label, selected_df, frequency, reverse=False, path=''):
    selected_df = selected_df.sort_values(by=['heatmap_neurons_order'])
    sorted_indices = selected_df.loc[selected_df['heatmap_neurons_order']<30, 'heatmap_neurons_order'].index
    neurons_sample = current_variable[:, :, sorted_indices]
    neurons_sample = np.mean(neurons_sample, axis=0)
    
    fig, grid = plt.subplots(nrows=2, ncols=2, figsize=(6,4),
                   gridspec_kw={'width_ratios':(1, 0.05), 'height_ratios':(0.85,0.15)}, sharex='col')
    heatmaps_ax, cax0 = grid[0]
    drifting_grating_ax, cax1 = grid[1]   
    cax1.set_visible(False)
    heatmaps_ax.clear()
    drifting_grating_ax.clear()
    
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    stimuli_init_time = 500
    stimuli_end_time = stimuli_init_time + 1000
    cs = heatmaps_ax.imshow(np.transpose(neurons_sample), norm=norm,
                         interpolation='none', aspect='auto', cmap=cm)
    heatmaps_ax.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    heatmaps_ax.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    heatmaps_ax.tick_params(axis='both', bottom=False, labelbottom=False, labelsize=10)
    heatmaps_ax.set_ylabel('Sorted neuron #', fontsize=10)
    cbar = fig.colorbar(cs, cax=cax0, extend='both')
    cbar.set_label(label='Input current [pA]', fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels([-2, -1, 0, 1, 2])

    time_length = neurons_sample.shape[0]
    drifting_grating_plot = DriftingGrating(frequency=frequency, drifting_init_time=stimuli_init_time, 
                                            drifting_end_time=stimuli_end_time, reverse=reverse)
    drifting_grating_plot(drifting_grating_ax, time_length)
    
    plt.subplots_adjust(wspace=0.03, hspace=0.07, left=0.19)
    drifting_grating_ax.set_xlim(heatmaps_ax.get_xlim())
    
    pos1 = heatmaps_ax.get_position() # get the original position 
    sample_row_height = pos1.height/30  
    selection = selected_df.loc[selected_df['heatmap_neurons_order']<30, 'neuron class']
    n_dvf = list(selection.values).count('dVf')
    n_hvf = list(selection.values).count('hVf')

    fig.patches.extend([plt.Rectangle((pos1.x0-0.045, pos1.y1-n_hvf*sample_row_height), 0.045, n_hvf*sample_row_height,
                                      fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.patches.extend([plt.Rectangle((pos1.x0-0.045, pos1.y0), 0.045, n_dvf*sample_row_height,
                                      fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.text(0.12, pos1.y1-2*sample_row_height, 'hVf', rotation='vertical', color='#F06233', visible=True)
    fig.text(0.12, pos1.y0, 'dVf', rotation='vertical', color='#33ABA2', visible=True)
  
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'heatmap_{variable_label}_stimulus.png'), dpi=300, transparent=True)
    plt.close(fig)


def firing_rate_figure(firing_rates, sampling_interval, frequency, hVf_mask, dVf_mask, other_firing_rates=None,
                       drifting_init_time=500, drifting_end_time=1500, reverse=False, simulation_length=2500, path=''):
    fig, grid = plt.subplots(nrows=3, ncols=2, figsize=(4,8),
                   gridspec_kw={'width_ratios':(1,0.03), 'height_ratios':(1,0.5,0.1)}, sharex='col')
    firing_rates_ax, cax1 = grid[0]
    firing_rates_classes_ax, cax2 = grid[1]
    drifting_grating_ax, cax3 = grid[2]
    cax2.set_visible(False)
    cax3.set_visible(False)
    firing_rates_ax.clear()
    firing_rates_classes_ax.clear()
    drifting_grating_ax.clear()
    
    smoothed_firing_rates = np.mean(firing_rates, axis=0)
    firing_rates_non_zero = np.where(np.std(smoothed_firing_rates, axis=0)!=0)[0]
    normalised_smoothed_firing_rates = smoothed_firing_rates/np.max(smoothed_firing_rates, axis=0)
    firing_rates = normalised_smoothed_firing_rates[:, firing_rates_non_zero]
    mean_variations = np.mean(firing_rates[int(500/sampling_interval): int(1500/sampling_interval), :], axis=0)
    sorted_indices = np.arange(firing_rates.shape[1])
    sorted_indices = [x for _, x in sorted(zip(mean_variations, sorted_indices), key=lambda element: (element[0]), reverse=True)]   
    firing_rates_ordered = firing_rates[:,sorted_indices]
            
    stimuli_init_time = 500
    stimuli_end_time = stimuli_init_time + 1000
    time_length = firing_rates.shape[0]
    cs = firing_rates_ax.imshow(np.transpose(firing_rates_ordered),
                                interpolation='none', aspect='auto', cmap='coolwarm')
    cbar = fig.colorbar(cs, cax=cax1, extend='neither')
    # cbar.set_label(label=r'Normalized firing rate (NFR)', fontsize=10)
    cbar.ax.set_title(label='NFR', fontsize=12)

    cbar.ax.tick_params(labelsize=10)
    firing_rates_ax.axvline(int(stimuli_init_time/sampling_interval), linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    firing_rates_ax.axvline(int(stimuli_end_time/sampling_interval), linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
    firing_rates_ax.set_ylabel('Sorted neuron #', fontsize=12)
    firing_rates_ax.tick_params(axis='both',          # changes apply to the x-axis
                                which='both',      # both major and minor ticks are affected
                                bottom=False,      # ticks along the bottom edge are off
                                top=False,         # ticks along the top edge are off
                                labelbottom=False,
                                labelsize=10) 

    times = np.linspace(0, time_length, time_length)
    
    if other_firing_rates is not None:
        other_firing_rates = np.mean(other_firing_rates, axis=0)
        other_firing_rates_non_zero = np.where(np.std(other_firing_rates, axis=0)!=0)[0]
        normalised_other_firing_rates = other_firing_rates/np.max(other_firing_rates, axis=0)
        firing_rates_classes_ax.plot(times, np.mean(normalised_other_firing_rates[:, other_firing_rates_non_zero], axis=1), 'g')

    firing_rates_classes_ax.plot(times, np.mean(firing_rates[:, hVf_mask[firing_rates_non_zero]], axis=1), '#F06233')
    firing_rates_classes_ax.plot(times, np.mean(firing_rates[:, dVf_mask[firing_rates_non_zero]], axis=1), '#33ABA2')
    
    # metadata_path = os.path.join(path, 'neurons_plotted_per_class.txt')
    # with open(metadata_path, 'w') as out_file:
    #     out_file.write(f'hVf:{firing_rates[:, hVf_mask[firing_rates_non_zero]].shape[1]} - {np.sum(hVf_mask)}\n')
    #     out_file.write(f'dVf:{firing_rates[:, dVf_mask[firing_rates_non_zero]].shape[1]} - {np.sum(dVf_mask)}\n')

    firing_rates_classes_ax.axvline(int(stimuli_init_time/sampling_interval), linestyle='dashed', color='k', linewidth=1.5, alpha=0.8, zorder=10)
    firing_rates_classes_ax.axvline(int(stimuli_end_time/sampling_interval), linestyle='dashed', color='k', linewidth=1.5, alpha=0.8, zorder=10)
    firing_rates_classes_ax.set_ylabel('NFR', fontsize=12)
    firing_rates_classes_ax.tick_params(axis='both',          # changes apply to the x-axis
                                        which='both',      # both major and minor ticks are affected
                                        bottom=False,      # ticks along the bottom edge are off
                                        top=False,         # ticks along the top edge are off
                                        labelbottom=False,
                                        labelsize=10) 
    firing_rates_classes_ax.set_ylim(0, 1)
    
    drifting_grating_plot = DriftingGrating(frequency=frequency, drifting_init_time=int(drifting_init_time/sampling_interval), 
                                            drifting_end_time=int(drifting_end_time/sampling_interval), reverse=reverse)
    drifting_grating_plot(drifting_grating_ax, time_length)
    
    plt.subplots_adjust(wspace=0.03, hspace=0.07, left=0.19)
    #fig.canvas.draw()
    drifting_grating_ax.set_xlim(firing_rates_ax.get_xlim())
    
    os.makedirs(path, exist_ok=True)
    # fig.tight_layout()
    fig.savefig(os.path.join(path, 'firing_rates_figure.png'), dpi=300, transparent=True)#
    

def currents_comparison_figure(current_variables, frequencies, reverses, selected_df, input_current_threshold=0.2, path=''):
    n_plots = len(current_variables)
    fig, grid = plt.subplots(nrows=2, ncols=n_plots+1, figsize=(3*n_plots,4),
                   gridspec_kw={'width_ratios':(1,)*n_plots + (0.05,), 'height_ratios':(0.85,0.15)}, sharex='col')
    heatmaps_ax = grid[0]
    drifting_grating_ax = grid[1]
    drifting_grating_ax[-1].set_visible(False)
    # for idx in range(n_plots-1):
    #     heatmaps_ax[idx].sharey(heatmaps_ax[idx+1])
    # heatmaps_ax[0].sharey(heatmaps_ax[1])
    
    selected_df = selected_df.sort_values(by=['heatmap_neurons_order'])
    sorted_indices = selected_df.loc[selected_df['heatmap_neurons_order']<30, 'heatmap_neurons_order'].index
    colors = ['#045195', '#1D85C4', '#FFFFFF', '#EE2C76', '#82142D']
    n_bin = 100  # Discretizes the interpolation into bins
    cm = mpl.colors.LinearSegmentedColormap.from_list('keller_heatmap', colors, N=n_bin)
    norm = mpl.colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
    stimuli_init_time = 500
    stimuli_end_time = stimuli_init_time + 1000
    for idx, current_variable in enumerate(current_variables):
        neurons_sample = current_variable[:, :, sorted_indices]
        neurons_sample = np.mean(neurons_sample, axis=0)
        time_length = neurons_sample.shape[0]
        cs = heatmaps_ax[idx].imshow(np.transpose(neurons_sample), norm=norm,
                             interpolation='none', aspect='auto', cmap=cm)
        heatmaps_ax[idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
        heatmaps_ax[idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5, alpha=0.8)
        heatmaps_ax[idx].tick_params(axis='both', labelsize=8)            
        drifting_grating_plot = DriftingGrating(frequency=frequencies[idx], drifting_init_time=stimuli_init_time, 
                                                drifting_end_time=stimuli_end_time, reverse=reverses[idx])
        drifting_grating_plot(drifting_grating_ax[idx], time_length)
        drifting_grating_ax[idx].set_xlim(heatmaps_ax[idx].get_xlim())
    
        if idx != 0:
            y_axis = drifting_grating_ax[idx].axes.get_yaxis()
            y_label = y_axis.get_label()
            y_label.set_visible(False)
    
    for idx in range(1, n_plots):
        heatmaps_ax[idx].set_yticks([])
        heatmaps_ax[idx].set_yticklabels([])
        
    heatmaps_ax[0].set_ylabel('Sorted neuron #', fontsize=10)
    cbar = fig.colorbar(cs, cax=heatmaps_ax[-1], extend='both')
    cbar.set_label(label='Input current [pA]', fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks([-2, -1, 0, 1, 2])
    cbar.set_ticklabels([-2, -1, 0, 1, 2])
    plt.subplots_adjust(wspace=0.13, hspace=0.07)
    pos1 = heatmaps_ax[0].get_position() # get the original position 
    sample_row_height = pos1.height/30  
    selection = selected_df.loc[selected_df['heatmap_neurons_order']<30, 'neuron class']
    n_dvf = list(selection.values).count('dVf')
    n_hvf = list(selection.values).count('hVf')
        
    fig.patches.extend([plt.Rectangle((pos1.x0-0.025, pos1.y1-n_hvf*sample_row_height), 0.025, n_hvf*sample_row_height,
                                      fill=True, color='#F06233', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.patches.extend([plt.Rectangle((pos1.x0-0.025, pos1.y0), 0.025, n_dvf*sample_row_height,
                                      fill=True, color='#33ABA2', alpha=0.3, zorder=1000,
                                      transform=fig.transFigure, figure=fig)])
    fig.text(0.085, pos1.y1-2*sample_row_height, 'hVf', rotation='vertical', color='#F06233', visible=True, fontsize=10, weight='bold')
    fig.text(0.085, pos1.y0, 'dVf', rotation='vertical', color='#33ABA2', visible=True, fontsize=10, weight='bold')
        
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'figure_comparison.png'), dpi=300, transparent=True)
    plt.close(fig)
    
def matrices_comparison_figure(all_confussion_matrices, orientation, orientations, frequency, frequencies, path=''):
    n_plots = len(orientations)
    fig, grid = plt.subplots(nrows=1, ncols=n_plots+1, figsize=(3*n_plots, 3),
                   gridspec_kw={'width_ratios':(1,)*n_plots+ (0.05,)})
    matrices_axs = grid[:-1]
    cax = grid[-1]
    for idx, confussion_matrix, orientation2, frequency2 in zip(np.arange(n_plots), all_confussion_matrices, orientations, frequencies):
        sns.heatmap(confussion_matrix, vmin=0, vmax=1, annot=True, annot_kws={"size": 10}, fmt=".2f", yticklabels=True, cmap='binary', cbar_ax=cax, ax=matrices_axs[idx])
        matrices_axs[idx].set_xlabel(f'Orientation {orientation2} - frequency {frequency2}')
    matrices_axs[0].set_ylabel(f'Orientation {orientation} - frequency {frequency}')
    for idx in range(1, n_plots):
        matrices_axs[idx].set_yticks([])
        matrices_axs[idx].set_yticklabels([])
        
    plt.subplots_adjust(wspace=0.13, hspace=0.07, bottom=0.15)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'figure_matrices_comparison.png'), dpi=300, transparent=True)
    plt.close(fig)

def degree_distribution_plot(network_info, weighted_network_info, filename='degree_distribution.png', path=''):
    fig, axs = plt.subplots(2,2, sharex='col', sharey='col')
    for i, network_type, network_name in zip(range(2), [network_info, weighted_network_info], ['Non-weighted', 'Weighted']):
        max_degree = int(max([network_type['Total k_in'].max(), network_type['k_out'].max()]))
        min_degree = int(min([network_type['Total k_in'].min(), network_type['k_out'].min()]))
        binwidth = int((max_degree+abs(min_degree))/100)
        axs[0][i].hist(network_type['k_in'], bins=range(min_degree, max_degree + binwidth, binwidth), label='Recurrent $k_{in}$', color='b', alpha=0.6, zorder=1) 
        axs[0][i].hist(network_type['Total k_in'], bins=range(min_degree, max_degree + binwidth, binwidth), label='Total $k_{in}$', color='g', alpha=0.6, zorder=1) 
        axs[1][i].hist(network_type['k_out'], bins=range(min_degree, max_degree + binwidth, binwidth), label='Recurrent $k_{out}$', color='r', alpha=0.6)
        axs[0][i].set_xlim(min_degree, max_degree)
        axs[0][i].legend() 
        axs[0][i].set_title(network_name)
        axs[i][0].set_ylabel('# neurons')
        axs[1][i].set_xlabel('Degree')
        axs[1][i].set_xlim(min_degree, max_degree)
        axs[1][i].legend()
        
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, filename), dpi=300, transparent=True)
        
    
def synapses_histogram(counts, y_label='', path=''):
    fig = plt.figure(figsize=(3, 5))
    key_record = {}
    colors_exc = 'r'
    colors_inh = {'Htr3a':'b', 'Pvalb':'g', 'Sst':'y'}
    inds = [x for _, x in sorted(zip(counts.keys(), range(len(counts.keys()))), key=lambda element: (element[0][1], element[0][0]))]   
    keys = counts.keys()[inds]
    counts = counts[inds]
    for key, count in zip(keys, counts):
        if key[:2] not in key_record.keys():
            key_record[key[:2]] = {}
            key_record[key[:2]]['record'] = count
            key_record[key[:2]]['index'] = 0
            if key[:2] == 'e2':
                plt.bar(key[:2]+'3', count, width=0.8, align='center', color=colors_exc, label=key)
            elif key[:2] == 'i2':
                plt.bar(key[:2]+'3', count, width=0.8, align='center', color=colors_inh[key[3:]], label=key)
            elif key[1]!=2 and key[0]=='e':
                plt.bar(key[:2], count, width=0.8, align='center', color=colors_exc, label=key)
            else:
                plt.bar(key[:2], count, width=0.8, align='center', color=colors_inh[key[2:]], label=key)
        else:
            key_record[key[:2]]['index'] += 1
            if key[:2] == 'e2':
                plt.bar(key[:2]+'3', count, width=0.8, bottom=key_record[key[:2]]['record'], align='center', color=colors_exc, label=key)
            elif key[:2] == 'i2':
                plt.bar(key[:2]+'3', count, width=0.8, bottom=key_record[key[:2]]['record'], align='center', color=colors_inh[key[3:]], label=key)        
            elif key[1]!=2 and key[0]=='e':
                plt.bar(key[:2], count, width=0.8, bottom=key_record[key[:2]]['record'], align='center', color=colors_exc, label=key)
            else:
                plt.bar(key[:2], count, width=0.8, bottom=key_record[key[:2]]['record'], align='center', color=colors_inh[key[2:]], label=key)
            
            key_record[key[:2]]['record'] += count
            
    exc_patch = mpatches.Patch(color='r', label='Exc')
    Htr3a_patch = mpatches.Patch(color='b', label='Htr3a')
    Pvalb_patch = mpatches.Patch(color='g', label='Pvalb')
    Sst_patch = mpatches.Patch(color='y', label='Sst')
                
    plt.tick_params(axis='x', labelrotation=90)
    plt.ylabel(y_label)
    plt.legend(handles=[exc_patch, Htr3a_patch, Pvalb_patch, Sst_patch])
    plt.tick_params(axis='x', labelrotation=90)
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f'{y_label}.png'), dpi=300, transparent=True)
    
def degree_distributions_per_class_plot(e23_df, n_bins=50, path=''):   
    max_degree = int(max([e23_df['Total k_in'].max(), e23_df['k_out'].max()]))
    min_degree = int(min([e23_df['Total k_in'].min(), e23_df['k_out'].min()]))
    binwidth = int((max_degree+abs(min_degree))/n_bins)
    max_weight = int(max([e23_df['Weighted Total k_in'].max(), e23_df['Weighted k_out'].max()]))
    min_weight = int(min([e23_df['Weighted Total k_in'].min(), e23_df['Weighted k_out'].min()]))
    binwidth_weight = int((max_weight+abs(min_weight))/n_bins)
    for neuron_class in ['hVf', 'dVf', 'unclassified']:
        fig, axs = plt.subplots(2,2, sharex='col', sharey='col')
        reduced_e23_df = e23_df[e23_df['neuron class']==neuron_class]   
        axs[0][0].hist(reduced_e23_df['k_in'], density=True, bins=range(0, max_degree + binwidth, binwidth), label='Recurrent $k_{in}$', color='b', alpha=0.6) 
        axs[0][0].hist(reduced_e23_df['Total k_in'], density=True, bins=range(0, max_degree + binwidth, binwidth), label='Total $k_{in}$', color='g', alpha=0.6) 
        axs[1][0].hist(reduced_e23_df['k_out'], density=True, bins=range(0, max_degree + binwidth, binwidth), label='Recurrent $k_{out}$', color='r', alpha=0.6)
        axs[0][0].set_xlim(0, max_degree)
        axs[0][0].legend() 
        axs[0][0].set_title('Non-weighted')
        axs[0][0].set_ylabel('Probability density')
        axs[1][0].set_ylabel('Probability density')
        axs[1][0].set_xlabel('Degree')
        axs[1][0].legend()
        axs[0][1].hist(reduced_e23_df['Weighted k_in'], density=True, bins=range(min_weight, max_weight + binwidth_weight, binwidth_weight), label='Recurrent $k_{in}$', color='b', alpha=0.6) 
        axs[0][1].hist(reduced_e23_df['Weighted Total k_in'], density=True, bins=range(min_weight, max_weight + binwidth_weight, binwidth_weight), label='Total $k_{in}$', color='g', alpha=0.6) 
        axs[1][1].hist(reduced_e23_df['Weighted k_out'], density=True, bins=range(min_weight, max_weight + binwidth_weight, binwidth_weight), label='Recurrent $k_{out}$', color='r', alpha=0.6)
        axs[0][1].set_xlim(min_weight, max_weight)
        axs[0][1].legend() 
        axs[0][1].set_title('Weighted')
        axs[1][1].set_xlabel('Weight')
        axs[1][1].legend()
        
        fig.tight_layout()
        class_path = os.path.join(path, neuron_class)
        os.makedirs(class_path, exist_ok=True)
        fig.savefig(os.path.join(class_path, 'degrees.png'), dpi=300, transparent=True)
    
def degree_distributions_classes_comparison_plot(first_df, second_df, degree_key, weighted_degree_key, path=''):         
    fig, axs = plt.subplots(1,2, figsize=(8, 4), sharex='col')
    axs[0].hist(first_df[degree_key], bins=50, label='dVf', color='#33ABA2', alpha=0.3, histtype='stepfilled') 
    axs[0].hist(second_df[degree_key], bins=50, label='hVf', color='#F06233', alpha=0.3, histtype='stepfilled') 
    axs[1].hist(first_df[weighted_degree_key], bins=50, label='dVf', color='#33ABA2', alpha=0.3, histtype='stepfilled') 
    axs[1].hist(second_df[weighted_degree_key], bins=50, label='hVf', color='#F06233', alpha=0.3, histtype='stepfilled') 
    # axs[0].set_xlim(0, self.max_degree)
    # axs[0].legend() 
    # axs[0].set_title(degree_key)
    # axs[1].set_title(weighted_degree_key)
    axs[0].set_ylabel('# neurons', fontsize=14)
    axs[1].set_ylabel('# neurons', fontsize=14)
    axs[0].set_xlabel('Degree', fontsize=14)
    axs[1].set_xlabel('Weight', fontsize=14)
    axs[0].tick_params(axis='both', labelsize=14)
    axs[1].tick_params(axis='both', labelsize=14)
    # axs[1].legend()
    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'hVf_vs_dVf_degree_distribution_'+degree_key+'.png'), dpi=300, transparent=True)
         

def preferred_angle_distribution(hVf_angles, dVf_angles, unc_angles, path=''):
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(9, 3), constrained_layout=True)
    neurons_color = ['#33ABA2', '#9CB0AE', '#F06233']
    variables = [dVf_angles, unc_angles, hVf_angles]
    for idx in range(3):
        axs[idx].hist(variables[idx], color=neurons_color[idx], density=True, bins=40)
        axs[idx].tick_params(axis='both', labelsize=12)
        axs[idx].set_xlabel(u'Orientation [\N{DEGREE SIGN}]', fontsize=12)
    axs[0].set_ylabel('# neurons', fontsize=12)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'preferred_angle_distribution.png'), dpi=300, transparent=True)
    
    
def mean_perturbation_responses_figure( currents_df, path='', directory=''):
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    orientations = np.array([0, 45, 90, 135])
    for idx, orientation in enumerate(orientations):
        reverse_dir = orientation + 180
        sel_or_mean = currents_df[str(orientation)]
        sel_or_sem = currents_df[str(orientation)+'_sem']
        sel_rev_dir_mean = currents_df[str(reverse_dir)]
        sel_rev_dir_sem = currents_df[str(reverse_dir)+'_sem']
    
        times = np.arange(0, 2500)
        if idx==0:
            ax = axs[0, 0]
        elif idx==1:
            ax = axs[0, 1]
        elif idx==2:
            ax = axs[1, 0]
        elif idx==3:
            ax = axs[1, 1]
            
        ax.plot(times, sel_or_mean, color='r', ms=1,
                     alpha=0.7, label=orientation)
        ax.fill_between(times, 
                              sel_or_mean + sel_or_sem, 
                              sel_or_mean - sel_or_sem, 
                              alpha=0.3, color='r')
        ax.plot(times, sel_rev_dir_mean, color='orange', ms=1,
                     alpha=0.7, label=reverse_dir)
        ax.fill_between(times, 
                              sel_rev_dir_mean + sel_rev_dir_sem, 
                              sel_rev_dir_mean - sel_rev_dir_sem, 
                              alpha=0.3, color='gray')
        ax.legend()
        stimuli_init_time = 500
        stimuli_end_time = stimuli_init_time + 1000
        ax.axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5)
        ax.axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5)
        for ax in axs.flat:
            ax.set(xlabel='Time [ms]', ylabel='Total input current [pA]')
            ax.label_outer()
          
    path = os.path.join(path, directory)
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, 'fig2A.png'), dpi=300, transparent=True)
    plt.close(fig)   
    
def mean_perturbation_responses_composite_figure(currents_dvf, currents_hvf, path=''):
    fig, axs = plt.subplots(2, 4, figsize=(12,5), sharex=True, sharey='row')
    orientations = np.array([0, 45, 90, 135])
    for idx, orientation in enumerate(orientations):
        reverse_dir = orientation + 180
        sel_or_mean_dvf = currents_dvf[str(orientation)]
        sel_or_sem_dvf = currents_dvf[str(orientation)+'_sem']
        sel_rev_dir_mean_dvf = currents_dvf[str(reverse_dir)]
        sel_rev_dir_sem_dvf = currents_dvf[str(reverse_dir)+'_sem']
        
        sel_or_mean_hvf = currents_hvf[str(orientation)]
        sel_or_sem_hvf = currents_hvf[str(orientation)+'_sem']
        sel_rev_dir_mean_hvf = currents_hvf[str(reverse_dir)]
        sel_rev_dir_sem_hvf = currents_hvf[str(reverse_dir)+'_sem']
    
        times = np.arange(0, 2500)
                        
        axs[1, idx].plot(times, sel_or_mean_hvf, color='#F06233', ms=1,
                         alpha=0.7, label=u'{orientation}\N{DEGREE SIGN}'.format(orientation=orientation))
        axs[1, idx].fill_between(times, 
                              sel_or_mean_hvf + sel_or_sem_hvf, 
                              sel_or_mean_hvf - sel_or_sem_hvf, 
                              alpha=0.3, color='#F06233')
        axs[1, idx].plot(times, sel_rev_dir_mean_hvf, color='#F06233', ms=1, ls='dashed',
                     alpha=0.7, label=u'{orientation}\N{DEGREE SIGN}'.format(orientation=reverse_dir))
        axs[1, idx].fill_between(times, 
                              sel_rev_dir_mean_hvf + sel_rev_dir_sem_hvf, 
                              sel_rev_dir_mean_hvf - sel_rev_dir_sem_hvf, 
                              alpha=0.3, color='#F06233')
        
        axs[0, idx].plot(times, sel_or_mean_dvf, color='#33ABA2', ms=1,
                         alpha=0.7, label=u'{orientation}\N{DEGREE SIGN}'.format(orientation=orientation))
        axs[0, idx].fill_between(times, 
                              sel_or_mean_dvf + sel_or_sem_dvf, 
                              sel_or_mean_dvf - sel_or_sem_dvf, 
                              alpha=0.3, color='#33ABA2')
        axs[0, idx].plot(times, sel_rev_dir_mean_dvf, color='#33ABA2', ms=1, ls='dashed',
                     alpha=0.7, label=u'{orientation}\N{DEGREE SIGN}'.format(orientation=reverse_dir))
        axs[0, idx].fill_between(times, 
                              sel_rev_dir_mean_dvf + sel_rev_dir_sem_dvf, 
                              sel_rev_dir_mean_dvf - sel_rev_dir_sem_dvf, 
                              alpha=0.3, color='#33ABA2')
        
        stimuli_init_time = 500
        stimuli_end_time = stimuli_init_time + 1000
        axs[0, idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5)
        axs[0, idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5)
        axs[1, idx].axvline(stimuli_init_time, linestyle='dashed', color='k', linewidth=1.5)
        axs[1, idx].axvline(stimuli_end_time, linestyle='dashed', color='k', linewidth=1.5)
        axs[0, idx].legend(loc='upper right')
        axs[1, idx].set_xlabel('Time [ms]')
        for ax in axs.flat:
            ax.set(xlabel='Time [ms]', ylabel='Total input current [pA]')
            ax.label_outer()
     
    axs[0, 0].set_ylabel('Total input current [pA]')
    axs[1, 0].set_ylabel('Total input current [pA]')
    plt.subplots_adjust(wspace=0.05, hspace=0.04)
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'fig2A_complete.png'), dpi=300, transparent=True)
    plt.close(fig)   
    
    
def plot_average_population_comparison(input_current_1, input_current_2, 
                                       pop1_name='pop1', pop2_name='pop2',
                                       path=''):
    n_simulations, simulation_length, n_neurons = input_current_1.shape
    fig = plt.figure(figsize=(6, 4))
    mean_1 = np.mean(input_current_1, axis=(0, 2))
    sem_1 = (np.std(input_current_1, ddof=1, axis=(0, 2)) /
                        np.sqrt(n_neurons))
    times = np.linspace(0, (simulation_length/1000), simulation_length)
    plt.plot(times, mean_1, color='#808BD0',
                label=pop1_name)
    plt.fill_between(times, 
                      mean_1 + sem_1, 
                      mean_1 - sem_1, 
                      alpha=0.3, color='#808BD0')
        
    mean_2 = np.mean(input_current_2, axis=(0, 2))
    sem_2 = (np.std(input_current_2, ddof=1, axis=(0, 2)) /
                        np.sqrt(n_neurons))
    plt.plot(times, mean_2, color='#92876B',
                label=pop2_name)
    plt.fill_between(times, 
                      mean_2 + sem_2, 
                      mean_2 - sem_2, 
                      alpha=0.3, color='#92876B')
    
    stimuli_init_time = 0.5
    stimuli_end_time = stimuli_init_time + 1
    plt.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    plt.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=1, zorder=10)
    plt.hlines(0, 0, (simulation_length/1000), color='k', linewidths=1,zorder=10)
    
    plt.ylabel('Input current response [pA]')
    plt.yticks(np.arange(-0.5, 2, 0.5))
    plt.ylim(-0.5, 1.5)
        
    plt.xlabel('Time [s]')
    plt.tick_params(axis='both', labelsize=10)
    leg = plt.legend()
    neurons_color = ['#808BD0', '#92876B']
    for text, color in zip(leg.get_texts(), neurons_color):
        plt.setp(text, color=color)
    leg.set_zorder(102)
    
    os.makedirs(path, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(path, f'input_current_comparison.png'), dpi=300, transparent=True)

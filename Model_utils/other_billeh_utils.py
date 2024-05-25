# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:43:44 2022

@author: javig
"""


import pandas as pd
import os
import sys
import numpy as np
import h5py
import time
from scipy.ndimage import gaussian_filter1d
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management


def pop_name_to_cell_type(pop_name):
    # Convert pop_name in the old format to cell types. E.g., 'e4Rorb' -> 'L4 Exc', 'i4Pvalb' -> 'L4 PV', 'i23Sst' -> 'L2/3 SST'
    shift = 0  # letter shift for L23
    layer = pop_name[1]
    if layer == "2":
        layer = "2/3"
        shift = 1
    elif layer == "1":
        return "L1 Htr3a"  # special case

    class_name = pop_name[2 + shift :]
    if class_name == "Pvalb":
        subclass = "PV"
    elif class_name == "Sst":
        subclass = "SST"
    elif (class_name == "Vip") or (class_name == "Htr3a"):
        subclass = "Htr3a"
    else:  # excitatory
        subclass = "Exc"

    return f"L{layer} {subclass}"

def get_layer_info(network):
    pop_name = pop_names(network)
    layer_query = ["e23", "e4", "e5", "e6"]
    layer_names = ["EXC_L23", "EXC_L4", "EXC_L5", "EXC_L6"]
    layer_info = {}
    for i in range(4):
        layer_info[layer_names[i]] = np.char.startswith(pop_name, layer_query[i])
    return layer_info


def pop_names(network, core_radius = None, n_selected_neurons=None, data_dir='GLIF_network'):
    path_to_csv = os.path.join(data_dir, 'network/v1_node_types.csv')
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')

    # Read data
    node_types = pd.read_csv(path_to_csv, sep=' ')
    with h5py.File(path_to_h5, mode='r') as node_h5:
        # Create mapping from node_type_id to pop_name
        node_types.set_index('node_type_id', inplace=True)
        node_type_id_to_pop_name = node_types['pop_name'].to_dict()

        # Map node_type_id to pop_name for all neurons and select population names of neurons in the present network 
        node_type_ids = node_h5['nodes']['v1']['node_type_id'][()][network['tf_id_to_bmtk_id']]
        true_pop_names = np.array([node_type_id_to_pop_name[nid] for nid in node_type_ids])

        if core_radius is not None:
            selected_mask = isolate_core_neurons(network, radius=core_radius, data_dir=data_dir)
        elif n_selected_neurons is not None:
            selected_mask = isolate_core_neurons(network, n_selected_neurons=n_selected_neurons, data_dir=data_dir)
        else:
            selected_mask = np.full(len(true_pop_names), True)
            
        true_pop_names = true_pop_names[selected_mask]

    return true_pop_names


def angle_tunning(network, data_dir='GLIF_network'):
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    node_h5 = h5py.File(path_to_h5, mode='r')
    angle_tunning = np.array(node_h5['nodes']['v1']['0']['tuning_angle'][:])[network['tf_id_to_bmtk_id']]

    return angle_tunning


def isolate_core_neurons(network, radius=None, n_selected_neurons=None, data_dir='GLIF_network'):
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    node_h5 = h5py.File(path_to_h5, mode='r')
    x = node_h5['nodes']['v1']['0']['x'][()][network['tf_id_to_bmtk_id']]
    z = node_h5['nodes']['v1']['0']['z'][()][network['tf_id_to_bmtk_id']]
    r = np.sqrt(x ** 2 + z ** 2)
    if radius is not None:
        selected_mask = r < radius
    # if a number of neurons is given, select the closest neurons
    elif n_selected_neurons is not None:
        selected_mask = np.argsort(r)[:n_selected_neurons]
        selected_mask = np.isin(np.arange(len(r)), selected_mask)
    
    return selected_mask


def isolate_neurons(network, neuron_population='e23', data_dir='GLIF_network'):
    n_neurons = network['n_nodes']
    node_types_path = os.path.join(data_dir, 'network/v1_node_types.csv')
    node_types = pd.read_csv(node_types_path, sep=' ')

    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')

    with h5py.File(path_to_h5, mode='r') as node_h5:
        # Create mapping from node_type_id to pop_name
        node_types.set_index('node_type_id', inplace=True)
        node_type_id_to_pop_name = node_types['pop_name'].to_dict()

        # Get node_type_ids for the current network
        node_type_ids = node_h5['nodes']['v1']['node_type_id'][()]
        true_node_type_ids = node_type_ids[network['tf_id_to_bmtk_id']]
        selected_mask = np.zeros(n_neurons, bool)

        # Vectorize the selection of neurons based on population
        for pop_id, pop_name in node_type_id_to_pop_name.items():
            # if pop_name[0] == neuron_population[0] and pop_name[1] == neuron_population[1]:
            if neuron_population in pop_name:
                # choose all the neurons of the given pop_id
                sel = true_node_type_ids == pop_id
                selected_mask = np.logical_or(selected_mask, sel)

    return selected_mask


def firing_rates_smoothing(z, sampling_rate=60, window_size=100):  # window_size=300
    n_simulations, simulation_length, n_neurons = z.shape
    sampling_interval = int(1000/sampling_rate)  # ms
    window_size = int(np.round(window_size/sampling_interval))
    # z = z.reshape(n_simulations, simulation_length, z.shape[1])
    z_chunks = [z[:, x:x+sampling_interval, :]
                for x in range(0, simulation_length, sampling_interval)]
    # (simulation_length, n_simulations, n_neurons)
    sampled_firing_rates = np.array(
        [np.sum(group, axis=1) * sampling_rate for group in z_chunks])
    smoothed_fr = gaussian_filter1d(sampled_firing_rates, window_size, axis=0)
    smoothed_fr = np.swapaxes(smoothed_fr, 0, 1)
    return smoothed_fr, sampling_interval


def voltage_spike_effect_correction(v, z, pre_spike_gap=2, post_spike_gap=3):
    n_simulations, simulation_length, n_neurons = v.shape
    for neuron_id in range(n_neurons):
        for sim in range(n_simulations):
            vs = v[sim, :, neuron_id]
            zs = z[sim, :, neuron_id].astype(dtype=bool)
            for t_idx, spike in enumerate(zs[:-post_spike_gap]):
                if spike and t_idx >= pre_spike_gap:
                    prev_value = vs[t_idx-pre_spike_gap]
                    post_value = vs[t_idx+post_spike_gap]
                    xp = [t_idx-pre_spike_gap, t_idx+post_spike_gap]
                    fp = [prev_value, post_value]
                    new_values = np.interp(
                        np.arange(t_idx-pre_spike_gap+1, t_idx+post_spike_gap, 1), xp, fp)
                    v[sim, t_idx-pre_spike_gap+1:t_idx +
                        post_spike_gap, neuron_id] = new_values
    return v


# def load_simulation_results(full_data_path, area='v1', n_simulations=None, skip_first_simulation=False,
#                             variables=None, simulation_length=2500, n_neurons=51978):
#     if n_simulations is None:
#         n_simulations = len(filter(os.listdir(full_data_path), 'v*.lzma'))
#     first_simulation = 0
#     last_simulation = n_simulations
#     if skip_first_simulation:
#         n_simulations -= 1
#         first_simulation += 1
#     if variables == None:
#         variables = ['v', 'z', 'input_current',
#                      'recurrent_current', 'bottom_up_current']  # missing z_lgn
#     if type(variables) == str:
#         variables = [variables]
#     data = {key: (np.zeros((n_simulations, simulation_length, 17400), np.float32)
#                   if key == 'z_lgn' else
#                   np.zeros((n_simulations, simulation_length, n_neurons), np.float32))
#             for key in variables}

#     for i in range(first_simulation, last_simulation):
#         for key, value in data.items():
#             data[key][(i-first_simulation):(i+1-first_simulation), :, :] = np.array(file_management.load_lzma(os.path.join(
#                 full_data_path, f'{area}_{key}_{n_neurons}_{i}.lzma')))

#     if len(variables) == 1:
#         data = data[key]

#     return data, n_simulations


############################ DATA SAVING AND LOADING METHODS #########################
class SaveSimDataHDF5:
    def __init__(self, flags, data_path, networks, save_core_only=True):
        self.v1_neurons = networks['v1']['n_nodes']
        self.lm_neurons = networks['lm']['n_nodes']
        self.v1_core_neurons = 51978
        self.lm_core_neurons = 7414
        self.data_path = data_path

        if self.v1_neurons > self.v1_core_neurons and save_core_only:
            # Isolate the core neurons from v1
            self.v1_core_mask = isolate_core_neurons(networks['v1'], n_selected_neurons=self.v1_core_neurons, data_dir=flags.data_dir) 
        else:
            self.v1_core_neurons = self.v1_neurons
            self.v1_core_mask = np.full(self.v1_core_neurons, True)
        
        if self.lm_neurons > self.lm_core_neurons and save_core_only:
            # Isolate the core neurons from lm
            self.lm_core_mask = isolate_core_neurons(networks['lm'], n_selected_neurons=self.lm_core_neurons, data_dir=flags.data_dir)
        else:
            self.lm_core_neurons = self.lm_neurons
            self.lm_core_mask = np.full(self.lm_core_neurons, True)

        # Define the shape of the data matrix
        self.v1_data_shape = (flags.n_trials, flags.seq_len, self.v1_core_neurons)
        self.lm_data_shape = (flags.n_trials, flags.seq_len, self.lm_core_neurons)
        self.LGN_data_shape = (flags.n_trials, flags.seq_len, flags.n_input)

        with h5py.File(os.path.join(self.data_path, 'simulation_data.hdf5'), 'w') as f:
            g = f.create_group('Data')
            # create a group for v1 and other for lm
            v1g = g.create_group('v1')
            lmg = g.create_group('lm')
            LGNg = g.create_group('LGN')
            v1g.create_dataset('z', self.v1_data_shape, dtype=np.uint8, 
                                chunks=True, compression='gzip', shuffle=True)
            lmg.create_dataset('z', self.lm_data_shape, dtype=np.uint8, 
                                chunks=True, compression='gzip', shuffle=True)
            LGNg.create_dataset('z', self.LGN_data_shape, dtype=np.uint8, 
                                chunks=True, compression='gzip', shuffle=True)
                
            for flag, val in flags.flag_values_dict().items():
                if isinstance(val, (float, int, str, bool)):
                    g.attrs[flag] = val
            g.attrs['Date'] = time.time()
                
    def __call__(self, simulation_data, trial):
        with h5py.File(os.path.join(self.data_path, 'simulation_data.hdf5'), 'a') as f:
            # iterate over the keys of simulation_data
            for area in simulation_data.keys():
                for key, val in simulation_data[area].items():
                    if area == 'LGN':
                        val = np.array(val).astype(np.uint8)
                        # val = np.packbits(val)
                    elif area == 'v1':
                        val = np.array(val)[:, :, self.v1_core_mask].astype(np.uint8)
                    elif area == 'lm':
                        val = np.array(val)[:, :, self.lm_core_mask].astype(np.uint8)

                    # Save the data
                    f['Data'][area][key][trial, :, :] = val


def load_simulation_results_hdf5(full_data_path, n_trials=None, skip_first_simulation=False):
    # Prepare dictionary to store the simulation metadata
    flags_dict = {}
    with h5py.File(full_data_path, 'r') as f:
        dataset = f['Data']
        flags_dict.update(dataset.attrs)
        # Get the simulation features
        if n_trials is None:
            n_trials = dataset['v1']['z'].shape[0]
        first_simulation = 0
        last_simulation = n_trials
        if skip_first_simulation:
            n_trials -= 1
            first_simulation += 1
        # Extract the simulation data
        data = {}
        for area in dataset.keys():
            data[area] = {}
            data[area]['z'] = np.array(dataset[area]['z'][first_simulation:last_simulation, :,:]).astype(np.uint8) 

    return data, flags_dict, n_trials




class SaveGaborSimDataHDF5:
    def __init__(self, flags, data_path, networks, n_rows=1, n_cols=1, n_directions=4, save_core_only=True):
        self.v1_neurons = networks['v1']['n_nodes']
        self.lm_neurons = networks['lm']['n_nodes']
        self.v1_core_neurons = 51978
        self.lm_core_neurons = 7414
        self.data_path = data_path

        if self.v1_neurons > self.v1_core_neurons and save_core_only:
            # Isolate the core neurons from v1
            self.v1_core_mask = isolate_core_neurons(networks['v1'], n_selected_neurons=self.v1_core_neurons, data_dir=flags.data_dir) 
        else:
            self.v1_core_neurons = self.v1_neurons
            self.v1_core_mask = np.full(self.v1_core_neurons, True)
        
        if self.lm_neurons > self.lm_core_neurons and save_core_only:
            # Isolate the core neurons from lm
            self.lm_core_mask = isolate_core_neurons(networks['lm'], n_selected_neurons=self.lm_core_neurons, data_dir=flags.data_dir)
        else:
            self.lm_core_neurons = self.lm_neurons
            self.lm_core_mask = np.full(self.lm_core_neurons, True)

        # Define the shape of the data matrix
        self.v1_data_shape = (flags.n_trials, 4*flags.seq_len, self.v1_core_neurons)
        self.lm_data_shape = (flags.n_trials, 4*flags.seq_len, self.lm_core_neurons)
        self.LGN_data_shape = (flags.n_trials, 4*flags.seq_len, flags.n_input)

        self.data_shapes = {'v1': self.v1_data_shape, 'lm': self.lm_data_shape, 'LGN': self.LGN_data_shape}

        row_ids = [6] #np.arange(0, n_rows)
        col_ids = [5] #np.arange(0, n_cols)
        # directions = np.arange(0, 180, 45)

        with h5py.File(os.path.join(self.data_path, 'simulation_data.hdf5'), 'w') as f:
            g = f.create_group('Data')
            # create a group for v1 and other for lm
            for key, data_shape in self.data_shapes.items():
                area = g.create_group(key)
                for row in row_ids:
                    for col in col_ids:
                        area.create_dataset(f'{row}_{col}', data_shape, dtype=np.uint8, 
                                                    chunks=True, compression='gzip', shuffle=True)
                
            for flag, val in flags.flag_values_dict().items():
                if isinstance(val, (float, int, str, bool)):
                    g.attrs[flag] = val
            g.attrs['Date'] = time.time()
                
    def __call__(self, simulation_data, trial, row, col):
        with h5py.File(os.path.join(self.data_path, 'simulation_data.hdf5'), 'a') as f:
            # iterate over the keys of simulation_data
            for area in simulation_data.keys():
                for key, val in simulation_data[area].items():
                    if area == 'LGN':
                        val = np.array(val).astype(np.uint8)
                        # val = np.packbits(val)
                    elif area == 'v1':
                        val = np.array(val)[:, :, self.v1_core_mask].astype(np.uint8)
                    elif area == 'lm':
                        val = np.array(val)[:, :, self.lm_core_mask].astype(np.uint8)

                    # Save the data
                    f['Data'][area][f'{row}_{col}'][trial, :, :] = val


def load_gabor_simulation_results_hdf5(full_data_path, n_trials=None, skip_first_simulation=False):
    # Prepare dictionary to store the simulation metadata
    flags_dict = {}
    with h5py.File(full_data_path, 'r') as f:
        dataset = f['Data']
        flags_dict.update(dataset.attrs)
        # Get the simulation features
        if n_trials is None:
            n_trials = dataset['v1']['0_0'].shape[0]
        first_simulation = 0
        last_simulation = n_trials
        if skip_first_simulation:
            n_trials -= 1
            first_simulation += 1
        # Extract the simulation data
        data = {}
        for area in dataset.keys():
            data[area] = {}
            for row_col in dataset[area].keys():
                data[area][row_col] = {}
                for direction in dataset[area][row_col].keys():
                    data[area][row_col][direction] = np.array(dataset[area][row_col][direction][first_simulation:last_simulation, :,:]).astype(np.uint8)
                data[area][row_col] = {}
                for direction in dataset[area][row_col].keys():
                    data[area][row_col][direction] = np.array(dataset[area][row_col][direction][first_simulation:last_simulation, :,:]).astype(np.uint8)
                
    return data, flags_dict, n_trials
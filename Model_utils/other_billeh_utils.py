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


def pop_names(network, core_radius= None, data_dir='GLIF_network'):
    path_to_csv = os.path.join(data_dir, 'network/V1_node_types.csv')
    path_to_h5 = os.path.join(data_dir, 'network/V1_nodes.h5')
    node_types = pd.read_csv(path_to_csv, sep=' ')
    node_h5 = h5py.File(path_to_h5, mode='r')
    node_type_id_to_pop_name = dict()
    for nid in np.unique(node_h5['nodes']['v1']['node_type_id']):
        # if not np.unique all of the 230924 model neurons ids are considered,
        # but nearly all of them are repeated since there are only 111 different indices
        ind_list = np.where(node_types.node_type_id == nid)[0]
        assert len(ind_list) == 1
        node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]
    true_pop_names = []  # it contains the pop_name of all the 230,924 neurons
    for nid in node_h5['nodes']['v1']['node_type_id']:
        true_pop_names.append(node_type_id_to_pop_name[nid])
     # Select population names of neurons in the present network (core)
    true_pop_names = np.array(true_pop_names)[network['tf_id_to_bmtk_id']]

    if core_radius is not None:
        selected_mask = isolate_core_neurons(network, radius=core_radius, data_dir=data_dir)
        true_pop_names = true_pop_names[selected_mask]

    return true_pop_names


def angle_tunning(network, data_dir='GLIF_network'):
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    node_h5 = h5py.File(path_to_h5, mode='r')
    angle_tunning = np.array(node_h5['nodes']['v1']['0']['tuning_angle'][:])[
        network['tf_id_to_bmtk_id']]

    return angle_tunning


def isolate_core_neurons(network, radius=400, data_dir='GLIF_network'):
    path_to_h5 = os.path.join(data_dir, 'network/V1_nodes.h5')
    node_h5 = h5py.File(path_to_h5, mode='r')
    x = np.array(node_h5['nodes']['v1']['0']['x'])
    z = np.array(node_h5['nodes']['v1']['0']['z'])
    r = np.sqrt(x ** 2 + z ** 2)
    r_network_nodes = r[network['tf_id_to_bmtk_id']]
    selected_mask = r_network_nodes < radius

    return selected_mask


def isolate_neurons(network, neuron_population='e23', data_dir='GLIF_network'):
    n_neurons = network['n_nodes']
    node_types = pd.read_csv(os.path.join(
        data_dir, 'network/v1_node_types.csv'), sep=' ')
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    node_h5 = h5py.File(path_to_h5, mode='r')
    node_type_id_to_pop_name = dict()
    for nid in np.unique(node_h5['nodes']['v1']['node_type_id']):
        # if not np.unique all of the 230924 model neurons ids are considered,
        # but nearly all of them are repeated since there are only 111 different indices
        ind_list = np.where(node_types.node_type_id == nid)[0]
        assert len(ind_list) == 1
        node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]

    node_type_ids = np.array(node_h5['nodes']['v1']['node_type_id'])
    true_node_type_ids = node_type_ids[network['tf_id_to_bmtk_id']]
    selected_mask = np.zeros(n_neurons, bool)
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


# def load_simulation_results(full_data_path, area='V1', n_simulations=None, skip_first_simulation=False,
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
    def __init__(self, flags, keys, data_path, networks, n_neurons, V1_to_LM_neurons_ratio, save_core_only=True, dtype=np.float16):
        self.keys = keys
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        self.dtype = dtype
        self.V1_neurons = n_neurons['V1']
        self.LM_neurons = n_neurons['LM']
        V1_core_radius = 400
        self.V1_core_neurons = 51978
        self.LM_core_neurons = 7285

        if self.V1_neurons > self.V1_core_neurons and save_core_only:
            # Isolate the core neurons from V1
            self.V1_core_mask = isolate_core_neurons(networks['V1'], radius=V1_core_radius, data_dir=flags.data_dir) 
        else:
            self.V1_core_neurons = self.V1_neurons
            self.V1_core_mask = np.full(self.V1_core_neurons, True)
        
        if self.LM_neurons > self.LM_core_neurons and save_core_only:
            # Isolate the core neurons from LM
            LM_core_radius = V1_core_radius/np.sqrt(V1_to_LM_neurons_ratio)
            self.LM_core_mask = isolate_core_neurons(networks['LM'], radius=LM_core_radius, data_dir=flags.data_dir)
        else:
            self.LM_core_neurons = self.LM_neurons
            self.LM_core_mask = np.full(self.LM_core_neurons, True)

        # Define the shape of the data matrix
        self.V1_data_shape = (flags.n_simulations, flags.seq_len, self.V1_neurons)
        self.V1_core_data_shape = (flags.n_simulations, flags.seq_len, self.V1_core_neurons)
        self.LM_data_shape = (flags.n_simulations, flags.seq_len, self.LM_neurons)
        self.LM_core_data_shape = (flags.n_simulations, flags.seq_len, self.LM_core_neurons)
        self.LGN_data_shape = (flags.n_simulations, flags.seq_len, flags.n_input)

        with h5py.File(os.path.join(self.data_path, 'simulation_data.hdf5'), 'w') as f:
            g = f.create_group('Data')
            # create a group for V1 and other for LM
            V1g = g.create_group('V1')
            LMg = g.create_group('LM')
            LGNg = g.create_group('LGN')
            for key in self.keys:
                if key=='z':
                    V1g.create_dataset(key, self.V1_data_shape, dtype=np.uint8, 
                                     chunks=True, compression='gzip', shuffle=True)
                    LMg.create_dataset(key, self.LM_data_shape, dtype=np.uint8, 
                                     chunks=True, compression='gzip', shuffle=True)
                elif key=='z_lgn':
                    LGNg.create_dataset(key, self.LGN_data_shape, dtype=np.uint8, 
                                     chunks=True, compression='gzip', shuffle=True)
                else:
                    V1g.create_dataset(key, self.V1_core_data_shape, dtype=self.dtype, 
                                     chunks=True, compression='gzip', shuffle=True)
                    LMg.create_dataset(key, self.LM_core_data_shape, dtype=self.dtype,
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
                    if key in ['z', 'z_lgn']:
                        val = np.array(val).astype(np.uint8)
                        # val = np.packbits(val)
                    else:
                        if area == 'V1':
                            val = np.array(val)[:, :, self.V1_core_mask].astype(self.dtype)
                        elif area == 'LM':
                            val = np.array(val)[:, :, self.LM_core_mask].astype(self.dtype)

                    # Save the data
                    f['Data'][area][key][trial, :, :] = val

     

def load_simulation_results_hdf5(full_data_path, n_simulations=None, skip_first_simulation=False, 
                                variables=None):
    # Prepare dictionary to store the simulation metadata
    flags_dict = {}
    with h5py.File(full_data_path, 'r') as f:
        dataset = f['Data']
        flags_dict.update(dataset.attrs)
        # Get the simulation features
        if n_simulations is None:
            n_simulations = dataset['Data']['V1']['z'].shape[0]
        first_simulation = 0
        last_simulation = n_simulations
        if skip_first_simulation:
            n_simulations -= 1
            first_simulation += 1
        # Select the variables for the extraction
        if variables == None:
            variables = ['v', 'z', 'input_current', 'recurrent_current', 'bottom_up_current', 'z_lgn']
        if type(variables) == str:
            variables = [variables]

        # Extract the simulation data
        data = {}
        if 'z_lgn' in variables:
            data['LGN'] = {}
            data['LGN']['z_lgn'] = np.array(dataset['LGN']['z_lgn'][first_simulation:last_simulation, :,:]).astype(np.uint8)

        else:
            for area in dataset.keys():
                if area != 'LGN':
                    data[area] = {}
                    for key in variables:
                        if key == 'z':
                            data[area][key] = np.array(dataset[area][key][first_simulation:last_simulation, :,:]).astype(np.uint8) 
                        else:
                            data[area][key] = np.array(dataset[area][key][first_simulation:last_simulation, :,:]).astype(np.float32)
            
    # if len(variables) == 1:
    #     data = data[key]
    #         
    return data, flags_dict, n_simulations
import os
import h5py
import numpy as np
from numba import njit
from scipy.stats import levy_stable
import pandas as pd
import pickle as pkl
import json
import logging
import tensorflow as tf

np.random.seed(0)


@njit
def sort_indices(indices, weights, delays):
    max_ind = np.max(indices) + 1
    # sort indices by considering first all the targets of node 0, then all of node 1, ...
    q = indices[:, 0] * max_ind + indices[:, 1]
    sorted_ind = np.argsort(q)
    return indices[sorted_ind], weights[sorted_ind], delays[sorted_ind]


@njit
def intersomatic_distance(sources_projected_x, targets_x, sources_projected_z, targets_z):
    return np.sqrt(np.square(sources_projected_x - targets_x) + np.square(sources_projected_z - targets_z))


@njit
def gaussian_decay(r, a, sigma):
    return a*np.exp(-(r/sigma)**2)


@njit
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)


@njit
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def direction_tuning_factor(src_tuning, tar_tuning, sigma):
    delta_tuning_180 = np.abs(
        np.abs(np.mod(np.abs(tar_tuning - src_tuning), 360.0) - 180.0) - 180.0)
    w_multiplier_180 = np.exp(-(delta_tuning_180 / sigma) ** 2)
    return w_multiplier_180


def node_type_id_to_pop_name(data_dir='GLIF_network'):
    path_to_csv = os.path.join(data_dir, 'network/V1_node_types.csv')
    path_to_h5 = os.path.join(data_dir, 'network/V1_nodes.h5')
    node_types = pd.read_csv(path_to_csv, sep=' ')
    node_h5 = h5py.File(path_to_h5, mode='r')
    node_type_id_to_pop_name_dict = dict()
    for nid in np.unique(node_h5['nodes']['v1']['node_type_id']):
        # if not np.unique all of the 230924 model neurons ids are considered,
        # but nearly all of them are repeated since there are only 111 different indices
        ind_list = np.where(node_types.node_type_id == nid)[0]
        assert len(ind_list) == 1
        node_type_id_to_pop_name_dict[nid] = node_types.pop_name[ind_list[0]]

    return node_type_id_to_pop_name_dict


def load_network(path,
                 h5_path,
                 n_neurons=None, seed=3000, core_only=True, connected_selection=False):
    rd = np.random.RandomState(seed=seed)

    with open(path, 'rb') as f:
        d = pkl.load(f)  # d is a dictionary with 'nodes' and 'edges' keys

  # This file contains the data related to each neuron class.
  # The nodes key is a list of 111 entries (one per neuron class) with the following information:
  #  'ids' (bmtk indices of the class neurons): array([173915, 174530, 175234, ..., 230780], dtype=uint32)
  #  'params': {'asc_init': [0.0, 0.0],
  #             'V_th': -34.78002413066345,
  #             'g': 4.332666343216805,
  #             'E_L': -71.3196309407552,
  #             'k': [0.003, 0.029999999999999992],
  #             'C_m': 61.776013140488196,
  #             'V_reset': -71.3196309407552,
  #             'V_dynamics_method': 'linear_exact',
  #             'tau_syn': [5.5, 8.5, 2.8, 5.8],
  #             't_ref': 2.2,
  #             'asc_amps': [-6.621493991981387, -68.56339310938284]}

  # The 'edges' key is a list of 1783 entries (one per edge class) with the following information:
  #  'source': array([   86,   195,   874, ..., 26266, 26563, 26755], dtype=uint64), # bmtk indices
  #  'target': array([13289, 13289, 13289, ..., 26843, 26843, 26843], dtype=uint64), # bmtk indices
  #  'params': {'model': 'static_synapse',
  #             'receptor_type': 1,
  #             'delay': 1.5,
  #             'weight': array([2.05360475e-07, 1.18761259e-20, 1.04067864e-12, ...,
  #                              3.33087865e-34, 1.26318969e-03, 1.20919572e-01])}

    n_nodes = sum([len(a['ids']) for a in d['nodes']])  # 230924 total neurons
    n_edges = sum([len(a['source'])
                  for a in d['edges']])  # 70139111 total edges
    # max_delay = max([a['params']['delay'] for a in d['edges']])

    bmtk_id_to_tf_id = np.arange(n_nodes)
    tf_id_to_bmtk_id = np.arange(n_nodes)

    edges = d['edges']
    h5_file = h5py.File(h5_path, 'r')
    # This file gives us the:
    # '0': coordinates of each point and tuning angle
    # 'node_group_id': all nodes have the index 0
    # 'node_group_index': same as node_id
    # 'node_id': bmtk index of each node (node_id[0]=0, node_id[1]=1, ...)
    # 'node_type_id': 518290966, 539742766,... for each node
    assert np.diff(h5_file['nodes']['v1']['node_id']).var() < 1e-12
    x = np.array(h5_file['nodes']['v1']['0']['x'])
    y = np.array(h5_file['nodes']['v1']['0']['y'])
    z = np.array(h5_file['nodes']['v1']['0']['z'])
    tuning_angle = np.array(h5_file['nodes']['v1']['0']['tuning_angle'])
    node_type_id = np.array(h5_file['nodes']['v1']['node_type_id'])
    # its a cylinder where the y variable is just the depth
    r = np.sqrt(x ** 2 + z ** 2)

    # sel is a boolean array with True value in the indices of selected neurons
    if connected_selection:  # this condition takes the n_neurons closest neurons
        # order according to radius distance. This option allows for more synapses since close neurons are more probably connected
        sorted_ind = np.argsort(r)
        sel = np.zeros(n_nodes, np.bool_)
        sel[sorted_ind[:n_neurons]] = True  # keep only the nearest n_neurons
        print(f'> Maximum sample radius: {r[sorted_ind[n_neurons - 1]]:.2f}')
    # this condition makes all the neurons to be within distance 400 micrometers from the origin (core)
    elif core_only:
        # 51,978 maximum value for n_neurons in this case
        sel = r < 400
        if n_neurons is not None and n_neurons > 0:
            inds, = np.where(sel)  # indices where the condition is satisfied
            take_inds = rd.choice(inds, size=n_neurons, replace=False)
            sel[:] = False
            sel[take_inds] = True

    # Choose n_neurons random neurons whithout any other requirement
    elif n_neurons is not None and n_neurons > 0:  # this condition takes random neurons from all the V1
        legit_neurons = np.arange(n_nodes)
        take_inds = rd.choice(legit_neurons, size=n_neurons, replace=False)
        sel = np.zeros(n_nodes, np.bool_)
        sel[take_inds] = True

    n_nodes = np.sum(sel)  # number of nodes selected
    # tf idx '0' corresponds to 'tf_id_to_bmtk_id[0]' bmtk idx
    tf_id_to_bmtk_id = tf_id_to_bmtk_id[sel]
    bmtk_id_to_tf_id = np.zeros_like(bmtk_id_to_tf_id) - 1
    for tf_id, bmtk_id in enumerate(tf_id_to_bmtk_id):
        bmtk_id_to_tf_id[bmtk_id] = tf_id

    # bmtk idx '0' corresponds to 'bmtk_id_to_tf_id[0]' tf idx which can be '-1' in case
    # the bmtk node is not in the tensorflow selection or another value in case it belongs the selection
    x = x[sel]
    y = y[sel]
    z = z[sel]
    tuning_angle = tuning_angle[sel]
    node_type_id = node_type_id[sel]

    # from all the model edges, lets see how many correspond to the selected nodes
    n_edges = 0
    for edge in edges:
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge['target'])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge['source'])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        n_edges += np.sum(edge_exists)

    print(f'> Number of Neurons: {n_nodes}')
    print(f'> Number of Synapses: {n_edges}')

    # Save in a dictionary the properties of each of the 111 node types
    n_node_types = len(d['nodes'])
    node_params = dict(
        V_th=np.zeros(n_node_types, np.float32),
        g=np.zeros(n_node_types, np.float32),
        E_L=np.zeros(n_node_types, np.float32),
        k=np.zeros((n_node_types, 2), np.float32),
        C_m=np.zeros(n_node_types, np.float32),
        V_reset=np.zeros(n_node_types, np.float32),
        tau_syn=np.zeros((n_node_types, 4), np.float32),
        t_ref=np.zeros(n_node_types, np.float32),
        asc_amps=np.zeros((n_node_types, 2), np.float32)
    )

    # give every selected node of a given node type an index according to tf ids
    node_type_ids = np.zeros(n_nodes, np.int64)
    for i, node_type in enumerate(d['nodes']):
        # get ALL the nodes of the given node type
        tf_ids = bmtk_id_to_tf_id[np.array(node_type['ids'])]
        # choose only those that belong to our model
        tf_ids = tf_ids[tf_ids >= 0]
        # assign them all the same id (which does not relate with the neuron type)
        node_type_ids[tf_ids] = i
        for k, v in node_params.items():
            # save in a dict the information of the nodes
            v[i] = node_type['params'][k]

    # each node has 4 different inputs (soma, dendrites, etc) with different properties each
    dense_shape = (4 * n_nodes, n_nodes)
    indices = np.zeros((n_edges, 2), dtype=np.int64)
    weights = np.zeros(n_edges, np.float32)
    delays = np.zeros(n_edges, np.float32)

    current_edge = 0
    for edge in edges:
        # Indentify the which of the 4 types of inputs we have
        r = edge['params']['receptor_type'] - 1
        # r takes values whithin 0 - 3
        target_tf_ids = bmtk_id_to_tf_id[np.array(edge['target'])]
        source_tf_ids = bmtk_id_to_tf_id[np.array(edge['source'])]
        edge_exists = np.logical_and(target_tf_ids >= 0, source_tf_ids >= 0)
        # select the edges whithin our model
        target_tf_ids = target_tf_ids[edge_exists]
        source_tf_ids = source_tf_ids[edge_exists]
        weights_tf = edge['params']['weight'][edge_exists]
        # all the edges of a given type have the same delay
        delays_tf = edge['params']['delay']
        n_new_edge = np.sum(edge_exists)
        indices[current_edge:current_edge +
                n_new_edge] = np.array([target_tf_ids * 4 + r, source_tf_ids]).T
        # we multiply by 4 and add r to identify the receptor_type easily:
        # if target id is divisible by 4 the receptor_type is 0,
        # if it is rest is 1 by dividing by 4 then its receptor type is 1, and so on...
        weights[current_edge:current_edge + n_new_edge] = weights_tf
        delays[current_edge:current_edge + n_new_edge] = delays_tf
        current_edge += n_new_edge
    # sort indices by considering first all the targets of node 0, then all of node 1, ...
    indices, weights, delays = sort_indices(indices, weights, delays)

    network = dict(
        x=x, y=y, z=z,
        tuning_angle=tuning_angle,
        node_type_id=node_type_id,
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_params=node_params,
        node_type_ids=node_type_ids,
        synapses=dict(indices=indices, weights=weights, delays=delays,
                      dense_shape=dense_shape),
        tf_id_to_bmtk_id=tf_id_to_bmtk_id,
        bmtk_id_to_tf_id=bmtk_id_to_tf_id,
        # interarea_synapses=dict(lower=dict(),higher=dict())
    )

    return network


# Here we load the 17400 neurons that act as input in the model
def load_input(path,
               start=0,
               duration=3000,
               dt=1,
               bmtk_id_to_tf_id=None):
    with open(path, 'rb') as f:
        # d contains two populations (LGN and background inputs), each of them with two elements:
        d = pkl.load(f)
        # [0] a dict with 'ids' and 'spikes'
        # [1] a list of edges types, each of them with their 'source', 'target' and 'params'
        # The first population (LGN) has 86 different edges and the source ids go from 0 to 17399
        # The second population (background) is only formed by the source index 0 (single background node)
        # and projects to all network neurons with 21 different edges types and weights

    input_populations = []
    for input_population in d:
        post_indices = []
        pre_indices = []
        weights = []
        delays = []

        for edge in input_population[1]:
            # Indentify the which of the 4 types of inputs we have
            r = edge['params']['receptor_type'] - 1
            # r takes values whithin 0 - 3
            target_bmtk_id = np.array(edge['target'])
            source_tf_id = np.array(edge['source'])
            weights_tf = np.array(edge['params']['weight'])
            delays_tf = np.zeros_like(weights_tf) + edge['params']['delay']
            # if bmtk_id_to_tf_id is not None:
            # check if the given edges exist in our model
            # (notice that only the target must exist since the source is whithin the LGN module)
            # This means that source index is whithin 0-17400
            target_tf_id = bmtk_id_to_tf_id[target_bmtk_id]
            edge_exists = target_tf_id >= 0
            target_tf_id = target_tf_id[edge_exists]
            source_tf_id = source_tf_id[edge_exists]
            weights_tf = weights_tf[edge_exists]
            delays_tf = delays_tf[edge_exists]
            # we multiply by 4 the indices and add r to identify the receptor_type easily:
            # if target id is divisible by 4 the receptor_type is 0,
            # if it is rest is 1 by dividing by 4 then its receptor type is 1, and so on...
            # extend acts by extending the list with the given object
            post_indices.extend(4 * target_tf_id + r)
            pre_indices.extend(source_tf_id)
            weights.extend(weights_tf)
            delays.append(delays_tf)
        # first column are the post indices and second column the pre indices
        indices = np.stack([post_indices, pre_indices], -1)
        weights = np.array(weights)
        delays = np.concatenate(delays)
        # sort indices by considering first all the sources of target node 0, then all of node 1, ...
        indices, weights, delays = sort_indices(indices, weights, delays)

        n_input_neurons = len(input_population[0]['ids'])  # 17400
        spikes = np.zeros((int(duration / dt), n_input_neurons))
        # now we save the spikes of the input population
        # ? no entiendo bien la presencia de estos spikes
        for i, sp in zip(input_population[0]['ids'], input_population[0]['spikes']):
            # consider only the spikes whithin the period we are taking
            sp = sp[np.logical_and(start < sp, sp < start + duration)] - start
            sp = (sp / dt).astype(np.int)
            for s in sp:
                spikes[s, i] += 1

        input_populations.append(dict(n_inputs=n_input_neurons, indices=indices.astype(
            np.int64), weights=weights, delays=delays, spikes=spikes))
    return input_populations


def reduce_input_population(input_population, new_n_input, seed=3000):
    rd = np.random.RandomState(seed=seed)

    in_ind = input_population['indices']
    in_weights = input_population['weights']
    in_delays = input_population['delays']

    # we take input_population['n_inputs'] neurons from a list of new_n_input with replace,
    # which means that repeated LGN neurons double their synaptic weights, ending with less
    # units than new_n_input
    assignment = rd.choice(np.arange(new_n_input),
                           size=input_population['n_inputs'], replace=True)

    weight_dict = dict()
    delays_dict = dict()
    # go through all the asignment selection made
    for input_neuron in range(input_population['n_inputs']):
        assigned_neuron = assignment[input_neuron]
        # consider neurons connected to the input_neuron
        sel = in_ind[:, 1] == input_neuron
        # keep that neurons connected to the input_neuron
        sel_post_inds = in_ind[sel, 0]
        sel_weights = in_weights[sel]
        sel_delays = in_delays[sel]
        for post_ind, weight, delay in zip(sel_post_inds, sel_weights, sel_delays):
            # for post_ind, weight in zip(sel_post_inds, sel_weights):
            # tuple with the indices of the post model neuron and the pre LGN neuron
            t_inds = post_ind, assigned_neuron
            if t_inds not in weight_dict.keys():  # in case the key hasnt been already created
                weight_dict[t_inds] = 0.
            # in case a LGN unit connection is repeated we consider that the weights are add up
            weight_dict[t_inds] += weight
            delays_dict[t_inds] = delay
    n_synapses = len(weight_dict)
    # we now save the synapses in arrays of indices and weights
    new_in_ind = np.zeros((n_synapses, 2), np.int64)
    new_in_weights = np.zeros(n_synapses)
    new_in_delays = np.zeros(n_synapses)
    for i, (t_ind, w) in enumerate(weight_dict.items()):
        new_in_ind[i] = t_ind
        new_in_weights[i] = w
        new_in_delays[i] = delays_dict[t_ind]
    new_in_ind, new_in_weights, new_in_delays = sort_indices(
        new_in_ind, new_in_weights, new_in_delays)
    # new_in_ind, new_in_weights = sort_indices(new_in_ind, new_in_weights)
    new_input_population = dict(
        n_inputs=new_n_input, indices=new_in_ind, weights=new_in_weights, delays=new_in_delays, spikes=None)
    # new_input_population = dict(
    # n_inputs=new_n_input, indices=new_in_ind, weights=new_in_weights, spikes=None)
    return new_input_population


def set_laminar_indices(df, network, flags=None):
    def get_one_layer_indices(EI, layer_number, neuron_subpop=None):
        types_indices = []
        for node_type in df.iterrows():
            if EI == 'e':
                if node_type[1]['pop_name'].startswith(f'e{layer_number}'):
                    types_indices.append(node_type[0])
                else:
                    continue
            elif EI == 'i':
                if (node_type[1]['pop_name'].startswith(f'i{layer_number}')) & (neuron_subpop in node_type[1]['pop_name']):
                    types_indices.append(node_type[0])
                else:
                    continue
            else:
                print('Error: Wrong population name')
                break

        types_indices = np.array(types_indices)
        neuron_sel = np.zeros(network['n_nodes'], np.bool_)
        for type_index in types_indices:
            is_type = network['node_type_ids'] == type_index
            neuron_sel = np.logical_or(neuron_sel, is_type)
        return np.where(neuron_sel)[0]

    network['laminar_indices'] = dict()
    for layer_number in [1, 2, 4, 5, 6]:
        network['laminar_indices'][f'L{layer_number}_e'] = get_one_layer_indices(
            'e', layer_number)

    for neuron_subpop in ['Pvalb', 'Sst', 'Htr3a']:
        for layer_number in [1, 2, 4, 5, 6]:
            network['laminar_indices'][f'L{layer_number}_i_{neuron_subpop}'] = get_one_layer_indices(
                'i', layer_number, neuron_subpop)

    # split 2 3 layers
    # vertical_coordinates_e = network['y'][network['laminar_indices']['L23e']]
    # vertical_coordinates_i = network['y'][network['laminar_indices']['L23i']]
    # vertical_coordinates = np.hstack((vertical_coordinates_e,vertical_coordinates_i))
    # L23_argindices_sorted = np.argsort(vertical_coordinates)
    # L23_neuorn_indices = np.hstack((network['laminar_indices']['L23e'],network['laminar_indices']['L23i']))

    # L2_argindices = L23_argindices_sorted[:np.int64(flags.L2_neuron_ratio*vertical_coordinates.size)]
    # L2e_argindices = L2_argindices[L2_argindices<vertical_coordinates_e.size]
    # network['laminar_indices']['L2e'] = L23_neuorn_indices[L2e_argindices]
    # L2i_argindices = L2_argindices[L2_argindices>vertical_coordinates_e.size]
    # network['laminar_indices']['L2i'] = L23_neuorn_indices[L2i_argindices]

    # L3_argindices = L23_argindices_sorted[np.int64(flags.L2_neuron_ratio*vertical_coordinates.size):]
    # L3e_argindices = L3_argindices[L3_argindices<vertical_coordinates_e.size]
    # network['laminar_indices']['L3e'] = L23_neuorn_indices[L3e_argindices]
    # L3i_argindices = L3_argindices[L3_argindices>vertical_coordinates_e.size]
    # network['laminar_indices']['L3i'] = L23_neuorn_indices[L3i_argindices]

    return network


class InterareaConnectivity:
    def __init__(self, target_network, source_network, target_column_name,
                 source_column_name, interarea_weight_distribution='billeh_like', seed=42,
                 data_dir='GLIF_network'):
        self.target_network = target_network
        self.source_network = source_network
        self.target_column_name = target_column_name
        self.source_column_name = source_column_name
        self.interarea_weight_distribution = interarea_weight_distribution
        self.seed = seed
        self.data_dir = data_dir
        self.rd = np.random.RandomState(seed=self.seed)

        # Calculate the ratio between the size of the source and target columns
        target_network_radius = np.sqrt(
            self.target_network['x']**2 + self.target_network['z']**2).max()
        source_network_radius = np.sqrt(
            self.source_network['x']**2 + self.source_network['z']**2).max()
        self.radius_ratio = target_network_radius/source_network_radius

        # Create dictionary between node_type_id and their pop names
        self.node_type_id_to_pop_name_dict = node_type_id_to_pop_name()

        # Load the dictionary with the connection probabilities parameters
        interarea_connectivity_path = os.path.join(
            self.data_dir, f'{self.source_column_name}_to_{self.target_column_name}_connection_probabilities.json')
        with open(interarea_connectivity_path, 'rb') as f:
            self.interarea_connectivity = json.load(f)

        # Load the CSV with the synaptic information (weights, delays, etc.) for every neuron pair
        edge_types_df = pd.read_csv(os.path.join(
            self.data_dir, 'network/v1_v1_edge_types.csv'), delimiter=' ')
        edge_types_df['target_query'] = edge_types_df['target_query'].str.slice(
            start=15, stop=24)
        edge_types_df['target_query'] = edge_types_df['target_query'].astype(
            np.uint32)
        edge_types_df['source_query'] = edge_types_df['source_query'].str.slice(
            start=11, stop=-1)

        self.edge_params_keys = [
            'syn_weight', 'weight_function', 'weight_sigma', 'delay', 'dynamics_params']
        self.edges_params = {target_query: {}
                             for target_query in set(edge_types_df['target_query'])}
        for index, row in edge_types_df.iterrows():
            self.edges_params[row['target_query']][row['source_query']] = {}
            for key in self.edge_params_keys:
                self.edges_params[row['target_query']
                                  ][row['source_query']][key] = row[key]

    def compute_pair_type_parameters(self, source_type, target_type):
        """ Takes in two strings for the source and target type. It determined the connectivity parameters needed based on
        distance dependence and orientation tuning dependence and returns a dictionary of these parameters. A description
        of the calculation and resulting formulas used herein can be found in the accompanying documentation. Note that the
        source and target can be the same as this function works on cell TYPES and not individual nodes. The first step of
        this function is getting the parameters that determine the connectivity probabilities reported in the literature.
        From there the calculation proceed based on adapting these values to our model implementation.

        :param source_type: string of the cell type that will be the source (pre-synaptic)
        :param target_type: string of the cell type that will be the targer (post-synaptic)
        :return: dictionary with the values to be used for distance dependent connectivity
                 and orientation tuning dependent connectivity (when applicable, else nans populate the dictionary).
        """

        target = '_'.join(target_type.split('_')[1:])
        source = '_'.join(source_type.split('_')[1:])

        ##### For distance dependence which is modeled as a Gaussian ####
        # P = A * exp(-r^2 / sigma^2)

        # A_0 is different for every source-target pair and was estimated from the BBP model.
        A_0 = self.interarea_connectivity[source_type][target_type]['amplitude']

        # Sigma is estimated from the BBP model.
        sigma = self.interarea_connectivity[source_type][target_type]['sigma']

        # We confirmed that A_0 is lower than 1. If this does happen, it is for a few cases and is not much higher than 1.0.
        if A_0 > 1.0:
            logging.warning('Adjusted calculated probability based on distance dependence is coming out to be '
                            'greater than 1 for ' + source_type + ' and ' + target_type + '. Setting to 1.0')
            A_0 = 1.0

        ##### To include orientation tuning ####
        # Many cells will show orientation tuning and the relative different in orientation tuning angle will influence
        # probability of connections as has been extensively report in the literature. This is modeled here with a linear
        # where B in the largest value from 0 to 90 (if the profile decays, then B is the intercept, if the profile
        # increases, then B is the value at 90). The value of G is the gradient of the curve.
        # The calculations and explanation can be found in the accompanying documentation with this code.

        # B_ratio takes value 0.8 for e-e5 connections, 0.5 for other e-e connections and np.nan for connections involving
        # inhibitory neurons
        if 'i' in source or 'i' in target:
            B_ratio = np.nan
        else:
            if '5' in target:
                B_ratio = 0.8
            else:
                B_ratio = 0.5

        # Check if there is orientation dependence in this source-target pair type. If yes, then a parallel calculation
        # to the one done above for distance dependence is made though with the assumption of a linear profile.
        if not np.isnan(B_ratio):
            # The scaling for distance and orientation must remain less than 1 which is calculated here and reset
            # if it is greater than one. We also ensure that the area under the p(delta_phi) curve is always equal
            # to one (see documentation). Hence the desired ratio by the user may not be possible, in which case
            # an warning message appears indicating the new ratio used. In the worst case scenario the line will become
            # horizontal (no orientation tuning) but will never "reverse" slopes.

            # B1 is the intercept which occurs at (0, B1)
            # B2 is the value when delta_phi equals 90 degree and hence the point (90, B2)
            B1 = 2.0 / (1.0 + B_ratio)
            B2 = B_ratio * B1

            AB = A_0 * max(B1, B2)
            if AB > 1.0:
                if B1 >= B2:
                    B1_new = 1.0 / A_0
                    delta = B1 - B1_new
                    B1 = B1_new
                    B2 = B2 + delta
                elif (B2 > B1):
                    B2_new = 1.0 / A_0
                    delta = B2 - B2_new
                    B2 = B2_new
                    B1 = B1 + delta

                B_ratio = B2 / B1
                logging.warning(
                    'Could not satisfy the desired B_ratio, probability of connectivity would become greater than one in'
                    'some cases. Rescaled, {} --> {} the ratio is set to {}'.format(
                        source_type, target_type, B_ratio)
                )

            G = (B2 - B1) / 90.0

        # If there is no orientation dependent, record this by setting the intercept to Not a Number (NaN).
        else:
            B1 = np.NaN
            G = np.NaN

        # Return the dictionary. Note, the new values are A_0 and intercept.
        return {
            'A_0': A_0,
            'sigma': sigma,
            'gradient': G,
            'intercept': B1,
            'nsyn_range': [3, 8]
        }

    def DirectionRule_EE(self, src_tf_ids, trg_tf_ids, src_x_projected, src_z_projected, nsyns_ret, synapsis_params):

        tar_tuning = self.target_network['tuning_angle'][trg_tf_ids]
        src_tuning = self.source_network['tuning_angle'][src_tf_ids]
        x_tar = self.target_network['x'][trg_tf_ids]
        z_tar = self.target_network['z'][trg_tf_ids]
        x_src = src_x_projected
        z_src = src_z_projected

        sigma = synapsis_params['weight_sigma']
        syn_weight = synapsis_params['syn_weight']

        w_multiplier_180 = direction_tuning_factor(
            src_tuning, tar_tuning, sigma)

        delta_x = (x_tar - x_src) * 0.07
        delta_z = (z_tar - z_src) * 0.04

        theta_pref = tar_tuning * (np.pi / 180.)
        xz = delta_x * np.cos(theta_pref) + delta_z * np.sin(theta_pref)
        sigma_phase = 1.0
        phase_scale_ratio = np.exp(- (xz ** 2 / (2 * sigma_phase ** 2)))

        # To account for the 0.07 vs 0.04 dimensions. This ensures the horizontal neurons are scaled by 5.5/4 (from the
        # midpoint of 4 & 7). Also, ensures the vertical is scaled by 5.5/7. This was a basic linear estimate to get the
        # numbers (y = ax + b).
        theta_tar_scale = abs(
            abs(abs(180.0 - np.mod(np.abs(tar_tuning), 360.0)) - 90.0) - 90.0)
        phase_scale_ratio = phase_scale_ratio * \
            (5.5 / 4.0 - 11.0 / 1680.0 * theta_tar_scale)

        return syn_weight * w_multiplier_180 * phase_scale_ratio * nsyns_ret

    def DirectionRule_others(self, src_tf_ids, trg_tf_ids, nsyns_ret, synapsis_params):

        tar_tuning = self.target_network['tuning_angle'][trg_tf_ids]
        src_tuning = self.source_network['tuning_angle'][src_tf_ids]

        sigma = synapsis_params['weight_sigma']
        syn_weight = synapsis_params['syn_weight']

        w_multiplier_180 = direction_tuning_factor(
            src_tuning, tar_tuning, sigma)

        return syn_weight * w_multiplier_180 * nsyns_ret

    def assign_weight_and_delay(self, source, target, source_tf_ids, target_tf_ids, nsyns_ret):
        """This function determined which nodes are connected based on the parameters in the dictionary params. The
        function iterates through every cell pair when called and hence no for loop is seen iterating pairwise
        although this is still happening.

        By iterating though every cell pair, a decision is made whether or not two cells are connected and this
        information is returned by the function. This function calculates these probabilities based on the distance between
        two nodes and (if applicable) the orientation tuning angle difference.

        :param source: the pop_name of the source
        :param target: the pop_name of the target
        :param source_network: the network dict of the source area
        :param target_network: the network dict of the target area
        :param params: parameters dictionary for probability of connection (see function: compute_pair_type_parameters)
        :return: if two cells are deemed to be connected, the return function automatically returns the source id
                 and the target id for that connection. The code further returns the number of synapses between
                 those two neurons
        """
        tgt_query_ids = self.target_network['node_type_id'][target_tf_ids]
        src_query_ids = self.source_network['node_type_id'][source_tf_ids]
        src_query = [self.node_type_id_to_pop_name_dict[node_id]
                     for node_id in src_query_ids]

        synapsis_params = {k: [] for k in self.edge_params_keys}
        for tgt_q, src_q in zip(tgt_query_ids, src_query):
            edge_parameters = self.edges_params[tgt_q][src_q]
            for k, v in edge_parameters.items():
                synapsis_params[k].append(v)

        json_to_receptor_type = {'e2i.json': 3,
                                 'e2e.json': 1, 'i2e.json': 2, 'i2i.json': 4}
        receptor_types = [json_to_receptor_type.get(
            key) for key in synapsis_params['dynamics_params']]

        if ('e' in source) and ('e' in target):
            synaptic_weights = self.DirectionRule_EE(source_tf_ids, target_tf_ids,
                                                     self.sources_projected_x, self.sources_projected_z,
                                                     nsyns_ret, synapsis_params)
        else:
            synaptic_weights = self.DirectionRule_others(source_tf_ids, target_tf_ids,
                                                         nsyns_ret, synapsis_params)

        return synaptic_weights, synapsis_params['delay'], receptor_types

    def connect_cells(self, source, target, params):
        """This function determined which nodes are connected based on the parameters in the dictionary params. The
        function iterates through every cell pair when called and hence no for loop is seen iterating pairwise
        although this is still happening.

        By iterating though every cell pair, a decision is made whether or not two cells are connected and this
        information is returned by the function. This function calculates these probabilities based on the distance between
        two nodes and (if applicable) the orientation tuning angle difference.

        :param source: the pop_name of the source
        :param target: the pop_name of the target
        :param params: parameters dictionary for probability of connection (see function: compute_pair_type_parameters)
        :return: if two cells are deemed to be connected, the return function automatically returns the source id
                 and the target id for that connection. The code further returns the number of synapses between
                 those two neurons
        """
        target = '_'.join(target.split('_')[1:])
        source = '_'.join(source.split('_')[1:])

        # Get the ids of all the source and target neurons of a given neuron pair type
        target_ids = self.target_network['laminar_indices'][target]
        source_ids = self.source_network['laminar_indices'][source]

        # Get the number of neurons in every population
        num_neuron_target = target_ids.size
        num_neuron_source = source_ids.size

        # Get the coordinates and tuning angle of every neuron
        targets_x = self.target_network['x'][target_ids]
        targets_z = self.target_network['z'][target_ids]
        targets_tuning_angle = self.target_network['tuning_angle'][target_ids]

        sources_x = self.source_network['x'][source_ids]
        sources_z = self.source_network['z'][source_ids]
        sources_tuning_angle = self.source_network['tuning_angle'][source_ids]

        # Project each neuron of the source area into its retinotopic location in the target area according
        # to the cylindrical form of the column.
        sources_r, sources_phi = cart2pol(sources_x, sources_z)
        sources_projected_x, sources_projected_z = pol2cart(
            sources_r*self.radius_ratio, sources_phi)

        # Replicate the sources and targets features for every possible pair
        sources_projected_x = np.repeat(sources_projected_x, num_neuron_target)
        sources_projected_z = np.repeat(sources_projected_z, num_neuron_target)
        sources_tuning_angle = np.repeat(
            sources_tuning_angle, num_neuron_target)

        targets_x = np.tile(targets_x, num_neuron_source)
        targets_z = np.tile(targets_z, num_neuron_source)
        targets_tuning_angle = np.tile(targets_tuning_angle, num_neuron_source)

        # Read parameter values needed for distance and orientation dependence
        A_0 = params['A_0']
        sigma = params['sigma']
        gradient = params['gradient']
        intercept = params['intercept']
        nsyn_range = params['nsyn_range']

        # Calculate the planar intersomatic distance between the current two cells (in 2D - not including depth)
        planar_intersomatic_distance = intersomatic_distance(
            sources_projected_x, targets_x, sources_projected_z, targets_z)

        # Check if there is orientation dependence
        if not np.isnan(gradient):
            # Calculate the difference in orientation tuning between the cells
            delta_orientation = np.array(
                sources_tuning_angle, dtype=float) - np.array(targets_tuning_angle, dtype=float)

            # For OSI, convert to quadrant from 0 - 90 degrees
            delta_orientation = abs(
                abs(abs(180.0 - abs(delta_orientation)) - 90.0) - 90.0)

            # Calculate the probability two cells are connected based on distance and orientation
            p_connect = gaussian_decay(planar_intersomatic_distance, A_0, sigma)\
                * (intercept + gradient * delta_orientation)

        # If no orientation dependence
        else:
            # Calculate the probability two cells are connection based on distance only
            p_connect = gaussian_decay(
                planar_intersomatic_distance, A_0, sigma)

        # # Sanity check warning
        # if p_connect > 1:
        #     print 'WARNING WARNING WARNING: p_connect is greater that 1.0 it is: ', p_connect

        # Set connections
        num_full_connections = num_neuron_target * num_neuron_source
        all_conn_idcs = np.arange(num_full_connections, dtype=np.int32)

        # Decide which cells get a connection based on the p_connect value calculated
        p_connected = self.rd.binomial(1, p_connect)
        synapsis_mask = p_connected.astype(np.bool_)

        # Assign the number of synapses for every connection
        nsyns_ret = np.copy(p_connected)
        nsyns_ret[nsyns_ret == 1] = self.rd.randint(
            nsyn_range[0], nsyn_range[1], len(nsyns_ret[nsyns_ret == 1]))

    #     synapses_mask = target_source_conn_prob > rd.uniform(low=0, high=1, size=num_full_connections)

        n_connections = int(np.sum(p_connected))
        n_synapses = int(np.sum(nsyns_ret))
        if n_connections != 0:
            print(
                f'> {self.source_column_name}-{source} to {self.target_column_name}-{target} Number of Connections: {n_connections}')
            # print(f'> {self.source_column_name}-{source} to {self.target_column_name}-{target} Number of Synapses: {n_synapses}\n')

        # Get the neuron indices for every connection pair
        conn_idcs = all_conn_idcs[synapsis_mask]
        self.sources_projected_x = sources_projected_x[synapsis_mask]
        self.sources_projected_z = sources_projected_z[synapsis_mask]

        src_idcs = np.int64(conn_idcs / num_neuron_target)  # \in [0, n_in-1]
        tgt_idcs = conn_idcs % num_neuron_target  # \in [0, n_out-1]

        target_tf_ids = target_ids[tgt_idcs]
        source_tf_ids = source_ids[src_idcs]

        nsyns_ret = nsyns_ret[synapsis_mask]
    #     nsyns_ret = [Nsyn if Nsyn != 0 else None for Nsyn in nsyns_ret]

        return source_tf_ids, target_tf_ids, nsyns_ret

    def __call__(self):
        target_tf_ids = []
        source_tf_ids = []
        interarea_weights = []
        interarea_delays = []
        interarea_receptor_types = []

        from time import time

        for source in self.interarea_connectivity.keys():
            for target in self.interarea_connectivity[source].keys():
                t0 = time()
                connectivity_params = self.compute_pair_type_parameters(
                    source, target)
                src_tf_ids, tgt_tf_ids, nsyns_ret = self.connect_cells(
                    source, target, connectivity_params)

                target_tf_ids.append(tgt_tf_ids*4)
                source_tf_ids.append(src_tf_ids)

                # Set synaptic weights and delays
                n_connections = len(target_tf_ids)
                if self.interarea_weight_distribution == 'levy_stable':
                    # alpha, betas were got from fitting recurrent weights within a column
                    random_weights = levy_stable.rvs(
                        0.6, 0, size=n_connections*10)
                    random_weights = random_weights[random_weights > 0]
                elif self.interarea_weight_distribution == 'normal':
                    # mean and std were got from fitting recurrent weights within a column
                    random_weights = self.rd.randn(n_connections*10) + 10
                    random_weights = random_weights[random_weights > 0]
                elif self.interarea_weight_distribution == 'billeh_like':
                    synaptic_weights, synaptic_delays, receptor_types = self.assign_weight_and_delay(source, target,
                                                                                                     src_tf_ids, tgt_tf_ids,
                                                                                                     nsyns_ret)
                interarea_weights.append(synaptic_weights)
                interarea_delays.append(synaptic_delays)
                interarea_receptor_types.append(receptor_types)

                print(f'{source} to {target} connection time: {time()-t0}.')
                t0 = time()

        if target_tf_ids and source_tf_ids:  # if both the lists are not empty
            # identify target receptor
            # Indentify the which of the 4 types of inputs we have
            r = np.concatenate(interarea_receptor_types) - 1
            indices = np.array([np.concatenate(
                target_tf_ids) + r, np.concatenate(source_tf_ids)], dtype=np.int64).T

            # (all neurons in target column*n_receptors, all neurons in source column)
            dense_shape = (
                self.target_network['n_nodes']*4, self.source_network['n_nodes'])
            max_tgt_ind, max_src_ind = indices.max(axis=0)
            # check legal indices
            assert max_tgt_ind <= self.target_network['n_nodes'] * \
                4, 'wrong inter-area indices from target!'
            assert max_src_ind <= self.source_network['n_nodes'], 'wrong inter-area indices from source!'
            # weights
            interarea_weights = np.concatenate(interarea_weights)
            # delays
            interarea_delays = np.concatenate(interarea_delays)

            # rand_delays = rd.randint(low=inter_area_min_delay, high=inter_area_max_delay, size=interarea_weights.shape)
            indices, interarea_weights, interarea_delays = sort_indices(
                indices, interarea_weights, interarea_delays)

        else:
            indices, interarea_weights, dense_shape, interarea_delays = None, None, None, np.ones(
                1)  # max_delay in model.py needs a non- None type

        self.target_network['interarea_synapses'] = dict()
        self.target_network['interarea_synapses'][self.source_column_name] = dict(
        )
        self.target_network['interarea_synapses'][self.source_column_name]['indices'] = indices
        self.target_network['interarea_synapses'][self.source_column_name]['weights'] = interarea_weights
        self.target_network['interarea_synapses'][self.source_column_name]['delays'] = interarea_delays
        self.target_network['interarea_synapses'][self.source_column_name]['dense_shape'] = dense_shape

        print(
            f'Number of connections from {self.source_column_name} to {self.target_column_name}: {len(interarea_weights)}')

        return self.target_network


def load_billeh(flags, n_neurons):

    networks = dict()
    bkg_weights = dict()

    # initialize every area
    for column_name in ['V1', 'LM']:
        df = pd.read_csv(os.path.join(
            flags.data_dir, f'network/{column_name}_node_types.csv'), delimiter=' ')

        print(f'{column_name} column')
        networks[column_name] = load_network(
            path=os.path.join(
                flags.data_dir, f'{column_name}_network_dat.pkl'),
            h5_path=os.path.join(flags.data_dir, f'network/{column_name}_nodes.h5'), core_only=flags.core_only, n_neurons=n_neurons[column_name],
            seed=flags.seed, connected_selection=flags.connected_selection)
        networks[column_name] = set_laminar_indices(df, networks[column_name])

        inputs = load_input(
            start=1000, duration=1000, dt=1, path=os.path.join(flags.data_dir, 'input_dat.pkl'),
            bmtk_id_to_tf_id=networks[column_name]['bmtk_id_to_tf_id'])

        if column_name == 'V1':
            input_population = inputs[0]  # LGN input direct to V1

        # contains the single background node that projects to all V1 neurons
        bkg = inputs[1]
        bkg_weights[column_name] = np.zeros(
            (networks[column_name]['n_nodes'] * 4,), np.float32)
        bkg_weights[column_name][bkg['indices'][:, 0]] = bkg['weights']

    for target_column_name in ['V1', 'LM']:
        for source_column_name in ['V1', 'LM']:
            if target_column_name != source_column_name:
                source_target_connectivity = InterareaConnectivity(networks[target_column_name], networks[source_column_name],
                                                                   target_column_name, source_column_name,
                                                                   interarea_weight_distribution=flags.interarea_weight_distribution,
                                                                   seed=flags.seed, data_dir=flags.data_dir)
                networks[target_column_name] = source_target_connectivity()

    n_input = flags.n_input
    if n_input != 17400:
        input_population = reduce_input_population(
            input_population, n_input, seed=flags.seed)

    # n_abstract_output = networks['LM']['laminar_indices']['L5e'].size
    # n_completed_output = networks['V1']['laminar_indices']['L5e'].size

    # , n_abstract_output, n_completed_output
    return input_population, networks, bkg_weights, n_input


# If the model already exist we can load it, or if it does not just save it for future occasions
def cached_load_billeh(flags, n_neurons):
    store = False
    # input_population, networks, bkg_weights, n_input, n_abstract_output, n_completed_output  = None, None, None, None, None, None
    input_population, networks, bkg_weights, n_input = None, None, None, None  # , None, None
    # flag_str = f'ratio{flags.area_neuron_ratio}_rec{flags.neurons}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}'
    V1_neurons = n_neurons['V1']
    LM_neurons = n_neurons['LM']
    flag_str = f'V1_{V1_neurons}_LM_{LM_neurons}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}'
    file_dir = os.path.split(__file__)[0]
    cache_path = os.path.join(
        file_dir, f'.cache/LM_V1_network_{flag_str}.pkl')
    # os.remove(cache_path)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                input_population, networks, bkg_weights, n_input = pkl.load(f)
                print(f'> Sucessfully restored Billeh model from {cache_path}')
        except Exception as e:
            print(e)
            store = True
    else:
        store = True
    # if input_population is None or networks is None or bkg_weights is None or n_input is None or n_abstract_output is None or n_completed_output is None:
    if input_population is None or networks is None or bkg_weights is None or n_input is None:
        # input_population, networks, bkg_weights, n_input, n_abstract_output, n_completed_output = load_billeh(
        #     flags=flags, n_neurons=n_neurons)
        input_population, networks, bkg_weights, n_input = load_billeh(
            flags=flags, n_neurons=n_neurons)
    if store:
        os.makedirs(os.path.join(file_dir, '.cache'), exist_ok=True)
        with open(cache_path, 'wb') as f:
            # pkl.dump((input_population, networks, bkg_weights, n_input, n_abstract_output, n_completed_output), f)
            pkl.dump((input_population, networks, bkg_weights, n_input), f)
        print(f'> Cached LM_V1 model in {cache_path}')
    # , n_abstract_output, n_completed_output
    return input_population, networks, bkg_weights, n_input

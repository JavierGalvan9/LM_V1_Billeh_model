import os
import h5py
import numpy as np
import pandas as pd
import pickle as pkl
from Model_utils.network_builder import build_network_dat, build_input_dat, sort_indices, sort_indices_tf
from Model_utils.interarea_connectivity_builder import InterareaConnectivity
from time import time

np.random.seed(0)


def load_network(path="GLIF_network/v1_network_dat.pkl",
                 h5_path="GLIF_network/network/v1_nodes.h5",
                 core_only=True,
                 n_neurons=51978, 
                 seed=3000, 
                 connected_selection=True,
                 column_name='v1',
                 tensorflow_speed_up=False,
                 random_weights=False):
    
    rd = np.random.RandomState(seed=seed)

    # Create / Load the network_dat pickle file from the SONATA files
    if not os.path.exists(path):
        print(f"Creating {column_name}_network_dat.pkl file...")
        d = build_network_dat(data_dir='GLIF_network', source=column_name, target=column_name)
    else:
        print(f"Loading {column_name}_network_dat.pkl file...")
        with open(path, "rb") as f:
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
    # Create arrays to convert between bmtk and tf ides
    tf_id_to_bmtk_id = np.arange(n_nodes, dtype=np.int32)
    bmtk_id_to_tf_id = np.full(n_nodes, -1, dtype=np.int32)

    # Extract from the SONATA file the nodes information
    # h5_file = h5py.File(h5_path, "r")
    with h5py.File(h5_path, "r") as h5_file:
        assert np.diff(h5_file["nodes"]["v1"]["node_id"]).var() < 1e-12
        x, y, z = [h5_file["nodes"]['v1']["0"][dim][:].astype(np.float32) for dim in ["x", "y", "z"]]
        tuning_angle = h5_file['nodes']['v1']['0']['tuning_angle'][()].astype(np.float32)
        node_type_id = h5_file['nodes']['v1']['node_type_id'][()].astype(np.int32)
        r = np.sqrt(x**2 + z**2)  # the maximum radius is 845

    ### CHOOSE THE NETWORK NODES ###
    if n_neurons > n_nodes:
        raise ValueError(f"There are only {n_nodes} neurons in the network")
    
    # sel is a boolean array with True value in the indices of selected neurons
    elif connected_selection:  # this condition takes the n_neurons closest neurons
        # order according to radius distance. This option allows for more synapses since close neurons are more probably connected
        sorted_ind = np.argsort(r)
        sel = np.zeros(n_nodes, dtype=np.bool_)
        sel[sorted_ind[:n_neurons]] = True  # keep only the nearest n_neurons
        print(f'> Maximum sample radius: {r[sorted_ind[n_neurons - 1]]:.2f}')
    # this condition makes all the neurons to be within a given distance from the origin (core)
    elif core_only:
        # 51,978 maximum value for n_neurons in v1 in this case
        if column_name == 'v1':
            core_radius = 400
        elif column_name == 'lm':
            v1_to_lm_neurons_ratio = 7.010391285652859 # based on BlueBrain Project data
            core_radius = 400/np.sqrt(v1_to_lm_neurons_ratio)
        else:
            raise ValueError('Column name not recognized')

        sel = r < core_radius
        if n_neurons > 51978:
            raise ValueError("There are only 51978 neurons in the network core")
        
        elif n_neurons > 0 and n_neurons <= 51978:
            (inds,) = np.where(sel)  # indices where the condition is satisfied
            take_inds = rd.choice(inds, size=n_neurons, replace=False)
            sel[:] = False
            sel[take_inds] = True

    # Choose n_neurons random neurons without any other requirement
    elif n_neurons > 0 and n_neurons <= n_nodes:  # this condition takes random neurons from all the v1
        legit_neurons = np.arange(n_nodes)
        take_inds = rd.choice(legit_neurons, size=n_neurons, replace=False)
        sel = np.empty(n_nodes, dtype=np.bool_)
        sel[take_inds] = True

    else:  # this condition takes all the neurons
        sel = np.ones(n_nodes, dtype=np.bool_)

    # Get the number of neurons in the chosen network and update the traslation arrays
    n_nodes = np.sum(sel)  # number of nodes selected
    print(f"> Number of Neurons: {n_nodes}")
    # tf idx '0' corresponds to 'tf_id_to_bmtk_id[0]' bmtk idx
    tf_id_to_bmtk_id = tf_id_to_bmtk_id[sel]
    bmtk_id_to_tf_id[tf_id_to_bmtk_id] = np.arange(n_nodes, dtype=np.int32) 
    # bmtk idx '0' corresponds to 'bmtk_id_to_tf_id[0]' tf idx which can be '-1' in case
    # the bmtk node is not in the tensorflow selection or another value in case it belongs the selection
    
    # Get the properties from the network neurons
    x = x[sel]
    y = y[sel]
    z = z[sel]
    tuning_angle = tuning_angle[sel]
    node_type_id = node_type_id[sel]

    # GET THE NODES PARAMETERS
    n_node_types = len(d["nodes"])
    node_params = dict(
        V_th=np.empty(n_node_types, np.float32),
        g=np.empty(n_node_types, np.float32),
        E_L=np.empty(n_node_types, np.float32),
        k=np.empty((n_node_types, 2), np.float32),
        C_m=np.empty(n_node_types, np.float32),
        V_reset=np.empty(n_node_types, np.float32),
        t_ref=np.empty(n_node_types, np.float32),
        asc_amps=np.empty((n_node_types, 2), np.float32),
    )

    node_type_ids = np.empty(n_nodes, np.int32)
    for i, node_type in enumerate(d["nodes"]):
        # get ALL the nodes of the given node type
        tf_ids = bmtk_id_to_tf_id[np.array(node_type["ids"], dtype=np.int32)]
        # choose only those that belong to our model
        tf_ids = tf_ids[tf_ids >= 0]
        # assign them all the same id (which does not relate with the neuron type)
        node_type_ids[tf_ids] = i
        for k, v in node_params.items():
            # save in a dict the information of the nodes
            v[i] = node_type["params"][k]

    # GET THE EDGES INFORMATION
    t0 = time()
    edges = d["edges"]
    n_edges = 0
    dense_shape = (n_nodes, n_nodes)
    # each node has 4 different inputs (soma, dendrites, etc) with different properties each
    # dense_shape = (4 * n_nodes, n_nodes)
    indices = []
    weights = []
    delays = []
    receptor_ids = [] # [0,1,2,3] possible values
    edge_type_ids = []
    
    for edge in edges:
        edge_type_id = edge["edge_type_id"]
        target_tf_ids = bmtk_id_to_tf_id[edge["target"]]
        source_tf_ids = bmtk_id_to_tf_id[edge["source"]]
        edge_exists = np.logical_and(target_tf_ids != -1, source_tf_ids != -1)
        # select the edges within our model
        target_tf_ids = target_tf_ids[edge_exists]
        source_tf_ids = source_tf_ids[edge_exists]
        weights_tf = edge["params"]["weight"][edge_exists]
        # if random weights, we assign random values to the weights maintaining the sign, the mean and the std of the original weights
        if random_weights:
            np.random.shuffle(weights_tf)
            # weights_tf = np.sign(weights_tf) * np.abs(rd.normal(loc=np.mean(weights_tf), scale=np.std(weights_tf), size=weights_tf.size))

        n_new_edge = len(target_tf_ids)
        n_edges += int(n_new_edge)
        # Identify which of the 4 types of receptor types we have. r takes values within 0 - 3
        r = edge["params"]["receptor_type"] - 1
        # indices[current_edge:current_edge + n_new_edge] = np.array([target_tf_ids * 4 + r, source_tf_ids]).T

        # all the edges of a given type have the same delay and synaptic id
        delays_tf = np.full(n_new_edge, edge["params"]["delay"], dtype=np.float16)
        receptor_id = np.full(n_new_edge, r, dtype=np.uint8)

        indices.append(np.array([target_tf_ids, source_tf_ids]).T)
        weights.append(weights_tf)
        delays.append(delays_tf)
        receptor_ids.append(receptor_id)
        edge_type_ids.append(np.full(n_new_edge, edge_type_id, dtype=np.uint16))

    print(f'> Number of Synapses: {n_edges}')
    indices = np.concatenate(indices, axis=0, dtype=np.int32)
    weights = np.concatenate(weights, axis=0, dtype=np.float32)
    delays = np.concatenate(delays, axis=0, dtype=np.float16)
    receptor_ids = np.concatenate(receptor_ids, axis=0, dtype=np.uint8)
    edge_type_ids = np.concatenate(edge_type_ids, axis=0, dtype=np.uint16)

    # sort indices by considering first all the targets of node 0, then all of node 1, ...
    if tensorflow_speed_up:
        # indices, weights, delays = sort_indices_tf(indices, weights, delays)
        indices, weights, delays, receptor_ids, edge_type_ids = sort_indices_tf(indices, weights, delays, receptor_ids, edge_type_ids)
    else:
        # indices, weights, delays = sort_indices(indices, weights, delays)
        indices, weights, delays, receptor_ids, edge_type_ids = sort_indices(indices, weights, delays, receptor_ids, edge_type_ids)

    network = dict(
        x=x, y=y, z=z,
        tuning_angle=tuning_angle,
        node_type_id=node_type_id,
        n_nodes=n_nodes,
        n_edges=n_edges,
        node_params=node_params,
        node_type_ids=node_type_ids,
        synapses=dict(indices=indices, weights=weights, delays=delays, 
                      receptor_ids=receptor_ids, edge_type_ids=edge_type_ids,
                      dense_shape=dense_shape),
        tf_id_to_bmtk_id=tf_id_to_bmtk_id,
        bmtk_id_to_tf_id=bmtk_id_to_tf_id,
    )

    return network


def set_laminar_indices(network, column_name='v1', data_dir='GLIF_network'):
    df = pd.read_csv(os.path.join(data_dir, f'network/{column_name}_node_types.csv'), delimiter=' ')

    def get_one_layer_indices(EI, layer_number, neuron_subpop=None):
        # Pre-filter based on EI to reduce dataframe size early on
        if EI == 'e':
            filtered_df = df[df['pop_name'].str.startswith(f'e{layer_number}')]
            types_indices = filtered_df['node_type_id'].values
        elif EI == 'i':
            # Include the subpopulation in the filter condition
            condition = df['pop_name'].str.startswith(f'i{layer_number}') & df['pop_name'].str.contains(neuron_subpop, na=False)
            filtered_df = df[condition]
            types_indices = filtered_df['node_type_id'].values
        else:
            print('Error: Wrong population name')
            return np.array([], dtype=np.int32)
        
        # Vectorized operation for selecting neurons
        neuron_sel = np.isin(network['node_type_id'], types_indices)
        layer_indices = np.where(neuron_sel)[0].astype(np.int32)

        return layer_indices

    network['laminar_indices'] = dict()
    for layer_number in [1, 2, 4, 5, 6]:
        network['laminar_indices'][f'L{layer_number}_e'] = get_one_layer_indices('e', layer_number)
        for neuron_subpop in ['Pvalb', 'Sst', 'Htr3a']:
            network['laminar_indices'][f'L{layer_number}_i_{neuron_subpop}'] = get_one_layer_indices('i', layer_number, neuron_subpop)

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


# Here we load the 17400 neurons that act as input in the model
def load_input(column_name='v1', 
               source='lgn',
               input_dat_path="GLIF_network/bkg_v1_network_dat.pkl",
               bmtk_id_to_tf_id=None,
               tensorflow_speed_up=False,
            #    start=0,
            #    duration=3000,
            #    dt=1
               ):

    # LOAD THE BACKGROUND INPUT
    if not os.path.exists(input_dat_path):
        print(f"Creating {source}_{column_name}_network_dat.pkl file...")
        # Process BKG input network
        input_dat = build_input_dat(data_dir='GLIF_network', source=source, target=column_name)
        print("Done.")
    else:
        with open(input_dat_path, "rb") as f:
            input_dat = pkl.load(f)

    # Unite the edges information of the LGN and the background noise
    input_edges = input_dat['edges']
    
    post_indices = []
    pre_indices = []
    weights = []
    delays = []
    receptor_ids = []

    for edge in input_edges:
        # Indentify the which of the 4 types of inputs we have
        r = edge['params']['receptor_type'] - 1
        # r takes values whithin 0 - 3
        target_bmtk_id = edge["target"].astype(np.int32)
        source_tf_id = edge["source"].astype(np.int32)
        weights_tf = edge["params"]["weight"].astype(np.float32)

        n_new_edge = len(target_bmtk_id)
        delays_tf = np.full(n_new_edge, edge["params"]["delay"], dtype=np.float16)
        receptor_id = np.full(n_new_edge, r, dtype=np.uint8)

        # check if the given edges exist in our model
        # (notice that only the target must exist since the source is within the LGN module)
        # This means that source index is within 0-17400
        if bmtk_id_to_tf_id is not None:
            target_tf_id = bmtk_id_to_tf_id[target_bmtk_id]
            edge_exists = target_tf_id >= 0
            target_tf_id = target_tf_id[edge_exists]
            source_tf_id = source_tf_id[edge_exists]
            weights_tf = weights_tf[edge_exists]
            delays_tf = delays_tf[edge_exists]
            receptor_id = receptor_id[edge_exists]
            # we multiply by 4 the indices and add r to identify the receptor_type easily:
            # if target id is divisible by 4 the receptor_type is 0,
            # if it is rest is 1 by dividing by 4 then its receptor type is 1, and so on...
            # extend acts by extending the list with the given object
            # post_indices.extend(4 * target_tf_id + r)
            # pre_indices.extend(source_tf_id)
            # weights.extend(weights_tf)
            # delays.append(delays_tf)
            # new_target_tf_id = target_tf_id * max_n_receptors + r
            new_target_tf_id = target_tf_id
            post_indices.append(new_target_tf_id)
            pre_indices.append(source_tf_id)
            weights.append(weights_tf)
            delays.append(delays_tf)
            receptor_ids.append(receptor_id)


    post_indices = np.concatenate(post_indices, axis=0, dtype=np.int32)
    pre_indices = np.concatenate(pre_indices, axis=0, dtype=np.int32)
    weights = np.concatenate(weights, axis=0, dtype=np.float32)
    delays = np.concatenate(delays, axis=0, dtype=np.float16)
    receptor_ids = np.concatenate(receptor_ids, axis=0, dtype=np.uint8)

    # first column are the post indices and second column the pre indices
    indices = np.stack([post_indices, pre_indices], -1)

    # Sort indices
    # indices, weights, delays, syn_ids = sort_indices_tf(indices, weights, delays, syn_ids)
    if tensorflow_speed_up:
        # indices, weights, delays = sort_indices_tf(indices, weights, delays)
        indices, weights, delays, receptor_ids = sort_indices_tf(indices, weights, delays, receptor_ids)
    else:
        # indices, weights, delays = sort_indices(indices, weights, delays)
        indices, weights, delays, receptor_ids = sort_indices(indices, weights, delays, receptor_ids)

    # n_inputs = len(set(pre_indices))
    # print('N_inputs: ', n_inputs)
    # we load the background nodes and their positions
    nodes_h5_file = h5py.File(f"GLIF_network/network/{source}_nodes.h5", "r")
    n_inputs = len(nodes_h5_file["nodes"][source]["node_id"])

    input_population = dict(
                            n_inputs=n_inputs,
                            indices=indices,
                            weights=weights,
                            delays=delays,
                            receptor_ids=receptor_ids,
        )

    # n_input_neurons = len(input_population[0]['ids'])  # 17400
    # spikes = np.zeros((int(duration / dt), n_input_neurons))
    
    # # now we save the spikes of the input population from the BMTK simulation
    # for i, sp in zip(input_population[0]['ids'], input_population[0]['spikes']):
    #     # consider only the spikes within the period we are taking
    #     sp = sp[np.logical_and(start < sp, sp < start + duration)] - start
    #     sp = (sp / dt).astype(np.int)
    #     for s in sp:
    #         spikes[s, i] += 1
    # input_populations.append(
    #     dict(n_inputs=n_input_neurons, indices=indices.astype(
    #     np.int64), weights=weights, delays=delays, spikes=spikes))

    return input_population

def reduce_input_population(input_population, new_n_input, seed=3000):
    # This is not optimal since we are randonmly taking and adding the effect of different LGN units, 
    # without considering their spatial arrangement and connectivity particularities
    rd = np.random.RandomState(seed=seed)

    in_ind = input_population['indices']
    in_weights = input_population['weights']
    in_delays = input_population['delays']
    in_receptor_ids = input_population["receptor_ids"]

    # we take input_population['n_inputs'] neurons from a list of new_n_input with replace,
    # which means that repeated LGN neurons double their synaptic weights, ending with less
    # units than new_n_input
    assignment = rd.choice(np.arange(new_n_input, dtype=np.int32), 
                           size=input_population['n_inputs'], replace=True)

    # Create a tuple of indices for vectorized operations
    post_indices = in_ind[:, 0]
    input_neuron_indices = in_ind[:, 1]
    assigned_neuron_indices = assignment[input_neuron_indices]

    # Calculate unique pairs and their indices in the original array
    unique_pairs, inverse_indices = np.unique(np.stack((post_indices, assigned_neuron_indices), axis=1), axis=0, return_inverse=True)

    # Accumulate weights for repeated pairs
    accumulated_weights = np.bincount(inverse_indices, weights=in_weights)

    # Lets get the average delay by the connection strength (synaptic weight), 
    # assuming stronger connections might have a more significant impact on the timing.
    # Calculate mean delays for each unique pair
    # First, accumulate the total delays for each unique pair
    total_delays = np.bincount(inverse_indices, weights=in_delays)
    # Count the occurrences of each unique pair to divide and get the mean
    counts = np.bincount(inverse_indices)
    # Calculate the mean by dividing total delays by counts
    mean_delays = total_delays / counts

    # same for the receptor_ids since all of them are excitatory
    total_receptor_ids = np.bincount(inverse_indices, weights=in_receptor_ids)
    mean_receptor_ids = total_receptor_ids // counts

    # Sort the data
    new_in_ind, new_in_weights, new_in_delays, new_in_receptor_ids = sort_indices(
        unique_pairs, accumulated_weights, mean_delays, mean_receptor_ids)

    new_input_population = {
        'n_inputs': new_n_input,
        'indices': new_in_ind.astype(np.int32),
        'weights': new_in_weights.astype(np.float32),
        'delays': new_in_delays.astype(np.float16),
        'receptor_ids': new_in_receptor_ids.astype(np.uint8),
        'spikes': None
    }

    return new_input_population

def load_billeh(flags, n_neurons, flag_str=''):
    # flag_str is an argument only used for the cached_load_billeh network
    networks = dict()
    lgn_inputs = dict()
    bkg_inputs = dict()
    # inputs = dict()
    # bkg_weights = dict()

    # initialize every area
    for column_name in ['v1', 'lm']:

        print(f'{column_name} column')
        networks[column_name] = load_network(
                                            path=os.path.join(flags.data_dir, f'{column_name}_network_dat.pkl'),
                                            h5_path=os.path.join(flags.data_dir, f'network/{column_name}_nodes.h5'), 
                                            core_only=flags.core_only, 
                                            n_neurons=n_neurons[column_name],
                                            seed=flags.seed, 
                                            connected_selection=flags.connected_selection,
                                            column_name=column_name,
                                            tensorflow_speed_up=False,
                                            random_weights=flags.random_weights)
        networks[column_name] = set_laminar_indices(networks[column_name], column_name=column_name, data_dir = flags.data_dir)

        ##### Select random l5e neurons for tracking output #########
        df = pd.read_csv(os.path.join(
            flags.data_dir, "network/v1_node_types.csv"), delimiter=" ")
        
        l5e_types_indices = []
        for a in df.iterrows():
            if a[1]["pop_name"].startswith("e5"):
                l5e_types_indices.append(a[0])
        l5e_types_indices = np.array(l5e_types_indices)
        l5e_neuron_sel = np.zeros(networks[column_name]["n_nodes"], np.bool_)
        for l5e_type_index in l5e_types_indices:
            is_l5_type = networks[column_name]["node_type_ids"] == l5e_type_index
            l5e_neuron_sel = np.logical_or(l5e_neuron_sel, is_l5_type)
        networks[column_name]["l5e_types"] = l5e_types_indices
        networks[column_name]["l5e_neuron_sel"] = l5e_neuron_sel
        print(f"> Number of L5e Neurons: {np.sum(l5e_neuron_sel)}")

        # assert that you have enough l5 neurons for all the outputs and then choose n_output * neurons_per_output random neurons
        assert np.sum(l5e_neuron_sel) > flags.n_output * flags.neurons_per_output
        rd = np.random.RandomState(seed=flags.seed)
        l5e_neuron_indices = np.where(l5e_neuron_sel)[0]
        readout_neurons = rd.choice(
            l5e_neuron_indices, size=flags.n_output * flags.neurons_per_output, replace=False
        )
        readout_neurons = readout_neurons.reshape((flags.n_output, flags.neurons_per_output))
        for i in range(10):
            # networks[column_name][f'localized_readout_neuron_ids_{i}'] = readout_neurons_random[i,:][None,:]
            networks[column_name][f"readout_neuron_ids_{i}"] = readout_neurons[i,:][None,:]
        #########################################

        if column_name == 'v1':
            lgn_input = load_input(
                                    column_name=column_name, 
                                    source='lgn',
                                    input_dat_path=os.path.join(flags.data_dir, f"lgn_{column_name}_network_dat.pkl"),
                                    bmtk_id_to_tf_id=networks[column_name]['bmtk_id_to_tf_id'],
                                    tensorflow_speed_up=False)
            if flags.n_input != 17400:
                lgn_inputs[column_name] = reduce_input_population(lgn_input, flags.n_input, seed=flags.seed)
            else:
                lgn_inputs[column_name] = lgn_input

        else:
            lgn_inputs[column_name] = None

        
        bkg_input = load_input(
                                column_name=column_name, 
                                source='bkg',
                                input_dat_path=os.path.join(flags.data_dir, f"bkg_{column_name}_network_dat.pkl"),
                                bmtk_id_to_tf_id=networks[column_name]['bmtk_id_to_tf_id'],
                                tensorflow_speed_up=False)

        bkg_inputs[column_name] = bkg_input


        # # contains the single background node that projects to all v1 neurons
        # bkg = inputs[1]
        # bkg_weights[column_name] = np.zeros((networks[column_name]['n_nodes'] * 4,), np.float32)
        # bkg_weights[column_name][bkg['indices'][:, 0]] = bkg['weights']

    for target_column_name in ['v1', 'lm']:
        for source_column_name in ['v1', 'lm']:
            if target_column_name != source_column_name:                
                source_target_connectivity = InterareaConnectivity(networks[target_column_name], networks[source_column_name],
                                                                   target_column_name, source_column_name,
                                                                   interarea_weight_distribution=flags.interarea_weight_distribution,
                                                                   seed=flags.seed, data_dir=flags.data_dir)
                networks[target_column_name] = source_target_connectivity()

    if flags.E4_weight_factor != 1:
        print('E4_weight_factor = ', flags.E4_weight_factor)
        L4_e_ids = networks['lm']['laminar_indices']['L4_e']

        ### Increase recurrent weights to E4 in lm 
        # rec_target_indices = networks['lm']['synapses']['indices'] [:, 0] // 4
        # # Get the recurrent excitatory projections to lm e4 neurons
        # L4_e_rec_exc_edges_mask =  np.logical_and(np.isin(rec_target_indices, L4_e_ids), networks['lm']['synapses']['weights'] > 0)
        # # Increase the weight of the excitatory recurrent projections to lm e4 neurons
        # networks['lm']['synapses']['weights'][L4_e_rec_exc_edges_mask] *= flags.E4_weight_factor

        ### Increase interarea weights to E4 in lm
        inter_target_indices = networks['lm']['interarea_synapses']['v1']['indices'][:, 0] 
        # Get the interarea excitatory projections to lm e4 neurons
        L4_e_inter_exc_edges_mask =  np.logical_and(np.isin(inter_target_indices, L4_e_ids), networks['lm']['interarea_synapses']['v1']['weights'] > 0)
        # Increase the weight of the excitatory interarea projections to lm e4 neurons
        networks['lm']['interarea_synapses']['v1']['weights'][L4_e_inter_exc_edges_mask] *= flags.E4_weight_factor

    # if flags.disconnect_lm_L6_inhibition:
    #     print('disconnect_lm_L6_inhibition = ', flags.disconnect_lm_L6_inhibition)
        
    #     # Get the indices of the lm L6 neurons
    #     L6_Sst_ids = networks['lm']['laminar_indices']['L6_i_Sst']
    #     L6_Htr3a_ids = networks['lm']['laminar_indices']['L6_i_Htr3a']
    #     L6_Pvalb_ids = networks['lm']['laminar_indices']['L6_i_Pvalb']
    #     L6_inh_ids = np.concatenate((L6_Sst_ids, L6_Htr3a_ids, L6_Pvalb_ids))

    #     # Get the recurrent inhibitory projections from lm i6 neurons
    #     rec_source_indices = networks['lm']['synapses']['indices'][:, 1]
    #     L6_inh_edges_mask = np.isin(rec_source_indices, L6_inh_ids)
    #     # Set the weight of the recurrent inhibitory projections from lm i6 neurons to zero
    #     networks['lm']['synapses']['weights'][L6_inh_edges_mask] = 0

    if flags.disconnect_v1_lm_L6_excitatory_projections:
        print('disconnect_v1_lm_L6_excitation = ', flags.disconnect_v1_lm_L6_excitatory_projections)
        
        # Get the indices of the lm L6 neurons
        L6_e_ids = networks['lm']['laminar_indices']['L6_e']
        ### Increase interarea weights to E4 in lm
        inter_target_indices = networks['lm']['interarea_synapses']['v1']['indices'][:, 0] 
        # Get the interarea excitatory projections to lm e4 neurons
        L6_e_inter_exc_edges_mask =  np.logical_and(np.isin(inter_target_indices, L6_e_ids), networks['lm']['interarea_synapses']['v1']['weights'] > 0)
        # Increase the weight of the excitatory interarea projections to lm e4 neurons
        networks['lm']['interarea_synapses']['v1']['weights'][L6_e_inter_exc_edges_mask] = 0

    # n_abstract_output = networks['lm']['laminar_indices']['L5e'].size
    # n_completed_output = networks['v1']['laminar_indices']['L5e'].size
    
    # return input_population, networks, bkg_weights, n_input # , n_abstract_output, n_completed_output
    # return networks, inputs # , n_abstract_output, n_completed_output
    return networks, lgn_inputs, bkg_inputs


# If the model already exist we can load it, or if it does not just save it for future occasions
def cached_load_billeh(flags, n_neurons, flag_str=''):
    store = False
    # input_population, networks, bkg_weights, n_input, n_abstract_output, n_completed_output  = None, None, None, None, None, None
    networks, lgn_inputs, bkg_inputs = None, None, None  # , None, None
    # flag_str = f'ratio{flags.area_neuron_ratio}_rec{flags.neurons}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}'
    v1_neurons = n_neurons['v1']
    lm_neurons = n_neurons['lm']
    if flag_str == '':
        flag_str = f'v1_{v1_neurons}_lm_{lm_neurons}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}_n_input_{flags.n_input}_interarea_weight_distribution_{flags.interarea_weight_distribution}_E4_weight_factor_{flags.E4_weight_factor}_disconnect_v1_lm_L6_excitatory_projections_{flags.disconnect_v1_lm_L6_excitatory_projections}_random_weights_{flags.random_weights}'
    
    file_dir = os.path.split(__file__)[0]
    cache_path = os.path.join(file_dir, f'.cache/lm_v1_network_{flag_str}.pkl')
    print(f"> Looking for cached lm/v1 model in {cache_path}")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                networks, lgn_inputs, bkg_inputs = pkl.load(f)
                print(f'> Sucessfully restored Billeh model from {cache_path}')
        except Exception as e:
            print(e)
            store = True
    else:
        store = True
    # if input_population is None or networks is None or bkg_weights is None or n_input is None or n_abstract_output is None or n_completed_output is None:
    if networks is None or lgn_inputs is None or bkg_inputs is None:
        # input_population, networks, bkg_weights, n_input, n_abstract_output, n_completed_output = load_billeh(
        #     flags=flags, n_neurons=n_neurons)
        networks, lgn_inputs, bkg_inputs = load_billeh(flags=flags, n_neurons=n_neurons)

    if store:
        os.makedirs(os.path.join(file_dir, '.cache'), exist_ok=True)
        with open(cache_path, 'wb') as f:
            # pkl.dump((input_population, networks, bkg_weights, n_input, n_abstract_output, n_completed_output), f)
            pkl.dump((networks, lgn_inputs, bkg_inputs), f)
        print(f'> Cached lm_v1 model in {cache_path}')
    # , n_abstract_output, n_completed_output
    return networks, lgn_inputs, bkg_inputs
import os
import h5py
import json
import numpy as np
import pandas as pd
import pickle as pkl
from numba import njit
from scipy.stats import levy_stable
import logging
from time import time

from Model_utils.network_builder import build_network_dat, build_input_dat, sort_indices, sort_indices_tf
from Model_utils.network_builder import DirectionRule_EE, DirectionRule_others


@njit
def intersomatic_distance(sources_projected_x, targets_x, sources_projected_z, targets_z):
    return np.sqrt((sources_projected_x - targets_x)**2 + (sources_projected_z - targets_z)**2)

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


def node_type_id_to_pop_name(data_dir='GLIF_network'):
    path_to_csv = os.path.join(data_dir, 'network/v1_node_types.csv')
    # Ensure the CSV reading is optimized by specifying usecols if only a subset of columns is needed
    node_types = pd.read_csv(path_to_csv, sep=' ', usecols=['node_type_id', 'pop_name'])
    node_type_id_to_pop_name_dict = node_types.set_index('node_type_id')['pop_name'].to_dict()
    return node_type_id_to_pop_name_dict


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
        target_network_radius = np.sqrt(self.target_network['x']**2 + self.target_network['z']**2).max()
        source_network_radius = np.sqrt(self.source_network['x']**2 + self.source_network['z']**2).max()
        self.radius_ratio = target_network_radius/source_network_radius

        # Create dictionary between node_type_id and their pop names
        self.node_type_id_to_pop_name_dict = node_type_id_to_pop_name()

        # Load the dictionary with the connection probabilities parameters
        interarea_connectivity_path = os.path.join(
            self.data_dir, f'network/{self.source_column_name}_to_{self.target_column_name}_connection_probabilities.json')
        with open(interarea_connectivity_path, 'rb') as f:
            self.interarea_connectivity = json.load(f)

        # Load the CSV with the synaptic information (weights, delays, etc.) for every neuron pair
        edge_types_df = pd.read_csv(os.path.join(self.data_dir, 'network/v1_v1_edge_types.csv'), delimiter=' ')
        edge_types_df['target_query'] = edge_types_df['target_query'].str.slice(start=15, stop=24)
        edge_types_df['target_query'] = edge_types_df['target_query'].astype(np.uint32)
        edge_types_df['source_query'] = edge_types_df['source_query'].str.slice(start=11, stop=-1)

        # Define the path with the information of the synaptic_models
        synaptic_models_path = os.path.join(data_dir, 'components', "synaptic_models")
        # Create a dictionary with the parameters for every edge type
        edge_params_keys = ['syn_weight', 'weight_function', 'weight_sigma', 'delay']
        self.edges_params = {target_query: {} for target_query in set(edge_types_df['target_query'])}
        for index, row in edge_types_df.iterrows():
            self.edges_params[row['target_query']][row['source_query']] = {}
            for key in edge_params_keys:
                self.edges_params[row['target_query']][row['source_query']][key] = row[key]
            # introduce the key , 'dynamics_params'
            synaptic_model = row['dynamics_params']
            with open(os.path.join(synaptic_models_path, synaptic_model)) as f:
                synaptic_model_dict = json.load(f)
                receptor_type = synaptic_model_dict["receptor_type"] - 1
            self.edges_params[row['target_query']][row['source_query']]['receptor_type'] = receptor_type
        # append receptor type to edge_params_keys
        self.edge_params_keys = edge_params_keys + ['receptor_type']

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
        src_query = [self.node_type_id_to_pop_name_dict[node_id] for node_id in src_query_ids]
        tgt_query = [self.node_type_id_to_pop_name_dict[node_id] for node_id in tgt_query_ids]

        synapsis_params = {k: [] for k in self.edge_params_keys}
        for tgt_q, src_q in zip(tgt_query_ids, src_query):
            edge_parameters = self.edges_params[tgt_q][src_q]
            for k, v in edge_parameters.items():
                synapsis_params[k].append(v)
        
        base_weights = synapsis_params['syn_weight']
        base_weights_sigma = synapsis_params['weight_sigma'] 
        delays = synapsis_params['delay'] 
        receptor_ids = synapsis_params['receptor_type']

        # calculate the synaptic weights
        weight_function = list(set(synapsis_params['weight_function']))[0]
        trg_tuning = self.target_network['tuning_angle'][target_tf_ids]
        src_tuning = self.source_network['tuning_angle'][source_tf_ids]
        if weight_function == 'DirectionRule_EE':
            x_trg = self.target_network['x'][target_tf_ids]
            z_trg = self.target_network['z'][target_tf_ids]
            x_src = self.sources_projected_x
            z_src = self.sources_projected_z
            synaptic_weights = DirectionRule_EE(src_tuning, trg_tuning, x_src, z_src, x_trg, z_trg, 
                                                nsyns_ret, base_weights, base_weights_sigma)
        elif weight_function == 'DirectionRule_others':
            trg_tuning = self.target_network['tuning_angle'][target_tf_ids]
            src_tuning = self.source_network['tuning_angle'][source_tf_ids]
            synaptic_weights = DirectionRule_others(src_tuning, trg_tuning,
                                                    nsyns_ret, base_weights, base_weights_sigma)
        else:
            print('There is no correct weight function!')
            return None

        return synaptic_weights, delays, receptor_ids

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

        # Get the coordinates and tuning angle of every neuron for the source and target networks
        targets_x = self.target_network['x'][target_ids]
        targets_z = self.target_network['z'][target_ids]
        targets_tuning_angle = self.target_network['tuning_angle'][target_ids]
        sources_x = self.source_network['x'][source_ids]
        sources_z = self.source_network['z'][source_ids]
        sources_tuning_angle = self.source_network['tuning_angle'][source_ids]

        # Project each neuron of the source area into its retinotopic location in the target area according
        # to the cylindrical form of the column.
        sources_r, sources_phi = cart2pol(sources_x, sources_z)
        sources_projected_x, sources_projected_z = pol2cart(sources_r*self.radius_ratio, sources_phi)

        # Read parameter values needed for distance and orientation dependence
        A_0, sigma = params['A_0'], params['sigma']
        gradient, intercept = params.get('gradient', np.nan), params['intercept']
        nsyn_range = params['nsyn_range']

        # Calculate the planar intersomatic distance between the current two cells (in 2D - not including depth)
        # planar_intersomatic_distance = intersomatic_distance(sources_projected_x, targets_x, sources_projected_z, targets_z)
        # Vectorized distance and orientation calculation
        distance_matrix = intersomatic_distance(sources_projected_x[:, np.newaxis], targets_x, sources_projected_z[:, np.newaxis], targets_z)

        # Check if there is orientation dependence
        if not np.isnan(gradient):
            # Calculate the difference in orientation tuning between the cells
            delta_orientation = np.abs(sources_tuning_angle[:, np.newaxis] - targets_tuning_angle)
            # For OSI, convert to quadrant from 0 - 90 degrees
            delta_orientation = np.abs(np.abs(np.abs(180.0 - delta_orientation) - 90.0) - 90.0)
            # Calculate the probability two cells are connected based on distance and orientation
            p_connect = gaussian_decay(distance_matrix, A_0, sigma) * (intercept + gradient * delta_orientation)
        else: # If no orientation dependence
            # Calculate the probability two cells are connection based on distance only
            p_connect = gaussian_decay(distance_matrix, A_0, sigma)

        # Sanity check warning
        if np.any(p_connect) > 1:
            raise ValueError('ERROR: p_connect > 1.0')

        # Reduce the dtype of p_connect and decide which cells get a connection based on the p_connect value calculated
        p_connect = p_connect.flatten().astype(np.float16)
        p_connected = self.rd.binomial(1, p_connect)
        connections_mask = p_connected.astype(np.bool_)
        
        # synapses_mask = target_source_conn_prob > rd.uniform(low=0, high=1, size=num_full_connections)
        n_connections = np.sum(connections_mask)
    
        # Assign the number of synapses for every connection
        nsyns_ret = np.zeros_like(p_connected, dtype=np.int32)
        nsyns_ret[connections_mask] = self.rd.randint(nsyn_range[0], nsyn_range[1]+1, n_connections)
        n_synapses = np.sum(nsyns_ret)
        if n_connections != 0:
            print(f'> {self.source_column_name}-{source} to {self.target_column_name}-{target} Number of Connections: {n_connections}')
            print(f'{self.source_column_name}-{source} to {self.target_column_name}-{target} Number of Synapses: {n_synapses}')
        else:
            print(f'> {self.source_column_name}-{source} to {self.target_column_name}-{target} No Connections\n')


        # Get the neuron indices for every connection pair
        conn_indices = np.nonzero(connections_mask)[0]
        src_indices = conn_indices // len(target_ids)
        tgt_indices = conn_indices % len(target_ids)
        
        target_tf_ids = target_ids[tgt_indices]
        source_tf_ids = source_ids[src_indices]

        nsyns_ret = nsyns_ret[connections_mask]
    #     nsyns_ret = [Nsyn if Nsyn != 0 else None for Nsyn in nsyns_ret]
        # self.sources_projected_x = sources_projected_x[connections_mask]
        # self.sources_projected_z = sources_projected_z[connections_mask]
        num_neuron_target = target_ids.size
        self.sources_projected_x = np.repeat(sources_projected_x, num_neuron_target).astype(np.float32)[connections_mask]
        self.sources_projected_z = np.repeat(sources_projected_z, num_neuron_target).astype(np.float32)[connections_mask]

        return source_tf_ids, target_tf_ids, nsyns_ret

    def __call__(self):
        target_tf_ids = []
        source_tf_ids = []
        interarea_weights = []
        interarea_delays = []
        interarea_receptor_ids = []

        for source in self.interarea_connectivity.keys():
            for target in self.interarea_connectivity[source].keys():
                t0 = time()
                connectivity_params = self.compute_pair_type_parameters(source, target)
                src_tf_ids, tgt_tf_ids, nsyns_ret = self.connect_cells(source, target, connectivity_params)
                # target_tf_ids.append(tgt_tf_ids*4)
                target_tf_ids.append(tgt_tf_ids)
                source_tf_ids.append(src_tf_ids)

                # Set synaptic weights and delays
                # n_connections = len(target_tf_ids)
                n_connections = len(tgt_tf_ids) 

                if n_connections == 0:
                    # If there are no connections, we don't need to calculate the synaptic weights and delays, so pass to next iteration
                    continue
                else:

                    if self.interarea_weight_distribution == 'levy_stable':
                        # alpha, betas were got from fitting recurrent weights within a column
                        random_weights = levy_stable.rvs(0.6, 0, size=n_connections*10)
                        random_weights = random_weights[random_weights > 0]
                    elif self.interarea_weight_distribution == 'normal':
                        # mean and std were got from fitting recurrent weights within a column
                        random_weights = self.rd.randn(n_connections*10) + 10
                        random_weights = random_weights[random_weights > 0]
                    elif self.interarea_weight_distribution == 'billeh_like':
                        synaptic_weights, synaptic_delays, receptor_ids = self.assign_weight_and_delay(source, target,
                                                                                                        src_tf_ids, tgt_tf_ids,
                                                                                                        nsyns_ret)
                    elif self.interarea_weight_distribution == 'disconnected':
                        synaptic_weights = np.zeros(n_connections)
                        synaptic_delays = np.ones(n_connections)  
                        receptor_ids = np.ones(n_connections)
                    else:
                        raise ValueError('Invalid weight distribution')
                    
                    interarea_weights.append(synaptic_weights)
                    interarea_delays.append(synaptic_delays)
                    interarea_receptor_ids.append(receptor_ids)
    
                    if n_connections != 0: 
                        print(f'{source} to {target} connection time: {round(time()-t0, 2)} s.\n')
                    
                    t0 = time()

        if target_tf_ids and source_tf_ids:  # if both the lists are not empty
            # identify target receptor
            # Indentify the which of the 4 types of inputs we have
            # r = np.concatenate(interarea_receptor_ids) - 1
            # indices = np.array([np.concatenate(target_tf_ids) + r, np.concatenate(source_tf_ids)], dtype=np.int32).T
            indices = np.array([np.concatenate(target_tf_ids), np.concatenate(source_tf_ids)], dtype=np.int32).T
            
            # (all neurons in target column*n_receptors, all neurons in source column)
            # dense_shape = (self.target_network['n_nodes']*4, self.source_network['n_nodes'])
            dense_shape = (self.target_network['n_nodes'], self.source_network['n_nodes'])
            max_tgt_ind, max_src_ind = indices.max(axis=0)
            # check legal indices
            # assert max_tgt_ind <= self.target_network['n_nodes'] * 4, 'wrong inter-area indices from target!'
            assert max_tgt_ind <= self.target_network['n_nodes'], 'wrong inter-area indices from target!'
            assert max_src_ind <= self.source_network['n_nodes'], 'wrong inter-area indices from source!'

            interarea_weights = np.concatenate(interarea_weights)
            interarea_delays = np.concatenate(interarea_delays)
            interarea_receptor_ids = np.concatenate(interarea_receptor_ids)

            # rand_delays = rd.randint(low=inter_area_min_delay, high=inter_area_max_delay, size=interarea_weights.shape)
            indices, interarea_weights, interarea_delays, interarea_receptor_ids = sort_indices(indices, interarea_weights, interarea_delays, interarea_receptor_ids)

        else:
            indices, interarea_weights, dense_shape, interarea_delays, interarea_receptor_ids = None, None, None, np.ones(1), None
            # max_delay in model.py needs a non- None type

        # Create a dictionary to save the interarea connections
        if 'interarea_synapses' not in self.target_network.keys():
            self.target_network['interarea_synapses'] = dict()
            if self.source_column_name not in self.target_network['interarea_synapses'].keys():
                self.target_network['interarea_synapses'][self.source_column_name] = dict()

        self.target_network['interarea_synapses'][self.source_column_name]['indices'] = indices
        self.target_network['interarea_synapses'][self.source_column_name]['weights'] = interarea_weights.astype(np.float32)
        self.target_network['interarea_synapses'][self.source_column_name]['delays'] = interarea_delays.astype(np.float16)
        self.target_network['interarea_synapses'][self.source_column_name]['receptor_ids'] = interarea_receptor_ids.astype(np.uint8)
        self.target_network['interarea_synapses'][self.source_column_name]['dense_shape'] = dense_shape

        print(f'Number of connections from {self.source_column_name} to {self.target_column_name}: {len(interarea_weights)}')

        return self.target_network

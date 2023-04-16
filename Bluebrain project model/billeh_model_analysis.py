# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 23:35:53 2022

@author: UX325
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import h5py
from collections import Counter

parentDir = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management

sys.path.append(os.path.join(parentDir, "billeh_model_utils"))
import other_billeh_utils

random.seed(10)

    
# See inhibitory neurons classification in:
# Tremblay, R., Lee, S., and Rudy, B. (2016). GABAergic interneurons in the neocortex: from cellular properties to circuits. Neuron 91, 260–292.
def standardize_pop_names(pop_name):
    ei = pop_name[0]
    layer = pop_name[1]
    if ei == 'e':
        return '_'.join([f'L{layer}', ei])
    elif ei == 'i':
        if 'Htr3a' in pop_name:
            sub_pop = 'Htr3a'
        elif 'Pvalb' in pop_name:
            sub_pop = 'Pvalb'
        elif 'Sst' in pop_name:
            sub_pop = 'Sst' 
    
        return '_'.join([f'L{layer}', ei, sub_pop])
    

def get_area_stats(array_node_types, area_name, path=''):
    counter = dict(Counter(array_node_types))
    # Sort the population names according to their layer depth
    all_pop_names = sorted(counter.keys())
    sorted_values = [x for _,x in sorted(zip(counter.keys(),counter.values()))]
    
    # Extract simplified pop_names
    layers = []
    pop_names = []
    for key in all_pop_names:
        items = key.split('_')
        if len(items)==2:
            layer, ie = items
            pop_names.append(ie)
        else:
            layer, ie, pop = items
            pop_names.append('_'.join([ie, pop]))
        layers.append(layer)
    
    # Compute the statistics and save the df
    multiindex = list(zip(layers, pop_names))
    index = pd.MultiIndex.from_tuples(multiindex, names=["Layer", "Population"])
    area_pop_stats = pd.DataFrame(sorted_values, index=index, columns=['# neurons'])
    n_neurons_layer = area_pop_stats.groupby(level=0, axis=0).sum()
    area_pop_stats['% layer'] = (area_pop_stats/n_neurons_layer)*100
    area_pop_stats['% total'] = (sorted_values/np.array(sorted_values).sum())*100
    
    area_pop_stats.to_csv(os.path.join(path, f'{area_name}_population_stats.csv'), sep='\t', float_format="%.4f", decimal=',')


    
def get_population_neurons(array_node_types):
    # Get the number of neurons per population
    counter = dict(Counter(array_node_types))
    all_pop_names = sorted(counter.keys())
    sorted_values = [x for _,x in sorted(zip(counter.keys(), counter.values()))]
    counter_df = pd.DataFrame(sorted_values, index=all_pop_names, columns=['# neurons'])
    
    return counter_df


def billeh_connectivity_matrices():
    # Path to save results
    results_path = 'Source_billeh_Target_billeh'
    os.makedirs(results_path, exist_ok=True)
        
    # Load recurrent connectivity from the Billeh model
    network_path = 'Data_billeh'
    recurrent_network = file_management.load_lzma(os.path.join(network_path, 'recurrent_network.lzma'))
    edges_df = recurrent_network[['Source type', 'Target type']]
    
    # Standardize the names of the neuron populations
    edges_df = edges_df.applymap(standardize_pop_names)
    
    # Count number of edges for each pair of source and target population
    short_edges_df = pd.crosstab(edges_df['Source type'], edges_df['Target type'])
    short_edges_df.sort_index(axis=0, inplace=True)
    short_edges_df.sort_index(axis=1, inplace=True)
    
    short_edges_df.to_csv(os.path.join(results_path, 'billeh_to_billeh_total_synapses_matrix.csv'), sep='\t', decimal=',')

    # Load the network of the model and label the neurons with their populations    
    network = file_management.load_lzma(os.path.join(network_path, 'network.lzma'))
    true_pop_names = other_billeh_utils.pop_names(network, data_dir=network_path)
    short_pop_names = [standardize_pop_names(pop_name) for pop_name in true_pop_names]
    
    # Get the abundances of neurons for each class
    get_area_stats(short_pop_names, 'billeh', path=results_path)

    # Obtain the probabilities of connection between neuron pairs
    n_neurons_df = get_population_neurons(short_pop_names)
    n_source = n_neurons_df.values
    n_target = n_neurons_df.transpose().values
    probability_edges_df = short_edges_df.div(n_source, axis=1)
    probability_edges_df = probability_edges_df.div(n_target, axis=1)
    probability_edges_df.index.names = ['Source']

    probability_edges_df.to_csv(os.path.join(results_path, 'billeh_to_billeh_connection_probability_matrix.csv'), sep='\t', decimal=',', index_label='Source')

    # Get the average weights of connection between the different neuron types
    weighted_edges_df = recurrent_network[['Source type', 'Target type', 'Weight']]
    weighted_edges_df[['Source type', 'Target type']] = weighted_edges_df[['Source type', 'Target type']].applymap(standardize_pop_names)
    # average_weighted_edges_df = weighted_edges_df.groupby(['Source type', 'Target type']).mean()
    average_weighted_edges_df = weighted_edges_df.groupby(['Source type', 'Target type']).mean()
    average_weighted_edges_df = average_weighted_edges_df.unstack(fill_value=0)
    col_names = average_weighted_edges_df.columns.get_level_values(1)
    average_weighted_edges_df.columns = col_names
    average_weighted_edges_df.index.names = ['Source']
    
    average_weighted_edges_df.to_csv(os.path.join(results_path, 'billeh_to_billeh_total_weights_matrix.csv'), sep='\t', decimal=',')
    
    # Get the std deviation of the connection weights
    std_weighted_edges_df = weighted_edges_df.groupby(['Source type', 'Target type']).std()
    std_weighted_edges_df = std_weighted_edges_df.unstack(fill_value=0)
    col_names = std_weighted_edges_df.columns.get_level_values(1)
    std_weighted_edges_df.columns = col_names
    std_weighted_edges_df.index.names = ['Source']
    
    std_weighted_edges_df.to_csv(os.path.join(results_path, 'billeh_to_billeh_total_weights_matrix_std.csv'), sep='\t', decimal=',')

   
if __name__ == "__main__":
    
    billeh_connectivity_matrices()        
    
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 23:35:53 2022

@author: UX325
"""

import os
import sys
import pandas as pd
import numpy as np
import random as python_random
import h5py
import io
import json
import argparse
import pickle
import warnings
import plotly.express as px
from collections import Counter
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, Callback, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

from time import time
from timeit import default_timer as timer


parentDir = os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd())))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management

my_seed = 42
os.environ['PYTHONHASHSEED']=str(my_seed)
rd = np.random.RandomState(seed=my_seed)
python_random.seed(my_seed)
tf.random.set_seed(my_seed)
        

def bbp_model_reduction(cortex_areas=['VISl', 'VISp'], hemisphere='right'):
    # It is generally recommended that applications avoid multiple opens of the same h5 file.
    if len(cortex_areas) == 0:
        print('Error: No cortex area selected.')
    elif cortex_areas[0]==cortex_areas[1]:
        edges_file = f'Data/{hemisphere}_local_edges.h5'
        cortex_region = f'{hemisphere}_local'
        patch_name = cortex_areas[0]
    else:
        edges_file = f'Data/{hemisphere}_ipsi_edges.h5'
        cortex_region = f'{hemisphere}_ipsi'
        patch_name = '_'.join(cortex_areas)
    
    # Get the whole cortex model from the BlueBrain project and extract only the regions of interest
    # Load node information from the BlueBrain project model
    path_to_csv ='Data/cortex_node_types.csv'
    node_types = pd.read_csv(path_to_csv, sep=' ')
    
    # Focus on one hemisphere and select source
    hemisphere_nodes = node_types.loc[node_types['hemisphere']==hemisphere]
    patch_node_types_mask = None
    for area in list(cortex_areas):
        patch_node_types_mask = np.logical_or(patch_node_types_mask, hemisphere_nodes['region']==area)
    patch_node_type_ids = hemisphere_nodes.loc[patch_node_types_mask]['node_type_id'].values

    # Load nodes of the Bbp cortex model
    path_to_h5 = 'Data/cortex_nodes.h5'
    node_h5 = h5py.File(path_to_h5, mode='r')
    
    # Extract node_type_ids and get the nodes mask
    cortex_node_type_ids = np.array(node_h5['nodes']['cortex']['node_type_id'])
    patch_nodes_mask = np.isin(cortex_node_type_ids, patch_node_type_ids)
    del cortex_node_type_ids
    print(f'{patch_name} neurons: ', patch_nodes_mask.sum())
    
    allen_id_to_bbp_id = np.array(node_h5['nodes']['cortex']['node_id'][patch_nodes_mask])
    bbp_id_to_allen_id = np.arange(allen_id_to_bbp_id.max()+1)
    bbp_id_to_allen_id = np.zeros_like(bbp_id_to_allen_id) - 1
    for allen_id, bbp_id in enumerate(allen_id_to_bbp_id):
        bbp_id_to_allen_id[bbp_id] = allen_id 
        
    # Save node data from the patch
    t0 = time()
    with h5py.File(f"Data/{patch_name}_nodes.h5", "w") as f:
        nodes_grp = f.create_group("nodes")
        patch_grp = nodes_grp.create_group(patch_name)
        for key in node_h5['nodes']['cortex'].keys():
            if key == '0':
                location_grp = patch_grp.create_group("0")
                for key in node_h5['nodes']['cortex']['0'].keys():
                    location_grp.create_dataset(key, data=node_h5['nodes']['cortex']['0'][key][patch_nodes_mask])
            else:
                patch_grp.create_dataset(key, data=node_h5['nodes']['cortex'][key][patch_nodes_mask])
        
        patch_grp.create_dataset('allen_id_to_bbp_id', data=allen_id_to_bbp_id.astype('u4'))
        patch_grp.create_dataset('bbp_id_to_allen_id', data=bbp_id_to_allen_id.astype('i4'))
    print(f'Time for saving nodes: {time()-t0}')
    
    # Load edges file with the BlueBrain project model ipsilateral synapses in the chosen hemisphere     
    edges_h5 = h5py.File(edges_file, mode='r')
    
    # Identify the connections between neurons within the patch
    patch_node_ids = node_h5['nodes']['cortex']['node_id'][patch_nodes_mask]
    t1 = time()
    source_edges_mask = np.isin(edges_h5['edges'][cortex_region]['source_node_id'], patch_node_ids)
    print(f'Time for masking source edges: {time()-t1}')
    target_edges_mask = np.isin(edges_h5['edges'][cortex_region]['target_node_id'], patch_node_ids)
    patch_edges_mask = np.logical_and(source_edges_mask, target_edges_mask)
    data_shape = patch_edges_mask.sum()
    print(f'{patch_name} synapses: ', data_shape)
    del source_edges_mask, target_edges_mask
    
    # Save edge data from the patch using chunks (dataset is too large and freezes the CPU)
    # https://www.oreilly.com/library/view/python-and-hdf5/9781491944981/ch04.html
    # https://docs.h5py.org/en/stable/high/dataset.html
    with h5py.File(f"Data/{patch_name}_edges.h5", "w") as f:
        edges_grp = f.create_group("edges")
        cortex_region_patch_grp = edges_grp.create_group(f"{cortex_region}_{patch_name}")
        for key in edges_h5['edges'][cortex_region].keys():
            #if key != '0' and key != 'indices':
            if key in ['source_node_id', 'target_node_id']:
                t0 = time()
                data = edges_h5['edges'][cortex_region][key]
                ds = cortex_region_patch_grp.create_dataset(key, shape=(data_shape,), chunks=True, dtype='u4')
                data_iter = 0
                chunks_iter = 0
                chunks_shape = ds.chunks[0]
                n_loops = int(data_shape/chunks_shape)-1
                for i in range(n_loops):
                    data_chunk_mask = patch_edges_mask[chunks_iter:chunks_iter+chunks_shape]
                    data_chunk = data[chunks_iter:chunks_iter+chunks_shape] 
                    data_chunk = data_chunk[data_chunk_mask]
                    n_elements = data_chunk_mask.sum()
                    ds[data_iter:data_iter+n_elements] = data_chunk
                    data_iter += n_elements
                    chunks_iter += chunks_shape
                
                if data_shape%chunks_shape!=0:
                    data_chunk_mask = patch_edges_mask[chunks_iter:]
                    data_chunk = data[chunks_iter:]
                    data_chunk = data_chunk[data_chunk_mask]
                    ds[data_iter:] = data_chunk
                print(f'Time for saving edges {key}: {time()-t0}')
    
# See inhibitory neurons classification in:
# Tremblay, R., Lee, S., and Rudy, B. (2016). GABAergic interneurons in the neocortex: from cellular properties to circuits. Neuron 91, 260–292.
def shorten_name(pop_name, exc_inh):
    i_classes = ['Sst', 'Pvalb', 'Htr3a']
    new_pop_name = 'L'
    new_pop_name = 'L' + pop_name[6] + '_' + exc_inh
    if exc_inh == 'e':
        return new_pop_name
    else:
        for i_class in i_classes:
            if i_class in pop_name:
                new_pop_name += '_'+i_class
        if len(new_pop_name)<5: # Vip and other inhibitory neurons are catalogued as Htr3a
            if 'Vip' in pop_name: 
                new_pop_name += '_'+'Htr3a'
            else: #only case is Ndnf-IRES2-dgCre
                new_pop_name += '_'+'Htr3a'       
    
        return new_pop_name
    
def new_pop_name_generator(pop_name, exc_inh):
    sep_idx = pop_name.find('; ') # Some Bbp populations correspond to two Allen populations. 
    if sep_idx != -1: 
        pop_name1 = pop_name[:sep_idx]
        abundance1 = pop_name1[:2]
        pop_name1 = pop_name1[4:]
        new_pop_name1 = shorten_name(pop_name1, exc_inh)
        
        pop_name2 = pop_name[sep_idx+2:]
        abundance2 = pop_name2[:2]
        pop_name2 = pop_name2[4:]
        new_pop_name2 = shorten_name(pop_name2, exc_inh)
        
        new_pop_name = abundance1 + new_pop_name1 + '%' + abundance2 + new_pop_name2
        
    else:
        new_pop_name = shorten_name(pop_name, exc_inh)
        
    return new_pop_name
    

def node_pop_assign(node_type_id, node_type_id_to_pop_name):
    pop_name = node_type_id_to_pop_name.get(node_type_id)
    idx = pop_name.find('%')
    if idx == -1:
        return pop_name
    else: # All the conflictive populations are of the style ACAd_70L2_i_Pvalb%30L2_i_Sst (mix of pvalb and sst). Random sample according to abbundance
        pop_name1 = pop_name[:idx] 
        pop_name1_split = pop_name1.split('_')
        region = pop_name1_split[0]
        abundance1 = float(pop_name1_split[1][:2])/100
        pop_name1 = '_'.join(pop_name1_split[1:])
        pop_name1 = pop_name1[2:]
        
        pop_name2 = pop_name[idx+1:]
        abundance2 = float(pop_name2[:2])/100
        pop_name2 = pop_name2[2:]
                
        pop_name = rd.choice([pop_name1, pop_name2], p=[abundance1, abundance2], size=1)[0] # sampling with replacement (no problem since we only take one sample)
        pop_name = region + '_' + pop_name
        
        return pop_name
    
    
def pop_renaming(pop_name):
    return '_'.join(pop_name.split('_')[1:])


def get_area_stats(array_node_types, area_name, path=''):
    counter = dict(Counter(array_node_types))
    # Sort the population names according to their layer depth
    all_pop_names = sorted(counter.keys())
    sorted_values = [x for _,x in sorted(zip(counter.keys(),counter.values()))]
    
    # Extract simplified pop_names
    layers = []
    pop_names = []
    for key in all_pop_names:
        items = key.split('_')[1:]
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
    keys = ['_'.join(key.split('_')[1:]) for key in counter.keys()]
    all_pop_names = sorted(keys)
    sorted_values = [x for _,x in sorted(zip(keys, counter.values()))]
    counter_df = pd.DataFrame(sorted_values, index=all_pop_names, columns=['# neurons'])
    
    return counter_df

def euclidean_distance_loss(y_true, y_pred):
    square_deviation = tf.math.square(y_true - y_pred)
    euclidean_distance = tf.math.reduce_sum(square_deviation, axis=1)
    return tf.math.reduce_mean(euclidean_distance)

def euclidean_distance_metric(y_true, y_pred):
    square_deviation = tf.math.square(y_true - y_pred)
    euclidean_distance = tf.math.sqrt(tf.math.reduce_sum(square_deviation, axis=1))
    return tf.math.reduce_mean(euclidean_distance)

# https://github.com/keras-team/keras/issues/13168
class PickleablePipeline(Pipeline):
    def _pickle_steps(self):
        steps = self.steps[:]
        classifier = steps[-1][1]
        bio = io.BytesIO()
        with h5py.File(bio) as f:
            classifier.model.save(f)
        steps[-1] = (steps[-1][0], bio)
        return steps

    def _unpickle_steps(self, steps):
        with h5py.File(steps[-1][1]) as f:
            steps[-1] = (steps[-1][0], load_model(f, custom_objects={'euclidean_distance_loss': euclidean_distance_loss,
                                                                    'euclidean_distance_metric': euclidean_distance_metric}))
        return steps

    def __getstate__(self):
        state = super(PickleablePipeline, self).__getstate__()
        state['steps'] = self._pickle_steps()
        return state

    def __setstate__(self, state):
        state['steps'] = self._unpickle_steps(state['steps'])
        return super(PickleablePipeline, self).__setstate__(state)


def gaussian_func(r, a, sigma):
    return a*np.exp(-r**2/sigma**2)

def freedman_diaconis(data, returnas="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return(result)


class ModelAnalysis:
    
    def __init__(self, source_area='VISl', target_area='VISp', hemisphere='right', neural_network_mapping=True):
        self.source_area = source_area
        self.target_area = target_area
        self.hemisphere = hemisphere
        self.neural_network_mapping = neural_network_mapping
        
    def __call__(self):
        
        if self.source_area != self.target_area:
            self.patch_name = '_'.join([self.source_area, self.target_area])
            self.patch_name_reversed = '_'.join([self.target_area, self.source_area])
            self.cortex_region = f'{self.hemisphere}_ipsi'
        else:
            self.patch_name = self.source_area
            self.patch_name_reversed = self.source_area
            self.cortex_region = f'{self.hemisphere}_local'
        
        if os.path.isfile(f'Data/{self.patch_name}_nodes.h5') and os.path.isfile(f'Data/{self.patch_name}_edges.h5'): 
            self.connectivity_analysis()  
        elif os.path.isfile(f'Data/{self.patch_name_reversed}_nodes.h5') and os.path.isfile(f'Data/{self.patch_name_reversed}_edges.h5'): 
            self.connectivity_analysis(reverse=True)        
        else:
            bbp_model_reduction(cortex_areas=[self.source_area, self.target_area], hemisphere=self.hemisphere)
            self.connectivity_analysis()
            
        self.retinotopic_mapping(self.neural_network_mapping)

    def connectivity_analysis(self, reverse=False):
        # Path to save results
        self.results_path = f'Source_{self.source_area}_Target_{self.target_area}'
        os.makedirs(self.results_path, exist_ok=True)
            
        # Load node information from the patch of the project model
        path_to_csv = 'Data/cortex_node_types.csv'
        self.node_types = pd.read_csv(path_to_csv, sep=' ')
        
        # Mapping csv from BlueBrain project model morphologies types to Allen Institute cell models
        path_to_mapping_df ='Data/mapping_for_bbp_morphology_models.csv'
        mapping_df = pd.read_csv(path_to_mapping_df, skiprows=0)
        mapping_df.drop(index=0, inplace=True) # Remove unnecesary first row
        mapping_df.dropna(axis=0, subset=['Mapping to Allen Institute cell models'], inplace=True) # Remove white rows written for separation between layers
        
        # Add short population names (e, iPvalb, iSst, iHtr3a) to the df
        short_allen_pop_names = []
        for pop_name, exc_inh in zip(mapping_df['Mapping to Allen Institute cell models'],
                                     mapping_df['exc/inh']):
            short_pop_name = new_pop_name_generator(pop_name, exc_inh)
            short_allen_pop_names.append(short_pop_name)   
        mapping_df['short_allen_cell_models'] = short_allen_pop_names
        
        # Create a 'pop_name' column to label the Allen Institute population name to each node_type and add the region to the pop name
        bbp_allen_mapping = dict(zip(mapping_df['Morphology type'], mapping_df['short_allen_cell_models']))
        self.node_types['pop_name'] = [bbp_allen_mapping.get(bbp_m_type) for bbp_m_type in self.node_types['m_type']]
        self.node_types['pop_name'] = self.node_types[['region', 'pop_name']].agg('_'.join, axis=1)
        # Create dictionary mapping from node_type_id to pop_name
        node_type_id_to_pop_name = dict(zip(self.node_types['node_type_id'], self.node_types['pop_name']))
        
        # Focus on one hemisphere and selected areas
        hemisphere_nodes = self.node_types.loc[self.node_types['hemisphere']==self.hemisphere]
        self.source_nodes = hemisphere_nodes.loc[(hemisphere_nodes['region']==self.source_area)]
        self.target_nodes = hemisphere_nodes.loc[(hemisphere_nodes['region']==self.target_area)]
        self.source_node_type_ids = self.source_nodes['node_type_id']
        self.target_node_type_ids = self.target_nodes['node_type_id']
            
        # Load nodes of the Bbp cortex model
        if reverse:
            self.patch_name = self.patch_name_reversed
        else:
            self.patch_name = self.patch_name
        path_to_h5 = f"Data/{self.patch_name}_nodes.h5"
        self.node_h5 = h5py.File(path_to_h5, mode='r')
        
        # Extract node_type_ids and create array of population names
        self.patch_node_type_ids = np.array(self.node_h5['nodes'][self.patch_name]['node_type_id'])
        self.patch_node_type = np.array([node_pop_assign(node_type_id, node_type_id_to_pop_name) for node_type_id in self.node_h5['nodes'][self.patch_name]['node_type_id']])
        
        # Obtain a mask for the source and target populations
        self.source_nodes_mask = np.isin(self.patch_node_type_ids, self.source_node_type_ids)
        self.target_nodes_mask = np.isin(self.patch_node_type_ids, self.target_node_type_ids)
        with open(os.path.join(self.results_path, 'Cortex_areas_stats.txt'), 'w') as f:
            f.write(f'Source - {self.source_area} neurons: {self.source_nodes_mask.sum()}\n', )
            f.write(f'Target - {self.target_area} neurons: {self.target_nodes_mask.sum()}\n', )
            
        # Load conversor from bbp node_id to allen node_id
        self.allen_id_to_bbp_id = np.array(self.node_h5['nodes'][self.patch_name]['allen_id_to_bbp_id']).astype('u4')
        self.bbp_id_to_allen_id = np.array(self.node_h5['nodes'][self.patch_name]['bbp_id_to_allen_id']).astype('i4')
        
        self.source_node_ids = self.allen_id_to_bbp_id[self.source_nodes_mask]
        self.target_node_ids = self.allen_id_to_bbp_id[self.target_nodes_mask]
        
        self.source_node_popnames = self.patch_node_type[self.source_nodes_mask]
        self.target_node_popnames = self.patch_node_type[self.target_nodes_mask]
        
        source_node_type_ids = self.patch_node_type_ids[self.source_nodes_mask]
        target_node_type_ids = self.patch_node_type_ids[self.target_nodes_mask]
        
        source_node_type = np.array([node_pop_assign(node_type_id, node_type_id_to_pop_name) for node_type_id in source_node_type_ids])
        target_node_type = np.array([node_pop_assign(node_type_id, node_type_id_to_pop_name) for node_type_id in target_node_type_ids])
        
        # Get the statistics of each area
        get_area_stats(source_node_type, self.source_area, path=self.results_path)
        get_area_stats(target_node_type, self.target_area, path=self.results_path)
        
        # Load file with the BlueBrain project model ipsilateral synapses in the chosen hemisphere 
        t0 = time()
        edges_file = f"Data/{self.patch_name}_edges.h5"
        edges_h5 = h5py.File(edges_file, mode='r')
        
        self.edges_source_ids = np.array(edges_h5['edges'][f'{self.cortex_region}_{self.patch_name}']['source_node_id'])
        self.edges_target_ids = np.array(edges_h5['edges'][f'{self.cortex_region}_{self.patch_name}']['target_node_id'])
        # del node_h5, edges_h5
        
        # Select only the synapses between the source and target area
        self.source_edges_mask = np.isin(self.edges_source_ids, self.source_node_ids)
        self.target_edges_mask = np.isin(self.edges_target_ids, self.target_node_ids)
        self.edges_mask = np.logical_and(self.source_edges_mask, self.target_edges_mask)
        # del source_nodes_mask, target_nodes_mask
        print(f'Time for masking edges: {time()-t0}')
        
        # Obtain the total synapses matrix between neuron types
        self.edges_df = pd.DataFrame()
        self.edges_df['Source_id'] = self.bbp_id_to_allen_id[self.edges_source_ids[self.edges_mask]].astype('u4')
        self.edges_df['Source'] = self.patch_node_type[(self.bbp_id_to_allen_id[self.edges_source_ids[self.edges_mask]]).astype('u4')]
        # del edges_source_ids
        self.edges_df['Target_id'] = self.bbp_id_to_allen_id[self.edges_target_ids[self.edges_mask]].astype('u4')
        self.edges_df['Target'] = self.patch_node_type[(self.bbp_id_to_allen_id[self.edges_target_ids[self.edges_mask]]).astype('u4')]
        # del edges_target_ids
        
        t0 = time()
        file_management.save_lzma(self.edges_df,  f'{self.source_area}_to_{self.target_area}_edges_df.lzma', self.results_path)
        print(f'Time for saving edges file: {time()-t0}')
        # edges_df = self.edges_df[['Source', 'Target']]
        # Count number of edges for each pair of source and target population
        edges_source_target_df = pd.crosstab(self.edges_df.Source, self.edges_df.Target)
        
        source_pop_names = set(self.node_types.loc[np.logical_and(self.node_types['region']==self.source_area, self.node_types['hemisphere']==self.hemisphere)]['pop_name'])
        source_pop_names = [pop_name for pop_name in source_pop_names if pop_name.find('%') == -1]
        target_pop_names = set(self.node_types.loc[np.logical_and(self.node_types['region']==self.target_area, self.node_types['hemisphere']==self.hemisphere)]['pop_name'])
        target_pop_names = [pop_name for pop_name in target_pop_names if pop_name.find('%') == -1]
        
        edges_source_target_df = edges_source_target_df.reindex(index=source_pop_names, columns=target_pop_names, fill_value=0)
        edges_source_target_df.rename(index=pop_renaming, columns=pop_renaming, inplace=True)
        # edges_df = edges_df.loc[~(edges_df==0).all(axis=1)] # Remove disconnected source populations
        edges_source_target_df.sort_index(axis=0, inplace=True)
        edges_source_target_df.sort_index(axis=1, inplace=True)
        
        edges_source_target_df.to_csv(os.path.join(self.results_path, f'{self.source_area}_to_{self.target_area}_total_synapses_matrix.csv'), sep='\t', decimal=',')
    
        # Obtain the probability of connection matrix
        n_source = get_population_neurons(source_node_type).values
        n_target = get_population_neurons(target_node_type).transpose().values
    
        probability_edges_df = edges_source_target_df.div(n_source, axis=1)
        probability_edges_df = probability_edges_df.div(n_target, axis=1)
    
        probability_edges_df.to_csv(os.path.join(self.results_path, f'{self.source_area}_to_{self.target_area}_connection_probability_matrix.csv'), sep='\t', decimal=',')

    def neuron_types_planar_density(self, neurons_df, shape_estimator='parallelepiped', center_cylinder_radius=175, side_length=175):
        # Get the center of the area
        center_coordinates = neurons_df.loc[:, ['x', 'y', 'z']].median(axis=0)
        neurons_df.loc[:, 'Distance_to_center'] = np.sqrt(((neurons_df[['x', 'y', 'z']]-center_coordinates)**2).sum(axis=1))
        all_closest_neurons = neurons_df.sort_values(['Distance_to_center']).groupby('Neuron_type', sort=False).head(100)
        # use excitatory neurons to determine the depth and planar coordinates
        exc_pops = [f'{self.target_area}_L2_e', f'{self.target_area}_L4_e', f'{self.target_area}_L5_e', f'{self.target_area}_L6_e']
        exc_closest_neurons = all_closest_neurons.loc[all_closest_neurons['Neuron_type'].isin(exc_pops)]
        pca=PCA(3)
        pca.fit(exc_closest_neurons[['x', 'y', 'z']])   
        # transform original coordinates to the new pca coordinates
        pca_center_coordinates = pca.transform([center_coordinates])[0]
        neurons_df.loc[:, ['x_trans', 'y_trans', 'z_trans']] = pca.transform(neurons_df[['x', 'y', 'z']])
        
        if shape_estimator == 'parallelepiped':
            neurons_df.loc[:, ['y_trans', 'z_trans']] -= np.array([pca_center_coordinates[1], pca_center_coordinates[2]])
            mask_y = np.abs(neurons_df.loc[:, 'y_trans']) < side_length
            mask_z = np.abs(neurons_df.loc[:, 'z_trans']) < side_length
            mask = np.logical_and(mask_y, mask_z)
            planar_area = side_length**2
        elif shape_estimator == 'cylinder':
            neurons_df.loc[:, 'planar_intersomatic_distance'] = np.sqrt(((neurons_df[['y_trans', 'z_trans']]-np.array([pca_center_coordinates[1], pca_center_coordinates[2]]))**2).sum(axis=1))
            # isolate the neurons located inside a cylinder around the center
            mask = neurons_df.loc[:, 'planar_intersomatic_distance'] < center_cylinder_radius
            planar_area = np.pi*center_cylinder_radius**2
            
        neurons_df = neurons_df.loc[mask]
        # count neurons of every type and estimate their densities
        n_neurons_in_area = neurons_df[['Neuron_type', 'Distance_to_center']].groupby('Neuron_type').count()
        planar_density = dict()
        for neuron_type, n_cylinder in zip(list(n_neurons_in_area.index), n_neurons_in_area.values):
            planar_density[neuron_type] = n_cylinder[0]/planar_area
            
        # Path to save results
        self.density_path = os.path.join(self.results_path, 'Density_estimation')
        os.makedirs(self.density_path, exist_ok=True)

        # Scatter closest excitatory neurons
        exc_closest_neurons.loc[:, ['x_trans', 'y_trans', 'z_trans']] = pca.transform(exc_closest_neurons[['x', 'y', 'z']])
        center_df = pd.DataFrame(index=np.arange(1), columns=exc_closest_neurons.columns)
        center_df.loc[0, ['x', 'y', 'z']] = center_coordinates
        center_df.loc[0, ['x_trans', 'y_trans', 'z_trans']] = pca_center_coordinates
        center_df = center_df.assign(marker_size = 15)
        center_df = center_df.assign(Neuron_type = f'{self.target_area} center')
        exc_closest_neurons = exc_closest_neurons.assign(marker_size = 2)
        representation_df = pd.concat([center_df, exc_closest_neurons], ignore_index=True, sort=False)
        fig = px.scatter_3d(representation_df, x='x', y='y', z='z', 
                            size='marker_size', color='Neuron_type', opacity=0.7)
        fig.update_layout(title_text=f'Closest 100 excitatory neurons to {self.target_area} center', margin=dict(l=0, r=0, b=0, t=0), 
                          scene=dict(aspectmode='data'))
        fig.write_html(os.path.join(self.density_path, f"closest_100_excitatory_neurons_to_{self.target_area}_center.html"))
        
        # Scatter closest excitatory neurons with cylindrical coordinates
        fig = px.scatter_3d(representation_df, x='x_trans', y='y_trans', z='z_trans', 
                            size='marker_size', color='Neuron_type', opacity=0.7)
        fig.update_layout(title_text=f'Closest 100 excitatory neurons to {self.target_area} center in PCA coordinates', margin=dict(l=0, r=0, b=0, t=0), 
                          scene=dict(aspectmode='data'))
        fig.write_html(os.path.join(self.density_path, f"closest_100_excitatory_neurons_to_{self.target_area}_center_PCA_coordinates.html"))
        
        # Scatter area around the center
        neurons_df = neurons_df.assign(marker_size = 2)
        representation_df = pd.concat([center_df, neurons_df], ignore_index=True, sort=False)
        fig = px.scatter_3d(representation_df, x='x_trans', y='y_trans', z='z_trans', 
                            size='marker_size', color='Neuron_type', opacity=0.7)
        fig.update_layout(title_text=f'{self.target_area} area around center in PCA coordinates', scene=dict(aspectmode='data'))
        fig.write_html(os.path.join(self.density_path, f"area_around_{self.target_area}_center_composition.html"))
        fig.show()
        
        # Representation of neurons inside the area within V1 and LM frame
        new_target_subset = self.target_subset.copy()
        new_source_subset = self.source_subset.copy()
        new_target_subset.loc[:, ['x_trans', 'y_trans', 'z_trans']] = pca.transform(new_target_subset[['x', 'y', 'z']])
        new_source_subset.loc[:, ['x_trans', 'y_trans', 'z_trans']] = pca.transform(new_source_subset[['x', 'y', 'z']])
        new_target_subset = new_target_subset.assign(marker_size = 2)
        new_source_subset = new_source_subset.assign(marker_size = 2)
        new_target_subset = new_target_subset.assign(Neuron_type = f'{self.target_area}')
        new_source_subset = new_source_subset.assign(Neuron_type = f'{self.source_area}')
        neurons_df = neurons_df.assign(Neuron_type = f'{self.target_area} central area')
        representation_df = pd.concat([new_target_subset, new_source_subset, center_df, neurons_df], ignore_index=True, sort=False)
        fig = px.scatter_3d(representation_df, x='x_trans', y='y_trans', z='z_trans', 
                            color='Neuron_type', size='marker_size',
                            opacity=0.7)
        fig.update_layout(title_text='Neurons inside central area in PCA coordinates', scene=dict(aspectmode='data'))
        fig.write_html(os.path.join(self.density_path, f"area_around_{self.target_area}_center.html"))
        fig.show()

        return planar_density
    

    def retinotopic_mapping(self, neural_network_mapping=True, source_area_central_radius=300):
     
        n_source_neurons = self.source_nodes_mask.sum()
        source_df = pd.DataFrame()
        source_df['node_id'] = self.allen_id_to_bbp_id[self.source_nodes_mask]
        source_df['x'] = self.node_h5['nodes'][self.patch_name]['0']['x'][self.source_nodes_mask]
        source_df['y'] = self.node_h5['nodes'][self.patch_name]['0']['y'][self.source_nodes_mask]
        source_df['z'] = self.node_h5['nodes'][self.patch_name]['0']['z'][self.source_nodes_mask]
        source_df['Neuron_type'] = self.source_node_popnames
        source_df['Visual_area'] = self.source_area
        source_df = source_df.set_index('node_id')
        # V1_df['color'] = 'r'
        percentage_of_selected_neurons = 0.005
        selected_tf_indices = rd.choice(np.arange(n_source_neurons), size=int(n_source_neurons*percentage_of_selected_neurons), replace=False)
        self.source_subset = source_df.iloc[selected_tf_indices]
        self.source_subset = self.source_subset.assign(marker_size = 2)
        
        n_target_neurons = self.target_nodes_mask.sum()
        target_df = pd.DataFrame()
        target_df['node_id'] = self.allen_id_to_bbp_id[self.target_nodes_mask]
        target_df['x'] = self.node_h5['nodes'][self.patch_name]['0']['x'][self.target_nodes_mask]
        target_df['y'] = self.node_h5['nodes'][self.patch_name]['0']['y'][self.target_nodes_mask]
        target_df['z'] = self.node_h5['nodes'][self.patch_name]['0']['z'][self.target_nodes_mask]
        target_df['Neuron_type'] = self.target_node_popnames
        target_df['Visual_area'] = self.target_area
        target_df = target_df.set_index('node_id')
        selected_tf_indices = rd.choice(np.arange(n_target_neurons), size=int(n_target_neurons*percentage_of_selected_neurons), replace=False)
        self.target_subset = target_df.iloc[selected_tf_indices]
        self.target_subset = self.target_subset.assign(marker_size = 2)
        
        # get the planar densities of the different neuron types at the target region
        planar_density = self.neuron_types_planar_density(target_df)
        with open(os.path.join(self.density_path, f'{self.target_area}_populations_planar_densities.json'), 'w') as handle:
            json.dump(planar_density, handle)
             
        self.edges_df['Source_id'] = self.allen_id_to_bbp_id[self.edges_df['Source_id']]
        self.edges_df['Target_id'] = self.allen_id_to_bbp_id[self.edges_df['Target_id']]
        self.edges_df.loc[:, ['x_target', 'y_target', 'z_target']] = target_df.loc[self.edges_df['Target_id'], ['x', 'y', 'z']].values
        # we use the median to estimate the projected sources positions
        projected_source_positions = self.edges_df[['Source_id', 'x_target', 'y_target', 'z_target']].groupby(['Source_id'], sort=False).median()
        projected_source_positions_std = self.edges_df[['Source_id', 'x_target', 'y_target', 'z_target']].groupby(['Source_id'], sort=False).std()
        
        # Focus only on the neurons from the source area that project to the target and save their projected positions
        source_connected_neurons_df = source_df.loc[projected_source_positions.index]
        
        # # Illustrate the retinotopic mapping
        self.mapping_path = os.path.join(self.results_path, 'Retinotopic_mapping')
        os.makedirs(self.mapping_path, exist_ok=True)

        if neural_network_mapping:
            # Illustrate the retinotopic mapping
            self.mapping_path = os.path.join(self.results_path, 'Retinotopic_mapping_prediction')
            os.makedirs(self.mapping_path, exist_ok=True)
            
            # Uncomment the following for focussing only on source neurons with at least 30 targets. 
            # This will speed up the computation since 10% of source neurons disappear and also will allow to more accurately
            # determine the projected location of the source neuron in the target region
            # 94% of source satisfy >10, 92% satisfy > 20, 90% satisfy >30, 88% >40, >50 87%, >100 81%
            source_filter = self.edges_df[['Source_id','Target_id']].groupby(['Source_id'], sort=False).count()
            source_filter_indices = source_filter.loc[source_filter['Target_id']>30].index # to reduce computation cost
            
            hub_source_positions = projected_source_positions.loc[source_filter_indices]
            hub_source_positions_std = projected_source_positions_std.loc[source_filter_indices]
            hub_source_df = source_connected_neurons_df.loc[source_filter_indices]
            
            uncertainty_projected_position = np.sqrt((hub_source_positions_std.mean(axis=0)**2).sum())

            hub_source_df.loc[:, ['x_projected', 'y_projected', 'z_projected']] = hub_source_positions.loc[:, ['x_target', 'y_target', 'z_target']].values
            hub_source_df.dropna(axis=0, inplace=True)
            hub_source_df.drop('Visual_area', axis=1, inplace=True)
            
            def get_model():
                model = Sequential()
                model.add(Dense(128, input_shape = (3, ), kernel_initializer='glorot_uniform', activation='relu'))
                model.add(Dense(64, kernel_initializer='glorot_uniform'))
                model.add(LeakyReLU(alpha=0.3)) # activation layer here instead 
                model.add(Dense(32, kernel_initializer='glorot_uniform'))
                model.add(LeakyReLU(alpha=0.3)) # activation layer here instead 
                model.add(Dense(16, kernel_initializer='glorot_uniform'))
                model.add(LeakyReLU(alpha=0.3)) # activation layer here instead 
                model.add(Dense(8, kernel_initializer='glorot_uniform'))
                model.add(LeakyReLU(alpha=0.3)) # activation layer here instead 
                model.add(Dense(3, kernel_initializer='glorot_uniform')) # no activation required for linear regression in the output layer
            #     model.compile(loss=euclidean_distance_loss, optimizer='adam', metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name='rmse'), euclidean_distance_metric])
                model.compile(loss=euclidean_distance_loss, optimizer='adam', metrics=[euclidean_distance_metric])
            
                return model
            
            X = hub_source_df[['x', 'y', 'z']].values
            y = hub_source_df[['x_projected', 'y_projected', 'z_projected']].values
            # n_inputs, n_outputs = X.shape[1], y.shape[1]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            # scaler
            # x_scaler = MinMaxScaler().fit(X_train)
            # X_train_scaled = x_scaler.transform(X_train)
            # X_test_scaled = x_scaler.transform(X_test)
            
            class MyCallback(Callback):
                def __init__(self, monitor='val_loss', value=50, verbose=1, patience=1000):
                    super(Callback, self).__init__()
                    self.monitor = monitor
                    self.value = value
                    self.verbose = verbose
                    self.patience=patience  
                    self.starttime = timer()
                    # best_weights para almacenar los pesos en los cuales ocurre la perdida minima.
                    self.best_weights = None
                    
                def on_train_begin(self, logs=None):
                    # El numero de epoch que ha esperado cuando la perdida ya no es minima.
                    self.wait = 0
                    # Initialize el best como infinito.
                    self.best = np.Inf
            
                def on_epoch_end(self, epoch, logs={}):
                    if epoch%100 == 0:
                        elapsed_time = float(timer()-self.starttime)
                        self.starttime = timer()
                        print('Epoch {}/{} - Elapsed_time: {:3.2f} s - Train loss: {:7.2f} - Train EUD {:7.2f} - Val loss: {:7.2f} - Val EUD: {:7.2f}.'.format(
                            epoch+1, self.params.get('epochs'), elapsed_time, logs['loss'], logs['euclidean_distance_metric'], logs['val_loss'], logs['val_euclidean_distance_metric']))
                        
                    current = logs.get(self.monitor)   
                    if np.less(current, self.best):
                        self.best = current
                        # Guardar los mejores pesos si el resultado actual es mejor (menos).
                        self.best_weights = self.model.get_weights()
                    
                    if current is None:
                        warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
            
                    if current < self.value:
                        if self.verbose > 0:
                            print("Epoch %05d: early stopping THR" % epoch)
                            
                        elapsed_time = float(timer()-self.starttime)
                        print('Epoch {}/{} - Elapsed_time: {:3.2f} s - Train loss: {:7.2f} - Train EUD {:7.2f} - Val loss: {:7.2f} - Val EUD: {:7.2f}.'.format(
                            epoch+1, self.params.get('epochs'), elapsed_time, logs['loss'], logs['euclidean_distance_metric'], logs['val_loss'], logs['val_euclidean_distance_metric']))
            
                        self.model.stop_training = True   
                        
                        
                    if ((epoch+1) == self.params.get('epochs')) and (epoch%100 != 0):
                        elapsed_time = float(timer()-self.starttime)
                        print('Epoch {}/{} - Elapsed_time: {:3.2f} s - Train loss: {:7.2f} - Train EUD {:7.2f} - Val loss: {:7.2f} - Val EUD: {:7.2f}.'.format(
                            epoch+1, self.params.get('epochs'), elapsed_time, logs['loss'], logs['euclidean_distance_metric'], logs['val_loss'], logs['val_euclidean_distance_metric']))
                
                def on_train_end(self, epoch, logs={}):
                    
                    print('Restaurando los pesos del modelo del final de la mejor epoch.')
                    self.model.set_weights(self.best_weights)        
            
            history_filename = os.path.join(self.mapping_path, 'history.csv')         
            callbacks = [
                MyCallback(monitor='val_euclidean_distance_metric', value=10, verbose=1),
                CSVLogger(history_filename, append=False),
                EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10000)
            ]
    
            model = KerasRegressor(build_fn= get_model, epochs=200000, batch_size=128, 
                             verbose=0)

            pipe = PickleablePipeline([
            ('scaler', MinMaxScaler()),
            ('model', model)
            ])
            
            # Fit first the preprocessing
            pipe[:-1].fit(X_train)
            pipe.fit(X_train, y_train, model__validation_data=(pipe[:-1].transform(X_test), y_test), model__callbacks=callbacks)
            
            y_pred = pipe.predict(X_test) 
            print('Best model val loss: {:3.2f}'.format(float(euclidean_distance_loss(y_test, y_pred))))
            print('BEst model val EUD: {:3.2f}'.format(float(euclidean_distance_metric(y_test, y_pred))))
            
            # model.save(os.path.join(self.mapping_path, f'saved_models/{self.source_area}_to_{self.target_area}_model'))
            model_path = os.path.join(self.mapping_path, 'saved_models')
            os.makedirs(model_path, exist_ok=True)
            filename = os.path.join(model_path, 'best_model.pk')
            with open(filename, 'wb') as file:
                pickle.dump(pipe, file)  
            
            # summarize history for loss
            history = pd.read_csv(history_filename)
            plt.plot(history['euclidean_distance_metric'][100:])
            plt.plot(history['val_euclidean_distance_metric'][100:])
            plt.title('model loss')
            plt.ylabel('Loss (microns)')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper right')
            plt.savefig(os.path.join(self.mapping_path, 'training_loss_curve.png'), dpi=300)
            

            # Plot some samples
            samples_mapping_path = os.path.join(self.mapping_path, 'Mapping_examples')
            os.makedirs(samples_mapping_path, exist_ok=True)
            
            for sid in rd.choice(source_connected_neurons_df.index, size=10, replace=False):
                df1 = target_df.loc[np.isin(target_df.index, self.edges_df['Target_id'].loc[self.edges_df['Source_id']==sid])]
                df2 = source_df.loc[source_df.index==sid]
                location_prediction = pipe.predict(df2[['x', 'y', 'z']])
                df_pred = pd.DataFrame(index=[sid], columns=target_df.columns)
                df_pred[['x', 'y', 'z']] = location_prediction
                df_pred['Visual_area'] = f'{self.target_area} prediction'
                df_pred['marker_size'] = 10
                df1 = df1.assign(Visual_area = f'{self.target_area} targets')
                df2 = df2.assign(Visual_area = f'{self.source_area} source')
                df1 = df1.assign(marker_size = 2)
                df2 = df2.assign(marker_size = 10)
                average_targets = pd.DataFrame(df1.median(axis=0)).transpose()
                average_targets = average_targets.assign(marker_size = 10)
                average_targets = average_targets.assign(Visual_area = f'{self.target_area} average targets position')
            
                representation_df = pd.concat([self.source_subset,self.target_subset,df1, df2, df_pred, average_targets], ignore_index=True, sort=False)
                fig = px.scatter_3d(representation_df, x='x', y='y', z='z', 
                                    color='Visual_area', size='marker_size',
                                    opacity=0.7)
            
                #fig.update_traces(marker_size = 2)
                fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0)) #, scene_camera=camera)
                fig.write_html(os.path.join(samples_mapping_path, f"{self.source_area}_node_{sid}_to_{self.target_area}_bbp_model.html"))
            
                ## Histograms plot
                edges_subset = self.edges_df[['Source_id', 'x_target', 'y_target', 'z_target']].loc[self.edges_df['Source_id']==sid]
                edges_subset_median = edges_subset.median()
                model_prediction =  pipe.predict([source_connected_neurons_df.loc[sid, ['x', 'y', 'z']].values])
                error_in_prediction = np.sqrt(((edges_subset_median[['x_target', 'y_target', 'z_target']] -model_prediction)**2).sum())
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.hist(edges_subset['x_target'], color='b', alpha=0.2)
                ax1.axvline(x=model_prediction[0], color='r')
                ax1.axvline(x=edges_subset_median['x_target'], color='g')
                ax1.set_xlabel('x (microns)')
                
                ax2.hist(edges_subset['y_target'], color='b')
                ax2.axvline(x=model_prediction[1], color='r')
                ax2.axvline(x=edges_subset_median['y_target'], color='g')
                ax2.set_xlabel('y (microns)')
                
                ax3.hist(edges_subset['z_target'], color='b')
                ax3.axvline(x=model_prediction[2], color='r', label=f'Total error: {error_in_prediction}')
                ax3.axvline(x=edges_subset_median['z_target'], color='g')
                ax3.set_xlabel('z (microns)')
                ax3.legend()
                
                fig.savefig(os.path.join(samples_mapping_path, f"{self.source_area}_node_{sid}_to_{self.target_area}_histogram.png"), dpi=300, transparent=True)
                plt.close()

            
            # Error analysis of the 10 worst predictions
            worst_mapping_path = os.path.join(self.mapping_path, 'Worst_mapping_examples')
            os.makedirs(worst_mapping_path, exist_ok=True)
            y_pred = pipe.predict(hub_source_df.loc[:, ['x', 'y', 'z']].values)
            y_approx = hub_source_df.loc[:, ['x_projected', 'y_projected', 'z_projected']].values
            error_in_prediction = np.sqrt(((y_approx -y_pred)**2).sum(axis=1))
            hub_source_df = hub_source_df.assign(Prediction_error = error_in_prediction)
            hub_source_df.sort_values('Prediction_error', ascending=True, inplace=True)
            print(hub_source_df)
            print(sorted(error_in_prediction))
            
            for sid in list(hub_source_df.iloc[-10:].index):
                pred_error = hub_source_df.loc[sid, 'Prediction_error']
                print(sid, pred_error)
                df_pred = pd.DataFrame(index=[sid], columns=target_df.columns)
                df1 = target_df.loc[np.isin(target_df.index, self.edges_df['Target_id'].loc[self.edges_df['Source_id']==sid])]
                df2 = source_df.loc[source_df.index==sid]
                location_prediction = pipe.predict(df2[['x', 'y', 'z']])
                df_pred[['x', 'y', 'z']] = location_prediction
                df_pred['Visual_area'] = f'{self.target_area} prediction'
                df_pred['marker_size'] = 10
                df1 = df1.assign(Visual_area = f'{self.target_area} targets')
                df2 = df2.assign(Visual_area = f'{self.source_area} source')
                df1 = df1.assign(marker_size = 2)
                df2 = df2.assign(marker_size = 10)
                average_targets = pd.DataFrame(df1.median(axis=0)).transpose()
                average_targets = average_targets.assign(marker_size = 10)
                average_targets = average_targets.assign(Visual_area = f'{self.target_area} average targets position')
                
                representation_df = pd.concat([self.source_subset,self.target_subset,df1, df2, df_pred, average_targets], ignore_index=True, sort=False)
                fig = px.scatter_3d(representation_df, x='x', y='y', z='z', 
                                    color='Visual_area', size='marker_size',
                                    opacity=0.7)
                
                #fig.update_traces(marker_size = 2)
                fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0)) #, scene_camera=camera)
                fig.write_html(os.path.join(worst_mapping_path, f"{self.source_area}_node_{sid}_to_{self.target_area}_error_{pred_error}.html"))               
           
            
            location_prediction = pipe.predict(source_connected_neurons_df[['x', 'y', 'z']])
            source_connected_neurons_df.loc[projected_source_positions.index, ['x_projected', 'y_projected', 'z_projected']] = location_prediction
            
        else:
            source_connected_neurons_df.loc[:, ['x_projected', 'y_projected', 'z_projected']] = projected_source_positions.loc[:, ['x_target', 'y_target', 'z_target']].values

# source_connected_neurons_df = source_df.loc[projected_source_positions.index]

        source_connected_neurons_df.dropna(axis=0, inplace=True)
        source_connected_neurons_df = source_connected_neurons_df.drop('Visual_area', axis=1)
        target_df = target_df.drop('Visual_area', axis=1)

        # Lets focus in neurons located in the center of V1(LM) which primarily project to the center of LM(V1)
        source_central_neurons_df = source_df.copy()
        # source_connected_central_neurons_df = source_connected_neurons_df.copy()
        center_coordinates = source_central_neurons_df.loc[:, ['x', 'y', 'z']].median(axis=0)
        source_central_neurons_df.loc[:, 'Distance_to_center'] = np.sqrt(((source_central_neurons_df[['x', 'y', 'z']]-center_coordinates)**2).sum(axis=1))
        all_closest_neurons = source_central_neurons_df.sort_values(['Distance_to_center']).groupby('Neuron_type', sort=False).head(100)
        # use excitatory neurons to determine the depth and planar coordinates
        exc_pops = [f'{self.source_area}_L2_e', f'{self.source_area}_L4_e', f'{self.source_area}_L5_e', f'{self.source_area}_L6_e']
        exc_closest_neurons = all_closest_neurons.loc[all_closest_neurons['Neuron_type'].isin(exc_pops)]
        pca=PCA(3)
        pca.fit(exc_closest_neurons[['x', 'y', 'z']])   
        # transform original coordinates to the new pca coordinates
        pca_center_coordinates = pca.transform([center_coordinates])[0]
        source_central_neurons_df.loc[:, ['x_trans', 'y_trans', 'z_trans']] = pca.transform(source_central_neurons_df[['x', 'y', 'z']])
        source_central_neurons_df.loc[:, 'planar_intersomatic_distance'] = np.sqrt(((source_central_neurons_df[['y_trans', 'z_trans']]-np.array([pca_center_coordinates[1], pca_center_coordinates[2]]))**2).sum(axis=1))
        source_central_neurons_df_mask = source_central_neurons_df.loc[:, 'planar_intersomatic_distance'] < source_area_central_radius
        
        source_central_neurons_df = source_central_neurons_df.loc[source_central_neurons_df_mask]
        
        source_connected_central_neurons_df_mask = source_connected_neurons_df.index.isin(source_central_neurons_df.index)
        source_connected_central_neurons_df = source_connected_neurons_df.loc[source_connected_central_neurons_df_mask]
        print(source_central_neurons_df.shape)
        print(source_connected_central_neurons_df.shape)
        
        source_ids = list(set(source_connected_central_neurons_df.index)) # some source neurons may be disconnected triggering nan values
        exc_pops = [f'{self.target_area}_L2_e', f'{self.target_area}_L4_e', f'{self.target_area}_L5_e', f'{self.target_area}_L6_e']
        n_source_ids = get_population_neurons(source_central_neurons_df['Neuron_type'])
        n_source_ids.index = f'{self.source_area}_' + n_source_ids.index.astype(str)
        print(n_source_ids)
        t0 = time()
        for idx, sid in enumerate(source_ids):
            if idx%1000 == 0:
                print(idx)
                print(time()-t0)
                t0 = time()
            xp, yp, zp = source_connected_central_neurons_df.loc[sid, ['x_projected', 'y_projected', 'z_projected']]
            # Use closest excitatory neurons to determine the cylindrical coordinates through a PCA and determine the neuron densities
            new_target_df = target_df.copy()
            new_target_df['Distance_to_projected_source'] = np.sqrt(((new_target_df[['x', 'y', 'z']]-np.array([xp, yp, zp]))**2).sum(axis=1))
            all_closest_neurons = new_target_df.sort_values(['Distance_to_projected_source']).groupby('Neuron_type', sort=False).head(100)
            exc_closest_neurons = all_closest_neurons.loc[all_closest_neurons['Neuron_type'].isin(exc_pops)]
            # Make principal component analysis
            pca=PCA(3)
            pca.fit(exc_closest_neurons[['x', 'y', 'z']])   
            #Get now neurons projections on the cylindrical basis
            xp_trans, yp_trans, zp_trans = pca.transform(np.array([[xp, yp, zp]]))[0]
            
            edges_subset_df = self.edges_df.loc[self.edges_df['Source_id']==sid]
            targets_projection = pca.transform(edges_subset_df.loc[:,['x_target', 'y_target', 'z_target']])
            edges_subset_df.loc[:, ['x_trans', 'y_trans', 'z_trans']] = targets_projection

            intersomatic_planar_dist = np.sqrt(((edges_subset_df[['y_trans', 'z_trans']] - np.array([yp_trans, zp_trans]))**2).sum(axis=1))
            source_edges_index = edges_subset_df.index
            self.edges_df.loc[source_edges_index, 'planar_intersomatic_distance'] = (intersomatic_planar_dist.values).astype(np.float32)


            if idx in [0,1]:
                # Scatter closest excitatory neurons
                exc_closest_neurons.loc[:, ['x_trans', 'y_trans', 'z_trans']] = pca.transform(exc_closest_neurons[['x', 'y', 'z']])
                all_closest_neurons.loc[:, ['x_trans', 'y_trans', 'z_trans']] = pca.transform(all_closest_neurons[['x', 'y', 'z']])
                sid_df = pd.DataFrame(index=np.arange(1), columns=exc_closest_neurons.columns)
                sid_df.loc[0, ['x', 'y', 'z']] = np.array([xp, yp, zp])
                sid_df.loc[0, ['x_trans', 'y_trans', 'z_trans']] = np.array([xp_trans, yp_trans, zp_trans])
                sid_df = sid_df.assign(marker_size = 15)
                sid_df = sid_df.assign(Neuron_type = 'Source')
                exc_closest_neurons = exc_closest_neurons.assign(marker_size = 2)
                representation_df = pd.concat([sid_df, exc_closest_neurons], ignore_index=True, sort=False)
                fig = px.scatter_3d(representation_df, x='x', y='y', z='z', 
                                    size='marker_size', color='Neuron_type', opacity=0.7)
                # fig.update_traces(marker_size = 2)
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(
                                  aspectmode='data'))
                fig.write_html(os.path.join(self.mapping_path, f"{self.source_area}_node_{sid}_to_{self.target_area}_closest_excitatory.html"))
                
                
                # Scatter closest excitatory neurons with cylindrical coordinates
                fig = px.scatter_3d(representation_df, x='x_trans', y='y_trans', z='z_trans', 
                    size='marker_size', color='Neuron_type', opacity=0.7)
                # fig.update_traces(marker_size = 2)
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(
                                  aspectmode='data'))
                fig.write_html(os.path.join(self.mapping_path, f"{self.source_area}_node_{sid}_to_{self.target_area}_closest_excitatory_PCA_coordinates.html"))
                
                
                # Scatter all closest neurons
                all_closest_neurons = all_closest_neurons.assign(marker_size = 2)
                representation_df = pd.concat([sid_df, all_closest_neurons], ignore_index=True, sort=False)
                
                fig = px.scatter_3d(representation_df, x='x', y='y', z='z', 
                                    size='marker_size', color='Neuron_type', opacity=0.7)
                # fig.update_traces(marker_size = 2)
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(
                                  aspectmode='data'))
                fig.write_html(os.path.join(self.mapping_path, f"{self.source_area}_node_{sid}_to_{self.target_area}_closest_all.html"))
                
                
                # Scatter all closest neurons with cylindrical coordinates
                fig = px.scatter_3d(representation_df, x='x_trans', y='y_trans', z='z_trans', 
                    size='marker_size', color='Neuron_type', opacity=0.7)
                # fig.update_traces(marker_size = 2)
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(
                                  aspectmode='data'))
                fig.write_html(os.path.join(self.mapping_path, f"{self.source_area}_node_{sid}_to_{self.target_area}_closest_all_PCA_coordinates.html"))
                
                
                # Scatter target neurons with cylindrical coordinates
                edges_subset_df = edges_subset_df.assign(marker_size = 2)
                sid_df = sid_df.assign(Target='Source')
                representation_df = pd.concat([sid_df, edges_subset_df], ignore_index=True, sort=False)
                fig = px.scatter_3d(representation_df, x='x_trans', y='y_trans', z='z_trans', 
                                    size='marker_size', color='Target', opacity=0.7)
                # fig.update_traces(marker_size = 2)
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(
                                  aspectmode='data'))
                fig.write_html(os.path.join(self.mapping_path, f"{self.source_area}_node_{sid}_to_{self.target_area}_targets_PCA_coordinates.html"))
                
                
        file_management.save_lzma(self.edges_df, f'{self.source_area}_to_{self.target_area}_edges_distances.lzma', self.mapping_path)
        
        # # Calculate connection probabilities distributions for every source_target pair
        new_edges_df = self.edges_df[['Source', 'Target', 'planar_intersomatic_distance']].dropna(axis=0)
        
        # # Create path to save the results
        self.gaussian_path = os.path.join(self.mapping_path, 'Gaussian probability')
        os.makedirs(self.gaussian_path, exist_ok=True)
        
        source_pop_names = list(set(new_edges_df['Source']))
        target_pop_names = list(set(new_edges_df['Target']))
        self.sigma_distance_dependence = pd.DataFrame(0, columns=target_pop_names, index=source_pop_names)
        self.connection_probability_origin_amplitude = pd.DataFrame(0, columns=target_pop_names, index=source_pop_names)
        connection_probabilities = dict()

        for source_pop in source_pop_names:
            connection_probabilities[source_pop] = {}
            n_source = n_source_ids.loc[source_pop, '# neurons']
            for target_pop in target_pop_names:
                mask = np.logical_and(new_edges_df['Source']==source_pop, new_edges_df['Target']==target_pop)
                intersomatic_distances = new_edges_df.loc[mask, 'planar_intersomatic_distance'].values
                # Use Freedman_diaconis rule to determine the proper number of bins in the histogram
                NBR_BINS = freedman_diaconis(intersomatic_distances, returnas="bins")

                fig = plt.figure()
                n, bins, patches = plt.hist(intersomatic_distances, bins=NBR_BINS, align='mid')
                plt.title(f'{source_pop} to {target_pop}')
                plt.savefig(os.path.join(self.gaussian_path, f'{source_pop}_to_{target_pop}_histogram.png'), dpi=300, transparent=True)
                plt.close()
                
                centered_bins = bins[:-1] + (bins[1]-bins[0])/2
                areas = np.pi*(bins[1:]**2 - bins[:-1]**2)
                non_zero_bins_mask = (n!=0)
                n = n[non_zero_bins_mask]
                centered_bins = centered_bins[non_zero_bins_mask]
                areas = areas[non_zero_bins_mask]
                
                targets_density = n/(n_source*areas)
                target_pop_density = planar_density[target_pop]
                conn_prob = targets_density/target_pop_density
                popt, pcov = curve_fit(gaussian_func, centered_bins, conn_prob, p0=np.array([1, 100]), method='lm')
                
                # with open(os.path.join(self.gaussian_path, f'{source_pop}_to_{target_pop}_metadata.txt'), 'w')  as f:
                #     f.write(f'Gaussian amplitude: {popt[0]}\n')
                #     f.write(f'Gaussian sigma: {popt[1]}\n')
                #     f.write(f'Parameters covariance: {pcov}')
                
                self.sigma_distance_dependence.loc[source_pop, target_pop] = popt[1]
                self.connection_probability_origin_amplitude.loc[source_pop, target_pop] = popt[0]
                connection_probabilities[source_pop][target_pop] = {'sigma': popt[1], 'amplitude': popt[0]}
                
                ym = gaussian_func(centered_bins, popt[0], popt[1])
                fig = plt.figure()
                plt.plot(centered_bins, conn_prob, '.')
                plt.plot(centered_bins, ym, c='r', label=f'Gaussian sigma: {popt[1]}')
                plt.ylabel('Relative Probability')
                plt.xlabel(r'Planar Intersomatic Distance ($\mu m$)')
                plt.title(f'{source_pop} to {target_pop}')
                plt.savefig(os.path.join(self.gaussian_path, f'{source_pop}_to_{target_pop}_histogram_fit.png'), dpi=300, transparent=True)
                plt.close()
                
        file_management.save_lzma(self.sigma_distance_dependence, f'{self.source_area}_to_{self.target_area}_sigma_distance_dependent.lzma', self.gaussian_path)
        file_management.save_lzma(self.connection_probability_origin_amplitude, f'{self.source_area}_to_{self.target_area}_connection_probability_amplitude.lzma', self.gaussian_path)
        self.sigma_distance_dependence.to_csv(os.path.join(self.gaussian_path, f'{self.source_area}_to_{self.target_area}_sigma_distance_dependent.csv'), sep='\t', float_format="%.4f", decimal=',')
        self.connection_probability_origin_amplitude.to_csv(os.path.join(self.gaussian_path, f'{self.source_area}_to_{self.target_area}_connection_probability_amplitude.csv'), sep='\t', float_format="%.4f", decimal=',')
        
        with open(os.path.join(self.gaussian_path, f'{self.source_area}_to_{self.target_area}_connection_probabilities.json'), 'w') as handle:
            json.dump(connection_probabilities, handle)
                    
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--hemisphere",
                        default='right',
                        help="Brain hemisphere whose ipsilateral connections we will focus in")
    
    parser.add_argument("--source_area",
                        default='VISl',
                        help="Brain area which acts as source")
    
    parser.add_argument("--target_area",
                        default='VISl',
                        help="Brain area which acts as target")
    
    args = parser.parse_args()
    
    PatchAnalysis = ModelAnalysis(source_area=args.source_area, target_area=args.target_area, hemisphere=args.hemisphere)
    PatchAnalysis()
    
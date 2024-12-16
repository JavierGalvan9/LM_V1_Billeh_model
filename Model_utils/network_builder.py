import os
import h5py
import json
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf


def convert_glif_lif_asc_psc(dynamics_params, tau_syns):
    config = dynamics_params
    coeffs = config["coeffs"]
    return {
        "V_m": config["El"] * 1.0e03 + config["El_reference"] * 1.0e03,
        "V_th": coeffs["th_inf"] * config["th_inf"] * 1.0e03
        + config["El_reference"] * 1.0e03,
        "g": coeffs["G"] / config["R_input"] * 1.0e09,
        "E_L": config["El"] * 1.0e03 + config["El_reference"] * 1.0e03,
        "C_m": coeffs["C"] * config["C"] * 1.0e12,
        "t_ref": config["spike_cut_length"] * config["dt"] * 1.0e03,
        "V_reset": config["El_reference"] * 1.0e03,
        "asc_init": list(np.array(config["init_AScurrents"]) * 1.0e12),
        "asc_decay": list(1.0 / np.array(config["asc_tau_array"]) * 1.0e-03),
        "asc_amps": list(
            np.array(config["asc_amp_array"])
            * np.array(coeffs["asc_amp_array"])
            * 1.0e12
        ),
        "tau_syn": tau_syns,
        "spike_dependent_threshold": False,
        "after_spike_currents": True,
        "adapting_threshold": False,
    }


def sort_indices(indices, *arrays):
    # if indices.size == 0:
    #     return (indices, *arrays)
    # else:
    max_ind = np.max(indices) + 1
    if np.iinfo(indices.dtype).max < max_ind * (max_ind + 1) :
        indices = indices.astype(np.int64)
    q = indices[:, 0] * max_ind + indices[:, 1]
    sorted_ind = np.argsort(q)
    sorted_arrays = list(map(lambda arr: arr[sorted_ind], [indices, *arrays]))
    return tuple(sorted_arrays)

# The following function, although it provided a speed up, it creates bunch of tensor copies 
# which are not garbage collected and thus it is not memory efficient
def sort_indices_tf(indices, *arrays):
    # if indices.size == 0:
    #     return (indices, *arrays)
    # else:
    indices = tf.cast(indices, dtype=tf.int64)    
    max_ind = tf.reduce_max(indices) + 1
    q = indices[:, 0] * max_ind + indices[:, 1]
    sorted_ind = tf.argsort(q)
    indices = tf.cast(indices, dtype=tf.int32)
    sorted_arrays = [tf.gather(arr, sorted_ind).numpy() for arr in [indices, *arrays]]
    return tuple(sorted_arrays)

def direction_tuning_factor(src_tuning, tar_tuning, sigma):
    delta_tuning_180 = np.abs(
        np.abs(np.mod(np.abs(tar_tuning - src_tuning), 360.0) - 180.0) - 180.0)
    w_multiplier_180 = np.exp(-(delta_tuning_180 / sigma) ** 2)
    return w_multiplier_180


def DirectionRule_EE(src_tuning, trg_tuning, x_src, z_src, x_trg, z_trg, nsyns_ret, syn_weight, weight_sigma):
    
    w_multiplier_180 = direction_tuning_factor(src_tuning, trg_tuning, weight_sigma)

    delta_x = (x_trg - x_src) * 0.07
    delta_z = (z_trg - z_src) * 0.04

    theta_pref = trg_tuning * (np.pi / 180.)
    xz = delta_x * np.cos(theta_pref) + delta_z * np.sin(theta_pref)
    sigma_phase = 1.0
    phase_scale_ratio = np.exp(- (xz ** 2 / (2 * sigma_phase ** 2)))

    # To account for the 0.07 vs 0.04 dimensions. This ensures the horizontal neurons are scaled by 5.5/4 (from the
    # midpoint of 4 & 7). Also, ensures the vertical is scaled by 5.5/7. This was a basic linear estimate to get the
    # numbers (y = ax + b).
    theta_tar_scale = np.abs(np.abs(np.abs(180.0 - np.mod(np.abs(trg_tuning), 360.0)) - 90.0) - 90.0)
    phase_scale_ratio = phase_scale_ratio * (5.5 / 4.0 - 11.0 / 1680.0 * theta_tar_scale)

    return syn_weight * w_multiplier_180 * phase_scale_ratio * nsyns_ret


def DirectionRule_others(src_tuning, trg_tuning, nsyns_ret, syn_weight, weight_sigma):

    w_multiplier_180 = direction_tuning_factor(src_tuning, trg_tuning, weight_sigma)

    return syn_weight * w_multiplier_180 * nsyns_ret


def add_nodes(data_dir='GLIF_network', source='v1'):
    # Load the nodes file and create a dictionary with the node ids and the node parameters
    node_file = f'network/{source}_nodes.h5'
    nodes_h5_path = os.path.join(data_dir, node_file)
    node_types_file = f'network/{source}_node_types.csv'
    node_types_df = pd.read_csv(os.path.join(data_dir, node_types_file), delimiter=" ")
    cell_models_path = os.path.join(data_dir, "components", "cell_models")

    with h5py.File(nodes_h5_path, "r") as nodes_h5_file:
        source_node_ids = nodes_h5_file["nodes"][source]["node_id"][()].astype(np.uint32)
        source_node_type_ids = nodes_h5_file["nodes"][source]["node_type_id"][()].astype(np.uint32)

    nodes_list = []
    for node_type_id in node_types_df["node_type_id"]:
        mask = source_node_type_ids == node_type_id
        new_pop_dict = {"node_type_id": node_type_id}
        new_pop_dict["ids"] = source_node_ids[mask]
        # Get the node parameters
        node_params_file = os.path.join(cell_models_path, f"{node_type_id}_glif_lif_asc_config.json")
        with open(node_params_file) as f:
            dynamics_params = json.load(f)
        tau_syns = [5.5, 8.5, 2.8, 5.8]
        cell_model_dict = convert_glif_lif_asc_psc(dynamics_params, tau_syns)
        # rename some parameterse
        cell_model_dict["V_reset"] = cell_model_dict.pop("V_m")
        cell_model_dict["k"] = cell_model_dict.pop("asc_decay")
        new_pop_dict["params"] = cell_model_dict
        nodes_list.append(new_pop_dict)
    
    return nodes_list

def add_edges(data_dir='GLIF_network', source='v1', target='v1'):
    edges_list = []
    edge_file = f'network/{source}_{target}_edges.h5'
    edge_types_file = f'network/{source}_{target}_edge_types.csv'
    edges_h5_path = os.path.join(data_dir, edge_file)
    edges_type_df = pd.read_csv(os.path.join(data_dir, edge_types_file), delimiter=" ")
    synaptic_models_path = os.path.join(data_dir, 'components', "synaptic_models")

    with h5py.File(edges_h5_path, "r") as edges_h5_file:
        source_to_target = f"{source}_to_{target}"
        edge_type_ids = edges_h5_file["edges"][source_to_target]["edge_type_id"][()].astype(np.uint32)
        source_node_ids = edges_h5_file["edges"][source_to_target]["source_node_id"][()].astype(np.uint32)
        target_node_ids = edges_h5_file["edges"][source_to_target]["target_node_id"][()].astype(np.uint32)
        nsyns = edges_h5_file["edges"][source_to_target]['0']['nsyns'][()].astype(np.float32)

    # Pre-load node data if applicable
    x, z, tuning_angle = None, None, None
    if source in ['v1', 'lm'] and source == target:
        node_file = f'network/v1_nodes.h5'
        nodes_h5_path = os.path.join(data_dir, node_file)
        with h5py.File(nodes_h5_path, "r") as nodes_h5_file:        
            x = nodes_h5_file["nodes"]['v1']["0"]['x'][()].astype(np.float32)
            z = nodes_h5_file["nodes"]['v1']["0"]['z'][()].astype(np.float32)
            tuning_angle = nodes_h5_file["nodes"]['v1']["0"]['tuning_angle'][()].astype(np.float32)

    for idx, edge_row in edges_type_df.iterrows():
        edge_type_id = edge_row['edge_type_id']
        mask = edge_type_ids == edge_type_id
        src_ids = source_node_ids[mask]
        trg_ids = target_node_ids[mask]
        nsyn_ret = nsyns[mask]
        syn_weight = edges_type_df.loc[idx, 'syn_weight']

        if 'weight_function' in edges_type_df.columns:
            weight_function = edges_type_df.loc[idx, 'weight_function']
            weight_sigma = edges_type_df.loc[idx, 'weight_sigma']
            weight_function = edges_type_df.loc[idx, 'weight_function']
            trg_tuning = tuning_angle[trg_ids]
            src_tuning = tuning_angle[src_ids]
            if weight_function == 'DirectionRule_EE':
                x_trg = x[trg_ids]
                z_trg = z[trg_ids]
                x_src = x[src_ids]
                z_src = z[src_ids]
                edge_weights = DirectionRule_EE(src_tuning, trg_tuning, x_src, z_src, x_trg, z_trg, nsyn_ret, syn_weight, weight_sigma)
            elif weight_function == 'DirectionRule_others':
                edge_weights = DirectionRule_others(src_tuning, trg_tuning, nsyn_ret, syn_weight, weight_sigma)
        else:
            edge_weights = np.full_like(src_ids, syn_weight, dtype=np.float32)*nsyn_ret

        synaptic_model = edge_row['dynamics_params']
        edge_params_file = os.path.join(synaptic_models_path, synaptic_model)
        with open(edge_params_file) as f:
            synaptic_model_dict = json.load(f)
            receptor_type = synaptic_model_dict["receptor_type"]

        new_pop_dict = {
            "edge_type_id": edge_type_id,
            "source": src_ids,
            "target": trg_ids,
            "params": {
                "model": edge_row["model_template"],
                "synaptic_model": synaptic_model.split('.')[0],
                "receptor_type": receptor_type,
                "delay": edge_row["delay"],
                "weight": edge_weights},
        }

        edges_list.append(new_pop_dict)

    return edges_list


def build_input_dat(data_dir='GLIF_network', target='v1', source='lgn', save_pkl=True):
    # Extract the network data into a pickle file from the SONATA files
    new_input_dat = {"nodes": [], "edges": []}

    if target == 'lm':
        target = 'v1'
        filename = f'{source}_lm_network_dat.pkl'
    elif target == 'v1':
        filename = f'{source}_{target}_network_dat.pkl'
    else:
        print('Error: Invalid target node')
     
    new_input_dat["edges"] = add_edges(data_dir, source, target)
    
    if save_pkl:    
        pkl_path = os.path.join(data_dir, filename)
        with open(pkl_path, "wb") as f:
            pkl.dump(new_input_dat, f)

    return new_input_dat


def build_network_dat(data_dir='GLIF_network', source='v1', target='v1', save_pkl=True):
    # Extract the network data into a pickle file from the SONATA files
    new_network_dat = {"nodes": [], "edges": []}

    if source == 'v1' and target == 'v1':
        new_network_dat["nodes"] = add_nodes(data_dir, source)
        new_network_dat["edges"] = add_edges(data_dir, source, target)
    elif source == 'lm' and target == 'lm':
        new_network_dat["nodes"] = add_nodes(data_dir, 'v1')
        new_network_dat["edges"] = add_edges(data_dir, 'v1', 'v1')
    else:
        print('Error: Invalid target node')
        return None

    if save_pkl:    
        if source == target:
            filename = f'{source}_network_dat.pkl'
        else:
            filename = f'{source}_{target}_network_dat.pkl'
        
        pkl_path = os.path.join(data_dir, filename)
        with open(pkl_path, "wb") as f:
            pkl.dump(new_network_dat, f)

    return new_network_dat

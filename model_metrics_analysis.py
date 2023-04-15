# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:28:39 2022

@author: UX325
"""

import os
import sys
import absl
import json
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# import plotting_figures as myplots
parentDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management
sys.path.append(os.path.join(os.getcwd(), "Model_utils"))
import other_billeh_utils
import load_sparse

mpl.style.use('default')
rd = np.random.RandomState(seed=42)


def calculateFiringRate(z, drifting_gratings_init=500):

    # print("Calculating Firing Rate")
    dg_spikes = z[:, drifting_gratings_init:, :]
    mean_dg_spikes = np.mean(dg_spikes, axis=0)
    mean_firing_rates = np.sum(mean_dg_spikes, axis=0)/2.0
    
    return mean_firing_rates

def pop_name_to_cell_type(pop_name):
    """convert pop_name in the old format to cell types.
    for example,
    'e4Rorb' -> 'L4 Exc'
    'i4Pvalb' -> 'L4 PV'
    'i23Sst' -> 'L2/3 SST'
    """
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
        subclass = "VIP"
    else:  # excitatory
        subclass = "Exc"

    return f"L{layer} {subclass}"

def neuropixels_cell_type_to_cell_type(pop_name):
    """convert pop_name in the neuropixels cell type to cell types.
    for example,
    'EXC_L23' -> 'L2/3 Exc'
    'PV_L5' -> 'L5 PV'
    """
    layer = pop_name.split('_')[1]
    class_name = pop_name.split('_')[0]
    if "2" in layer:
        layer = "L2/3"
    elif layer == "L1":
        return "L1 Htr3a"  # special case
    if class_name == "EXC":
        class_name = "Exc"

    return f"{layer} {class_name}"

def get_borders(ticklabel):
    prev_layer = "1"
    borders = [-0.5]
    for i in ticklabel:
        x = i.get_position()[0]
        text = i.get_text()
        if text[1] != prev_layer:
            # detected layer change
            borders.append(x - 0.5)
            prev_layer = text[1]
    borders.append(x + 0.5)
    return borders


def draw_borders(ax, borders, ylim):
    for i in range(0, len(borders), 2):
        w = borders[i + 1] - borders[i]
        h = ylim[1] - ylim[0]
        ax.add_patch(
            Rectangle((borders[i], ylim[0]), w, h, alpha=0.08, color="k", zorder=-10)
        )
    return ax


def plot_one(ax, df, metric_name, ylim, cpal=None):
    sns.boxplot(
        x="cell_type",
        y=metric_name,
        hue="data_type",
        data=df,
        ax=ax,
        width=0.7,
        palette=cpal,
    )
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(loc="upper right")
    ax.set_ylim(ylim)
    ax.set_xlabel("")
    # apply shadings to each layer
    ticklabel = ax.get_xticklabels()
    borders = get_borders(ticklabel)
    draw_borders(ax, borders, ylim)
    return ax


def calculate_OSI_DSI_from_DF(rates_df, network, area='V1', angles=range(0,360, 45), basedir=''):
    num_angles = len(angles)
    num_neurons = len(rates_df) // num_angles
    all_rates = np.zeros((num_neurons, num_angles))

    angle_counts = 0
    for angle, g in rates_df.groupby("DG_angle"):
        all_rates[:, angle_counts] = g["Avg_rate(Hz)"]
        angle_counts += 1

    preferred_angle_ind = np.argmax(all_rates, axis=1)
    preferred_rates = all_rates[
        np.arange(len(preferred_angle_ind)), preferred_angle_ind
    ]

    phase_rad = np.array(angles) * math.pi / 180

    dsi = np.abs(
        (all_rates * np.exp(1j * phase_rad)).sum(axis=1) / all_rates.sum(axis=1)
    )
    osi = np.abs(
        (all_rates * np.exp(2j * phase_rad)).sum(axis=1) / all_rates.sum(axis=1)
    )

    osi_df = pd.DataFrame()
    osi_df["node_id"] = range(num_neurons)
    osi_df["pop_name"] = other_billeh_utils.pop_names(network)
    osi_df["DSI"] = dsi
    osi_df["OSI"] = osi
    osi_df["preferred_angle"] = np.array(angles)[preferred_angle_ind]
    osi_df["max_mean_rate(Hz)"] = preferred_rates
    osi_df["Avg_Rate(Hz)"] = np.mean(all_rates, axis=1)

    osi_df.to_csv(os.path.join(basedir, f"{area}_OSI_DSI_DF.csv"), sep=" ", index=False)

    return osi_df


def get_osi_df(network=None, area='', basedir='', metric_file="OSI_DSI_DF.csv", metric_name="", radius=None):
    # special case is neuropixels experimental data
    if basedir == "neuropixels":
        df = pd.read_csv(f"Neuropixels_data/{area}_{metric_file}", sep=" ")
        df = df[df['cell_type'].notna()]
        df["cell_type"] = df["cell_type"].apply(neuropixels_cell_type_to_cell_type)
    else:
        df = pd.read_csv(f"{basedir}/{area}_{metric_file}", sep=" ")
        df["cell_type"] = df["pop_name"].apply(pop_name_to_cell_type)
        # df = pick_core(df, radius=radius)
        # Analyze only neurons within a radius
        if radius:
            core_mask = other_billeh_utils.isolate_core_neurons(network, radius=radius)
            df = df[core_mask]

    df.rename(
        columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"}, inplace=True
    )

    # exclude OSI and DSI for neurons with <0.5 Hz preferred direction response
    nonresponding = df["Rate at preferred direction (Hz)"] < 0.5
    df.loc[nonresponding, "OSI"] = np.nan
    df.loc[nonresponding, "DSI"] = np.nan

    df = df.sort_values(by="cell_type")

    if len(metric_name) > 0:
        df["data_type"] = metric_name
    else:
        df["data_type"] = basedir
    return df


def Billeh_style_boxplot(network, area, basedir='', radius=None):
    osi_dfs = []

    color_pal = {
        "LM/V1 GLIF model": "tab:orange",
        "Neuropixels": "tab:gray",
    }
    osi_dfs.append(get_osi_df(network, area=area, basedir=basedir, metric_file=f"OSI_DSI_DF.csv", metric_name="LM/V1 GLIF model", radius=radius))
    osi_dfs.append(get_osi_df(area=area, basedir="neuropixels", metric_file="OSI_DSI_DF.csv", metric_name="Neuropixels"))

    df = pd.concat(osi_dfs)
    
    print(df)

    fig, axs = plt.subplots(3, 1, figsize=(12, 20))
    plot_one(axs[0], df, "Rate at preferred direction (Hz)", [0, 100], color_pal)
    plot_one(axs[1], df, "DSI", [0, 1], color_pal)
    plot_one(axs[2], df, "OSI", [0, 1], color_pal)
    # plot_scat(axs[3], df, "Rate at preferred direction (Hz)", "DSI", color_pal)

    plt.tight_layout()
    plt.savefig(os.path.join(basedir, f'{area}_metrics.png'), dpi=300, transparent=True)



class ModelMetricsAnalysis:    
    def __init__(self, V1_neurons, LM_neurons, analyze_core_only = True,
                 data_dir='GLIF_network', simulation_length=2500, n_simulations=None, 
                 skip_first_simulation=False):

        self.V1_neurons = V1_neurons
        self.LM_neurons = LM_neurons
        self.simulation_length = simulation_length
        self.n_simulations = n_simulations
        self.skip_first_simulation = skip_first_simulation
        self.analyze_core_only = analyze_core_only
        self.data_dir = data_dir
        self.save_dir = 'Metrics_analysis'
        os.makedirs(self.save_dir, exist_ok=True)

        
    def calculate_DG_Rates_DF(self, n_neurons, area='V1', n_simulations=10, angles=np.arange(0, 360, 45)):

        Rates_DF = pd.DataFrame(
            index=range(n_neurons * len(angles)),
            columns=["DG_angle", "node_id", "Avg_rate(Hz)"],
        )
        
        for i, ori in enumerate(angles): 
            directory = f'orien_{str(ori)}_freq_2_V1_{self.V1_neurons}_LM_{self.LM_neurons}'
            full_data_path = os.path.join('Simulation_results', directory, 'Data')
            # if the file is not found, skip it
            try:
                spikes, _ = other_billeh_utils.load_simulation_results(full_data_path, area=area, n_simulations = self.n_simulations, 
                                                                       skip_first_simulation=self.skip_first_simulation, 
                                                                       variables='z', simulation_length=self.simulation_length, 
                                                                       n_neurons=n_neurons)
                firingRates = calculateFiringRate(spikes, drifting_gratings_init=500)
            except FileNotFoundError:
                print(f"File not found: orientation_{ori}_n_simulations_{n_simulations} .  Skipping...")
                firingRates = np.nan
                # the value remains nan, and ignored in the mean calculation
                continue
    
            Rates_DF.loc[i * n_neurons : (i + 1) * n_neurons - 1, "DG_angle"] = ori
            Rates_DF.loc[i * n_neurons : (i + 1) * n_neurons - 1, "node_id"] = np.arange(
                n_neurons
            )
            Rates_DF.loc[i * n_neurons : (i + 1) * n_neurons - 1, "Avg_rate(Hz)"] = firingRates
    
        Rates_DF.to_csv(os.path.join(self.save_dir, f"{area}_DG_Rates_DF.csv"), sep=" ", index=False)
        return Rates_DF
    
    
    def __call__(self):
        flags = absl.app.flags.FLAGS
        
        # self.full_data_path = os.path.join(self.full_path, 'Data')
        n_neurons = {'V1': self.V1_neurons,
                     'LM': self.LM_neurons}
        
        load_fn = load_sparse.cached_load_billeh
        input_population, networks, bkg_weights, n_input = load_fn(flags, n_neurons)
        
        # Calculate the firing rates along every orientation
        angles=np.arange(0, 360, 45)
        v1_rates_df = self.calculate_DG_Rates_DF(self.V1_neurons, area='V1', n_simulations=self.n_simulations, angles=angles)
        lm_rates_df = self.calculate_DG_Rates_DF(self.LM_neurons, area='LM', n_simulations=self.n_simulations, angles=angles)
        
        # Calculate the orientation and direction selectivity indices
        calculate_OSI_DSI_from_DF(v1_rates_df, networks['V1'], area='V1', angles=angles, basedir=self.save_dir)
        calculate_OSI_DSI_from_DF(lm_rates_df, networks['LM'], area='LM', angles=angles, basedir=self.save_dir)

        # Make the boxplots to compare with the neuropixels data
        if self.analyze_core_only:
            LM_core_radius = 151.1
            V1_core_radius = 400
        else:
            V1_core_radius = V1_core_radius = None
            
        Billeh_style_boxplot(networks['V1'], 'V1', basedir=self.save_dir, radius=V1_core_radius)
        Billeh_style_boxplot(networks['LM'], 'LM', basedir=self.save_dir, radius=LM_core_radius)
        

def main(_):
    flags = absl.app.flags.FLAGS    
    model_analysis = ModelMetricsAnalysis(flags.V1_neurons, flags.LM_neurons, analyze_core_only=flags.analyze_core_only,
                                          data_dir=flags.data_dir, simulation_length=flags.simulation_length,
                                          n_simulations=flags.n_simulations, skip_first_simulation=flags.skip_first_simulation
                                          )
    model_analysis()
    
if __name__ == '__main__':
    
    absl.app.flags.DEFINE_integer('V1_neurons', 230924, '')
    absl.app.flags.DEFINE_integer('LM_neurons', 32940, '')
    absl.app.flags.DEFINE_integer('simulation_length', 2500, '')
    absl.app.flags.DEFINE_integer('n_simulations', None, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_boolean('skip_first_simulation', True, '')
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('analyze_core_only', True, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_string('data_dir', 'GLIF_network', '')
    
    
    absl.app.run(main)  
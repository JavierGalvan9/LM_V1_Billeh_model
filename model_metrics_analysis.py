# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:28:39 2022

@author: UX325
"""

import os
import sys
import absl
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
sys.path.append(os.path.join(os.getcwd(), "Model_utils"))
import other_billeh_utils
import load_sparse

mpl.style.use('default')
rd = np.random.RandomState(seed=42)


class MetricsBoxplot:
    def __init__(self, area, save_dir='Metrics_analysis'):
        self.area = area
        self.save_dir = save_dir
        self.osi_dfs = []

    @staticmethod
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
            subclass = "VIP"
        else:  # excitatory
            subclass = "Exc"

        return f"L{layer} {subclass}"

    @staticmethod
    def neuropixels_cell_type_to_cell_type(pop_name):
        # Convert pop_name in the neuropixels cell type to cell types. E.g, 'EXC_L23' -> 'L2/3 Exc', 'PV_L5' -> 'L5 PV'
        layer = pop_name.split('_')[1]
        class_name = pop_name.split('_')[0]
        if "2" in layer:
            layer = "L2/3"
        elif layer == "L1":
            return "L1 Htr3a"  # special case
        if class_name == "EXC":
            class_name = "Exc"

        return f"{layer} {class_name}"

    @staticmethod
    def get_borders(ticklabel):
        prev_layer = "1"
        borders = [-0.5]
        for i in ticklabel:
            x = i.get_position()[0]
            text = i.get_text()
            if text[1] != prev_layer:
                borders.append(x - 0.5)
                prev_layer = text[1]
        borders.append(x + 0.5)
        return borders

    @staticmethod
    def draw_borders(ax, borders, ylim):
        for i in range(0, len(borders), 2):
            w = borders[i + 1] - borders[i]
            h = ylim[1] - ylim[0]
            ax.add_patch(
                Rectangle((borders[i], ylim[0]), w, h, alpha=0.08, color="k", zorder=-10)
            )
        return ax

    @staticmethod
    def plot_one_metric(ax, df, metric_name, ylim, cpal=None):
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
        ax.set_ylim(ylim)
        ax.set_xlabel("")
        # Modify the label sizes
        ax.set_ylabel(metric_name, fontsize=25)
        yticklabel = ax.get_yticklabels()
        for label in yticklabel:
            label.set_fontsize(25)

        # Apply shadings to each layer
        xticklabel = ax.get_xticklabels()
        borders = MetricsBoxplot.get_borders(xticklabel)
        MetricsBoxplot.draw_borders(ax, borders, ylim)

        # Hide the legend
        ax.get_legend().remove()

        return ax

    def get_osi_dsi_df(self, metric_file="V1_OSI_DSI_DF.csv", data_source_name="", data_dir=""):
        # Load the data csv file and remove rows with empty cell type
        df = pd.read_csv(f"{data_dir}/{metric_file}", sep=" ")
        
        # Rename the cell types
        if data_dir == "Neuropixels_data":
            df = df[df['cell_type'].notna()]
            df["cell_type"] = df["cell_type"].apply(self.neuropixels_cell_type_to_cell_type)
        elif data_dir == 'Billeh_column_metrics':
            df["cell_type"] = df["pop_name"].apply(self.pop_name_to_cell_type)
        else:
            df["cell_type"] = df["pop_name"].apply(self.pop_name_to_cell_type)

        # Rename the maximum rate column
        df.rename(columns={"max_mean_rate(Hz)": "Rate at preferred direction (Hz)"}, inplace=True)

        # Cut off neurons with low firing rate at the preferred direction
        nonresponding = df["Rate at preferred direction (Hz)"] < 0.5
        df.loc[nonresponding, "OSI"] = np.nan
        df.loc[nonresponding, "DSI"] = np.nan

        # Sort the neurons by neuron types
        df = df.sort_values(by="cell_type")

        # Add a column for the data source name
        if len(data_source_name) > 0:
            df["data_type"] = data_source_name
        else:
            df["data_type"] = data_dir

        return df

    def plot(self, metrics=["Rate at preferred direction (Hz)", "OSI", "DSI"]):
        # Get the dataframes for the model and Neuropixels OSI and DSI 
        self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"{self.area}_OSI_DSI_DF.csv", data_source_name="LM/V1 GLIF model", data_dir=self.save_dir))
        self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"{self.area}_OSI_DSI_DF.csv", data_source_name="Neuropixels", data_dir='Neuropixels_data'))
        if self.area == 'V1':
            self.osi_dfs.append(self.get_osi_dsi_df(metric_file=f"{self.area}_OSI_DSI_DF.csv", data_source_name="Billeh et al (2020)", data_dir='Billeh_column_metrics'))
        
        df = pd.concat(self.osi_dfs)
        # df.to_csv(os.path.join(self.save_dir, f"help_DG_firing_rates_df.csv"), sep=" ", index=False)

        # Create a figure to compare several model metrics against Neuropixels data
        n_metrics = len(metrics)
        height = int(7*n_metrics)
        fig, axs = plt.subplots(n_metrics, 1, figsize=(12, height))
        # fig, axs = plt.subplots(n_metrics, 1, figsize=(12, 20))

        color_pal = {
            "LM/V1 GLIF model": "tab:orange",
            "Neuropixels": "tab:gray",
            "Billeh et al (2020)": "tab:blue"
        }

        for idx, metric in enumerate(metrics):
            if metric == "Rate at preferred direction (Hz)":
                ylims = [0, 100]
            else:
                ylims = [0, 1]

            self.plot_one_metric(axs[idx], df, metric, ylims, color_pal)

        axs[0].legend(loc="upper right", fontsize=20)
        axs[0].set_xticklabels([])
        axs[1].set_xticklabels([])
        xticklabel = axs[n_metrics-1].get_xticklabels()
        for label in xticklabel:
            label.set_fontsize(25)
            label.set_weight("bold")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{self.area}_metrics.png'), dpi=1000, transparent=True)
        plt.close()


def calculate_Firing_Rate(z, drifting_gratings_init=500, drifting_gratings_end=1500):
    dg_spikes = z[:, drifting_gratings_init:drifting_gratings_end, :]
    mean_dg_spikes = np.mean(dg_spikes, axis=0)
    mean_firing_rates = np.sum(mean_dg_spikes, axis=0)/((drifting_gratings_end-drifting_gratings_init)/1000)
    
    return mean_firing_rates

def calculate_OSI_DSI(rates_df, network, DG_angles=range(0,360, 45), core_radius=None, remove_zero_rate_neurons=True):
    
    # Get the pop names of the neurons
    if core_radius is not None:
        pop_names = other_billeh_utils.pop_names(network, core_radius=core_radius) 
    else:
        pop_names = other_billeh_utils.pop_names(network)

    # Get the number of neurons and DG angles
    n_neurons = len(pop_names)
    node_ids = np.arange(n_neurons)
    
    # Get the firing rates for every neuron and DG angle
    all_rates = np.array([g["Avg_rate(Hz)"] for _, g in rates_df.groupby("DG_angle")]).T
    average_rates = np.mean(all_rates, axis=1)

    # Find the preferred DG angle for each neuron
    preferred_angle_ind = np.argmax(all_rates, axis=1)
    preferred_rates = np.max(all_rates, axis=1)
    preferred_DG_angle = np.array(DG_angles)[preferred_angle_ind]

    # Calculate the DSI and OSI
    phase_rad = np.array(DG_angles) * math.pi / 180.0
    denominator = all_rates.sum(axis=1)
    dsi = np.abs((all_rates * np.exp(1j * phase_rad)).sum(axis=1) / denominator)
    osi = np.abs((all_rates * np.exp(2j * phase_rad)).sum(axis=1) / denominator)

    # Save the results in a dataframe
    osi_df = pd.DataFrame()
    osi_df["node_id"] = node_ids
    osi_df["pop_name"] = pop_names
    osi_df["DSI"] = dsi
    osi_df["OSI"] = osi
    osi_df["preferred_angle"] = preferred_DG_angle
    osi_df["max_mean_rate(Hz)"] = preferred_rates
    osi_df["Avg_Rate(Hz)"] = average_rates

    if remove_zero_rate_neurons:
        osi_df = osi_df[osi_df["Avg_Rate(Hz)"] != 0]

    return osi_df

class ModelMetricsAnalysis:    

    def __init__(self, flags):
        self.flags = flags
        self.V1_neurons = flags.V1_neurons
        self.LM_neurons = flags.LM_neurons
        self.gratings_frequency = flags.gratings_frequency
        self.data_dir = flags.data_dir 
        self.n_simulations = flags.n_simulations 
        self.skip_first_simulation = flags.skip_first_simulation
        self.analyze_core_only = flags.analyze_core_only

        # Create a dictionary with the number of neurons in each area
        self.n_neurons = {'V1': self.V1_neurons, 'LM': self.LM_neurons}

        # Define the simulation data path
        # self.simulation_results_path = f'Simulation_results/V1_{flags.V1_neurons}_LM_{flags.LM_neurons}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}'
        simulation_results_path = f'Simulation_results/V1_{flags.V1_neurons}_LM_{flags.LM_neurons}'
        for name, value in flags.flag_values_dict().items():
            if value != flags[name].default and name not in ['save_dir', 'v', 'verbosity', 'n_simulations', 'caching', 'V1_neurons', 'LM_neurons', 'gratings_orientation', 'gratings_frequency']:
                simulation_results_path += f'_{name}_{value}'
        self.simulation_results_path = simulation_results_path
        # self.simulation_results_path = os.path.join(simulation_results_path, f'orien_{flags.gratings_orientation}_freq_{flags.gratings_frequency}')
        print('Simulation results path: ')
        print(simulation_results_path)

        # Create a path to save the analysis results
        # self.save_dir = f'Metrics_analysis/V1_{flags.V1_neurons}_LM_{flags.LM_neurons}_s{flags.seed}_c{flags.core_only}_con{flags.connected_selection}'
        self.save_dir = os.path.join(simulation_results_path, 'Metrics_analysis')
        os.makedirs(self.save_dir, exist_ok=True)
    
    def __call__(self, area):
        # Load the network and input parameters
        load_fn = load_sparse.cached_load_billeh
        input_population, networks, bkg_weights, n_input = load_fn(self.flags, self.n_neurons)
        network = networks[area]
        n_neurons = self.n_neurons[area]

        # Isolate the core neurons if necessary
        if self.analyze_core_only:
            if area == 'V1':
                core_neurons = 51978
                core_radius = 400
            elif area == 'LM':
                core_neurons = 7285
                V1_to_LM_neurons_ratio = 7.010391285652859
                core_radius = 400/np.sqrt(V1_to_LM_neurons_ratio)
            else:
                raise ValueError(f"Invalid area: {area}")
            
            # Calculate the core_neurons mask
            if n_neurons > core_neurons:
                self.core_mask = other_billeh_utils.isolate_core_neurons(network, radius=core_radius, data_dir=self.data_dir) 
                n_neurons = core_neurons
            else:
                self.core_mask = np.full(n_neurons, True)

        else:
            self.core_mask = np.full(n_neurons, True)
       
        # Calculate the firing rates along every orientation
        DG_angles = np.arange(0, 360, 45)
        firing_rates_df = self.create_firing_rates_df(n_neurons, area=area, n_simulations=self.n_simulations, DG_angles=DG_angles)
        
        # Save the firing rates
        firing_rates_df.to_csv(os.path.join(self.save_dir, f"{area}_DG_firing_rates_df.csv"), sep=" ", index=False)
        print(firing_rates_df)

        # Calculate the orientation and direction selectivity indices
        metrics_df = calculate_OSI_DSI(firing_rates_df, network, DG_angles=DG_angles, core_radius=core_radius)
        metrics_df.to_csv(os.path.join(self.save_dir, f"{area}_OSI_DSI_DF.csv"), sep=" ", index=False)
        print(metrics_df)

        # Make the boxplots to compare with the neuropixels data
        metrics = ["Rate at preferred direction (Hz)", "OSI", "DSI"]
        boxplot = MetricsBoxplot(area, save_dir=self.save_dir)
        boxplot.plot(metrics)

    def create_firing_rates_df(self, n_neurons, area='V1', n_simulations=10, DG_angles=np.arange(0, 360, 45)):
        
        # Calculate the firing rates for each neuron in each orientation
        firing_rates_df = pd.DataFrame(columns=["DG_angle", "node_id", "Avg_rate(Hz)"])
        node_ids = np.arange(n_neurons)

        # Iterate through each orientation
        for angle in DG_angles: 
            # Load the simulation results
            directory = f'orien_{angle}_freq_{self.gratings_frequency}'
            full_data_path = os.path.join(self.simulation_results_path, directory, 'Data', 'simulation_data.hdf5')
            
            try:
                sim_data, sim_metadata, _ = other_billeh_utils.load_simulation_results_hdf5(full_data_path, n_simulations=self.n_simulations,  
                                                                                            skip_first_simulation=self.skip_first_simulation, 
                                                                                            variables='z')
                # Calculate the firing rates
                spikes = sim_data[area]['z'][:, :, self.core_mask]
                firingRates = calculate_Firing_Rate(spikes, drifting_gratings_init=500, drifting_gratings_end=3000)

                data = {
                        "DG_angle": angle,
                        "node_id": node_ids,
                        "Avg_rate(Hz)": firingRates
                    }
                df = pd.DataFrame(data)
                firing_rates_df = pd.concat([firing_rates_df, df], ignore_index=True)

            except FileNotFoundError:
                print(f"File not found: orientation_{angle}_n_simulations_{n_simulations} .  Skipping...")
                continue

        return firing_rates_df
        

def main(_):
    flags = absl.app.flags.FLAGS    
    model_analysis = ModelMetricsAnalysis(flags)
    model_analysis('V1')
    model_analysis('LM')
    
if __name__ == '__main__':
    
    absl.app.flags.DEFINE_integer('V1_neurons', 230924, '')
    absl.app.flags.DEFINE_integer('LM_neurons', 32940, '')
    absl.app.flags.DEFINE_integer('gratings_frequency', 2, '')
    absl.app.flags.DEFINE_integer('n_simulations', None, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_boolean('skip_first_simulation', False, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('caching', True, '')
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('hard_reset', False, '')
    absl.app.flags.DEFINE_string('interarea_weight_distribution', 'billeh_like', '')
    absl.app.flags.DEFINE_boolean('analyze_core_only', True, '')
    absl.app.flags.DEFINE_float('E4_weight_factor', 1., '')
    absl.app.flags.DEFINE_boolean('disconnect_LM_L6_inhibition', False, '')
    absl.app.flags.DEFINE_integer('n_input', 17400, '')
    absl.app.flags.DEFINE_integer('seq_len', 3000, '')
    absl.app.flags.DEFINE_string('data_dir', 'GLIF_network', '')
    
    
    absl.app.run(main)  
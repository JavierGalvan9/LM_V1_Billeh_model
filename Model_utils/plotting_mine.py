import os
import numpy as np
import pandas as pd
from matplotlib import patches
import matplotlib.pyplot as plt
import h5py
import toolkit


class InputActivityFigure:
    def __init__(self, networks, data_dir, images_dir='Images', filename="Raster_plot", batch_ind=0, scale=3., frequency=2, drifting_init_time=500, drifting_end_time=1500, reverse=False):
        self.figure = plt.figure(
            figsize=toolkit.cm2inch((15 * scale, 11 * scale)))
        gs = self.figure.add_gridspec(11, 1)
        self.input_ax = self.figure.add_subplot(gs[:2])
        self.V1_activity_ax = self.figure.add_subplot(gs[2:6])
        self.LM_activity_ax = self.figure.add_subplot(gs[6:-1])
        self.drifting_grating_ax = self.figure.add_subplot(gs[-1])

        self.inputs_plot = RasterPlot(
            batch_ind=batch_ind, scale=scale, y_label='LGN Neuron ID', alpha=.05)
        self.V1_laminar_plot = LaminarPlot(
            networks['V1'], data_dir, area_name='V1', batch_ind=batch_ind, scale=scale, alpha=.2)
        self.LM_laminar_plot = LaminarPlot(
            networks['LM'], data_dir, area_name='LM', batch_ind=batch_ind, scale=scale, alpha=.2)
        self.drifting_grating_plot = DriftingGrating(frequency=frequency, drifting_init_time=drifting_init_time, drifting_end_time=drifting_end_time, reverse=reverse, scale=scale)

        self.tightened = True  # False
        self.scale = scale
        self.networks = networks
        self.n_neurons = self.networks['V1']['n_nodes'] + self.networks['LM']['n_nodes']
        self.batch_ind = batch_ind
        self.images_dir = images_dir
        self.filename = filename

    def __call__(self, inputs, V1_spikes, LM_spikes):
        self.input_ax.clear()
        self.V1_activity_ax.clear()
        self.LM_activity_ax.clear()
        self.drifting_grating_ax.clear()

        self.inputs_plot(self.input_ax, inputs)
        self.input_ax.set_xticklabels([])
        toolkit.apply_style(self.input_ax, scale=self.scale)

        self.V1_laminar_plot(self.V1_activity_ax, V1_spikes)
        self.V1_activity_ax.set_xticklabels([])
        toolkit.apply_style(self.V1_activity_ax, scale=self.scale)
        
        self.LM_laminar_plot(self.LM_activity_ax, LM_spikes)
        self.LM_activity_ax.set_xticklabels([])
        toolkit.apply_style(self.LM_activity_ax, scale=self.scale)
        
        self.drifting_grating_plot(self.drifting_grating_ax, V1_spikes)
        toolkit.apply_style(self.drifting_grating_ax, scale=self.scale)
        
        if not self.tightened:
            self.figure.tight_layout()
            self.tightened = True

        self.figure.savefig(os.path.join(self.images_dir, self.filename), dpi=300)

        return self.figure


def pop_ordering(x):
    if x[1:3].count('23') > 0:  # count('str') finds if the string belongs to the given string
        # Those neurons belonging to layers 2/3 assign then to layer 2 by default (representation purposes)
        p_c = 2  # p_c represents the layer number
    else:
        p_c = int(x[1:2])
    if x[0] == 'e':
        inter_order = 4  # inter_order represents the neurons type order inside the layer
    elif x.count('Htr') > 0:
        inter_order = 1
    elif x.count('Sst') > 0:
        inter_order = 2
    elif x.count('Pvalb') > 0:
        inter_order = 3
    else:
        print(x)
        raise ValueError()
    ordering = p_c * 10 + inter_order
    return ordering


class RasterPlot:
    def __init__(self, batch_ind=0, scale=2., marker_size=1., alpha=.03, color='r', y_label='Neuron ID'):
        self.batch_ind = batch_ind
        self.scale = scale
        self.marker_size = marker_size
        self.alpha = alpha
        self.color = color
        self.y_label = y_label

    def __call__(self, ax, spikes):
        # This method plots the spike train (spikes) that enters the network
        n_elements = np.prod(spikes.shape)
        non_binary_frac = np.sum(np.logical_and(
            spikes > 1e-3, spikes < 1 - 1e-3)) / n_elements
        if non_binary_frac > .01:
            rate = -np.log(1 - spikes[self.batch_ind] / 1.3) * 1000
            # rate = rate.reshape((rate.shape[0], int(rate.shape[1] / 100), 100)).mean(-1)
            p = ax.pcolormesh(rate.T, cmap='cividis')
            toolkit.do_inset_colorbar(ax, p, '')
            ax.set_ylim([0, rate.shape[-1]])
            ax.set_yticks([0, rate.shape[-1]])
            # ax.set_yticklabels([0, rate.shape[-1] * 100])
            ax.set_yticklabels([0, rate.shape[-1]])
            ax.set_ylabel(self.y_label, fontsize=20)
        else:
            # Take the times where the spikes occur
            times, ids = np.where(spikes[self.batch_ind].astype(np.float) > .5)
            ax.plot(times, ids, '.', color=self.color,
                    ms=self.marker_size, alpha=self.alpha)
            ax.set_ylim([0, spikes.shape[-1]])
            ax.set_yticks([0, spikes.shape[-1]])
            ax.set_ylabel(self.y_label, fontsize=20)

        ax.axvline(500, linestyle='dashed', color='k', linewidth=1.5, alpha=1)
        ax.axvline(1500, linestyle='dashed', color='k', linewidth=1.5, alpha=1)
        ax.set_xlim([0, spikes.shape[1]])
        ax.set_xticks([0, spikes.shape[1]])
        ax.tick_params(axis='both', which='major', labelsize=18)


class LaminarPlot:
    def __init__(self, network, data_dir, area_name='V1', batch_ind=0, scale=2., marker_size=1., alpha=.2):
        self.batch_ind = batch_ind
        self.scale = scale
        self.marker_size = marker_size
        self.alpha = alpha
        self.area_name = area_name

        node_types = pd.read_csv(os.path.join(
            data_dir, f'network/{self.area_name}_node_types.csv'), sep=' ')
        path_to_h5 = os.path.join(data_dir, f'network/{self.area_name}_nodes.h5')
        node_h5 = h5py.File(path_to_h5, mode='r')

        node_type_id_to_pop_name = dict()
        for nid in np.unique(node_h5['nodes']['v1']['node_type_id']): 
            # if not np.unique all of the 230924 model neurons ids are considered, 
            # but nearly all of them are repeated since there are only 111 different indices
            ind_list = np.where(node_types.node_type_id == nid)[0]
            assert len(ind_list) == 1
            node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]
            
        node_type_ids = np.array(node_h5['nodes']['v1']['node_type_id'])
        true_pop_names = []  # it contains the pop_name of all the 230,924 neurons
        for nid in node_h5['nodes']['v1']['node_type_id']:
            true_pop_names.append(node_type_id_to_pop_name[nid])
        
        # Select population names of neurons in the present network (core)
        true_pop_names = np.array(true_pop_names)[network['tf_id_to_bmtk_id']]
        true_node_type_ids = node_type_ids[network['tf_id_to_bmtk_id']]
        
        # Now order the pop_names according to their layer and type
        pop_orders = dict(sorted(node_type_id_to_pop_name.items(), key=lambda item: pop_ordering(item[1])))
        
        # We can have an array with the population names in order as follows:
        n_neurons = network['n_nodes']
        self.network = network
        self.n_neurons = n_neurons

        # Now we convert the neuroon id (related to its pop_name) to an index related to its position in the y axis
        neuron_id_to_y = np.zeros(n_neurons, np.int32) - 1 # rest 1 to check at the end if every neuron has an index
        current_ind = 0

        self.e_mask = np.zeros(n_neurons, bool)
        self.htr3_mask = np.zeros(n_neurons, bool)
        self.sst_mask = np.zeros(n_neurons, bool)
        self.pvalb_mask = np.zeros(n_neurons, bool)

        layer_bounds = []
        ie_bounds = []
        current_pop_name = 'e0'

        for pop_id, pop_name in pop_orders.items():
            # choose all the neurons of the given pop_id
            sel = true_node_type_ids == pop_id
            _n = np.sum(sel)
            # order the neurons by type in the y axis
            neuron_id_to_y[sel] = np.arange(current_ind, current_ind + _n)

            if int(pop_name[1]) > int(current_pop_name[1]):
                # register the change of layer
                layer_bounds.append(current_ind)
            if current_pop_name[0] == 'i' and pop_name[0] == 'e':
                # register the change of neuron type: exc -> inh
                ie_bounds.append(current_ind)

            # #Now introduce the masks for the different neuron types
            if pop_name[0] == 'e':
                self.e_mask = np.logical_or(self.e_mask, sel)
            elif pop_name.count('Htr3') > 0:
                self.htr3_mask = np.logical_or(self.htr3_mask, sel)
            elif pop_name.count('Sst') > 0:
                self.sst_mask = np.logical_or(self.sst_mask, sel)
            elif pop_name.count('Pvalb') > 0:
                self.pvalb_mask = np.logical_or(self.pvalb_mask, sel)
            else:
                raise ValueError(f'Unknown population {pop_name}')
            current_ind += _n
            current_pop_name = pop_name
        # check that an y id has been given to every neuron
        assert np.sum(neuron_id_to_y < 0) == 0
        self.layer_bounds = layer_bounds

        ######### For l5e neurons  ###########
        # l5e_min, l5e_max = ie_bounds[-2], layer_bounds[-1]
        # n_l5e = l5e_max - l5e_min

        # n_readout_pops = network['readout_neuron_ids'].shape[0]
        # dist = int(n_l5e / n_readout_pops)
        # #####################################

        y_to_neuron_id = np.zeros(n_neurons, np.int32)
        y_to_neuron_id[neuron_id_to_y] = np.arange(n_neurons)
        assert np.all(y_to_neuron_id[neuron_id_to_y] == np.arange(n_neurons))
        # y_to_neuron_id: E.g., la neurona séptima por orden de capas tiene id 0, y_to_neuron_id[7]=0
        # neuron_id_to_y: E.g., la neurona con id 0 es la séptima por orden de capas, neuron_id_to_y[0] = 7
        
        # ##### For l5e neurons #####
        # neurons_per_readout = network['readout_neuron_ids'].shape[1]

        # for i in range(n_readout_pops):
        #     desired_y = np.arange(neurons_per_readout) + \
        #         int(dist / 2) + dist * i + l5e_min
        #     for j in range(neurons_per_readout):
        #         other_id = y_to_neuron_id[desired_y[j]]
        #         readout_id = network['readout_neuron_ids'][i, j]
        #         old_readout_y = neuron_id_to_y[readout_id]
        #         neuron_id_to_y[readout_id], neuron_id_to_y[other_id] = desired_y[j], neuron_id_to_y[readout_id]
        #         y_to_neuron_id[old_readout_y], y_to_neuron_id[desired_y[j]
        #                                                       ] = other_id, readout_id
        ###########################

        self.neuron_id_to_y = n_neurons - neuron_id_to_y  # plot the L1 top and L6 bottom

    def __call__(self, ax, spikes):
        scale = self.scale
        ms = self.marker_size
        alpha = self.alpha

        layer_label = ['1', '2/3', '4', '5', '6']
        for i, (y, h) in enumerate(zip(self.layer_bounds, np.diff(self.layer_bounds, append=[self.n_neurons]))):
            ax.annotate(
                f'L{layer_label[i]}', (5, (self.n_neurons - y - h / 2)), fontsize=5 * scale, va='center')

            if i % 2 != 0:
                continue
            rect = patches.Rectangle(
                (0, self.n_neurons - y - h), spikes.shape[1], h, color='gray', alpha=.1)
            ax.add_patch(rect)

        # e
        times, ids = np.where(
            spikes[self.batch_ind] * self.e_mask[None, :].astype(np.float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, '.', color='r', ms=ms, alpha=alpha)

        # htr3
        times, ids = np.where(
            spikes[self.batch_ind] * self.htr3_mask[None, :].astype(np.float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, '.', color='darkviolet', ms=ms, alpha=alpha)

        # sst
        times, ids = np.where(
            spikes[self.batch_ind] * self.sst_mask[None, :].astype(np.float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, '.', color='g', ms=ms, alpha=alpha)

        # pvalb
        times, ids = np.where(
            spikes[self.batch_ind] * self.pvalb_mask[None, :].astype(np.float))
        _y = self.neuron_id_to_y[ids]
        ax.plot(times, _y, '.', color='b', ms=ms, alpha=alpha)

        ##### For l5e neurons #####

        # for i, readout_neuron_ids in enumerate(self.network['readout_neuron_ids']):
        #     if len(self.network['readout_neuron_ids']) == 2 and i == 0:
        #         continue
        #     sel = np.zeros(self.n_neurons)
        #     sel[readout_neuron_ids] = 1.
        #     times, ids = np.where(
        #         spikes[self.batch_ind] * sel[None, :].astype(np.float))
        #     _y = self.neuron_id_to_y[ids]
        #     ax.plot(times, _y, '.', color='k', ms=ms, alpha=alpha)

        ###########################

        ax.plot([-1, -1], [-1, -1], '.', color='darkviolet',
                ms=6, alpha=.9, label='Htr3a')
        ax.plot([-1, -1], [-1, -1], '.', color='g',
                ms=6, alpha=.9, label='Sst')
        ax.plot([-1, -1], [-1, -1], '.', color='b',
                ms=6, alpha=.9, label='Pvalb')
        ax.plot([-1, -1], [-1, -1], '.', color='r',
                ms=6, alpha=.9, label='Excitatory')
        # ax.plot([-1, -1], [-1, -1], '.', color='k',
        #         ms=4, alpha=.9, label='Readout (L5e)')
        seq_len = spikes.shape[1]
        # bg = patches.Rectangle((480 / 2050 * seq_len, 0), 300 / 2050 * seq_len,
        #                        220 / 1000 * self.n_neurons, color='white', alpha=.9, zorder=101)
        # ax.add_patch(bg)
        # ax.legend(frameon=True, facecolor='white', framealpha=.9, edgecolor='white',
        #           fontsize=5 * scale, loc='center', bbox_to_anchor=(.3, .12)).set_zorder(102)
        ax.axvline(500, linestyle='dashed', color='k', linewidth=1.5, alpha=1)
        ax.axvline(1500, linestyle='dashed', color='k', linewidth=1.5, alpha=1)
        ax.set_ylim([0, self.n_neurons])
        ax.set_yticks([0, self.n_neurons])
        ax.set_ylabel(f'{self.area_name} Neuron ID', fontsize=20)
        ax.set_xlim([0, seq_len])
        ax.set_xticks([0, seq_len])        
        ax.tick_params(axis='both', which='major', labelsize=18)


class DriftingGrating:
    def __init__(self, scale=2., frequency=2., drifting_init_time=500, drifting_end_time=1500, reverse=False, marker_size=1., alpha=1, color='g'):
        self.marker_size = marker_size
        self.alpha = alpha
        self.color = color
        self.scale = scale
        self.drifting_init_time = drifting_init_time
        self.drifting_end_time = drifting_end_time
        self.reverse = reverse
        self.frequency = frequency

    def __call__(self, ax, spikes):
        times = np.arange(spikes.shape[1])
        stimuli_speed = np.zeros((spikes.shape[1]))
        if self.reverse:
            stimuli_speed[:self.drifting_init_time] = self.frequency
            stimuli_speed[self.drifting_end_time:] = self.frequency
        else:
            stimuli_speed[self.drifting_init_time:self.drifting_end_time] = self.frequency
        
        ax.plot(times, stimuli_speed, color=self.color,
                    ms=self.marker_size, alpha=self.alpha, linewidth=2*self.scale)
        ax.set_ylabel('Visual flow \n [Hz]')
        ax.set_yticks([0, self.frequency], ['0', f'{self.frequency}'])
        ax.set_xlim([0, spikes.shape[1]])
        ax.set_xticks(np.linspace(0, spikes.shape[1], 6))
        ax.set_xlabel('Time [ms]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        
class LGN_sample_plot:
    def __init__(self, firing_rates, spikes, stimuli_init_time=0.5, stimuli_end_time=1.5, images_dir='Images', n_samples=2, directory='LGN units'):
        self.firing_rates = firing_rates[0,:,:]
        self.spikes = spikes
        self.stimuli_init_time = stimuli_init_time
        self.stimuli_end_time = stimuli_end_time
        self.firing_rates_shape = self.firing_rates.shape
        self.n_samples = n_samples
        self.images_dir = images_dir
        self.directory = directory
        
    def __call__(self):
        for neuron_idx in np.random.choice(range(self.firing_rates_shape[1]), size=self.n_samples):
            times = np.linspace(0, (self.firing_rates_shape[0]/1000), self.firing_rates_shape[0])
            
            fig, axs = plt.subplots(2, sharex=True)
            axs[0].plot(times, self.firing_rates[:, neuron_idx], color='r', ms=1, alpha=0.7)
            axs[0].set_ylabel('Firing rate [Hz]')
            axs[1].plot(times, self.spikes[0, :, neuron_idx], color='b', ms=1, alpha=0.7)
            axs[1].set_yticks([0, 1])
            axs[1].set_ylim(0, 1)
            axs[1].set_xlabel('Time [s]')
            axs[1].set_ylabel('Spikes')
            
            for subplot in range(2):
                axs[subplot].axvline(self.stimuli_init_time, linestyle='dashed', color='gray', linewidth=3)
                axs[subplot].axvline(self.stimuli_end_time, linestyle='dashed', color='gray', linewidth=3)
                
            fig.suptitle(f'LGN unit idx:{neuron_idx}')
            path = os.path.join(self.images_dir, self.directory)
            os.makedirs(path, exist_ok=True)
            fig.savefig(os.path.join(path, f'LGN unit idx_{neuron_idx}.png'), dpi=300)
            

class PopulationActivity:
    def __init__(self, orientation, frequency, n_neurons, network, image_path):
        self.data_dir = 'GLIF_network'
        self.orientation = orientation
        self.frequency = frequency
        self.n_neurons = n_neurons
        self.network = network
        self.images_path = image_path
        os.makedirs(self.images_path, exist_ok=True)
        
    def __call__(self, spikes, bin_size=10):
        self.spikes = np.array(spikes)[0]
        self.neurons_ordering()
        self.plot_populations_activity(bin_size)
        self.subplot_populations_activity(bin_size)
        
    def neurons_ordering(self):
        node_types = pd.read_csv(os.path.join(self.data_dir, 'network/v1_node_types.csv'), sep=' ')
        path_to_h5 = os.path.join(self.data_dir, 'network/v1_nodes.h5')
        node_h5 = h5py.File(path_to_h5, mode='r')
        node_type_id_to_pop_name = dict()
        for nid in np.unique(node_h5['nodes']['v1']['node_type_id']): 
            # if not np.unique all of the 230924 model neurons ids are considered, 
            # but nearly all of them are repeated since there are only 111 different indices
            ind_list = np.where(node_types.node_type_id == nid)[0]
            assert len(ind_list) == 1
            node_type_id_to_pop_name[nid] = node_types.pop_name[ind_list[0]]
            
        node_type_ids = np.array(node_h5['nodes']['v1']['node_type_id'])
        # Select population names of neurons in the present network (core)
        true_node_type_ids = node_type_ids[self.network['tf_id_to_bmtk_id']]
        # Now order the pop_names according to their layer and type
        pop_orders = dict(sorted(node_type_id_to_pop_name.items(), key=lambda item: pop_ordering(item[1])))

        # Now we convert the neuroon id (related to its pop_name) to an index related to its position in the y axis
        neuron_id_to_y = np.zeros(self.n_neurons, np.int32) - 1 # rest 1 to check at the end if every neuron has an index
        current_ind = 0
        self.layer_bounds = []
        self.ie_bounds = []
        current_pop_name = 'e0'
        
        for pop_id, pop_name in pop_orders.items():
            # choose all the neurons of the given pop_id
            sel = true_node_type_ids == pop_id
            _n = np.sum(sel)
            # order the neurons by type in the y axis
            neuron_id_to_y[sel] = np.arange(current_ind, current_ind + _n)
            if int(pop_name[1]) > int(current_pop_name[1]):
                # register the change of layer
                self.layer_bounds.append(current_ind)
            if current_pop_name[0] == 'i' and pop_name[0] == 'e':
                # register the change of neuron type: exc -> inh
                self.ie_bounds.append(current_ind)
            current_ind += _n
            current_pop_name = pop_name
            
        # check that an y id has been given to every neuron
        assert np.sum(neuron_id_to_y < 0) == 0
        self.y_to_neuron_id = np.zeros(self.n_neurons, np.int32)
        self.y_to_neuron_id[neuron_id_to_y] = np.arange(self.n_neurons)
        assert np.all(self.y_to_neuron_id[neuron_id_to_y] == np.arange(self.n_neurons))

    def plot_populations_activity(self, bin_size=10):
        layers_label = ['i1', 'i23', 'e23', 'i4', 'e4', 'i5', 'e5', 'i6', 'e6']
        neuron_class_bounds = np.concatenate((self.ie_bounds, self.layer_bounds))
        neuron_class_bounds = np.append(neuron_class_bounds, self.n_neurons)
        neuron_class_bounds.sort()
        
        for idx, label in enumerate(layers_label):
            init_idx = neuron_class_bounds[idx]
            end_idx = neuron_class_bounds[idx+1]
            neuron_ids = self.y_to_neuron_id[init_idx: end_idx]
            n_neurons_class = len(neuron_ids)
            class_spikes = self.spikes[:, neuron_ids]
            m,n = class_spikes.shape
            H,W = int(m/bin_size), 1 # block-size
            n_spikes_bin = class_spikes.reshape(H,m//H,W,n//W).sum(axis=(1,3))
            population_activity = n_spikes_bin/(n_neurons_class*bin_size*0.001)
            
            fig = plt.figure()
            plt.plot(np.arange(0, self.spikes.shape[0], bin_size), population_activity)
            plt.xlabel('Time (ms)')
            plt.ylabel('Population activity (Hz)')
            plt.suptitle(f'Population activity of {label} neurons')
            path = os.path.join(self.images_path, 'Populations activity')
            os.makedirs(path, exist_ok=True)
            fig.tight_layout()
            fig.savefig(os.path.join(path, f'{label}_population_activity.png'), dpi=300)
            
    def subplot_populations_activity(self, bin_size=10):
        layers_label = ['Inhibitory L1 neurons', 'Inhibitory L23 neurons', 'Excitatory L23 neurons', 
                        'Inhibitory L4 neurons', 'Excitatory L4 neurons', 'Inhibitory L5 neurons', 
                        'Excitatory L5 neurons', 'Inhibitory L6 neurons', 'Excitatory L6 neurons']
        neuron_class_bounds = np.concatenate((self.ie_bounds, self.layer_bounds))
        neuron_class_bounds = np.append(neuron_class_bounds, self.n_neurons)
        neuron_class_bounds.sort()
        
        population_activity_dict = {}
        
        for idx, label in enumerate(layers_label):
            init_idx = neuron_class_bounds[idx]
            end_idx = neuron_class_bounds[idx+1]
            neuron_ids = self.y_to_neuron_id[init_idx: end_idx]
            n_neurons_class = len(neuron_ids)
            class_spikes = self.spikes[:, neuron_ids]
            m,n = class_spikes.shape
            H,W = int(m/bin_size), 1 # block-size
            n_spikes_bin = class_spikes.reshape(H,m//H,W,n//W).sum(axis=(1,3))
            population_activity = n_spikes_bin/(n_neurons_class*bin_size*0.001)
            population_activity_dict[label] = population_activity
            
        time = np.arange(0, self.spikes.shape[0], bin_size)
        fig = plt.figure(constrained_layout=False)
        # fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, hspace=0.15, wspace=0.15)
        ax1 = plt.subplot(5, 1, 1)
        plt.plot(time, population_activity_dict['Inhibitory L1 neurons'], label='Inhibitory L1 neurons', color='b')
        plt.legend(fontsize=6)
        plt.tick_params(axis='both', labelsize=7)
        # plt.xlabel('Time (ms)', fontsize=7)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.ylabel('Population \n activity (Hz)', fontsize=7)
        plt.axvline(500, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        plt.axvline(1500, linestyle='dashed', color='gray', linewidth=1, zorder=10) 
        
        ax2=None
        for i in range(3, 9):
            if i%2 == 1:
                ax1 = plt.subplot(5, 2, i, sharex=ax1, sharey=ax1)
                plt.plot(time, population_activity_dict[layers_label[i-2]], label=layers_label[i-2], color='b')
                plt.ylabel('Population \n activity (Hz)', fontsize=7)
                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.legend(fontsize=6, loc='upper right')
                plt.tick_params(axis='both', labelsize=7)
                plt.axvline(500, linestyle='dashed', color='gray', linewidth=1, zorder=10)
                plt.axvline(1500, linestyle='dashed', color='gray', linewidth=1, zorder=10)  
            else:
                if ax2==None:
                    ax2 = plt.subplot(5, 2, i, sharex=ax1)
                else:
                    ax2 = plt.subplot(5, 2, i, sharex=ax2, sharey=ax2)
                plt.plot(time, population_activity_dict[layers_label[i-2]], label=layers_label[i-2], color='r')
                plt.setp(ax2.get_xticklabels(), visible=False)
                plt.legend(fontsize=6, loc='upper right')
                plt.tick_params(axis='both', labelsize=7)
                plt.axvline(500, linestyle='dashed', color='gray', linewidth=1, zorder=10)
                plt.axvline(1500, linestyle='dashed', color='gray', linewidth=1, zorder=10)  
            
        ax1 = plt.subplot(5, 2, 9, sharex=ax1, sharey=ax1)
        plt.plot(time, population_activity_dict[layers_label[7]], label=layers_label[7], color='b')
        plt.ylabel('Population \n activity (Hz)', fontsize=7)
        plt.xlabel('Time [ms]', fontsize=7)
        plt.tick_params(axis='both', labelsize=7)
        plt.legend(fontsize=6, loc='upper right')
        plt.axvline(500, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        plt.axvline(1500, linestyle='dashed', color='gray', linewidth=1, zorder=10) 
        
        ax2 = plt.subplot(5, 2, 10, sharex=ax2, sharey=ax2)
        plt.plot(time, population_activity_dict[layers_label[8]], label=layers_label[8], color='r')
        plt.xlabel('Time [ms]', fontsize=7)
        plt.tick_params(axis='both', labelsize=7)
        plt.legend(fontsize=6, loc='upper right')
        plt.axvline(500, linestyle='dashed', color='gray', linewidth=1, zorder=10)
        plt.axvline(1500, linestyle='dashed', color='gray', linewidth=1, zorder=10)  
            
        plt.subplots_adjust(left=0.1,
                    bottom=0.07, 
                    right=0.99, 
                    top=0.99, 
                    wspace=0.17, 
                    hspace=0.17)
        
        path = os.path.join(self.images_path, 'Populations activity')
        os.makedirs(path, exist_ok=True)
        # fig.tight_layout()
        fig.savefig(os.path.join(path, 'subplot_population_activity.png'), dpi=300)
        
        
class Saleem_Fig1:
    def __init__(self, n_neurons, simulation_length, results_path):
        self.n_neurons = n_neurons
        self.simulation_length = simulation_length
        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)
        
    def plot_firing_rates_w_wo(self, df, neuron_id, directory=''):
        fr_w_mean = df.xs((True, neuron_id), level=('Perturbation', 'Neuron')).mean(axis=0)
        fr_wo_mean = df.xs((False, neuron_id), level=('Perturbation', 'Neuron')).mean(axis=0)
        fr_w_sem = df.xs((True, neuron_id), level=('Perturbation', 'Neuron')).sem(axis=0)
        fr_wo_sem = df.xs((False, neuron_id), level=('Perturbation', 'Neuron')).sem(axis=0)

        times = np.linspace(0, self.simulation_length, len(fr_w_mean))
        fig = plt.figure()
        plt.plot(times, fr_w_mean, color='r', ms=1,
                     alpha=0.7, label='Trials w. pert.')
        plt.fill_between(times, 
                              fr_w_mean + fr_w_sem, 
                              fr_w_mean - fr_w_sem, 
                              alpha=0.3, color='r')
        plt.plot(times, fr_wo_mean, color='gray', ms=1,
                     alpha=0.7, label='Trials w/o. pert.')
        plt.fill_between(times, 
                              fr_wo_mean + fr_wo_sem, 
                              fr_wo_mean - fr_wo_sem, 
                              alpha=0.3, color='gray')
        plt.ylabel(r'Spikes/s')
        plt.xlabel('Time [s]')
        plt.legend()
        
        stimuli_init_time = 4500
        stimuli_end_time = stimuli_init_time + 1000
        plt.axvline(stimuli_init_time, linestyle='dashed', color='gray', linewidth=3)
        plt.axvline(stimuli_end_time, linestyle='dashed', color='gray', linewidth=3)
        fig.suptitle(f'Firing rate neuron id:{neuron_id}')
        path = os.path.join(self.results_path, 'Figure 1', directory)
        os.makedirs(path, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(path, f'fr_neurons_{self.n_neurons}_index_{neuron_id}.png'), dpi=300)
        plt.close(fig)   
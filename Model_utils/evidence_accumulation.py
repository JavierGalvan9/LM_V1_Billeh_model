import os
import absl
import json
import numpy as np
import pickle as pkl
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import load_sparse
import models
import toolkit
import callbacks
import data_sets
import plotting
#import simmanager


class PlotCallback(tf.keras.callbacks.Callback):
    """Periodically plot the activity of the model based on the same example"""

    def __init__(self, test_data_set, extractor_model, network, data_dir, batch_ind=0, scale=2, path=None, prefix=''):
        super().__init__()
        test_iter = iter(test_data_set)
        self._test_example = next(test_iter)
        self._extractor_model = extractor_model

        self.figure = plt.figure(
            figsize=toolkit.cm2inch((15 * scale, 11 * scale)))
        gs = self.figure.add_gridspec(5, 1)
        self.input_ax = self.figure.add_subplot(gs[0])
        self.activity_ax = self.figure.add_subplot(gs[1:-1])
        self.output_ax = self.figure.add_subplot(gs[-1])

        self.inputs_plot = plotting.RasterPlot(
            batch_ind=batch_ind, scale=scale, y_label='Input Neuron ID', alpha=1.)
        self.laminar_plot = plotting.LaminarPlot(
            network, data_dir, batch_ind=batch_ind, scale=scale, alpha=.4)
        self.tightened = False
        self.scale = scale
        self.network = network
        self.batch_ind = batch_ind
        self._counter = 0
        self._path = path
        self._prefix = prefix

    def on_epoch_begin(self, epoch, logs=None):
        inputs = self._test_example[0]
        targets = self._test_example[1]
        (z, v), prediction, all_prediction = self._extractor_model(inputs)

        self.input_ax.clear()
        self.activity_ax.clear()
        self.output_ax.clear()
        self.inputs_plot(self.input_ax, inputs[0].numpy())
        self.input_ax.set_xticklabels([])
        toolkit.apply_style(self.input_ax, scale=self.scale)
        self.laminar_plot(self.activity_ax, z.numpy())
        self.activity_ax.set_xticklabels([])
        toolkit.apply_style(self.activity_ax, scale=self.scale)

        all_pred_np = tf.nn.softmax(models.exp_convolve(
            all_prediction[self.batch_ind], axis=0, decay=.8)).numpy()
        self.output_ax.plot(
            all_pred_np[:, 1], 'r', alpha=.7, lw=self.scale, label='Up')
        self.output_ax.plot(
            all_pred_np[:, 0], 'b', alpha=.7, lw=self.scale, label='Down')
        self.output_ax.set_ylim([0, 1])
        self.output_ax.set_yticks([0, 1])
        self.output_ax.set_xlim([0, all_pred_np.shape[0]])
        self.output_ax.set_xticks([0, all_pred_np.shape[0]])
        self.output_ax.legend(frameon=False, fontsize=5 * self.scale)
        self.output_ax.set_xlabel('Time in ms')
        self.output_ax.set_ylabel('Probability')
        toolkit.apply_style(self.output_ax, scale=self.scale)

        if not self.tightened:
            self.figure.tight_layout()
            self.tightened = True

        self._counter += 1
        if self._path is not None:
            self.figure.savefig(os.path.join(
                self._path, f'{self._prefix}raster_epoch_{self._counter}.png'), dpi=300)


def main(_):
    flags = absl.app.flags.FLAGS
    global_batch_size = flags.batch_size

    # Allow for memory growth (also to observe memory consumption
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for dev in physical_devices:
            tf.config.experimental.set_memory_growth(dev, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # Can be used to try half precision training
    if flags.float16:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        dtype = tf.float16
    else:
        dtype = tf.float32

    # Define 2 outputs that correspond to having more cues top or bottom
    # Note that two different output conventions can be used:
    # 1) Linear readouts from all neurons in the model (softmax)
    # 2) Selecting a population of neurons that report a binary decision
    # with high firing rate (flag --neuron_output)
    n_output = 2

    # Load data of Billeh et al. (2020) and select appropriate number of neurons and inputs
    if flags.caching:
        load_fn = load_sparse.cached_load_billeh
    else:
        load_fn = load_sparse.load_billeh

    n_input = 40
    input_weight_scale = 1.
    input_population, network, bkg_weights = load_fn(
        n_input, flags.neurons, flags.core_only, flags.data_dir, flags.seed, flags.connected_selection, n_output,
        flags.neurons_per_output)
    # ---

    # Load a typical distribution of firing rates to which the model is regularized to
    # during training
    with open(os.path.join(flags.data_dir, 'garrett_firing_rates.pkl'), 'rb') as f:
        firing_rates = pkl.load(f)
    sorted_firing_rates = np.sort(firing_rates)
    percentiles = (np.arange(
        firing_rates.shape[-1]) + 1).astype(np.float32) / firing_rates.shape[-1]
    rate_rd = np.random.RandomState(seed=3000)
    x_rand = rate_rd.uniform(size=flags.neurons)
    target_firing_rates = np.sort(
        np.interp(x_rand, percentiles, sorted_firing_rates))
    # ---

    # tf.random.set_seed(flags.seed)

    model = models.create_model(
        network, input_population, bkg_weights, seq_len=flags.seq_len, n_input=n_input,
        n_output=n_output, cue_duration=flags.recall_duration, dtype=dtype,
        input_weight_scale=input_weight_scale, dampening_factor=flags.dampening_factor,
        gauss_std=flags.gauss_std, lr_scale=flags.lr_scale, train_recurrent=flags.train_recurrent,
        neuron_output=flags.neuron_output, recurrent_dampening_factor=flags.recurrent_dampening_factor,
        batch_size=flags.batch_size, pseudo_gauss=flags.pseudo_gauss)

    model.build((flags.batch_size, flags.seq_len, n_input))

    # Extract outputs of intermediate keras layers to get access to
    # spikes and membrane voltages of the model
    rsnn_layer = model.get_layer('rsnn')
    prediction_layer = model.get_layer('prediction')
    extractor_model = tf.keras.Model(inputs=model.inputs,
                                     outputs=[rsnn_layer.output, model.output[0], prediction_layer.output])

    # Loss used for training (evidence accumulation is a classification task)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                reduction=tf.keras.losses.Reduction.SUM)

    # Training disrupts the firing properties of the model
    # To counteract, two types of regularizations are used
    # 1) Firing rate regularization keeps the distribution of the firing rates close
    # to the previously loaded distribution of firing rates
    rate_distribution_regularizer = models.SpikeRateDistributionRegularization(
        target_firing_rates, flags.rate_cost)
    # 2) Voltage regularization penalizes membrane voltages that are below resting potential or above threshold
    voltage_regularizer = models.VoltageRegularization(
        rsnn_layer.cell, flags.voltage_cost)

    rate_loss = rate_distribution_regularizer(rsnn_layer.output[0])
    voltage_loss = voltage_regularizer(rsnn_layer.output[1])
    model.add_loss(rate_loss)
    model.add_loss(voltage_loss)
    model.add_metric(rate_loss, name='rate_loss')
    model.add_metric(voltage_loss, name='voltage_loss')

    def compute_loss(_target, _pred):
        return loss_object(_target, _pred) / global_batch_size

    # Adaptive learning rates
    optimizer = tf.keras.optimizers.Adam(flags.learning_rate)
    model.compile(optimizer, compute_loss, metrics=['accuracy'])

    # Restore weights from a checkpoint if desired
    if flags.restore_from != '':
        model.load_weights(flags.restore_from)
        print(f'> Model successfully restored from {flags.restore_from}')

    # These "dummy" zeros are injected to the models membrane voltage
    # Provides the opportunity to compute gradients wrt. membrane voltages at all time steps
    # Not important for general use
    def _expand(_x, _y):
        return (_x, tf.zeros((flags.batch_size, flags.seq_len, flags.neurons), dtype)), _y

    def get_dataset_fn(_n, _n_cues):
        def _f(_):
            _data_set = data_sets.create_evidence_accumulation(
                batch_size=global_batch_size, n_input=n_input, seq_len=flags.seq_len,
                recall_duration=flags.recall_duration, examples_in_epoch=_n, n_cues=_n_cues,
                hard_only=flags.hard_only, t_cue=flags.cue_duration, t_interval=flags.interval_duration,
                input_f0=flags.input_f0
            ).repeat().map(_expand)
            return _data_set
        return _f

    test_data_set = get_dataset_fn(
        flags.validation_examples, flags.n_cues)(None)

    # Bookkeeping of simulations
    sim_name = toolkit.get_random_identifier('b_')
    results_dir = os.path.join(flags.results_dir, 'evidence_accumulation')
    print(f'> Results will be stored in {os.path.join(results_dir, sim_name)}')
    os.makedirs(results_dir, exist_ok=True)

    # cm = simmanager.SimManager(sim_name, results_dir, write_protect_dirs=False, tee_stdx_to='output.log')

    # with cm:
    #     # Save the settings with which the script was invoked
    #     with open(os.path.join(cm.paths.data_path, 'flags.json'), 'w') as f:
    #         json.dump(flags.flag_values_dict(), f, indent=4)

    #     # Apply a learning curriculum using iteratively more cues up to the desired number
    #     for n_cues in range(1, flags.n_cues + 1, 2):
    #         train_data_set = get_dataset_fn(flags.examples_in_epoch, n_cues)(None)

    #         vis_data = test_data_set if flags.visualize_test else train_data_set
    #         # Define callbacks that are used for visualizing network activity (see above),
    #         # for stopping the training if the task is solved, and for saving the model
    #         plot_callback = PlotCallback(vis_data, extractor_model, network, flags.data_dir,
    #                                      path=cm.paths.results_path, prefix=f'cue_{n_cues}_')
    #         fit_callbacks = [
    #             plot_callback,
    #             callbacks.StopAt('accuracy', .99),
    #             tf.keras.callbacks.ModelCheckpoint(
    #                 filepath=os.path.join(cm.paths.results_path, 'model'),
    #                 monitor='val_accuracy', save_weights_only=True),
    #             tf.keras.callbacks.TensorBoard(log_dir=cm.paths.results_path)
    #         ]

    #         # Perform training
    #         model.fit(train_data_set, steps_per_epoch=flags.examples_in_epoch, epochs=flags.n_epochs,
    #                   validation_data=test_data_set, validation_steps=flags.validation_examples,
    #                   callbacks=fit_callbacks)


if __name__ == '__main__':
    _data_dir = 'GLIF_network'
    _results_dir = '/tmp/output/billeh'

    # Try and except can be used to prevent DuplicateFlagError (this does not happen
    # when working in the command line and flags allow us to change initial
    # code conditions from command line)

    absl.app.flags.DEFINE_string('data_dir', _data_dir, '')
    absl.app.flags.DEFINE_string('results_dir', _results_dir, '')
    absl.app.flags.DEFINE_string('restore_from', '', '')
    absl.app.flags.DEFINE_string('comment', '', '')

    absl.app.flags.DEFINE_float('learning_rate', .02, '')
    absl.app.flags.DEFINE_float('rate_cost', 0.1, '')
    absl.app.flags.DEFINE_float('voltage_cost', .00001, '')
    absl.app.flags.DEFINE_float('dampening_factor', .3, '')
    absl.app.flags.DEFINE_float('recurrent_dampening_factor', 1., '')
    absl.app.flags.DEFINE_float('gauss_std', .3, '')
    absl.app.flags.DEFINE_float('lr_scale', 1., '')
    absl.app.flags.DEFINE_float('input_f0', 0.2, '')

    absl.app.flags.DEFINE_integer('n_epochs', 20, '')
    absl.app.flags.DEFINE_integer('batch_size', 12, '')
    absl.app.flags.DEFINE_integer(
        'neurons', 5000, '')  # -1 to take all neurons
    absl.app.flags.DEFINE_integer('seq_len', 400, '')
    absl.app.flags.DEFINE_integer('n_cues', 7, '')
    absl.app.flags.DEFINE_integer('recall_duration', 40, '')
    absl.app.flags.DEFINE_integer('cue_duration', 30, '')
    absl.app.flags.DEFINE_integer('interval_duration', 40, '')
    absl.app.flags.DEFINE_integer('examples_in_epoch', 32, '')
    absl.app.flags.DEFINE_integer('validation_examples', 16, '')
    absl.app.flags.DEFINE_integer('seed', 3000, '')
    absl.app.flags.DEFINE_integer('neurons_per_output', 16, '')

    absl.app.flags.DEFINE_boolean('float16', False, '')
    absl.app.flags.DEFINE_boolean('caching', True, '')
    absl.app.flags.DEFINE_boolean('core_only', False, '')
    absl.app.flags.DEFINE_boolean('train_recurrent', True, '')
    absl.app.flags.DEFINE_boolean('connected_selection', True, '')
    absl.app.flags.DEFINE_boolean('neuron_output', True, '')
    absl.app.flags.DEFINE_boolean('hard_only', False, '')
    absl.app.flags.DEFINE_boolean('visualize_test', False, '')
    absl.app.flags.DEFINE_boolean('pseudo_gauss', False, '')

    absl.app.run(main)

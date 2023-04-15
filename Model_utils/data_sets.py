import numpy as np
import numpy.random as rd
import tensorflow as tf


def generate_poisson_noise_np(prob_pattern, freezing_seed=None):
    if isinstance(prob_pattern, list):
        return [generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

    shp = prob_pattern.shape

    if not(freezing_seed is None):
        rng = rd.RandomState(freezing_seed)
    else:
        rng = rd.RandomState()

    spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
    return spikes


def generate_click_task_data(batch_size, seq_len, n_neuron, recall_duration, p_group, f0=0.5,
                             n_cues=7, t_cue=100, t_interval=150,
                             n_input_symbols=4, hard_only=False):
    t_seq = seq_len
    n_channel = n_neuron // n_input_symbols

    # randomly assign group A and B
    prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
    idx = rd.choice([0, 1], batch_size)
    probs = np.zeros((batch_size, 2), dtype=np.float32)
    # assign input spike probabilities
    probs[:, 0] = prob_choices[idx]
    probs[:, 1] = prob_choices[1 - idx]

    cue_assignments = np.zeros((batch_size, n_cues), dtype=np.int32)
    # for each example in batch, draw which cues are going to be active (left or right)
    for b in range(batch_size):
        if hard_only:
            p = np.ones(n_cues + 1)
            p[n_cues // 2] = 10.
            p[n_cues // 2 + 1] = 10.
            p = p / np.sum(p)
            num = rd.choice(np.arange(n_cues + 1), p=p)
            inds = rd.choice(np.arange(n_cues), size=num, replace=False)
            cue_assignments[b, inds] = 1
        else:
            cue_assignments[b, :] = rd.choice([0, 1], n_cues, p=probs[b])

    # generate input nums - 0: left, 1: right, 2:recall, 3:background noise
    input_nums = 3*np.ones((batch_size, seq_len), dtype=np.int32)
    input_nums[:, :n_cues] = cue_assignments
    input_nums[:, -1] = 2

    # generate input spikes
    input_spike_prob = np.zeros((batch_size, t_seq, n_neuron))
    d_silence = t_interval - t_cue
    for b in range(batch_size):
        for k in range(n_cues):
            # input channels only fire when they are selected (left or right)
            c = cue_assignments[b, k]
            # reverse order of cues
            #idx = sequence_length - int(recall_cue) - k - 1
            idx = k
            input_spike_prob[b, d_silence+idx*t_interval:d_silence +
                             idx*t_interval+t_cue, c*n_channel:(c+1)*n_channel] = f0

    # recall cue
    input_spike_prob[:, -recall_duration:, 2*n_channel:3*n_channel] = f0
    # background noise
    input_spike_prob[:, :, 3*n_channel:] = f0/4.
    input_spikes = generate_poisson_noise_np(input_spike_prob)

    # generate targets
    target_mask = np.zeros((batch_size, seq_len), dtype=bool)
    target_mask[:, -1] = True
    target_nums = np.zeros((batch_size, seq_len), dtype=np.int32)
    target_nums[:, :] = np.transpose(
        np.tile(np.sum(cue_assignments, axis=1) > int(n_cues/2), (seq_len, 1)))

    return input_spikes, input_nums, target_nums, target_mask


def create_sequential_mnist(path='mnist.npz', batch_size=20, n_input=100, cue_duration=20, is_test=False,
                            worker_id=0, n_workers=1):
    def map_fn(_x, _y):
        thresholds = tf.cast(tf.linspace(
            0., 254., (n_input - 1) // 2), tf.uint8)
        _x = tf.reshape(_x, (-1,))
        lower = _x[:, None] < thresholds[None, :]
        higher = _x[:, None] >= thresholds[None, :]
        transition_onset = tf.logical_and(lower[:-1], higher[1:])
        transition_offset = tf.logical_and(higher[:-1], lower[1:])
        onset_spikes = tf.cast(transition_onset, tf.float32)
        onset_spikes = tf.concat(
            (onset_spikes, tf.zeros_like(onset_spikes[:1])), 0)
        offset_spikes = tf.cast(transition_offset, tf.float32)
        offset_spikes = tf.concat(
            (offset_spikes, tf.zeros_like(offset_spikes[:1])), 0)

        touch_spikes = tf.cast(tf.equal(_x, 255), tf.float32)[..., None]

        out_spikes = tf.concat((onset_spikes, offset_spikes, touch_spikes), -1)
        out_spikes = tf.tile(out_spikes[:, None], (1, 2, 1))
        out_spikes = tf.reshape(out_spikes, (-1, n_input - 1))
        out_spikes = tf.concat(
            (out_spikes, tf.zeros_like(out_spikes[:cue_duration])), 0)
        signal_spikes = tf.concat(
            (tf.zeros_like(out_spikes[:-cue_duration, :1]), tf.ones_like(touch_spikes[:cue_duration])), 0)
        out_spikes = tf.concat((out_spikes, signal_spikes), -1)

        return out_spikes, _y

    train_data, test_data = tf.keras.datasets.mnist.load_data(path=path)
    data = (train_data, test_data)

    if is_test:
        d = test_data
    else:
        d = train_data
    x, y = d

    samples_per_worker = int(x.shape[0] / n_workers)
    x = x[worker_id * samples_per_worker:(worker_id + 1) * samples_per_worker]
    y = y[worker_id * samples_per_worker:(worker_id + 1) * samples_per_worker]
    y = y.astype(np.int32)
    data_set = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(
        samples_per_worker).map(map_fn).batch(batch_size)
    return data_set


def create_store_recall(batch_size=10, seq_len=400, n_input=200, symbol_duration=40, cue_duration=40,
                        examples_in_epoch=1000, input_rate=.05, noise_rate=.01):
    inputs_per_group = int(n_input / 4)
    assert inputs_per_group * 4 == n_input

    def _f(_):
        _noise = tf.cast(tf.random.uniform(
            shape=(batch_size, seq_len, inputs_per_group)) < noise_rate, tf.float32)
        _bin_label = tf.random.uniform(shape=(batch_size,)) > .5
        _float_label = tf.cast(_bin_label, tf.float32)
        _label = tf.cast(_bin_label, tf.int64)
        _in1 = tf.cast(tf.random.uniform(
            shape=(batch_size, symbol_duration, inputs_per_group)) < input_rate, tf.float32)
        _in1 = _in1 * (1 - _float_label)[:, None, None]
        _in2 = tf.cast(tf.random.uniform(
            shape=(batch_size, symbol_duration, inputs_per_group)) < input_rate, tf.float32)
        _in2 = _in2 * _float_label[:, None, None]
        _cue = tf.cast(tf.random.uniform(
            shape=(batch_size, cue_duration, inputs_per_group)) < input_rate, tf.float32)
        _in_zeros = tf.zeros(
            (batch_size, seq_len - symbol_duration, inputs_per_group))
        _in1 = tf.concat((_in1, _in_zeros), 1)
        _in2 = tf.concat((_in2, _in_zeros), 1)
        _cue = tf.concat(
            (tf.zeros((batch_size, seq_len - cue_duration, inputs_per_group)), _cue), 1)
        _inputs = tf.concat((_in1, _in2, _cue, _noise), -1)
        return _inputs, _label

    data_set = tf.data.Dataset.from_tensors(0).repeat(
        int(examples_in_epoch / batch_size)).map(_f)
    return data_set


def create_evidence_accumulation(batch_size=10, seq_len=2250, n_input=40, recall_duration=150, p_group=.3,
                                 t_cue=100, n_cues=7, t_interval=150, input_f0=.04, examples_in_epoch=1,
                                 hard_only=False):
    def _g():
        while True:
            spk_data, _, target_data, _ = generate_click_task_data(
                batch_size=batch_size, seq_len=seq_len, n_neuron=n_input, recall_duration=recall_duration,
                p_group=p_group, t_cue=t_cue, n_cues=n_cues, t_interval=t_interval, f0=input_f0,
                n_input_symbols=4, hard_only=hard_only)
            yield spk_data, target_data[..., -1]

    output_dtypes = (tf.bool, tf.bool)
    output_shapes = (tf.TensorShape(
        (batch_size, seq_len, n_input)), tf.TensorShape((batch_size,)))
    data_set = tf.data.Dataset.from_generator(_g, output_dtypes, output_shapes=output_shapes).take(
        examples_in_epoch).map(lambda _a, _b: (tf.cast(_a, tf.float32), tf.cast(_b, tf.int32)))
    return data_set


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # evidence accumulation
    d = create_evidence_accumulation()
    for el in d:
        x, y = el[0][0].numpy(), el[1][0].numpy()
        fig, ax = plt.subplots(figsize=(12, 4))
        times, ids = np.where(x > 0.5)
        ax.plot(times, x.shape[-1] - ids, 'k.', ms=1, alpha=.6)
        print(y)
        plt.show()

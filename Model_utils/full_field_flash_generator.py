import numpy as np
import tensorflow as tf
import lgn_model.lgn as lgn_module


@tf.function # Sometimes this function is called with different phase, causing 
def make_drifting_grating_stimulus(row_size=120, col_size=240, moving_flag=True, image_duration=100, cpd=0.05,
                                   temporal_f=2, theta=None, phase=0, contrast=1.0):
    '''
    Create the grating movie with the desired parameters
    :param t_min: start time in seconds
    :param t_max: end time in seconds
    :param cpd: cycles per degree
    :param temporal_f: in Hz
    :param theta: orientation angle
    :return: Movie object of grating with desired parameters
    '''
    #  Franz's code will accept something larger than 101 x 101 because of the
    #  kernel size.
    # row_size = row_size*2 # somehow, Franz's code only accept larger size; thus, i did the mulitplication
    # col_size = col_size*2
    frame_rate = 1000  # Hz
    t_min = 0
    t_max = tf.cast(image_duration, tf.float32) / 1000
    if phase is None:
        phase = tf.random.uniform(shape=(1,), minval=0, maxval=360) 
    if theta is None:
        theta = tf.random.uniform(shape=(1,), minval=0, maxval=360) 

    # assert contrast <= 1, "Contrast must be <= 1"
    # assert contrast > 0, "Contrast must be > 0"
    # tf.debugging.assert_less_equal(contrast, 1.0, message="Contrast must be <= 1")
    # tf.debugging.assert_greater(contrast, 0.0, message="Contrast must be > 0")


    # physical_spacing = 1. / (float(cpd) * 10)    #To make sure no aliasing occurs
    physical_spacing = 1.0  # 1 degree, fixed for now. tf version lgn model need this to keep true cpd;
    row_range = tf.linspace(0.0, row_size, int(row_size / physical_spacing))
    col_range = tf.linspace(0.0, col_size, int(col_size / physical_spacing))
    # number_frames_needed = int(round(frame_rate * t_max))
    number_frames_needed = tf.cast(tf.math.round(frame_rate * t_max), tf.int32)
    time_range = tf.linspace(0.0, t_max, number_frames_needed)

    tt, yy, xx = tf.meshgrid(time_range, row_range, col_range, indexing='ij')

    # theta_rad = tf.constant(np.pi * (180 - theta) / 180.0, dtype=tf.float32) #Add negative here to match brain observatory angles!
    # phase_rad = tf.constant(np.pi * (180 - phase) / 180.0, dtype=tf.float32)

    theta_rad = np.pi * (180 - theta) / 180.0
    phase_rad = np.pi * (180 - phase) / 180.0


    xy = xx * tf.cos(theta_rad) + yy * tf.sin(theta_rad)
    data = contrast * tf.sin(2 * np.pi * (cpd * xy + temporal_f * tt) + phase_rad)

    if moving_flag: # decide whether the gratings drift or they are static 
        return tf.cast(data, tf.float32) 
    else:
        return tf.tile(data[0][tf.newaxis, ...], (image_duration, 1, 1))

@tf.function
def movies_concat(movie, pre_delay, post_delay):
    movie = tf.expand_dims(movie, axis=-1) #movie[...,None]  # add dim
    # add an empty period before a period of gray image
    # z1 = tf.tile(tf.zeros_like(movie[0,...])[None,...], (pre_delay, 1, 1, 1))
    # z2 = tf.tile(tf.zeros_like(movie[0,...])[None,...], (post_delay, 1, 1, 1))
    z1 = tf.zeros((pre_delay, movie.shape[1], movie.shape[2], movie.shape[3]))
    z2 = tf.zeros((post_delay, movie.shape[1], movie.shape[2], movie.shape[3]))
    videos = tf.concat((z1, movie, z2), 0)
    return videos

def generate_drifting_grating_tuning(phase = None, orientation=None, temporal_f=2, cpd=0.04, contrast=0.8, 
                                     row_size=120, col_size=240,
                                     seq_len=600, pre_delay=50, post_delay=50,
                                     current_input=False, regular=False, n_input=17400,
                                     bmtk_compat=True, moving_flag=True):
    """ make a drifting gratings stimulus for FR and OSI tuning."""
    # mimc_lgn_std, mimc_lgn_mean = 0.02855, 0.02146

    lgn = lgn_module.LGN(row_size=row_size, col_size=col_size, n_input=n_input)

    # seq_len = pre_delay + duration + post_delay
    duration =  seq_len - pre_delay - post_delay

    def _g():
        if regular:
            theta = -45  # to make the first one 0
        while True:
            if orientation is None:
                # generate randomly.
                if regular:
                    theta = (theta + 45) % 360
                else:
                    theta = tf.random.uniform([], 0, 360) #np.random.uniform(0, 360)
            else:
                theta = orientation

            theta = tf.cast(theta, tf.float32)
            movie = make_drifting_grating_stimulus(image_duration=duration, cpd=cpd, temporal_f=temporal_f, 
                                           theta=theta, phase=phase, contrast=contrast, moving_flag=moving_flag)
            # movie = tf.expand_dims(movie, axis=-1) #movie[...,None]  # add dim

            # # add an empty period before a period of gray image
            # # z1 = tf.tile(tf.zeros_like(movie[0,...])[None,...], (pre_delay, 1, 1, 1))
            # # z2 = tf.tile(tf.zeros_like(movie[0,...])[None,...], (post_delay, 1, 1, 1))
            # z1 = tf.zeros((pre_delay, movie.shape[1], movie.shape[2], movie.shape[3]))
            # z2 = tf.zeros((post_delay, movie.shape[1], movie.shape[2], movie.shape[3]))
            # videos = tf.concat((z1, movie, z2), 0)
            # del movie

            videos = movies_concat(movie, pre_delay, post_delay)
            del movie

            spatial = lgn.spatial_response(videos, bmtk_compat)
            del videos

            firing_rates = lgn.firing_rates_from_spatial(*spatial)
            del spatial
            # sample rate
            # assuming dt = 1 ms
            _p = 1 - tf.exp(-firing_rates / 1000.) # probability of having a spike before dt = 1 ms
            del firing_rates
            # _z = tf.cast(fixed_noise < _p, dtype)
            if current_input:
                _z = _p * 1.3
            else:
                _z = tf.random.uniform(tf.shape(_p)) < _p
            del _p

            yield _z, tf.constant(theta, dtype=tf.float32, shape=(1,)), tf.constant(contrast, dtype=tf.float32, shape=(1,)), tf.constant(duration, dtype=tf.float32, shape=(1,))
            # yield _z, np.array([theta], dtype=np.float32)

    output_dtypes = (tf.bool, tf.float32, tf.float32, tf.float32)
    
    # output_dtypes = (tf.float32, tf.float32)
    # when using generator for dataset, it should not contain the batch dim
    output_shapes = (tf.TensorShape((seq_len, n_input)), tf.TensorShape((1)), tf.TensorShape((1)), tf.TensorShape((1)))
    # output_shapes = (tf.TensorShape((seq_len, 17400)), tf.TensorShape((1)))
    data_set = tf.data.Dataset.from_generator(_g, output_dtypes, output_shapes=output_shapes).map(lambda _a, _b, _c, _d:
                (tf.cast(_a, tf.bool), tf.cast(_b, tf.float32), tf.cast(_c, tf.float32), tf.cast(_d, tf.float32)))
    # data_set = tf.data.Dataset.from_generator(_g, output_dtypes, output_shapes=output_shapes).map(lambda _a, _b:
                # (tf.cast(_a, tf.float32), tf.cast(_b, tf.float32)))
    return data_set
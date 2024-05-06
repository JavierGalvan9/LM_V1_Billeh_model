import numpy as np
import tensorflow as tf
import lgn_model.lgn as lgn_module

def circle_heatmap(x0, y0, r = 10):
    heatmap = np.zeros((120, 240))

    r = r-0.5 # to make the circle of radius r pixels

    grid_x = np.linspace(50, 190, 13) # 13 points
    grid_y = np.linspace(10, 110, 11) # 11 points

    x0 = grid_x[x0] 
    y0 = grid_y[y0] 

    for x in range(heatmap.shape[1]):
        for y in range(heatmap.shape[0]):
            if (x - x0)**2 + (y - y0)**2 <= r**2:
                heatmap[y, x] = 1

    return heatmap

def create_gabor_mask(x0, y0, seq_len, r = 10):
    """
    Create a 3D mask from a circle heatmap.

    Args:
        x0 (int): The index of the x-coordinate of the center of the circle
        y0 (int): The index of y-coordinate of the center of the circle.
        seq_len (int): The length of the sequence.
        r (int): The radius of the circle.

    Returns:
        tf.Tensor: The 3D mask.
    """
    mask = circle_heatmap(x0, y0, r)
    mask_3d = tf.tile(mask[None, :, :], [seq_len, 1, 1])
    mask_3d = tf.cast(mask_3d, tf.float32)
    
    return mask_3d

@tf.function # Sometimes this function is called with different phase, causing 
def make_moving_gabors_stimulus(row_size=120, col_size=240, moving_flag=True, image_duration=100, cpd=0.05,
                                   temporal_f=2, theta=None, phase=0, contrast=1.0, x0 = 0, y0 = 0, r=10):
    """
    Generates a drifting grating stimulus.

    Args:
        row_size (int): Number of rows in the stimulus.
        col_size (int): Number of columns in the stimulus.
        moving_flag (bool): Flag indicating whether the gratings drift or are static.
        image_duration (int): Duration of the stimulus in milliseconds.
        cpd (float): Cycles per degree of the grating.
        temporal_f (float): Temporal frequency of the grating.
        theta (float): Orientation angle of the grating in degrees. If None, a random angle is chosen.
        phase (float): Phase of the grating in degrees.
        contrast (float): Contrast of the grating.

    Returns:
        tf.Tensor: The generated drifting grating stimulus.
    """
    frame_rate = 1000  # Hz
    t_max = tf.cast(image_duration, tf.float32) / 1000
    if phase is None:
        phase = tf.random.uniform(shape=(1,), minval=0, maxval=360) 
    if theta is None:
        theta = tf.random.uniform(shape=(1,), minval=0, maxval=360) 

    physical_spacing = 1.0  # 1 degree, fixed for now. tf version lgn model need this to keep true cpd;
    row_range = tf.linspace(0.0, row_size, int(row_size / physical_spacing))
    col_range = tf.linspace(0.0, col_size, int(col_size / physical_spacing))
    # number_frames_needed = int(round(frame_rate * t_max))
    number_frames_needed = tf.cast(tf.math.round(frame_rate * t_max), tf.int32)
    time_range = tf.linspace(0.0, t_max, number_frames_needed)

    tt, yy, xx = tf.meshgrid(time_range, row_range, col_range, indexing='ij')

    theta_rad = np.pi * (180 - theta) / 180.0
    phase_rad = np.pi * (180 - phase) / 180.0

    xy = xx * tf.cos(theta_rad) + yy * tf.sin(theta_rad)
    data = contrast * tf.sin(2 * np.pi * (cpd * xy + temporal_f * tt) + phase_rad)

    # create a gabor filter at grey background -----------------------------------------------
    mask_3d = create_gabor_mask(x0, y0, image_duration, r = r) # returns the mask already in tf format

    # Apply the mask to the data
    masked_data = data * mask_3d  # 0 is the grey background
    # ---------------------------------------------------------------------------------------

    if moving_flag: # decide whether the gratings drift or they are static 
        return tf.cast(masked_data, tf.float32) 
    else:
        return tf.tile(masked_data[0][tf.newaxis, ...], (image_duration, 1, 1))

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
                                     bmtk_compat=True, moving_flag=True, x0 = 0, y0 = 0, r = 10):
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
                    theta = (theta + 45) % 180
                else:
                    theta = tf.random.uniform([], 0, 180) #np.random.uniform(0, 360)
            else:
                theta = orientation

            theta = tf.cast(theta, tf.float32)
            movie = make_moving_gabors_stimulus(image_duration=duration, cpd=cpd, temporal_f=temporal_f, 
                                           theta=theta, phase=phase, contrast=contrast, moving_flag=moving_flag, 
                                           x0 = x0, y0 = y0, r = r)
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
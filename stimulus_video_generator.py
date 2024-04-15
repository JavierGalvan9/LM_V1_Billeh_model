'''
19/03/2024
Generating a video of the stimulus 
'''
#%% Importing libraries
import numpy as np
import numpy as np
import tensorflow as tf
import imageio

# Functions
@tf.function # Sometimes this function is called with different phase, causing 
def make_drifting_grating_stimulus(row_size=120, col_size=240, moving_flag=True, image_duration=100, cpd=0.05,
                                   temporal_f=2, theta=None, phase=0, contrast=1.0):
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

    if moving_flag: # decide whether the gratings drift or they are static 
        return tf.cast(data, tf.float32) 
    else:
        return tf.tile(data[0][tf.newaxis, ...], (image_duration, 1, 1))

@tf.function
def movies_concat(movie, pre_delay, post_delay):
    """
    Concatenates a movie with pre and post delay periods of gray images.

    Args:
        movie (tf.Tensor): The movie to be concatenated. It should have shape (frames, height, width, channels).
        pre_delay (int): The number of frames to add before the movie.
        post_delay (int): The number of frames to add after the movie.

    Returns:
        tf.Tensor: The concatenated movie with pre and post delay periods.

    """
    movie = tf.expand_dims(movie, axis=-1) #movie[...,None]  # add dim
    # add an empty period before a period of gray image
    z1 = tf.zeros((pre_delay, movie.shape[1], movie.shape[2], movie.shape[3]))
    z2 = tf.zeros((post_delay, movie.shape[1], movie.shape[2], movie.shape[3]))

    z1 = tf.cast(z1, tf.float32)
    z2 = tf.cast(z2, tf.float32)
    movie = tf.cast(movie, tf.float32)

    videos = tf.concat((z1, movie, z2), 0)
    return videos

def generate_drifting_grating_video(phase=None, orientation=None, temporal_f=2, cpd=0.04, contrast=0.8, 
                                     seq_len=600, pre_delay=50, post_delay=50, moving_flag=True):
    """
    Generates a drifting grating tuning video.

    Parameters:
    - phase (float): The phase of the grating in degrees. If None, a random phase is used.
    - orientation (float): The orientation of the grating in degrees. If None, a random orientation is used.
    - temporal_f (float): The temporal frequency of the grating in Hz.
    - cpd (float): The spatial frequency of the grating in cycles per degree.
    - contrast (float): The contrast of the grating, ranging from -1 to 1.
    - seq_len (int): The total length of the video sequence in frames.
    - pre_delay (int): The number of frames to show before the grating starts drifting.
    - post_delay (int): The number of frames to show after the grating stops drifting.
    - moving_flag (bool): If True, the grating will be moving. If False, the grating will be static.

    Returns:
    - video (ndarray): The generated drifting grating tuning video as a numpy array.
    """
    # Set the orientation angle to the given orientation
    theta = orientation
    # Calculate the duration of the stimulus video
    duration =  seq_len - pre_delay - post_delay

    # Generate the drifting grating stimulus for the given duration and parameters
    movie = make_drifting_grating_stimulus(image_duration=duration, cpd=cpd, temporal_f=temporal_f, 
                                           theta=theta, phase=phase, contrast=contrast, moving_flag=moving_flag)
    
    # Concatenate the movie with pre and post delay periods
    video = movies_concat(movie, pre_delay, post_delay)

    # Run the session to obtain the video as a numpy array
    #with tf.Session() as sess:
    #    video = sess.run(video)
    video = video.numpy()

    # Remove the extra dimension from the video
    video = video.squeeze(axis=-1)

    return video

# Generate GIF
#-- STIMULUS TYPES AND PARAMETERS ---------------------------------------------
# For full-field flash:
# temporal_f = 0
# cpd = 0
# contrast = 1
# seq_len = 350 (with default pre_delay and post_delay to 50)
# phase = 90
#--------------------------------------------------------------------------
video = generate_drifting_grating_video(temporal_f=0, cpd=0, contrast=1, seq_len=350, moving_flag=False, phase = 90) # full-field flash
#video = generate_drifting_grating_video(temporal_f=40, cpd=0.04, contrast=0.8, seq_len=350, moving_flag=True) # moving gratings
video = ((video + 1) * 127.5).astype(np.uint8) # normalize to 0-255
imageio.mimsave('animations/full_field_flash.gif', video, fps=30)  # Adjust fps as needed

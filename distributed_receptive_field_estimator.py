import subprocess
import json
import os
import re
import argparse
import numpy as np
import pandas as pd
# import tensorflow as tf
# import tensorflow as tf
from Model_utils import toolkit, load_sparse
# # script_path = "bash d

# Create argument parser
parser = argparse.ArgumentParser()

# Define arguments
parser.add_argument('--task_name', default='receptive_fields_estimation', type=str)
parser.add_argument('--results_dir', default='Receptive_field', type=str)
parser.add_argument('--restore_from', default='', type=str)
parser.add_argument('--n_trials', default=50, type=int)
parser.add_argument('--seq_len', default=350, type=int)

parser.add_argument('--delays', default='100,0', type=str)

# flags for gabor patches for orientation, circle_row, circle_column, radius_circle, batch_size, n_angles
parser.add_argument('--row_size', default=120, type=int)
parser.add_argument('--col_size', default=240, type=int)
parser.add_argument('--radius_circle', default=10, type=int)
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--n_angles', default=3, type=int)

parser.add_argument('--data_dir', default='GLIF_network', type=str)
parser.add_argument('--comment', default='', type=str)
parser.add_argument('--interarea_weight_distribution', default='billeh_weights', type=str)
parser.add_argument('--dtype', default='float16', type=str)
parser.add_argument('--rotation', default='ccw', type=str)

parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--dampening_factor', default=0.5, type=float)
parser.add_argument('--recurrent_dampening_factor', default=0.5, type=float)
parser.add_argument('--input_weight_scale', default=1.0, type=float)
parser.add_argument('--gauss_std', default=0.28, type=float)
parser.add_argument('--lr_scale', default=1.0, type=float)
parser.add_argument('--input_f0', default=0.2, type=float)
parser.add_argument('--E4_weight_factor', default=1.0, type=float)
parser.add_argument('--temporal_f', default=2.0, type=float)

parser.add_argument('--n_epochs', default=50, type=int)
parser.add_argument('--v1_neurons', default=10, type=int)
parser.add_argument('--lm_neurons', default=10, type=int)
parser.add_argument('--steps_per_epoch', default=20, type=int)
parser.add_argument('--n_input', default=17400, type=int)
parser.add_argument('--n_cues', default=3, type=int)
parser.add_argument('--recall_duration', default=40, type=int)
parser.add_argument('--cue_duration', default=40, type=int)
parser.add_argument('--interval_duration', default=40, type=int)
parser.add_argument('--examples_in_epoch', default=32, type=int)
parser.add_argument('--validation_examples', default=16, type=int)
parser.add_argument('--seed', default=3000, type=int)
parser.add_argument('--neurons_per_output', default=30, type=int)
parser.add_argument('--n_output', default=10, type=int)

parser.add_argument('--caching', default=True, action='store_true')
parser.add_argument('--core_only', default=False, action='store_true')
parser.add_argument('--hard_reset', default=False, action='store_true')
parser.add_argument('--disconnect_lm_L6_inhibition', default=False, action='store_true')
parser.add_argument('--disconnect_v1_lm_L6_excitatory_projections', default=False, action='store_true')
parser.add_argument('--realistic_neurons_ratio', default=False, action='store_true')
parser.add_argument('--train_recurrent_v1', default=False, action='store_true')
parser.add_argument('--train_recurrent_lm', default=False, action='store_true')
parser.add_argument('--train_input', default=False, action='store_true')
parser.add_argument('--train_interarea_lm_v1', default=False, action='store_true')
parser.add_argument('--train_interarea_v1_lm', default=False, action='store_true')
parser.add_argument('--train_noise', default=False, action='store_true')
parser.add_argument('--connected_selection', default=True, action='store_true')
parser.add_argument('--neuron_output', default=True, action='store_true')
parser.add_argument('--visualize_test', default=False, action='store_true')
parser.add_argument('--pseudo_gauss', default=False, action='store_true')
parser.add_argument('--bmtk_compat_lgn', default=True, action='store_true')
parser.add_argument('--connected_areas', dest='connected_areas', action='store_true', default=True)
parser.add_argument('--no-connected_areas', dest='connected_areas', action='store_false')
parser.add_argument('--connected_recurrent_connections', dest='connected_recurrent_connections', action='store_true', default=True)
parser.add_argument('--no-connected_recurrent_connections', dest='connected_recurrent_connections', action='store_false')
parser.add_argument('--connected_noise', dest='connected_noise', action='store_true', default=True)
parser.add_argument('--no-connected_noise', dest='connected_noise', action='store_false')
parser.add_argument('--current_input', action='store_true', default=False)
parser.add_argument('--random_weights', default=False, action='store_true')


def submit_job(command):
    """ 
    Submit a job to the cluster using the command provided.
    """
    result = subprocess.run(command, capture_output=True, text=True)
    job_id = re.search(r'\d+', result.stdout.strip())
    job_id = job_id.group()

    return job_id

def main():                
    # Initialize the flags and customize the simulation main characteristics
    flags = parser.parse_args()

    # Save the configuration of the model based on the main features
    flag_str = f'v1_{flags.v1_neurons}_lm_{flags.lm_neurons}'
    for name, value in vars(flags).items():
        if value != parser.get_default(name) and name in ['n_input', 'core_only', 'connected_selection', 'interarea_weight_distribution', 'E4_weight_factor', 'random_weights']:
            flag_str += f'_{name}_{value}'

    # Define flag string as the second part of results_path
    results_dir = f'{flags.results_dir}/{flag_str}'
    os.makedirs(results_dir, exist_ok=True)
    print('Simulation results path: ', results_dir)
    # Save the flags configuration as a dictionary in a JSON file
    with open(os.path.join(results_dir, 'flags_config.json'), 'w') as fp:
        json.dump(vars(flags), fp)

    # Generate a ticker for the current simulation
    if flags.restore_from == '':
        sim_name = toolkit.get_random_identifier('b_')
        logdir = os.path.join(results_dir, sim_name)
        initial_benchmark_model = ''
    else:
        sim_name = os.path.basename(os.path.dirname(flags.restore_from))
        logdir = os.path.dirname(flags.restore_from) 
        initial_benchmark_model = flags.restore_from

    flag_str = logdir.split(os.path.sep)[-2]
    print(f'> Results for {flags.task_name} will be stored in:\n {logdir} \n')

    # Characterize the simulations
    n_cols =  int(np.ceil((flags.col_size - flags.radius_circle) / flags.radius_circle))
    n_rows = int(np.ceil((flags.row_size - flags.radius_circle) / flags.radius_circle))

    # Load LM-V1 column data
    if flags.caching:
        load_fn = load_sparse.cached_load_billeh
    else:
        load_fn = load_sparse.load_billeh

    n_neurons = {'v1': flags.v1_neurons, 'lm': flags.lm_neurons}
    _, lgn_inputs, _ = load_fn(flags, n_neurons, flag_str=flag_str)
    # Load the LGN units data for find the extreme possible values for the RF locations
    path = '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/lgn_model/data/lgn_full_col_cells_240x120.csv'
    lgn_data = pd.read_csv(path, sep=' ')

    # Get the extreme locations of the lgn connected neurons
    lgn_ids = np.array(list(set(lgn_inputs['v1']['indices'][:, 1])))
    x, y = lgn_data['x'][lgn_ids].values, lgn_data['y'][lgn_ids].values
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # using the min_x and max_x, calculate the real initial and final row for iteration
    initial_row = int((min_y - flags.radius_circle) / flags.radius_circle)
    final_row = int(max_y / flags.radius_circle) + 1
    # same for the columns
    initial_col = int((min_x - flags.radius_circle) / flags.radius_circle)
    final_col = int(max_x / flags.radius_circle) + 1

    # # Define the job submission commands for the training and evaluation scripts
    evaluation_commands = ["run", "-g", "1", "-m", "48", "-c", "4", "-t", "1:00"]

    # Define the evaluation script calls
    evaluation_script = "python receptive_field_estimator.py " 

    # Append each flag to the string
    for name, value in vars(flags).items():
        if name not in ['seed']: 
            if type(value) == bool and value == False:
                evaluation_script += f"--no{name} "
            elif type(value) == bool and value == True:
                evaluation_script += f"--{name} "
            else:
                evaluation_script += f"--{name} {value} "

    job_ids = []

    # circle_rows = np.arange(0, n_rows)
    # circle_columns = np.arange(0, n_cols)
    # # # get all the possible combinations using itertools
    # import itertools
    # all_combinations = list(itertools.product(circle_rows, circle_columns))

    # for row, col in all_combinations:
    for col in range(initial_col, final_col):
        # Submit all the receptive field evaluation jobs
        new_evaluation_command = evaluation_commands + ["-o", f"Out/{sim_name}_receptive_field_col_{col}.out", "-e", f"Error/{sim_name}_receptive_field_col_{col}.err", "-j", f"{sim_name}_col_{col}"]
        new_first_evaluation_script = evaluation_script + f"--circle_column {col} --seed {flags.seed + col} --ckpt_dir {logdir} --restore_from {initial_benchmark_model} "
        new_first_evaluation_command = new_evaluation_command + [new_first_evaluation_script]
        job_id = submit_job(new_first_evaluation_command)
        job_ids.append(job_id)
    
    print("Submitted receptive field evaluation jobs with the following JOBIDs:", job_ids)

if __name__ == '__main__':
    main()

#! /bin/bash

# run -g 1 -m 24 -t 0:30 -o Out/drifting_gratings2.out -e Error/drifting_gratings2.err -j small "python multi_training.py --batch_size 1 --v1_neurons 10000 --seq_len 600 --n_runs 1 --n_epochs 1 --steps_per_epoch 1"

# python parallel_training_testing.py --spontaneous_training --v1_neurons 100000 --lm_neurons 30000 --realistic_neurons_ratio --osi_cost 0.1 --train_interarea_lm_v1 --train_noise --learning_rate 0.001 --n_runs 2 --n_epochs 100 --steps_per_epoch 20
# python parallel_training_testing.py --delays '500,0' --seq_len 1000 --v1_neurons 100000 --lm_neurons 30000 --realistic_neurons_ratio --osi_cost 1 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --learning_rate 0.001 --n_runs 1 --n_epochs 10 --steps_per_epoch 50

# python parallel_training_testing.py --interarea_weight_distribution 'billeh_weights' --delays '0,0' --seq_len 500 --v1_neurons 10000 --lm_neurons 3000 --realistic_neurons_ratio --osi_cost 1 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --learning_rate 0.001 --n_runs 2 --n_epochs 20 --steps_per_epoch 20

# run -g 1 -m 65 -t 2:00 -o Out/gordo.out -e Error/gordo.err -j gordo "python osi_dsi_estimator.py --core_loss --interarea_weight_distribution 'billeh_weights' --E4_weight_factor 4.0 --v1_neurons 90000 --lm_neurons 30000 --ckpt_dir '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_90000_lm_30000_E4_weight_factor_4.0/b_m4bo' --restore_from 'Best_model' --run_session 1000 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm"

python parallel_training_testing.py --core_loss --interarea_weight_distribution 'billeh_weights' --E4_weight_factor 4.0 --v1_neurons 10000 --lm_neurons 3000 --rate_cost 10000 --osi_cost 20 --sync_cost 0.1 --recurrent_weight_regularization 1 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --n_runs 1 --n_epochs 50 --steps_per_epoch 25
# python parallel_training_testing.py --core_loss --interarea_weight_distribution 'random_weights' --random_weights --E4_weight_factor 4.0 --v1_neurons 100000 --lm_neurons 30000 --rate_cost 10000 --osi_cost 20 --sync_cost 0.1 --recurrent_weight_regularization 1 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --n_runs 10 --n_epochs 50 --steps_per_epoch 25
# python parallel_training_testing.py --core_loss --interarea_weight_distribution 'billeh_weights' --E4_weight_factor 4.0 --v1_neurons 90000 --lm_neurons 30000 --rate_cost 10000 --osi_cost 10 --sync_cost 0.25 --recurrent_weight_regularization 0.05 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --n_runs 20 --n_epochs 1 --steps_per_epoch 256

# scp -r Benchmark_models sofiagil@nuredduna2020:/home/sofiagil/tfm/LM_V1_Billeh_model_sofia
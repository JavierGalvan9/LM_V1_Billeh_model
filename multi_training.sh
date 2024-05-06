#! /bin/bash

# run -g 1 -m 24 -t 0:30 -o Out/drifting_gratings2.out -e Error/drifting_gratings2.err -j small "python multi_training.py --batch_size 1 --v1_neurons 10000 --seq_len 600 --n_runs 1 --n_epochs 1 --steps_per_epoch 1"
# run -g 1 -m 24 -t 2:30 -o Out/gordo.out -e Error/gordo.err -j gordo "python osi_dsi_estimator.py --batch_size 1 --v1_neurons 10000 --seq_len 600 --n_epochs 1 --steps_per_epoch 1"

# python parallel_training_testing.py --spontaneous_training --v1_neurons 100000 --lm_neurons 30000 --realistic_neurons_ratio --osi_cost 0.1 --train_interarea_lm_v1 --train_noise --learning_rate 0.001 --n_runs 2 --n_epochs 100 --steps_per_epoch 20
# python parallel_training_testing.py --delays '500,0' --seq_len 1000 --v1_neurons 100000 --lm_neurons 30000 --realistic_neurons_ratio --osi_cost 1 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --learning_rate 0.001 --n_runs 1 --n_epochs 10 --steps_per_epoch 50
python parallel_training_testing.py --delays '0,0' --seq_len 500 --v1_neurons 100000 --lm_neurons 30000 --realistic_neurons_ratio --osi_cost 1 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --learning_rate 0.001 --n_runs 20 --n_epochs 1 --steps_per_epoch 500

# scp -r Benchmark_models sofiagil@nuredduna2020:/home/sofiagil/tfm/LM_V1_Billeh_model_sofia
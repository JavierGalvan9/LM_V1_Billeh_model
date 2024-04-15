#! /bin/bash

# run -g 1 -m 24 -t 0:30 -o Out/drifting_gratings2.out -e Error/drifting_gratings2.err -j small "python multi_training.py --batch_size 1 --v1_neurons 10000 --seq_len 600 --n_runs 1 --n_epochs 1 --steps_per_epoch 1"
# run -g 1 -m 24 -t 2:30 -o Out/gordo.out -e Error/gordo.err -j gordo "python multi_training.py --batch_size 1 --v1_neurons 51978 --seq_len 600 --n_epochs 110 --steps_per_epoch 20"

python parallel_training_testing.py --v1_neurons 100000 --interarea_weight_distribution 'billeh_weights' --osi_loss_method 'crowd_osi' --osi_cost 1 --rate_cost 100 --voltage_cost 1 --recurrent_weight_regularization 0 --train_interarea --train_recurrent_v1 --train_recurrent_lm --learning_rate 0.001 --n_runs 10 --n_epochs 20 --steps_per_epoch 20

# run -g 1 -m 100 -t 2:30 -o Out/gordo.out -e Error/gordo.err -j gordo "python generate_dg_lgn_fr.py" 

# scp -r Benchmark_models sofiagil@nuredduna2020:/home/sofiagil/tfm/LM_V1_Billeh_model_sofia
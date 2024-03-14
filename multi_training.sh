#! /bin/bash

# run -g 1 -m 24 -t 0:30 -o Out/drifting_gratings2.out -e Error/drifting_gratings2.err -j small "python multi_training.py --batch_size 1 --v1_neurons 10000 --seq_len 600 --n_runs 1 --n_epochs 1 --steps_per_epoch 1"
# run -g 1 -m 24 -t 2:30 -o Out/gordo.out -e Error/gordo.err -j gordo "python multi_training.py --batch_size 1 --v1_neurons 51978 --seq_len 600 --n_epochs 110 --steps_per_epoch 20"
python parallel_training_testing.py --v1_neurons 80000 --core_loss --learning_rate 0.01 --n_runs 1 --n_epochs 1 --steps_per_epoch 2




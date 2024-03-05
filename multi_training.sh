#! /bin/bash

run -g 1 -m 24 -t 0:30 -o Out/small.out -e Error/small.err -j small "python multi_training.py --batch_size 1 --v1_neurons 10000 --seq_len 600 --n_epochs 1 --steps_per_epoch 1"
# run -g 1 -m 24 -t 2:30 -o Out/gordo.out -e Error/gordo.err -j gordo "python multi_training.py --batch_size 1 --v1_neurons 51978 --seq_len 600 --n_epochs 110 --steps_per_epoch 20"
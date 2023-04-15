#! /bin/bash

run -c 1 -m 100 -t 0:30 -o Out/metrics_analysis.out -e Error/metrics_analysis.err "python model_metrics_analysis.py --V1_neurons 230924 --LM_neurons 32940 --n_simulations 10"

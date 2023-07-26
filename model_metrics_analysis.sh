#! /bin/bash

# ### NORMAL MODELS ###
# run -c 1 -m 10 -t 0:30 -o Out/metrics_analysis_normal.out -e Error/metrics_analysis_normal.err "python model_metrics_analysis.py --V1_neurons 230924 --LM_neurons 32940 --n_simulations 2 --seq_len 3000"

### E4 MODELS ###
# run -c 1 -m 100 -t 0:30 -o Out/metrics_analysis_normal.out -e Error/metrics_analysis_normal.err "python model_metrics_analysis.py --V1_neurons 230924 --LM_neurons 32940 --n_simulations 10 --seq_len 3000 --E4_weight_factor 1"

# ### DISCONNECTED MODELS ###
# for orientation in 0 45 90 135 180 225 270 315
# do for frequency in 2 #1 3 5 7 9 10 2 0 4 6 8
run -c 1 -m 100 -t 10:30 -o Out/metrics_analysis_disconnected.out -e Error/metrics_analysis_disconnected.err "python model_metrics_analysis.py --V1_neurons 230924 --LM_neurons 32940 --seq_len 3000 --n_simulations 10 --interarea_weight_distribution disconnected"

# ### NORMAL MODELS ###
# for orientation in 0 #45 90 135 180 225 270 315
# do for frequency in 2 #1 3 5 7 9 10 2 0 4 6 8
# # do run -c 1 -m 150 -t 40:30 -o Out/or-"$orientation"_freq-"$frequency".out -e Error/or-"$orientation"_freq-"$frequency".err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 10 --seq_len 3000"
# do run -c 1 -m 10 -t 20:30 -o Out/or-"$orientation"_freq-"$frequency"_small.out -e Error/or-"$orientation"_freq-"$frequency"_small.err "python drifting_gratings.py --V1_neurons 1000 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# done
# done

# ### DISCONNECTED MODELS ###
# for orientation in 0 45 90 135 180 225 270 315
# do for frequency in 2 #1 3 5 7 9 10 2 0 4 6 8
# do run -c 1 -m 10 -t 0:30 -o Out/or-"$orientation"_freq-"$frequency"_connected.out -e Error/or-"$orientation"_freq-"$frequency"_connected.err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 2 --interarea_weight_distribution disconnected"
# # do run -c 1 -m 10 -t 20:30 -o Out/or-"$orientation"_freq-"$frequency".out -e Error/or-"$orientation"_freq-"$frequency".err "python drifting_gratings.py --V1_neurons 2309 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 2"
# done
# done
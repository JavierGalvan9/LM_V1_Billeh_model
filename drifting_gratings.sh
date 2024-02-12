#! /bin/bash

### NORMAL MODELS ###
# for orientation in 0 #45 90 135 180 225 270 315
# do for frequency in 2 #1 3 5 7 9 10 2 0 4 6 8
# # do run -c 1 -m 150 -t 40:30 -o Out/or-"$orientation"_freq-"$frequency".out -e Error/or-"$orientation"_freq-"$frequency".err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 10 --seq_len 3000"
# do run -c 1 -m 150 -t 20:30 -o Out/or-"$orientation"_freq-"$frequency".out -e Error/or-"$orientation"_freq-"$frequency".err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# done
# done

# ### DISCONNECTED MODELS ###
# for orientation in 0 45 90 135 180 225 270 315
# do for frequency in 2 #1 3 5 7 9 10 2 0 4 6 8
# do run -c 1 -m 150 -t 40:30 -o Out/or-"$orientation"_freq-"$frequency"_connected.out -e Error/or-"$orientation"_freq-"$frequency"_connected.err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 10 --seq_len 3000 --interarea_weight_distribution disconnected"
# # do run -c 1 -m 10 -t 20:30 -o Out/or-"$orientation"_freq-"$frequency".out -e Error/or-"$orientation"_freq-"$frequency".err "python drifting_gratings.py --V1_neurons 2309 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 2"
# done
# done

# E4 weight factor ###
# for E4_weight_factor in 1. 1.25 1.5 1.75 2
# do run -c 1 -m 150 -t 40:30 -o Out/or-0_freq-2-L4weight-$E4_weight_factor.out -e Error/or-0_freq-2-L4weight-$E4_weight_factor.err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --disconnect_V1_LM_L6_excitatory_projections --E4_weight_factor $E4_weight_factor --n_simulations 3"
# # do run -c 1 -m 10 -t 20:30 -o Out/or-"$orientation"_freq-"$frequency".out -e Error/or-"$orientation"_freq-"$frequency".err "python drifting_gratings.py --V1_neurons 2309 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 2"
# done


# for orientation in 0 45 90 135 180 225 270 315
# do run -c 1 -m 150 -t 40:30 -o Out/or_freq-2-L4weight-$E4_weight_factor.out -e Error/or_freq-2-L4weight-$E4_weight_factor.err "python drifting_gratings.py --V1_neurons 230924 --gratings_orientation $orientation --batch_size 1 --seq_len 3000 --E4_weight_factor 2 --n_simulations 2"
# # do run -c 1 -m 10 -t 20:30 -o Out/or-"$orientation"_freq-"$frequency".out -e Error/or-"$orientation"_freq-"$frequency".err "python drifting_gratings.py --V1_neurons 2309 --batch_size 1 --seq_len 3000 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 2"
# done


## DISCONNECT LM L6 INHIBITORY NEURONS ###
# run -c 1 -m 150 -t 40:30 -o Out/or-0_freq-2-L4weight-$E4_weight_factor.out -e Error/or-0_freq-2-L4weight-$E4_weight_factor.err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --disconnect_LM_L6_inhibition --n_simulations 3"

# run -c 1 -m 150 -t 40:30 -o Out/or-0_freq-2-L4weight-$E4_weight_factor.out -e Error/or-0_freq-2-L4weight-$E4_weight_factor.err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --disconnect_LM_L6_inhibition --E4_weight_factor 1.5 --n_simulations 3"

# run -c 1 -m 150 -t 40:30 -o Out/or-0_freq-2-L4weight-$E4_weight_factor.out -e Error/or-0_freq-2-L4weight-$E4_weight_factor.err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --disconnect_LM_L6_inhibition --E4_weight_factor 2 --n_simulations 3"

run -c 1 -m 15 -t 0:30 -o Out/new_or-0_freq-2-L4weight-$E4_weight_factor.out -e Error/new_or-0_freq-2-L4weight-$E4_weight_factor.err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 3000 --n_simulations 1 --gratings_orientation 45"

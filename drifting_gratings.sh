#! /bin/bash

for orientation in 0 #45 90 135 180 225 270 315
do for frequency in 2 #1 3 5 7 9 10 2 0 4 6 8
do run -c 1 -m 100 -t 12:30 -o Out/or-"$orientation"_freq-"$frequency".out -e Error/or-"$orientation"_freq-"$frequency".err "python drifting_gratings.py --V1_neurons 230924 --batch_size 1 --seq_len 2500 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 1"
# do run -g 1 -t 2:30 -o Out/or-"$orientation"_freq-"$frequency".out -e Error/or-"$orientation"_freq-"$frequency".err "python drifting_gratings.py --V1_neurons 100 --LM_neurons 80 --batch_size 1 --seq_len 2500 --gratings_orientation $orientation --gratings_frequency $frequency --n_simulations 2"
done
done

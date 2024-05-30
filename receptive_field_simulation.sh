

run -g 1 -m 24 -t 2:00 -o Out/receptive.out -e Error/receptive.err -j t2fs "python receptive_field_simulation.py -seq_len 600 -delays '100,0' -n_trials 100 -circle_row 6 -circle_column 5 -orientation 0 -radius_circle 10"

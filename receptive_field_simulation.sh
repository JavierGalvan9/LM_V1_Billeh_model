# only runs one position of the gabor patches

run -g 1 -m 24 -t 5:00 -o Out/receptive.out -e Error/receptive.err -j t2fs "python receptive_field_estimator.py -seq_len 350 -delays '100,0' -n_trials 1 -circle_row 0 -circle_column 0 -orientation 0 -radius_circle 10"

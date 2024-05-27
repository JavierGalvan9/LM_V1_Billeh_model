

run -g 1 -m 24 -t 1:00 -o Out/t2fs.out -e Error/t2fs.err -j t2fs "python time_to_first_spike_simulation.py -seq_len 1250 -delays '1000,0' -n_trials 100"

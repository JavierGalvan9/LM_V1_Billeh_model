run -g 1 -c 4 -m 48 -t 1:00 -o Out/power.out -e Error/power.err -j t2fs "python power_spectrum_simulation.py -seq_len 2100 -delays '100,0' -n_trials 100"

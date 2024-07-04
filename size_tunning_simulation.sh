for r in 5 10 15 20 25 30 35 40 45 50
do
run -g 1 -m 24 -t 1:00 -o Out/receptive_radius${r}.out -e Error/receptive_radius${r}.err -j t2fs "python receptive_field_simulation.py -seq_len 350 -delays '100,0' -n_trials 50 -circle_row 6 -circle_column 5 -orientation 0 -radius_circle $r"
sleep 2
done
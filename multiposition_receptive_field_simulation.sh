for row in {0..12}
do
for col in {0..10}
do
run -g 1 -m 24 -t 3:00 -o Out/receptive_col${col}.out -e Error/receptive_col${col}.err -j t2fs "python receptive_field_simulation.py -seq_len 350 -delays '100,0' -n_trials 50 -circle_row $row -circle_column $col -orientation 0 -radius_circle 10"
sleep 2
done
done
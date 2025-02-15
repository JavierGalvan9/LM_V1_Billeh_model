# for row in {0..10}
# do
# for col in {0..22}
# do
# run -g 1 -c 4 -m 24 -t 0:15 -o Out/receptive_col${col}_row${row}.out -e Error/receptive_col${col}_row${row}.err -j receptive_col${col}_row${row} "python receptive_field_estimator.py --ckpt_dir '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_gvbz' --restore_from 'Best_model' --noconnected_areas --seq_len 350 --delays '100,0' --n_trials 50 --circle_row $row --circle_column $col --radius_circle 10 --dtype float16 --batch_size 5"
# # sleep 2
# done
# done

# col_size=240
# row_size=120
# radius_circle=5
# # Calculate the number of columns and rows
# n_cols=$(echo "($col_size - $radius_circle) / $radius_circle" | bc -l)
# n_cols=$(echo "($n_cols + 0.999) / 1" | bc)  # Ceil operation

# n_rows=$(echo "($row_size - $radius_circle) / $radius_circle" | bc -l)
# n_rows=$(echo "($n_rows + 0.999) / 1" | bc)  # Ceil operation

# # print n_cols and n_rows
# echo "Number of columns: $n_cols"
# echo "Number of rows: $n_rows"

# # for row in $(seq 0 $((n_rows - 1))); do
# for col in $(seq 0 $((n_cols - 1))); do
# run -g 1 -c 4 -m 48 -t 1:00 -o Out/receptive_col${col}.out -e Error/receptive_col${col}.err -j receptive_col${col} "python receptive_field_estimator.py --ckpt_dir '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_gvbz' --restore_from 'Best_model' --seq_len 350 --delays '100,0' --n_trials 50 --circle_column $col --radius_circle $radius_circle --dtype float16 --batch_size 5"
# done
# # done


# run -g 1 -c 4 -m 80 -t 7:15 -o Out/3receptive_col${col}_row${row}.out -e Error/3receptive_col${col}_row${row}.err -j receptive_col${col}_row${row} "python receptive_field_estimator.py --ckpt_dir '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_6mvi' --seq_len 350 --delays '100,0' --n_trials 50 --circle_row 0 --circle_column 0 --radius_circle 10 --dtype float16 --batch_size 5"
# b_3514_mnist
# b_hfm3
# --noconnected_areas


python distributed_receptive_field_estimator.py --no-connected_areas --restore_from '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_gvbz/Best_model' --v1_neurons 100000 --lm_neurons 30000 --seq_len 350 --delays '100,0' --n_trials 50 --row_size 120 --col_size 240 --radius_circle 5 --dtype float16 --batch_size 5
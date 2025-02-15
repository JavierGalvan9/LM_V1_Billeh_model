#! /bin/bash

# run -g 1 -m 24 -t 0:30 -o Out/drifting_gratings2.out -e Error/drifting_gratings2.err -j small "python multi_training.py --batch_size 1 --v1_neurons 10000 --seq_len 600 --n_runs 1 --n_epochs 1 --steps_per_epoch 1"

# python parallel_training_testing.py --spontaneous_training --v1_neurons 100000 --lm_neurons 30000 --realistic_neurons_ratio --osi_cost 0.1 --train_interarea_lm_v1 --train_noise --learning_rate 0.001 --n_runs 2 --n_epochs 100 --steps_per_epoch 20
# python parallel_training_testing.py --delays '500,0' --seq_len 1000 --v1_neurons 100000 --lm_neurons 30000 --realistic_neurons_ratio --osi_cost 1 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --learning_rate 0.001 --n_runs 1 --n_epochs 10 --steps_per_epoch 50

# python parallel_training_testing.py --interarea_weight_distribution 'billeh_weights' --delays '0,0' --seq_len 500 --v1_neurons 10000 --lm_neurons 3000 --realistic_neurons_ratio --osi_cost 1 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --learning_rate 0.001 --n_runs 2 --n_epochs 20 --steps_per_epoch 20

# run -g 1 -m 65 -t 2:00 -o Out/gordo.out -e Error/gordo.err -j gordo "python osi_dsi_estimator.py --core_loss --interarea_weight_distribution 'billeh_weights' --E4_weight_factor 4.0 --v1_neurons 90000 --lm_neurons 30000 --ckpt_dir '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_90000_lm_30000_E4_weight_factor_4.0/b_m4bo' --restore_from 'Best_model' --run_session 1000 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm"

num_replicas=3
training_samples=60000
test_samples=10000
batch_size=14
global_batch_size=$((batch_size*num_replicas))
# calculate steps_per_epoch as integer division
steps_per_epoch=$((training_samples/global_batch_size))
val_steps=$((test_samples/global_batch_size))


# mnist_noise_std=0.0
# run -g 1 -c 4 -m 80 -t 1:00 -o Out/test.out -e Error/test.err -j test "python classification_training.py --mnist_noise_std $mnist_noise_std --test_only --restore_from '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_gvbz_mnist/Best_model' --seq_len 200 --dtype 'float16' --core_loss --v1_neurons 100000 --lm_neurons 30000 --rate_cost 100 --voltage_cost 1 --recurrent_weight_regularization 0 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm"

# for mnist_noise_std in $(awk 'BEGIN{for(i=0.01;i<=100;i*=10)print i}')
# do
#     run -g 1 -c 4 -m 80 -t 1:00 -o Out/test_$mnist_noise_std.out -e Error/test_$mnist_noise_std.err -j test_$mnist_noise_std "python classification_training.py --noconnected_areas --mnist_noise_std $mnist_noise_std --test_only --restore_from '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_oo67_mnist/Best_model' --seq_len 200 --dtype 'float16' --core_loss --v1_neurons 100000 --lm_neurons 30000 --rate_cost 100 --voltage_cost 1 --recurrent_weight_regularization 0 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm"
# done

python classification_training_testing.py --gradient_checkpointing --restore_from '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_86qi/Best_model' --recurrent_dampening_factor 0.5 --dampening_factor 0.5 --batch_size $batch_size --dtype 'float16' --n_gpus $num_replicas --core_loss --v1_neurons 100000 --lm_neurons 30000 --rate_cost 100 --voltage_cost 1 --recurrent_weight_regularization 0 --train_noise --train_recurrent_lm --train_interarea_lm_v1 --train_interarea_v1_lm --n_runs 3 --n_epochs 10 --steps_per_epoch $steps_per_epoch --val_steps $val_steps

# python classification_training_testing.py --test_only --restore_from '/home/jgalvan/Desktop/Neurocoding/LM_V1_Billeh_model/Simulation_results/v1_100000_lm_30000/b_gvbz_mnist/Best_model' --batch_size $batch_size --dtype 'float16' --n_gpus 1 --core_loss --v1_neurons 100000 --lm_neurons 30000 --val_steps $val_steps
# python classification_training_testing.py --recurrent_dampening_factor 0.5 --dampening_factor 0.5 --batch_size $batch_size --dtype 'float16' --n_gpus $num_replicas --core_loss --v1_neurons 100000 --lm_neurons 30000 --rate_cost 100 --voltage_cost 1 --recurrent_weight_regularization 0 --train_noise --train_recurrent_v1 --train_recurrent_lm --train_interarea_lm_v1 --train_interarea_v1_lm --n_runs 1 --n_epochs 10 --steps_per_epoch $steps_per_epoch --val_steps $val_steps

# python classification_training_testing.py --delays '50,50' --seq_len 200 --batch_size $batch_size --dtype 'float16' --n_gpus $num_replicas --core_loss --v1_neurons 100000 --lm_neurons 30000 --rate_cost 100 --voltage_cost 1 --recurrent_weight_regularization 0 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --n_runs 6 --n_epochs 6 --steps_per_epoch $steps_per_epoch --val_steps $val_steps

# python parallel_training_testing.py --core_loss --interarea_weight_distribution 'random_weights' --random_weights --E4_weight_factor 4.0 --v1_neurons 100000 --lm_neurons 30000 --rate_cost 10000 --osi_cost 20 --sync_cost 0.1 --recurrent_weight_regularization 1 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --n_runs 10 --n_epochs 50 --steps_per_epoch 25
# python parallel_training_testing.py --core_loss --interarea_weight_distribution 'billeh_weights' --E4_weight_factor 4.0 --v1_neurons 90000 --lm_neurons 30000 --rate_cost 10000 --osi_cost 10 --sync_cost 0.25 --recurrent_weight_regularization 0.05 --train_noise --train_interarea_lm_v1 --train_interarea_v1_lm --train_recurrent_v1 --train_recurrent_lm --n_runs 20 --n_epochs 1 --steps_per_epoch 254

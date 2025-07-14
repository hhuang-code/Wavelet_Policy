#!/usr/bin/env bash

source ~/.bashrc
conda activate wavelet_policy

export WANDB_MODE=disabled
#export WANDB_MODE=online


scaler=attn # id, conv, mlp, attn (only test attn)

# Wavelet
for converter in 'conv'; do
    for loss_weight in 0.1; do
        for n_layer in 3; do
            for window_size in 10; do
                for seed in 31 42 53; do
                    echo "Running with window_size=$window_size, converter=$converter, loss_weight=$loss_weight, n_layer=$n_layer"

                    CUDA_VISIBLE_DEVICES=0 python train.py --config-name=train_carla state_prior=waveletnet_carla_best \
                    project=causal_wavelet_policy_submit experiment=carla_train_waveletnet \
                    converter=${converter} scaler=${scaler} n_layer=${n_layer} window_size=${window_size} seed=${seed} \
                    detail_loss_weight=${loss_weight} smooth_loss_weight=${loss_weight} \
                    num_prior_epochs=50
                done
            done
        done
    done
done



#!/usr/bin/env bash

source ~/.bashrc

conda activate wavelet_policy

module load gcc/9.2.0
module load mesa/20.2.1

export WANDB_MODE=disabled

export HYDRA_FULL_ERROR=1


python run.py --config-name=sorting_2_config \
              --multirun seed=0 device=cuda:1 \
              agents=wavelet_agent \
              agent_name=wavelet \
              window_size=5 \
              group=sorting_2_wavelet_seeds \
              simulation.n_cores=1 \
              simulation.n_contexts=600 \
              simulation.n_trajectories_per_context=1 \
              simulation.render=True \
              +agents.model.vocab_size=24 \
              +agents.model.offset_loss_scale=1.0 \
              +agents.model.converter='conv' \
              +agents.model.scaler='attn' \
              +agents.model.detail_loss_weight=0.1 \
              +agents.model.smooth_loss_weight=0.1

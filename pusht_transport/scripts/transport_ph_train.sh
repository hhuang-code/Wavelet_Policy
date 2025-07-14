#!/usr/bin/env bash


source ~/.bashrc
conda activate wavelet_policy


export WANDB_MODE=disabled
#export WANDB_MODE=online


# Waveletnet
PROJECT='diffusion_policy'
METHOD_NAME='diffusion_policy_waveletnet'
TASK_NAME='lowdim_transport_ph' # ONLY NEED TO CHANGE THIS
CONFIG_NAME=${TASK_NAME}'_'${METHOD_NAME}'.yaml'

SCALER='attn'
CONVERTER='conv'

SMOOTH_LOSS_WEIGHT=0.1
DETAIl_LOSS_WEIGHT=0.1

# Waveletnet
# To save time, change seed in 42, 43, 44 and run it
for seed in 44; do
    python train.py --config-dir=. --config-name=${CONFIG_NAME} \
    logging.project=${PROJECT} +method_name=${METHOD_NAME} task_name=${TASK_NAME} \
    policy.model.scaler=${SCALER} policy.model.converter=${CONVERTER} \
    policy.smooth_loss_weight=${SMOOTH_LOSS_WEIGHT} policy.detail_loss_weight=${DETAIl_LOSS_WEIGHT} \
    training.seed=${seed} training.device=cuda:0 \
    hydra.run.dir='outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${method_name}_${task_name}_${policy.model.scaler}_${policy.model.converter}_dw${policy.detail_loss_weight}'
done
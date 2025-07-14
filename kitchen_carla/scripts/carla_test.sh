#!/usr/bin/env bash

source ~/.bashrc
conda activate wavelet_policy

export WANDB_MODE=disabled
#export WANDB_MODE=online

export MUJOCO_GL=glfw

#export ASSET_PATH=$(pwd)

#export PYTHONPATH=$PYTHONPATH:$(pwd)/envs

export CUDA_VISIBLE_DEVICES='0'


folder="exp_local/2025.xx.xx"


model="waveletnet"

scaler=attn
converters=("id" "conv" "mlp" "attn")  # id, conv, mlp, attn

# Traverse the folder and select matching subfolders
for subfolder in "${folder}"/*; do
    if [[ -d "${subfolder}" ]] && [[ "${subfolder}" == *"train"* ]] && [[ "${subfolder}" == *"carla"* ]] && [[ "${subfolder}" == *"$model"* ]]; then

        # Extract the integers after "layer" and "ws" using pattern matching
        n_layer=$(echo "${subfolder}" | sed -n 's/.*layer\([0-9]\+\).*/\1/p')
        window_size=$(echo "${subfolder}" | sed -n 's/.*ws\([0-9]\+\).*/\1/p')
        seed=$(echo "${subfolder}" | sed -n 's/.*seed\([0-9]\+\).*/\1/p')
        detail_loss_weight=$(echo "${subfolder}" | grep -oP '(?<=dw)[0-9.]+')
        smooth_loss_weight=$(echo "${subfolder}" | grep -oP '(?<=sw)[0-9.]+')

        for converter in "${converters[@]}"; do
            if [[ ${subfolder} == *"${converter}"* ]]; then
                break
            fi
        done

        echo ${subfolder}
        echo ${converter}
        echo ${n_layer}
        echo ${window_size}
        echo ${seed}
        echo ${detail_loss_weight}
        echo ${smooth_loss_weight}

        # Traverse through the given folder and find all files with the pattern "snapshot_xx.pt"
        find "${subfolder}" -type f -name "snapshot_*.pt" | while read -r file; do
            # Extract the integer part from the filename
            filename=$(basename "${file}")
            snapshot_number=$(echo "${filename}" | grep -oP '(?<=snapshot_)\d+')

            # Create a new subfolder named "epoch_integer" under the given folder
            new_folder="${subfolder}/epoch_${snapshot_number}"
            mkdir -p "${new_folder}"

            # Move the file into the new subfolder and rename it to "snapshot.pt"
            cp "${file}" "${new_folder}/snapshot.pt"

            echo ${new_folder}

            # Loop indefinitely until the file is found
            while true; do
                if [ -f "${new_folder}/snapshot.pt" ]; then
                    echo "File found: ${new_folder}/snapshot.pt"
                    break
                else
                    echo "File not found, checking again..."
                    sleep 3  # Pause for 5 seconds before checking again
                fi
            done

            python run_on_env.py --config-name=eval_carla state_prior=waveletnet_carla_best experiment=carla_eval_wavelet \
            converter=${converter} scaler=${scaler} n_layer=${n_layer} window_size=${window_size} \
            seed=${seed} epoch=${snapshot_number} \
            detail_loss_weight=${detail_loss_weight} smooth_loss_weight=${smooth_loss_weight} \
            env.load_dir=${new_folder} \
            record_video=False

        done

    fi
done

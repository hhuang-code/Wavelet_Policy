#!/usr/bin/env bash


source ~/.bashrc
conda activate wavelet_policy


export WANDB_MODE=disabled
#export WANDB_MODE=online


BASE_DIR='/scratch/hh1811/projects/diffusion_policy-main/outputs'

TASK_NAME='transport_mh'

MODEL_NAME='waveletnet'

# Use find to locate all .ckpt files under BASE_DIR
find "$BASE_DIR" -type f -name "*.ckpt" -print0 | while IFS= read -r -d '' CHECKPOINT; do

    # Extract the filename without the path
    file_name=$(dirname "$CHECKPOINT")

    # Check if filename contains the keyword
    if [[ "$file_name" != *"$TASK_NAME"* || "$file_name" != *"$MODEL_NAME"* ]]; then
#        echo "Skipping $CHECKPOINT - does not contain keyword '$KEYWORD'"
        continue
    fi

    # Extract the checkpoints directory path
    checkpoints_dir=$(dirname "$CHECKPOINT")
    # Extract the filename without the path
    FILE_NAME=$(basename "$CHECKPOINT")
    # Remove .ckpt extension to get FILE_DIR
    FILE_DIR="${FILE_NAME%.ckpt}"
    # Construct the output directory path
    OUTPUT_DIR="$checkpoints_dir/$FILE_DIR"

    # Check if the output directory exists
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Processing checkpoint: $CHECKPOINT"
        echo "Output directory: $OUTPUT_DIR"
        # Run the evaluation command
        python eval.py \
            --checkpoint "$CHECKPOINT" \
            --output_dir "$OUTPUT_DIR" \
            --device cuda:0
    else
        :
#        echo "Output directory '$OUTPUT_DIR' already exists. Skipping checkpoint: $CHECKPOINT"
    fi
done
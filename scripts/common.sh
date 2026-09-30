#!/bin/bash
# Shared settings for the run scripts. Override via environment variables.
DATASET=${DATASET:-harvard}          # harvard | librispeech | commands
ASR_MODEL=${ASR_MODEL:-whisper-tiny} # whisper-tiny | whisper-large-v3-turbo | wav2vec2-large | ...
SEED=${SEED:-0}
GPU=${GPU:-0}
START=${START:-1}
END=${END:-100}

source ~/miniconda3/etc/profile.d/conda.sh

generate_references() {
    python scripts/generate_reference_audios.py --dataset $DATASET --start $START --end $END --seed $SEED --gpu $GPU
}

evaluate() {  # $1: method
    python scripts/evaluate_results.py --method $1 --dataset $DATASET --gpu $GPU
}

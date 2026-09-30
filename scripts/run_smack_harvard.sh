#!/bin/bash
# SMACK on Harvard sentences 1–100, single process.
# Run from project root: bash scripts/run_smack_harvard.sh
source ~/miniconda3/etc/profile.d/conda.sh

ASR_MODEL=${ASR_MODEL:-whisper}
SEED=${SEED:-0}
GPU=${GPU:-0}
START=1
END=100

# ---------- Generate shared reference audios first ----------
python scripts/generate_harvard_audios.py --start $START --end $END --seed $SEED --gpu $GPU

conda run --no-capture-output -n smack python scripts/adversarial_smack_harvard.py \
    --start $START --end $END \
    --seed $SEED \
    --gpu $GPU \
    --asr_model $ASR_MODEL

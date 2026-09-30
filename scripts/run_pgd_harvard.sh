#!/bin/bash
# PGD on Harvard sentences 1–100, single process (one robust_speech run per sentence).
# Run from project root: bash scripts/run_pgd_harvard.sh
source ~/miniconda3/etc/profile.d/conda.sh

ASR_MODEL=${ASR_MODEL:-whisper}
SEED=${SEED:-0}
GPU=${GPU:-0}
START=1
END=100

# ---------- Generate shared reference audios first ----------
python scripts/generate_harvard_audios.py --start $START --end $END --seed $SEED --gpu $GPU

conda run --no-capture-output -n pgd python scripts/adversarial_pgd_harvard.py \
    --start $START --end $END \
    --seed $SEED \
    --gpu $GPU \
    --asr_model $ASR_MODEL

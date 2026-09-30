#!/bin/bash
# Waveform baseline on Harvard sentences 1–100, single process.
# Run from project root: bash scripts/run_waveform_harvard.sh
ASR_MODEL=${ASR_MODEL:-whisper}
SEED=${SEED:-0}
GPU=${GPU:-0}
START=1
END=100

# ---------- Generate shared reference audios first ----------
python scripts/generate_harvard_audios.py --start $START --end $END --seed $SEED --gpu $GPU

python scripts/adversarial_waveform_harvard.py \
    --sentence_start $START \
    --sentence_end $END \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --noise_scale 0.05 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target \
    --seed $SEED \
    --gpu $GPU \
    --asr_model $ASR_MODEL

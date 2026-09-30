#!/bin/bash
# GATAS on Harvard sentences 1–100, single process.
# Run from project root: bash scripts/run_gatas_harvard.sh
ASR_MODEL=${ASR_MODEL:-whisper}
SEED=${SEED:-0}
GPU=${GPU:-0}
START=1
END=100

# ---------- Generate shared reference audios first ----------
python scripts/generate_harvard_audios.py --start $START --end $END --seed $SEED --gpu $GPU

python scripts/adversarial_gatas_harvard.py \
    --harvard_sentences_start $START \
    --harvard_sentences_end $END \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --iv_scalar 0.5 \
    --size_per_phoneme 1 \
    --num_rms_candidates 1 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target \
    --seed $SEED \
    --gpu $GPU \
    --asr_model $ASR_MODEL

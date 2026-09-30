#!/bin/bash
# Waveform baseline. Run from project root: bash scripts/run_waveform.sh
source scripts/common.sh
generate_references

python scripts/adversarial_waveform.py \
    --dataset $DATASET --start $START --end $END \
    --loop_count 1 \
    --num_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate Waveform

#!/bin/bash
# Waveform baseline. Run from project root: bash scripts/run_waveform.sh
source scripts/common.sh
# Search budget: POP_SIZE x NUM_GENERATIONS ASR queries per sentence (no early stopping)
POP_SIZE=${POP_SIZE:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-10}
generate_references

python scripts/adversarial_waveform.py \
    --dataset $DATASET --start $START --end $END \
    --loop_count 1 \
    --num_generations $NUM_GENERATIONS \
    --min_generations $NUM_GENERATIONS \
    --pop_size $POP_SIZE \
    --batch_size $POP_SIZE \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate Waveform

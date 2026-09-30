#!/bin/bash
# GATAS (untargeted). Run from project root: bash scripts/run_gatas.sh
source scripts/common.sh
# Search budget: POP_SIZE x NUM_GENERATIONS ASR queries per sentence (no early stopping)
POP_SIZE=${POP_SIZE:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-10}
generate_references

python scripts/adversarial_gatas.py \
    --dataset $DATASET --start $START --end $END \
    --loop_count 1 \
    --num_generations $NUM_GENERATIONS \
    --min_generations $NUM_GENERATIONS \
    --pop_size $POP_SIZE \
    --batch_size $POP_SIZE \
    --iv_scalar 0.5 \
    --size_per_phoneme 1 \
    --num_rms_candidates 1 \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target \
    --method_name ${METHOD_NAME:-GATAS} \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate ${METHOD_NAME:-GATAS}

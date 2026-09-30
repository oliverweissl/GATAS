#!/bin/bash
# GATAS (targeted, for the comparison with native SMACK). Run from project root: bash scripts/run_gatas_targeted.sh
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
    --size_per_phoneme 1 \
    --objectives "PESQ=0.2, WER_TARGET=0.0" \
    --mode TARGETED \
    --seed_target \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate GATAS_targeted

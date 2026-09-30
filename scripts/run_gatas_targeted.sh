#!/bin/bash
# GATAS (targeted, for the comparison with native SMACK). Run from project root: bash scripts/run_gatas_targeted.sh
source scripts/common.sh
generate_references

python scripts/adversarial_gatas.py \
    --dataset $DATASET --start $START --end $END \
    --loop_count 1 \
    --num_generations 100 \
    --min_generations 100 \
    --pop_size 100 \
    --batch_size 100 \
    --size_per_phoneme 1 \
    --objectives "PESQ=0.2, WER_TARGET=0.0" \
    --mode TARGETED \
    --seed_target \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate GATAS_targeted

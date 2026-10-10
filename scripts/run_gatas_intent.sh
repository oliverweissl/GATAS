#!/bin/bash
# GATAS, intent matching: the objective is 1 - P(target intent | transcription) of the intent classifier (commands only).
# Run from project root: DATASET=commands bash scripts/run_gatas_intent.sh
source scripts/common.sh
POP_SIZE=${POP_SIZE:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-10}
generate_references

python scripts/adversarial_gatas.py \
    --dataset $DATASET --start $START --end $END \
    --loop_count 1 \
    --num_generations $NUM_GENERATIONS \
    --pop_size $POP_SIZE \
    --batch_size $POP_SIZE \
    --iv_scalar 4 --adaptive_scale \
    --basis jacobian --text_dct 8 --algorithm cmaes \
    --objectives "PESQ=0.2, INTENT_TARGET=0.5" \
    --mode NOISE_UNTARGETED \
    --method_name GATAS_intent \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate GATAS_intent

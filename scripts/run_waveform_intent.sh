#!/bin/bash
# Waveform baseline, intent matching (commands only). Run from project root: DATASET=commands bash scripts/run_waveform_intent.sh
source scripts/common.sh
POP_SIZE=${POP_SIZE:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-15}
generate_references

conda run --no-capture-output -n gatas python scripts/adversarial_waveform.py \
    --dataset $DATASET --start $START --end $END \
    --loop_count 1 \
    --num_generations $NUM_GENERATIONS \
    --pop_size $POP_SIZE \
    --batch_size $POP_SIZE \
    --objectives "PESQ=0.2, INTENT_TARGET=0.5" \
    --mode NOISE_UNTARGETED \
    --seed_target \
    --method_name Waveform_intent \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate Waveform_intent

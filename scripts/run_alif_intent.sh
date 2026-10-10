#!/bin/bash
# ALIF, intent matching: loss is 1 - P(target intent | transcription) (commands only).
# Run from project root: DATASET=commands bash scripts/run_alif_intent.sh
source scripts/common.sh
generate_references

conda run --no-capture-output -n alif python scripts/adversarial_alif.py \
    --dataset $DATASET --start $START --end $END \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL \
    --intent_targeted

evaluate ALIF_intent

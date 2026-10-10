#!/bin/bash
# SMACK, intent matching: fitness uses the intent classifier output (commands only).
# Run from project root: DATASET=commands bash scripts/run_smack_intent.sh
source scripts/common.sh
generate_references

conda run --no-capture-output -n smack python scripts/adversarial_smack.py \
    --dataset $DATASET --start $START --end $END \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL \
    --intent_targeted

evaluate SMACK_intent

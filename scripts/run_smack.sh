#!/bin/bash
# SMACK, untargeted adaptation. Run from project root: bash scripts/run_smack.sh
source scripts/common.sh
generate_references

conda run --no-capture-output -n smack python scripts/adversarial_smack.py \
    --dataset $DATASET --start $START --end $END \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL \
    --untargeted

evaluate SMACK

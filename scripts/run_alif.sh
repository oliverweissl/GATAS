#!/bin/bash
# ALIF (ALIF-OTA, untargeted adaptation). Run from project root: bash scripts/run_alif.sh
source scripts/common.sh
generate_references

conda run --no-capture-output -n alif python scripts/adversarial_alif.py \
    --dataset $DATASET --start $START --end $END \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate ALIF

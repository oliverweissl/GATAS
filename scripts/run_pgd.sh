#!/bin/bash
# PGD (white-box, attacks Whisper-tiny only). Run from project root: bash scripts/run_pgd.sh
source scripts/common.sh
generate_references

conda run --no-capture-output -n pgd python scripts/adversarial_pgd.py \
    --dataset $DATASET --start $START --end $END \
    --seed $SEED --gpu $GPU

evaluate PGD

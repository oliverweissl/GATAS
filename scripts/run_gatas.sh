#!/bin/bash
# GATAS (untargeted) with the best configuration of the ablations (METHODOLOGY.md): CMA-ES on the Chebyshev scalarization,
# Jacobian basis, DCT text coefficients, adaptive scale in [0.5, 4].
# Run from project root: bash scripts/run_gatas.sh   (pin a GPU with CUDA_VISIBLE_DEVICES=<n> GPU=0: UTMOS is hardcoded to cuda:0)
source scripts/common.sh
# Search budget: POP_SIZE x NUM_GENERATIONS ASR queries per sentence (300, full budget, no early stopping)
POP_SIZE=${POP_SIZE:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-15}
generate_references

conda run --no-capture-output -n gatas python scripts/adversarial_gatas.py \
    --dataset $DATASET --start $START --end $END \
    --loop_count 1 \
    --num_generations $NUM_GENERATIONS \
    --pop_size $POP_SIZE \
    --batch_size $POP_SIZE \
    --iv_scalar 4 --adaptive_scale \
    --basis jacobian --text_dct 8 --algorithm cmaes \
    --objectives "PESQ=0.2, SET_OVERLAP=0.5" \
    --mode NOISE_UNTARGETED \
    --method_name ${METHOD_NAME:-GATAS} \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate ${METHOD_NAME:-GATAS}

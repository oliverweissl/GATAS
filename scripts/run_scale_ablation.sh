#!/bin/bash
# Perturbation-scale ablation of GATAS on 20 sentences: fixed scale 0.5 / 1 / 2 and an adaptive scale (gene in [0.5, 4]).
# Run from project root: bash scripts/run_scale_ablation.sh   (analysis: analysis/RQ0_Ablation.ipynb)
source scripts/common.sh
END=20  # the ablation uses sentences 1-20 (set after common.sh, which defaults END to 100)
POP_SIZE=${POP_SIZE:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-15}
generate_references

run() {  # $1: method name, remaining: scale arguments
    name=$1; shift
    conda run --no-capture-output -n gatas python scripts/adversarial_gatas.py \
        --dataset $DATASET --start $START --end $END \
        --loop_count 1 --num_generations $NUM_GENERATIONS --pop_size $POP_SIZE --batch_size $POP_SIZE \
        --objectives "PESQ=0.2, SET_OVERLAP=0.5" --mode NOISE_UNTARGETED \
        --method_name $name --seed $SEED --gpu $GPU --asr_model $ASR_MODEL "$@"
    evaluate $name
}

run GATAS_scale_0.5 --iv_scalar 0.5
run GATAS_scale_1 --iv_scalar 1
run GATAS_scale_2 --iv_scalar 2
run GATAS_adaptive --iv_scalar 4 --adaptive_scale

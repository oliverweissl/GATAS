#!/bin/bash
# Ablation of GATAS on intent matching (commands, sentences 1-20): intent-first candidate selection and the margin objective
# (sigmoid of target logit - best other logit, threshold 0.5 = target is the predicted intent) instead of 1 - P(target).
# The reference is GATAS_intent from scripts/run_gatas_intent.sh (sentences 1-20).
# Run from project root: CUDA_VISIBLE_DEVICES=1 bash scripts/run_intent_ablation.sh   (UTMOS is hardcoded to cuda:0)
DATASET=commands
source scripts/common.sh
END=20  # the ablation uses sentences 1-20 (set after common.sh, which defaults END to 100)
POP_SIZE=${POP_SIZE:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-15}
generate_references

run() {  # $1: method name, $2: intent objective, remaining: extra arguments
    name=$1; objective=$2; shift 2
    conda run --no-capture-output -n gatas python scripts/adversarial_gatas.py \
        --dataset $DATASET --start $START --end $END \
        --loop_count 1 --num_generations $NUM_GENERATIONS --pop_size $POP_SIZE --batch_size $POP_SIZE \
        --iv_scalar 4 --adaptive_scale --basis jacobian --text_dct 8 --algorithm cmaes \
        --objectives "PESQ=0.2, $objective=0.5" --mode NOISE_UNTARGETED \
        --method_name $name --seed $SEED --gpu $GPU --asr_model $ASR_MODEL "$@"
    evaluate $name
}

run GATAS_intent_sel INTENT_TARGET --intent_selection
run GATAS_intent_margin INTENT_MARGIN
run GATAS_intent_margin_sel INTENT_MARGIN --intent_selection

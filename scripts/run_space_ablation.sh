#!/bin/bash
# Search-space / optimizer ablation of GATAS on 20 sentences, all with the adaptive scale (gene in [0.5, 4]):
# DCT text basis, output-sensitive (Jacobian) basis, CMA-ES and MO-CMA-ES instead of NSGA-II. The reference is GATAS_adaptive
# from scripts/run_scale_ablation.sh.
# Run from project root: bash scripts/run_space_ablation.sh   (analysis: analysis/Ablation_Space.ipynb)
source scripts/common.sh
END=20  # the ablation uses sentences 1-20 (set after common.sh, which defaults END to 100)
POP_SIZE=${POP_SIZE:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-15}
TEXT_DCT=${TEXT_DCT:-8}
generate_references

run() {  # $1: method name, remaining: search-space arguments; ONLY=<method name> runs just that configuration
    name=$1; shift
    [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && return
    conda run --no-capture-output -n gatas python scripts/adversarial_gatas.py \
        --dataset $DATASET --start $START --end $END \
        --loop_count 1 --num_generations $NUM_GENERATIONS --pop_size $POP_SIZE --batch_size $POP_SIZE \
        --objectives "PESQ=0.2, SET_OVERLAP=0.5" --mode NOISE_UNTARGETED --iv_scalar 4 --adaptive_scale \
        --method_name $name --seed $SEED --gpu $GPU --asr_model $ASR_MODEL "$@"
    evaluate $name
}

run GATAS_dct --text_dct $TEXT_DCT
run GATAS_jac --basis jacobian
run GATAS_jacdct --basis jacobian --text_dct $TEXT_DCT
run GATAS_cmaes --algorithm cmaes
run GATAS_mocma --algorithm mocma
run GATAS_cmaes_jacdct --algorithm cmaes --basis jacobian --text_dct $TEXT_DCT
run GATAS_mocma_jacdct --algorithm mocma --basis jacobian --text_dct $TEXT_DCT

#!/bin/bash
# Sequential queue of runs on ONE physical GPU. Run from project root:
#   nohup bash scripts/queue_remaining.sh 0 > outputs/logs/queue_gpu0.log 2>&1 &
#   nohup bash scripts/queue_remaining.sh 1 > outputs/logs/queue_gpu1.log 2>&1 &
# GATAS / Waveform / PGD / evaluation: CUDA_VISIBLE_DEVICES=<n> + --gpu 0 (UTMOS is hardcoded to cuda:0).
# ALIF: adversarial_alif.py sets CUDA_VISIBLE_DEVICES from --gpu itself (physical index); its evaluation is remapped to cuda:0.
cd "$(dirname "$0")/.."
G=${1:?usage: queue_remaining.sh <0|1>}

run() {  # $1: dataset, $2: run script
    echo "=== $(date '+%F %T') START $2 [$1] gpu$G"
    CUDA_VISIBLE_DEVICES=$G DATASET=$1 GPU=0 bash scripts/$2.sh
    echo "=== $(date '+%F %T') END $2 [$1]"
}

run_alif() {  # $1: dataset, $2: "" | "--intent_targeted", $3: result method name
    echo "=== $(date '+%F %T') START alif $2 [$1] gpu$G"
    (
        source scripts/common.sh
        DATASET=$1
        CUDA_VISIBLE_DEVICES=$G GPU=0 generate_references
        conda run --no-capture-output -n alif python scripts/adversarial_alif.py \
            --dataset $1 --start $START --end $END --seed $SEED --gpu $G --asr_model $ASR_MODEL $2
        CUDA_VISIBLE_DEVICES=$G GPU=0 evaluate $3
    )
    echo "=== $(date '+%F %T') END alif $2 [$1]"
}

if [ "$G" = 0 ]; then
    run harvard run_gatas
    run harvard run_waveform
    run harvard run_pgd
    run harvard run_gatas_targeted
    run_alif harvard "" ALIF
    run_alif commands "--intent_targeted" ALIF_intent
else
    run commands run_gatas
    run commands run_waveform
    run commands run_pgd
    run commands run_gatas_intent
    run commands run_waveform_intent
    run_alif commands "" ALIF
fi
echo "=== QUEUE DONE gpu$G"

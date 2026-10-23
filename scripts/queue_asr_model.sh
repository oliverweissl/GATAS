#!/bin/bash
# Sequential queue of the paper's experiment matrix for ONE ASR model on ONE physical GPU (no PGD: it is Whisper-specific).
# Run from project root, one queue per GPU:
#   ASR_MODEL=wav2vec2-large nohup bash scripts/queue_asr_model.sh 0 > outputs/logs/queue_w2v_gpu0.log 2>&1 &
#   ASR_MODEL=wav2vec2-large nohup bash scripts/queue_asr_model.sh 1 > outputs/logs/queue_w2v_gpu1.log 2>&1 &
# GATAS / Waveform / evaluation: CUDA_VISIBLE_DEVICES=<n> + --gpu 0 (UTMOS is hardcoded to cuda:0).
# SMACK / ALIF: the attack script sets CUDA_VISIBLE_DEVICES from --gpu itself (physical index); the evaluation is remapped to cuda:0.
cd "$(dirname "$0")/.."
G=${1:?usage: queue_asr_model.sh <0|1>}
export ASR_MODEL=${ASR_MODEL:?set ASR_MODEL}
export SEED=${SEED:-0}
source scripts/common.sh

run() {  # $1: dataset, $2: run script (GATAS and Waveform scripts)
    echo "=== $(date '+%F %T') START $2 [$1] $ASR_MODEL gpu$G"
    CUDA_VISIBLE_DEVICES=$G DATASET=$1 GPU=0 bash scripts/$2.sh || echo "!!! FAILED $2 [$1]"
    echo "=== $(date '+%F %T') END $2 [$1]"
}

run_ext() {  # $1: dataset, $2: smack|alif, $3: attack flags, $4: result method name
    echo "=== $(date '+%F %T') START $2 $3 [$1] $ASR_MODEL gpu$G"
    (
        DATASET=$1
        CUDA_VISIBLE_DEVICES=$G GPU=0 generate_references
        conda run --no-capture-output -n $2 python scripts/adversarial_$2.py \
            --dataset $1 --start $START --end $END --seed $SEED --gpu $G --asr_model $ASR_MODEL $3 \
            && CUDA_VISIBLE_DEVICES=$G GPU=0 evaluate $4
    ) || echo "!!! FAILED $2 $3 [$1]"
    echo "=== $(date '+%F %T') END $2 $3 [$1]"
}

if [ "$G" = 0 ]; then  # Harvard (ASR task)
    run harvard run_gatas
    run harvard run_waveform
    run harvard run_gatas_targeted
    run_ext harvard smack "--untargeted" SMACK_untargeted
    run_ext harvard smack "" SMACK
    run_ext harvard alif "" ALIF
else                   # commands (intent matching)
    run commands run_gatas
    run commands run_waveform
    run commands run_gatas_intent
    run commands run_waveform_intent
    run_ext commands alif "" ALIF
    run_ext commands alif "--intent_targeted" ALIF_intent
    run_ext commands smack "" SMACK  # native SMACK, target text = intent rewrite (as GATAS_llm)
fi
echo "=== QUEUE DONE gpu$G"

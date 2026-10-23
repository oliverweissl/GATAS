#!/bin/bash
# Native targeted SMACK for one (dataset, ASR model) on one GPU, with its own SMACK work directory so that several runs
# can share the machine. On commands the target text is the intent rewrite (as GATAS_llm).
#   bash scripts/queue_smack_rerun.sh <gpu> <dataset> <asr_model> [--resume]
cd "$(dirname "$0")/.."
G=${1:?gpu}; export DATASET=${2:?dataset}; export ASR_MODEL=${3:?asr_model}; RESUME=${4:-}
source scripts/common.sh  # the smack conda env no longer exists; alif was cloned from it (+unidecode)
export SMACK_WORKDIR=outputs/smack_workdir_${DATASET}_${ASR_MODEL}_gpu$G
SMACK_WORKDIR=$(realpath -m $SMACK_WORKDIR); export SMACK_WORKDIR
CUDA_VISIBLE_DEVICES=$G GPU=0 generate_references
conda run --no-capture-output -n ${SMACK_ENV:-alif} python scripts/adversarial_smack.py \
    --dataset $DATASET --start $START --end $END --seed $SEED --gpu $G --asr_model $ASR_MODEL $RESUME \
    && CUDA_VISIBLE_DEVICES=$G GPU=0 evaluate SMACK

#!/bin/bash
# Full experiment matrix of the paper, sequentially. Run from project root: bash scripts/run_experiments.sh
# Each block can also be run on its own with the individual run scripts and DATASET / ASR_MODEL set.
set -e
export SEED=${SEED:-0} GPU=${GPU:-0}

# 1. Main comparison (RQ1-RQ3): all methods, Whisper-tiny, Harvard (ASR task) + SLURP commands (intent matching)
for dataset in harvard commands; do
    for script in run_gatas run_waveform run_smack run_alif run_pgd; do
        DATASET=$dataset ASR_MODEL=whisper-tiny bash scripts/$script.sh
    done
done

# 2. Objective formulation: native targeted SMACK vs targeted GATAS (untargeted SMACK is part of block 1)
for script in run_smack_targeted run_gatas_targeted; do
    DATASET=harvard ASR_MODEL=whisper-tiny bash scripts/$script.sh
done

# 3. Intent matching (commands): the methods that can use the intent classifier output optimize toward the target intent
#    (PGD is white-box on Whisper's text decoder and stays untargeted, see block 1)
for script in run_gatas_intent run_waveform_intent run_smack_intent run_alif_intent; do
    DATASET=commands ASR_MODEL=whisper-tiny bash scripts/$script.sh
done

#!/bin/bash
# Full experiment matrix of the paper, sequentially. Run from project root: bash scripts/run_experiments.sh
# Each block can also be run on its own with the individual run scripts and DATASET / ASR_MODEL set.
set -e
export SEED=${SEED:-0} GPU=${GPU:-0}

# 1. Main comparison (RQ1-RQ3): all methods, Whisper-tiny, Harvard + LibriSpeech
for dataset in harvard librispeech; do
    for script in run_gatas run_waveform run_smack run_alif run_pgd; do
        DATASET=$dataset ASR_MODEL=whisper-tiny bash scripts/$script.sh
    done
done

# 2. Objective formulation: native targeted SMACK vs targeted GATAS (untargeted SMACK is part of block 1)
for script in run_smack_targeted run_gatas_targeted; do
    DATASET=harvard ASR_MODEL=whisper-tiny bash scripts/$script.sh
done

# 3. Generalization: GATAS and its waveform ablation on all datasets and ASR models
for asr in whisper-tiny whisper-large-v3-turbo wav2vec2-large; do
    for dataset in harvard librispeech commands; do
        if [ "$asr" = "whisper-tiny" ] && [ "$dataset" != "commands" ]; then continue; fi  # done in block 1
        for script in run_gatas run_waveform; do
            DATASET=$dataset ASR_MODEL=$asr bash scripts/$script.sh
        done
    done
done

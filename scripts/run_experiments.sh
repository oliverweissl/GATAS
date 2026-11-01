#!/bin/bash
# Full experiment matrix of the paper, sequentially. Run from project root: bash scripts/run_experiments.sh
# Each block can also be run on its own with the individual run scripts and DATASET / ASR_MODEL set.
set -e
export SEED=${SEED:-0} GPU=${GPU:-0}

for asr in whisper-tiny wav2vec2-large; do
    # 1. ASR testing (Harvard, RQ1-RQ3). PGD is white-box on Whisper only.
    for script in run_gatas run_waveform run_smack_untargeted run_smack run_alif; do
        DATASET=harvard ASR_MODEL=$asr bash scripts/$script.sh
    done
    if [ "$asr" = whisper-tiny ]; then DATASET=harvard ASR_MODEL=$asr bash scripts/run_pgd.sh; fi

    # 2. Intent matching (SLURP commands, RQ1-RQ3): the methods that can use the classifier output optimize toward the
    #    target intent; SMACK uses the LLM intent rewrite as its native target text; PGD stays untargeted.
    for script in run_gatas_intent run_gatas_llm run_waveform_intent run_smack run_alif_intent; do
        DATASET=commands ASR_MODEL=$asr bash scripts/$script.sh
    done
    if [ "$asr" = whisper-tiny ]; then DATASET=commands ASR_MODEL=$asr bash scripts/run_pgd.sh; fi
done

# 3. Ablations (RQ0, Whisper-tiny, sentences 1-20)
ASR_MODEL=whisper-tiny bash scripts/run_scale_ablation.sh
ASR_MODEL=whisper-tiny bash scripts/run_space_ablation.sh
ASR_MODEL=whisper-tiny bash scripts/run_intent_ablation.sh

# 4. Post-processing for the analysis notebooks: semantic outcome (needs a local ollama server) and normalized intents
conda run --no-capture-output -n gatas python scripts/semantic_rescore.py
(cd analysis && conda run --no-capture-output -n gatas python normalize_intents.py)

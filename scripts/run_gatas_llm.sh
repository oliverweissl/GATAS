#!/bin/bash
# GATAS_llm, intent matching: the search interpolates in StyleTTS2's latent space between the seed sentence and an LLM rewrite of it
# that has the target intent (src/data/seed_sentences/commands.intent_rewrites.txt, scripts/generate_intent_rewrites.py).
# Objective as GATAS_intent (1 - P(target intent)), intent-first candidate selection, 8 alphas along the phonemes + 2 style alphas.
# Run from project root: CUDA_VISIBLE_DEVICES=<n> GPU=0 bash scripts/run_gatas_llm.sh   (UTMOS is hardcoded to cuda:0)
DATASET=commands
source scripts/common.sh
POP_SIZE=${POP_SIZE:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-15}
generate_references

conda run --no-capture-output -n gatas python scripts/adversarial_gatas.py \
    --dataset $DATASET --start $START --end $END \
    --loop_count 1 \
    --num_generations $NUM_GENERATIONS \
    --pop_size $POP_SIZE \
    --batch_size $POP_SIZE \
    --target_interp --interp_knots 8 --algorithm cmaes \
    --objectives "PESQ=0.2, INTENT_TARGET=0.5" \
    --mode NOISE_UNTARGETED \
    --intent_selection \
    --method_name GATAS_llm \
    --seed $SEED --gpu $GPU --asr_model $ASR_MODEL

evaluate GATAS_llm

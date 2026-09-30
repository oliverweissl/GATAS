"""
SMACK baseline (third_party/SMACK via src/baselines/smack.py), untargeted (--untargeted) or targeted.

Generate reference audios first (main env):
    python scripts/generate_reference_audios.py --dataset harvard --start 1 --end 100

Then run from project root (smack env):
    python scripts/adversarial_smack.py --dataset harvard --start 1 --end 100 --untargeted
"""

import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import soundfile as sf

from src.baselines import smack
from src.data.datasets import DATASETS, load_sentences
from src.helper import set_seed, run_seed, reference_paths, results_dir, experiment_name, target_sentence
from src.models import ASR_MODEL_CHOICES, canonical_asr_model_name
from src.trainer.result_writer import save_attack_result


def main():
    parser = argparse.ArgumentParser(description='SMACK baseline')
    parser.add_argument('--dataset', type=str, default='harvard', choices=DATASETS)
    parser.add_argument('--start', type=int, default=1, help='First sentence ID (1-based)')
    parser.add_argument('--end', type=int, default=100, help='Last sentence ID (1-based, inclusive)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use')
    parser.add_argument("--untargeted", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (per sentence: seed + sentence_id)")
    parser.add_argument("--asr_model", type=str, default="whisper-tiny", choices=ASR_MODEL_CHOICES,
                        help="ASR model attacked by SMACK")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")

    asr_model = canonical_asr_model_name(args.asr_model)
    method = 'SMACK' if args.untargeted else 'SMACK_targeted'
    sentences = load_sentences(args.dataset)
    output_base = results_dir(method, args.dataset, experiment_name(asr_model))

    print(f"Dataset: {args.dataset} | sentences {args.start} → {args.end}")
    print(f"Population: {smack.POPULATION_SIZE} | Genetic: {smack.GENETIC_ITERATIONS} | Gradient: {smack.GRADIENT_ITERATIONS}")
    print(f"ASR model: {asr_model} | method: {method} | results: {output_base}")
    print('=' * 60)

    smack.setup(asr_model)

    for sentence_id in range(args.start, args.end + 1):
        sentence_text = sentences[sentence_id - 1]
        reference_audio, _ = reference_paths(args.dataset, sentence_id)
        if not os.path.exists(reference_audio):
            print(f"[{sentence_id:3d}] Reference audio not found, skipping: {reference_audio}")
            continue

        print(f"\n{'='*60}\n[Sentence {sentence_id}] {sentence_text}\n{'='*60}")

        seed = run_seed(args.seed, sentence_id)
        set_seed(seed)
        target = None if args.untargeted else target_sentence(sentences, sentence_id, args.seed)

        # End-to-end: from attack start until the adversarial audio exists
        attack_start = time.time()
        audio_16k = smack.attack(reference_audio, sentence_text, target)
        elapsed = time.time() - attack_start
        queries = smack.query_count
        print(f"Attack finished. Time: {elapsed:.2f}s | queries: {queries}\n")

        gt_audio = sf.read(reference_audio, dtype='float32')[0]
        save_attack_result(
            output_dir=output_base,
            method=method,
            dataset=args.dataset,
            sentence_id=sentence_id,
            asr_model=asr_model,
            audio=audio_16k,
            gt_audio=gt_audio,
            gt_text=sentence_text,
            transcription=smack.transcribe(audio_16k),
            gt_transcription=smack.transcribe(gt_audio),
            elapsed_seconds=elapsed,
            queries=queries,
            generations=smack.GENETIC_ITERATIONS + smack.GRADIENT_ITERATIONS,
            pop_size=smack.POPULATION_SIZE,
            seed=seed,
            target_text=target,
            # Benign SMACK synthesis (no perturbation) as PESQ reference; not part of the attack runtime
            clean_reference_audio=smack.clean_reference(reference_audio, sentence_text),
            params={
                'genetic_iterations': smack.GENETIC_ITERATIONS,
                'gradient_iterations': smack.GRADIENT_ITERATIONS,
                'gradient_K': smack.GRADIENT_K,
                'gradient_sigma': smack.GRADIENT_SIGMA,
                'gradient_learning_rate': smack.GRADIENT_LEARNING_RATE,
            },
        )

    print("\n[Done]")


if __name__ == '__main__':
    main()

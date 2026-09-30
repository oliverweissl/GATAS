"""
Adversarial SMACK — Harvard Sentences Experiment

Runs SMACK (third_party/SMACK via src/baselines/smack.py) on each Harvard sentence,
targeted by default or untargeted with --untargeted.

Generate reference audios first (styletts2 env):
    python scripts/generate_harvard_audios.py --start 1 --end 100

Then run this script (smack env) from project root:
    python scripts/adversarial_smack_harvard.py --start 1 --end 10
"""

import os
import random

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import shutil
import argparse
import datetime
import soundfile as sf

from src.baselines import smack
from src.data.harvard_sentences import HARVARD_SENTENCES
from src.helper import set_seed, run_seed, reference_paths, results_dir
from src.models import ASR_MODEL_CHOICES
from src.trainer.attack_summary import compute_attack_summary

AUDIO_DIR = 'outputs/'


def main():
    parser = argparse.ArgumentParser(description='SMACK Attack — Harvard Sentences')
    parser.add_argument('--start', type=int, default=1,
                        help='First Harvard sentence index (1-based)')
    parser.add_argument('--end', type=int, default=10,
                        help='Last Harvard sentence index (1-based, inclusive)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use')
    parser.add_argument("--untargeted", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (per sentence: seed + sentence_id)")
    parser.add_argument("--asr_model", type=str, default="whisper", choices=ASR_MODEL_CHOICES,
                        help="ASR backend attacked by SMACK and used for evaluation")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")

    print(f"Sentences: {args.start} → {args.end}")
    print(f"Population: {smack.POPULATION_SIZE} | Genetic: {smack.GENETIC_ITERATIONS} | Gradient: {smack.GRADIENT_ITERATIONS}")
    print(f"ASR model: {args.asr_model} | untargeted: {args.untargeted}")
    print(f"Audio directory: {AUDIO_DIR}")
    print('=' * 60)

    smack.setup(args.asr_model)

    run_timestamp = f"{args.asr_model}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    method = 'SMACK' if args.untargeted else 'SMACK_targeted'
    output_base = results_dir(method, run_timestamp)

    for sentence_id in range(args.start, args.end + 1):
        sentence_text = HARVARD_SENTENCES[sentence_id - 1]
        reference_audio, _ = reference_paths(sentence_id, AUDIO_DIR)

        if not os.path.exists(reference_audio):
            print(f"[{sentence_id:3d}] Reference audio not found, skipping: {reference_audio}")
            continue

        print(f"\n{'='*60}")
        print(f"[Sentence {sentence_id}] {sentence_text}")
        print('=' * 60)

        sentence_dir = os.path.join(output_base, f'sentence_{sentence_id:03d}')
        os.makedirs(sentence_dir, exist_ok=True)

        set_seed(run_seed(args.seed, sentence_id))
        target_sentence = None if args.untargeted else random.choice([HARVARD_SENTENCES[i] for i in range(len(HARVARD_SENTENCES)) if i != sentence_id - 1])

        # End-to-end: from attack start until the adversarial audio exists
        attack_start = time.time()
        audio_16k = smack.attack(reference_audio, sentence_text, target_sentence)
        adv_path = os.path.join(sentence_dir, 'best_smack.wav')
        gt_dst = os.path.join(sentence_dir, 'ground_truth.wav')
        sf.write(adv_path, audio_16k, 16000)
        elapsed = time.time() - attack_start
        print(f"Attack finished. Time: {elapsed:.2f}s\n")

        shutil.copy(reference_audio, gt_dst)

        # Clean SMACK synthesis (no perturbation) as PESQ reference; not part of the attack runtime
        pesq_ref_path = os.path.join(sentence_dir, 'smack_clean_reference.wav')
        sf.write(pesq_ref_path, smack.clean_reference(reference_audio, sentence_text), 16000)

        compute_attack_summary(
            adversarial_audio_path=adv_path,
            gt_audio_path=gt_dst,
            gt_text=sentence_text,
            target_text=target_sentence,
            attack_method=method,
            num_generations=smack.GENETIC_ITERATIONS + smack.GRADIENT_ITERATIONS,
            pop_size=smack.POPULATION_SIZE,
            elapsed_time_seconds=elapsed,
            output_path=os.path.join(sentence_dir, 'smack_summary.json'),
            pesq_reference_audio_path=pesq_ref_path,
            sentence_id=sentence_id,
            whisper_transcription=smack.transcribe(audio_16k),
            gt_transcription=smack.transcribe(sf.read(gt_dst, dtype='float32')[0]),
            extra={
                'asr_model': args.asr_model,
                'seed': run_seed(args.seed, sentence_id),
            },
        )

    print("\n[Done]")


if __name__ == '__main__':
    main()

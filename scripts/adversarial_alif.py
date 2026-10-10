"""
ALIF baseline (third_party/TASER via src/baselines/alif.py), ALIF-OTA adapted to the untargeted setting.

Generate reference audios first (main env):
    python scripts/generate_reference_audios.py --dataset harvard --start 1 --end 100

Then run from project root (alif env, configs/setup_alif_env.sh):
    python scripts/adversarial_alif.py --dataset harvard --start 1 --end 100
"""

import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import soundfile as sf

from src.baselines import alif
from src.data.datasets import DATASETS, load_sentences, load_target_intents
from src.helper import set_seed, run_seed, reference_paths, results_dir, experiment_name
from src.models import ASR_MODEL_CHOICES, canonical_asr_model_name
from src.trainer.result_writer import save_attack_result


def main():
    parser = argparse.ArgumentParser(description='ALIF baseline')
    parser.add_argument('--dataset', type=str, default='harvard', choices=DATASETS)
    parser.add_argument('--start', type=int, default=1, help='First sentence ID (1-based)')
    parser.add_argument('--end', type=int, default=100, help='Last sentence ID (1-based, inclusive)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use')
    parser.add_argument("--intent_targeted", action="store_true",
                        help="Intent matching: minimize 1 - P(target intent | transcription) instead of SET_OVERLAP")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (per sentence: seed + sentence_id)")
    parser.add_argument("--asr_model", type=str, default="whisper-tiny", choices=ASR_MODEL_CHOICES,
                        help="ASR model attacked by ALIF")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")

    asr_model = canonical_asr_model_name(args.asr_model)
    METHOD = 'ALIF_intent' if args.intent_targeted else 'ALIF'
    sentences = load_sentences(args.dataset)
    output_base = results_dir(METHOD, args.dataset, experiment_name(asr_model))

    print(f"Dataset: {args.dataset} | sentences {args.start} → {args.end}")
    print(f"Particles: {alif.NUM_PARTICLES} | Epochs: {alif.EPOCHS} | ASR model: {asr_model} | results: {output_base}")
    print('=' * 60)

    alif.setup(asr_model)

    for sentence_id in range(args.start, args.end + 1):
        sentence_text = sentences[sentence_id - 1]
        reference_audio, _ = reference_paths(args.dataset, sentence_id)
        if not os.path.exists(reference_audio):
            print(f"[{sentence_id:3d}] Reference audio not found, skipping: {reference_audio}")
            continue

        print(f"\n{'='*60}\n[Sentence {sentence_id}] {sentence_text}\n{'='*60}")
        seed = run_seed(args.seed, sentence_id)
        set_seed(seed)
        target_intent = load_target_intents(args.dataset)[sentence_id - 1] if args.intent_targeted else None
        if target_intent:
            print(f"[Target intent] {target_intent}")

        try:
            # End-to-end: from attack start until the adversarial audio exists (includes the clean synthesis)
            attack_start = time.time()
            audio_16k, clean_audio = alif.attack(sentence_text, target_intent)
            elapsed = time.time() - attack_start
            print(f"Attack finished. Time: {elapsed:.2f}s | queries: {alif.query_count}\n")

            gt_audio = sf.read(reference_audio, dtype='float32')[0]
            save_attack_result(
                output_dir=output_base,
                method=METHOD,
                dataset=args.dataset,
                sentence_id=sentence_id,
                asr_model=asr_model,
                audio=audio_16k,
                gt_audio=gt_audio,
                gt_text=sentence_text,
                transcription=alif.transcribe(audio_16k),
                gt_transcription=alif.transcribe(gt_audio),
                elapsed_seconds=elapsed,
                queries=alif.query_count,
                generations=alif.EPOCHS,
                pop_size=alif.NUM_PARTICLES,
                seed=seed,
                target_intent=target_intent,
                clean_reference_audio=clean_audio,
                params={'variant': 'ALIF-OTA (PSO), ' + ('intent matching adaptation' if target_intent else 'untargeted adaptation'), 'particles': alif.NUM_PARTICLES, 'epochs': alif.EPOCHS},
            )
        except Exception as e:
            print(f"[{sentence_id:3d}] Attack failed, skipping: {e!r}")

    print("\n[Done]")


if __name__ == '__main__':
    main()

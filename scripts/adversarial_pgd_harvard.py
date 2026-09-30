"""
Adversarial PGD — Harvard Sentences Experiment

Loads pre-generated reference audios and runs the PGD untargeted attack
(third_party/whisper_attack via src/baselines/pgd.py) on each Harvard sentence.

Generate reference audios first (styletts2 env):
    python scripts/generate_harvard_audios.py --start 1 --end 100

Then run this script (pgd env) from project root:
    python scripts/adversarial_pgd_harvard.py --start 1 --end 100
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import shutil
import datetime
import argparse
import soundfile as sf
import torch

from src.baselines import pgd
from src.data.harvard_sentences import HARVARD_SENTENCES
from src.helper import run_seed, reference_paths, results_dir
from src.metrics import evaluate_attack, utmos_score
from src.models import ASR_MODEL_CHOICES, load_asr_model
from src.trainer.result_writer import save_attack_result

AUDIO_DIR = 'outputs'


def organize_output(output_dir, sentence_id, asr_model, asr_model_name: str, elapsed_time_seconds: float, seed: int, device: str):
    save_path = pgd.save_audio_dir(AUDIO_DIR)
    adv_src = os.path.join(save_path, f'sentence_{sentence_id:03d}_adv.wav')
    if not os.path.exists(adv_src):
        print(f"[{sentence_id:3d}] Adversarial audio not found, skipping")
        shutil.rmtree(save_path, ignore_errors=True)
        return

    audio, sr = sf.read(adv_src)
    texts, _ = asr_model.inference(torch.from_numpy(audio).float(), sample_rate=sr)
    transcription = texts[0]

    gt_text = HARVARD_SENTENCES[sentence_id - 1]
    reference_audio, _ = sf.read(reference_paths(sentence_id, AUDIO_DIR)[0])
    method_name = 'PGD'
    save_attack_result(
        output_dir=output_dir,
        sentence_id=sentence_id,
        method=method_name,
        audio=audio,
        transcription=transcription,
        gt_text=gt_text,
        elapsed=elapsed_time_seconds,
        params={
            'num_generations': pgd.NB_ITER,
            'pop_size': 1,
            'snr': pgd.SNR,
            'seed': seed,
            'attack_asr_model': 'whisper',
            'eval_asr_model': asr_model_name,
        },
        evaluation=evaluate_attack(reference_audio, audio, gt_text, transcription),
        gt_audio=reference_audio,
        gt_transcription=asr_model.inference(torch.from_numpy(reference_audio).float(), sample_rate=16000)[0][0],
        naturalness={
            'utmos_best': utmos_score(audio, device),
            'utmos_gt': utmos_score(reference_audio, device),
        },
    )

    shutil.rmtree(save_path, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description='PGD Untargeted Attack — Harvard Sentences')
    parser.add_argument('--start', type=int, default=1, help='First sentence index (1-based)')
    parser.add_argument('--end', type=int, default=100, help='Last sentence index (1-based, inclusive)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed (per sentence: seed + sentence_id)')
    parser.add_argument('--asr_model', type=str, default='whisper', choices=ASR_MODEL_CHOICES,
                        help='ASR backend used to transcribe/evaluate the generated PGD audio')
    args = parser.parse_args()

    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")

    print(f"Sentences: {args.start} → {args.end}")
    print(f"SNR: {pgd.SNR} | Iterations: {pgd.NB_ITER} | Seed: {args.seed}")
    print(f"PGD attack model: whisper | evaluation ASR model: {args.asr_model}")
    print('=' * 60)

    device = f"cuda:{args.gpu}" if (args.gpu is not None and torch.cuda.is_available()) else ("cuda:0" if torch.cuda.is_available() else "cpu")
    asr_model = load_asr_model(args.asr_model, device=device)

    # robust_speech working files (csv/, attacks/, pgd_save/) go to AUDIO_DIR; results to outputs/results/PGD/
    output_dir = AUDIO_DIR
    os.makedirs(output_dir, exist_ok=True)
    run_timestamp = f"{args.asr_model}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    results_path = results_dir('PGD', run_timestamp)
    print(f"Results: {results_path}")

    for sentence_id in range(args.start, args.end + 1):
        seed = run_seed(args.seed, sentence_id)

        # End-to-end per sentence: robust_speech runs in a subprocess, so this includes its model loading
        attack_start = time.time()
        reference_audio, _ = reference_paths(sentence_id, AUDIO_DIR)
        if not os.path.exists(reference_audio):
            print(f"[{sentence_id:3d}] Audio not found, skipping: {reference_audio}")
            continue
        pgd.write_csv(output_dir, f'sentence_{sentence_id:03d}', reference_audio, HARVARD_SENTENCES[sentence_id - 1])
        pgd.run_attack(output_dir, seed, gpu=args.gpu)
        elapsed = time.time() - attack_start

        organize_output(results_path, sentence_id, asr_model, args.asr_model, elapsed, seed, device)

    print('\n[Done]')


if __name__ == '__main__':
    main()

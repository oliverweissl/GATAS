"""
PGD baseline (third_party/whisper_attack via src/baselines/pgd.py): white-box, untargeted, Whisper-tiny.

Generate reference audios first (main env):
    python scripts/generate_reference_audios.py --dataset harvard --start 1 --end 100

Then run from project root (pgd env):
    python scripts/adversarial_pgd.py --dataset harvard --start 1 --end 100
"""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import shutil
import argparse
import soundfile as sf
import torch

from src.baselines import pgd
from src.data.datasets import DATASETS, load_sentences
from src.helper import run_seed, reference_paths, results_dir, experiment_name
from src.models import load_asr_model
from src.trainer.result_writer import save_attack_result

METHOD = 'PGD'
ASR_MODEL = 'whisper-tiny'  # the model attacked by attack_configs/whisper/pgd.yaml
WORK_DIR = os.path.join('outputs', 'pgd_workdir')  # robust_speech working files (csv/, attacks/, pgd_save/)


def main():
    parser = argparse.ArgumentParser(description='PGD baseline')
    parser.add_argument('--dataset', type=str, default='harvard', choices=DATASETS)
    parser.add_argument('--start', type=int, default=1, help='First sentence ID (1-based)')
    parser.add_argument('--end', type=int, default=100, help='Last sentence ID (1-based, inclusive)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed (per sentence: seed + sentence_id)')
    args = parser.parse_args()

    sentences = load_sentences(args.dataset)
    output_base = results_dir(METHOD, args.dataset, experiment_name(ASR_MODEL))
    print(f"Dataset: {args.dataset} | sentences {args.start} → {args.end}")
    print(f"SNR: {pgd.SNR} | Iterations: {pgd.NB_ITER} | Seed: {args.seed} | results: {output_base}")
    print('=' * 60)

    device = f"cuda:{args.gpu}" if (args.gpu is not None and torch.cuda.is_available()) else ("cuda:0" if torch.cuda.is_available() else "cpu")
    asr_model = load_asr_model(ASR_MODEL, device=device)
    os.makedirs(WORK_DIR, exist_ok=True)

    for sentence_id in range(args.start, args.end + 1):
        sentence_text = sentences[sentence_id - 1]
        reference_audio, _ = reference_paths(args.dataset, sentence_id)
        if not os.path.exists(reference_audio):
            print(f"[{sentence_id:3d}] Reference audio not found, skipping: {reference_audio}")
            continue
        seed = run_seed(args.seed, sentence_id)
        utterance_id = f'sentence_{sentence_id:03d}'

        # End-to-end per sentence: robust_speech runs in a subprocess, so this includes its model loading
        attack_start = time.time()
        pgd.write_csv(WORK_DIR, utterance_id, reference_audio, sentence_text)
        pgd.run_attack(WORK_DIR, seed, gpu=args.gpu)
        elapsed = time.time() - attack_start

        save_path = pgd.save_audio_dir(WORK_DIR)
        adv_path = os.path.join(save_path, f'{utterance_id}_adv.wav')
        if not os.path.exists(adv_path):
            print(f"[{sentence_id:3d}] Adversarial audio not found, skipping")
            shutil.rmtree(save_path, ignore_errors=True)
            continue

        audio, sr = sf.read(adv_path, dtype='float32')
        gt_audio = sf.read(reference_audio, dtype='float32')[0]
        save_attack_result(
            output_dir=output_base,
            method=METHOD,
            dataset=args.dataset,
            sentence_id=sentence_id,
            asr_model=ASR_MODEL,
            audio=audio,
            gt_audio=gt_audio,
            gt_text=sentence_text,
            transcription=asr_model.inference(torch.from_numpy(audio), sample_rate=sr)[0][0],
            gt_transcription=asr_model.inference(torch.from_numpy(gt_audio))[0][0],
            elapsed_seconds=elapsed,
            queries=pgd.NB_ITER,  # white-box: gradient iterations
            generations=pgd.NB_ITER,
            pop_size=1,
            seed=seed,
            params={'snr': pgd.SNR, 'nb_iter': pgd.NB_ITER, 'attack_config': pgd.ATTACK_CONFIG},
        )
        shutil.rmtree(save_path, ignore_errors=True)

    print('\n[Done]')


if __name__ == '__main__':
    main()

"""
Generate the shared reference (ground-truth) audio for each seed sentence with StyleTTS2 (16 kHz).
Every attack method starts from these files:
    outputs/references/<dataset>/sentence_XXX/reference.wav   audio
    outputs/references/<dataset>/sentence_XXX/reference.pt    StyleTTS2 embeddings (used by GATAS / Waveform)

Run from project root (main env):
    python scripts/generate_reference_audios.py --dataset harvard --start 1 --end 100
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import soundfile as sf
import torch

from src.data.datasets import DATASETS, load_sentences
from src.helper import set_seed, run_seed, reference_paths
from src.models._styletts2 import StyleTTS2


def main():
    parser = argparse.ArgumentParser(description='Generate reference audios via StyleTTS2')
    parser.add_argument('--dataset', type=str, default='harvard', choices=DATASETS)
    parser.add_argument('--start', type=int, default=1, help='First sentence ID (1-based)')
    parser.add_argument('--end', type=int, default=100, help='Last sentence ID (1-based, inclusive)')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed (per sentence: seed + sentence_id)')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index to use (default: 0)')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    sentences = load_sentences(args.dataset)
    print(f"Device: {device} | dataset: {args.dataset} | sentences {args.start} → {args.end}")
    print('=' * 60)

    tts_model = None
    for sentence_id in range(args.start, args.end + 1):
        sentence_text = sentences[sentence_id - 1]
        output_path, embeddings_path = reference_paths(args.dataset, sentence_id)

        if os.path.exists(output_path) and os.path.exists(embeddings_path):
            print(f"[{sentence_id:3d}] Already exists, skipping: {output_path}")
            continue

        if tts_model is None:
            print("Loading StyleTTS2...")
            tts_model = StyleTTS2(device=device)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        set_seed(run_seed(args.seed, sentence_id))
        noise = torch.randn(1, 1, 256).to(device)
        embeddings = tts_model.extract_embeddings(tts_model.preprocess_text(sentence_text), noise)
        # inference_on_embedding already resamples to 16 kHz internally
        audio_numpy = tts_model.inference_on_embedding(embeddings).flatten().cpu().detach().numpy()

        sf.write(output_path, audio_numpy, 16000)
        torch.save(embeddings, embeddings_path)

        print(f"[{sentence_id:3d}] Saved: {output_path}  |  {sentence_text}")
    print("\n[Done]")


if __name__ == '__main__':
    main()

"""
Adapter around the unmodified whisper_attack code (git submodule third_party/whisper_attack,
https://github.com/RaphaelOlivier/whisper_attack), which runs the robust_speech PGD attack on Whisper.

whisper_attack reads a SpeechBrain-style CSV (ID, duration, wav, wrd) and writes adversarial audio to
save_audio_path. It is run as a subprocess from the submodule directory in the pgd conda environment.
"""
import os
import csv
import sys
import subprocess

import soundfile as sf
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WHISPER_ATTACK_DIR = os.path.join(REPO_ROOT, "third_party", "whisper_attack")

ATTACK_CONFIG = "attack_configs/whisper/pgd.yaml"  # attacks whisper-tiny (model_label in the config)
NB_ITER = 200
SNR = 35
CSV_NAME = "harvard"


def write_csv(output_dir: str, utterance_id: str, audio_path: str, text: str) -> str:
    """Write a single-utterance CSV at <output_dir>/csv/<CSV_NAME>.csv (the path whisper_attack expects)."""
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{CSV_NAME}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "duration", "wav", "wrd"])
        writer.writerow([utterance_id, f"{sf.info(audio_path).duration:.3f}", os.path.abspath(audio_path), text.upper()])
    return csv_path


def save_audio_dir(output_dir: str) -> str:
    return os.path.join(os.path.abspath(output_dir), "pgd_save")


def run_attack(output_dir: str, seed: int, gpu: int = None) -> int:
    """Run PGD on the CSV in output_dir. Adversarial audio: <save_audio_dir(output_dir)>/<ID>_adv.wav."""
    abs_output_dir = os.path.abspath(output_dir)
    device = f"cuda:{gpu}" if (gpu is not None and torch.cuda.is_available()) else ("cuda:0" if torch.cuda.is_available() else "cpu")
    cmd = [
        sys.executable, "-W", "ignore", "run_attack.py",
        ATTACK_CONFIG,
        f"--root={abs_output_dir}",
        f"--data_folder={abs_output_dir}",
        f"--data_csv_name={CSV_NAME}",
        f"--nb_iter={NB_ITER}",
        "--load_audio=False",
        f"--seed={seed}",
        "--attack_name=pgd_harvard",
        f"--snr={SNR}",
        "--skip_prep=True",
        f"--save_audio_path={save_audio_dir(output_dir)}",
        f"--device={device}",
    ]
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print(f"[PGD] Running attack on device={device} | nb_iter={NB_ITER} | snr={SNR} | seed={seed}")
    result = subprocess.run(cmd, cwd=WHISPER_ATTACK_DIR, env=env)
    print(f"[PGD] subprocess exited with code {result.returncode}")
    return result.returncode

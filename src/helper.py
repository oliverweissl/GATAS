import os
import random

import numpy as np
import torch
import soundfile as sf


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_seed(seed: int, sentence_id: int, run_id: int = 0) -> int:
    """Seed for one (sentence, run) so results do not depend on which sentence range is executed."""
    return seed + sentence_id + 1000 * run_id


# Method names, used for result folders and the 'attack_method' field of every summary JSON
METHODS = ("GATAS", "Waveform", "SMACK", "SMACK_targeted", "PGD")


def results_dir(method: str, run_timestamp: str) -> str:
    """outputs/results/<method>/<run_timestamp>; method is one of METHODS."""
    assert method in METHODS, f"Unknown method {method!r}, expected one of {METHODS}"
    return os.path.join("outputs", "results", method, run_timestamp)


def reference_paths(sentence_id: int, audio_dir: str = "outputs") -> tuple[str, str]:
    """(wav path, embedding path) of the shared reference audio written by generate_harvard_audios.py."""
    sentence_dir = os.path.join(audio_dir, f"harvard_sentence_{sentence_id:03d}")
    return os.path.join(sentence_dir, "harvard_audio.wav"), os.path.join(sentence_dir, "harvard_audio.pt")


def load_reference_embedding(sentence_id: int, device: str, audio_dir: str = "outputs"):
    """Load the GT AudioEmbeddingData saved by generate_harvard_audios.py, or None if missing."""
    _, embeddings_path = reference_paths(sentence_id, audio_dir)
    if not os.path.exists(embeddings_path):
        return None
    embedding = torch.load(embeddings_path, map_location=device, weights_only=False)
    for field in ("input_length", "text_mask", "h_bert", "h_text", "style_vector_acoustic", "style_vector_prosodic"):
        setattr(embedding, field, getattr(embedding, field).to(device))
    return embedding


def calculate_2d_hypervolume(pareto_front, ref_point):
    """
    Calculates the area (Hypervolume) for a 2D Pareto front.
    pareto_front: np.ndarray of shape (N, 2)
    ref_point: list or array [r1, r2] (the 'worst' possible values)
    """
    if pareto_front.size == 0:
        return 0.0

    # 1. Sort the front by the first objective
    front = pareto_front[pareto_front[:, 0].argsort()]

    # 2. Ensure all points are within the reference point bounds
    # (Ignore points worse than the reference point)
    mask = (front[:, 0] <= ref_point[0]) & (front[:, 1] <= ref_point[1])
    front = front[mask]

    if len(front) == 0:
        return 0.0

    # 3. Calculate the area of the rectangles
    area = 0.0
    last_y = ref_point[1]

    for x, y in front:
        # Area = Width (distance to ref_x) * Height (distance between steps)
        area += (ref_point[0] - x) * (last_y - y)
        last_y = y

    return area

def save_audio(audio, file_path):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy().squeeze()
    sf.write(file_path, audio, samplerate=16000)

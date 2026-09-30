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
METHODS = ("GATAS", "GATAS_targeted", "Waveform", "SMACK", "SMACK_targeted", "ALIF", "PGD")

REFERENCE_DIR = os.path.join("outputs", "references")
RESULTS_DIR = os.path.join("outputs", "results")


def experiment_name(asr_model: str) -> str:
    """Name of one experiment folder: <asr_model>_<timestamp>."""
    import datetime
    return f"{asr_model}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")


def results_dir(method: str, dataset: str, experiment: str) -> str:
    """outputs/results/<method>/<dataset>/<experiment>; method is one of METHODS."""
    assert method in METHODS, f"Unknown method {method!r}, expected one of {METHODS}"
    return os.path.join(RESULTS_DIR, method, dataset, experiment)


def sentence_dir(results_path: str, sentence_id: int) -> str:
    return os.path.join(results_path, f"sentence_{sentence_id:03d}")


def reference_paths(dataset: str, sentence_id: int, reference_dir: str = REFERENCE_DIR) -> tuple[str, str]:
    """(wav path, embedding path) of the shared reference audio written by generate_reference_audios.py."""
    folder = os.path.join(reference_dir, dataset, f"sentence_{sentence_id:03d}")
    return os.path.join(folder, "reference.wav"), os.path.join(folder, "reference.pt")


def load_reference_embedding(dataset: str, sentence_id: int, device: str, reference_dir: str = REFERENCE_DIR):
    """Load the GT AudioEmbeddingData saved by generate_reference_audios.py, or None if missing."""
    _, embeddings_path = reference_paths(dataset, sentence_id, reference_dir)
    if not os.path.exists(embeddings_path):
        return None
    embedding = torch.load(embeddings_path, map_location=device, weights_only=False)
    for field in ("input_length", "text_mask", "h_bert", "h_text", "style_vector_acoustic", "style_vector_prosodic"):
        setattr(embedding, field, getattr(embedding, field).to(device))
    return embedding


def target_sentence(sentences, sentence_id: int, seed: int) -> str:
    """Target text for targeted attacks: another sentence of the same dataset, identical for all methods."""
    candidates = [s for i, s in enumerate(sentences) if i != sentence_id - 1]
    return random.Random(run_seed(seed, sentence_id)).choice(candidates)


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

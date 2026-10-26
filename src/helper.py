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
METHODS = ("GATAS", "GATAS_targeted", "Waveform", "SMACK_untargeted", "SMACK", "ALIF", "PGD",
           "GATAS_intent", "Waveform_intent", "ALIF_intent")  # *_intent: optimize the intent classifier output
# Scale ablation of GATAS (scripts/run_scale_ablation.sh, analysis/Ablation_Scale.ipynb)
ABLATION_METHODS = ("GATAS_scale_0.5", "GATAS_scale_1", "GATAS_scale_2", "GATAS_adaptive")
# Search-space / optimizer ablation of GATAS (scripts/run_space_ablation.sh, analysis/RQ0_Ablation.ipynb)
# GATAS variant: latent interpolation toward an LLM intent rewrite of the sentence (scripts/run_gatas_llm.sh)
VARIANT_METHODS = ("GATAS_llm",)
INTENT_ABLATION_METHODS = ("GATAS_intent_sel", "GATAS_intent_margin", "GATAS_intent_margin_sel")  # run_intent_ablation.sh
SPACE_METHODS = ("GATAS_dct", "GATAS_jac", "GATAS_jacdct", "GATAS_cmaes", "GATAS_mocma", "GATAS_cmaes_jacdct", "GATAS_mocma_jacdct")

REFERENCE_DIR = os.path.join("outputs", "references")
RESULTS_DIR = os.path.join("outputs", "results")


def experiment_name(asr_model: str) -> str:
    """Name of one experiment folder: <asr_model>_<timestamp>."""
    import datetime
    return f"{asr_model}_" + datetime.datetime.now().strftime("%Y%m%d_%H%M")


def results_dir(method: str, dataset: str, experiment: str) -> str:
    """outputs/results/<method>/<dataset>/<experiment>; method is one of METHODS."""
    assert method in METHODS + ABLATION_METHODS + SPACE_METHODS + INTENT_ABLATION_METHODS + VARIANT_METHODS, f"Unknown method {method!r}, expected one of {METHODS + ABLATION_METHODS + SPACE_METHODS + INTENT_ABLATION_METHODS + VARIANT_METHODS}"
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


def save_audio(audio, file_path):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy().squeeze()
    sf.write(file_path, audio, samplerate=16000)

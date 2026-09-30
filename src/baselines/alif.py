"""
Adapter around the unmodified ALIF code (git submodule third_party/TASER, https://github.com/TASER2023/TASER;
Cheng et al., "ALIF: Low-Cost Adversarial Audio Attacks on Black-Box Speech Platforms Using Linguistic Features").

ALIF perturbs the Tacotron2 encoder outputs (one linguistic-feature vector per input symbol) and searches with
particle swarm optimization (ALIF-OTA: POS2, perturbations bounded by |x| <= 0.4). Its native goal is the reverse
of ours: keep the ASR transcription correct (constraint, loss = inf otherwise) while maximizing mel distortion.

Untargeted adaptation used here (same structure, roles swapped to our goal):
    loss = SET_OVERLAP(GT, transcription)   if PESQ fitness(clean synthesis, candidate) <= PESQ threshold
           inf                              otherwise
ALIF's search space, perturbation bound, PSO optimizer and hyperparameters (pN=20 particles, 10 epochs) are kept.
Its mel-distortion knobs (alpha, beta, gamma, eta) exist to degrade audio and are left neutral.

The adapter also:
  - builds a working directory with the Tacotron2/WaveGlow checkpoints ALIF loads via cwd-relative paths,
  - stubs modules imported but unused at inference (TensorFlow for HParams, tensorboard logger, ALIF's ASR APIs),
  - routes ASR queries to our ASR models (src.models) and resamples ALIF's 22.05 kHz output to 16 kHz.

Import this module from the ALIF conda environment only (configs/setup_alif_env.sh).
"""
import os
import sys
import types

import librosa
import numpy as np
import torch

from ..metrics import SUCCESS_THRESHOLDS, pesq_fitness, set_overlap
from ..models import load_asr_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALIF_DIR = os.path.join(REPO_ROOT, "third_party", "TASER")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
WORK_DIR = os.path.join(REPO_ROOT, "outputs", "alif_workdir")

TACOTRON_SAMPLE_RATE = 22_050
SAMPLE_RATE = 16_000
WAVEGLOW_SIGMA = 0.666  # as in ALIF

# ALIF-OTA hyperparameters (defaults of alif_ota.py)
NUM_PARTICLES = 20
EPOCHS = 10

_alif = None
_optimizer = None
_tts = None
_asr_model = None
query_count = 0  # attacked-ASR queries (reset per attack)


class _HParams:
    """Minimal stand-in for tf.contrib.training.HParams (Tacotron2's hparams.py only needs these methods)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def parse(self, values: str):
        for item in filter(None, (values or "").split(",")):
            key, value = item.split("=")
            current = getattr(self, key.strip())
            setattr(self, key.strip(), type(current)(value.strip()))
        return self

    def values(self):
        return dict(self.__dict__)


def _install_stubs() -> None:
    tensorflow = types.ModuleType("tensorflow")
    tensorflow.contrib = types.SimpleNamespace(training=types.SimpleNamespace(HParams=_HParams))
    sys.modules["tensorflow"] = tensorflow

    logger = types.ModuleType("logger")  # training-only (tensorboard)
    logger.Tacotron2Logger = object
    sys.modules["logger"] = logger

    asrs = types.ModuleType("ASRs")  # ALIF's cloud ASR hooks; replaced by our models
    for name in ("get_trans_tencent", "get_trans_iflytek", "get_trans_amazon", "get_trans_google", "get_trans_azure"):
        setattr(asrs, name, lambda *_args: "")
    sys.modules["ASRs"] = asrs


def _link(target: str, link_name: str) -> None:
    if os.path.lexists(link_name):
        return
    if not os.path.exists(target):
        raise FileNotFoundError(f"ALIF requires {target} (see README: ALIF checkpoints)")
    os.symlink(target, link_name)


def _prepare_work_dir() -> None:
    """WORK_DIR/tacotron2 mirrors the submodule's tacotron2/ (symlinks) plus the two checkpoints ALIF expects there."""
    tacotron_dir = os.path.join(WORK_DIR, "tacotron2")
    os.makedirs(tacotron_dir, exist_ok=True)
    source_dir = os.path.join(ALIF_DIR, "tacotron2")
    for name in os.listdir(source_dir):
        _link(os.path.join(source_dir, name), os.path.join(tacotron_dir, name))
    _link(os.path.join(CHECKPOINT_DIR, "tacotron2_statedict.pt"), os.path.join(tacotron_dir, "tacotron2_statedict.pt"))
    _link(os.path.join(CHECKPOINT_DIR, "waveglow_256channels_universal_v5.pt"),
          os.path.join(tacotron_dir, "waveglow_256channels_universal_v5.pt"))


def setup(asr_model_name: str, device: str = "cuda") -> None:
    """Load ALIF's Tacotron2/WaveGlow (from its submodule) and our ASR model. Must be called before attack()."""
    global _alif, _optimizer, _tts, _asr_model
    if _alif is not None:
        return

    _prepare_work_dir()
    _install_stubs()
    sys.path.insert(0, ALIF_DIR)

    cwd = os.getcwd()
    os.chdir(WORK_DIR)  # ALIF adds 'tacotron2/' to sys.path and loads checkpoints relative to cwd
    try:
        import alif_ota
        import optimizer
        tts = alif_ota.TTS()
    finally:
        os.chdir(cwd)

    _alif, _optimizer, _tts = alif_ota, optimizer, tts
    _asr_model = load_asr_model(asr_model_name, device=device)


def _vocode(mel: torch.Tensor) -> np.ndarray:
    """WaveGlow synthesis as in ALIF (peak-normalized), returned as float32 in [-1, 1] at 16 kHz."""
    with torch.no_grad():
        audio = _tts.waveglow.infer(mel, sigma=WAVEGLOW_SIGMA)[0].data.cpu().numpy()
    audio = audio / max(np.max(np.abs(audio)), 1e-8) * (10000 / 32768)  # ALIF writes peak 10000 as int16
    return librosa.resample(audio.astype(np.float32), orig_sr=TACOTRON_SAMPLE_RATE, target_sr=SAMPLE_RATE)


def transcribe(audio_16k: np.ndarray) -> str:
    texts, _ = _asr_model.inference(torch.from_numpy(np.asarray(audio_16k, dtype=np.float32)).unsqueeze(0))
    return texts[0] if texts else ""


def attack(text: str) -> tuple:
    """
    Run untargeted ALIF-OTA for one sentence.
    Returns (adversarial audio, clean Tacotron2 synthesis), both float32 in [-1, 1] at 16 kHz.
    """
    global query_count
    query_count = 0

    with torch.no_grad():
        feature = _tts.get_feature(text)
        _, ori_mel, _, _ = _tts.model.inference2(feature)
    clean_audio = _vocode(ori_mel)
    pesq_threshold = SUCCESS_THRESHOLDS["PESQ"]

    num_symbols, feature_dim = feature.shape[1], feature.shape[2]
    noise = np.zeros((NUM_PARTICLES, num_symbols, feature_dim))
    noise[:, -1, :] = 0
    pos = _optimizer.POS2(NUM_PARTICLES, num_symbols, feature_dim, noise)

    best_audio, best_loss = clean_audio, np.inf
    for epoch in range(EPOCHS):
        features = np.array([feature[0].cpu().detach().numpy()] * NUM_PARTICLES) + noise
        with torch.no_grad():
            adv_mels, _ = _tts.get_mels_gates(features)

        losses = []
        for adv_mel in adv_mels:
            audio = _vocode(adv_mel)
            transcription = transcribe(audio)
            query_count += 1
            if pesq_fitness(clean_audio, audio) <= pesq_threshold:
                loss = set_overlap(text, transcription)
            else:
                loss = np.inf
            losses.append(loss)
            if loss < best_loss:
                best_loss, best_audio = loss, audio

        noise, _, _, _ = pos.update(losses)
        noise[:, -1, :] = 0
        print(f"[ALIF] epoch {epoch}: best SET_OVERLAP under PESQ constraint = {best_loss}")

    return best_audio, clean_audio

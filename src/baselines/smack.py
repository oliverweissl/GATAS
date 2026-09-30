"""
Adapter around the unmodified SMACK code (git submodule third_party/SMACK,
https://github.com/WUSTL-CSPL/SMACK).

SMACK only ships Google/iFlytek ASR and speaker-recognition targets and expects to be run from its own
directory. This adapter:
  - builds a working directory with the files SMACK loads via cwd-relative paths
    (./LJ.ckpt, ./waveglow_256channels_universal_v5.pt, ./waveglow, ./SampleDir, ./SuccessDir),
  - stubs the SMACK modules that are imported but unused here (Google/iFlytek clients, FAKEBOB speaker models),
  - routes SMACK's ASR queries (its 'googleASR' code path) to our ASR models from src.models,
  - adds an untargeted variant (SMACK itself is targeted only),
  - converts SMACK's int16-scaled 22.05 kHz synthesis output to float 16 kHz audio,
  - synthesizes a clean ETTS reference (prosody taken from the GT audio) for PESQ.

Import this module from the SMACK conda environment only.
"""
import os
import sys
import types

import librosa
import numpy as np
import soundfile as sf
import torch

from ..models import load_asr_model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMACK_DIR = os.path.join(REPO_ROOT, "third_party", "SMACK")
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
WORK_DIR = os.path.join(REPO_ROOT, "outputs", "smack_workdir")

SYNTHESIS_SAMPLE_RATE = 22_050
SAMPLE_RATE = 16_000

# SMACK routes every ASR query through this target name; the function behind it is replaced below.
_SMACK_ASR_TARGET = "googleASR"

# SMACK attack hyperparameters (defaults of SMACK's attack.py)
POPULATION_SIZE = 20
GENETIC_ITERATIONS = 10
GRADIENT_ITERATIONS = 5
GRADIENT_SIGMA = 0.1
GRADIENT_LEARNING_RATE = 0.01
GRADIENT_K = 20

_genetic = None
_gradient = None
_synthesis = None
_asr_model = None


def _link(target: str, link_name: str) -> None:
    if os.path.lexists(link_name):
        return
    if not os.path.exists(target):
        raise FileNotFoundError(f"SMACK requires {target} (see README: SMACK checkpoints)")
    os.symlink(target, link_name)


def _prepare_work_dir() -> None:
    os.makedirs(os.path.join(WORK_DIR, "SampleDir"), exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "SuccessDir"), exist_ok=True)
    _link(os.path.join(CHECKPOINT_DIR, "LJ.ckpt"), os.path.join(WORK_DIR, "LJ.ckpt"))
    _link(os.path.join(CHECKPOINT_DIR, "waveglow_256channels_universal_v5.pt"),
          os.path.join(WORK_DIR, "waveglow_256channels_universal_v5.pt"))
    _link(os.path.join(SMACK_DIR, "waveglow"), os.path.join(WORK_DIR, "waveglow"))


def _install_stubs() -> None:
    """Stub SMACK modules whose imports need credentials or FAKEBOB, which the ASR attack never calls."""
    def unused(*_args, **_kwargs):
        raise RuntimeError("This SMACK target is not supported by the GATAS adapter")

    stubs = {
        "google_ASR": ["google_ASR"],
        "iflytek_ASR": ["iflytek_ASR"],
        "speaker_sv": ["speaker_verification_gmm", "speaker_verification_iv"],
        "speaker_csi": ["gmm_ubm_csi", "iv_plda_csi"],
        "speaker_osi": ["gmm_ubm_osi", "iv_plda_osi"],
    }
    for module_name, functions in stubs.items():
        module = types.ModuleType(module_name)
        for function in functions:
            setattr(module, function, unused)
        sys.modules[module_name] = module


def load_synthesis_audio(path: str) -> np.ndarray:
    """Read a SMACK synthesis WAV (22.05 kHz) as float32 in [-1, 1] at 16 kHz."""
    audio, sample_rate = sf.read(path, dtype="float32")
    return librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)


def transcribe(audio_16k: np.ndarray) -> str:
    texts, _ = _asr_model.inference(torch.from_numpy(audio_16k).float().unsqueeze(0))
    return texts[0] if texts and texts[0] else "NA"


def _smack_asr(audio_file: str) -> str:
    """Replacement for SMACK's google_ASR(path): transcribe with our ASR model ('NA' when empty, as SMACK expects)."""
    return transcribe(load_synthesis_audio(audio_file))


def _safe_nisqa(nisqa_score):
    """NISQA crashes on zero-length synthesis (and leaves stdout redirected); score such audio as worst MOS."""
    def score(audio_file, *args, **kwargs):
        try:
            return nisqa_score(audio_file, *args, **kwargs)
        except RuntimeError:
            sys.stdout = sys.__stdout__
            return 1.0
    return score


def _write_wav_short_name(path, data, samplerate, *args, **kwargs):
    """SMACK names WAVs after the transcription; keep file names below the 255-byte filesystem limit."""
    directory, name = os.path.split(path)
    if len(name.encode()) > 200:
        name = name.encode()[:196].decode(errors="ignore") + ".wav"
    return sf.write(os.path.join(directory, name), data, samplerate, *args, **kwargs)


def setup(asr_model_name: str, device: str = "cuda") -> None:
    """Load SMACK (from its submodule) and our ASR model. Must be called before attack()."""
    global _genetic, _gradient, _synthesis, _asr_model
    if _genetic is not None:
        return

    import nltk
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)  # needed by g2p_en with recent nltk

    _prepare_work_dir()
    _install_stubs()
    sys.path.insert(0, SMACK_DIR)

    cwd = os.getcwd()
    os.chdir(WORK_DIR)  # SMACK loads checkpoints relative to cwd at import time
    try:
        import genetic
        import gradient
        import synthesis
    finally:
        os.chdir(cwd)

    for module in (genetic, gradient):
        module.google_ASR = _smack_asr
        module.NISQA_score = _safe_nisqa(module.NISQA_score)
        module.sf = types.SimpleNamespace(write=_write_wav_short_name)

    _genetic, _gradient, _synthesis = genetic, gradient, synthesis
    _asr_model = load_asr_model(asr_model_name, device=device)


def _untargeted_classes():
    """
    Untargeted variant of SMACK's ASR attack. SMACK's text terms reward similarity to the target;
    here the reference (GT) text is the comparison and the text terms are negated, so the attack rewards
    distance from the GT. 'NA' (no transcription) keeps SMACK's worst-case values. Everything else is SMACK.
    """
    genetic, gradient = _genetic, _gradient

    def text_score(transcription, reference_text):
        if transcription == "NA":
            return 100, 0, 10000
        lev = genetic.levenshteinDistance(transcription, reference_text) / ((len(transcription) + len(reference_text)) / 2)
        return lev, genetic.CMU_similarity(transcription, reference_text), genetic.ALINE_dissimilarity(transcription, reference_text)

    class UntargetedGeneticAlgorithm(genetic.GeneticAlgorithm):
        def _calculate_fitness(self):
            population_fitness = []
            for individual, _ in self.population:
                genetic.audio_synthesis(individual.reshape(-1, 32), self.reference_audio, self.reference_text)
                tmp_audio_file = './SampleDir/synthesis.wav'
                audio_quality = genetic.NISQA_score(tmp_audio_file)
                transcription = genetic.google_ASR(tmp_audio_file)

                lev, cmu, aline = text_score(transcription, self.reference_text)
                # SMACK (targeted, maximized): -10*lev + 0.1*cmu - 0.0001*aline + 0.05*quality
                text_fitness = -10 * lev + 0.1 * cmu - 0.0001 * aline
                if transcription != "NA":
                    text_fitness = -text_fitness
                population_fitness.append(text_fitness + 0.05 * audio_quality)
            return population_fitness

    class UntargetedGradientEstimation(gradient.GradientEstimation):
        def _calculate_loss(self, p_i):
            gradient.audio_synthesis(p_i.reshape(-1, 32), self.reference_audio, self.reference_text)
            tmp_audio_file = './SampleDir/synthesis.wav'
            audio_quality = gradient.NISQA_score(tmp_audio_file)
            transcription = gradient.google_ASR(tmp_audio_file)

            lev, cmu, aline = text_score(transcription, self.reference_text)
            # SMACK (targeted): 10*lev - 0.1*cmu + 0.0001*aline - 0.05*quality
            text_loss = 10 * lev - 0.1 * cmu + 0.0001 * aline
            if transcription != "NA":
                text_loss = -text_loss
            return text_loss - 0.05 * audio_quality

    return UntargetedGeneticAlgorithm, UntargetedGradientEstimation


def _clear_sample_dir() -> None:
    """SMACK writes one WAV per query into SampleDir; drop them between sentences."""
    sample_dir = os.path.join(WORK_DIR, "SampleDir")
    for name in os.listdir(sample_dir):
        if name.endswith(".wav"):
            os.remove(os.path.join(sample_dir, name))


def attack(reference_audio: str, reference_text: str, target_text: str = None) -> np.ndarray:
    """
    Run SMACK (genetic algorithm + gradient estimation, as in SMACK's attack.py) for one sentence.

    reference_audio: 16 kHz WAV of the ground truth. target_text: None for the untargeted variant.
    Returns the adversarial audio as float32 in [-1, 1] at 16 kHz.
    """
    reference_audio = os.path.abspath(reference_audio)
    if target_text is None:
        genetic_cls, gradient_cls = _untargeted_classes()
    else:
        genetic_cls, gradient_cls = _genetic.GeneticAlgorithm, _gradient.GradientEstimation

    cwd = os.getcwd()
    os.chdir(WORK_DIR)
    try:
        _clear_sample_dir()
        ga = genetic_cls(reference_audio, reference_text, _SMACK_ASR_TARGET, target_text, POPULATION_SIZE)
        fittest_individual = ga.run(GENETIC_ITERATIONS)

        gradient_estimator = gradient_cls(
            reference_audio, reference_text, _SMACK_ASR_TARGET, target_text, None,
            sigma=GRADIENT_SIGMA, learning_rate=GRADIENT_LEARNING_RATE, K=GRADIENT_K,
        )
        p_refined = gradient_estimator.refine_prosody_vector(fittest_individual, GRADIENT_ITERATIONS)

        # SMACK's synthesis returns int16-scaled samples at 22.05 kHz
        audio_int16 = _synthesis.audio_synthesis(p_refined.reshape(-1, 32), reference_audio, reference_text)
    finally:
        os.chdir(cwd)

    audio_float = audio_int16.astype(np.float32) / 32768.0
    return librosa.resample(audio_float, orig_sr=SYNTHESIS_SAMPLE_RATE, target_sr=SAMPLE_RATE)


def clean_reference(reference_audio: str, reference_text: str) -> np.ndarray:
    """
    Benign SMACK synthesis of the sentence: ETTS with global and local emotion extracted from the GT audio
    (SMACK's synthesize_with_ref), i.e. the same TTS/vocoder as the attack but without a perturbed prosody
    vector. Used as the PESQ reference, since the StyleTTS2 GT is a different voice and not time-aligned.
    Returns float32 in [-1, 1] at 16 kHz.
    """
    gt_audio, sample_rate = librosa.load(os.path.abspath(reference_audio), sr=None)
    assert sample_rate == SAMPLE_RATE  # SMACK's emotion model expects 16 kHz reference audio

    cwd = os.getcwd()
    os.chdir(WORK_DIR)
    try:
        _synthesis.model_syn.synthesize_with_ref(reference_text, gt_audio, gt_audio, "clean_reference.wav", wav_input=True)
    finally:
        os.chdir(cwd)
    return load_synthesis_audio(os.path.join(WORK_DIR, "SampleDir", "clean_reference.wav"))

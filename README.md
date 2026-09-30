# GATAS — Setup & Reproduction Guide

GATAS attacks ASR systems by optimizing the text embedding of a TTS model (StyleTTS2) with a
multi-objective genetic algorithm (NSGA-II, pymoo): the generated speech should stay natural (PESQ)
while the ASR transcription loses the original content (SET_OVERLAP). It is compared against a
waveform-space baseline and two external baselines, SMACK and PGD.

## Repository layout

| Path | Content |
|---|---|
| `src/` | GATAS: StyleTTS2 wrapper (`models/`, `tts_core/`), ASR wrappers (`models/`), objectives, optimizer, trainers, logging |
| `src/metrics.py` | Shared evaluation metrics and success criterion for all methods |
| `src/baselines/` | Adapters for the external baselines (`smack.py`, `pgd.py`) |
| `third_party/` | Unmodified external baselines as git submodules |
| `scripts/` | Experiment entry points (`adversarial_*_harvard.py`) and run scripts (`run_*_harvard.sh`) |
| `configs/` | StyleTTS2 config, conda environments for SMACK and PGD |
| `analysis/` | Notebooks for RQ1 (effectiveness), RQ2 (validity / survey), RQ3 (efficiency), qualitative analysis |

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
# or, in an existing clone:
git submodule update --init
```

Three Python environments are used:

| Environment | Used for | Setup |
|---|---|---|
| main (venv/conda) | GATAS, waveform baseline, reference audio generation | `pip install -r requirements.txt` and `sudo apt-get install espeak-ng` |
| `smack` | SMACK | `conda env create -f configs/smack.yml` |
| `pgd` | PGD | `cd configs && bash setup_pgd_env.sh` |

The external baselines are unmodified upstream code, called through adapters:

| Baseline | Submodule | Upstream | Adapter |
|---|---|---|---|
| SMACK | `third_party/SMACK` | [WUSTL-CSPL/SMACK](https://github.com/WUSTL-CSPL/SMACK) | `src/baselines/smack.py` |
| PGD | `third_party/whisper_attack` | [RaphaelOlivier/whisper_attack](https://github.com/RaphaelOlivier/whisper_attack) | `src/baselines/pgd.py` |

The SMACK adapter routes SMACK's ASR queries to the same ASR models GATAS uses (`src/models`), so no
Google/iFlytek credentials are needed, and adds an untargeted variant (SMACK itself is targeted only).
Known issues in the upstream SMACK code are kept as is.

## Checkpoints

All checkpoints go into `checkpoints/` (git-ignored).

**StyleTTS2** ([LJSpeech model](https://huggingface.co/yl4579/StyleTTS2-LJSpeech)):

```bash
wget -O checkpoints/STT2.pth https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth
```

StyleTTS2 also needs its text aligner and pitch extractor from the [StyleTTS2 repository](https://github.com/yl4579/StyleTTS2/tree/main/Utils):
`Utils/ASR/epoch_00080.pth` as `checkpoints/asr.pth` and `Utils/JDC/bst.t7` as `checkpoints/bst.t7`
(paths set in `configs/style_tts2_config.yml`). PL-BERT is included in `src/tts_core/pretrained/plbert`.

**SMACK:** `LJ.ckpt` and `waveglow_256channels_universal_v5.pt` (download links in the
[SMACK README](https://github.com/WUSTL-CSPL/SMACK?tab=readme-ov-file#installation)). SMACK downloads
`facebook/wav2vec2-base` from the Hugging Face Hub on first use.

ASR models (Whisper tiny, `facebook/wav2vec2-base-960h`, `speechbrain/asr-transformer-transformerlm-librispeech`)
are downloaded automatically.

## Running the experiments

Run everything from the project root. Every run script processes Harvard sentences 1–100 in one sequential
process and first generates the shared reference audios (`outputs/harvard_sentence_XXX/harvard_audio.{wav,pt}`,
StyleTTS2, 16 kHz), so all methods attack the same ground-truth audio. Existing reference audios are reused;
delete `outputs/` for a fresh run.

Environment variables for all run scripts: `ASR_MODEL` (`whisper` | `wav2vec2` | `speechbrain`, default `whisper`),
`SEED` (default `0`), `GPU` (default `0`).

All results go to `outputs/results/<Method>/<asr>_<timestamp>/sentence_XXX/`:

| Method | Command | `<Method>` | Files per sentence |
|---|---|---|---|
| GATAS | `bash scripts/run_gatas_harvard.sh` | `GATAS` | `run_0/`: `run_summary.json`, `best_mixed.wav`, `ground_truth.wav`, `fitness_history.csv`, plots |
| Waveform baseline | `bash scripts/run_waveform_harvard.sh` | `Waveform` | same as GATAS |
| SMACK (untargeted) | `bash scripts/run_smack_harvard_untargeted.sh` | `SMACK` | `smack_summary.json`, `best_smack.wav`, `ground_truth.wav`, `smack_clean_reference.wav` |
| SMACK (targeted) | `bash scripts/run_smack_harvard.sh` | `SMACK_targeted` | same as SMACK |
| PGD | `bash scripts/run_pgd_harvard.sh` | `PGD` | `pgd_summary.json`, `best_pgd.wav`, `ground_truth.wav` |

GATAS and the waveform baseline also write `all_results.{json,csv}` per experiment. Every summary JSON contains
`attack_method` (one of `GATAS`, `Waveform`, `SMACK`, `SMACK_targeted`, `PGD`), `success_metrics` (shared criterion)
and `naturalness_scores` (UTMOS).

GATAS settings used in the run script: NSGA-II with population 100 for up to 100 generations,
objectives `PESQ=0.2, SET_OVERLAP=0.5` (the values are early-stopping thresholds), mode `NOISE_UNTARGETED`
(interpolation of the StyleTTS2 text embedding between the GT and a noise embedding), and `--seed_target`
(one initial individual is the pure noise target). See `python scripts/adversarial_gatas_harvard.py --help` for all options.

PGD attacks Whisper tiny (200 iterations, SNR 35 dB) and is evaluated with `ASR_MODEL`.

The analysis notebooks load results through `analysis/results_loader.py` (`load_results(asr_model="whisper")`):
it takes the latest experiment per method for the given ASR model, normalizes the three summary formats, and keeps
only sentences present for every method, sorted by sentence ID, so paired statistics compare the same sentences.
Success is always read from `success_metrics`.

## Evaluation

All methods report the same metrics and success criterion, defined in `src/metrics.py`:

- **PESQ fitness**: wideband PESQ mapped to [0, 1] (0 = perfect quality, 1 = worst).
- **SET_OVERLAP**: fraction of ground-truth content words (lowercased, stopwords removed, WordNet-lemmatized)
  that survive in the ASR transcription (0 = none left).
- **Success**: every metric at or below `SUCCESS_THRESHOLDS` (default PESQ ≤ 0.2 and SET_OVERLAP ≤ 0.5).
  Change the thresholds there to change success for all methods.

PESQ needs a time-aligned reference in the same voice. GATAS, the waveform baseline and PGD use the StyleTTS2
ground truth. SMACK re-synthesizes speech with its own TTS (ETTS + WaveGlow), so its PESQ reference is SMACK's
clean synthesis of the sentence (prosody taken from the ground-truth audio, no adversarial perturbation).

Naturalness is reported as **UTMOS** (predicted MOS in [1, 5], `src/metrics.py`,
[balacoon/utmos](https://huggingface.co/balacoon/utmos), downloaded on first use) for the adversarial and the
ground-truth audio of every method. It is not part of the success criterion.

Runtimes are end-to-end wall clock per sentence, from attack start until the adversarial audio exists
(one-time model loading excluded). PGD runs whisper_attack in a subprocess per sentence, so its runtime
includes loading the Whisper model.

## Reproducibility

Every method takes `--seed`. Each sentence is seeded with `seed + sentence_id` (plus `1000 * run_id` for
repeated GATAS runs), so results do not depend on which sentence range is run. Reference audios use the
same per-sentence seed.

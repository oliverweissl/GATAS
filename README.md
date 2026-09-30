# GATAS — Setup & Reproduction Guide

GATAS is a black-box test generator for ASR systems. It perturbs the phoneme-level text embedding of a
TTS model (StyleTTS2) with a multi-objective genetic algorithm (NSGA-II, pymoo): the generated speech should
stay natural (PESQ) while the ASR transcription loses the original content (SET_OVERLAP). It is compared
against a waveform-space ablation and three external baselines: SMACK, ALIF and PGD.

## Repository layout

| Path | Content |
|---|---|
| `src/` | GATAS: StyleTTS2 wrapper (`models/`, `tts_core/`), ASR wrappers (`models/`), objectives, optimizer, trainers, logging |
| `src/data/` | Seed-sentence datasets (`datasets.py`, `seed_sentences/`) |
| `src/metrics.py` | Shared metrics, success criteria and validity check for all methods |
| `src/evaluation.py` | Uniform evaluation of saved results (used by `scripts/evaluate_results.py`) |
| `src/baselines/` | Adapters for the external baselines (`smack.py`, `alif.py`, `pgd.py`) |
| `third_party/` | Unmodified external baselines as git submodules |
| `scripts/` | Attack entry points (`adversarial_<method>.py`), reference generation, evaluation, run scripts (`run_*.sh`) |
| `configs/` | StyleTTS2 config, conda environments for SMACK, ALIF and PGD |
| `analysis/` | Notebooks: RQ1 (effectiveness), RQ2 (validity / survey), RQ3 (efficiency), qualitative analysis, revision analyses |

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
# or, in an existing clone:
git submodule update --init
```

Four Python environments are used:

| Environment | Used for | Setup |
|---|---|---|
| main (venv/conda) | GATAS, waveform baseline, reference audios, evaluation, notebooks | `pip install -r requirements.txt` and `sudo apt-get install espeak-ng` |
| `smack` | SMACK | `conda env create -f configs/smack.yml` |
| `alif` | ALIF | `bash configs/setup_alif_env.sh` (clones `smack`, adds `unidecode`) |
| `pgd` | PGD | `cd configs && bash setup_pgd_env.sh` |

The external baselines are unmodified upstream code, called through adapters:

| Baseline | Submodule | Upstream | Adapter |
|---|---|---|---|
| SMACK | `third_party/SMACK` | [WUSTL-CSPL/SMACK](https://github.com/WUSTL-CSPL/SMACK) | `src/baselines/smack.py` |
| ALIF | `third_party/TASER` | [TASER2023/TASER](https://github.com/TASER2023/TASER) | `src/baselines/alif.py` |
| PGD | `third_party/whisper_attack` | [RaphaelOlivier/whisper_attack](https://github.com/RaphaelOlivier/whisper_attack) | `src/baselines/pgd.py` |

- **SMACK** perturbs the prosody latent of an emotional TTS (ETTS + WaveGlow). The adapter routes its ASR queries to
  our ASR models (no Google/iFlytek credentials needed) and adds an untargeted variant, since SMACK is targeted only.
  Known issues in the upstream code are kept as is.
- **ALIF** perturbs Tacotron2 encoder outputs (linguistic features) with particle swarm optimization (ALIF-OTA).
  Its native goal is the reverse of ours (keep the transcription, maximize distortion). The adapter keeps ALIF's
  search space, perturbation bound, optimizer and hyperparameters and swaps objective and constraint:
  minimize SET_OVERLAP subject to PESQ fitness ≤ 0.2. ALIF's audio-degrading mel knobs are left neutral.
- **PGD** is a white-box attack on Whisper-tiny and serves as a reference upper bound.

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

**ALIF:** `tacotron2_statedict.pt` (NVIDIA Tacotron2 LJSpeech model, link in the
[Tacotron2 README](https://github.com/NVIDIA/tacotron2#inference-demo)) and the same `waveglow_256channels_universal_v5.pt`.

ASR models and UTMOS are downloaded automatically.

## Seed datasets and ASR models

Sentences are only used as text: every method synthesizes the reference speech with StyleTTS2 (`generate_reference_audios.py`).

| Dataset | Content | Source |
|---|---|---|
| `harvard` | Harvard sentences (phonetically balanced), IDs 1–100 used | `src/data/harvard_sentences.py` |
| `librispeech` | 100 LibriSpeech test-clean transcripts, 8–20 words | `src/data/seed_sentences/librispeech.txt` |
| `commands` | 100 SLURP voice-assistant commands, ≥ 4 words | `src/data/seed_sentences/commands.txt` |

`scripts/prepare_datasets.py` documents and reproduces the selection of the two text lists.

| ASR model (`ASR_MODEL`) | Checkpoint |
|---|---|
| `whisper-tiny` | OpenAI Whisper tiny |
| `whisper-large-v3-turbo` | OpenAI Whisper large-v3-turbo |
| `wav2vec2-large` | `facebook/wav2vec2-large-960h-lv60-self` (CTC) |
| `whisper-large-v3` | reference ASR of the validity check only (never attacked) |

## Running the experiments

Run everything from the project root. `bash scripts/run_experiments.sh` runs the full experiment matrix sequentially:

| Study | Methods | ASR | Datasets |
|---|---|---|---|
| Main comparison (RQ1–RQ3) | GATAS, Waveform, SMACK, ALIF, PGD | whisper-tiny | harvard, librispeech |
| Objective formulation | SMACK (native targeted), GATAS (targeted) | whisper-tiny | harvard |
| Generalization | GATAS, Waveform | whisper-tiny, whisper-large-v3-turbo, wav2vec2-large | harvard, librispeech, commands |
| Budget ablation (experimental) | GATAS 20x10, 20x15 (`GATAS_budget300`), 100x100 (`GATAS_budget10000`) | whisper-tiny | harvard |

Each method also has its own run script. Every run script generates the shared reference audios for its
sentences, runs the attack in one sequential process, and evaluates the results. Settings come from
environment variables (defaults in `scripts/common.sh`): `DATASET`, `ASR_MODEL`, `SEED` (0), `GPU` (0), `START`/`END` (1/100).

```bash
DATASET=librispeech ASR_MODEL=whisper-large-v3-turbo bash scripts/run_gatas.sh
```

| Method | Run script | Environment |
|---|---|---|
| `GATAS` | `run_gatas.sh` | main |
| `GATAS_targeted` | `run_gatas_targeted.sh` | main |
| `GATAS_budget300`, `GATAS_budget10000` | `run_gatas_budget_ablation.sh` | main |
| `Waveform` | `run_waveform.sh` | main |
| `SMACK` (untargeted adaptation) | `run_smack.sh` | smack |
| `SMACK_targeted` (native) | `run_smack_targeted.sh` | smack |
| `ALIF` | `run_alif.sh` | alif |
| `PGD` | `run_pgd.sh` (Whisper-tiny only) | pgd |

GATAS settings: NSGA-II with population 20 for 10 generations, i.e. 200 ASR queries per sentence (the budget of ALIF;
SMACK uses 300). Early stopping is disabled (`--min_generations` = generations); the first generation meeting the
thresholds is logged as `generation_found`. Budget and population are set with `POP_SIZE` / `NUM_GENERATIONS`
(same for Waveform). Objectives `PESQ=0.2, SET_OVERLAP=0.5` (the values are the thresholds used to select the final
candidate from the Pareto front), mode `NOISE_UNTARGETED` (interpolation of the StyleTTS2 text embedding
between the GT and a noise embedding), and `--seed_target` (one initial individual is the pure noise target). The
targeted variant uses `PESQ=0.2, WER_TARGET=0.0` in mode `TARGETED`; targets are drawn per sentence from the same
dataset, identically for GATAS and SMACK (`src/helper.target_sentence`). See `python scripts/adversarial_gatas.py --help`.

Existing reference audios are reused; delete `outputs/` for a fresh run.

### Output layout

```
outputs/references/<dataset>/sentence_XXX/reference.{wav,pt}
outputs/results/<Method>/<dataset>/<asr>_<timestamp>/sentence_XXX/[run_0/]
```

Per sentence: the adversarial audio `best_*.wav`, `ground_truth.wav`, a summary (`run_summary.json` for GATAS /
Waveform, `<method>_summary.json` for the baselines), `clean_reference.wav` for SMACK / ALIF, and `evaluation.json`.
GATAS / Waveform also store the final Pareto front with transcriptions and `archive_history.json` (archive after
every generation).

## Evaluation

`python scripts/evaluate_results.py` (main env; the run scripts call it) evaluates every saved result the same way
and writes `evaluation.json`. All definitions are in `src/metrics.py`:

- **SET_OVERLAP**: fraction of ground-truth content words (stopwords removed, WordNet-lemmatized) left in the
  transcription (bag-of-words; 0 = none left). Used as optimization objective.
- **WER** (order-sensitive) and **SBERT similarity** (meaning-level) between ground truth and transcription.
- **PESQ fitness**: wideband PESQ mapped to [0, 1] (0 = perfect). PESQ needs a time-aligned reference in the same
  voice: GATAS, Waveform and PGD use the StyleTTS2 ground truth; SMACK and ALIF re-synthesize with their own TTS and
  use their own clean synthesis of the sentence as reference. PESQ is also a GATAS objective.
- **UTMOS**: predicted MOS in [1, 5] ([balacoon/utmos](https://huggingface.co/balacoon/utmos)), no reference; the
  naturalness metric not optimized by any method.
- **Success**: untargeted `PESQ ≤ 0.2 and SET_OVERLAP ≤ 0.5` (`SUCCESS_THRESHOLDS`); targeted `PESQ ≤ 0.2 and
  TARGET_WER = 0` (`TARGETED_SUCCESS_THRESHOLDS`).
- **Validity**: the reference ASR (Whisper large-v3, not attacked) still recognizes the ground truth in the
  adversarial audio, `WER ≤ 0.3` (`VALIDITY_MAX_WER`). **Valid success** = success and valid.

Runtimes are end-to-end wall clock per sentence, from attack start until the adversarial audio exists (one-time
model loading excluded). PGD runs whisper_attack in a subprocess per sentence, so its runtime includes loading the
Whisper model. Query budgets count attacked-ASR queries (PGD: gradient iterations).

## Analysis

The notebooks load results through `analysis/results_loader.py` (`load_results(dataset=..., asr_model=...)`): the
latest experiment per method, only sentences present for every method, sorted by sentence ID (for paired tests).

- `RQ1_Effectiveness`: success, semantic divergence (SET_OVERLAP, WER, SBERT), naturalness (UTMOS), main comparison per
  dataset, threshold sensitivity (from the stored Pareto fronts), native vs. adapted SMACK, generalization matrix.
- `RQ2_Validity`: human study (survey) and the automatic validity check (reference ASR).
- `RQ3_Efficiency`: runtime, query budget, convergence, success vs. query budget (from `archive_history.json`),
  budget ablation.
- `QualAnalysis`: qualitative examples.

## Reproducibility

Every method takes `--seed`. Each sentence is seeded with `seed + sentence_id` (plus `1000 * run_id` for repeated
GATAS runs), so results do not depend on which sentence range is run. Reference audios use the same per-sentence seed.

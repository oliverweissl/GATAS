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
| `src/metrics.py` | Shared metrics and success criterion for all methods |
| `src/evaluation.py` | Uniform evaluation of saved results (used by `scripts/evaluate_results.py`) |
| `src/baselines/` | Adapters for the external baselines (`smack.py`, `alif.py`, `pgd.py`) |
| `third_party/` | Unmodified external baselines as git submodules |
| `scripts/` | Attack entry points (`adversarial_<method>.py`), reference generation, evaluation, run scripts (`run_*.sh`) |
| `configs/` | StyleTTS2 config, conda environments for SMACK, ALIF and PGD |
| `analysis/` | Notebooks, one per research question: RQ0 (ablation), RQ1 (effectiveness), RQ2 (efficiency), RQ3 (validity / survey) |

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
  search space, perturbation bound (its quality control), optimizer and hyperparameters and swaps the objective:
  minimize SET_OVERLAP. ALIF's audio-degrading mel knobs are left neutral. Unlike GATAS, ALIF's search has no PESQ
  constraint: its Tacotron2/WaveGlow synthesis is stochastic (dropout stays on, WaveGlow noise), so two syntheses of
  the same text already have PESQ fitness ≈ 0.69 and a bound of 0.2 is never met. PESQ is only measured in the evaluation.
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

**ALIF:** `tacotron2_statedict.pt` (NVIDIA Tacotron2 LJSpeech model) and the same `waveglow_256channels_universal_v5.pt`.
The Google Drive link of the [Tacotron2 README](https://github.com/NVIDIA/tacotron2#inference-demo) requires a Google
login. Instead, convert the checkpoint of the [NVIDIA torch hub model](https://pytorch.org/hub/nvidia_deeplearningexamples_tacotron2/)
(same LJSpeech Tacotron2, its keys match ALIF's model exactly):

```bash
wget -O /tmp/nvidia_tacotron2.pt https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427
python -c "import torch; sd = torch.load('/tmp/nvidia_tacotron2.pt', map_location='cpu', weights_only=True)['state_dict']; \
torch.save({'state_dict': {k.removeprefix('module.'): v for k, v in sd.items()}}, 'checkpoints/tacotron2_statedict.pt')"
```

ASR models and UTMOS are downloaded automatically.

## Seed datasets and ASR models

Sentences are only used as text: every method synthesizes the reference speech with StyleTTS2 (`generate_reference_audios.py`).

| Dataset | Content | Source |
|---|---|---|
| `harvard` | Harvard sentences (phonetically balanced), IDs 1–100 used | `src/data/harvard_sentences.py` |
| `commands` | 100 SLURP voice-assistant commands, ≥ 4 words, with reference intents (intent matching task) | `src/data/seed_sentences/commands.txt`, `commands.intents.txt` |

`scripts/prepare_datasets.py` documents and reproduces the selection of the `commands` list.

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
| Main comparison (RQ1–RQ3) | GATAS, Waveform, SMACK, ALIF, PGD | whisper-tiny | harvard (ASR), commands (intent matching) |
| Objective formulation | SMACK (native targeted), GATAS (targeted) | whisper-tiny | harvard |
| Intent matching | GATAS, Waveform, SMACK, ALIF toward the target intent (PGD untargeted, block 1) | whisper-tiny | commands |

Each method also has its own run script. Every run script generates the shared reference audios for its
sentences, runs the attack in one sequential process, and evaluates the results. Settings come from
environment variables (defaults in `scripts/common.sh`): `DATASET`, `ASR_MODEL`, `SEED` (0), `GPU` (0), `START`/`END` (1/100).

```bash
DATASET=commands ASR_MODEL=whisper-large-v3-turbo bash scripts/run_gatas.sh
```

| Method | Run script | Environment |
|---|---|---|
| `GATAS` | `run_gatas.sh` | main |
| `GATAS_targeted` | `run_gatas_targeted.sh` | main |
| `Waveform` | `run_waveform.sh` | main |
| `SMACK` (native targeted, as in the SMACK repository; on `commands` the target text is the intent rewrite) | `run_smack.sh` | smack |
| `SMACK_untargeted` (our untargeted adaptation, ASR task only) | `run_smack_untargeted.sh` | smack |
| `ALIF` | `run_alif.sh` | alif |
| `GATAS_intent`, `Waveform_intent`, `ALIF_intent` (intent matching, `commands`) | `run_gatas_intent.sh`, `run_waveform_intent.sh`, `run_alif_intent.sh` | main / main / alif |
| `PGD` | `run_pgd.sh` (Whisper-tiny only) | pgd |

GATAS settings: CMA-ES with population 20 for up to 15 generations, i.e. a budget of 300 ASR queries per sentence, the
same for all methods (Waveform 20 x 15, ALIF 20 particles x 15 epochs, PGD 300 iterations, SMACK upstream settings,
about 300). There is no early stopping: every method uses the full budget. The first generation meeting the thresholds is logged as `generation_found`. Budget and population are set with `POP_SIZE` / `NUM_GENERATIONS`
(same for Waveform). Objectives `PESQ=0.2, SET_OVERLAP=0.5` (the values are the thresholds used to select the final
candidate from the Pareto front, among the candidates whose transcription differs from the ground truth text, i.e.
that already are test cases), mode `NOISE_UNTARGETED` (interpolation of the StyleTTS2 text embedding
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

The attacked ASR model is re-run on the saved original audio (`ground_truth.wav`) and the final candidate
(`best_*.wav`); the attack-time transcription is kept as `attack_transcription`. `original_passes` is the control
(the original audio is transcribed correctly).

- **Test case found**: the transcription differs from the ground truth text, `WER(gt_text, transcription) > 0`.
- **Lexical distance**: WER between ground truth text and transcription.
- **Semantic distance**: 1 − cosine similarity of SBERT embeddings (all-MiniLM-L6-v2) of ground truth text and
  transcription.
- **Spectrogram distance**: Frobenius norm of the mel spectrogram (dB) difference between the original audio and the final candidate, divided by the spectrogram size.

- **Naturalness**: UTMOS of the final candidate (UTMOS of the original audio stored as reference). PESQ and SET_OVERLAP are
  optimization objectives of GATAS / Waveform and therefore not used for evaluation.

Runtimes are end-to-end wall clock per sentence, from attack start until the adversarial audio exists (one-time
model loading excluded). PGD runs whisper_attack in a subprocess per sentence, so its runtime includes loading the
Whisper model. Query budgets count attacked-ASR queries (PGD: gradient iterations).

## Analysis

The notebooks load results through `analysis/results_loader.py` (`load_results(dataset=..., asr_model=...)`): the
latest experiment per method, only sentences present for every method, sorted by sentence ID (for paired tests).

- `RQ0_Ablation`: scale, search space, optimizer and selection ablations.
- `RQ2_Efficiency`: runtime, query budget, queries per m-valid success.
- `RQ1_Effectiveness`: test case found, lexical / semantic / spectrogram distance, UTMOS, paired tests, trade-off and spectrogram figures.
- `RQ3_Validity`: m-valid share and the human study (survey responses): data exclusions, realism vs. UTMOS / m-validity, attack success as perceived by humans.

## Intent matching

The `commands` dataset adds an intent matching task. The system under test is the ASR model followed by the
pretrained text classifier [`cartesinus/slurp-intent_baseline-xlm_r-en`](https://huggingface.co/cartesinus/slurp-intent_baseline-xlm_r-en)
(`src/intent.py`, downloaded automatically, no training). Every sentence has a *reference intent* (classifier
prediction on the clean text, `commands.intents.txt`) and a *target intent* (best other intent of a different scenario,
`commands.target_intents.txt`; SLURP gold labels are in `commands.gold_intents.txt`).

The intent output is used as optimization signal by the methods that can use a score: `GATAS_intent` and
`Waveform_intent` (objective `INTENT_TARGET = 1 - P(target intent)` instead of SET_OVERLAP), `ALIF_intent` (PSO loss)
(PSO loss); run scripts `run_<method>_intent.sh` with `DATASET=commands`. SMACK is not modified: its native targeted attack gets the LLM intent rewrite (`commands.intent_rewrites.txt`, also the target of `GATAS_llm`) as target text. PGD stays untargeted.
`evaluation.json` gets an `intent` block for every method on `commands`: `target_match` (predicted intent equals the
target intent), `changed` (differs from the intent of the original audio's transcription), `original_correct`
(control) and `targeted` (the method optimized toward the target).

## Reproducibility

Every method takes `--seed`. Each sentence is seeded with `seed + sentence_id` (plus `1000 * run_id` for repeated
GATAS runs), so results do not depend on which sentence range is run. Reference audios use the same per-sentence seed.

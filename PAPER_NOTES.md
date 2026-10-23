# Notes for the paper

Things to mention that are easy to forget. Append new items below.

## ALIF: no PESQ constraint in the search

- ALIF's quality control is its perturbation bound (|x| <= 0.4, kept as is). Its native loss is mel MSE + gate loss, not PESQ.
- Our GATAS objective uses a PESQ constraint (fitness <= 0.2). Applying it to ALIF fails: ALIF's Tacotron2 stays in
  train mode (dropout active) and WaveGlow samples with sigma 0.666, so synthesis is stochastic.
  Measured on harvard sentence 1 (whisper-tiny):
  - clean synthesis vs. a second clean synthesis of the same text: PESQ fitness 0.69
  - same mel vocoded twice: 0.35
  - zero-noise particle vs. clean synthesis: 0.69
  A bound of 0.2 is therefore never met, every particle gets `inf`, and PSO has no signal.
- Decision: keep ALIF as conceptualized. Search loss = SET_OVERLAP only (`src/baselines/alif.py`).
  PESQ and UTMOS are measured in the common evaluation.
- Suggested wording: "ALIF controls quality through its perturbation bound, and its Tacotron2/WaveGlow pipeline
  synthesizes stochastically, so even repeated syntheses of the same text have PESQ fitness of about 0.69. A PESQ
  constraint as used by GATAS is therefore not applicable to ALIF's search. We keep ALIF's own bound and report PESQ
  and UTMOS in the common evaluation."
- Limitation to state: ALIF's search is not PESQ-constrained, so PESQ/UTMOS comparisons favor GATAS.
- TODO check: ALIF's PESQ in `evaluate_results.py` is computed against `clean_reference.wav` (one random clean
  synthesis), so it carries the same ~0.69 noise floor. Decide how to report it.

## ALIF: Tacotron2 checkpoint

- The Google Drive `tacotron2_statedict.pt` needs a Google login. We use the checkpoint of the NVIDIA torch hub model
  (`nvidia_tacotron2pyt_fp16_20190427`, LJSpeech), with the `module.` key prefix removed. Its keys match ALIF's
  Tacotron2 exactly (`strict=True`), but it is a different release than the file ALIF's authors used.

## Experimental scope (decisions)

- Two tasks: **ASR** (harvard, 100 Harvard sentences) and **intent matching** (commands, 100 SLURP test commands).
  LibriSpeech was dropped; so was the "generalization across ASR models and datasets" study (RQ1 section, run block,
  README row). The attacked ASR model is whisper-tiny everywhere. `--asr_model` still accepts the other Whisper and
  wav2vec2-large models, but no reported experiment uses them.
- Main comparison (GATAS, Waveform, SMACK, ALIF, PGD): 300 attacked-ASR queries per sentence (20 x 15), seed 0.
  No early stopping for any method (a stop rule cannot guarantee the evaluated success, the evaluation re-transcribes the saved audio).
  ALIF epochs 10 -> 15 and PGD iterations 200 -> 300 to match (SMACK keeps its upstream settings, ~263 measured, <=300). Objective-formulation study (native targeted SMACK vs
  targeted GATAS) on harvard only.
- Ablations (scale, space) use 20 sentences and one seed (stated in METHODOLOGY.md, limitations).
- The code base was reduced to what the experiments use: only PESQ, SET_OVERLAP and WER_TARGET remain as objectives
  (other objectives were never part of a reported run). Per-run plots (Pareto/hypervolume/spectrogram PNGs) and
  `fitness_history.csv` were removed; reported figures come from the notebooks (`archive_history.json`).

## Intent matching

- **Intent classifier:** pretrained, no training in this work: `cartesinus/slurp-intent_baseline-xlm_r-en`
  (XLM-R base fine-tuned on SLURP, 93 intents, MIT, 87.5 % test accuracy on gold text per its model card).
  On our 100 commands it gets 87/100 on the gold text. The intent is predicted from the ASR transcription.
  Alternatives considered: fine-tuning our own classifier (rejected, no training in the paper), sentence-embedding
  kNN, an LLM.
- **Reference intents (benchmark labels):** the reference intent of a seed sentence is the classifier's prediction on its
  clean text, not SLURP's gold label. For a benchmark that only needs a stable label per sentence this is enough, and
  it removes the classifier's own errors from the metrics: the classifier agrees with the reference on all clean
  sentences by construction. It reproduces SLURP's gold intent for 87/100 sentences (13 misses: chit-chat
  `general_quirky`, near-duplicate music intents, calendar set/remove, ambiguous `query`); gold labels are kept in
  `commands.gold_intents.txt`. State this in the paper and do not claim SLURP-gold intent accuracy.
- **Dataset:** 100 SLURP test-set commands (seed 0 sample; >= 4 words, >= 2 content words, unique, exactly one gold
  intent), 38 distinct reference intents (`src/data/seed_sentences/commands.intents.txt`). Intent = SLURP
  `intent` field space (scenario_action, e.g. `alarm_query`). The classifier was trained on the SLURP train split, our
  sentences come from the test split.
- **Targets:** targeted attacks (GATAS targeted, SMACK native) get a target sentence of a *different reference intent*
  (seeded, identical for all methods). On harvard (no intents) the target is any other sentence, as before.
- **Metrics (`evaluation.json` -> `intent`):**
  - `changed`: predicted intent of the adversarial transcription differs from the predicted intent of the original
    audio's transcription. The original (not the reference label) is the reference point, so classifier or ASR errors on clean
    audio do not count as attack success.
  - `target_match` (targeted attacks only): predicted intent equals the reference intent of the target sentence.
  - `original_correct`: the ASR transcription of the clean audio still maps to the reference intent (control, report the rate and
    optionally restrict results to these sentences).
- **Targets (decision):** the target intent of a sentence is the classifier's most likely intent on the clean text after
  the reference that is well-formed (`scenario_action`) and in a *different scenario* (35 distinct targets over 100
  sentences). Plain "2nd most likely intent" was rejected: for 36/100 sentences it is a scenario-less SLURP label twin
  (`query`, `set`, `radio`, ... label noise) and for 32/100 it stays in the same scenario.
- **Optimization signal (decision):** the SUT of the task is ASR + intent classifier, so its output is a legitimate
  black-box signal. Every method that can use a score does: INTENT_TARGET = 1 - P(target intent | transcription)
  (threshold 0.5) replaces SET_OVERLAP in GATAS and Waveform (`GATAS_intent`, `Waveform_intent`), is the PSO loss of ALIF
  (`ALIF_intent`) while SMACK stays upstream and gets the intent rewrite (`commands.intent_rewrites.txt`, same as GATAS_llm) as its native target text (`SMACK` on commands). PGD is
  white-box on Whisper's text decoder, cannot use it, and is only compared untargeted. Note in the paper: GATAS and the
  others use the classifier during the search, the budget (300 SUT queries) counts one ASR+classifier call each.
- **Metrics (`evaluation.json` -> `intent`):** `target_match` (predicted intent of the adversarial transcription equals
  the target intent; computed for every method, `targeted` marks the methods that optimized toward it, untargeted ones
  can hit it by chance), `changed` (differs from the intent of the original audio's transcription; the original, not
  the reference label, is the baseline so classifier/ASR errors on clean audio are no attack success), and
  `original_correct` (control: the ASR transcription of the clean audio still maps to the reference intent;
  optionally restrict results to these sentences).
- **Caveats:** the PESQ constraint of GATAS/Waveform/(not ALIF, see above) still applies; SMACK's 10x scaling of the
  intent term mirrors the scale of its original text terms (not tuned). With the 300-query budget some targets may be
  unreachable; report `target_match` and `changed` side by side.

## Ablations at 300 queries
Scale and space ablations run the full 300-query budget (20 x 15, no early stopping), same as the main comparison. Numbers quoted in METHODOLOGY.md (ablation tables) were measured at 200 queries and must be refreshed after the re-run.

## Analysis decisions (notebooks)

- **Two tasks, same methods:** the ASR task (harvard) and intent matching (commands) are analysed separately (RQ1 and RQ3
  notebooks). The same methods appear under the same names in both (GATAS, Waveform, SMACK, ALIF, PGD); for commands they
  are the `*_intent` runs (PGD untargeted). `results_loader.load_intent_results` maps the names. The untargeted commands runs are
  not part of the analysis. The query budget is identical for all methods (300) and not compared in the intent section.
- **Sentences with a wrong original are excluded** (ASR: clean audio already fails, `original_passes` false; intent:
  clean transcription classified differently than the reference, `original_correct` false). Tables no longer show `n`,
  `original fails %` or `found % (original passes)`. Intent table: `intent changed %` over the valid sentences,
  `target intent %` = share of the changed intents that are the target intent (not over all sentences).
- **Intent statistics:** exact McNemar test (`binomtest` on discordant pairs) per sentence, GATAS vs each method.
- **Snapshot of the numbers (2026-10-11, whisper-tiny, 100 sentences, one seed; before the final runs, will change):**
  harvard test case found % / UTMOS: GATAS 88.7 / 4.20, Waveform 100 / 1.37, SMACK_untargeted 90.3 / 2.57, SMACK 69.4 / 3.19.
  Intent matching: intent changed % / target intent % (of changed): GATAS 20.4 / 45, Waveform 82.7 / 24.7, SMACK 28.6 / 14.3.
  SMACK is more than 20x slower (about 550 s per sentence on commands vs 13-25 s for the others).

## Waveform is only "effective" because it destroys the audio

- Waveform test cases have a median SNR of about -1.4 dB to the original (noise louder than speech), UTMOS 1.37 vs 4.39 for
  the original. Whisper-tiny then hallucinates ("you", "Thank you", or an empty transcription), so WER = 1.0 almost always.
  Its spectrogram distance (0.16) is small, so the objectives do not see the problem. In the intent task it changes the
  intent most often (83 %) but only with unnatural audio; none of its test cases has UTMOS >= 3 (GATAS: 97 of 98, SMACK: 75 of 98).
- Paper: always report success together with UTMOS (and ideally a validity filter, e.g. UTMOS >= 3, in the intent task). GATAS
  is the best method among the valid test cases (target intent 9 vs 1 for SMACK among UTMOS >= 3). Do not rank on success alone.
- **What Waveform does exactly** (`scripts/adversarial_waveform.py`, `WaveformAdversarialTrainer`): the same search as GATAS but in waveform space.
  The "target audio" is the TTS rendering of a random embedding (Gaussian noise with the per-tensor mean/std of the seed's
  `h_text`, `h_bert` and both style vectors; the loudest of `num_rms_candidates` draws), cut/padded to the seed length and RMS-matched
  to the seed. A candidate is one weight per audio sample, `audio = (1 - alpha) * seed + alpha * target`, alpha in [0, 1] (tens of
  thousands of genes), clamped to [-1, 1]. NSGA-II, population 20, 15 generations, objectives PESQ and SET_OVERLAP (or INTENT_TARGET).
  With `--seed_target` (used in `run_waveform.sh`) the first individual of the initial population is the all-ones genome, i.e. the pure
  target audio, so a candidate with no word overlap with the seed is in the archive from query 1. This anchor and the unconstrained
  sample-wise mixing explain the 100 % found, SNR about -1.4 dB and UTMOS about 1.4. Describe Waveform in the paper as a deliberately
  unconstrained waveform-space baseline, not as a competitive natural-audio attacker.
- Not analysed: why the PESQ threshold (0.2) in the waveform search does not keep the audio natural (the pure-target anchor and the selection
  fallback when no candidate meets all thresholds are likely, not verified).

## Intent matching: why GATAS is weaker, what helped

- GATAS's additive TTS-embedding perturbation keeps the content (mean WER 0.27, the words stay almost the same), the intent
  classifier is robust to that. Intent changed 20 %, target intent reached for about 9 of 98 sentences.
- PESQ fitness is $(4.5-\text{PESQ})/5$; the GATAS threshold 0.2 means PESQ >= 3.5. From the archives of GATAS_intent: loosening the
  limit does not help (P(target) > 0.5 reached for 1/2/2/3/4 sentences at fitness 0.2/0.3/0.4/0.5/0.6; 26 sentences only at
  fitness 0.7 = PESQ about 1, i.e. unusable audio). The quality constraint is not the main limit.
- **Selection fix:** the old selection used the ASR criterion (WER > 0) and a balanced PESQ/intent trade-off. Intent-first
  selection (`--intent_selection`: among members that reach the target take the most natural, else the closest) gave on 20
  sentences: target intent 0 -> 2, intent changed 4 -> 10, UTMOS 4.15 -> 3.56. The margin objective
  (sigmoid of target logit minus best other logit, threshold 0.5 = target is the argmax) gave no gain. Report as an ablation.
  **The main GATAS_intent run used the old selection**; decide whether to rerun it with `--intent_selection` for the paper
  (free, no extra queries; baselines unaffected, but the selection code is shared with Waveform so keep it a flag).
- Small samples: 2 vs 0 target hits out of 19 is weak evidence; intent changed (10 vs 4) is clearer, no test run yet.

## GATAS_llm: latent interpolation toward an LLM rewrite (intent matching)

- Description in METHODOLOGY.md Section 8. Facts to keep: local Ollama `qwen2.5:latest` (7B), minimal-edit prompt, 8 samples,
  verified with the intent classifier, cached in `commands.intent_rewrites.txt`; 70/100 rewrites are classified as the target intent;
  target embedding uses the seed's style noise and is resampled to the seed's phoneme count; genome = 8 phoneme-axis alphas + 2
  style alphas, interpolation only, CMA-ES, PESQ<=0.2 and INTENT_TARGET<=0.5, 300 queries, intent-first selection.
- Related work: Mimicry (Weißl et al., "Targeted Deep Learning System Boundary Testing", TOSEM 2025, arXiv 2408.06258) mixes StyleGAN latent
  features of a source and a target input guided by classifier confidence to approximate the decision boundary. Similarity:
  latent mixing of source/target, black-box confidence feedback, content/style mixing. Differences: our target is an LLM rewrite
  synthesized by the TTS, the system under test is ASR + intent classifier, we search per-segment alphas with CMA-ES under a PESQ
  constraint and aim to cross to the target intent rather than to sit at the boundary (boundary distance, e.g. the intent
  margin of the selected test case, could be reported to make it comparable).
- Sanity check on sentences 1, 2, 4: alpha=0 -> seed text, alpha=1 -> rewrite text and target intent, alpha=0.5 everywhere -> garbled.
- Results: run on all 100 sentences started 2026-10-11 (log `outputs/logs/gatas_llm.log`); numbers to be added after the run.

## Run issues to mention / fix before the final numbers

- **PGD race condition (invalid first run):** PGD runs on the two datasets were run in parallel and shared the working directory
  (`outputs/pgd_workdir`: csv/ and pgd_save/), so they overwrote each other; only 31/100 sentences per dataset were attacked
  and some of them may use the other dataset's csv. Fixed (per-dataset workdir `outputs/pgd_workdir/<dataset>`); the PGD runs of
  both datasets must be rerun, and the old PGD result folders (31 sentences each) must be ignored/deleted before analysis.
  Independent: some PGD sentences exit with code 1 (Whisper tokenizer error on a hallucinated language token `<|ru|>`).
- ALIF (harvard and commands, plus ALIF_intent) was still running at the time of writing (about 2.5 min per sentence); evaluation happens only after all
  100 sentences. ALIF's workdir is read-only, so parallel runs are safe.
- GPU handling: GATAS/Waveform/PGD/evaluation use `CUDA_VISIBLE_DEVICES=<n>` with `--gpu 0` because the UTMOS TorchScript model is
  hardcoded to cuda:0; ALIF/SMACK take the physical GPU index themselves. A previous run on commands failed for that reason
  (cross-device error), and results of GATAS/Waveform/GATAS_intent/Waveform_intent were regenerated on 2026-10-11 (queues in
  `scripts/queue_remaining.sh`).
- Intent runs of SMACK are redone as native targeted SMACK with the intent rewrite as target text (the earlier `SMACK_intent`,
  a modified fitness, is removed).

## GATAS_llm results (commands, whisper-tiny, 98 sentences with correct clean intent, one seed)
- Intent changed 84.7 %, target intent 72/83 of the changed intents (86.7 %), UTMOS 4.15, spectrogram distance 0.10.
- GATAS_intent: 20 changed, 9 target hits, UTMOS 4.19. Waveform_intent: 82.7 % changed but 24.7 % target, UTMOS 1.40. (old modified-fitness SMACK_intent results discarded.)
- Runtime 30.4 s per sentence vs 25.2 s for GATAS_intent. The LLM rewrites are generated once (cached); 70/100 rewrites are classified as the target intent.
- Added to RQ1/RQ3 intent sections as `GATAS_llm` (loader key in `INTENT_FOLDERS`); notebook outputs need a rerun.

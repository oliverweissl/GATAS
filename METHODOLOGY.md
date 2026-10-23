# GATAS: Methodology

GATAS (Generative Adversarial Test generation for ASR) is a **black-box** test generator for automatic speech
recognition (ASR) systems. Instead of perturbing a waveform, it perturbs the *inputs of a text-to-speech (TTS) model*.
Every candidate is therefore speech that a TTS model could plausibly produce, and the search only needs the
transcription of the system under test (SUT), with no gradients or logits.

This document describes the best configuration found in the ablation studies (`analysis/Ablation_Space.ipynb`:
scale, search space and optimizer):

| Component | Choice | Flag |
|---|---|---|
| Search space | Low-rank additive perturbation of text embedding and both style vectors | |
| Directions | Jacobian (output-sensitive) basis | `--basis jacobian` |
| Text coefficients | DCT over the phoneme axis, 8 components | `--text_dct 8` |
| Perturbation scale | Adaptive, one extra gene in [0.5, 4] | `--iv_scalar 4 --adaptive_scale` |
| Objectives | PESQ and SET_OVERLAP (minimized) | `--objectives "PESQ=0.2, SET_OVERLAP=0.5"` |
| Optimizer | CMA-ES (diagonal) on the Chebyshev scalarization | `--algorithm cmaes` |
| Budget | 20 individuals × 15 generations = 300 SUT queries | `--pop_size 20 --num_generations 15` |

## 1. Problem

Given a seed sentence $s$ and an ASR system $A$, a **test case** is an audio $x$ that sounds like $s$ but for which
$A(x)$ differs from $s$ (WER $> 0$). A good test case changes the audio as little as possible, so that a failure
reflects a weakness of the ASR and not unintelligible input. The goal is to find such audios with few queries to $A$.

## 2. Generator: StyleTTS2

GATAS uses StyleTTS2 as the generator. For one sentence the TTS pipeline has these intermediate representations:

| Representation | Shape | Role |
|---|---|---|
| `h_text` | 512 × P | acoustic text encoding, one column per phoneme (P phonemes) |
| `h_bert` | – | prosodic (BERT) text encoding; predicts duration and F0 |
| `style_vector_acoustic` | 128 | style from the diffusion sampler; steers duration / pitch |
| `style_vector_prosodic` | 128 | style from the diffusion sampler; steers the decoder |

`h_bert` stays fixed. The reference embedding of every sentence is computed once (`scripts/generate_reference_audios.py`,
seed-fixed style noise), so **all methods attack the same reference audio**. The decoder (iSTFTNet) maps the
embeddings to a 24 kHz waveform, which is resampled to 16 kHz.

## 3. Search space

A candidate is the reference embedding plus a perturbation in a low-dimensional subspace.
For the three perturbed tensors $h$ (text), $a$ (acoustic style), $p$ (prosodic style):

$$
\tilde h_{:,j} = h_{:,j} + \sigma \sum_{r} c^{(h)}_{j,r}\, b^{(h)}_r ,\qquad
\tilde a = a + \sigma \sum_r c^{(a)}_r b^{(a)}_r ,\qquad
\tilde p = p + \sigma \sum_r c^{(p)}_r b^{(p)}_r
$$

with directions $b_r$ (Section 3.1), coefficients $c$ (Section 3.2) and scale $\sigma$ (Section 3.3).

### 3.1 Directions: the Jacobian basis

Directions are computed **per sentence, without any dataset**.

1. **Candidate subspace (PCA).**
   - Text: SVD of the sentence's own phoneme embeddings (P × 512, uncentered), top $3\cdot 8 = 24$ directions.
   - Style: 48 style vectors are sampled for the same text with random diffusion noise; PCA (centered) of the
     acoustic and the prosodic vectors, top $\min(3\cdot 16, 47) = 47$ directions each.
   - Every direction is scaled by its standard deviation, so a coefficient of 1 means one standard deviation.
2. **Audibility probing.** Directions are ranked by how much they change the *audio*, not by embedding variance.
   Each candidate direction $b_k$ is applied with coefficient 1 and the TTS output is rendered. For the text, 8
   phonemes spread evenly over the sentence are probed one at a time; for the styles the whole vector is perturbed.
   The response is the difference of the log-mel spectrograms (80 mels, 25 ms / 10 ms; zero-padded in time when the
   duration changes) to the unperturbed audio, $r_k = \log M(\tilde x_k) - \log M(x)$.
3. **Selection.** The Gram matrix $G_{kl} = \sum_{\text{probes}} \langle r_k, r_l \rangle$ is a Gauss–Newton metric
   of the TTS in mel space. Its eigenvectors with the largest eigenvalues are the directions that change the
   output most per unit step. The kept basis is those top eigenvectors, mapped back to embedding space:
   **8 directions for the text, 16 for each style vector**.

Only the TTS is queried in this step (about 260 forward passes per sentence, a few seconds). **No SUT query is
spent**, so the budget in Section 5 is unaffected. The eigenvectors are orthonormal in coefficient space, so a
coefficient keeps its meaning of "one standard deviation along a direction".

### 3.2 Coefficients: DCT over the phoneme axis

Free per-phoneme text coefficients would be $P \times 8$ genes whose neighbours are independent. GATAS instead
parameterizes the text coefficients along the phoneme axis with the first $K=8$ components of an orthonormal
DCT-II:

$$
c^{(h)}_{j,r} = \sqrt{P/K}\sum_{k=0}^{K-1} D_{jk}\, g_{k,r},\qquad
D_{jk} = \sqrt{\tfrac{2}{P}}\cos\!\Big(\tfrac{\pi (2j+1) k}{2P}\Big)\ (D_{j0}=\sqrt{1/P})
$$

The text therefore has $K\times 8 = 64$ genes regardless of the sentence length, and the perturbation varies
smoothly over the sentence (low-frequency, prosody-like changes) instead of independently per phoneme. The factor
$\sqrt{P/K}$ keeps the per-phoneme perturbation size equal to the free parameterization. The style vectors have
no time axis and keep one coefficient per direction (16 + 16).

### 3.3 Genome and adaptive scale

The genome $g \in [0,1]^{n}$ has $n = 64 + 16 + 16 + 1 = 97$ genes. Gene values are mapped to coefficients by
$c = 2g - 1$, so **$g = 0.5$ is the unperturbed sentence**. The last gene sets the scale of the individual:

$$
\sigma = 0.5 + (4 - 0.5)\, g_{n}
$$

The scale ablation showed that the perturbation size is the main trade-off (larger scale: more test cases, lower
UTMOS). Encoding it in the genome lets the search raise the scale when nothing is found and lower it when it is
close; the selected scale was 1.77 on average (median 1.56) for NSGA-II, below the fixed best of 2.

## 4. Objectives and optimizer

### 4.1 Objectives (both minimized)

| Objective | Definition | Threshold |
|---|---|---|
| **PESQ** | Wideband PESQ between the reference audio and the candidate, mapped to $[0,1]$: $1-(\text{PESQ}+0.5)/5$. Failures count as worst | 0.2 |
| **SET_OVERLAP** | Fraction of the ground-truth content-word lemmas (lowercase, no punctuation, no stopwords, WordNet-lemmatized) that survive in the transcription | 0.5 |

PESQ keeps the audio close to the reference, SET_OVERLAP drives the transcription away from the content. Both
are computed from the audio and the SUT transcription only. They are optimization objectives, so they are
**not** used to evaluate the results (Section 6).

### 4.2 CMA-ES on the Chebyshev scalarization

The two objectives are scalarized by

$$
f(g) = \max_i \frac{f_i(g)}{\tau_i},\qquad \tau = (0.2,\ 0.5)
$$

$f \le 1$ means that **every threshold is met**, and minimizing the maximum removes the largest relative violation
first, with no weighting parameter. The scalar is minimized with CMA-ES (pycma):

- population 20 (= `pop_size`), start $g_0 = 0.5$ (unperturbed), initial step size 0.3,
- **diagonal covariance** (`CMA_diagonal`): 97 genes and about 300 evaluations cannot support a full
  covariance matrix,
- box constraints $[0,1]$ via pycma's boundary handling.

The ablation compared NSGA-II, CMA-ES and MO-CMA-ES: CMA-ES with the Jacobian basis and DCT gave the most
natural audio with the smallest change at the same 100 % success; MO-CMA-ES offered no gain because its per-parent
covariances cannot be learned in 10 generations.

### 4.3 Archive and selection

Independent of the optimizer, all evaluated individuals are collected in a **Pareto archive** over the two
(unscalarized) objectives. After the last generation one candidate is selected:

1. Prefer archive members whose stored transcription already differs from the ground truth (WER $> 0$). This uses
   the transcriptions recorded during the search, so it costs **no extra query**.
2. Among those, prefer members that meet all thresholds, if any.
3. Divide each objective by its threshold and take the member with the smallest $L_3$ norm (it rewards small and
   balanced values: (0.3, 0.3) beats (0.1, 0.9)).

**Intent-first selection (`--intent_selection`, intent matching only).** Step 1 selects by the ASR criterion (WER $> 0$),
which is unrelated to the intent task, and step 3 trades the intent objective off against PESQ. With an intent
objective the selection is instead: among the archive members whose intent fitness is within its threshold (they reach
the target intent), take the one with the smallest PESQ fitness (most natural); if none reaches it, take the one with
the lowest intent fitness (closest to the target). It uses the stored transcriptions only (no extra query). It is
optional and off by default, so the other runs are unchanged; see Section 9.

## 5. Procedure

```
for each seed sentence s:
    reference embedding e(s) and audio x0 = TTS(e(s))                 # fixed, shared by all methods
    bases = JacobianBasis(TTS, e(s))                                   # Section 3.1, 0 SUT queries
    es = CMA-ES(n = 97, x0 = 0.5, sigma0 = 0.3, popsize = 20, diagonal)
    for generation in 1..15:                                           # 300 SUT queries in total
        G = es.ask()                                                   # 20 genomes in [0,1]^97
        for g in G:
            embedding = perturb(e(s), bases, DCT, scale(g), g)         # Section 3
            x = TTS(embedding);  text = SUT(x)
            f = (PESQ(x0, x), SET_OVERLAP(s, text))
            archive.update(g, f, x, text)
        es.tell(G, max_i f_i / tau_i)
    return select(archive)                                             # Section 4.3
```

Early stopping is disabled: the full 300-query budget is always used, so all methods are compared at an equal query budget. (A stop rule cannot guarantee that the evaluated final test case is a success: the evaluation re-transcribes the saved audio.)

## 6. Evaluation

The selected audio and the original audio are re-transcribed by the SUT (`scripts/evaluate_results.py`), identically
for every method:

| Metric | Definition |
|---|---|
| **Test case found** | transcription differs from the ground truth text (WER $> 0$) |
| Lexical distance | WER between ground truth and transcription |
| Semantic distance | $1-$ cosine similarity of SBERT (all-MiniLM-L6-v2) embeddings |
| Spectrogram distance | normalized Frobenius distance of the mel spectrograms of original and adversarial audio |
| **Naturalness** | UTMOS of the adversarial audio (not an optimization objective) |

**Intent matching (`commands` dataset).** Besides the ASR task (harvard), the framework evaluates an intent-matching
task on 100 SLURP test commands. Its system under test is the ASR model followed by a pretrained text intent
classifier (`cartesinus/slurp-intent_baseline-xlm_r-en`, `src/intent.py`; 93 intents, no training in this work).

- *Reference intent* of a sentence: the classifier's most likely intent on its clean text
  (`src/data/seed_sentences/commands.intents.txt`). The classifier therefore agrees with the reference on every clean
  sentence by construction; SLURP's gold labels (reproduced for 87/100 sentences) are kept in `commands.gold_intents.txt`.
- *Target intent* (`commands.target_intents.txt`): the classifier's most likely intent on the clean text, after the
  reference, that is well-formed (`scenario_action`) and in a different scenario, i.e. the nearest decision boundary
  that is a real change of intent.
- *Optimization signal.* The intent output of the SUT is a graded black-box signal. The INTENT_TARGET objective is
  $1 - P(\text{target intent} \mid \text{transcription})$ (threshold 0.5: the target is more likely than not). It replaces
  SET_OVERLAP in GATAS and the waveform baseline (`GATAS_intent`, `Waveform_intent`; PESQ stays the second objective).
  ALIF uses it as its PSO loss (`ALIF_intent`). SMACK stays as upstream (native targeted attack) and receives the
  LLM intent rewrite of the sentence as target text (`SMACK` on commands), the same sentence `GATAS_llm` interpolates toward. PGD is white-box on Whisper's
  text decoder, cannot use a classifier score, and stays untargeted.

| Intent metric | Definition |
|---|---|
| `target_match` | the intent of the adversarial transcription equals the target intent (reported for every method; `targeted` says whether the method optimized toward it) |
| `changed` | the intent of the adversarial transcription differs from the intent of the *original* audio's transcription |
| `original_correct` | control: the ASR transcription of the original audio still maps to the reference intent |

`changed` is measured against the original transcription, so classifier or ASR errors on clean audio do not count as
attack success.

In the notebooks the ASR task and the intent-matching task are two separate subsets of the experiments (harvard and
commands); the same methods are applied to both under the same names (the `*_intent` folders of the commands dataset are
the methods run with the intent objective, PGD untargeted). Runtime is compared per task; the query budget is the same
for all methods (300) and not compared.

**Margin objective (`INTENT_MARGIN`, ablation).** $	ext{sigmoid}(-m/4)$ with $m$ the target logit minus the largest other
logit of the classifier on the transcription. 0.5 is the decision boundary (target = predicted intent), the value does
not saturate at 1 for far transcriptions. It was tested against $1-P(	ext{target})$ and gave no gain (Section 9), so
the reported intent runs use $1-P(	ext{target})$.

**Control: sentences with a wrong original are excluded.** Some seed audios are already mis-transcribed by the ASR
(`original_passes` is false: WER of the clean audio $> 0$ by the success criterion) or, for intent matching, already
classified as a different intent than the reference (`original_correct` false). For these a "found" test case is not
caused by the attack, so the analysis notebooks (`results_loader.load_results`, `load_intent_results`) exclude them from
every table and test. All methods share the same reference audio and ASR, so the same sentences are excluded for all
methods and the pairing is kept. The reported rates are therefore over the remaining sentences only.
ASR task: **test case found %** over the sentences with a correct original. Intent task: **intent changed %** over the
sentences with a correct original intent, and **target intent %** as the share of the *changed* intents that are the
target intent (untargeted methods reach it by chance). Methods are compared with paired Wilcoxon tests per sentence
(per-sentence outcomes of the intent task with the exact McNemar test).

## 7. Results of the ablations (harvard, whisper-tiny, 20 sentences, one seed)

| Configuration | found % | WER | spectrogram | UTMOS |
|---|---|---|---|---|
| NSGA-II, PCA, free text, adaptive scale | 100 | 0.374 | 0.082 | 3.81 |
| + DCT text | 100 | 0.319 | 0.080 | 4.10 |
| CMA-ES | 95 | 0.204 | 0.074 | 4.17 |
| **CMA-ES + Jacobian + DCT (this document)** | **100** | **0.168** | **0.068** | **4.29** |

The UTMOS gain over the reference is $+0.48$ (paired Wilcoxon $p=0.008$, uncorrected; about 0.056 after Bonferroni
correction over the 7 compared configurations).

## 8. Latent-interpolation variant: GATAS_llm (intent matching)

Besides the additive perturbation of Section 3, one GATAS variant searches an **interpolation in StyleTTS2's latent space
between the seed sentence and a target sentence** (`--target_interp`, `scripts/run_gatas_llm.sh`, `InterpolationManipulator`
in `src/trainer/vector_manipulator.py`). It is related to Mimicry (Weißl et al., "Targeted Deep Learning System Boundary
Testing", TOSEM 2025), which mixes latent style features of a source and a target input guided by the classifier confidence.

- **Target sentence.** A local LLM (Ollama, `qwen2.5:latest`, 7B) rewrites each seed sentence with as few word changes as
  possible so that its intent becomes the sentence's target intent (`scripts/generate_intent_rewrites.py`; prompt gives the
  current and target intent with scenario/action). 8 samples (temperature $0.2+0.1t$) per sentence are checked with the intent
  classifier; kept is the one predicted as the target intent with the fewest changed words (else the most probable).
  The rewrites are cached in `src/data/seed_sentences/commands.intent_rewrites.txt`, so all runs and seeds use identical
  target text and no LLM call happens during the attack (it costs no SUT query). **70/100 rewrites are classified as the
  target intent** (the other 30 cannot reach the target even at the interpolation end point).
- **Target embedding.** Its StyleTTS2 embedding is extracted with the same diffusion style noise as the seed's reference
  (`seed + sentence_id`); the text and BERT features are linearly resampled to the seed's phoneme count, so the seed's
  duration frame and text mask are kept (the audio length follows the seed's frame).
- **Genome.** $g\in[0,1]^{10}$: $K=8$ alphas along the phoneme axis (piecewise linear, applied to `h_text` and `h_bert`;
  0 = seed, 1 = target) and one alpha each for the acoustic and the prosodic style vector.
  $e(g) = (1-\alpha)\,e_\text{seed} + \alpha\,e_\text{target}$ per component. Interpolation only: no additive perturbation,
  no scale gene.
- **Search.** Same as GATAS_intent: CMA-ES (start 0.5, $\sigma_0=0.3$, diagonal), objectives PESQ ($\le 0.2$) and
  INTENT_TARGET ($\le 0.5$), 20 x 15 = 300 queries, intent-first selection.
- **Sanity check** (sentences 1, 2, 4): $\alpha = 0$ is transcribed as the seed, $\alpha = 1$ as the rewrite and classified as
  the target intent; $\alpha = 0.5$ everywhere is garbled speech. A successful test case therefore needs a partial
  interpolation (some phonemes or only the style) that keeps the audio close to the seed.
- **Result** (98 sentences with a correct clean intent): intent changed 84.7 %, of which 72/83 (86.7 %) are the target intent;
  UTMOS 4.15. GATAS_intent: 20 changed, 9 target hits, UTMOS 4.19; Waveform_intent: 82.7 % changed but 24.7 % target
  at UTMOS 1.40. Runtime 30.4 s vs 25.2 s per sentence.
- The search space is small (10 genes) and the target is fixed per sentence, so, unlike the additive search, it can only
  reach intents the rewrite expresses.

## 9. Ablation of GATAS on intent matching (commands, 20 sentences, one seed)

| Variant | target intent | intent changed | WER | UTMOS |
|---|---|---|---|---|
| GATAS_intent ($1-P$, balanced selection) | 0 | 4 | 0.29 | 4.15 |
| + intent-first selection | 2 | 10 | 0.51 | 3.56 |
| margin objective | 0 | 4 | 0.27 | 4.15 |
| margin objective + intent-first selection | 2 | 10 | 0.49 | 3.57 |

(19 of 20 sentences with a correct original intent; counts, not rates.) The selection rule is the effective change: the
archive contains members that move the intent which the balanced selection discards, at a cost of about 0.6 UTMOS; the
margin objective has no effect, so the search signal was not the bottleneck. **The main GATAS_intent run in the main
comparison used the balanced selection** (intent-first is not part of `run_gatas_intent.sh`).
Archive analysis of the main run (100 sentences): P(target) > 0.5 is reached by some archive member for only 26 sentences
even without a PESQ limit, and only at PESQ $\approx 1$ (fitness 0.7); with the PESQ bound at fitness 0.2/0.3/0.4/0.5/0.6
it is reached for 1/2/2/3/4 sentences. The quality constraint is therefore not the only limit: the additive search space
rarely changes the content enough.

## 10. Limitations

- The ablation uses 20 sentences and one seed; one sentence is 5 points of success. The configuration should be
  confirmed on 100 sentences, further datasets, and several seeds.
- **Weak search signal.** SET_OVERLAP is a coarse word-set metric: in all configurations the best archive value
  only drops from 1.0 to about 0.95, so the thresholds are almost never met. "Found" (any transcription change)
  is achieved nevertheless, but a graded objective (e.g. WER or semantic distance) would give the optimizer a
  stronger gradient. The Chebyshev scalarization then reduces to its more informative term.
- Reachable test cases are limited to what StyleTTS2 can generate in the chosen subspace; a smooth DCT
  perturbation cannot target a single phoneme.
- The Jacobian basis is local (finite differences around the reference, coefficient 1) and does not model the
  SUT, only the TTS.
- The CMA-ES variant optimizes a scalar. The Pareto archive still records both objectives, but the search does
  not spread the population along the front.
- **Intent matching.** The additive search rarely changes the content enough to flip an intent (about 20 % intent changed,
  Section 9); a successful intent change comes with lower naturalness. Waveform reaches more intent changes but only with
  heavily destroyed audio (UTMOS about 1.4 vs 4.4 for the original), so its
  success is not comparable; report naturalness alongside every success rate. The PESQ threshold in the waveform search
  did not keep the audio natural (reason not analysed).
- Several seeds are not run in the main comparison (seed 0 only); the intent ablation is 19 sentences.
- GATAS_llm depends on the LLM rewrite: 30/100 rewrites do not reach the target intent in the classifier, and the rewrites
  come from a 7B model.

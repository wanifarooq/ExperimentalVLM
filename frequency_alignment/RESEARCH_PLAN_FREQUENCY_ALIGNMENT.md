# Research Plan: Language Conditioning as a Learned Frequency Filter in Vision-Language Models

## Building on: "Same Answer, Different Representations" (Paper 1)

## Implementation Status (April 15, 2026)

This file mixes long-horizon research goals with the current implementation. The code in `frequency_alignment/` now concretely implements the main pipeline on `GQA` for experiments `1-5` and `PartImageNet` for experiment `6`, with the authoritative operational behavior documented in `frequency_alignment/experiment_run.md`.

Current model profiles in code and configs:
- Local smoke profile: `Qwen/Qwen3-VL-2B-Instruct`
- Server profile: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Recent supported family alternatives: `llava-hf/llava-v1.6-mistral-7b-hf`, `OpenGVLab/InternVL3-8B-hf`

Implementation clarifications that supersede older references below:
- The main VQA dataset path is now `GQA` scene graphs. The loader builds same-image primary levels `L1-L4` plus matched wordy controls `L5-L8`. Primary hypothesis tests remain on `L1-L4`; descriptive plots and control comparisons include `L5-L8`.
- `L5-L8` are prompt-load controls: `L5` repeats L1 semantics with long filler wording, `L6` repeats L2 semantics, `L7` repeats L3 semantics, and `L8` repeats L4 semantics. Their prompt-load proxy is required to exceed the matched base prompt, but there is no fixed word or token cap.
- Experiment 1 now stores both raw and relative perturbation spectra in image space (`delta_f`, `delta_f_relative`) and, when enabled, the matching raw and relative vision-feature spectra (`delta_f_vision`, `delta_f_vision_relative`).
- Experiment 1 stores signed and directional correct-answer log-likelihood metrics: `loglik_drift`, `loglik_erosion`, `loglik_recovery`, and `loglik_volatility`.
- Experiments `1`, `2`, `3`, `4`, and `5` now keep discrete level labels but also attach a continuous semantic complexity score based on a structured semantic program plus a grounding-ambiguity term. Prompt/load complexity and MCQ option hardness are tracked separately as controls.
- Horse-race regressions use residualized semantic logic (`complexity_score_residual`) as the main logic variable, where semantic complexity is residualized against prompt load.
- Regression and grouped-correlation reporting now uses three views: Primary (`L1-L4`), Wordy (`L5-L8`), and Pooled (`L1-L8`). Legacy unsuffixed keys remain for compatibility.
- Accuracy-drop horse races are gated to clean-correct points only. This tests fragility of existing knowledge rather than mixing in already-wrong questions.
- Experiment 1 now includes clean-image prediction entropy as an additional horse-race control, separating spectral fragility from flat/confused MCQ score distributions.
- Experiment 1 also runs paired mirror tests comparing each primary level against its wordy counterpart (`L1/L5`, `L2/L6`, `L3/L7`, `L4/L8`) and plots the linguistic stabilization effect as base-minus-wordy drift/drop.
- Experiment 1 also runs perturbation-specific horse races for accuracy drop, log-likelihood drift, erosion, recovery, and volatility.
- Experiment 2 does not rely on a named encoder-decoder cross-attention block. It derives the effective language-to-vision attention map from the self-attention slice, supports optional FFT windowing before the 2D FFT, and computes `W_t` separately for `overall`, `early`, `mid`, and `late` layer groups. The current configs default this windowing to `none`.
- Experiment 2 also includes prompt-only controls (`empty_language`, `random_language`) so task-conditioned filters can be compared against semantically weak prompts.
- Experiment 3 now uses a late decoder hidden state as the default post-fusion representation, treats pre-fusion band drift as the controlled perturbation input, and measures the task-conditioned post-fusion response with an all-token scalar drift. It then tests whether that response follows the overlap between `W_t` and the pre-fusion drift profile for `overall`, `early`, `mid`, and `late`, with `late` treated as the main post-fusion comparison.
- Experiment 5 now runs overlap prediction on both image-space and vision-feature perturbation spectra, keeps both raw and relative normalization branches, and evaluates multiple targets: `accuracy_drop`, `loglik_erosion`, `loglik_volatility`, `net_drop`, and grouped `relative_accuracy_drop`. The primary paper target is `loglik_volatility`, with accuracy-based targets retained as controls.
- Experiment 5 also keeps a standardized "comparable bridge" analysis in parallel with the raw overlap results: the prediction is transformed with `log1p` and both prediction and observed target are z-scored so effect sizes can be compared on a common scale. This does not replace the raw overlap metrics; it complements them.
- Spectral binning is adaptive by default: the runner probes the model patch grid, resolves one linear radial bin count for the run, and freezes it so all spectra remain aligned. DC suppression and log-scale plotting are config/visualization choices, not fixed theory assumptions.
- Complexity scatter plots use residualized semantic logic on the x-axis when available, so wordy controls do not visually collapse the semantic trend.
- Monotonicity and Spearman granularity tests stay on the primary ladder, with secondary wordy-ladder checks where available. Pooled monotonicity is intentionally not reported because it would conflate semantic and prompt-load interventions.
- Perturbation overlays are label-free by default so text, random text, and box overlays are nuisance perturbations shared across levels rather than answer-conditioned interventions.
- GQA dataset construction now uses early stopping and a local granularity cache so repeated runs do not rebuild the same sample list from scratch.
- Experiment 6 now evaluates clean and perturbed `mIoU` against GT PartImageNet masks and uses GT boxes for the SAM2 control.
- Older project notes and prior-result references should be read as research-history context. The current operational behavior is the implementation summarized here and in `frequency_alignment/experiment_run.md`.

---

## 1. HYPOTHESIS

### Primary Hypothesis (H1)
**Language conditioning in vision-language models learns a task-specific frequency filter W_t that binds text concepts to specific spectral signatures in vision features. The sensitivity of this binding to perturbations scales monotonically with the spectral support required by the task -- coarse tasks (object detection, segmentation) require narrow frequency support and are robust, while fine-grained tasks (attribute recognition, spatial reasoning, MCQ) require broad cross-frequency support and are fragile.**

### Sub-Hypotheses

**H1a (Frequency-Selective Attention):** Cross-attention maps from language tokens to vision patches exhibit frequency-selective structure -- coarse concept prompts ("a bicycle") concentrate attention energy in narrow spatial frequencies, while fine-grained prompts ("the red spoke on the front wheel") distribute attention across wider range frequency bands.

**H1b (Drift Amplification):** The language-vision fusion layer amplifies representation drift in frequency bands that the text token attends to and suppresses drift in unattended bands. The ratio of post-fusion to pre-fusion drift per frequency band correlates with cross-attention energy in that band.

**H1c (Granularity Scaling):** For a fixed perturbation type and severity, the performance degradation increases monotonically with task granularity, measured as the bandwidth of the effective frequency filter W_t.

**H1d (Perturbation-Filter Overlap):** The magnitude of performance degradation for a given (task, perturbation) pair is predicted by the spectral overlap between the perturbation's energy injection pattern and the task's frequency filter W_t. High overlap → high degradation, low overlap → robustness.

**H1e (Two-Factor Language Control):** Semantic program complexity and prompt load exert separable forces. Residualized semantic logic increases spectral grounding demand and perturbation-induced drift, while prompt load can act as linguistic anchoring that damps drift. The `L5-L8` wordy controls test this directly by increasing prompt load while holding the underlying semantic program fixed.

### Connection to Paper 1
Paper 1 established:
- All perturbations (spatial and adversarial) manifest as phase/frequency misalignment at the representation level
- Cross-frequency sensitivity (H2), not low-frequency dominance (H1), explains VLM fragility
- Output stability masks representation drift (hidden drift phenomenon)
- Language conditioning is the vulnerability surface

This follow-up explains **why** language conditioning is the vulnerability surface (it's a learned frequency filter) and **when** it breaks (when perturbations corrupt the specific bands the filter depends on).

---

## 2. THEORETICAL FRAMEWORK

### 2.0 Two-Factor Model: Compositional Precision vs Linguistic Anchoring

The current implementation now treats VLM robustness as the balance of two opposing language effects:

- **Compositional Precision**: richer semantic logic forces the model to ground finer visual distinctions and therefore broadens the effective task filter `W_t`. This is the mechanism that increases frequency-based fragility.
- **Linguistic Anchoring**: extra prompt scaffolding can stabilize the model through textual shortcuts or stronger language priors even when it does not increase genuine visual grounding demand.

In practice, this means the project no longer treats prompt length as a proxy for task difficulty. Instead:

- `semantic complexity` is the primary causal variable
- `prompt load` is an explicit control
- `option hardness` is an explicit MCQ-discrimination control

The fixed-effects and multivariate regressions are designed to identify the semantic-complexity effect after controlling for both of these confounders. The intended scientific claim is therefore:

> semantic complexity is the true driver of frequency-based fragility, while prompt length is controlled rather than conflated with semantic demand.

### 2.1 Language Conditioning as Frequency Filter

#### Setup
Let I be an input image, F(I) its 2D Fourier transform, and t a text token (e.g., "bicycle").

The vision encoder maps I to patch tokens: V = Enc_v(I) ∈ R^{N×d}

The language encoder maps text to tokens: L = Enc_l(t) ∈ R^{M×d}

Cross-attention computes:
```
A = softmax(Q_l · K_v^T / √d)    [M × N attention matrix]
Z = A · V_v                        [M × d fused representation]
```

#### Key Insight
Each row of A selects a weighted combination of vision patches. In the spatial domain, this is a spatial filter. We can analyze this filter in the frequency domain.

Define the effective spatial attention map for language token j:
```
a_j(x, y) = A[j, patch_at(x,y)]    [spatial attention map, N patches → H×W grid]
```

Its Fourier transform reveals the frequency selectivity:
```
Â_j(ω_x, ω_y) = F{a_j(x, y)}
```

**Claim**: For coarse concepts, |Â_j|² is concentrated in a narrow spectral band. For fine concepts, |Â_j|² is distributed across a broader range of frequencies.


#### Formal Filter Model
Define the effective frequency filter for text token t:
```
W_t(ω) = E_j[ |Â_j(ω)|² ]    [averaged over language tokens]
```

The fused representation in the frequency domain is approximately:
```
Z(ω) ≈ W_t(ω) · F(V)(ω)
```

#### Perturbation Sensitivity
A perturbation P transforms: F(V) → F(V) + ΔF

Post-perturbation fused representation:
```
Z'(ω) ≈ W_t(ω) · (F(V)(ω) + ΔF(ω))
```

Representation drift:
```
||Z' - Z||² = ∫ |W_t(ω)|² · |ΔF(ω)|² dω
```

This is the **spectral overlap** between the filter and the perturbation. This directly gives us:

**Theorem 1 (Sensitivity Scaling):**
```
Sensitivity(task, perturbation) = ∫ |W_t(ω)|² · |ΔF(ω)|² dω
```

### 2.2 Task Granularity as Spectral Bandwidth

Define granularity G(t) as the effective bandwidth of W_t:
```
G(t) = [∫ |W_t(ω)|² dω]² / ∫ |W_t(ω)|⁴ dω    [inverse participation ratio]
```

- Coarse tasks: G(t) is small (filter concentrated in few frequency bands)
- Fine tasks: G(t) is large (filter spread across many bands)

**Theorem 2 (Granularity-Sensitivity Monotonicity):**
For perturbations with flat spectral energy (e.g., white noise, rotation-induced phase scrambling):
```
|ΔF(ω)|² ≈ σ² (constant across ω)

⟹ Sensitivity ≈ σ² · ∫ |W_t(ω)|² dω ∝ σ² · G(t)
```

Therefore sensitivity scales linearly with task granularity for broadband perturbations.

For narrowband perturbations (e.g., HighPassKeep):
```
|ΔF(ω)|² is large only for |ω| > ω_cutoff

⟹ Sensitivity ∝ ∫_{|ω|>ω_cutoff} |W_t(ω)|² dω
```

Fine tasks have more energy above the cutoff, so they're more affected.

### 2.3 Information-Theoretic Bound

The task requires extracting I(Y; I | t) -- mutual information about label Y from image I given text t.

Define the frequency-band mutual information:
```
I_ω(Y; I | t) = mutual information contributed by frequency band ω
```

The total is:
```
I(Y; I | t) = ∫ I_ω(Y; I | t) dω
```

A perturbation that corrupts band ω reduces I_ω by some amount δ_ω. The total information loss:
```
ΔI = ∫ δ_ω · 1[ω ∈ support(W_t)] dω
```

Fine tasks require more bands (larger support), so ΔI is larger.

**Theorem 3 (Alignment Tax):**
The robustness R of a language-conditioned model is bounded by:
```
R(t) ≤ R_vision_only · [1 - G(t) · ε_perturbation]
```

where ε_perturbation is the average per-band perturbation energy. Language conditioning always reduces robustness relative to the vision-only baseline, and the reduction grows with task granularity.

This formalizes why SAM3 (language-conditioned) is less robust than SAM2 (vision-only) to high-pass filtering, but the gap is small for coarse tasks and large for fine tasks.

### 2.4 Proofs to Write

1. **Theorem 1**: Derive sensitivity as spectral overlap (straightforward from linear filter model)
2. **Theorem 2**: Prove monotonicity under flat-spectrum perturbations (follows from definition of G)
3. **Theorem 3**: Derive alignment tax bound (information-theoretic argument)
4. **Proposition**: Show that rotation, translation, scaling each have characteristic ΔF(ω) profiles, and predict which tasks are most affected by each

---

## 3. EMPIRICAL PROOF STRATEGY

### Experiment 1: Task Granularity Spectrum
**Goal**: Show that sensitivity scales with task granularity on the SAME images.

**Design**: For each image, define 4 task granularity levels:

| Level | Task Type | Example Prompt | Granularity |
|-------|-----------|----------------|-------------|
| L1 (Coarse) | Object presence | "Is there a bicycle in this image?" | Lowest |
| L2 (Medium) | Attribute recognition | "What color is the bicycle?" | Medium |
| L3 (Fine) | Spatial relationship | "Is the bicycle to the left of the person?" | High |
| L4 (Very Fine) | MCQ reasoning | "Which option best describes..." | Highest |

The current GQA implementation also creates matched wordy controls:

| Level | Control Type | Semantic Program | Purpose |
|-------|--------------|------------------|---------|
| L5 | Wordy-simpleton | Same as L1 | More prompt load than L1, low semantic complexity |
| L6 | Wordy-medium | Same as L2 | More prompt load than L2 at attribute-query semantics |
| L7 | Wordy-fine | Same as L3 | More prompt load than L3 at relationship-verify semantics |
| L8 | Wordy-very-fine | Same as L4 | More prompt load than L4 at compositional MCQ semantics |

The primary granularity hypothesis is still evaluated on `L1-L4`. The `L5-L8` controls are used to test whether long prompts alone explain the effect; their semantic programs are copied from matched base levels, while residualized logic is fit on `L1-L4` and then applied to the wordy mirrors.

For segmentation variant:
| Level | Task Type | Example Prompt | Granularity |
|-------|-----------|----------------|-------------|
| L1 | Whole object | "segment the bicycle" | Lowest |
| L2 | Part-level | "segment the bicycle wheel" | Medium |
| L3 | Fine part | "segment the bicycle chain" | High |
| L4 | Boundary detail | "segment the bicycle spoke pattern" | Highest |

**Perturbation suite** (same as Paper 1):
- Natural: Translation (±4, ±8, ±12), Rotation (±10, ±20, ±30), Scale, PadCrop, TextOverlay, BoxOverlay
- Frequency: LowPassKeep, HighPassKeep, LowBandNoise, HighBandNoise, AllBandNoise
- Severity: 1, 2, 3

**Metric**: For each (granularity level, perturbation, severity), measure:
- Accuracy drop (VQA) or mIoU drop (segmentation)
- Representation drift (cosine distance of vision tokens from clean baseline)
- Dirichlet energy change (from Paper 1)
- Correct-answer log-likelihood drift:
  - signed `loglik_drift = clean_ll(correct) - perturbed_ll(correct)`
  - `loglik_erosion = max(drift, 0)`
  - `loglik_recovery = min(drift, 0)`
  - `loglik_volatility = |drift|`

**Continuous view**:
- In addition to `L1-L4`, assign each prompt a semantic complexity score from the underlying program:
  - referenced entity count
  - attribute-query count
  - relation count
  - reasoning-operator count
  - program depth
  - grounding ambiguity `log(1 + k)` where `k` is the candidate grounding count
- Keep prompt/load complexity separate as a control rather than folding option wording into the main semantic score.
- Add an explicit option-hardness control `H_opt` so semantic complexity can be separated from MCQ discrimination difficulty.
- Add prediction entropy as a model-confusion control, computed over the clean-image MCQ option distribution.
- In the current code, the continuous analysis is tested both as pooled slopes and as within-image fixed-effects regressions, plus multivariate horse-race regressions over semantic complexity, prompt load, option hardness, and clean prediction entropy where available.
- Horse-race regressions use residualized semantic logic as the main semantic variable: `complexity_score_residual = residual(complexity_score ~ prompt_complexity_score)`.
- Accuracy-drop regressions are filtered to clean-correct points only, so they measure fragility of already-known answers.
- Paired mirror tests compare terse levels against matched wordy controls on the same image, directly estimating linguistic stabilization as `base - wordy`.
- Perturbation-specific horse races are produced for accuracy drop and all directional log-likelihood outcomes.
- Analyze degradation as a function of this continuous score so the robustness trend can be tested as a slope, not only as four discrete bins.

**Expected Result**: For primary levels `L1-L4`, degradation and internal drift should increase with semantic complexity. For matched wordy controls `L5-L8`, increased prompt load without increased semantics should not mimic the primary semantic-complexity effect; if anything, prompt load may damp drift through linguistic anchoring.

**Dataset**: GQA scene graphs for the current VQA path; PartImageNet for segmentation.
**Sample size**: Configurable. Server defaults target up to 1000 GQA samples for Exp 1, subject to valid same-image sample construction and yes/no balancing.

---

### Experiment 2: Cross-Attention Frequency Analysis
**Goal**: Show that cross-attention maps have frequency-selective structure correlated with task granularity.

**Design**:
1. For each `(image, text prompt)` pair, extract the effective language-to-vision attention map by slicing the model self-attention tensor into non-vision queries over vision tokens.
2. Split the selected layers into `early`, `mid`, and `late` groups, and also keep an `overall` aggregate.
3. Reshape each attention map to the vision patch grid `a_j(x, y)`.
4. Optionally apply a configured 2D window before the FFT to reduce spectral leakage from patch-grid borders. The current configs default this to `none`.
5. Compute the 2D FFT `Â_j(ω_x, ω_y)` and the radial power spectrum `|Â_j|²`.
6. Normalize the spectrum to obtain `W_t(ω)` and compute the effective bandwidth `G(t)` with the inverse participation ratio.
7. Repeat the same procedure for prompt controls (`empty_language`, `random_language`) to test whether the measured filter is language-meaning dependent rather than only image-content dependent.
8. Aggregate the spectra over samples for each granularity level and for each layer group.

**Analysis**:
- Plot mean power spectrum `|Â_j|²` for all available levels, including `L5-L8` wordy controls in descriptive plots
- Plot `overall`, `early`, `mid`, and `late` `W_t(ω)` filters and their bandwidths
- Plot matched wordy-control bandwidth comparisons: `L1` vs `L5`, `L2` vs `L6`, `L3` vs `L7`, `L4` vs `L8`
- Plot effective bandwidth against continuous semantic complexity to test whether each additional prompt atom broadens the task filter
- Run within-image fixed-effects regressions so the semantic-complexity slope is estimated on the same image rather than across pooled image content.
- Run multivariate horse-race regressions `bandwidth ~ semantic + prompt_load + option_hardness`.
- Compare task prompts to prompt-only controls using `L2`, cosine similarity, and Jensen-Shannon divergence between `W_t` distributions
- Correlate `G(t)` with sensitivity from Experiment 1

**Expected Result**:
- L1 prompts: power concentrated at DC and low frequencies
- L4 prompts: power spread across frequency spectrum
- Effective bandwidth should rise smoothly with semantic complexity, not only when moving between coarse discrete level labels
- Wordy controls should reveal whether prompt length alone broadens `W_t`; the core theory predicts semantic logic should be the stronger bandwidth driver after controls
- Control prompts should be closer to each other than to the real task prompts, especially for fine-grained levels
- Strong positive correlation between G(t) and sensitivity

**Models for extraction**: Current primary path is Qwen3-VL via `QwenAdapter`; LLaVA and InternVL adapters remain available for generalization.

---

### Experiment 3: Pre-Fusion vs Post-Fusion Drift
**Goal**: Show that the same pre-fusion perturbation spectrum can produce different post-fusion internal responses depending on the task, and that this difference follows the task filter `W_t`.

**Design**:
1. Extract vision features BEFORE fusion (pre-fusion): `V`
2. For each clean / perturbed image pair, compute the task-agnostic pre-fusion drift spectrum `ΔV(ω)` from vision features only.
3. Extract a late language-conditioned hidden state `Z` from the decoder and measure post-fusion drift as an all-token scalar response:
   - `ΔZ_all = scalar_drift(Z_perturbed, Z_clean)`
4. Group perturbations with similar normalized `ΔV(ω)` profiles so the perturbation spectrum is approximately held fixed inside each group.
5. For each task and filter group (`overall`, `early`, `mid`, `late`), compute the spectral overlap:
   - `overlap(t, p) = ∫ W_t(ω) · ΔV_p(ω) dω`
6. Test whether `ΔZ_all` correlates with this overlap, especially within the matched pre-drift profile groups.

**Expected Result**:
- The same pre-fusion perturbation profile should induce different post-fusion responses for different tasks.
- Tasks whose `W_t` overlaps more strongly with the pre-fusion drift profile should show larger `ΔZ_all`.
- The clearest controlled effect should appear on the `late` filter group.
- Continuous complexity regressions should show that residualized semantic logic predicts `ΔZ_all` and response amplification after prompt-load and option-hardness controls.
- Wordy-control plots should show whether increased prompt load alone changes post-fusion drift relative to the matched base level.

**Models**: Qwen3-VL checkpoints are the primary implemented path because hidden states, vision-token spans, and attention slices are accessible through the adapter.

---

### Experiment 4: Synthetic Frequency Ablation
**Goal**: Directly measure the frequency threshold at which concept grounding fails, for different granularity levels.

**Design**:
1. Take clean images from COCO
2. Create frequency-controlled versions:
   - Keep only frequencies below cutoff ω_c: I_low(ω_c)
   - Keep only frequencies above cutoff ω_c: I_high(ω_c)
   - Sweep ω_c from 0 to Nyquist in 20 steps
3. For each cutoff, evaluate:
   - VQA accuracy at each granularity level
   - Segmentation mIoU at each granularity level
   - Concept grounding success rate

**Expected Result**:
- Coarse tasks remain accurate with low ω_c (only need low frequencies)
- Fine tasks require higher ω_c (need more frequency content)
- The critical cutoff ω_c* where accuracy drops below 50% shifts to higher frequencies as granularity increases
- This directly measures the "bandwidth" of the effective filter W_t
- The same cutoff should also increase smoothly with the continuous semantic complexity score
- The same relationship should survive within-image fixed-effects and multivariate horse-race controls.
- Wordy-control cutoff plots test whether prompt load changes cutoff thresholds independently of semantic program complexity.

**Dataset**: GQA multilevel samples in the current implemented path.

---

### Experiment 5: Perturbation-Filter Overlap Prediction
**Goal**: Show that Theorem 1 (sensitivity = spectral overlap) quantitatively predicts observed degradation.

**Design**:
1. From Experiment 2, measure `W_t(ω)` for each `(image, prompt)` pair and for each filter group: `overall`, `early`, `mid`, `late`.
2. From Experiment 1, load both image-space and vision-feature perturbation spectra, in both raw and relative-normalized form.
3. Compute predicted sensitivity:
   - raw branch: `S_pred = ∫ |W_t(ω)|² · |ΔF(ω)|² dω`
   - relative branch: `S_pred^rel = ∫ |W_t(ω)|² · |ΔF_rel(ω)|² dω`
4. Treat the `late` filter group as the primary hypothesis test, while `overall`, `early`, and `mid` remain control analyses.
5. Compare `S_pred` against multiple observed targets from Experiment 1:
   - `accuracy_drop`
   - `loglik_erosion`
   - `loglik_volatility`
   - `net_drop = (N_CI - N_IC) / N_total`
   - grouped `relative_accuracy_drop = (Acc_clean - Acc_pert) / max(Acc_clean, ε)`
6. Treat `loglik_volatility` as the primary target because it captures confidence motion even when the discrete answer does not flip. Accuracy-based targets remain controls.
7. Compute both grouped correlations over `(level, perturbation)` pairs and raw per-sample correlations.
8. Run a multivariate prediction-factor horse race for observed `accuracy_drop`: `accuracy_drop ~ zscore(log1p(S_pred)) + prompt_load + option_hardness`.
9. Generate level-wise and level-by-perturbation predicted-vs-observed plots. These use separate axes for predicted overlap and observed behavior when their scales differ.
10. Generate wordy-control comparison plots for predicted sensitivity and observed targets.

**Expected Result**:
- Strong positive grouped correlation between predicted overlap and correct-answer log-likelihood volatility, with the cleanest signal typically appearing on the `late` branch
- `loglik_volatility` is the headline target because the theory predicts internal signal motion under perturbation whether that motion is erosion or recovery; `loglik_erosion` and binary `accuracy_drop` remain important controls
- The prediction-factor horse race should show the standardized spectral-overlap coefficient as the strongest independent predictor of accuracy failure after prompt-load and option-hardness controls
- Relative-normalization should make cross-image comparisons more stable without eliminating the raw-overlap control view
- Prediction error and predicted sensitivity should have interpretable continuous relationships with semantic complexity if the spectral-overlap theory captures task-specific vulnerability.

---

### Experiment 6: Segmentation Extension (SAM2 vs SAM3 Granularity)
**Goal**: Extend Paper 1's segmentation results with granularity analysis.

**Design**:
1. Use COCO instances with hierarchical annotations:
   - Whole object: "segment the dog"
   - Part level: "segment the dog's head"
   - Fine part: "segment the dog's eye"
2. Run SAM3 (language-conditioned) and SAM2 (vision-only with GT boxes) on all levels
3. Apply perturbation suite at all severity levels
4. Measure mIoU drop at each granularity level

**Expected Result**:
- SAM3 whole-object segmentation: robust (small mIoU drop)
- SAM3 part segmentation: moderate degradation
- SAM3 fine-part segmentation: significant degradation
- SAM2 shows no granularity effect (box prompts don't have concept-frequency coupling)

**Dataset**: COCO + COCO-Parts or PartImageNet (provides part-level masks)

---

## 4. MODELS TO USE

### Primary Models (must include for credibility)

| Model | Role | Why |
|-------|------|-----|
| **Qwen/Qwen3-VL-2B-Instruct** | Local smoke / debugging path | Fits local 12 GB GPU profile |
| **Qwen/Qwen3-VL-8B-Instruct** | Single-large-GPU path | Larger Qwen3 checkpoint with manageable memory |
| **Qwen/Qwen3-VL-30B-A3B-Instruct** | Server primary VQA path | Current default publishable-run profile with `device_map: auto` |
| **llava-hf/llava-v1.6-mistral-7b-hf** | Generalization check | Different VLM architecture |
| **OpenGVLab/InternVL3-8B-hf** | Generalization check | Different vision-language stack |
| **SAM2 (Hiera-B+)** | Vision-only segmentation baseline | Continuity with segmentation experiments |
| **SAM3 / GroundingDINO+SAM2** | Language-conditioned segmentation | Tests grounding fragility |

### Optional Models (for comprehensive evaluation)

| Model | Role | Why |
|-------|------|-----|
| **BLIP-2** | Cross-architecture validation | Q-Former fusion is different from cross-attention |
| **Gemini 2.0 Flash** | Proprietary model comparison | Tests if proprietary models have same vulnerability |
| **GPT-4o** | Proprietary model comparison | Industry standard |
| **Molmo-7B** | Open alternative | Tests different training recipe |

### Model Selection Rationale
- **Minimum viable**: Qwen3-VL-2B/8B local validation + Qwen3-VL-30B server run
- **Recommended**: Add LLaVA-v1.6 and InternVL3-8B generalization checks
- **Comprehensive**: Add BLIP-2 + one proprietary model (8 models, covers all major architectures)

---

## 5. DATASETS TO USE

### Primary Datasets

| Dataset | Samples | Use For | Why |
|---------|---------|---------|-----|
| **COCO val2017** | 5,000 images | All experiments | Rich annotations, standard benchmark |
| **VQAv2** | 214K questions | Granularity spectrum (VQA) | Provides questions at different granularities |
| **GQA** | 22M questions | Fine-grained spatial reasoning | Compositional questions with scene graphs |
| **SEEDBench** | 19K questions | Continuity with Paper 1 | Direct comparison to prior results |

### Part-Level Segmentation Datasets

| Dataset | Samples | Use For | Why |
|---------|---------|---------|-----|
| **PartImageNet** | 24,095 images | Part segmentation granularity | Hierarchical part annotations (object → part → subpart) |
| **PASCAL-Part** | 10,103 images | Part segmentation | Part-level masks for 20 categories |
| **PACO** | 75K images | Part and attribute | Parts, attributes, and objects in COCO images |

### Frequency Analysis Datasets

| Dataset | Samples | Use For | Why |
|---------|---------|---------|-----|
| **ImageNet-C** | 50K images × 19 corruptions | Corruption robustness baseline | Standard benchmark, reviewers expect it |
| **Stylized-ImageNet** | 50K images | Texture-shape bias | Tests frequency band reliance |

### Recommended Minimum
- **COCO val2017** (all experiments)
- **GQA** (granularity spectrum -- has scene graphs enabling multi-granularity questions)
- **SEEDBench** (continuity with Paper 1)
- **PartImageNet** (segmentation granularity)

---

## 6. STEP-BY-STEP IMPLEMENTATION PLAN

### Phase 1: Foundation (Weeks 1-3)

**Step 1.1: Data Preparation**
- Download COCO val2017, GQA, PartImageNet
- Build multi-granularity question sets from GQA scene graphs:
  - L1: Object presence ("Is there a [object]?")
  - L2: Attribute ("What color/size/material is the [object]?")
  - L3: Relationship ("Is the [object] to the left/right/above of [object2]?")
  - L4: Compositional ("Which [object] that is [attribute] is closest to [object2]?")
  - L5-L8: matched wordy controls for L1-L4 semantics
- Build part-level segmentation manifests from PartImageNet:
  - L1: Whole object mask
  - L2: Part mask (head, body, legs)
  - L3: Subpart mask (eye, ear, paw)
- Target: up to 1000 images with valid same-image `L1-L8` VQA tasks, and 500 images with 3 levels for segmentation

**Step 1.2: Perturbation Engine**
- Reuse perturbation code from Paper 1 (vlm_invariance_check.py)
- Ensure all perturbations also output ΔF(ω) -- the spectral signature of each perturbation
- Add frequency sweep perturbation: progressive low-pass/high-pass with 20 cutoff values
- Use label-free overlays by default, with answer-conditioned overlays retained only as a legacy ablation mode
- Validate against Paper 1 results on SEEDBench (sanity check)

**Step 1.3: Model Setup**
- Set up Qwen3-VL checkpoints with intermediate layer extraction hooks
- Set up LLaVA-v1.6 with same hooks as the main generalization model
- Set up SAM2 + SAM3/GroundingDINO (reuse segmentation_robustness code)
- Validate all models produce expected baseline accuracy on clean data

### Phase 2: Core Experiments (Weeks 4-8)

**Step 2.1: Experiment 1 -- Task Granularity Spectrum** (Week 4-5)
- Run all perturbations × all severity levels × all granularity levels on:
  - Qwen3-VL with GQA questions (`L1-L4` primary, `L5-L8` wordy controls)
  - SAM3 with PartImageNet (L1-L3)
- Record: accuracy/mIoU, correct-answer log-likelihood drift, directional drift, optional cosine drift, optional Dirichlet energy
- Compute: degradation curves (metric vs severity) for each granularity level
- Statistical test: primary `L1-L4` monotonicity plus continuous fixed-effects and horse-race regressions
- Target output: primary degradation curves plus wordy-control comparisons and perturbation-specific coefficient plots

**Step 2.2: Experiment 2 -- Cross-Attention Frequency Analysis** (Week 5-6)
- For configured samples × available prompts, extract effective language-to-vision attention maps from Qwen3-VL
- Implementation:
  ```python
  # Derive the language-to-vision slice from self-attention
  attn_spatial = attention.reshape(H_patches, W_patches)

  # Optionally apply a configured window before the FFT
  attn_windowed = maybe_window(attn_spatial, fft_window)
  attn_freq = np.fft.fft2(attn_windowed)
  power_spectrum = np.abs(attn_freq)**2

  # Compute effective bandwidth (inverse participation ratio)
  G = (np.sum(power_spectrum)**2) / np.sum(power_spectrum**2)
  ```
- Also compute `overall`, `early`, `mid`, and `late` group filters plus prompt-only controls
- Plot: average power spectrum for all available levels, matched wordy-control bandwidth comparisons, and prompt-control divergences
- Compute: G(t) for each granularity level and layer group
- Statistical test: primary `L1-L4` bandwidth trend plus continuous fixed-effects and horse-race regressions, with `L5-L8` as prompt-load controls

**Step 2.3: Experiment 3 -- Pre/Post Fusion Drift** (Week 6-7)
- For configured samples × available prompts × configured perturbations:
  - Extract pre-fusion vision features once per perturbation
  - Compute band-wise `ΔV(ω)` from the vision encoder
  - Extract post-fusion hidden states from a late language-conditioned decoder layer
  - Measure post-fusion drift as an all-token scalar `ΔZ_all`
  - Group perturbations by similar normalized `ΔV(ω)` profiles
  - Correlate `ΔZ_all` with `∫ W_t(ω) · ΔV(ω) dω` for `overall`, `early`, `mid`, and `late`
- Plot: controlled overlap-vs-response correlations, profile-group response heatmaps, sample-level `ΔV(ω)` profiles, complexity regressions, and wordy-control internal-drift comparisons
- Statistical test: Pearson correlation between controlled post-fusion response and spectral overlap for each group, with `late` as the primary post-fusion test, plus continuous response regressions

**Step 2.4: Experiment 4 -- Synthetic Frequency Ablation** (Week 7-8)
- For 500 images, create frequency-swept versions:
  ```python
  for omega_c in np.linspace(0.01, 0.5, 20):  # normalized frequency
      # Low-pass: keep only below omega_c
      I_low = apply_ideal_lowpass(image, omega_c)
      # High-pass: keep only above omega_c
      I_high = apply_ideal_highpass(image, omega_c)

      # Evaluate at each granularity
      for level in [L1, L2, L3, L4]:
          acc = evaluate(model, I_low, prompt[level])
          record(omega_c, level, acc)
  ```
- Plot: accuracy vs cutoff frequency for each granularity level
- Measure: critical cutoff ω_c* (accuracy = 50%) for each level
- Expected: ω_c*(L1) < ω_c*(L2) < ω_c*(L3) < ω_c*(L4)

### Phase 3: Validation and Theory (Weeks 9-12)

**Step 3.1: Experiment 5 -- Overlap Prediction** (Week 9)
- Combine `W_t(ω)` from Experiment 2 with image-space and vision-feature perturbation spectra from Experiment 1
- Run both raw and relative-normalized overlap calculations
- Compute predicted sensitivity for `overall`, `early`, `mid`, and `late` filter groups
- Correlate the predictions with `loglik_volatility` as the primary target and with `accuracy_drop`, `loglik_erosion`, `net_drop`, and grouped `relative_accuracy_drop` as controls
- Treat the `late` branch as the main layer-group test and the others as controls
- Target: grouped Pearson `r > 0.7` on the primary branch, with consistent positive controls

**Step 3.2: Generalization Check** (Week 10)
- Repeat Experiment 1 (task granularity) on:
  - `llava-hf/llava-v1.6-mistral-7b-hf` (different architecture)
  - `OpenGVLab/InternVL3-8B-hf` (different vision-language stack)
- Verify same monotonic granularity-sensitivity pattern
- Repeat Experiment 2 (attention frequency) on LLaVA-v1.6
- Verify same frequency-selective attention structure

**Step 3.3: Experiment 6 -- Segmentation Extension** (Week 10-11)
- Run SAM3 on PartImageNet at 3 granularity levels
- Apply full perturbation suite
- Compare to SAM2 (vision-only) baseline
- Verify granularity effect appears in SAM3 but not SAM2

**Step 3.4: Theoretical Write-up** (Week 11-12)
- Formalize Theorems 1, 2, 3 with proofs
- Connect empirical W_t measurements to theoretical predictions
- Derive alignment tax bound from data
- Write proof sketches for all main claims

### Phase 4: Mitigation Prototype (Weeks 13-15)

**Step 4.1: Frequency-Aware Augmentation**
- During fine-tuning, augment specifically in bands where W_t is large
- Compare to random augmentation baseline
- Measure robustness improvement

**Step 4.2: Spectral Regularization**
- Add loss term: penalize large ||∂Z/∂F(I)(ω)|| in attended frequency bands
- Train for 1 epoch of fine-tuning on COCO
- Measure robustness improvement

**Step 4.3: Multi-Scale Grounding**
- Decompose image into frequency sub-bands
- Run grounding independently on each sub-band
- Ensemble the results
- Measure robustness improvement

### Phase 5: Paper Writing (Weeks 16-18)

Structure:
1. Introduction: Paper 1 showed what → this paper explains why and predicts when
2. Theory: Frequency filter formalism, Theorems 1-3
3. Experiments 1-6
4. Mitigation results
5. Discussion: implications for VLM deployment, architecture design
6. Conclusion

---

## 7. EXPECTED RESULTS AND WHAT TO LOOK FOR

### Key Plots to Generate

| Plot | X-axis | Y-axis | Lines/Bars | Expected Pattern |
|------|--------|--------|------------|-----------------|
| **Granularity scaling** | Perturbation severity | Accuracy/mIoU drop | One line per primary level, controls shown separately | Primary lines fan out: L4 drops most, L1 drops least |
| **Wordy-control comparison** | Matched pair | Drop / drift / bandwidth / cutoff | Base vs wordy bars | Wordy controls reveal prompt-load anchoring independent of semantic logic |
| **Attention power spectrum** | Spatial frequency ω | Power |Â|² | One line per available level | L1 concentrated at low ω, L4 spread across all ω; controls test prompt load |
| **Prompt-control divergence** | Granularity level | JS(W_task, W_control) | Empty vs random control | Divergence grows with task specificity |
| **Effective bandwidth** | Granularity level | G(t) | Bar chart, split by overall/early/mid/late | Monotonically increasing |
| **Coefficient / forest plots** | Standardized beta | Predictor | Pooled and within-image dots with CIs | Residualized semantics dominates prompt/load controls |
| **Linguistic stabilization** | Matched mirror pair | Base-minus-wordy drop/drift | One bar per pair | Positive values show wordy prompts damp fragility |
| **Perturbation-specific horse races** | Standardized beta | Predictor | One subplot per perturbation type | Shows which perturbations follow the theory most closely |
| **Directional confidence plots** | Semantic complexity | Drift / erosion / recovery / volatility | Scatter by level | Semantics increases volatility; prompt load can damp movement |
| **Controlled post-fusion response** | Filter group / profile group | Correlation or mean ΔZ_all | Coarse vs Fine prompt | Higher overlap predicts larger task-conditioned post-fusion response |
| **Frequency threshold** | Cutoff frequency ω_c | Accuracy | One curve per granularity | Critical ω_c shifts right with granularity |
| **Overlap prediction** | Predicted sensitivity | Actual sensitivity | Scatter plot by overall/early/mid/late and raw/relative branches | Strongest linear correlation on the primary late branch |
| **Predicted vs observed by level and perturbation** | Perturbation type | Predicted / observed bars | One panel per level | Reveals where overlap prediction matches or misses each perturbation |
| **Segmentation granularity** | Severity | mIoU drop | Whole object vs Part vs Subpart | Same fan-out pattern |

### Key Numbers to Report

| Metric | What it proves | Target value |
|--------|---------------|-------------|
| Spearman ρ(granularity, sensitivity) | H1c: monotonic scaling | ρ > 0.8, p < 0.001 |
| Fixed-effects slope of bandwidth or drift on residualized semantic complexity | H1a/H1e: frequency-selective attention after image and prompt-load controls | Positive slope, CI excluding 0 |
| Horse-race beta for residualized semantics vs prompt load, option hardness, and prediction entropy | H1e: semantic logic dominates prompt length, MCQ difficulty, and clean uncertainty | Semantic beta largest and stable across perturbations |
| Correlation(ΔZ_all, ∫W_t·ΔV) per prompt and layer group | H1b: task-conditioned internal response follows the filter | r > 0.6 |
| Pearson r(predicted, actual sensitivity) on primary late branch | H1d: spectral overlap prediction | r > 0.7 |
| ω_c*(L4) / ω_c*(L1) ratio | Bandwidth scales with granularity | Ratio > 2.0 |
| Robustness improvement from mitigation | Practical value | > 15% reduction in worst-case drop |

### Red Flags (What Would Weaken the Hypothesis)

| Observation | What it means | How to handle |
|-------------|---------------|---------------|
| No monotonic granularity scaling | Coupling tightness isn't the right abstraction | Investigate per-perturbation; may hold for some but not all |
| Attention maps show no frequency structure | Cross-attention operates differently than hypothesized | Check if frequency selectivity emerges in deeper layers |
| Overlap prediction correlation is weak (r < 0.4) | Linear filter model is too simple | Try nonlinear filter models or layer-specific analysis |
| Effect disappears in larger models | Scale overcomes the vulnerability | Important finding in itself; report as "scale mitigates" |
| SAM2 also shows granularity effect | It's not about language conditioning | Check if box prompt size correlates with granularity |

---

## 8. IMPACT ASSESSMENT

### If Results Confirm the Hypothesis

**Scientific impact: HIGH**
- First mechanistic theory of VLM robustness
- First predictive framework (not just post-hoc analysis)
- Unifies frequency robustness, task granularity, and cross-modal alignment
- Theoretical contributions (Theorems 1-3) provide formal foundations

**Practical impact: HIGH**
- Deployment risk assessment: predict vulnerabilities before deployment
- Targeted hardening: frequency-aware augmentation
- Prompt engineering: reformulate fragile fine-grained queries as coarse + reasoning
- Architecture design: frequency-decoupled attention

**Venue potential:**
- Top-tier: NeurIPS, ICML, ICLR (theory + experiments)
- CV-specific: CVPR, ECCV (segmentation extension)
- Journal: TPAMI, IJCV (comprehensive version)

**Follow-up papers enabled:**
1. Frequency-aware training for robust VLMs
2. Certified robustness bounds for VLMs
3. Task-adaptive frequency routing architectures
4. Alignment-robustness tradeoff formalization
5. Extension to video, audio, and other modalities

### If Results Partially Confirm

The hypothesis may hold for some perturbation types but not others, or for some architectures but not all. This is still publishable:
- "For which perturbation classes does the frequency filter model hold?"
- "Architecture-dependent frequency coupling in VLMs"

### If Results Refute

Even negative results are valuable:
- "Task granularity does NOT predict robustness" would challenge the community's implicit assumptions
- Would redirect the field toward alternative explanations
- Publishable at a good workshop or as a shorter findings paper

---

## 9. COMPUTATIONAL REQUIREMENTS

| Experiment | GPU Hours (est.) | Storage |
|------------|-----------------|---------|
| Exp 1: Granularity spectrum (4 models) | ~200 hours | ~50 GB |
| Exp 2: Attention extraction | ~40 hours | ~100 GB (attention maps) |
| Exp 3: Pre/post fusion drift | ~60 hours | ~80 GB |
| Exp 4: Frequency ablation | ~80 hours | ~30 GB |
| Exp 5: Overlap prediction | ~10 hours (compute only) | ~5 GB |
| Exp 6: Segmentation extension | ~100 hours | ~40 GB |
| Mitigation experiments | ~150 hours | ~50 GB |
| **Total** | **~640 GPU hours** | **~355 GB** |

Recommended hardware: 4× A100 80GB or equivalent; ~1 week wall-clock time for all experiments in parallel.

---

## 10. TIMELINE SUMMARY

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| Foundation | 1-3 | Data, code, model setup, sanity checks |
| Core experiments | 4-8 | Experiments 1-4 complete with plots |
| Validation + theory | 9-12 | Experiments 5-6, generalization, proofs |
| Mitigation | 13-15 | Prototype solutions with measurements |
| Paper writing | 16-18 | Complete manuscript |
| **Total** | **18 weeks** | **Submission-ready paper** |

---

## 11. SUGGESTED PAPER TITLE

Options:
1. "The Frequency Filter Hypothesis: Why Language Conditioning Breaks Under Perturbation"
2. "Task Granularity Predicts VLM Fragility: A Spectral Theory of Cross-Modal Alignment"
3. "Same Filter, Different Drift: How Language Conditioning Creates Frequency-Selective Vulnerabilities in Vision-Language Models"
4. "The Alignment Tax: A Frequency-Domain Theory of Vision-Language Model Robustness"

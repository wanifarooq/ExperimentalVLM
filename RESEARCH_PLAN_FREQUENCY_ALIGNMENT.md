# Research Plan: Language Conditioning as a Learned Frequency Filter in Vision-Language Models

## Building on: "Same Answer, Different Representations" (Paper 1)

---

## 1. HYPOTHESIS

### Primary Hypothesis (H1)
**Language conditioning in vision-language models learns a task-specific frequency filter W_t that binds text concepts to specific spectral signatures in vision features. The sensitivity of this binding to perturbations scales monotonically with the spectral support required by the task -- coarse tasks (object detection, segmentation) require narrow low-frequency support and are robust, while fine-grained tasks (attribute recognition, spatial reasoning, MCQ) require broad cross-frequency support and are fragile.**

### Sub-Hypotheses

**H1a (Frequency-Selective Attention):** Cross-attention maps from language tokens to vision patches exhibit frequency-selective structure -- coarse concept prompts ("a bicycle") concentrate attention energy in low spatial frequencies, while fine-grained prompts ("the red spoke on the front wheel") distribute attention across higher frequency bands.

**H1b (Drift Amplification):** The language-vision fusion layer amplifies representation drift in frequency bands that the text token attends to and suppresses drift in unattended bands. The ratio of post-fusion to pre-fusion drift per frequency band correlates with cross-attention energy in that band.

**H1c (Granularity Scaling):** For a fixed perturbation type and severity, the performance degradation increases monotonically with task granularity, measured as the bandwidth of the effective frequency filter W_t.

**H1d (Perturbation-Filter Overlap):** The magnitude of performance degradation for a given (task, perturbation) pair is predicted by the spectral overlap between the perturbation's energy injection pattern and the task's frequency filter W_t. High overlap → high degradation, low overlap → robustness.

### Connection to Paper 1
Paper 1 established:
- All perturbations (spatial and adversarial) manifest as phase/frequency misalignment at the representation level
- Cross-frequency sensitivity (H2), not low-frequency dominance (H1), explains VLM fragility
- Output stability masks representation drift (hidden drift phenomenon)
- Language conditioning is the vulnerability surface

This follow-up explains **why** language conditioning is the vulnerability surface (it's a learned frequency filter) and **when** it breaks (when perturbations corrupt the specific bands the filter depends on).

---

## 2. THEORETICAL FRAMEWORK

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

**Claim**: For coarse concepts, |Â_j|² is concentrated at low (ω_x, ω_y). For fine concepts, |Â_j|² has significant energy at higher frequencies.

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

**Expected Result**: Monotonic increase in all degradation metrics as granularity increases, for every perturbation type.

**Dataset**: COCO + VQAv2 annotations (provides images with objects, attributes, spatial relationships)
**Sample size**: 1000 images × 4 granularity levels × 51 perturbations = ~204,000 evaluations

---

### Experiment 2: Cross-Attention Frequency Analysis
**Goal**: Show that cross-attention maps have frequency-selective structure correlated with task granularity.

**Design**:
1. For each (image, text prompt) pair, extract cross-attention weights A from every layer
2. Reshape attention to spatial grid: a_j(x, y)
3. Compute 2D FFT: Â_j(ω_x, ω_y)
4. Compute power spectrum: |Â_j|²
5. Compute effective bandwidth G(t) using inverse participation ratio
6. Average across layers and images

**Analysis**:
- Plot mean power spectrum |Â_j|² for L1 vs L2 vs L3 vs L4 prompts
- Compute G(t) for each granularity level
- Correlate G(t) with sensitivity from Experiment 1

**Expected Result**:
- L1 prompts: power concentrated at DC and low frequencies
- L4 prompts: power spread across frequency spectrum
- Strong positive correlation between G(t) and sensitivity

**Models for extraction**: Qwen2.5-VL-7B (same as Paper 1), LLaVA-1.5-13B (for generalization)

---

### Experiment 3: Pre-Fusion vs Post-Fusion Drift
**Goal**: Show that language conditioning amplifies drift in attended bands and suppresses it in unattended bands.

**Design**:
1. Extract vision features BEFORE cross-attention (pre-fusion): V
2. Extract features AFTER cross-attention (post-fusion): Z
3. For clean and perturbed image, compute:
   - Pre-fusion drift: ΔV = V_perturbed - V_clean
   - Post-fusion drift: ΔZ = Z_perturbed - Z_clean
4. Decompose both into frequency bands (apply band-pass filters to the spatial arrangement of patches)
5. Compute drift ratio per band: R(ω) = ||ΔZ(ω)|| / ||ΔV(ω)||

**Expected Result**:
- R(ω) > 1 in bands where W_t(ω) is large (amplification in attended bands)
- R(ω) < 1 in bands where W_t(ω) is small (suppression in unattended bands)
- The amplification pattern differs between coarse and fine prompts

**Models**: Qwen2.5-VL-7B (has accessible intermediate representations)

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

**Dataset**: 500 COCO images with multi-granularity annotations

---

### Experiment 5: Perturbation-Filter Overlap Prediction
**Goal**: Show that Theorem 1 (sensitivity = spectral overlap) quantitatively predicts observed degradation.

**Design**:
1. From Experiment 2, measure W_t(ω) for each (image, prompt) pair
2. For each perturbation type, compute ΔF(ω) (the spectral signature of the perturbation)
3. Compute predicted sensitivity: S_pred = ∫ |W_t(ω)|² · |ΔF(ω)|² dω
4. From Experiment 1, measure actual sensitivity: S_actual = accuracy/mIoU drop
5. Compute correlation between S_pred and S_actual

**Expected Result**:
- Strong positive correlation (r > 0.7) between predicted and actual sensitivity
- This validates the theoretical framework

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
| **Qwen2.5-VL-7B** | Main VLM for VQA experiments | Continuity with Paper 1 |
| **Qwen2.5-VL-72B** | Scaling analysis | Tests if the phenomenon persists at scale |
| **LLaVA-1.5-13B** | Generalization check | Different architecture (linear projection vs Q-Former) |
| **InternVL2-8B** | Generalization check | Different vision encoder (InternViT) |
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
- **Minimum viable**: Qwen2.5-VL-7B + LLaVA-1.5-13B + SAM2 + SAM3 (4 models, matches Paper 1 scope)
- **Recommended**: Add InternVL2-8B + Qwen2.5-VL-72B (6 models, strong generalization claim)
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
- Build part-level segmentation manifests from PartImageNet:
  - L1: Whole object mask
  - L2: Part mask (head, body, legs)
  - L3: Subpart mask (eye, ear, paw)
- Target: 1000 images with all 4 granularity levels for VQA, 500 images with 3 levels for segmentation

**Step 1.2: Perturbation Engine**
- Reuse perturbation code from Paper 1 (vlm_invariance_check.py)
- Ensure all perturbations also output ΔF(ω) -- the spectral signature of each perturbation
- Add frequency sweep perturbation: progressive low-pass/high-pass with 20 cutoff values
- Validate against Paper 1 results on SEEDBench (sanity check)

**Step 1.3: Model Setup**
- Set up Qwen2.5-VL-7B with intermediate layer extraction hooks
- Set up LLaVA-1.5-13B with same hooks
- Set up SAM2 + SAM3/GroundingDINO (reuse segmentation_robustness code)
- Validate all models produce expected baseline accuracy on clean data

### Phase 2: Core Experiments (Weeks 4-8)

**Step 2.1: Experiment 1 -- Task Granularity Spectrum** (Week 4-5)
- Run all perturbations × all severity levels × all granularity levels on:
  - Qwen2.5-VL-7B with GQA questions (L1-L4)
  - SAM3 with PartImageNet (L1-L3)
- Record: accuracy/mIoU, representation drift (cosine sim), Dirichlet energy
- Compute: degradation curves (metric vs severity) for each granularity level
- Statistical test: Spearman correlation between granularity level and degradation magnitude
- Target output: 4 degradation curves (one per granularity) for each perturbation type

**Step 2.2: Experiment 2 -- Cross-Attention Frequency Analysis** (Week 5-6)
- For 200 images × 4 prompts each, extract cross-attention maps from Qwen2.5-VL-7B
- Implementation:
  ```python
  # Hook into cross-attention layers
  def attention_hook(module, input, output):
      attention_weights = output[1]  # [batch, heads, lang_tokens, vis_patches]
      store_attention(attention_weights)

  # For each (image, prompt):
  for layer in cross_attention_layers:
      layer.register_forward_hook(attention_hook)

  # Reshape attention to spatial grid
  attn_spatial = attention.reshape(H_patches, W_patches)

  # 2D FFT
  attn_freq = np.fft.fft2(attn_spatial)
  power_spectrum = np.abs(attn_freq)**2

  # Compute effective bandwidth (inverse participation ratio)
  G = (np.sum(power_spectrum)**2) / np.sum(power_spectrum**2)
  ```
- Plot: average power spectrum for L1, L2, L3, L4 prompts
- Compute: G(t) for each granularity level
- Statistical test: one-way ANOVA on G(t) across granularity levels

**Step 2.3: Experiment 3 -- Pre/Post Fusion Drift** (Week 6-7)
- For 200 images × 4 prompts × 10 perturbations:
  - Extract pre-fusion vision features (before cross-attention)
  - Extract post-fusion features (after cross-attention)
  - Compute drift at each stage
  - Decompose drift into frequency bands (band-pass filter the patch grid)
  - Compute amplification ratio R(ω) = ||ΔZ(ω)|| / ||ΔV(ω)||
- Plot: R(ω) heatmaps for coarse vs fine prompts
- Statistical test: paired t-test on R(ω) in high-frequency bands between L1 and L4 prompts

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
- Combine W_t(ω) from Experiment 2 with ΔF(ω) from perturbation analysis
- Compute predicted sensitivity for each (task, perturbation) pair
- Correlate with actual sensitivity from Experiment 1
- Target: Pearson r > 0.7

**Step 3.2: Generalization Check** (Week 10)
- Repeat Experiment 1 (task granularity) on:
  - LLaVA-1.5-13B (different architecture)
  - InternVL2-8B (different vision encoder)
- Verify same monotonic granularity-sensitivity pattern
- Repeat Experiment 2 (attention frequency) on LLaVA-1.5-13B
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
| **Granularity scaling** | Perturbation severity | Accuracy/mIoU drop | One line per granularity level (L1-L4) | Lines fan out: L4 drops most, L1 drops least |
| **Attention power spectrum** | Spatial frequency ω | Power |Â|² | One line per granularity level | L1 concentrated at low ω, L4 spread across all ω |
| **Effective bandwidth** | Granularity level | G(t) | Bar chart | Monotonically increasing |
| **Drift amplification** | Frequency band ω | Ratio R(ω) | Coarse vs Fine prompt | Fine prompt amplifies more in high-freq bands |
| **Frequency threshold** | Cutoff frequency ω_c | Accuracy | One curve per granularity | Critical ω_c shifts right with granularity |
| **Overlap prediction** | Predicted sensitivity | Actual sensitivity | Scatter plot | Strong linear correlation (r > 0.7) |
| **Segmentation granularity** | Severity | mIoU drop | Whole object vs Part vs Subpart | Same fan-out pattern |

### Key Numbers to Report

| Metric | What it proves | Target value |
|--------|---------------|-------------|
| Spearman ρ(granularity, sensitivity) | H1c: monotonic scaling | ρ > 0.8, p < 0.001 |
| ANOVA F-statistic on G(t) across levels | H1a: frequency-selective attention | F > 10, p < 0.001 |
| Correlation(R(ω), W_t(ω)) per prompt | H1b: drift amplification in attended bands | r > 0.6 |
| Pearson r(predicted, actual sensitivity) | H1d: spectral overlap prediction | r > 0.7 |
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

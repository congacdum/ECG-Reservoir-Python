# PROJECT KNOWLEDGE — LOCKED PROJECT CONTRACT

> **Purpose:** This file is the compact source of truth for humans and coding agents. Read this file and `CHANGE_LOG.md` before scanning the repository. Do not silently change locked decisions.

## 1. Project goal

Build a **hardware-oriented Reservoir Computing / Liquid State Machine (LSM)** for **binary classification of 20-point ECG signal windows**, with the final goal of **FPGA inference**.

Priority order:
1. Correctness and reproducibility.
2. No data leakage / no fake test result.
3. No bias anywhere in the learnable/inference path.
4. High classification quality.
5. FPGA suitability.
6. Low resource usage and power.
7. Low latency.

## 2. Problem definition

- Input: exactly **20 sequential ECG values** per sample.
- Output: binary class `0` or `1`.
- Current safe semantic interpretation:
  - `1`: target waveform / single-peak target class.
  - `0`: non-target / hard-negative class.
- **Unknown:** exact clinical/medical meaning of label `1`.
- Therefore DO NOT rename class `1` to QRS, arrhythmia, disease, abnormal beat, etc. until upstream documentation confirms it.

## 3. Dataset facts verified from the supplied CSV

Source file:
`data/raw/training_data_20_points_with_single_peak_hardneg.csv`

Verified structure:
- 17,214 rows.
- 9,094 unique `original_index` groups.
- Columns:
  - `value_sequence`
  - `finalLabel`
  - `original_index`
  - `original_len`
  - `generated_window_id`
  - `new_len`
- Every parsed model input has exactly 20 points.
- `new_len == 20` for all rows.
- Each `original_index` has exactly one label.

Only `value_sequence` may be used as model input.

Never use as predictive features:
- `finalLabel`
- `original_index`
- `original_len`
- `generated_window_id`
- `new_len`

## 4. Leakage policy — CRITICAL

The dataset contains multiple generated windows from the same `original_index`.

Therefore:
- Split by **`original_index`**, never by individual row.
- All windows from one `original_index` must stay in exactly one split.
- Exact waveform overlap between train/validation/test must be checked.

Locked deterministic split (`random_state=42`):
- Train: 6,365 groups / 12,059 rows.
- Validation: 1,364 groups / 2,594 rows.
- Test: 1,365 groups / 2,561 rows.

The test set is sealed.

Test data MUST NOT be used for:
- seed selection
- hyperparameter selection
- reservoir size selection
- quantization decisions
- architecture decisions
- threshold tuning

`05_test_final.py` is the intended final-test entry point and refuses repeat evaluation by default when final metrics already exist.

### Current limitation
No `patient_id` or `record_id` exists in the supplied file. Therefore patient-independent splitting cannot be verified. Group-wise `original_index` splitting is the strongest leakage protection currently possible.

## 5. Locked model architecture

### Reservoir
- 64 logical LIF neurons.
- Fixed sparse recurrent reservoir.
- Recurrent connectivity: approximately 10%.
- Seed: 42 for locked run.
- Input weights: `Win ∈ {0.5, 1, 2}`.
- Recurrent weights: `Wres ∈ {-1, 0, +1}`.
- No trainable reservoir weights.
- No additive bias current.
- Leak: `7/8`.
- Input gain: `3/2`.
- Recurrent gain: `1/16`.
- Threshold: `2.5`.
- Reset: `0` after spike.
- Reservoir state used by the classifier: **spike count per neuron over 20 timesteps**.

### Readout
- One linear output.
- Ridge regression.
- `fit_intercept=False`.
- Labels during readout training: `0 -> -1`, `1 -> +1`.
- Prediction rule: `score >= 0 -> class 1`, else class 0.
- Class-balanced sample weights are used for readout fitting.
- Final hardware readout target: signed INT8 weights.

## 6. No-bias requirement

Confirmed hardware constraint: **bias is not supported**.

Forbidden:
- trainable `+b` term
- `fit_intercept=True`
- constant trainable neuron bias current (`I_bias`)

Current model uses:
`score = Wout * state`

not:
`score = Wout * state + b`

## 7. FPGA-oriented arithmetic

Current golden hardware model:
- ECG input: signed 12-bit fixed point, 10 fractional bits.
- Membrane: signed 16-bit, 10 fractional bits.
- Spike: 1 bit.
- Spike count: 5 bits sufficient for range 0..20.
- Recurrent weights: ternary.
- Readout weights: INT8.
- Accumulator: 20-bit signed target.

Multiplier-light reservoir:
- `Win=0.5,1,2` with input gain `3/2` maps to shift/add-friendly gains `3/4, 3/2, 3`.
- Leak `7/8`: `v - (v >> 3)`.
- Recurrent gain `1/16`: shift by 4.
- Ternary recurrence: add / subtract / skip.

Target FPGA microarchitecture:
- 64 logical neurons.
- 1 time-multiplexed physical LIF processing element as the initial resource-minimal target.
- Sparse recurrent edge list.
- 64 spike counters.
- Shared linear readout MAC.
- Offline training only; FPGA performs inference only.

`src/hardware/hardware_model.py` is the bit-exact behavioral reference for RTL. Golden-vector generation is implemented in `src/hardware/golden_vectors.py` and `scripts/07_generate_rtl_vectors.py`; Brian2 must not generate FPGA vectors. RTL Phase 1 LIF PE simulation is verified with MSYS2 UCRT64 Icarus Verilog 13.0: 10,246 timestep vectors and 40,984 state fields matched with zero mismatches.

RTL Phase 2 is also experimentally verified with Icarus 13.0. The corrected logical sparse edge export contains 403 source→destination edges, 9.838867% density, destination fan-in 2–12 with mean 6.296875, 311 positive edges, 92 negative edges, no self-edges, and no duplicates. The maximum ternary incoming sum is therefore bounded by −12…+12, so the internal signed edge accumulator uses 5 bits (−16…+15). The sparse engine matched 132 directed vectors and 10,240 real golden neuron vectors exactly; the one-timestep wrapper matched 71,680 comparisons exactly. RTL integration beyond one timestep, synthesis, and board deployment remain pending.

RTL Phase 3 is experimentally verified with the same Icarus 13.0 simulator. `rtl/src/reservoir_controller.sv` uses current/next membrane arrays, current/next spike vectors, and current/next 5-bit counters. The exact timestep order is input[t] plus committed state from t−1, recurrent contribution, LIF update, spike/reset, counter update, then a full commit after logical neuron 63. The controller matched 14 complete train/validation/synthetic segments over 280 checkpoints with zero mismatches.

RTL Phase 4 is experimentally verified with Icarus 13.0. `rtl/src/readout_mac.sv` scans the 64 final spike counts and reads `outputs/fpga/weights/w_out_int8.mem` as signed INT8 two's-complement values. The theoretical legal dot-product bounds are −163,840…+162,560 for counts 0…20 and INT8 weights −128…+127; signed 19 bits would contain that interval, while the locked 20-bit signed accumulator provides one additional safety bit. The MAC applies explicit signed saturation and the no-bias class rule `score >= 0 -> class 1`. Fourteen standalone vectors produced 896 exact product/accumulator checkpoints, including zero-score class 1. `rtl/src/ecg_classifier_core.sv` bridges the verified controller to the MAC after the final count commit. Eight train/validation golden samples passed end-to-end with exact scores/classes, sequential reset isolation, and 1,386 cycles from start acceptance to classifier done (1,320 reservoir cycles, two bridge edges, 64 MAC cycles). Synthesis, board timing, resources, power, and deployment remain pending.

Do not assume final FPGA device, clock, DSP count, BRAM limit, or signed-weight restriction until hardware requirements are supplied.

## 8. Evaluation metrics

Primary metric:
- Balanced Accuracy.

Always report when available:
- Accuracy
- Balanced Accuracy
- Precision
- Sensitivity / Recall
- Specificity
- F1
- MCC
- ROC-AUC
- PR-AUC
- Confusion matrix

## 9. Locked run status (`outputs/runs/run_001`)

The packaged run was built with the deterministic split and locked architecture.

Validation, fixed-point + INT8 readout:
- Balanced Accuracy: **0.9614098226**
- Accuracy: **0.9653045490**
- F1: **0.9510337323**

Final sealed test, fixed-point + INT8 readout:
- Balanced Accuracy: **0.9679954626**
- Accuracy: **0.9703240922**
- Sensitivity: **0.9607843137**
- Specificity: **0.9752066116**
- Precision: **0.9520000000**
- F1: **0.9563719862**
- MCC: **0.9339101129**
- ROC-AUC: **0.9933389982**
- PR-AUC: **0.9785456699**
- Confusion matrix: `[[1652, 42], [34, 833]]`

These numbers are measured on the supplied dataset; they are not clinical-generalization claims.

## 10. Brian2 status

Brian2 is required by the assignment direction and an optional reference backend exists at:
`src/model/reservoir_brian2.py`

Brian2 was installed and executed in a network-enabled Python 3.13 environment on 2026-09-19. The runtime audit passed:
- import and minimal Brian2 construction;
- one actual ECG sample and multiple actual ECG samples;
- independent-sample reset isolation;
- comparison against the NumPy reference.

The audit corrected two implementation discrepancies: the input equation now scales the continuous-time term so one Euler step applies the intended direct input current, and recurrent edges map `w_res[destination, source]` to source→destination as in NumPy/hardware. A controlled recurrent case and actual ECG rows with recurrence disabled match NumPy exactly. On two actual ECG rows with recurrence enabled, 100/128 spike-count elements matched; remaining sparse differences are from Brian2 event-driven floating-point recurrent accumulation semantics. Brian2 is therefore a runnable exploratory reference, but it is **not bit-equivalent** and is not the FPGA golden model.

The full train/validation Brian2 run is saved at `outputs/runs/run_brian2_20260919`. A no-bias Ridge readout trained on Brian2 train states achieved validation Balanced Accuracy `0.9530366169`; train and validation non-zero spike-count rates were `0.9991111` and `0.9985965`. The final test was not used.

The tested executable reference paths in this repository are:
1. NumPy float LIF reservoir.
2. Integer/fixed-point hardware golden model.
3. Brian2 exploratory reference (runtime-tested on full train/validation, dynamics not fully equivalent).

## 11. Do-not-expand rules

Do NOT add the following unless current architecture fails or real synthesis evidence justifies it:
- CNN
- Transformer
- STDP
- trainable reservoir
- backpropagation through reservoir
- multiple spike encoders
- PCA / wavelet preprocessing
- NAS / genetic algorithm / PSO reservoir search
- clinical interpretation unsupported by dataset metadata

The project goal is a robust FPGA-ready classifier, not an open-ended architecture research project.

## 12. Expected development flow

1. Inspect `PROJECT_KNOWLEDGE.md`.
2. Inspect `CHANGE_LOG.md`.
3. Run tests.
4. Change the smallest relevant module.
5. Run tests again.
6. Use train/validation only for development.
7. Do not inspect final test metrics to tune new changes.
8. Build/validate bit-exact fixed-point behavior.
9. Generate/load golden vectors with `scripts/07_generate_rtl_vectors.py`.
10. Implement RTL.
11. Compare RTL sample-by-sample against Python hardware golden model.
12. Synthesize in Vivado and report measured LUT/FF/BRAM/DSP/Fmax/power.

## 13. Published repository

The verified project was published to the official remote repository:

`https://github.com/congacdum/ECG-Reservoir-Python`

The published branch is `main`. The initial publication commit is `04d6a49d50cd3b08905ba8fbd0e9447e0f2176ca`.

## 14. Scientific analysis audit — 2026-09-21

A train/validation-only analysis pass was completed without changing the locked model or RTL.

Measured evidence:
- RAW20 linear baselines: validation BA 0.828203–0.837068.
- Decision Tree depth 5 on RAW20: validation BA 0.994166.
- HistGradientBoosting: validation BA 0.997289 on RAW20 and 0.998915 on RAW20_DIFF19.
- Recurrence ablation with retrained readout: Wres ON 0.958323 versus Wres=0 0.674066.
- Locked spike-count feature SVD: 11/24/50 components for 90%/95%/99% variance.
- Ten-seed validation BA: mean 0.962125, standard deviation 0.007437, range 0.953108–0.973564.
- Robustness evidence shows sensitivity to DC offset and gain scaling; small Gaussian noise is less damaging.
- Expanded integrated RTL verification passed 64 validation-only samples with zero score/class mismatches.

These results are evidence reports, not a model-v2 selection. The locked architecture, seed, graph, golden model and RTL remain unchanged. Final test was not used.

## 15. CI reproducibility audit — 2026-09-21

The pre-fix workflow used Python 3.11 and unconstrained requirements such as numpy>=1.26 and brian2>=2.7. The reported CI traceback is consistent with Brian2 releases 2.7.1–2.9.0 accessing np.ndarray.ptp, an API removed from NumPy 2.x. Wheel source inspection confirmed that Brian2 2.10.1 adds the compatibility guard and requires Python >=3.12.

The verified CI environment is now:

- Python 3.13
- NumPy 2.2.6
- Brian2 2.10.1
- pandas 2.3.2
- SciPy 1.16.3
- scikit-learn 1.7.2
- pytest 9.1.1

Clean-environment verification passed import smoke, pip check, compileall and 30 Python tests. No model, numerical result, RTL behavior, golden artifact or final test was changed or rerun.

Push/PR CI runs Python verification and deterministic RTL smoke. Full RTL regression is preserved in a separate manual/nightly workflow.

## 16. Icarus simulator compatibility audit — 2026-09-22

The same repository source, recurrent smoke runner, and smoke workload were tested with both simulator versions.

- Local Windows: Icarus Verilog 13.0 stable (v13_0), VVP 13.0; recurrent smoke passed. N=1 passed 7 comparisons and N=8 passed 56 comparisons. Representative local compile/simulation times were about 0.12–0.15 s / 0.04 s.
- WSL Ubuntu: apt Icarus Verilog 12.0 (iverilog 12.0-3); N=1 and N=8 compiled in about 0.03 s but VVP did not complete within 15 s. The same runner smoke reached its 60 s timeout in tb_reservoir_step.vvp golden simulation.
- Icarus 12 vvp -v showed only two simulation time steps and 102,860 scheduler events in about 10 s, with no PASS output. This is diagnostic evidence of a version-specific scheduler/runtime pathology.
- The official Icarus 13.0 tag was independently verified as commit 30a7d1a11b7586aa0fc868e509f04f514effc0ad. Production workflows build that exact tag and log both iverilog -V and vvp -V.

Conclusion: the A/B result supports an Icarus 12-specific runtime pathology for this unchanged workload; it does not justify redesigning RTL. Production CI is pinned to exact Icarus 13.0, while .github/workflows/rtl-simulator-compat.yml preserves a temporary 12/13 diagnostic matrix. RTL, model, graph, fixed-point behavior, golden vectors, metrics and scientific behavior were unchanged. The final ML test was not run.
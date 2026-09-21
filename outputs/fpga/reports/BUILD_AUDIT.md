# BUILD AUDIT

This report records what was actually executed during the project build.

## Environment
- Python: 3.13.5 in the build container.
- Available and executed: NumPy, pandas, scikit-learn, matplotlib, PyYAML, pytest.
- Brian2: not preinstalled.

## Brian2 installation attempt
A real `pip install brian2` command was executed.
Result: failed because the build environment could not resolve the external package host (`Temporary failure in name resolution`).

Therefore Brian2 execution is NOT claimed.

## Data audit executed
- Raw CSV loaded successfully.
- Shape verified: 17,214 x 6.
- 20 values parsed for every row.
- 9,094 unique original groups.
- Deterministic group-wise split generated.
- Cross-split group overlap check passed.
- Cross-split exact waveform hash overlap check passed.

## Leakage-safe split
- Train: 12,059 rows / 6,365 groups.
- Validation: 2,594 rows / 1,364 groups.
- Test: 2,561 rows / 1,365 groups.

## Model stages executed
1. NumPy float LIF reservoir + no-bias Ridge readout.
2. Integer fixed-point LIF golden model.
3. INT8 readout quantization.
4. Validation evaluation.
5. Final sealed-test evaluation after architecture lock.
6. FPGA weight/config export.

## Measured validation
Float reference BA: 0.9601935412.
Fixed-point + INT8 BA: 0.9614098226.

## Measured final test
- Accuracy: 0.9703240922
- Balanced Accuracy: 0.9679954626
- Precision: 0.9520000000
- Sensitivity: 0.9607843137
- Specificity: 0.9752066116
- F1: 0.9563719862
- MCC: 0.9339101129
- ROC-AUC: 0.9933389982
- PR-AUC: 0.9785456699
- Confusion matrix: [[1652, 42], [34, 833]]

## FPGA export executed
Generated:
- `w_in_codes.mem`
- `w_res_edges.csv`
- `w_out_int8.mem`
- `fpga_export_metadata.json`
- locked config snapshot

## Follow-up audit — 2026-09-19
- Installed Brian2 2.10.1 in a network-enabled Python 3.13 environment.
- Brian2 import, minimal construction, one actual ECG sample, multiple actual ECG samples, and reset isolation executed successfully.
- Brian2/NumPy comparison completed. On two actual training rows, 0/128 spike-count elements matched; Brian2 input scaling by `tau` is a known dynamics difference and Brian2 is not the FPGA golden model.
- Added fixed-point clipping, underflow, membrane/recurrent/readout saturation, and train/validation health diagnostics.
- Development-only diagnostics run `outputs/runs/run_diagnostics_20260919` used train/validation only.
- Current suite: 14 passed when Brian2 is available.

## Not executed / not claimed in the initial build
- RTL simulation.
- Vivado synthesis/place-and-route.
- FPGA resource utilization.
- FPGA Fmax.
- FPGA power measurement/estimation.
- Board deployment.

## Follow-up audit — Brian2 reconciliation and RTL vectors — 2026-09-19
- Corrected Brian2 direct-current input scaling and recurrent edge orientation.
- Added chunked independent sample groups; full train/validation execution completed.
- Brian2 no-bias readout validation Balanced Accuracy: `0.9530366169`.
- Controlled recurrent Brian2/NumPy case matched exactly; actual ECG recurrent comparison matched 100/128 spike-count elements on two train rows.
- Added `src/hardware/golden_vectors.py` and `scripts/07_generate_rtl_vectors.py`.
- Generated eight RTL vectors from four train and four validation samples only.
- Golden-vector model SHA-256: `fad34b75150bc80f2c43ed8ffb302c29f446993122b9cb8f877ad4d5c3305b11`.
- Golden-vector tests verify determinism, direct-model equality, legal ranges, no-bias manifest, and lossless reload.
- Current suite: **17 passed** with Brian2 available.

## Current not executed / not claimed
- RTL implementation.
- RTL simulation against the generated vectors.
- Vivado synthesis/place-and-route.
- FPGA resource utilization, Fmax, power, or board deployment.

## RTL Phase 1 audit — 2026-09-19
- Added `rtl/src/fixed_point_pkg.sv`, `rtl/src/lif_pe.sv`, `rtl/tb/tb_lif_pe.sv`, and `rtl/scripts/run_lif_tb.py`.
- The testbench consumes the exported golden-vector trace and adds Python-generated threshold/saturation boundary cases.
- Checked `iverilog`, `vvp`, Verilator, Questa/ModelSim: none were available.
- Direct `pip install iverilog` was attempted and returned no matching package.
- RTL simulation was **not executed or claimed**.
- Python regression: 19 passed, 1 simulator-dependent test skipped.

These are explicitly left for the next development stage.

## RTL Phase 1.1 audit — 2026-09-19
- WSL2 was checked but `wsl --status` and `wsl -l -v` returned `E_ACCESSDENIED` while enumerating distributions.
- MSYS2 was available; its UCRT64 Icarus package was installed as `mingw-w64-ucrt-x86_64-iverilog`.
- Icarus Verilog smoke test: `RTL_SIM_OK`, compile exit `0`, run exit `0`.
- Icarus version: `13.0 (stable) (v13_0)`.
- `python rtl/scripts/run_lif_tb.py`: PASS, 40,984 state comparisons across 10,246 timestep vectors, exit `0`.
- Complete regression: `20 passed`; `compileall` passed.
- No RTL mismatch occurred. RTL Phase 1 bit-exactness is verified for the exported train/validation vectors and directed threshold/saturation cases.

RTL integration, synthesis/place-and-route, resource/Fmax/power measurement, and board deployment remain pending.

## RTL Phase 2 audit — 2026-09-19
- Verified and corrected the exported edge-list orientation against `spikes @ w_res.T`; the frozen `w_res` matrix and model hash remain unchanged.
- Graph: 403 edges, 9.838867% density, fan-in 2–12, mean fan-in 6.296875, 311 positive, 92 negative, no self-edges or duplicates.
- Derived recurrent edge-sum range: `[-12, +12]`; selected 5-bit signed accumulator range `[-16, +15]`.
- `python rtl/scripts/run_recurrent_tb.py`: Icarus 13.0, 132 directed vectors and 10,240 real golden neuron vectors, zero recurrent mismatches.
- One-step integration: 71,680 exact comparisons, zero mismatches.
- Full Python suite: **23 passed**; compileall passed.

The full multi-timestep controller, readout MAC, synthesis, implementation, resource/Fmax/power measurement, and board deployment remain pending.

## RTL Phase 3 audit — 2026-09-19
- Added `rtl/src/reservoir_controller.sv` with explicit current/next state storage and a time-multiplexed 64-neuron FSM.
- Controller test sequence: 14 complete segments, including synthetic zero/low/mid/high/pulse activity, train samples, validation samples, and train-A/validation-B/train-A reset isolation.
- Checkpoints: 280 complete timesteps.
- Membrane-before comparisons: 17,920 exact.
- Membrane-after comparisons: 17,920 exact.
- Persistent membrane-reset comparisons: 17,920 exact.
- Input and recurrent comparisons: 17,920 exact each.
- Spike and spike-count comparisons: 17,920 exact each.
- Previous-spike-vector comparisons: 280 exact.
- Final 64-count vector comparisons: 896 exact.
- `python rtl/scripts/run_controller_tb.py`: Icarus Verilog 13.0, zero mismatches.
- Full Python suite: **26 passed**; compileall passed.

Readout integration, synthesis/place-and-route, resource/Fmax/power measurement, and board deployment remain pending.

## RTL Phase 4 audit — 2026-09-19
- Ran the required pre-change baseline: Python suite **26 passed**; Phase 1 LIF, Phase 2 recurrent, and Phase 3 controller regressions all remained zero-mismatch under Icarus Verilog 13.0.
- Added `rtl/src/readout_mac.sv` with one signed INT8 weight read per 64-neuron MAC term, explicit signed product/accumulator checkpoints, 20-bit saturation, and `score >= 0` classification.
- Added `rtl/src/ecg_classifier_core.sv` with a one-cycle registered controller-done bridge so the MAC captures the committed final count vector.
- Standalone readout: 14 directed/golden vectors, 896 product and accumulator comparisons, zero mismatches. Exact zero-score vectors classify as class 1.
- Full classifier: all eight train/validation golden samples, exact scores `[-170, 164, 249, 284, -170, -167, -167, -170]`, exact classes `[0, 1, 1, 1, 0, 0, 0, 0]`, zero mismatches.
- Multi-sample reset sequence passed for all eight samples. Measured start-acceptance to classifier-done latency: **1,386 clock edges** per sample, composed of 1,320 reservoir edges, two controller/readout bridge edges, and 64 MAC edges.
- Readout accumulator proof: theoretical range `[-163,840, +162,560]`; signed 20-bit range `[-524,288, +524,287]`; no legal input can saturate this accumulator.
- Full Python suite after Phase 4: **29 passed**; `compileall` passed.
- No sealed test data was used for vector generation or RTL development.
- Synthesis/place-and-route, LUT/FF/BRAM/DSP, Fmax, power, and board deployment remain unexecuted and unclaimed.

## Scientific analysis audit — 2026-09-21
- Added train/validation-only baseline, dataset, recurrence, SVD, seed, and robustness analyses under outputs/analysis.
- Baseline validation BA: linear RAW20 0.828203–0.837068; Decision Tree depth 5 0.994166; HistGradientBoosting 0.997289 RAW20 and 0.998915 RAW20_DIFF19; locked fixed INT8 reservoir 0.961410.
- Recurrence ablation with retrained readout: Wres ON 0.958323; Wres=0 0.674066.
- SVD components for 90/95/99% variance: 11/24/50.
- Ten-seed validation BA mean/std/min/max: 0.962125/0.007437/0.953108/0.973564.
- Robustness shows strong DC/gain sensitivity and milder small-noise sensitivity.
- Expanded RTL classifier verification: 64 validation-only samples, zero mismatches.
- Python suite after analysis additions: 30 passed; compileall passed.
- Vivado, Yosys and Verilator unavailable; synthesis and RTL optimization were not claimed.

## CI repair audit — 2026-09-21
- Root workflow dependency ranges were replaced for CI execution by requirements-lock.txt: Python 3.13, NumPy 2.2.6, Brian2 2.10.1 and pinned project test dependencies.
- Push/PR RTL verification uses deterministic smoke subsets with the existing default full runners unchanged.
- Local RTL smoke passed for LIF, recurrent, controller, readout and classifier with zero mismatches.
- Full RTL regression remains available in .github/workflows/rtl-full.yml for manual/nightly execution.

## RTL smoke performance audit -- 2026-09-21

- Root cause: run_recurrent_tb.py --smoke built all 10,240 reservoir-step vectors and only then discarded all but 128; the recurrent testbench itself was already parameterized, but the smoke reduction did not propagate through Python vector generation.
- Baseline local recurrent smoke: 128 reservoir-step vectors, 896 comparisons, 0.592 s total (0.075 s generation, 0.093 s/0.091 s unit compile/simulation, 0.091 s/0.085 s step compile/simulation).
- Smoke after fix: 8 deterministic reservoir-step cases, 56 comparisons; unit graph coverage remains 132 vectors / 396 exact comparisons.
- Smoke coverage includes zero, sparse and dense previous-spike vectors, positive and negative recurrent sums, mixed signs, and fan-in 2..12.
- After-fix local recurrent smoke: 0.530 s, zero mismatches.
- After-fix complete local RTL smoke suite: 6.795 s, all runners passed.
- Full recurrent regression remains unchanged: 10,240 vectors and 71,680 reservoir-step comparisons, zero mismatches; local runtime 3.730 s.
- Added phase timing logs for generation, Icarus compile, vvp simulation, comparison, and per-runner smoke elapsed time.
- RTL datapath, recurrent graph, golden vectors, model, expected outputs, and scientific behavior were not changed.

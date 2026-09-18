# CHANGE LOG

> Read this file after `PROJECT_KNOWLEDGE.md`. It is intentionally concise so future work does not require rescanning the entire repository.

## 2026-09-19 — Initial audited project build

### Added
- Clean project structure for data, model, evaluation, hardware simulation, scripts, outputs and tests.
- Detailed `README.md`.
- Locked knowledge contract in `PROJECT_KNOWLEDGE.md`.
- Deterministic group-wise data split by `original_index`.
- Leakage checks for group overlap and exact waveform overlap.
- NumPy no-bias LIF reservoir reference.
- Optional Brian2 LIF reference backend.
- Fixed-point integer FPGA golden model.
- No-bias Ridge readout.
- INT8 readout quantization.
- FPGA weight/config export.
- Metrics and plotting utilities.
- Final-test guard against accidental repeated evaluation.
- Automated pytest suite.

### Locked architecture
`20 ECG points -> 64 fixed LIF neurons -> spike-count state -> no-bias linear readout -> 0/1`

Hardware-oriented choices:
- `Win = {0.5, 1, 2}`
- `Wres = {-1, 0, +1}`
- leak `7/8`
- input gain `3/2`
- recurrent gain `1/16`
- 12-bit input
- 16-bit membrane
- 1-bit spike
- 5-bit spike count
- INT8 readout
- no bias

### Data audit
- Raw rows: 17,214.
- Unique `original_index`: 9,094.
- Train: 12,059 rows / 6,365 groups.
- Validation: 2,594 rows / 1,364 groups.
- Test: 2,561 rows / 1,365 groups.
- Group overlap: 0.
- Exact waveform overlap across splits: 0.

### Local measured results
Float NumPy reference validation BA: `0.9601935412`.

Fixed-point + INT8 validation:
- BA `0.9614098226`
- Accuracy `0.9653045490`
- F1 `0.9510337323`

Final sealed test:
- BA `0.9679954626`
- Accuracy `0.9703240922`
- Sensitivity `0.9607843137`
- Specificity `0.9752066116`
- F1 `0.9563719862`
- MCC `0.9339101129`
- Confusion matrix `[[1652,42],[34,833]]`

### Tests
Final packaged suite: **10 passed**.

### Brian2 limitation
- Brian2 was absent in the build environment.
- `pip install brian2` was genuinely attempted.
- Installation failed due external network/name-resolution failure.
- Brian2 module was syntax-checked but not executed.
- NumPy and fixed-point pipelines were executed end-to-end.

### FPGA status
Completed:
- Quantized software golden model.
- Sparse ternary recurrent weights.
- INT8 readout.
- `.mem`/CSV weight export.

Not completed yet:
- Verilog/VHDL RTL.
- RTL simulation against Python golden vectors.
- Vivado synthesis/place-and-route.
- Real LUT/FF/BRAM/DSP/Fmax/power measurements.
- Physical FPGA board deployment.

### Next recommended step
Install Brian2 in a network-enabled environment and execute the Brian2 reference backend on a small subset first. Then implement bit-exact RTL using `src/hardware/hardware_model.py` as the golden specification. Do not redesign the ML architecture unless an actual measured issue appears.

## 2026-09-19 02:17

### Goal
Complete the initial audit cycle, add the required model-health and fixed-point numerical diagnostics, and execute the Brian2 runtime checks without using the sealed test set for development.

### Changes
- Added fixed-point input clipping and nonzero-to-zero underflow diagnostics.
- Added membrane, recurrent-accumulator, and readout-accumulator saturation diagnostics.
- Added reusable train/validation balanced-accuracy gap, overfitting-warning, and underfitting-warning evaluation.
- Added train metrics and model-health output to `02_train.py`.
- Added train/validation fixed-point diagnostics to `04_quantize.py`.
- Added diagnostic thresholds to `configs/default.yaml`.
- Added diagnostic unit tests and Brian2 smoke/reset coverage.
- Updated Brian2, README, project knowledge, and build-audit status with measured runtime differences.
- Saved a separate development run at `outputs/runs/run_diagnostics_20260919`; it read train/validation only.

### Tests Executed
- `python -m compileall -q src scripts tests` — passed.
- Dependency-free CSV audit of all split files — passed.
- Initial `PYTHONPATH=. pytest -q` — not executed because `pytest` was not on PATH.
- Initial `PYTHONPATH=. python -m pytest -q` — failed because pytest/dependencies were unavailable in the sandbox interpreter.
- `python -m pip install -r requirements.txt` — first sandbox attempt failed at network access; elevated network-enabled attempt succeeded.
- `PYTHONPATH=. python -m pytest -q` in the repaired dependency environment — **14 passed**.
- Brian2 custom smoke/reset/comparison script — passed execution and reset checks.
- Brian2 actual ECG subset script on two train rows — passed execution and reset check.
- `python scripts/02_train.py --run-dir outputs/runs/run_diagnostics_20260919` — passed.
- `python scripts/03_validate.py --run-dir outputs/runs/run_diagnostics_20260919` — passed.
- `python scripts/04_quantize.py --run-dir outputs/runs/run_diagnostics_20260919` — passed.

### Results
- Locked split audit reproduced 12,059/2,594/2,561 rows and 6,365/1,364/1,365 groups for train/validation/test.
- Pairwise group overlap and exact waveform overlap were zero.
- Train balanced accuracy: `0.968137592`.
- Validation balanced accuracy: `0.960193541`.
- Generalization gap: `0.007944051`; overfitting warning: `false`; underfitting warning: `false`.
- Fixed-point train/validation input clipping and underflow rates: `0.0`.
- Fixed-point train/validation membrane and recurrent saturation rates: `0.0`.
- Fixed-point train/validation readout saturation rates: `0.0`.
- Fixed-point + INT8 validation balanced accuracy: `0.961409823`; quantization-drop field: `-0.003086436`.
- Brian2 import, one sample, multiple samples, and reset isolation passed.
- On two actual training ECG rows, Brian2 and NumPy matched `0/128` spike-count elements; first-row nonzero counts were Brian2 `0/64` and NumPy `64/64`.

### Failures / Limitations
- The first full pytest collection was blocked by a SciPy/scikit-learn binary mismatch after dependency installation. Reinstalling compatible SciPy/scikit-learn wheels resolved it.
- Brian2 is runnable but not bit-equivalent to NumPy: its current equation divides input drive by `tau`, whereas the locked discrete NumPy model applies input current directly. Brian2 remains exploratory and is not the FPGA golden model.
- No RTL simulation, Vivado synthesis, FPGA resource, timing, power, or board result was produced.
- The sealed test set was not read by the development scripts or diagnostics run.

### Files Changed
- `src/hardware/fixed_point.py`
- `src/hardware/hardware_model.py`
- `src/evaluation/metrics.py`
- `src/model/reservoir_brian2.py`
- `scripts/02_train.py`
- `scripts/04_quantize.py`
- `configs/default.yaml`
- `tests/test_diagnostics.py`
- `tests/test_brian2_backend.py`
- `PROJECT_KNOWLEDGE.md`
- `README.md`
- `outputs/fpga/reports/BUILD_AUDIT.md`
- `outputs/runs/run_diagnostics_20260919/`

### Current Project State
The NumPy and fixed-point software paths remain the measured working references, now with executable model-health and numerical-safety diagnostics. The full suite passes 14/14 in the Brian2-enabled environment. Brian2 runtime behavior is documented but still requires dynamics reconciliation before it can support equivalence claims. The final test remains sealed.

### Next Recommended Step
Use validation-only evidence to reconcile Brian2 input/leak timing with the locked NumPy equations, then generate bit-exact RTL vectors from `src/hardware/hardware_model.py`; do not use the final test for either decision.

## 2026-09-19 02:19

### Goal
Make the recurrent-accumulator diagnostic width explicit in the configuration and rerun the relevant validation cycle.

### Changes
- Added `hardware.recurrent_accumulator_bits: 16` to `configs/default.yaml`.
- Wired the configured width into `04_quantize.py` and the sealed-test entry point `05_test_final.py` without running the sealed test.

### Tests Executed
- `python scripts/04_quantize.py --run-dir outputs/runs/run_diagnostics_20260919` — passed; train/validation only.
- `PYTHONPATH=. python -m pytest -q` in the repaired dependency environment — **14 passed**.

### Results
- INT8 validation balanced accuracy: `0.961409823`.
- Quantization-drop field: `-0.003086436`.
- Recurrent accumulator saturation remained `0.0` on train and validation.

### Failures / Limitations
- Final test was not rerun; it remains sealed.
- No RTL or synthesis work was performed.

### Files Changed
- `configs/default.yaml`
- `scripts/04_quantize.py`
- `scripts/05_test_final.py`
- `CHANGE_LOG.md`
- `outputs/runs/run_diagnostics_20260919/quantization_summary.json`

### Current Project State
The diagnostic width is now explicit and reproducible; software tests and the train/validation fixed-point integration remain passing.

### Next Recommended Step
Resolve the measured Brian2/NumPy dynamics mismatch on validation, then proceed to golden-vector and RTL work.

## 2026-09-19 02:19

### Goal
Perform the final sealed-test safeguard check before handoff.

### Changes
No source or model changes.

### Tests Executed
- Dependency-free source scan for `test.csv` in `02_train.py`, `03_validate.py`, and `04_quantize.py` — passed after correcting a PowerShell interpolation typo in the audit command.
- Artifact guard check — passed: diagnostics run has no final-test artifact; locked final metrics remain present.

### Results
- Development scripts remain test-blind.
- The sealed `outputs/runs/run_001/test_final_metrics.json` artifact exists and was not regenerated.

### Failures / Limitations
- The first safeguard command had a PowerShell variable interpolation syntax error; it was corrected and rerun successfully.

### Files Changed
- `CHANGE_LOG.md`

### Current Project State
The repository is left with the final test sealed and the diagnostics run separated from the locked final run.

### Next Recommended Step
Reconcile Brian2 dynamics using validation-only evidence, then generate RTL golden vectors.

## 2026-09-19 02:38

### Goal
Reconcile the Brian2 backend as an exploratory LSM reference, freeze the fixed-point FPGA golden model, and generate/test reproducible RTL golden vectors using train and validation only.

### Changes
- Corrected Brian2 input scaling so one Euler step applies the intended direct input current.
- Corrected Brian2 recurrent edge orientation to match `w_res[destination, source]` in NumPy/hardware.
- Added chunked disjoint neuron blocks for practical full train/validation Brian2 execution while preserving sample isolation.
- Added Brian2 parameter-audit metadata and reproducible validation script `scripts/08_validate_brian2.py`.
- Added `IntegerLIFReservoir.transform_trace` for per-timestep bit-exact hardware state capture.
- Added `src/hardware/golden_vectors.py` and `scripts/07_generate_rtl_vectors.py`.
- Added deterministic, direct-model, legal-range, no-bias, and lossless-reload golden-vector tests.
- Updated `README.md`, `PROJECT_KNOWLEDGE.md`, and `outputs/fpga/reports/BUILD_AUDIT.md`.

### Tests Executed
- Baseline full suite before changes: **14 passed**.
- `python -m compileall -q src scripts tests` — passed.
- Controlled Brian2/NumPy recurrent comparison — exact match.
- Actual train ECG comparison with recurrence disabled — exact match, 128/128 elements.
- Actual train ECG comparison with recurrence enabled — 100/128 elements matched.
- `python scripts/08_validate_brian2.py` — passed on all train/validation rows.
- `python scripts/07_generate_rtl_vectors.py --force` — passed; generated 8 vectors.
- Final full suite: **17 passed**.
- Final manifest/range/sealed-test audit — passed.

### Results
- Brian2 train runtime: `27.054 s`.
- Brian2 validation runtime: `6.191 s`.
- Brian2 no-bias validation Balanced Accuracy: `0.9530366169`.
- Brian2 train non-zero spike-count rate: `0.9991111`.
- Brian2 validation non-zero spike-count rate: `0.9985965`.
- Golden-vector bundle: 8 samples, 20 timesteps, 64 neurons.
- Golden-vector model SHA-256: `fad34b75150bc80f2c43ed8ffb302c29f446993122b9cb8f877ad4d5c3305b11`.
- Golden-vector score range: `-170..284`; spike-count range: `1..10` for the selected samples.
- Golden-vector sources were exactly `train` and `validation`; no `test.csv` was opened.

### Failures / Limitations
- Brian2 remains non-bit-equivalent on real recurrent ECG traces because event-driven floating-point recurrent accumulation can diverge sparsely from vectorized NumPy recurrence. This is documented and Brian2 is not the FPGA golden reference.
- A zero-edge Brian2 experiment initially exposed an empty-Synapses runtime error; the backend now omits the Synapses object when there are no recurrent edges.
- RTL implementation, RTL simulation, synthesis, resource/timing/power measurement, and board deployment remain pending.

### Files Changed
- `src/model/reservoir_brian2.py`
- `src/hardware/hardware_model.py`
- `src/hardware/golden_vectors.py`
- `scripts/07_generate_rtl_vectors.py`
- `scripts/08_validate_brian2.py`
- `tests/test_brian2_backend.py`
- `tests/test_golden_vectors.py`
- `README.md`
- `PROJECT_KNOWLEDGE.md`
- `outputs/fpga/reports/BUILD_AUDIT.md`
- `outputs/fpga/golden_vectors/`
- `outputs/runs/run_brian2_20260919/`
- `CHANGE_LOG.md`

### Current Project State
The NumPy and fixed-point software paths remain stable. The fixed-point model is explicitly frozen as the FPGA behavioral reference, with a reproducible trace/vector bundle and 17 passing tests. Brian2 is runnable across train/validation with meaningful activity and validation performance, but its recurrent floating-point dynamics are documented as exploratory rather than bit-exact.

### Next Recommended Step
Implement the first RTL module and testbench against `outputs/fpga/golden_vectors`, comparing intermediate membrane/spike states before considering synthesis.

## 2026-09-19 02:40

### Goal
Complete the final documentation/status pass and verify the repository after all Task A–D changes.

### Changes
- Updated README and project knowledge structure/status for Brian2 validation, golden vectors, and pending RTL stages.
- Added scripts 07/08 to the documented project structure.

### Tests Executed
- `PYTHONPATH=. python -m pytest -q` — **17 passed**.
- Golden-vector manifest/range/source audit — passed.
- Confirmed vector scripts contain no `test.csv` access.

### Results
- Final suite: **17 passed in 5.91 s**.
- Golden vectors remain 8 train/validation samples with model hash `fad34b75150bc80f2c43ed8ffb302c29f446993122b9cb8f877ad4d5c3305b11`.
- Sealed final test artifact was not regenerated.

### Failures / Limitations
- RTL, RTL simulation, synthesis, and FPGA deployment remain pending.

### Files Changed
- `README.md`
- `PROJECT_KNOWLEDGE.md`
- `CHANGE_LOG.md`

### Current Project State
Brian2 reconciliation, fixed-point golden-model tracing, reproducible RTL vectors, validation tests, and documentation are complete. The final test remains sealed.

### Next Recommended Step
Implement the first RTL module and compare its intermediate states against the generated golden vectors.

## 2026-09-19 02:49

### Goal
Implement RTL Phase 1: a bit-exact-capable one-neuron LIF processing element, fixed-point helpers, golden-vector testbench, and simulator regression runner.

### Changes
- Added `rtl/src/fixed_point_pkg.sv` with signed membrane saturation and locked threshold/leak constants.
- Added `rtl/src/lif_pe.sv` as a combinational PE with external membrane state, input/recurrent contributions, saturation, threshold, spike, and reset output.
- Added `rtl/tb/tb_lif_pe.sv` to compare membrane-before, membrane-after, spike, and post-reset states.
- Added `rtl/scripts/run_lif_tb.py` to flatten exported train/validation traces, add Python-generated threshold/saturation directed cases, invoke Icarus, and parse PASS/FAIL.
- Added RTL structure, manifest, no-bias, and simulator-dependent regression tests.
- Updated README and build audit with RTL Phase 1 status.

### Tests Executed
- Baseline before RTL changes: **17 passed**.
- `python -m compileall -q rtl scripts src tests` — passed.
- Simulator discovery for Icarus, Verilator, Questa/ModelSim — all unavailable.
- `python -m pip install iverilog` — attempted; no matching package found.
- `python rtl/scripts/run_lif_tb.py` — returned `SKIP`, exit code `2`; RTL was not executed.
- Final Python suite: **19 passed, 1 skipped**; skip is simulator-dependent RTL regression.

### Results
- RTL source structure exists under `rtl/src`, `rtl/tb`, `rtl/mem`, and `rtl/scripts`.
- Static no-bias audit passed.
- RTL manifest width checks passed: input 12-bit, membrane 16-bit, spike count 5-bit, accumulator 20-bit.
- Python golden-vector tests remain passing and cover deterministic generation, direct hardware-model equality, legal ranges, threshold/reset behavior, saturation cases, and lossless reload.

### Failures / Limitations
- No SystemVerilog simulator is installed, so bit-exact RTL execution is not claimed.
- `pip install iverilog` is not a viable installation path in this environment.
- RTL synthesis, integration, and FPGA deployment remain pending.

### Files Changed
- `rtl/src/fixed_point_pkg.sv`
- `rtl/src/lif_pe.sv`
- `rtl/tb/tb_lif_pe.sv`
- `rtl/scripts/run_lif_tb.py`
- `rtl/mem/.gitkeep`
- `tests/test_rtl_phase1.py`
- `README.md`
- `outputs/fpga/reports/BUILD_AUDIT.md`
- `CHANGE_LOG.md`

### Current Project State
The first RTL LIF PE and its golden-data testbench are implemented, but simulation remains pending because no supported simulator is available. The Python fixed-point model remains authoritative and the known-good suite is passing.

### Next Recommended Step
Install or provide a supported SystemVerilog simulator, run `python rtl/scripts/run_lif_tb.py`, and do not claim bit-exact RTL until the testbench reports zero mismatches.

## 2026-09-19 02:59

### Goal
Install a real SystemVerilog simulator and complete RTL Phase 1.1 bit-exact verification of the LIF processing element.

### Changes
- Checked WSL2; `wsl --status` and `wsl -l -v` returned `E_ACCESSDENIED` while enumerating distributions.
- Installed MSYS2 UCRT64 Icarus Verilog package `mingw-w64-ucrt-x86_64-iverilog`.
- Verified Icarus Verilog and VVP version `13.0 (stable) (v13_0)`.
- Updated `rtl/scripts/run_lif_tb.py` to discover common native MSYS2 Icarus installations and report the simulator version.
- Fixed the runner to pass project-relative RTL source paths; absolute paths containing `Học tập` caused Icarus to reject the source file before elaboration.
- Updated project knowledge, README status, and the build audit with the verified result.

### Tests Executed
- Tiny SystemVerilog smoke test: `RTL_SIM_OK`, compile exit `0`, run exit `0`.
- `python rtl/scripts/run_lif_tb.py` — passed with Icarus 13.0.
- `python -m compileall -q src scripts tests rtl/scripts` — passed.
- `python -m pytest -q` — **20 passed**.

### Results
- RTL testbench compared **10,246 timestep vectors** and **40,984 state fields** covering membrane-before, membrane-after, spike, and post-spike reset.
- **Zero mismatches** were observed, including directed exact-threshold and positive/negative saturation cases.
- RTL Phase 1 LIF PE bit-exactness is verified against the Python fixed-point golden vectors.
- The sealed test set was not used or regenerated.

### Failures / Limitations
- WSL2 remains unavailable to this process because distro enumeration is denied.
- RTL integration beyond the single LIF PE, synthesis, implementation, resource/Fmax/power measurement, and board deployment remain pending.

### Files Changed
- `rtl/scripts/run_lif_tb.py`
- `README.md`
- `PROJECT_KNOWLEDGE.md`
- `outputs/fpga/reports/BUILD_AUDIT.md`
- `CHANGE_LOG.md`

### Current Project State
The Python fixed-point model remains the FPGA behavioral reference. The Phase 1 LIF PE now has an executed Icarus regression with zero mismatches; the recurrent reservoir engine must still wait until its RTL integration is implemented and verified.

### Next Recommended Step
Implement the next RTL integration stage only after preserving the verified LIF PE contract, then extend the same golden-vector comparison before considering synthesis.

## 2026-09-19 03:25

### Goal
Implement and verify the sparse ternary recurrent engine and one complete reservoir timestep without changing the locked LIF PE or frozen recurrent matrix.

### Changes
- Ran the required baseline: **20 Python tests passed** and the Phase 1 LIF regression remained at 40,984 exact comparisons.
- Verified the frozen recurrent semantics: Python computes `spikes @ w_res.T`; the logical edge is `source=column -> destination=row`.
- Fixed `src/hardware/export_fpga.py` so `w_res_edges.csv` serializes that logical orientation. The frozen `w_res` values, seed, and model hash were not regenerated or changed.
- Added `rtl/src/sparse_recurrent_engine.sv` with sparse edge-list scanning, ternary add/subtract, signed accumulation, Q10 recurrent scaling, and 16-bit saturation.
- Added `rtl/src/reservoir_step.sv` to connect the recurrent engine to the verified LIF PE for one logical neuron/timestep.
- Added directed and golden-vector testbenches plus `rtl/scripts/run_recurrent_tb.py`.
- Added graph checksum, orientation, edge-count, sign, index, fan-in, duplicate, self-edge, and RTL regression tests.

### Tests Executed
- Single-spike orientation audit for source neuron 7: corrected export now matches Python destinations and signs exactly.
- First recurrent RTL run exposed a scaling mismatch: expected `+/-64`, RTL produced `+/-16`.
- Fixed scaling to match Python: `(1 << membrane_fractional_bits) >> recurrent_gain_shift = 64`.
- Final `python rtl/scripts/run_recurrent_tb.py` with Icarus Verilog 13.0:
  - 132 directed vectors: 396 exact recurrent comparisons.
  - 10,240 real golden neuron vectors: 30,720 exact recurrent comparisons.
  - One-step integration: 71,680 exact comparisons.
  - Mismatches: **0**.
- `python rtl/scripts/run_lif_tb.py` — retained Phase 1 zero-mismatch result.
- `python -m compileall -q src scripts tests rtl/scripts` — passed.
- `python -m pytest -q` — **23 passed**.

### Results
- Graph statistics: 403 edges, 9.838867% density, destination fan-in 2–12, mean fan-in 6.296875, 311 positive edges, 92 negative edges, no self-edges, no duplicates.
- Raw ternary recurrent sum is bounded by `[-12, +12]`; the selected 5-bit signed accumulator range `[-16, +15]` cannot overflow.
- The sparse recurrent engine and one-timestep reservoir integration are bit-exact against the Python FPGA golden model.
- The sealed final test set was not accessed or regenerated.

### Failures / Limitations
- The initial edge export had reversed logical labels; this was fixed in the exporter and verified by matrix reconstruction and a single-spike test.
- The initial RTL scaling used a literal left shift of 4; this was fixed to the Python Q10 recurrent unit of 64.
- The full 20-timestep controller, streaming interface, readout MAC, synthesis, and board deployment remain intentionally unimplemented.

### Files Changed
- `src/hardware/export_fpga.py`
- `outputs/fpga/weights/w_res_edges.csv`
- `rtl/src/sparse_recurrent_engine.sv`
- `rtl/src/reservoir_step.sv`
- `rtl/tb/tb_sparse_recurrent_engine.sv`
- `rtl/tb/tb_reservoir_step.sv`
- `rtl/scripts/run_recurrent_tb.py`
- `tests/test_rtl_phase2.py`
- `README.md`
- `PROJECT_KNOWLEDGE.md`
- `outputs/fpga/reports/BUILD_AUDIT.md`
- `CHANGE_LOG.md`

### Current Project State
Phase 1 LIF and Phase 2 sparse recurrent plus one-timestep integration are verified bit-exactly with Icarus Verilog 13.0. The project must remain in the current phase until any future multi-timestep controller is separately verified.

### Next Recommended Step
Review and preserve the verified recurrent/one-step contracts, then implement the next controller stage only with new train/validation golden vectors and an independent zero-mismatch regression.

## 2026-09-19 03:19

### Goal
Complete the final Phase 2 regression pass and ensure recurrent integration mismatch output contains the required edge-mask context.

### Changes
- Added expected contributing-edge masks to reservoir-step test vectors and mismatch diagnostics.

### Tests Executed
- `python rtl/scripts/run_recurrent_tb.py` — passed with Icarus Verilog 13.0.
- `python -m compileall -q src scripts tests rtl/scripts` — passed.
- `python -m pytest -q` — **23 passed**.

### Results
- Recurrent directed comparisons: **396 exact**.
- Golden recurrent comparisons: **30,720 exact**.
- Reservoir-step comparisons: **71,680 exact**.
- Mismatches: **0**.

### Failures / Limitations
- No new failures. Multi-timestep controller, readout MAC, synthesis, and board deployment remain pending.

### Files Changed
- `rtl/scripts/run_recurrent_tb.py`
- `rtl/tb/tb_reservoir_step.sv`
- `CHANGE_LOG.md`

### Current Project State
Phase 2 sparse recurrence and one-timestep integration remain bit-exact and regression-clean.

### Next Recommended Step
Keep the verified contracts fixed while implementing only the next separately testable controller stage.

## 2026-09-19 03:32

### Goal
Implement and verify the complete 20-timestep, 64-neuron time-multiplexed reservoir controller without adding the Phase 4 readout.

### Changes
- Ran the required baseline before changes: **23 Python tests passed**, Phase 1 LIF passed with 40,984 exact comparisons, and Phase 2 passed with zero mismatches.
- Added `rtl/src/reservoir_controller.sv` with explicit `IDLE`, `LOAD_SAMPLE`, `PROCESS_NEURON`, `COMMIT_TIMESTEP`, and `DONE` states.
- Added current/next membrane arrays, current/next spike vectors, current/next 5-bit counters, input-code mapping, segment reset, and checkpoint outputs.
- Added `rtl/tb/tb_reservoir_controller.sv` with first-mismatch diagnostics and checkpoint comparisons.
- Added `rtl/scripts/run_controller_tb.py` using existing simulator discovery and Python fixed-point golden traces.
- Added synthetic sparse/high-activity segments, real train/validation segments, and train-A/validation-B/train-A reset coverage.
- Added `tests/test_rtl_phase3.py` for locked dimensions, Win mapping, no-readout/no-bias checks, and controller regression.
- Documented exact timestep ordering and Phase 3 verification in README, project knowledge, and build audit.

### Tests Executed
- `python rtl/scripts/run_controller_tb.py` — Icarus Verilog 13.0, zero mismatches.
- 14 complete segments and 280 timestep checkpoints.
- Membrane-before: **17,920 exact**.
- Membrane-after: **17,920 exact**.
- Persistent membrane-reset: **17,920 exact**.
- Input: **17,920 exact**.
- Recurrent: **17,920 exact**.
- Spikes: **17,920 exact**.
- Spike counters: **17,920 exact**.
- Previous-spike vectors: **280 exact**.
- Final 64-count vectors: **896 exact**.
- `python -m compileall -q src scripts tests rtl/scripts` — passed.
- `python -m pytest -q` — **26 passed**.
- Phase 1 and Phase 2 regressions rerun successfully after controller changes.

### Results
- Complete 20-timestep reservoir control flow is bit-exact against the Python fixed-point golden model.
- Explicit A/B/A reset isolation passed with no stale membrane, spike-vector, counter, timestep, or neuron state.
- No spike counter exceeded 20.
- No readout MAC or classifier logic was added.
- The sealed final test set was not accessed or regenerated.

### Failures / Limitations
- No controller mismatch remained after the final run.
- Readout integration, synthesis, timing/resource measurement, power analysis, and board deployment remain pending.

### Files Changed
- `rtl/src/reservoir_controller.sv`
- `rtl/tb/tb_reservoir_controller.sv`
- `rtl/scripts/run_controller_tb.py`
- `tests/test_rtl_phase3.py`
- `README.md`
- `PROJECT_KNOWLEDGE.md`
- `outputs/fpga/reports/BUILD_AUDIT.md`
- `CHANGE_LOG.md`

### Current Project State
Phase 1 LIF, Phase 2 sparse recurrence/one-step integration, and Phase 3 full 20-timestep reservoir control are simulation-verified with Icarus Verilog 13.0. The readout remains intentionally deferred to Phase 4.

### Next Recommended Step
Preserve the verified controller contract and begin Phase 4 readout integration only after reviewing the checkpoint and reset evidence.

## 2026-09-19 03:49

### Goal
Implement and verify the bit-exact Phase 4 readout MAC and full controller-to-classifier integration without changing the locked Python model or verified Phase 1–3 RTL contracts.

### Changes
- Ran the required clean baseline before edits: **26 Python tests passed**, LIF passed with 40,984 exact comparisons, recurrent regression passed with zero mismatches, and controller regression passed with zero mismatches.
- Added `rtl/src/readout_mac.sv`: 64-term time-multiplexed MAC, signed INT8 weight ROM, unsigned 5-bit count handling, signed 32-bit intermediate product, explicit signed 20-bit saturation, per-term checkpoints, and no-bias `score >= 0` classifier.
- Added `rtl/src/ecg_classifier_core.sv`: verified reservoir controller followed by a registered final-count-to-readout start bridge.
- Added `rtl/tb/tb_readout_mac.sv` and `rtl/scripts/run_readout_tb.py` with directed and frozen golden-vector tests.
- Added `rtl/tb/tb_ecg_classifier_core.sv` and `rtl/scripts/run_classifier_tb.py` with all eight train/validation vectors and sequential reset coverage.
- Added `tests/test_rtl_phase4.py` for architecture, manifest, signed-width proof, simulator regression, and latency checks.
- Updated `README.md`, `PROJECT_KNOWLEDGE.md`, and `outputs/fpga/reports/BUILD_AUDIT.md` with the verified Phase 4 contract and measured results.

### Tests Executed
- Standalone readout: 14 vectors × 64 terms = **896 exact MAC comparisons**, zero mismatches.
- Directed zero-score vectors: exact zero classified as class 1.
- Full classifier: eight train/validation golden samples, exact scores `[-170, 164, 249, 284, -170, -167, -167, -170]`, exact classes `[0, 1, 1, 1, 0, 0, 0, 0]`, zero mismatches.
- Multi-sample reset coverage: all eight sequential samples passed with no stale reservoir state.
- Measured integrated latency: **1,386 clock edges** from start acceptance to classifier done; 1,320 reservoir edges + two bridge edges + 64 MAC edges.
- Legal accumulator proof: `64 × 20 × [-128, +127] = [-163,840, +162,560]`, contained by signed 20-bit `[-524,288, +524,287]`; no valid vector saturates.
- Icarus Verilog: **13.0 stable**.
- Full `python -m pytest -q`: **29 passed**; `python -m compileall -q src scripts tests rtl/scripts`: passed.

### Results
- Phase 4 readout and full classifier are bit-exact against the Python fixed-point golden reference for the executed train/validation vectors.
- The readout weight file remains the single exported source of signed INT8 weights; no bias/intercept was added.
- The sealed final test set was not accessed or regenerated.

### Failures / Limitations
- No RTL mismatch remains in the executed Phase 4 regressions.
- Synthesis, implementation, resource utilization, Fmax, power, and board deployment remain unexecuted.
- The current integration interface is a simulation-verified 20-sample streaming interface; physical FPGA pin/clock protocol is still unspecified.

### Files Changed
- `rtl/src/readout_mac.sv`
- `rtl/src/ecg_classifier_core.sv`
- `rtl/tb/tb_readout_mac.sv`
- `rtl/tb/tb_ecg_classifier_core.sv`
- `rtl/scripts/run_readout_tb.py`
- `rtl/scripts/run_classifier_tb.py`
- `tests/test_rtl_phase4.py`
- `README.md`
- `PROJECT_KNOWLEDGE.md`
- `outputs/fpga/reports/BUILD_AUDIT.md`
- `CHANGE_LOG.md`

### Current Project State
RTL Phases 1–4 are simulator-verified with Icarus Verilog 13.0. The complete reservoir and readout classifier produce exact train/validation golden results. Device-specific implementation work has not started.

### Next Recommended Step
Run a synthesis/implementation flow only after the target FPGA device, clock, reset, and streaming I/O contract are supplied; report measured LUT, FF, BRAM, DSP, Fmax, and power results rather than estimates.

## 2026-09-19 04:23

### Goal
Finalize the repository documentation in Vietnamese and prepare the complete verified project for GitHub publication.

### Changes
- Rewrote `README.md` in Vietnamese while preserving precise English hardware and ML terminology.
- Cross-checked dataset counts, split sizes, model metrics, Brian2 audit values, golden-vector hashes, graph statistics, RTL comparison counts, and Phase 4 latency against stored artifacts.
- Expanded `.gitignore` for Python caches, virtual environments, IDE files, secrets, OS files, waveform files, and temporary simulator binaries while retaining useful golden vectors and reports.
- Completed the pre-push secret-pattern audit; no credentials or private keys were found.

### Tests Executed
- `python -m pytest -q`
- `python -m compileall .`
- `python rtl/scripts/run_lif_tb.py`
- `python rtl/scripts/run_recurrent_tb.py`
- `python rtl/scripts/run_controller_tb.py`
- `python rtl/scripts/run_readout_tb.py`
- `python rtl/scripts/run_classifier_tb.py`

### Results
- Python suite: **29 passed**.
- Compileall: passed.
- All five RTL regressions: passed with zero mismatches.
- README now documents the current verified software/RTL state and explicitly marks synthesis, timing, resources, power, and board deployment as pending.

### Failures / Limitations
- Git initialization, remote inspection, commit, and push remain to be performed.
- The local directory was not a Git repository at audit time.

### Files Changed
- `README.md`
- `.gitignore`
- `CHANGE_LOG.md`

### Current Project State
The local project is regression-clean and has a Vietnamese, team-leader-oriented README. No GitHub state has been changed yet.

### Next Recommended Step
Inspect the target GitHub repository history, initialize/link Git safely, commit the complete project, push without force, and verify the remote commit and file contents.

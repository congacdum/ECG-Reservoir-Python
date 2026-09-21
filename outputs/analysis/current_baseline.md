# Current baseline snapshot

Ngày audit: 2026-09-21

## Current model configuration

- 20 timestep/sample.
- 64 logical LIF neurons.
- 403 sparse recurrent edges.
- Spike-count readout.
- No trainable bias or intercept.
- Fixed-point input: signed 12-bit Q10.
- Membrane: signed 16-bit Q10.
- Spike count: unsigned 5-bit, range 0..20.
- Readout: signed INT8 weights, signed 20-bit score.
- Locked seed: 42.
- RTL latency: 1386 architectural cycles/sample.

## Measured metrics

- Locked fixed-point + INT8 validation Balanced Accuracy: 0.9614098226.
- Development diagnostic train Balanced Accuracy: 0.9681375920.
- Development diagnostic validation Balanced Accuracy: 0.9601935412.
- Known locked final-test Balanced Accuracy: 0.9679954626; this stored result was not rerun for this task.
- Python test suite: 29 passed.

## Current limitations

- Dataset does not contain patient_id or record_id.
- Exact semantic/generation provenance of finalLabel is not established.
- Vivado, Yosys and Verilator are unavailable in the current environment.
- No synthesis, Fmax, LUT/FF/BRAM/DSP or power result is claimed.
- New analysis artifacts use train and validation only; test.csv was not read.


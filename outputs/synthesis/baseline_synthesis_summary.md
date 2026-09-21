# RTL baseline synthesis status

Ngày audit: 2026-09-21

## Tool availability

- Vivado: unavailable.
- Yosys: unavailable.
- Icarus Verilog 13.0: available.
- Verilator: unavailable.

## Result

Baseline synthesis was not executed because no synthesis tool is installed in the current environment.

Therefore the following values are intentionally unclaimed:

- LUT
- FF
- BRAM
- DSP
- critical path
- WNS
- Fmax
- power
- energy/classification

The logical top identified for a future run is rtl/src/ecg_classifier_core.sv. A target FPGA part has not been supplied.

## RTL optimization

No RTL optimization was performed. The verified Phase 1–4 RTL is preserved. Optimization must follow a real baseline synthesis report and a bit-exact regression.


from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .hardware_model import HardwareParams, IntegerLIFReservoir, integer_readout_score


def _params_payload(params: HardwareParams) -> dict:
    return {
        "input_format": asdict(params.input_format),
        "membrane_bits": params.membrane_bits,
        "membrane_fractional_bits": params.membrane_fractional_bits,
        "threshold": params.threshold,
        "leak_shift": params.leak_shift,
        "recurrent_shift": params.recurrent_shift,
        "recurrent_accumulator_bits": params.recurrent_accumulator_bits,
        "saturation": params.saturation,
    }


def model_sha256(
    reservoir: IntegerLIFReservoir,
    w_out_int8: np.ndarray,
    accumulator_bits: int,
) -> str:
    """Hash all model values and fixed-point parameters used by the vectors."""
    digest = hashlib.sha256()
    for name, array in (
        ("w_in", reservoir.w_in),
        ("w_res", reservoir.w_res),
        ("w_out_int8", np.asarray(w_out_int8, dtype=np.int8)),
    ):
        digest.update(name.encode("ascii"))
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    digest.update(json.dumps(_params_payload(reservoir.params), sort_keys=True).encode("utf-8"))
    digest.update(str(int(accumulator_bits)).encode("ascii"))
    return digest.hexdigest()


def generate_golden_vectors(
    reservoir: IntegerLIFReservoir,
    X: np.ndarray,
    labels: np.ndarray,
    w_out_int8: np.ndarray,
    sample_ids: Sequence[str],
    source_splits: Sequence[str],
    accumulator_bits: int = 20,
) -> dict[str, np.ndarray]:
    """Generate trace and final outputs directly from the FPGA golden model."""
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    w_out_int8 = np.asarray(w_out_int8, dtype=np.int8).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must have shape [batch,time]")
    if len(labels) != len(X) or len(sample_ids) != len(X) or len(source_splits) != len(X):
        raise ValueError("Sample metadata length mismatch")
    if w_out_int8.size != reservoir.neurons:
        raise ValueError("Readout weight dimension mismatch")

    trace = reservoir.transform_trace(X)
    features = trace["spike_counts"]
    readout_raw = features.astype(np.int32) @ w_out_int8.astype(np.int32)
    readout_score = integer_readout_score(features, w_out_int8, accumulator_bits)
    result = {
        **trace,
        "labels_debug": labels,
        "sample_ids": np.asarray(sample_ids, dtype="U"),
        "source_splits": np.asarray(source_splits, dtype="U"),
        "w_out_int8": w_out_int8,
        "readout_accumulator_raw": readout_raw.astype(np.int32),
        "readout_accumulator": readout_score.astype(np.int32),
        "signed_score": readout_score.astype(np.int32),
        "predicted_class": (readout_score >= 0).astype(np.int64),
    }
    return result


def _unsigned_hex(value: int, bits: int) -> str:
    mask = (1 << bits) - 1
    return f"{int(value) & mask:0{(bits + 3) // 4}x}"


def _array_metadata(array: np.ndarray) -> dict:
    array = np.asarray(array)
    info = {"shape": list(array.shape), "dtype": str(array.dtype)}
    if array.size and np.issubdtype(array.dtype, np.number):
        info["min"] = int(np.min(array))
        info["max"] = int(np.max(array))
    return info


def _json_row(values: np.ndarray, bits: int | None = None) -> str:
    values = np.asarray(values).reshape(-1)
    if bits is None:
        return json.dumps([int(v) for v in values], separators=(",", ":"))
    return json.dumps([_unsigned_hex(int(v), bits) for v in values], separators=(",", ":"))


def write_golden_vectors(
    output_dir: str | Path,
    vectors: Mapping[str, np.ndarray],
    reservoir: IntegerLIFReservoir,
    accumulator_bits: int,
    reservoir_seed: int,
) -> dict:
    """Write lossless NPZ vectors plus RTL-friendly memory files and manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in vectors.items()}
    np.savez_compressed(output_dir / "golden_vectors.npz", **arrays)

    input_q = arrays["input_quantized"]
    with (output_dir / "inputs_q12.mem").open("w", encoding="ascii", newline="\n") as f:
        for row in input_q:
            f.write(" ".join(_unsigned_hex(v, 12) for v in row) + "\n")

    with (output_dir / "spike_counts_5bit.mem").open("w", encoding="ascii", newline="\n") as f:
        for row in arrays["spike_counts"]:
            f.write(" ".join(_unsigned_hex(v, 5) for v in row) + "\n")

    with (output_dir / "expected_scores_20bit.mem").open("w", encoding="ascii", newline="\n") as f:
        for value in arrays["signed_score"]:
            f.write(_unsigned_hex(value, accumulator_bits) + "\n")

    with (output_dir / "expected_classes.mem").open("w", encoding="ascii", newline="\n") as f:
        for value in arrays["predicted_class"]:
            f.write(f"{int(value)}\n")

    trace_fields = (
        "membrane_before",
        "input_contribution",
        "recurrent_contribution",
        "recurrent_contribution_raw",
        "membrane_update_raw",
        "membrane_after_update",
        "threshold_result",
        "reset_membrane",
        "spike_count_state",
    )
    with (output_dir / "timestep_trace.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "sample_id", "source_split", "t", "input_quantized", *trace_fields])
        for sample_index in range(input_q.shape[0]):
            for t in range(input_q.shape[1]):
                row = [
                    sample_index,
                    arrays["sample_ids"][sample_index],
                    arrays["source_splits"][sample_index],
                    t,
                    int(input_q[sample_index, t]),
                ]
                for field in trace_fields:
                    bits = 1 if field == "threshold_result" else None
                    row.append(_json_row(arrays[field][sample_index, t], bits=bits))
                writer.writerow(row)

    model_hash = model_sha256(reservoir, arrays["w_out_int8"], accumulator_bits)
    array_names = [
        "input_quantized",
        "membrane_before",
        "input_contribution",
        "recurrent_contribution",
        "recurrent_contribution_raw",
        "membrane_update_raw",
        "membrane_after_update",
        "threshold_result",
        "reset_membrane",
        "spike_count_state",
        "spike_counts",
        "w_out_int8",
        "readout_accumulator_raw",
        "readout_accumulator",
        "signed_score",
        "predicted_class",
        "labels_debug",
    ]
    manifest = {
        "format_version": 1,
        "model_sha256": model_hash,
        "reservoir_seed": int(reservoir_seed),
        "sample_count": int(input_q.shape[0]),
        "sequence_length": int(input_q.shape[1]),
        "sample_ids": [str(v) for v in arrays["sample_ids"]],
        "source_splits": [str(v) for v in arrays["source_splits"]],
        "fixed_point": {
            "input": {"bits": reservoir.params.input_format.bits, "fractional_bits": reservoir.params.input_format.fractional_bits, "signed": True},
            "membrane": {"bits": reservoir.params.membrane_bits, "fractional_bits": reservoir.params.membrane_fractional_bits, "signed": True},
            "spike": {"bits": 1, "signed": False},
            "spike_count": {"bits": 5, "signed": False, "range": [0, 20]},
            "recurrent_weight": {"values": [-1, 0, 1]},
            "readout_weight": {"bits": 8, "signed": True},
            "readout_accumulator": {"bits": int(accumulator_bits), "signed": True},
            "rounding": "numpy.rint (ties-to-even)",
            "saturation": bool(reservoir.params.saturation),
        },
        "no_bias": {
            "reservoir_bias_current": False,
            "readout_intercept": False,
            "exported_bias": False,
        },
        "model_parameters": _params_payload(reservoir.params),
        "files": {
            "arrays": "golden_vectors.npz",
            "inputs": "inputs_q12.mem",
            "spike_counts": "spike_counts_5bit.mem",
            "scores": "expected_scores_20bit.mem",
            "classes": "expected_classes.mem",
            "trace": "timestep_trace.csv",
        },
        "arrays": {name: _array_metadata(arrays[name]) for name in array_names},
    }
    with (output_dir / "golden_vectors_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest


def load_golden_vectors(output_dir: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    output_dir = Path(output_dir)
    with (output_dir / "golden_vectors_manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    with np.load(output_dir / manifest["files"]["arrays"], allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    return manifest, arrays

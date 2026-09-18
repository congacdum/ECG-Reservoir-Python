from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def _twos_complement_hex(value: int, bits: int) -> str:
    mask = (1 << bits) - 1
    width = (bits + 3) // 4
    return f"{value & mask:0{width}X}"


def export_hardware_artifacts(
    output_dir: str | Path,
    w_in: np.ndarray,
    w_res: np.ndarray,
    w_out_int8: np.ndarray,
    metadata: dict,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    w_in = np.asarray(w_in)
    w_res = np.asarray(w_res, dtype=np.int8)
    w_out_int8 = np.asarray(w_out_int8, dtype=np.int8)

    # Win encoded as 2-bit operation codes: 0=>0.5, 1=>1, 2=>2.
    in_codes = np.where(w_in == 0.5, 0, np.where(w_in == 1.0, 1, 2)).astype(np.uint8)
    with (output_dir / "w_in_codes.mem").open("w", encoding="ascii") as f:
        for value in in_codes:
            f.write(f"{int(value):02b}\n")

    # The golden model evaluates spikes @ w_res.T, so w_res[row, col]
    # represents an edge col (source) -> row (destination).
    # Export source,destination in that logical direction.
    destination, source = np.nonzero(w_res)
    with (output_dir / "w_res_edges.csv").open("w", encoding="ascii") as f:
        f.write("source,destination,sign\n")
        for dst, src in zip(destination, source):
            sign = 1 if int(w_res[dst, src]) < 0 else 0
            f.write(f"{int(src)},{int(dst)},{sign}\n")

    with (output_dir / "w_out_int8.mem").open("w", encoding="ascii") as f:
        for value in w_out_int8:
            f.write(_twos_complement_hex(int(value), 8) + "\n")

    metadata = dict(metadata)
    metadata.update(
        {
            "neurons": int(w_in.size),
            "recurrent_edges": int(len(destination)),
            "w_in_encoding": {"00": 0.5, "01": 1.0, "10": 2.0},
            "w_res_encoding": "sparse edge list; sign 0=+1, 1=-1",
            "w_out_encoding": "signed INT8 two's-complement hex",
        }
    )
    with (output_dir / "fpga_export_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata

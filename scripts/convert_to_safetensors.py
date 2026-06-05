#!/usr/bin/env python3
"""Convert a PyTorch checkpoint (.ckpt / .pt state dict) to safetensors.

OFFLINE TOOLING ONLY. FrankenWhisper's Rust engine NEVER invokes this script and
NEVER unpickles anything — reading a PyTorch pickle can execute arbitrary code,
so unpickling is a deliberate, human-run, out-of-band step. The Rust loader
(src/native_engine/weights.rs) reads the resulting *.safetensors only.

Every tensor is written as contiguous f32 (the format the Rust loader expects),
with a `__metadata__` block recording the source filename, the sha256 of the
input checkpoint, and the conversion timestamp, so the artifact is auditable.

Requires: python3, torch, safetensors. Install with:
    pip install torch safetensors

Usage:
    python3 convert_to_safetensors.py INPUT.ckpt OUTPUT.safetensors [--key KEY]

    --key KEY  if the checkpoint is a dict wrapping the state dict under a key
               (e.g. "state_dict" / "model"), unwrap that key first.

On success the output path and its sha256 are printed (pin this in
fetch_aux_models.sh).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .ckpt/.pt state dict to f32 safetensors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="input .ckpt / .pt file")
    parser.add_argument("output", type=Path, help="output .safetensors file")
    parser.add_argument(
        "--key",
        default=None,
        help="unwrap the state dict from this top-level key (e.g. state_dict)",
    )
    args = parser.parse_args()

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"error: missing dependency ({exc}); pip install torch safetensors", file=sys.stderr)
        return 2

    if not args.input.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    input_sha = _sha256(args.input)
    # weights_only=True refuses arbitrary pickled callables (defense in depth).
    obj = torch.load(args.input, map_location="cpu", weights_only=True)
    if args.key is not None:
        obj = obj[args.key]
    if not isinstance(obj, dict):
        print(f"error: checkpoint is not a state dict (got {type(obj).__name__})", file=sys.stderr)
        return 2

    tensors: dict[str, "torch.Tensor"] = {}
    for name, value in obj.items():
        if not isinstance(value, torch.Tensor):
            print(f"  skip non-tensor entry: {name} ({type(value).__name__})", file=sys.stderr)
            continue
        tensors[name] = value.detach().to(torch.float32).contiguous()

    if not tensors:
        print("error: no tensors found in checkpoint", file=sys.stderr)
        return 2

    metadata = {
        "source": args.input.name,
        "source_sha256": input_sha,
        "conversion_date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "converter": "franken_whisper/scripts/convert_to_safetensors.py",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)

    out_sha = _sha256(args.output)
    print(f"wrote {len(tensors)} tensors -> {args.output}")
    print(f"input  sha256: {input_sha}")
    print(f"output sha256: {out_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

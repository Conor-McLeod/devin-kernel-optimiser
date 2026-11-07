"""Generate harness/shapes.json from a Hugging Face model config.

    uv run --extra shapes python -m harness.gen_shapes Qwen/Qwen3.6-35B-A3B
    uv run python -m harness.gen_shapes path/to/config.json

Emits the linear layers W4A16 would quantise (attention, linear-attention,
dense MLP, MoE experts, shared expert), deduplicated, skipping any whose N/K
break Marlin's legality rules (N % 64, K % 128). Review the output by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import SHAPES_FILE
from .shapes import K_MULTIPLE, N_MULTIPLE


def _load_config(src: str) -> dict:
    p = Path(src)
    if p.exists():
        cfg = json.loads(p.read_text())
    else:
        from huggingface_hub import hf_hub_download

        cfg = json.loads(Path(hf_hub_download(src, "config.json")).read_text())
    return cfg.get("text_config", cfg)


def shapes_from_config(c: dict) -> list[dict]:
    H = c["hidden_size"]
    out: list[tuple[str, int, int]] = []
    nh, nkv = c.get("num_attention_heads"), c.get("num_key_value_heads", c.get("num_attention_heads"))
    hd = c.get("head_dim") or (H // nh if nh else None)
    if nh and hd:
        out.append(("attn_qkv", (nh + 2 * nkv) * hd, H))
        out.append(("attn_o", H, nh * hd))
    # Gated DeltaNet / linear attention (Qwen3-Next style key names).
    lk, lkd = c.get("linear_num_key_heads"), c.get("linear_key_head_dim")
    lv, lvd = c.get("linear_num_value_heads"), c.get("linear_value_head_dim")
    if lk and lkd and lv and lvd:
        out.append(("linattn_qkvz", 2 * lk * lkd + 2 * lv * lvd, H))
        out.append(("linattn_o", H, lv * lvd))
    if c.get("intermediate_size"):
        I = c["intermediate_size"]
        out.append(("mlp_gate_up", 2 * I, H))
        out.append(("mlp_down", H, I))
    if c.get("moe_intermediate_size"):
        I = c["moe_intermediate_size"]
        out.append(("expert_gate_up", 2 * I, H))
        out.append(("expert_down", H, I))
    if c.get("shared_expert_intermediate_size"):
        I = c["shared_expert_intermediate_size"]
        out.append(("shared_gate_up", 2 * I, H))
        out.append(("shared_down", H, I))

    seen, shapes = set(), []
    for name, N, K in out:
        if N % N_MULTIPLE or K % K_MULTIPLE:
            print(f"skip {name}: N={N} K={K} not legal for Marlin", file=sys.stderr)
            continue
        if (N, K) in seen:
            continue
        seen.add((N, K))
        shapes.append({"name": name, "N": N, "K": K})
    return shapes


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    cfg = _load_config(sys.argv[1])
    old = json.loads(SHAPES_FILE.read_text())
    new = {
        "_note": f"generated from {sys.argv[1]}",
        "format": old.get("format", "int4_g128"),
        "ms": old["ms"],
        "shapes": shapes_from_config(cfg),
    }
    SHAPES_FILE.write_text(json.dumps(new, indent=2) + "\n")
    print(json.dumps(new, indent=2))


if __name__ == "__main__":
    main()

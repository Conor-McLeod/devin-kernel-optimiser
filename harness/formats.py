"""Weight formats and the ground-truth dequantisation.

Every submission receives a `QuantWeight` in a canonical, *unpacked* layout and
is free to repack it however it likes in `prepare()` (untimed).

Formats
-------
int4_g128  GPTQ-style symmetric int4 ("uint4b8"): w = (code - 8) * scale,
           one bf16 scale per 128 consecutive K elements. Marlin's main dense path.
mxfp4      OCP MX FP4: E2M1 codes, one E8M0 (power-of-two) scale per 32 K elements.
           Marlin's MXFP4 MoE path on SM90.

Canonical layout (W is [N, K], y = x @ W.T):
  codes  uint8 [N, K]          values 0..15
  scales int4_g128: bf16 [N, K // 128]
         mxfp4:     uint8 [N, K // 32]   (biased exponent, value = 2**(e - 127))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

FORMATS = {
    "int4_g128": {"group": 128},
    "mxfp4": {"group": 32},
}

# E2M1 code -> value. Bit 3 is the sign.
E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


@dataclass
class QuantWeight:
    fmt: str
    N: int
    K: int
    group: int
    codes: torch.Tensor  # uint8 [N, K]
    scales: torch.Tensor  # see module docstring

    @property
    def device(self) -> torch.device:
        return self.codes.device

    def dequant_fp32(self) -> torch.Tensor:
        """Exact dequantised weight [N, K] in fp32. This is the ground truth."""
        if self.fmt == "int4_g128":
            vals = self.codes.to(torch.float32) - 8.0
            s = self.scales.to(torch.float32)
        elif self.fmt == "mxfp4":
            vals = E2M1_LUT.to(self.device)[self.codes.long()]
            s = torch.exp2(self.scales.to(torch.float32) - 127.0)
        else:
            raise ValueError(self.fmt)
        return vals * s.repeat_interleave(self.group, dim=1)

    def nbytes_packed(self) -> int:
        """Bytes a perfectly packed kernel must read for the weight + scales."""
        return self.N * self.K // 2 + self.scales.numel() * self.scales.element_size()


def random_weight(fmt: str, N: int, K: int, gen: torch.Generator, device="cpu") -> QuantWeight:
    group = FORMATS[fmt]["group"]
    assert K % group == 0, f"K={K} must be a multiple of group {group}"
    codes = torch.randint(0, 16, (N, K), generator=gen, dtype=torch.uint8)
    if fmt == "int4_g128":
        scales = (torch.rand(N, K // group, generator=gen) * 0.015 + 0.005).to(torch.bfloat16)
    else:
        scales = torch.randint(119, 125, (N, K // group), generator=gen, dtype=torch.uint8)
    return QuantWeight(fmt, N, K, group, codes.to(device), scales.to(device))


# ---------------------------------------------------------------------------
# Packing helpers that kernels / baselines commonly want.
# ---------------------------------------------------------------------------

def pack_int4_along_k(codes: torch.Tensor) -> torch.Tensor:
    """GPTQ qweight layout: codes [N, K] -> int32 [K // 8, N].

    Element k of a column lives in word k // 8 at bits 4 * (k % 8) (low nibble first).
    """
    N, K = codes.shape
    c = codes.to(torch.int32).t().contiguous().view(K // 8, 8, N)
    out = torch.zeros(K // 8, N, dtype=torch.int32, device=codes.device)
    for i in range(8):
        out |= c[:, i, :] << (4 * i)
    return out


def scales_kn(qw: QuantWeight) -> torch.Tensor:
    """Scales transposed to [K // group, N] (GPTQ / Marlin convention)."""
    return qw.scales.t().contiguous()

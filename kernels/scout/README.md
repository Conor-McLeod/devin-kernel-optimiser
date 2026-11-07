# scout: measurement tools (not a competing kernel)

- `roofline.py` — min bytes / FLOPs / floor time for every (shape, M) on H100 SXM.
- `prof_marlin.py <baseline> NxKxM ...` — runs points under the GPU lock for `ncu --profile-from-start off`.
- `prof_floor.py <baseline> NxKxM ...` — flush-then-call loop under the GPU lock for `nsys` (real kernel time vs harness time).

Example:
    ncu --profile-from-start off --kernel-name regex:'[Mm]arlin' --section SpeedOfLight --section WarpStateStats \
        -o prof python kernels/scout/prof_marlin.py sglang_marlin 12288x2048x16

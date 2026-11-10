<script setup lang="ts">
// Geomean speedup vs sglang Marlin, each agent's best full 55-point run (board "best" view, Nov 7).
const cols = ['Overall', 'Decode (M≤16)', 'Mid (32–256)', 'Prefill (≥512)']
const rows = [
  { name: 'decode', lang: 'CUDA / PTX', v: [1.72, 1.51, 1.80, 1.81], wins: 55 },
  { name: 'cuda-wgmma', lang: 'CUDA / PTX', v: [1.67, 1.15, 1.97, 1.86], wins: 52 },
  { name: 'triton', lang: 'Triton', v: [1.63, 1.25, 1.96, 1.66], wins: 53 },
  { name: 'cute-dsl', lang: 'CuTe DSL', v: [1.56, 1.18, 1.82, 1.66], wins: 52 },
  { name: 'torch_dequant', lang: 'cuBLAS bf16', v: [1.51, 0.91, 1.72, 1.93], wins: 44, ref: 'not W4: full bf16 copy' },
  { name: 'vLLM Machete', lang: 'CUTLASS', v: [1.32, 0.87, 1.49, 1.61], wins: 41, ref: 'Hopper-native ref' },
  { name: 'sglang Marlin', lang: 'CUDA', v: [1.0, 1.0, 1.0, 1.0], wins: null, ref: 'baseline' },
]
// Best agent kernel per column (reference rows excluded).
const best = cols.map((_, i) => Math.max(...rows.filter(r => !r.ref).map(r => r.v[i])))

// Diverging around 1.0: red below Marlin, green above, saturating at 2.0×.
function bg(s: number) {
  if (s < 1) return `rgba(248,113,113,${0.18 + Math.min(1, (1 - s) / 0.15) * 0.5})`
  const t = Math.min(1, (s - 1) / 1.0)
  return `rgba(74,222,128,${0.04 + t * 0.62})`
}
const winBg = (w: number | null) => (w == null ? 'transparent' : `rgba(96,165,250,${0.05 + (w / 55) ** 3 * 0.5})`)
</script>

<template>
  <div class="hm">
    <div class="row head">
      <div>Kernel</div><div>Language</div>
      <div v-for="c in cols" :key="c" class="num">{{ c }}</div>
      <div class="num">Wins</div>
    </div>
    <div v-for="r in rows" :key="r.name" class="row" :class="{ ref: r.ref }">
      <div class="name">{{ r.name }}<span v-if="r.ref" class="sub">{{ r.ref }}</span></div>
      <div class="lang">{{ r.lang }}</div>
      <div
        v-for="(s, i) in r.v" :key="i" class="cell num"
        :class="{ best: !r.ref && s === best[i] }" :style="{ background: bg(s) }"
      >{{ s.toFixed(2) }}×</div>
      <div class="cell num" :style="{ background: winBg(r.wins) }">{{ r.wins == null ? '' : `${r.wins}/55` }}</div>
    </div>
    <div class="legend">
      <span class="sw" style="background: rgba(248,113,113,.6)" /> slower than Marlin
      <span class="sw" style="background: rgba(74,222,128,.12)" /> ~1.1×
      <span class="sw" style="background: rgba(74,222,128,.6)" /> ~1.9×
      <span class="sw ring" /> best agent kernel in column
    </div>
  </div>
</template>

<style scoped>
.hm { font-size: 14px; text-align: left; }
.row { display: grid; grid-template-columns: 1.6fr 1.1fr repeat(4, 1fr) .8fr; gap: 4px; margin-bottom: 4px; align-items: stretch; }
.head { font-size: 12px; opacity: .65; margin-bottom: 6px; }
.name { font-weight: 700; display: flex; flex-direction: column; justify-content: center; }
.ref .name { font-weight: 500; opacity: .8; }
.sub { font-size: 10px; font-weight: 400; opacity: .6; font-style: italic; }
.lang { font-size: 12px; opacity: .7; display: flex; align-items: center; }
.num { text-align: center; }
.cell { padding: 7px 0; border-radius: 5px; font-variant-numeric: tabular-nums; font-weight: 600; }
.ref .cell { font-weight: 400; }
.best { outline: 2px solid rgba(255,255,255,.85); outline-offset: -2px; }
.legend { margin-top: 8px; font-size: 11px; opacity: .7; display: flex; gap: 6px; align-items: center; }
.sw { display: inline-block; width: 14px; height: 10px; border-radius: 2px; margin-left: 10px; }
.sw.ring { outline: 2px solid rgba(255,255,255,.85); outline-offset: -2px; }
</style>

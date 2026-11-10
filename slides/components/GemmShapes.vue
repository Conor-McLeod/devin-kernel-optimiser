<script setup lang="ts">
import { computed, ref } from 'vue'

// From harness/shapes.json (generated from the Qwen/Qwen3.6-35B-A3B config, hidden size 2048)
const blocks = [
  {
    block: 'Attention',
    ops: [
      { name: 'attn_qkv', N: 5120, K: 2048, what: 'Turns each token into a query, key and value' },
      { name: 'attn_o', N: 2048, K: 4096, what: 'Projects the attention output back to 2048 numbers', also: 'linear-attention output' },
    ],
  },
  {
    block: 'Linear attention',
    ops: [
      { name: 'linattn_qkvz', N: 12288, K: 2048, what: 'Same idea for the cheaper linear-attention layers: the widest matrix' },
    ],
  },
  {
    block: 'Mixture of experts',
    ops: [
      { name: 'expert_gate_up', N: 1024, K: 2048, what: 'Each small "expert" MLP expands the token', also: 'shared expert' },
      { name: 'expert_down', N: 2048, K: 512, what: '…and shrinks it back: the smallest matrix' },
    ],
  },
]
const ops = blocks.flatMap(b => b.ops)
const sel = ref('linattn_qkvz')
const op = computed(() => ops.find(o => o.name === sel.value)!)

const ms = [1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
const mi = ref(0)
const M = computed(() => ms[mi.value])

// Drawing: one fixed scale for every shape so sizes compare honestly
const S = 0.03 // px per element
const MIN = 2
const px = (n: number) => Math.max(MIN, n * S)
const G = 24 // gap between boxes
const geo = computed(() => {
  const { N, K } = op.value
  const k = px(K), n = px(N), m = px(M.value)
  // x (M×K) bottom-left, Wᵀ (K×N) top-right, y (M×N) bottom-right
  const x0 = 70, y0 = 24
  return {
    W: { x: x0 + k + G, y: y0, w: n, h: k },
    X: { x: x0, y: y0 + k + G, w: k, h: m },
    Y: { x: x0 + k + G, y: y0 + k + G, w: n, h: m },
  }
})

const RIDGE = 989e12 / 3.35e12 / 4 // ≈ 74: tokens where int4 weights stop being the bottleneck
const stats = computed(() => {
  const { N, K } = op.value
  const wBytes = N * K * 0.5 + (N * K / 128) * 2
  const flops = 2 * M.value * N * K
  return {
    params: N * K,
    wMB: wBytes / 1e6,
    flops,
    memBound: M.value < RIDGE,
  }
})
const fmtN = (n: number) => (n >= 1e9 ? (n / 1e9).toFixed(1) + 'B' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1e3 ? (n / 1e3).toFixed(0) + 'K' : String(n))
</script>

<template>
  <div class="gs select-none">
    <div class="grid grid-cols-[290px_1fr] gap-6">
      <!-- op picker -->
      <div class="flex flex-col gap-2">
        <div v-for="b in blocks" :key="b.block">
          <div class="blk">{{ b.block }}</div>
          <button
            v-for="o in b.ops" :key="o.name"
            class="op" :class="{ on: sel === o.name }"
            @click="sel = o.name"
          >
            <span class="nm">{{ o.name }}</span>
            <span class="dims">{{ o.N }} × {{ o.K }}</span>
          </button>
        </div>
        <div class="desc">
          {{ op.what }}<span v-if="op.also" class="opacity-60"> (also the {{ op.also }}, same shape)</span>
        </div>
      </div>

      <!-- picture -->
      <div>
        <div class="flex items-center gap-3 mb-1">
          <span class="text-sm opacity-70 whitespace-nowrap">Tokens M = <b class="mval">{{ M }}</b></span>
          <input v-model.number="mi" type="range" min="0" :max="ms.length - 1" step="1" class="flex-1">
        </div>

        <svg width="100%" height="320" viewBox="0 0 560 320" class="overflow-visible">
          <!-- Wᵀ -->
          <rect v-bind="{ x: geo.W.x, y: geo.W.y, width: geo.W.w, height: geo.W.h }" class="w" />
          <text :x="geo.W.x + geo.W.w / 2" :y="geo.W.y - 6" class="lbl">N = {{ op.N }}</text>
          <text
            v-if="geo.W.h >= 30 && geo.W.w >= 110"
            :x="geo.W.x + 6" :y="geo.W.y + 16" class="in w-t"
          >W · 4-bit weights</text>
          <text v-else :x="geo.W.x + geo.W.w + 8" :y="geo.W.y + geo.W.h / 2 + 4" class="in w-t">W · 4-bit weights</text>
          <!-- x -->
          <rect v-bind="{ x: geo.X.x, y: geo.X.y, width: geo.X.w, height: geo.X.h }" class="x" />
          <text :x="geo.X.x + geo.X.w / 2" :y="geo.W.y + geo.W.h + 5" class="lbl sm">K = {{ op.K }}</text>
          <text :x="geo.X.x - 6" :y="geo.X.y + Math.max(geo.X.h, 10) / 2 + 4" class="lbl end">M = {{ M }}</text>
          <!-- y -->
          <rect v-bind="{ x: geo.Y.x, y: geo.Y.y, width: geo.Y.w, height: geo.Y.h }" class="y" />
          <!-- K guide from x to W -->
          <line :x1="geo.X.x + geo.X.w" :x2="geo.W.x" :y1="geo.W.y + geo.W.h / 2" :y2="geo.W.y + geo.W.h / 2" class="guide" />
          <!-- legend -->
          <g :transform="`translate(${geo.X.x}, ${Math.min(geo.Y.y + geo.Y.h + 22, 312)})`">
            <rect width="10" height="10" class="x" /><text x="15" y="9" class="leg">x · your tokens (bf16)</text>
            <rect x="160" width="10" height="10" class="y" /><text x="175" y="9" class="leg">y = x · Wᵀ (bf16 out)</text>
          </g>
        </svg>
      </div>
    </div>

    <div class="stats">
      <div><div class="k">Weights to read</div><div class="v">{{ stats.wMB.toFixed(1) }} MB</div></div>
      <div><div class="k">Multiply-adds</div><div class="v">{{ fmtN(stats.flops / 2) }}</div></div>
      <div><div class="k">Each weight is reused</div><div class="v">{{ M }}×</div></div>
      <div class="verdict" :class="stats.memBound ? 'mem' : 'cmp'">
        <template v-if="stats.memBound">Thin <b>x</b>: the GPU mostly waits for <b>W</b> to arrive. <b>Memory-bound</b></template>
        <template v-else>Fat <b>x</b>: plenty of math per weight read. <b>Compute-bound</b></template>
      </div>
    </div>
    <div class="foot">
      5 shapes × 11 values of M = the <b>55 benchmark points</b>. W is fixed by the model; only M changes at runtime.
      The switch from memory-bound to compute-bound happens around M ≈ {{ Math.round(RIDGE) }}.
    </div>
  </div>
</template>

<style scoped>
.gs { --x: #60a5fa; --w: #4ade80; --y: #fbbf24; --mut: rgba(255,255,255,.12); width: 100%; text-align: left; font-size: 14px; }
.blk { font-size: 10px; text-transform: uppercase; letter-spacing: .06em; opacity: .55; margin: 2px 0 3px; }
.op { display: flex; justify-content: space-between; width: 100%; padding: 4px 10px; border: 1px solid var(--mut); border-radius: 7px; margin-bottom: 4px; font-size: 13px; transition: all .15s; }
.op .nm { font-family: ui-monospace, monospace; }
.op .dims { opacity: .55; font-variant-numeric: tabular-nums; }
.op.on { border-color: var(--w); background: rgba(74,222,128,.1); }
.op.on .nm { color: var(--w); }
.desc { font-size: 12px; line-height: 1.35; opacity: .85; margin-top: 4px; min-height: 48px; }
.mval { color: var(--x); font-variant-numeric: tabular-nums; }
input[type=range] { accent-color: var(--x); }
rect { transition: all .35s ease; }
rect.w { fill: rgba(74,222,128,.22); stroke: var(--w); }
rect.x { fill: rgba(96,165,250,.45); stroke: var(--x); }
rect.y { fill: rgba(251,191,36,.35); stroke: var(--y); }
text { transition: all .35s ease; }
.lbl { fill: rgba(255,255,255,.75); font-size: 11px; text-anchor: middle; font-family: ui-monospace, monospace; }
.lbl.sm { dominant-baseline: hanging; }
.lbl.end { text-anchor: end; }
.in { font-size: 11px; }
.w-t { fill: var(--w); }
.leg { fill: rgba(255,255,255,.7); font-size: 11px; }
.guide { stroke: rgba(255,255,255,.18); stroke-dasharray: 3 3; }
.stats { display: grid; grid-template-columns: 130px 130px 150px 1fr; gap: 14px; align-items: center; margin-top: 6px; }
.stats .k { font-size: 11px; opacity: .6; }
.stats .v { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
.verdict { font-size: 13px; padding: 6px 10px; border-radius: 8px; }
.verdict.mem { background: rgba(96,165,250,.14); }
.verdict.cmp { background: rgba(250,204,21,.13); }
.foot { font-size: 11px; opacity: .55; margin-top: 8px; }
</style>

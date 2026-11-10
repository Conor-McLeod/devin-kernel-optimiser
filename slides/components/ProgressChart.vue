<script setup lang="ts">
import { computed, ref } from 'vue'
import runs from './progress.json'

// Every full 55-point, 0-incorrect run the general kernels posted (results.jsonl on the VM).
// decode only appears once it covered all M properly (after the prefill redirect).
type Run = { agent: string; t: number; gm: number; v: string; note: string; run: string }
const data = runs as Run[]

const agents = [
  { name: 'cuda-wgmma', label: 'CUDA / PTX', color: '#60a5fa' },
  { name: 'triton', label: 'Triton', color: '#4ade80' },
  { name: 'cute-dsl', label: 'CuTe DSL', color: '#fbbf24' },
  { name: 'decode', label: 'CUDA', color: '#f472b6' },
]
const REDIRECT_T = 48 // minutes (between decode #48 at 46m and scout #49 at 50m): all agents told to switch to prefill (M ≥ 512)

const W = 820, H = 330
const pad = { l: 48, r: 96, t: 14, b: 36 }
const tMax = Math.ceil(Math.max(...data.map(d => d.t)) / 10) * 10 + 2
const yMin = 0.9, yMax = 1.75
const x = (t: number) => pad.l + (t / tMax) * (W - pad.l - pad.r)
const y = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (H - pad.t - pad.b)
const xTicks = Array.from({ length: Math.floor(tMax / 10) + 1 }, (_, i) => i * 10)
const yTicks = [1.0, 1.2, 1.4, 1.6]

// Best-so-far step line per agent, extended to the right edge.
const series = computed(() => agents.map(a => {
  const rs = data.filter(d => d.agent === a.name).sort((p, q) => p.t - q.t)
  let best = -Infinity
  let path = ''
  const labelled = new Set<string>()
  const seenV = new Set<string>()
  for (const r of rs) {
    if (r.gm < 1) labelled.add(r.run)
    if (r.gm <= best) continue
    if (!seenV.has(r.v)) labelled.add(r.run)
    seenV.add(r.v)
    path += path ? ` H${x(r.t)} V${y(r.gm)}` : `M${x(r.t)} ${y(r.gm)}`
    best = r.gm
  }
  path += ` H${x(tMax)}`
  return { ...a, rs, path, best, labelled }
}))

const hover = ref<Run | null>(null)
const tip = computed(() => {
  const h = hover.value
  if (!h) return null
  const left = Math.min(x(h.t), W - 300)
  return { left, top: y(h.gm) < 120 ? y(h.gm) + 14 : y(h.gm) - 70 }
})
</script>

<template>
  <div class="pc">
    <svg :viewBox="`0 0 ${W} ${H}`" width="100%">
      <g v-for="v in yTicks" :key="v">
        <line :x1="pad.l" :x2="W - pad.r" :y1="y(v)" :y2="y(v)" class="grid" />
        <text :x="pad.l - 8" :y="y(v) + 4" class="ylab">{{ v.toFixed(1) }}×</text>
      </g>
      <g v-for="t in xTicks" :key="t">
        <text :x="x(t)" :y="H - pad.b + 18" class="xlab">{{ t }}</text>
      </g>
      <text :x="(pad.l + W - pad.r) / 2" :y="H - 2" class="xlab">minutes since the board opened</text>

      <!-- references -->
      <line :x1="pad.l" :x2="W - pad.r" :y1="y(1)" :y2="y(1)" class="marlin" />
      <text :x="W - pad.r + 6" :y="y(1) + 4" class="ref">Marlin 1.00×</text>
      <line :x1="pad.l" :x2="W - pad.r" :y1="y(1.32)" :y2="y(1.32)" class="machete" />
      <text :x="W - pad.r + 6" :y="y(1.32) + 4" class="ref">Machete 1.32×</text>

      <line :x1="x(REDIRECT_T)" :x2="x(REDIRECT_T)" :y1="pad.t" :y2="H - pad.b" class="redirect" />
      <text :x="x(REDIRECT_T) + 6" :y="pad.t + 10" class="ann">new goal for everyone: prefill</text>

      <g v-for="s in series" :key="s.name">
        <path :d="s.path" :stroke="s.color" class="step" />
        <circle
          v-for="r in s.rs" :key="r.run" :cx="x(r.t)" :cy="y(Math.max(r.gm, yMin))" r="5"
          :fill="s.color" class="dot" :class="{ on: hover === r }"
          @mouseenter="hover = r" @mouseleave="hover = null"
        />
        <text
          v-for="r in s.rs.filter(r => r.v && s.labelled.has(r.run))" :key="'l' + r.run"
          :x="x(r.t) + 6" :y="y(Math.max(r.gm, yMin)) - 8" class="vlab" :fill="s.color"
        >{{ r.v }}</text>
      </g>

      <!-- the one regression worth calling out -->
      <text :x="x(25.5) + 26" :y="y(0.93) + 4" class="ann">cluster split-K: dropped below Marlin, reworked in v5</text>
    </svg>

    <div v-if="hover && tip" class="tip" :style="{ left: `${(tip.left / W) * 100}%`, top: `${tip.top}px` }">
      <b>{{ hover.agent }}</b> · {{ hover.gm.toFixed(2) }}× · {{ hover.t }} min · run {{ hover.run }}
      <div class="opacity-70">{{ hover.note }}</div>
    </div>

    <div class="legend">
      <span v-for="s in series" :key="s.name">
        <i :style="{ background: s.color }" />{{ s.name }} <span class="opacity-60">({{ s.label }})</span>
        <b :style="{ color: s.color }">{{ s.best.toFixed(2) }}×</b>
      </span>
    </div>
  </div>
</template>

<style scoped>
.pc { position: relative; width: 100%; text-align: left; }
.grid { stroke: rgba(255,255,255,.08); }
.ylab { fill: rgba(255,255,255,.55); font-size: 12px; text-anchor: end; }
.xlab { fill: rgba(255,255,255,.55); font-size: 12px; text-anchor: middle; }
.marlin { stroke: #f87171; stroke-width: 1.5; }
.machete { stroke: rgba(255,255,255,.45); stroke-dasharray: 5 4; }
.ref { fill: rgba(255,255,255,.7); font-size: 12px; }
.step { fill: none; stroke-width: 2.5; opacity: .9; }
.dot { stroke: #111; stroke-width: 1.5; cursor: pointer; transition: r .15s; }
.dot.on { r: 8; }
.vlab { font-size: 11px; font-family: ui-monospace, monospace; opacity: .85; pointer-events: none; }
.redirect { stroke: rgba(255,255,255,.35); stroke-dasharray: 2 4; }
.ann { fill: rgba(255,255,255,.6); font-size: 11px; font-style: italic; }
.tip { position: absolute; max-width: 300px; background: rgba(20,20,24,.95); border: 1px solid rgba(255,255,255,.15); border-radius: 6px; padding: 6px 9px; font-size: 12px; pointer-events: none; }
.legend { display: flex; gap: 22px; justify-content: center; font-size: 13px; margin-top: 2px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.legend b { margin-left: 6px; }
</style>

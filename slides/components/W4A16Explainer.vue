<script setup lang="ts">
import { computed, ref } from 'vue'

const tab = ref(0)
const tabs = ['1 · Shrink the weights', '2 · Why smaller = faster', '3 · The catch']

// ---------- 1. Shrink ----------
// 16 fixed "weights" (one slice of a 128-wide group), roughly normal like real LLM weights
const weights = [0.42, -0.13, 0.07, -0.61, 0.28, 0.91, -0.35, 0.02, -0.18, 0.55, -0.83, 0.11, 0.33, -0.47, 0.69, -0.05]
const bitOptions = [16, 8, 4, 3, 2]
const bits = ref(4)
const quant = computed(() => {
  const b = bits.value
  if (b === 16) return { scale: 0, q: weights.map(() => 0), deq: weights.slice() }
  const qmax = 2 ** (b - 1) - 1
  const scale = Math.max(...weights.map(Math.abs)) / qmax
  const q = weights.map(w => Math.max(-qmax - 1, Math.min(qmax, Math.round(w / scale))))
  return { scale, q, deq: q.map(v => v * scale) }
})
const meanErr = computed(() => {
  const e = weights.reduce((s, w, i) => s + Math.abs(w - quant.value.deq[i]), 0) / weights.length
  const m = weights.reduce((s, w) => s + Math.abs(w), 0) / weights.length
  return (100 * e) / m
})
// bits/weight incl. one bf16 scale per 128 weights
const bitsPerWeight = computed(() => (bits.value === 16 ? 16 : bits.value + 16 / 128))
const modelGB = computed(() => (35e9 * bitsPerWeight.value) / 8 / 1e9)
const levels = computed(() => {
  if (bits.value > 4) return []
  const qmax = 2 ** (bits.value - 1) - 1
  const out: number[] = []
  for (let v = -qmax - 1; v <= qmax; v++) out.push(v * quant.value.scale)
  return out
})
const BAR_W = 36
const H = 190
const y = (v: number) => H / 2 - v * (H / 2 - 6)
const nibble = (q: number) => (q + 8).toString(16).toUpperCase()

// ---------- 2. Faster ----------
const shapes = [
  { name: 'attn_qkv', N: 5120, K: 2048 },
  { name: 'attn_o', N: 2048, K: 4096 },
  { name: 'linattn_qkvz', N: 12288, K: 2048 },
  { name: 'expert_down', N: 2048, K: 512 },
]
const shapeIdx = ref(2)
const mExp = ref(0) // M = 2^mExp
const M = computed(() => 2 ** mExp.value)
const BW = 3.35e12 // H100 HBM bytes/s
const PEAK = 989e12 // H100 dense bf16 FLOP/s
function cost(bytesPerWeight: number) {
  const { N, K } = shapes[shapeIdx.value]
  const m = M.value
  const bytes = N * K * bytesPerWeight + 2 * m * K + 2 * m * N
  const flops = 2 * m * N * K
  const tMem = bytes / BW
  const tCmp = flops / PEAK
  return { t: Math.max(tMem, tCmp) * 1e6, memBound: tMem >= tCmp, bytes, tMem: tMem * 1e6, tCmp: tCmp * 1e6 }
}
const bf16 = computed(() => cost(2))
const w4 = computed(() => cost(0.5 + 2 / 128))
const speedup = computed(() => bf16.value.t / w4.value.t)
const maxT = computed(() => Math.max(bf16.value.t, w4.value.t))
const fmtT = (us: number) => (us < 10 ? us.toFixed(2) : us < 100 ? us.toFixed(1) : us.toFixed(0)) + ' µs'
const fmtMB = (b: number) => (b / 1e6).toFixed(1) + ' MB'

// ---------- 3. The catch ----------
const step = ref(0)
const scale3 = 0.021
const byte = 0x3b
const lo = byte & 0xf // 11
const hi = byte >> 4 // 3
const stages = [
  { title: 'Load from memory', where: 'GPU memory → chip', body: 'One byte holds two 4-bit weights.' },
  { title: 'Split the byte', where: 'registers', body: 'Mask and shift out each 4-bit "nibble".' },
  { title: 'Re-centre', where: 'registers', body: 'Stored as 0…15, meaning −8…7: subtract 8.' },
  { title: 'Scale', where: 'registers', body: 'Multiply by the group\'s shared bf16 scale.' },
  { title: 'Multiply', where: 'tensor cores', body: 'Now bf16 × bf16, the only thing tensor cores do.' },
]
</script>

<template>
  <div class="w4 select-none">
    <div class="tabs">
      <button v-for="(t, i) in tabs" :key="i" :class="{ on: tab === i }" @click="tab = i">{{ t }}</button>
    </div>

    <!-- 1. SHRINK -->
    <div v-if="tab === 0" class="panel grid grid-cols-[1fr_250px] gap-6">
      <div>
        <div class="text-xs opacity-70 mb-1">
          16 weights from one row. Outline = original value, fill = what we store.
        </div>
        <svg :width="weights.length * BAR_W" :height="H + 22" class="overflow-visible">
          <line
            v-for="(l, i) in levels" :key="'l' + i"
            x1="0" :x2="weights.length * BAR_W" :y1="y(l)" :y2="y(l)"
            class="lvl"
          />
          <line x1="0" :x2="weights.length * BAR_W" :y1="H / 2" :y2="H / 2" class="axis" />
          <g v-for="(w, i) in weights" :key="i">
            <rect
              :x="i * BAR_W + 5" :width="BAR_W - 10"
              :y="Math.min(y(quant.deq[i]), H / 2)" :height="Math.abs(y(quant.deq[i]) - H / 2)"
              class="bar"
            />
            <rect
              :x="i * BAR_W + 4" :width="BAR_W - 8"
              :y="Math.min(y(w), H / 2)" :height="Math.abs(y(w) - H / 2)"
              class="orig"
            />
            <text v-if="bits === 4" :x="i * BAR_W + BAR_W / 2" :y="H + 16" class="nib">{{ nibble(quant.q[i]) }}</text>
          </g>
        </svg>
        <div v-if="bits === 4" class="text-xs opacity-70 mt-1">
          Each weight becomes one hex digit (0–F) + the group shares one scale
          = <code>{{ quant.scale.toFixed(3) }}</code>
        </div>
        <div v-else-if="bits <= 3" class="text-xs opacity-70 mt-1">Only {{ 2 ** bits }} levels: visibly lossy.</div>
        <div v-else class="text-xs opacity-70 mt-1">Too many levels to draw: nearly lossless.</div>
      </div>

      <div class="flex flex-col gap-3">
        <div>
          <div class="text-xs opacity-70 mb-1">Bits per weight</div>
          <div class="seg">
            <button v-for="b in bitOptions" :key="b" :class="{ on: bits === b, star: b === 4 }" @click="bits = b">{{ b }}</button>
          </div>
        </div>
        <div class="stat">
          <div class="k">35B-param model</div>
          <div class="v">{{ modelGB.toFixed(1) }} GB</div>
          <div class="meter"><div :style="{ width: (modelGB / 70) * 100 + '%' }" /></div>
        </div>
        <div class="stat">
          <div class="k">Average rounding error</div>
          <div class="v" :class="{ bad: meanErr > 15 }">{{ meanErr.toFixed(1) }}%</div>
        </div>
        <div class="text-xs leading-snug opacity-80">
          <b>W4A16</b> = <b>W</b>eights in <b>4</b> bits, <b>A</b>ctivations (your input tokens) stay <b>16</b>-bit.
          4 bits is the sweet spot: ~4× smaller, accuracy mostly kept.
        </div>
      </div>
    </div>

    <!-- 2. FASTER -->
    <div v-else-if="tab === 1" class="panel">
      <div class="flex items-center gap-4 text-sm mb-3">
        <span class="opacity-70">Layer</span>
        <div class="seg">
          <button v-for="(s, i) in shapes" :key="s.name" :class="{ on: shapeIdx === i }" @click="shapeIdx = i">{{ s.name }}</button>
        </div>
      </div>
      <div class="flex items-center gap-4 mb-4">
        <span class="text-sm opacity-70 w-44">Tokens at once (M): <b class="text-white opacity-100">{{ M }}</b></span>
        <input v-model.number="mExp" type="range" min="0" max="12" step="1" class="flex-1">
        <span class="text-xs opacity-60 w-56">{{ M <= 16 ? 'chatbot replying (decode)' : M <= 256 ? 'many users batched' : 'reading a long prompt (prefill)' }}</span>
      </div>

      <div v-for="row in [{ label: 'bf16 weights (16-bit)', c: bf16, cls: 'b16' }, { label: 'W4A16 (4-bit weights)', c: w4, cls: 'b4' }]" :key="row.label" class="row">
        <div class="lab">{{ row.label }}</div>
        <div class="track">
          <div class="fill" :class="row.cls" :style="{ width: (row.c.t / maxT) * 100 + '%' }">
            <span>{{ fmtT(row.c.t) }}</span>
          </div>
        </div>
        <div class="tag" :class="row.c.memBound ? 'mem' : 'cmp'">{{ row.c.memBound ? 'waiting on memory' : 'busy doing math' }}</div>
      </div>

      <div class="grid grid-cols-[auto_1fr] gap-6 mt-4 items-center">
        <div class="big">{{ speedup.toFixed(2) }}×</div>
        <div class="text-sm leading-snug opacity-85">
          <template v-if="w4.memBound">
            With few tokens the GPU spends almost all its time <b>fetching weights</b> ({{ fmtMB(bf16.bytes) }} → {{ fmtMB(w4.bytes) }}).
            Fewer bytes to fetch = directly faster. This is why int4 models feel snappy when chatting.
          </template>
          <template v-else>
            With lots of tokens, each weight is reused {{ M }} times, so the GPU is limited by <b>arithmetic</b>, not memory.
            Smaller weights stop helping, and unpacking them becomes extra work.
          </template>
        </div>
      </div>
      <div class="text-[10px] opacity-50 mt-2">Ideal H100 lower bounds: 3.35 TB/s memory, 989 TFLOP/s bf16. Real kernels are slower.</div>
    </div>

    <!-- 3. CATCH -->
    <div v-else class="panel">
      <div class="text-sm opacity-85 mb-3">
        GPU tensor cores can't multiply 4-bit numbers by 16-bit ones. So every weight has to be <b>unpacked back to bf16
        on the fly</b>, inside the chip, billions of times per second, without slowing anything down.
      </div>
      <div class="pipe">
        <div v-for="(s, i) in stages" :key="i" class="stage" :class="{ on: step === i, done: step > i }" @click="step = i">
          <div class="where">{{ s.where }}</div>
          <div class="ttl">{{ s.title }}</div>
        </div>
      </div>

      <div class="demo">
        <div class="desc">{{ stages[step].body }}</div>
        <div class="vals">
          <template v-if="step === 0">
            <div class="bitbox">
              <span v-for="(b, i) in byte.toString(2).padStart(8, '0')" :key="i" :class="i < 4 ? 'h' : 'l'">{{ b }}</span>
            </div>
            <span class="opacity-60 ml-3">= 0x3B</span>
          </template>
          <template v-else-if="step === 1">
            <div class="chip h">{{ hi.toString(2).padStart(4, '0') }} → {{ hi }}</div>
            <div class="chip l">{{ lo.toString(2).padStart(4, '0') }} → {{ lo }}</div>
          </template>
          <template v-else-if="step === 2">
            <div class="chip h">{{ hi }} − 8 = {{ hi - 8 }}</div>
            <div class="chip l">{{ lo }} − 8 = {{ lo - 8 }}</div>
          </template>
          <template v-else-if="step === 3">
            <div class="chip h">{{ hi - 8 }} × {{ scale3 }} = {{ ((hi - 8) * scale3).toFixed(3) }}</div>
            <div class="chip l">{{ lo - 8 }} × {{ scale3 }} = {{ ((lo - 8) * scale3).toFixed(3) }}</div>
          </template>
          <template v-else>
            <div class="chip h">{{ ((hi - 8) * scale3).toFixed(3) }} × x</div>
            <div class="chip l">{{ ((lo - 8) * scale3).toFixed(3) }} × x</div>
            <span class="opacity-60 ml-2">→ accumulate into y</span>
          </template>
        </div>
        <button class="next" @click="step = (step + 1) % stages.length">{{ step === stages.length - 1 ? '↺' : 'Next →' }}</button>
      </div>

      <div class="text-sm mt-4 leading-snug">
        <b>Marlin</b> does this dance with GPU features from 2020. Our agents' job: redo it with the H100's newer
        hardware (async bulk loads, bigger tensor-core instructions) so the unpacking hides behind the memory traffic.
      </div>
    </div>
  </div>
</template>

<style scoped>
.w4 { --acc: #4ade80; --acc2: #60a5fa; --warn: #f87171; --mut: rgba(255,255,255,.12); font-size: 14px; text-align: left; width: 100%; }
.panel { width: 100%; }
.tabs { display: flex; gap: 6px; margin-bottom: 12px; }
.tabs button { padding: 4px 12px; border-radius: 999px; border: 1px solid var(--mut); opacity: .65; font-size: 13px; white-space: nowrap; flex: none; }
.tabs button.on { opacity: 1; border-color: var(--acc); color: var(--acc); }
.panel { min-height: 330px; }
.seg { display: inline-flex; border: 1px solid var(--mut); border-radius: 8px; overflow: hidden; }
.seg button { padding: 3px 10px; font-size: 13px; border-right: 1px solid var(--mut); position: relative; }
.seg button:last-child { border-right: none; }
.seg button.on { background: var(--acc); color: #111; font-weight: 600; }
.seg button.star:not(.on)::after { content: ''; position: absolute; top: 3px; right: 3px; width: 5px; height: 5px; border-radius: 50%; background: var(--acc); }
.lvl { stroke: rgba(255,255,255,.1); stroke-dasharray: 2 3; }
.axis { stroke: rgba(255,255,255,.35); }
.bar { fill: var(--acc); opacity: .85; transition: all .25s; }
.orig { fill: none; stroke: rgba(255,255,255,.7); stroke-width: 1; stroke-dasharray: 3 2; }
.nib { fill: var(--acc); font-family: ui-monospace, monospace; font-size: 12px; text-anchor: middle; }
.stat { border-left: 3px solid var(--acc2); padding-left: 10px; }
.stat .k { font-size: 11px; opacity: .65; }
.stat .v { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.stat .v.bad { color: var(--warn); }
.meter { height: 5px; background: var(--mut); border-radius: 3px; margin-top: 3px; }
.meter div { height: 100%; background: var(--acc2); border-radius: 3px; transition: width .25s; }
.row { display: grid; grid-template-columns: 170px 1fr 140px; gap: 12px; align-items: center; margin: 8px 0; }
.lab { font-size: 13px; opacity: .8; }
.track { height: 30px; background: rgba(255,255,255,.05); border-radius: 6px; }
.fill { height: 100%; border-radius: 6px; display: flex; align-items: center; padding-left: 8px; min-width: 70px; transition: width .3s; font-size: 12px; font-weight: 600; color: #111; }
.fill.b16 { background: #94a3b8; }
.fill.b4 { background: var(--acc); }
.tag { font-size: 11px; padding: 2px 8px; border-radius: 999px; text-align: center; }
.tag.mem { background: rgba(96,165,250,.18); color: var(--acc2); }
.tag.cmp { background: rgba(250,204,21,.15); color: #facc15; }
.big { font-size: 44px; font-weight: 800; color: var(--acc); font-variant-numeric: tabular-nums; min-width: 130px; }
input[type=range] { accent-color: var(--acc); }
.pipe { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
.stage { border: 1px solid var(--mut); border-radius: 8px; padding: 6px 8px; cursor: pointer; opacity: .5; transition: all .2s; }
.stage.done { opacity: .75; border-color: rgba(74,222,128,.35); }
.stage.on { opacity: 1; border-color: var(--acc); background: rgba(74,222,128,.08); }
.stage .where { font-size: 10px; text-transform: uppercase; letter-spacing: .05em; opacity: .6; }
.stage .ttl { font-size: 13px; font-weight: 600; }
.demo { display: grid; grid-template-columns: 1fr auto; grid-template-rows: auto auto; gap: 6px 12px; margin-top: 12px; padding: 12px; border-radius: 10px; background: rgba(255,255,255,.04); min-height: 92px; }
.desc { font-size: 13px; opacity: .8; }
.vals { display: flex; align-items: center; gap: 10px; font-family: ui-monospace, monospace; font-size: 18px; grid-row: 2; }
.next { grid-row: 1 / span 2; grid-column: 2; align-self: center; padding: 6px 14px; border-radius: 8px; background: var(--acc); color: #111; font-weight: 600; font-size: 13px; }
.bitbox { display: inline-flex; gap: 2px; }
.bitbox span { width: 22px; text-align: center; border-radius: 4px; padding: 2px 0; }
.bitbox .h, .chip.h { background: rgba(96,165,250,.2); color: var(--acc2); }
.bitbox .l, .chip.l { background: rgba(74,222,128,.18); color: var(--acc); }
.chip { padding: 3px 10px; border-radius: 6px; }
</style>

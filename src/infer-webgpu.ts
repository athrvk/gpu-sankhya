// WebGPU backend behind the same shape as infer-cpu's forward(), batched.
// Importing this module never touches `navigator.gpu` at load time — all
// device/shader setup happens lazily on first `run`/`parseBatch` call.

import type { LoadedWeights } from "./weights.ts";

export interface BackendResult {
  bio: Float32Array; // (batch*length*3)
  cls: Float32Array; // (batch*length*nCls)
}

const EMBED_WGSL = /* wgsl */ `
struct Params { batch: u32, length: u32, embedDim: u32, vocab: u32, stride: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> charIds: array<i32>;
@group(0) @binding(2) var<storage, read> embed: array<f32>;
@group(0) @binding(3) var<storage, read_write> xOut: array<f32>; // (batch, C, L)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * p.stride + gid.x;
  let total = p.batch * p.length;
  if (idx >= total) { return; }
  let b = idx / p.length;
  let t = idx % p.length;
  let cid = u32(charIds[idx]);
  for (var c: u32 = 0u; c < p.embedDim; c = c + 1u) {
    xOut[(b * p.embedDim + c) * p.length + t] = embed[cid * p.embedDim + c];
  }
}
`;

const CONV_WGSL = /* wgsl */ `
struct Params { batch: u32, length: u32, cIn: u32, cOut: u32, k: u32, padding: u32, dilation: u32, relu: u32, residual: u32, stride: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>; // (batch, cIn, L)
@group(0) @binding(2) var<storage, read> w: array<f32>; // (cOut, cIn, k)
@group(0) @binding(3) var<storage, read> bias: array<f32>; // (cOut)
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // (batch, cOut, L)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * p.stride + gid.x;
  let total = p.batch * p.cOut * p.length;
  if (idx >= total) { return; }
  let t = idx % p.length;
  let rest = idx / p.length;
  let oc = rest % p.cOut;
  let b = rest / p.cOut;

  var acc: f32 = bias[oc];
  for (var ic: u32 = 0u; ic < p.cIn; ic = ic + 1u) {
    let xBase = (b * p.cIn + ic) * p.length;
    let wBase = (oc * p.cIn + ic) * p.k;
    for (var tap: u32 = 0u; tap < p.k; tap = tap + 1u) {
      let srcT: i32 = i32(t) + i32(tap * p.dilation) - i32(p.padding);
      if (srcT >= 0 && srcT < i32(p.length)) {
        acc = acc + w[wBase + tap] * x[xBase + u32(srcT)];
      }
    }
  }
  if (p.relu != 0u && acc < 0.0) { acc = 0.0; }
  if (p.residual != 0u) {
    acc = acc + x[(b * p.cIn + oc) * p.length + t];
  }
  out[(b * p.cOut + oc) * p.length + t] = acc;
}
`;

const HEAD_WGSL = /* wgsl */ `
struct Params { batch: u32, length: u32, cIn: u32, nOut: u32, stride: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>; // (batch, cIn, L)
@group(0) @binding(2) var<storage, read> w: array<f32>; // (nOut, cIn)
@group(0) @binding(3) var<storage, read> bias: array<f32>; // (nOut)
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // (batch, L, nOut)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * p.stride + gid.x;
  let total = p.batch * p.length * p.nOut;
  if (idx >= total) { return; }
  let o = idx % p.nOut;
  let rest = idx / p.nOut;
  let t = rest % p.length;
  let b = rest / p.length;

  var acc: f32 = bias[o];
  for (var ic: u32 = 0u; ic < p.cIn; ic = ic + 1u) {
    acc = acc + w[o * p.cIn + ic] * x[(b * p.cIn + ic) * p.length + t];
  }
  out[(b * p.length + t) * p.nOut + o] = acc;
}
`;

/** Cheap synchronous check: the WebGPU API is present on `navigator`. This
 * does not guarantee an adapter is actually available -- headless browsers
 * and some CI/sandboxed environments expose `navigator.gpu` but return null
 * from `requestAdapter()`. Use `probeWebGPU()` for an authoritative check. */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== "undefined" && !!(navigator as any).gpu;
}

let probePromise: Promise<boolean> | null = null;

/** Authoritative (but async) check: resolves false immediately if
 * `navigator.gpu` is missing, otherwise awaits `requestAdapter()` and
 * resolves to whether an adapter was actually obtained. The result is
 * cached at module scope, so repeated calls only probe once. */
export function probeWebGPU(): Promise<boolean> {
  if (probePromise) return probePromise;
  probePromise = (async () => {
    if (!isWebGPUAvailable()) return false;
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  })();
  return probePromise;
}

interface GPUState {
  device: GPUDevice;
  embedModule: GPUShaderModule;
  convModule: GPUShaderModule;
  headModule: GPUShaderModule;
  embedBuf: GPUBuffer;
  convBufs: { w: GPUBuffer; b: GPUBuffer }[];
  bioBuf: { w: GPUBuffer; b: GPUBuffer };
  clsBuf: { w: GPUBuffer; b: GPUBuffer };
}

export class WebGPUBackend {
  private weights: LoadedWeights;
  private state: GPUState | null = null;

  constructor(weights: LoadedWeights) {
    this.weights = weights;
  }

  private async init(): Promise<GPUState> {
    if (this.state) return this.state;
    if (!isWebGPUAvailable()) throw new Error("WebGPU is not available in this environment");
    const adapter = await (navigator as any).gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter available");
    const device: GPUDevice = await adapter.requestDevice();

    const mkBuf = (data: Float32Array, usage: number) => {
      const buf = device.createBuffer({ size: Math.max(4, data.byteLength), usage, mappedAtCreation: true });
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    };
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    const w = this.weights;
    const convBufs = w.conv.map((l) => ({ w: mkBuf(l.w.data, STORAGE), b: mkBuf(l.b, STORAGE) }));

    this.state = {
      device,
      embedModule: device.createShaderModule({ code: EMBED_WGSL }),
      convModule: device.createShaderModule({ code: CONV_WGSL }),
      headModule: device.createShaderModule({ code: HEAD_WGSL }),
      embedBuf: mkBuf(w.embed.data, STORAGE),
      convBufs,
      bioBuf: { w: mkBuf(w.bio.w.data, STORAGE), b: mkBuf(w.bio.b, STORAGE) },
      clsBuf: { w: mkBuf(w.cls.w.data, STORAGE), b: mkBuf(w.cls.b, STORAGE) },
    };
    return this.state;
  }

  /** Run the forward pass for a batch of equal-length char-id rows.
   * charIds is (batch*length) row-major int32. */
  async run(charIds: Int32Array, batch: number, length: number): Promise<BackendResult> {
    const st = await this.init();
    const { device } = st;
    const w = this.weights;
    const embedDim = w.embed.shape[1];

    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

    const charBuf = device.createBuffer({ size: charIds.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Int32Array(charBuf.getMappedRange()).set(charIds);
    charBuf.unmap();

    const mkUniform = (arr: Uint32Array) => {
      const buf = device.createBuffer({ size: arr.byteLength, usage: UNIFORM, mappedAtCreation: true });
      new Uint32Array(buf.getMappedRange()).set(arr);
      buf.unmap();
      return buf;
    };

    // WebGPU caps dispatchWorkgroups() at device.limits.maxComputeWorkgroupsPerDimension
    // (commonly 65535) *per dimension*. A single 1D dispatch of
    // ceil(total/64) workgroups silently exceeds that for large batches
    // (e.g. batch=500 * length~72 * nOut=120 / 64 = 67500 workgroups for the
    // cls head alone), which is a validation error -- the dispatch is
    // dropped and its output buffer is left zero-initialized, corrupting
    // every downstream read of it. Spread workgroups across x and y instead
    // (workgroup_size is 1 in y/z) and have the shader recombine
    // global_invocation_id.{x,y} into the flat element index via `stride`
    // (= workgroupsX * 64), so no dimension ever exceeds the device limit.
    const maxWorkgroupsPerDim = device.limits?.maxComputeWorkgroupsPerDimension ?? 65535;
    const dispatchDims = (total: number): { x: number; y: number; stride: number } => {
      const totalWorkgroups = Math.max(1, Math.ceil(total / 64));
      const x = Math.min(totalWorkgroups, maxWorkgroupsPerDim);
      const y = Math.ceil(totalWorkgroups / x);
      return { x, y, stride: x * 64 };
    };

    const dispatch = (pipeline: GPUComputePipeline, entries: GPUBindGroupEntry[], dims: { x: number; y: number }) => {
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(dims.x, dims.y);
      pass.end();
      device.queue.submit([encoder.finish()]);
    };

    // embed gather
    let cur = device.createBuffer({ size: batch * embedDim * length * 4, usage: STORAGE });
    {
      const dims = dispatchDims(batch * length);
      const params = mkUniform(new Uint32Array([batch, length, embedDim, w.embed.shape[0], dims.stride]));
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: st.embedModule, entryPoint: "main" } });
      dispatch(
        pipeline,
        [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: charBuf } },
          { binding: 2, resource: { buffer: st.embedBuf } },
          { binding: 3, resource: { buffer: cur } },
        ],
        dims,
      );
    }

    let cIn = embedDim;
    const convLayers = w.conv;
    const convPipeline = device.createComputePipeline({ layout: "auto", compute: { module: st.convModule, entryPoint: "main" } });
    for (let li = 0; li < convLayers.length; li++) {
      const layer = convLayers[li];
      const [cOut, , k] = layer.w.shape;
      const dilation = layer.dilation;
      const padding = layer.padding;
      const outBuf = device.createBuffer({ size: batch * cOut * length * 4, usage: STORAGE });
      const dims = dispatchDims(batch * cOut * length);
      const params = mkUniform(new Uint32Array([batch, length, cIn, cOut, k, padding, dilation, 1, layer.residual ? 1 : 0, dims.stride]));
      dispatch(
        convPipeline,
        [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: cur } },
          { binding: 2, resource: { buffer: st.convBufs[li].w } },
          { binding: 3, resource: { buffer: st.convBufs[li].b } },
          { binding: 4, resource: { buffer: outBuf } },
        ],
        dims,
      );
      cur = outBuf;
      cIn = cOut;
    }

    const headPipeline = device.createComputePipeline({ layout: "auto", compute: { module: st.headModule, entryPoint: "main" } });
    const runHead = async (headBuf: { w: GPUBuffer; b: GPUBuffer }, nOut: number) => {
      const outBuf = device.createBuffer({ size: batch * length * nOut * 4, usage: STORAGE });
      const dims = dispatchDims(batch * length * nOut);
      const params = mkUniform(new Uint32Array([batch, length, cIn, nOut, dims.stride]));
      dispatch(
        headPipeline,
        [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: cur } },
          { binding: 2, resource: { buffer: headBuf.w } },
          { binding: 3, resource: { buffer: headBuf.b } },
          { binding: 4, resource: { buffer: outBuf } },
        ],
        dims,
      );
      const readBuf = device.createBuffer({ size: batch * length * nOut * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, batch * length * nOut * 4);
      device.queue.submit([encoder.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      return data;
    };

    const bio = await runHead(st.bioBuf, w.bio.w.shape[0]);
    const cls = await runHead(st.clsBuf, w.cls.w.shape[0]);
    return { bio, cls };
  }
}

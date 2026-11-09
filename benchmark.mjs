#!/usr/bin/env node
/**
 * GPU teaching benchmark: CPU sequential vs CPU worker threads vs OpenCL GPU.
 *
 * Node.js >= 20. No package is required for the synthetic-image CPU-only run.
 * OpenCL is loaded dynamically through @node-3d/opencl, so a missing OpenCL ICD
 * never gets mislabeled as GPU execution.
 */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { Worker } from 'node:worker_threads';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const KERNEL_PATH = path.join(HERE, 'kernels', 'bilateral_filter.cl');
const KERNEL_SOURCE = fs.readFileSync(KERNEL_PATH, 'utf8');

/* One knob controls both data size and neighbourhood size. */
const WORKLOADS = {
  1: { width: 256, height: 256, radius: 1, label: 'Intro / smoke test' },
  2: { width: 512, height: 512, radius: 2, label: 'Light' },
  3: { width: 768, height: 768, radius: 3, label: 'Medium' },
  4: { width: 1024, height: 1024, radius: 4, label: 'Heavy' },
  5: { width: 1280, height: 1280, radius: 5, label: 'Very heavy' },
};

const DEFAULT_THREADS = Math.max(1, (os.availableParallelism?.() ?? os.cpus().length) - 1);

function parseArgs(argv) {
  const config = {
    workload: 2,
    threads: DEFAULT_THREADS,
    chunkRows: 8,
    localX: 16,
    localY: 16,
    globalX: null,
    globalY: null,
    repeats: 3,
    warmups: 1,
    seed: 20261109,
    input: null,
    outputDir: path.resolve(process.cwd(), 'benchmark-results'),
    buildOptions: '',
    skipGpu: false,
    requireGpu: false,
    help: false,
  };

  const valueOptions = new Set([
    '--workload', '--threads', '--chunk-rows', '--local-x', '--local-y', '--global-x', '--global-y',
    '--repeats', '--warmups', '--seed', '--input', '--output-dir', '--build-options',
  ]);

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      config.help = true;
      continue;
    }
    if (arg === '--skip-gpu') {
      config.skipGpu = true;
      continue;
    }
    if (arg === '--require-gpu') {
      config.requireGpu = true;
      continue;
    }
    if (!valueOptions.has(arg)) {
      throw new Error(`Unknown argument: ${arg}. Use --help to see supported options.`);
    }
    const value = argv[i + 1];
    if (value === undefined || value.startsWith('--')) {
      throw new Error(`Missing value for ${arg}.`);
    }
    i += 1;

    switch (arg) {
      case '--workload': config.workload = Number(value); break;
      case '--threads': config.threads = Number(value); break;
      case '--chunk-rows': config.chunkRows = Number(value); break;
      case '--local-x': config.localX = Number(value); break;
      case '--local-y': config.localY = Number(value); break;
      case '--global-x': config.globalX = Number(value); break;
      case '--global-y': config.globalY = Number(value); break;
      case '--repeats': config.repeats = Number(value); break;
      case '--warmups': config.warmups = Number(value); break;
      case '--seed': config.seed = Number(value); break;
      case '--input': config.input = path.resolve(value); break;
      case '--output-dir': config.outputDir = path.resolve(value); break;
      case '--build-options': config.buildOptions = value; break;
      default: throw new Error(`Unhandled argument: ${arg}`);
    }
  }

  if (!(config.workload in WORKLOADS)) throw new RangeError('--workload must be an integer from 1 to 5.');
  for (const [key, value, min, max] of [
    ['threads', config.threads, 1, 256],
    ['chunk-rows', config.chunkRows, 1, 4096],
    ['local-x', config.localX, 1, 1024],
    ['local-y', config.localY, 1, 1024],
    ...(config.globalX === null ? [] : [['global-x', config.globalX, 1, 100000000]]),
    ...(config.globalY === null ? [] : [['global-y', config.globalY, 1, 100000000]]),
    ['repeats', config.repeats, 1, 100],
    ['warmups', config.warmups, 0, 20],
  ]) {
    if (!Number.isInteger(value) || value < min || value > max) {
      throw new RangeError(`--${key} must be an integer from ${min} to ${max}.`);
    }
  }
  if (!Number.isInteger(config.seed) || config.seed < 0 || config.seed > 0xffffffff) {
    throw new RangeError('--seed must be an unsigned 32-bit integer.');
  }
  return config;
}

function printHelp() {
  console.log(`
GPU COMPUTE TEACHING LAB — Bilateral Filter Benchmark

Usage:
  node benchmark.mjs [options]

Workload (the single calculation-heaviness control):
  --workload 1..5       Selects image resolution AND filter radius (default: 2).
                        1 = 256x256 / radius 1; 5 = 1280x1280 / radius 5.

CPU parallelization:
  --threads N           Worker-thread count (default: logical CPUs minus one).
  --chunk-rows N        Rows assigned per atomic scheduling chunk (default: 8).

OpenCL GPU launch configuration:
  --local-x N           Work-items per work-group in X (default: 16).
  --local-y N           Work-items per work-group in Y (default: 16).
  --global-x N          Requested global X work-item extent (default: image width).
  --global-y N          Requested global Y work-item extent (default: image height).
                        Must cover the image; extents are padded to local-size multiples.

Measurement / reproducibility:
  --warmups N           Untimed warmup runs per backend (default: 1).
  --repeats N           Timed repetitions; median is the primary comparison (default: 3).
  --seed N              Deterministic synthetic-image seed (default: 20261109).
  --input FILE          Optional PNG/JPEG image instead of a generated image.
  --build-options TEXT  OpenCL compiler options, e.g. "-cl-fast-relaxed-math".
  --output-dir DIR      Root folder for timestamped run artifacts.
  --skip-gpu            Do not initialize OpenCL; run the two CPU cases only.
  --require-gpu         Exit with an error if OpenCL is unavailable.
  --help                Show this help.

Examples:
  node benchmark.mjs --workload 1 --threads 4 --local-x 8 --local-y 8
  node benchmark.mjs --workload 4 --threads 8 --local-x 16 --local-y 16 --repeats 5
  node benchmark.mjs --workload 5 --build-options "-cl-fast-relaxed-math"
`);
}

function createRunId() {
  const now = new Date();
  const stamp = now.toISOString().replaceAll(':', '-').replaceAll('.', '-');
  return `${stamp}-pid${process.pid}`;
}

function round(value, digits = 3) {
  if (!Number.isFinite(value)) return null;
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function mb(bytes) {
  return bytes / (1024 * 1024);
}

function gb(bytes) {
  return bytes / 1e9;
}

function hashBuffer(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function makeSyntheticImage(width, height, seed) {
  const pixels = new Uint8Array(width * height * 4);
  let state = seed >>> 0;
  const nextRandom = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state >>> 24;
  };

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4;
      const noise = nextRandom() & 31;
      const region = ((x >> 5) ^ (y >> 5)) & 1;
      pixels[i + 0] = (x * 3 + y + noise + region * 48) & 255;
      pixels[i + 1] = (x + y * 2 + (noise >> 1) + region * 24) & 255;
      pixels[i + 2] = ((x >> 1) + y * 3 + noise * 2 + region * 72) & 255;
      pixels[i + 3] = 255;
    }
  }
  return { width, height, data: pixels, source: 'synthetic', sourcePath: null };
}

async function loadImage(filename) {
  if (!fs.existsSync(filename)) throw new Error(`Input image does not exist: ${filename}`);
  const extension = path.extname(filename).toLowerCase();
  const file = fs.readFileSync(filename);

  if (extension === '.png') {
    const { PNG } = await import('pngjs');
    const decoded = PNG.sync.read(file);
    return {
      width: decoded.width,
      height: decoded.height,
      data: new Uint8Array(decoded.data),
      source: 'file',
      sourcePath: filename,
    };
  }
  if (extension === '.jpg' || extension === '.jpeg') {
    const jpegModule = await import('jpeg-js');
    const decoded = jpegModule.default.decode(file, { useTArray: true });
    return {
      width: decoded.width,
      height: decoded.height,
      data: new Uint8Array(decoded.data),
      source: 'file',
      sourcePath: filename,
    };
  }
  throw new Error('Unsupported image. Use PNG, JPG, or JPEG.');
}

function clampByte(value) {
  return Math.trunc(Math.max(0, Math.min(255, value)));
}

function bilateralFilterRows(input, output, width, height, radius, sigmaSpaceSquared, sigmaColorSquared, yStart, yEnd) {
  for (let y = yStart; y < yEnd; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const centerIndex = (y * width + x) * 4;
      const centerR = input[centerIndex + 0];
      const centerG = input[centerIndex + 1];
      const centerB = input[centerIndex + 2];

      let weightedR = 0;
      let weightedG = 0;
      let weightedB = 0;
      let totalWeight = 0;

      for (let dy = -radius; dy <= radius; dy += 1) {
        const sampleY = Math.max(0, Math.min(height - 1, y + dy));
        for (let dx = -radius; dx <= radius; dx += 1) {
          const sampleX = Math.max(0, Math.min(width - 1, x + dx));
          const sampleIndex = (sampleY * width + sampleX) * 4;
          const sampleR = input[sampleIndex + 0];
          const sampleG = input[sampleIndex + 1];
          const sampleB = input[sampleIndex + 2];

          const spatialDistanceSquared = dx * dx + dy * dy;
          const dr = sampleR - centerR;
          const dg = sampleG - centerG;
          const db = sampleB - centerB;
          const colorDistanceSquared = dr * dr + dg * dg + db * db;
          const exponent =
            -(spatialDistanceSquared / sigmaSpaceSquared)
            -(colorDistanceSquared / sigmaColorSquared);
          const weight = Math.exp(exponent);

          weightedR += sampleR * weight;
          weightedG += sampleG * weight;
          weightedB += sampleB * weight;
          totalWeight += weight;
        }
      }

      output[centerIndex + 0] = clampByte(weightedR / totalWeight);
      output[centerIndex + 1] = clampByte(weightedG / totalWeight);
      output[centerIndex + 2] = clampByte(weightedB / totalWeight);
      output[centerIndex + 3] = 255;
    }
  }
}

class WorkerPool {
  constructor(workerCount) {
    this.workers = [];
    this.pending = new Map();
    this.runSequence = 0;
    this.readyWorkers = 0;
    this.closed = false;

    this.readyPromise = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });

    for (let index = 0; index < workerCount; index += 1) {
      const worker = new Worker(new URL('./cpu-worker.mjs', import.meta.url));
      worker.on('message', (message) => this.#handleMessage(message));
      worker.on('error', (error) => this.#handleWorkerError(error));
      worker.on('exit', (code) => {
        if (!this.closed && code !== 0) {
          this.#handleWorkerError(new Error(`CPU worker exited with code ${code}.`));
        }
      });
      this.workers.push(worker);
    }
  }

  static async create(workerCount) {
    const pool = new WorkerPool(workerCount);
    await pool.readyPromise;
    return pool;
  }

  #handleMessage(message) {
    if (message?.type === 'ready') {
      this.readyWorkers += 1;
      if (this.readyWorkers === this.workers.length) this.resolveReady();
      return;
    }

    const pending = this.pending.get(message?.runId);
    if (!pending) return;
    if (message.type === 'error') {
      this.pending.delete(message.runId);
      pending.reject(new Error(message.error || 'Unknown worker error.'));
      return;
    }
    if (message.type === 'done') {
      pending.remaining -= 1;
      if (pending.remaining === 0) {
        this.pending.delete(message.runId);
        pending.resolve();
      }
    }
  }

  #handleWorkerError(error) {
    if (this.readyWorkers < this.workers.length) this.rejectReady(error);
    for (const [runId, pending] of this.pending) {
      this.pending.delete(runId);
      pending.reject(error);
    }
  }

  async run(sharedInput, sharedOutput, width, height, radius, sigmaSpaceSquared, sigmaColorSquared, chunkRows) {
    const runId = ++this.runSequence;
    const controlBuffer = new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT);
    const job = new Promise((resolve, reject) => {
      this.pending.set(runId, { remaining: this.workers.length, resolve, reject });
    });

    for (const worker of this.workers) {
      worker.postMessage({
        type: 'run',
        runId,
        inputBuffer: sharedInput,
        outputBuffer: sharedOutput,
        controlBuffer,
        width,
        height,
        radius,
        sigmaSpaceSquared,
        sigmaColorSquared,
        chunkRows,
      });
    }
    await job;
  }

  async close() {
    if (this.closed) return;
    this.closed = true;
    await Promise.all(this.workers.map((worker) => worker.terminate().catch(() => undefined)));
  }
}

function summarizeSamples(samples) {
  if (!samples.length) return null;
  const sorted = [...samples].sort((a, b) => a.wallMs - b.wallMs);
  const values = samples.map((sample) => sample.wallMs);
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / values.length;
  const median = sorted.length % 2
    ? sorted[(sorted.length - 1) / 2].wallMs
    : (sorted[sorted.length / 2 - 1].wallMs + sorted[sorted.length / 2].wallMs) / 2;
  const avg = (field) => samples.reduce((sum, sample) => sum + (sample[field] ?? 0), 0) / samples.length;

  return {
    repetitions: samples.length,
    meanMs: round(mean),
    medianMs: round(median),
    minMs: round(sorted[0].wallMs),
    maxMs: round(sorted[sorted.length - 1].wallMs),
    stdDevMs: round(Math.sqrt(variance)),
    cpuUserMsMean: round(avg('cpuUserMs')),
    cpuSystemMsMean: round(avg('cpuSystemMs')),
    cpuTotalMsMean: round(avg('cpuTotalMs')),
    cpuCoreUtilizationPercentMean: round(avg('cpuCoreUtilizationPercent')),
    heapUsedDeltaMBMean: round(avg('heapUsedDeltaMB')),
    rssDeltaMBMean: round(avg('rssDeltaMB')),
    externalDeltaMBMean: round(avg('externalDeltaMB')),
    arrayBuffersDeltaMBMean: round(avg('arrayBuffersDeltaMB')),
    peakRssMBMax: samples.some((sample) => Number.isFinite(sample.peakRssMB))
      ? round(Math.max(...samples.map((sample) => sample.peakRssMB ?? 0)))
      : null,
  };
}

async function runMeasured(fn, warmups, repeats) {
  for (let i = 0; i < warmups; i += 1) await fn();
  const samples = [];

  for (let repetition = 0; repetition < repeats; repetition += 1) {
    const memoryBefore = process.memoryUsage();
    const cpuBefore = process.cpuUsage();
    const startedAt = performance.now();

    await fn();

    const wallMs = performance.now() - startedAt;
    const cpu = process.cpuUsage(cpuBefore);
    const memoryAfter = process.memoryUsage();
    const resourceAfter = process.resourceUsage?.();
    const cpuUserMs = cpu.user / 1000;
    const cpuSystemMs = cpu.system / 1000;
    const cpuTotalMs = cpuUserMs + cpuSystemMs;

    samples.push({
      repetition: repetition + 1,
      wallMs,
      cpuUserMs,
      cpuSystemMs,
      cpuTotalMs,
      cpuCoreUtilizationPercent: wallMs > 0 ? (cpuTotalMs / wallMs) * 100 : 0,
      heapUsedDeltaMB: mb(memoryAfter.heapUsed - memoryBefore.heapUsed),
      rssDeltaMB: mb(memoryAfter.rss - memoryBefore.rss),
      externalDeltaMB: mb(memoryAfter.external - memoryBefore.external),
      arrayBuffersDeltaMB: mb(memoryAfter.arrayBuffers - memoryBefore.arrayBuffers),
      peakRssMB: Number.isFinite(resourceAfter?.maxRSS) ? resourceAfter.maxRSS / 1024 : null,
      memoryBefore: {
        rssMB: round(mb(memoryBefore.rss)),
        heapUsedMB: round(mb(memoryBefore.heapUsed)),
        externalMB: round(mb(memoryBefore.external)),
        arrayBuffersMB: round(mb(memoryBefore.arrayBuffers)),
      },
      memoryAfter: {
        rssMB: round(mb(memoryAfter.rss)),
        heapUsedMB: round(mb(memoryAfter.heapUsed)),
        externalMB: round(mb(memoryAfter.external)),
        arrayBuffersMB: round(mb(memoryAfter.arrayBuffers)),
      },
    });
  }

  return { samples, summary: summarizeSamples(samples) };
}

function compareOutputs(reference, candidate, width, height) {
  if (reference.length !== candidate.length) {
    return { valid: false, reason: 'Output buffers have different lengths.' };
  }
  let absoluteErrorSum = 0;
  let squaredErrorSum = 0;
  let maxAbsoluteError = 0;
  let differingChannels = 0;
  let differingPixels = 0;

  for (let pixel = 0; pixel < width * height; pixel += 1) {
    const base = pixel * 4;
    let pixelDiffers = false;
    for (let channel = 0; channel < 4; channel += 1) {
      const difference = Math.abs(reference[base + channel] - candidate[base + channel]);
      absoluteErrorSum += difference;
      squaredErrorSum += difference * difference;
      if (difference > maxAbsoluteError) maxAbsoluteError = difference;
      if (difference !== 0) {
        differingChannels += 1;
        pixelDiffers = true;
      }
    }
    if (pixelDiffers) differingPixels += 1;
  }

  const channelCount = reference.length;
  const mae = absoluteErrorSum / channelCount;
  const rmse = Math.sqrt(squaredErrorSum / channelCount);
  const psnr = rmse === 0 ? null : 20 * Math.log10(255 / rmse);
  return {
    valid: true,
    exactMatch: differingChannels === 0,
    maxAbsoluteError,
    meanAbsoluteError: round(mae, 6),
    rootMeanSquareError: round(rmse, 6),
    psnrDb: psnr === null ? 'infinite' : round(psnr, 3),
    differingChannels,
    differingPixels,
    differingPixelPercent: round((differingPixels / (width * height)) * 100, 6),
    tolerance1Pass: maxAbsoluteError <= 1,
  };
}

function safeDeviceInfo(cl, device) {
  const fields = [
    ['name', 'DEVICE_NAME'],
    ['vendor', 'DEVICE_VENDOR'],
    ['driverVersion', 'DRIVER_VERSION'],
    ['deviceVersion', 'DEVICE_VERSION'],
    ['openclCVersion', 'DEVICE_OPENCL_C_VERSION'],
    ['deviceType', 'DEVICE_TYPE'],
    ['computeUnits', 'DEVICE_MAX_COMPUTE_UNITS'],
    ['maxClockFrequencyMHz', 'DEVICE_MAX_CLOCK_FREQUENCY'],
    ['maxWorkGroupSize', 'DEVICE_MAX_WORK_GROUP_SIZE'],
    ['maxWorkItemSizes', 'DEVICE_MAX_WORK_ITEM_SIZES'],
    ['globalMemoryBytes', 'DEVICE_GLOBAL_MEM_SIZE'],
    ['localMemoryBytes', 'DEVICE_LOCAL_MEM_SIZE'],
  ];
  const info = {};
  if (typeof cl.getDeviceInfo !== 'function') return { queryAvailable: false };

  for (const [field, constantName] of fields) {
    const constant = cl[constantName];
    if (constant === undefined) continue;
    try {
      const value = cl.getDeviceInfo(device, constant);
      if (typeof value === 'bigint') info[field] = value.toString();
      else if (ArrayBuffer.isView(value)) info[field] = Array.from(value);
      else if (Array.isArray(value)) info[field] = value;
      else if (value && typeof value === 'object') info[field] = String(value);
      else info[field] = value;
    } catch {
      // Some device properties are not implemented by every ICD/binding build.
    }
  }
  info.queryAvailable = true;
  return info;
}

function safeProgramBuildLog(cl, program, device) {
  if (typeof cl.getProgramBuildInfo !== 'function' || cl.PROGRAM_BUILD_LOG === undefined) return null;
  try {
    const value = cl.getProgramBuildInfo(program, device, cl.PROGRAM_BUILD_LOG);
    return typeof value === 'string' ? value : String(value ?? '');
  } catch (error) {
    return `Build log query unavailable: ${error?.message || error}`;
  }
}

function writeBinaryFromUnknownValue(filename, value) {
  if (Buffer.isBuffer(value)) {
    fs.writeFileSync(filename, value);
    return value.length;
  }
  if (value instanceof ArrayBuffer) {
    const bytes = Buffer.from(value);
    fs.writeFileSync(filename, bytes);
    return bytes.length;
  }
  if (ArrayBuffer.isView(value)) {
    const bytes = Buffer.from(value.buffer, value.byteOffset, value.byteLength);
    fs.writeFileSync(filename, bytes);
    return bytes.length;
  }
  if (Array.isArray(value) && value.length > 0) {
    return writeBinaryFromUnknownValue(filename, value[0]);
  }
  return 0;
}

function tryExportOpenCLBinary(cl, program, filename) {
  if (typeof cl.getProgramInfo !== 'function' || cl.PROGRAM_BINARIES === undefined) {
    return { status: 'not-exposed-by-binding', explanation: 'This binding did not expose getProgramInfo/PROGRAM_BINARIES.' };
  }
  try {
    const value = cl.getProgramInfo(program, cl.PROGRAM_BINARIES);
    const bytes = writeBinaryFromUnknownValue(filename, value);
    if (bytes > 0) return { status: 'saved', path: filename, bytes };
    return {
      status: 'exposed-but-no-binary-value-returned',
      explanation: 'The binding exposed the query but did not return a binary buffer in the attempted overload.',
    };
  } catch (error) {
    return { status: 'query-failed', explanation: error?.message || String(error) };
  }
}

async function runOpenCLBenchmark({ input, output, width, height, radius, sigmaSpaceSquared, sigmaColorSquared, config, artifactDir }) {
  let cl;
  let context;
  let device;
  let queue;
  let program;
  let kernel;
  let deviceInput;
  let deviceOutput;
  const buildLogPath = path.join(artifactDir, 'opencl-build.log');
  const buildLogLines = [];
  const setupStarted = performance.now();
  const runtimeInitStarted = performance.now();

  try {
    cl = await import('@node-3d/opencl');
    ({ context, device } = cl.quickStart(true));
    if (!context || !device) throw new Error('OpenCL quickStart did not return a context and device.');

    /* Queue profiling is requested when supported; host timing remains the fallback. */
    const profilingProperty = cl.QUEUE_PROFILING_ENABLE ?? 2;
    try {
      queue = cl.createCommandQueue(context, device, profilingProperty);
      buildLogLines.push(`Command queue: profiling requested (property ${profilingProperty}).`);
    } catch (queueError) {
      queue = cl.createCommandQueue(context, device);
      buildLogLines.push(`Command queue: default properties; profiling queue creation failed: ${queueError.message}`);
    }

    const runtimeInitMs = performance.now() - runtimeInitStarted;
    const deviceInfo = safeDeviceInfo(cl, device);
    buildLogLines.push('OpenCL platform/device selected by quickStart(): first available device.');
    buildLogLines.push(`Device information: ${JSON.stringify(deviceInfo, null, 2)}`);
    buildLogLines.push(`Compiler options: ${config.buildOptions || '(default)'}`);
    buildLogLines.push(`Kernel source: ${path.join(artifactDir, 'bilateral_filter.cl')}`);

    const compileStarted = performance.now();
    program = cl.createProgramWithSource(context, KERNEL_SOURCE);
    if (config.buildOptions) cl.buildProgram(program, config.buildOptions);
    else cl.buildProgram(program);
    const compileMs = performance.now() - compileStarted;

    const buildLog = safeProgramBuildLog(cl, program, device);
    if (buildLog) buildLogLines.push(`\n--- OpenCL compiler build log ---\n${buildLog}`);
    else buildLogLines.push('\nCompiler build log query unavailable or empty.');

    const kernelAndBufferSetupStarted = performance.now();
    kernel = cl.createKernel(program, 'bilateral_filter');
    deviceInput = cl.createBuffer(context, cl.MEM_READ_ONLY, input.byteLength);
    deviceOutput = cl.createBuffer(context, cl.MEM_WRITE_ONLY, output.byteLength);

    cl.setKernelArg(kernel, 0, 'uchar*', deviceInput);
    cl.setKernelArg(kernel, 1, 'uchar*', deviceOutput);
    cl.setKernelArg(kernel, 2, 'uint', width);
    cl.setKernelArg(kernel, 3, 'uint', height);
    cl.setKernelArg(kernel, 4, 'uint', radius);
    cl.setKernelArg(kernel, 5, 'float', sigmaSpaceSquared);
    cl.setKernelArg(kernel, 6, 'float', sigmaColorSquared);
    const kernelAndBufferSetupMs = performance.now() - kernelAndBufferSetupStarted;

    const requestedGlobalX = config.globalX ?? width;
    const requestedGlobalY = config.globalY ?? height;
    if (requestedGlobalX < width || requestedGlobalY < height) {
      throw new RangeError(`Global work-item extent ${requestedGlobalX}x${requestedGlobalY} must cover the image ${width}x${height}.`);
    }
    const globalX = Math.ceil(requestedGlobalX / config.localX) * config.localX;
    const globalY = Math.ceil(requestedGlobalY / config.localY) * config.localY;
    const globalSize = [globalX, globalY];
    const localSize = [config.localX, config.localY];

    const setupMs = performance.now() - setupStarted;
    const samples = [];

    const executeOne = () => {
      const uploadStart = performance.now();
      cl.enqueueWriteBuffer(queue, deviceInput, true, 0, input.byteLength, input);
      const uploadMs = performance.now() - uploadStart;

      const kernelStart = performance.now();
      cl.enqueueNDRangeKernel(queue, kernel, 2, undefined, globalSize, localSize);
      cl.finish(queue); // Required: OpenCL kernel launches are asynchronous.
      const kernelMs = performance.now() - kernelStart;

      const downloadStart = performance.now();
      cl.enqueueReadBuffer(queue, deviceOutput, true, 0, output.byteLength, output);
      const downloadMs = performance.now() - downloadStart;
      return { uploadMs, kernelMs, downloadMs, pipelineMs: uploadMs + kernelMs + downloadMs };
    };

    for (let i = 0; i < config.warmups; i += 1) executeOne();

    for (let repetition = 0; repetition < config.repeats; repetition += 1) {
      const memoryBefore = process.memoryUsage();
      const cpuBefore = process.cpuUsage();
      const startedAt = performance.now();
      const phaseTimes = executeOne();
      const wallMs = performance.now() - startedAt;
      const cpu = process.cpuUsage(cpuBefore);
      const memoryAfter = process.memoryUsage();
      const resourceAfter = process.resourceUsage?.();
      const cpuUserMs = cpu.user / 1000;
      const cpuSystemMs = cpu.system / 1000;
      const cpuTotalMs = cpuUserMs + cpuSystemMs;
      samples.push({
        repetition: repetition + 1,
        wallMs,
        ...phaseTimes,
        cpuUserMs,
        cpuSystemMs,
        cpuTotalMs,
        cpuCoreUtilizationPercent: wallMs > 0 ? (cpuTotalMs / wallMs) * 100 : 0,
        heapUsedDeltaMB: mb(memoryAfter.heapUsed - memoryBefore.heapUsed),
        rssDeltaMB: mb(memoryAfter.rss - memoryBefore.rss),
        externalDeltaMB: mb(memoryAfter.external - memoryBefore.external),
        arrayBuffersDeltaMB: mb(memoryAfter.arrayBuffers - memoryBefore.arrayBuffers),
        peakRssMB: Number.isFinite(resourceAfter?.maxRSS) ? resourceAfter.maxRSS / 1024 : null,
        memoryBefore: { rssMB: round(mb(memoryBefore.rss)), heapUsedMB: round(mb(memoryBefore.heapUsed)), externalMB: round(mb(memoryBefore.external)), arrayBuffersMB: round(mb(memoryBefore.arrayBuffers)) },
        memoryAfter: { rssMB: round(mb(memoryAfter.rss)), heapUsedMB: round(mb(memoryAfter.heapUsed)), externalMB: round(mb(memoryAfter.external)), arrayBuffersMB: round(mb(memoryAfter.arrayBuffers)) },
      });
    }

    const binaryPath = path.join(artifactDir, 'bilateral_filter.opencl.bin');
    const binaryExport = tryExportOpenCLBinary(cl, program, binaryPath);
    buildLogLines.push(`\nProgram binary export: ${JSON.stringify(binaryExport)}`);
    fs.writeFileSync(buildLogPath, `${buildLogLines.join('\n')}\n`);

    const pipelineSummary = summarizeSamples(samples.map((sample) => ({ ...sample, wallMs: sample.pipelineMs })));
    const kernelSummary = summarizeSamples(samples.map((sample) => ({ ...sample, wallMs: sample.kernelMs })));
    const uploadSummary = summarizeSamples(samples.map((sample) => ({ ...sample, wallMs: sample.uploadMs })));
    const downloadSummary = summarizeSamples(samples.map((sample) => ({ ...sample, wallMs: sample.downloadMs })));
    const wallSummary = summarizeSamples(samples);
    const phaseMean = (field) => round(samples.reduce((sum, sample) => sum + sample[field], 0) / samples.length);

    return {
      status: 'completed',
      deviceInfo,
      setupMs: round(setupMs),
      runtimeInitMs: round(runtimeInitMs),
      compileMs: round(compileMs),
      kernelAndBufferSetupMs: round(kernelAndBufferSetupMs),
      launch: {
        dimensions: 2,
        requestedGlobalSize: [requestedGlobalX, requestedGlobalY],
        globalSize,
        localSize,
        workGroupCount: [globalX / config.localX, globalY / config.localY],
        totalWorkItemsLaunched: globalX * globalY,
        usefulWorkItems: width * height,
        paddedWorkItems: (globalX * globalY) - (width * height),
      },
      memory: {
        deviceInputBufferBytes: input.byteLength,
        deviceOutputBufferBytes: output.byteLength,
        totalDeviceBufferAllocationBytes: input.byteLength + output.byteLength,
        note: 'This is the size of buffers allocated by this program, not a measurement of total VRAM consumption by the driver or GPU.',
      },
      timing: {
        summary: wallSummary,
        pipelineSummary,
        kernelSummary,
        uploadSummary,
        downloadSummary,
        uploadMsMean: phaseMean('uploadMs'),
        kernelHostSynchronizedMsMean: phaseMean('kernelMs'),
        downloadMsMean: phaseMean('downloadMs'),
        pipelineMsMean: phaseMean('pipelineMs'),
        samples,
        timingNote: 'Kernel time is host wall time from enqueue through cl.finish(); it includes driver/queue synchronization. It is not a hardware-counter event duration.',
      },
      binaryExport,
      buildLogPath,
    };
  } catch (error) {
    buildLogLines.push(`\nOPENCL ERROR:\n${error?.stack || error}`);
    fs.writeFileSync(buildLogPath, `${buildLogLines.join('\n')}\n`);
    return {
      status: 'unavailable-or-failed',
      error: error?.stack || String(error),
      buildLogPath,
    };
  } finally {
    if (cl) {
      try { if (kernel) cl.releaseKernel(kernel); } catch { /* best-effort cleanup */ }
      try { if (program) cl.releaseProgram(program); } catch { /* best-effort cleanup */ }
      try { if (deviceInput) cl.releaseMemObject(deviceInput); } catch { /* best-effort cleanup */ }
      try { if (deviceOutput) cl.releaseMemObject(deviceOutput); } catch { /* best-effort cleanup */ }
      try { if (queue) cl.releaseCommandQueue(queue); } catch { /* best-effort cleanup */ }
      try { if (context) cl.releaseContext(context); } catch { /* best-effort cleanup */ }
    }
  }
}

function writeRgbaFile(filename, data) {
  fs.writeFileSync(filename, Buffer.from(data.buffer, data.byteOffset, data.byteLength));
}

async function tryWritePng(filename, width, height, data) {
  try {
    const { PNG } = await import('pngjs');
    const image = new PNG({ width, height });
    image.data = Buffer.from(data);
    fs.writeFileSync(filename, PNG.sync.write(image));
    return true;
  } catch {
    return false;
  }
}

function csvEscape(value) {
  const text = value === null || value === undefined ? '' : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function writeCsv(filename, rows) {
  const headers = Object.keys(rows[0] ?? {});
  const lines = [headers.join(',')];
  for (const row of rows) lines.push(headers.map((header) => csvEscape(row[header])).join(','));
  fs.writeFileSync(filename, `${lines.join('\n')}\n`);
}

function makeCsvRows(benchmarks, metrics, comparisons) {
  const cases = [
    ['CPU Sequential', benchmarks.cpuSequential, 'CPU Sequential'],
    ['CPU Parallel (worker_threads)', benchmarks.cpuParallel, 'CPU Parallel (worker_threads)'],
    ['GPU OpenCL', benchmarks.openclGpu, 'GPU OpenCL (transfer + kernel pipeline)'],
  ];
  return cases.map(([name, result, speedupKey]) => {
    const summary = name === 'GPU OpenCL'
      ? (result?.timing?.pipelineSummary ?? null)
      : (result?.timing?.summary ?? result?.summary ?? null);
    const medianMs = summary?.medianMs ?? null;
    return {
      benchmark: name,
      status: result?.status ?? (summary ? 'completed' : 'failed'),
      median_ms: medianMs,
      mean_ms: summary?.meanMs ?? null,
      min_ms: summary?.minMs ?? null,
      max_ms: summary?.maxMs ?? null,
      stddev_ms: summary?.stdDevMs ?? null,
      megapixels_per_second: medianMs ? round((metrics.pixelCount / 1e6) / (medianMs / 1000), 3) : null,
      neighbour_samples_per_second: medianMs ? round(metrics.neighbourSamples / (medianMs / 1000), 0) : null,
      cpu_user_ms_mean: summary?.cpuUserMsMean ?? null,
      cpu_system_ms_mean: summary?.cpuSystemMsMean ?? null,
      cpu_core_utilization_percent: summary?.cpuCoreUtilizationPercentMean ?? null,
      rss_delta_mb_mean: summary?.rssDeltaMBMean ?? null,
      peak_rss_mb_max: summary?.peakRssMBMax ?? null,
      upload_ms_mean: result?.timing?.uploadMsMean ?? null,
      kernel_ms_mean: result?.timing?.kernelHostSynchronizedMsMean ?? null,
      kernel_ms_median: result?.timing?.kernelSummary?.medianMs ?? null,
      download_ms_mean: result?.timing?.downloadMsMean ?? null,
      speedup_vs_sequential: comparisons?.speedupVsSequential?.[speedupKey] ?? null,
      kernel_speedup_vs_sequential: name === 'GPU OpenCL'
        ? (comparisons?.speedupVsSequential?.['GPU OpenCL (kernel phase)'] ?? null)
        : null,
    };
  });
}

function buildMarkdownReport(profile) {
  const { config, workload, metrics, benchmarks, comparisons, validation, artifacts } = profile;
  const rows = [
    ['CPU Sequential', benchmarks.cpuSequential?.timing?.summary?.medianMs, benchmarks.cpuSequential?.status],
    ['CPU Parallel', benchmarks.cpuParallel?.timing?.summary?.medianMs, benchmarks.cpuParallel?.status],
    ['GPU OpenCL', benchmarks.openclGpu?.timing?.pipelineSummary?.medianMs, benchmarks.openclGpu?.status],
  ];
  const table = [
    '| Execution path | Median time (ms) | Status |',
    '|---|---:|---|',
    ...rows.map(([name, time, status]) => `| ${name} | ${time ?? '—'} | ${status ?? 'not run'} |`),
  ].join('\n');

  const speedups = Object.entries(comparisons.speedupVsSequential ?? {})
    .map(([name, speedup]) => `- ${name}: ${speedup === null ? 'not available' : `${speedup}×`}`)
    .join('\n');
  const validationLines = Object.entries(validation ?? {})
    .map(([name, result]) => `- ${name}: ${result?.exactMatch ? 'bit-identical' : result?.tolerance1Pass ? `within ±1 channel value (max error ${result.maxAbsoluteError})` : result?.reason ?? `max error ${result?.maxAbsoluteError ?? 'n/a'}`}`)
    .join('\n');

  return `# GPU Compute Benchmark Report\n\n` +
    `Run: \`${profile.runId}\`  \nGenerated: ${profile.generatedAt}\n\n` +
    `## Workload\n\n` +
    `- Profile: ${workload.label} (level ${config.workload})\n` +
    `- Image: ${workload.width} × ${workload.height} (${metrics.pixelCount.toLocaleString()} pixels)\n` +
    `- Bilateral radius: ${workload.radius} (window ${(workload.radius * 2 + 1)} × ${(workload.radius * 2 + 1)})\n` +
    `- Neighbour samples: ${metrics.neighbourSamples.toLocaleString()}\n` +
    `- Timed repetitions: ${config.repeats}; warmups: ${config.warmups}\n\n` +
    `## Results\n\n${table}\n\n` +
    `GPU median above is the end-to-end measured pipeline time (host-to-device transfer + synchronized kernel + device-to-host transfer). The GPU kernel-only host-synchronized median is ${benchmarks.openclGpu?.timing?.kernelSummary?.medianMs ?? 'not available'} ms. Compilation/setup is reported separately and is not included in repeated pipeline time.\n\n` +
    `## Speedup versus CPU Sequential\n\n${speedups || '- Not available'}\n\n` +
    `## Correctness\n\n${validationLines || '- No comparisons available'}\n\n` +
    `## Interpretation notes\n\n` +
    `- CPU parallelism uses Node.js worker threads and shared memory; it is not just asynchronous I/O.\n` +
    `- OpenCL work-items are pixel computations. A 2-D NDRange launches ${benchmarks.openclGpu?.launch?.totalWorkItemsLaunched ?? 'n/a'} work-items, with ${benchmarks.openclGpu?.launch?.workGroupCount?.join(' × ') ?? 'n/a'} work-groups.\n` +
    `- GPU total device-buffer allocation requested by this program: ${benchmarks.openclGpu?.memory?.totalDeviceBufferAllocationBytes ?? 'n/a'} bytes. This is not a measurement of all VRAM occupied by the driver.\n` +
    `- Memory deltas are Node.js process snapshots before and after each repetition. They are not a sampled peak and do not include a direct VRAM-used counter.\n` +
    `- The logical data-volume model counts RGB neighbour reads and RGBA output writes; it does not measure physical DRAM transactions or cache misses.\n` +
    `- OpenCL kernel timing uses host wall time around enqueue + cl.finish(). Hardware event profiling is not reported unless a future binding adapter exposes it.\n` +
    `- A GPU result is only present when an OpenCL context/device actually executed the kernel. CPU fallback is never labeled GPU.\n\n` +
    `## Saved artifacts\n\n` +
    Object.entries(artifacts).map(([name, value]) => `- ${name}: \`${value}\``).join('\n') +
    `\n`;
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  if (config.help) {
    printHelp();
    return;
  }

  const runId = createRunId();
  const artifactDir = path.join(config.outputDir, runId);
  fs.mkdirSync(artifactDir, { recursive: true });
  fs.mkdirSync(path.join(artifactDir, 'source'), { recursive: true });
  fs.writeFileSync(path.join(artifactDir, 'bilateral_filter.cl'), KERNEL_SOURCE);
  fs.copyFileSync(fileURLToPath(import.meta.url), path.join(artifactDir, 'source', 'benchmark.mjs'));
  fs.copyFileSync(path.join(HERE, 'cpu-worker.mjs'), path.join(artifactDir, 'source', 'cpu-worker.mjs'));

  const preset = WORKLOADS[config.workload];
  const image = config.input
    ? await loadImage(config.input)
    : makeSyntheticImage(preset.width, preset.height, config.seed);
  const { width, height, data: input } = image;
  const outputSequential = new Uint8Array(input.length);
  const outputParallelShared = new SharedArrayBuffer(input.length);
  const outputParallel = new Uint8Array(outputParallelShared);
  const outputOpenCL = new Uint8Array(input.length);

  const radius = preset.radius;
  const sigmaSpaceSquared = Math.max(1, 2 * radius * radius);
  const sigmaColorSquared = 7200;
  const pixelCount = width * height;
  const samplesPerPixel = (radius * 2 + 1) ** 2;
  const neighbourSamples = pixelCount * samplesPerPixel;
  const logicalBytesPerPixel = samplesPerPixel * 3 + 4;
  const logicalBytes = pixelCount * logicalBytesPerPixel;

  const inputBinaryPath = path.join(artifactDir, 'input.rgba');
  writeRgbaFile(inputBinaryPath, input);

  console.log('\n==============================================================');
  console.log(' GPU COMPUTE TEACHING LAB — BILATERAL FILTER');
  console.log('==============================================================');
  console.log(`Workload: ${config.workload}/5 — ${preset.label}`);
  console.log(`Image: ${width} × ${height} (${pixelCount.toLocaleString()} pixels)`);
  console.log(`Filter: radius ${radius}; ${(radius * 2 + 1)} × ${(radius * 2 + 1)} neighbourhood`);
  console.log(`Neighbour evaluations: ${neighbourSamples.toLocaleString()}`);
  console.log(`CPU workers: ${config.threads}; chunk rows: ${config.chunkRows}`);
  console.log(`OpenCL local size: ${config.localX} × ${config.localY}`);
  console.log(`Warmups: ${config.warmups}; repetitions: ${config.repeats}`);
  console.log('--------------------------------------------------------------\n');

  const cpuSequentialRaw = await runMeasured(() => {
    bilateralFilterRows(input, outputSequential, width, height, radius, sigmaSpaceSquared, sigmaColorSquared, 0, height);
  }, config.warmups, config.repeats);
  const cpuSequential = {
    status: 'completed',
    timing: cpuSequentialRaw,
    throughput: {
      megapixelsPerSecondMedian: round((pixelCount / 1e6) / (cpuSequentialRaw.summary.medianMs / 1000)),
      neighbourSamplesPerSecondMedian: round(neighbourSamples / (cpuSequentialRaw.summary.medianMs / 1000), 0),
    },
  };
  console.log(`[1/3] CPU Sequential: ${cpuSequentialRaw.summary.medianMs} ms median`);

  let cpuParallel = { status: 'failed', error: 'Not run.' };
  let workerPool;
  try {
    const parallelInputBuffer = new SharedArrayBuffer(input.byteLength);
    new Uint8Array(parallelInputBuffer).set(input);
    workerPool = await WorkerPool.create(config.threads);
    const cpuParallelRaw = await runMeasured(async () => {
      await workerPool.run(
        parallelInputBuffer,
        outputParallelShared,
        width,
        height,
        radius,
        sigmaSpaceSquared,
        sigmaColorSquared,
        config.chunkRows,
      );
    }, config.warmups, config.repeats);
    cpuParallel = {
      status: 'completed',
      threads: config.threads,
      chunkRows: config.chunkRows,
      timing: cpuParallelRaw,
      throughput: {
        megapixelsPerSecondMedian: round((pixelCount / 1e6) / (cpuParallelRaw.summary.medianMs / 1000)),
        neighbourSamplesPerSecondMedian: round(neighbourSamples / (cpuParallelRaw.summary.medianMs / 1000), 0),
      },
    };
    console.log(`[2/3] CPU Parallel (${config.threads} workers): ${cpuParallelRaw.summary.medianMs} ms median`);
  } catch (error) {
    cpuParallel = { status: 'failed', error: error?.stack || String(error) };
    console.error(`[2/3] CPU Parallel failed: ${error.message}`);
  } finally {
    if (workerPool) await workerPool.close();
  }

  let openclGpu;
  if (config.skipGpu) {
    openclGpu = { status: 'skipped', reason: 'Disabled with --skip-gpu.' };
    fs.writeFileSync(path.join(artifactDir, 'opencl-build.log'), 'OpenCL was skipped with --skip-gpu.\n');
    console.log('[3/3] GPU OpenCL: skipped by request');
  } else {
    openclGpu = await runOpenCLBenchmark({
      input,
      output: outputOpenCL,
      width,
      height,
      radius,
      sigmaSpaceSquared,
      sigmaColorSquared,
      config,
      artifactDir,
    });
    if (openclGpu.status === 'completed') {
      console.log(`[3/3] GPU OpenCL: ${openclGpu.timing.pipelineSummary.medianMs} ms median pipeline; kernel ${openclGpu.timing.kernelHostSynchronizedMsMean} ms mean`);
    } else {
      console.warn(`[3/3] GPU OpenCL: ${openclGpu.status} — ${openclGpu.error || openclGpu.reason}`);
    }
  }

  const validation = {
    cpuParallelVsSequential: cpuParallel.status === 'completed'
      ? compareOutputs(outputSequential, outputParallel, width, height)
      : { valid: false, reason: 'CPU parallel result unavailable.' },
    openclGpuVsSequential: openclGpu.status === 'completed'
      ? compareOutputs(outputSequential, outputOpenCL, width, height)
      : { valid: false, reason: 'OpenCL GPU result unavailable.' },
  };

  const sequentialMedian = cpuSequential.timing.summary.medianMs;
  const parallelMedian = cpuParallel.timing?.summary?.medianMs ?? null;
  const gpuKernelMedian = openclGpu.status === 'completed'
    ? openclGpu.timing.kernelSummary.medianMs
    : null;
  const gpuPipelineMedian = openclGpu.status === 'completed'
    ? openclGpu.timing.pipelineSummary.medianMs
    : null;
  const speedupVsSequential = {
    'CPU Parallel (worker_threads)': parallelMedian ? round(sequentialMedian / parallelMedian, 3) : null,
    'GPU OpenCL (kernel phase)': gpuKernelMedian ? round(sequentialMedian / gpuKernelMedian, 3) : null,
    'GPU OpenCL (transfer + kernel pipeline)': gpuPipelineMedian ? round(sequentialMedian / gpuPipelineMedian, 3) : null,
  };

  const metrics = {
    pixelCount,
    samplesPerPixel,
    neighbourSamples,
    logicalBytesPerPixel,
    logicalBytes,
    logicalDataVolumeGB: round(gb(logicalBytes), 6),
    logicalDataVolumeNote: 'Model: 3 RGB channel reads per neighbour sample plus 4 RGBA output bytes per pixel; not physical DRAM traffic.',
  };

  const artifacts = {
    inputRgbaBinary: inputBinaryPath,
    sequentialOutputRgbaBinary: path.join(artifactDir, 'output-cpu-sequential.rgba'),
    parallelOutputRgbaBinary: path.join(artifactDir, 'output-cpu-parallel.rgba'),
    openclOutputRgbaBinary: openclGpu.status === 'completed' ? path.join(artifactDir, 'output-opencl-gpu.rgba') : 'not-generated',
    openclKernelSource: path.join(artifactDir, 'bilateral_filter.cl'),
    openclBuildLog: path.join(artifactDir, 'opencl-build.log'),
    benchmarkSource: path.join(artifactDir, 'source', 'benchmark.mjs'),
    cpuWorkerSource: path.join(artifactDir, 'source', 'cpu-worker.mjs'),
    openclProgramBinary: openclGpu.binaryExport?.status === 'saved'
      ? openclGpu.binaryExport.path
      : `not-exported (${openclGpu.binaryExport?.status ?? 'OpenCL not executed'})`,
  };

  writeRgbaFile(artifacts.sequentialOutputRgbaBinary, outputSequential);
  if (cpuParallel.status === 'completed') writeRgbaFile(artifacts.parallelOutputRgbaBinary, outputParallel);
  if (openclGpu.status === 'completed') writeRgbaFile(artifacts.openclOutputRgbaBinary, outputOpenCL);

  const pngArtifacts = {};
  pngArtifacts.sequential = await tryWritePng(path.join(artifactDir, 'output-cpu-sequential.png'), width, height, outputSequential);
  if (cpuParallel.status === 'completed') {
    pngArtifacts.parallel = await tryWritePng(path.join(artifactDir, 'output-cpu-parallel.png'), width, height, outputParallel);
  }
  if (openclGpu.status === 'completed') {
    pngArtifacts.opencl = await tryWritePng(path.join(artifactDir, 'output-opencl-gpu.png'), width, height, outputOpenCL);
  }

  const profile = {
    schemaVersion: '1.0.0',
    runId,
    generatedAt: new Date().toISOString(),
    runtime: {
      node: process.version,
      platform: process.platform,
      architecture: process.arch,
      osRelease: os.release(),
      cpuModel: os.cpus()[0]?.model ?? 'unknown',
      logicalCpuCount: os.cpus().length,
      availableParallelism: os.availableParallelism?.() ?? null,
      totalSystemMemoryBytes: os.totalmem(),
      processId: process.pid,
    },
    config: {
      ...config,
      input: config.input,
      outputDir: config.outputDir,
    },
    workload: {
      label: preset.label,
      level: config.workload,
      width,
      height,
      radius,
      windowWidth: radius * 2 + 1,
      windowHeight: radius * 2 + 1,
      sigmaSpaceSquared,
      sigmaColorSquared,
      imageSource: image.source,
      originalInputPath: image.sourcePath,
      seed: image.source === 'synthetic' ? config.seed : null,
      inputSha256: hashBuffer(Buffer.from(input.buffer, input.byteOffset, input.byteLength)),
    },
    metrics,
    benchmarks: {
      cpuSequential,
      cpuParallel,
      openclGpu,
    },
    comparisons: {
      speedupVsSequential,
      speedupNote: 'Speedup ratios use medians. The OpenCL pipeline ratio includes host-device transfers; the kernel-only ratio excludes the transfer phases but still includes host enqueue/synchronization overhead.',
    },
    validation,
    memoryNotes: {
      processMemory: 'RSS, heapUsed, external, and arrayBuffers are process snapshots before and after each measured repetition. Deltas can be negative and are not peak memory.',
      gpuMemory: 'The GPU memory metric is the explicit input+output buffer allocation requested by this program. Driver/runtime allocations and total VRAM-used counters are not exposed by the current wrapper workflow.',
    },
    binaryAndSourceArtifacts: {
      rawRgbaIsBinary: true,
      rawRgbaDescription: 'RGBA byte stream in row-major order, four bytes per pixel: R, G, B, A. Width and height are recorded in profile.json.',
      openclProgramBinaryExport: openclGpu.binaryExport ?? { status: 'not-attempted' },
      pngArtifacts,
    },
    artifacts,
  };

  const profilePath = path.join(artifactDir, 'profile.json');
  const csvPath = path.join(artifactDir, 'profile.csv');
  const reportPath = path.join(artifactDir, 'report.md');
  fs.writeFileSync(profilePath, `${JSON.stringify(profile, null, 2)}\n`);
  writeCsv(csvPath, makeCsvRows(profile.benchmarks, metrics, profile.comparisons));
  fs.writeFileSync(reportPath, buildMarkdownReport(profile));

  console.log('\n==============================================================');
  console.log(' FINAL COMPARISON (median time)');
  console.log('==============================================================');
  console.log(`CPU Sequential       : ${cpuSequential.timing.summary.medianMs} ms`);
  console.log(`CPU Parallel         : ${parallelMedian ?? 'not available'} ms`);
  console.log(`GPU OpenCL pipeline  : ${gpuPipelineMedian ?? 'not available'} ms`);
  console.log(`GPU OpenCL kernel    : ${gpuKernelMedian ?? 'not available'} ms (median synchronized host timing)`);
  console.log('--------------------------------------------------------------');
  console.log(`Parallel CPU speedup : ${speedupVsSequential['CPU Parallel (worker_threads)'] ?? 'n/a'}×`);
  console.log(`GPU kernel speedup   : ${speedupVsSequential['GPU OpenCL (kernel phase)'] ?? 'n/a'}×`);
  console.log(`GPU pipeline speedup : ${speedupVsSequential['GPU OpenCL (transfer + kernel pipeline)'] ?? 'n/a'}×`);
  console.log('--------------------------------------------------------------');
  console.log(`Artifacts saved to   : ${artifactDir}`);
  console.log(`Profile JSON         : ${profilePath}`);
  console.log(`Profile CSV          : ${csvPath}`);
  console.log(`Readable report      : ${reportPath}`);
  console.log('==============================================================\n');

  if (config.requireGpu && openclGpu.status !== 'completed') {
    process.exitCode = 2;
  }
}

main().catch((error) => {
  console.error(`\n[FATAL] ${error?.stack || error}`);
  process.exitCode = 1;
});

# GPU Compute Teaching Lab — CPU vs OpenCL GPU

A reproducible, configurable benchmark built around a **bilateral image filter**. It compares three implementations of the same calculation:

1. **CPU Sequential** — one JavaScript thread processes all output pixels.
2. **CPU Parallel** — Node.js `worker_threads` share the input/output buffers and distribute rows across independent workers.
3. **GPU OpenCL** — one OpenCL work-item computes one output pixel; 2-D work-groups are controlled by the configured local size.

The project is designed as a teaching example, not as a claim that a GPU must always win. It records transfer time, synchronized kernel time, full GPU pipeline time, CPU process time, process memory snapshots, work-item geometry, correctness comparisons, and reproducibility artifacts.

## 1. Requirements

- Node.js 20 or newer.
- An OpenCL 1.2 runtime/ICD and a compatible GPU driver to execute the GPU case.
- Optional: PNG/JPEG input support and PNG output support are provided by the optional `pngjs` and `jpeg-js` packages.

The project uses [`@node-3d/opencl`](https://www.npmjs.com/package/@node-3d/opencl), a Node.js binding for OpenCL 1.2. An installed OpenCL runtime is still required; installing the JavaScript package does not install a GPU driver or guarantee that a GPU is available.

## 2. Install

```bash
npm install
```

The OpenCL binding is optional so the CPU demonstrations still run when no OpenCL runtime is installed.

## 3. Run the benchmark

```bash
# Small test; 4 CPU workers; 8x8 OpenCL local work-group
node benchmark.mjs --workload 1 --threads 4 --local-x 8 --local-y 8

# Medium workload, default local size and default worker count
node benchmark.mjs --workload 3

# Heavy workload and five measured repetitions
node benchmark.mjs --workload 4 --threads 8 --local-x 16 --local-y 16 --global-x 1024 --global-y 1024 --repeats 5

# Run only the CPU paths (does not attempt to initialize OpenCL)
node benchmark.mjs --workload 2 --skip-gpu

# Require real OpenCL execution; return exit code 2 if it cannot run
node benchmark.mjs --workload 2 --require-gpu
```

Run `node benchmark.mjs --help` for all arguments.

## 4. One parameter controls calculation heaviness

`--workload` is the main student-facing complexity control. It changes both the image size and bilateral-filter radius, so the number of neighbourhood evaluations grows sharply.

| Level | Image | Radius | Neighbours per pixel | Total neighbourhood evaluations |
|---:|---:|---:|---:|---:|
| 1 | 256 × 256 | 1 | 9 | 589,824 |
| 2 | 512 × 512 | 2 | 25 | 6,553,600 |
| 3 | 768 × 768 | 3 | 49 | 28,901,376 |
| 4 | 1024 × 1024 | 4 | 81 | 84,934,656 |
| 5 | 1280 × 1280 | 5 | 121 | 198,246,400 |

The work count is:

\[
N_{work} = width \times height \times (2r+1)^2
\]

This is an intentionally understandable model: the image area grows quadratically with width/height, while the filter window grows quadratically with radius. Level 5 can take a long time on a CPU-only machine. Begin with level 1 or 2, then increase it gradually.

When `--input` supplies a PNG/JPEG image, that image's dimensions are used; the workload level still controls the filter radius.

## 5. Parameters students can tune

### CPU parallelization

| Parameter | Meaning | Default |
|---|---|---:|
| `--threads` | Number of Node.js worker threads | Logical CPU count minus one |
| `--chunk-rows` | Number of image rows claimed at a time using an atomic counter | 8 |

The parallel implementation uses shared memory. It does not copy the complete image into every worker. Each worker repeatedly claims a disjoint row range with `Atomics.add`, reads the shared input, and writes its own output pixels. This makes the chunk size an explicit load-scheduling parameter.

### OpenCL launch configuration

| Parameter | Meaning | Default |
|---|---|---:|
| `--local-x` | Work-items per work-group along X | 16 |
| `--local-y` | Work-items per work-group along Y | 16 |
| `--global-x`, `--global-y` | Requested NDRange extents; must cover the image and are padded to local-size multiples | Image width/height |
| `--build-options` | OpenCL compiler options | Driver defaults |

By default the global range is derived from image width and height (the workload level therefore controls the global work-item count). Advanced users can override the requested X/Y extents with `--global-x` and `--global-y`, but these must still cover every image pixel. The actual launch range is padded up to a multiple of the local range. The kernel checks `x >= width` and `y >= height`, so padded work-items exit safely. The report records the true image size, padded global size, work-group count, and total launched work-items.

The OpenCL helper currently uses `quickStart()` and therefore selects the first device exposed by the binding. This example exposes work-group geometry and compiler options as tuning controls; it does not pretend that `quickStart()` provides a user-selectable device index.

## 6. What is measured?

### CPU cases

For every timed repetition the profiler records:

- wall-clock duration;
- process CPU user time and system time;
- approximate CPU-core utilization (`process CPU time / wall time`);
- before/after RSS, V8 heap, external memory, and ArrayBuffer memory snapshots;
- the process RSS high-water mark from `process.resourceUsage()` when available (cumulative, not isolated to one benchmark case).

The CPU parallel timing includes worker message dispatch and synchronization but excludes one-time worker-pool startup. Worker startup is reported separately only by the runtime environment, not mixed into every timed repetition.

### OpenCL GPU case

The following are kept separate:

- OpenCL initialization, program compilation, and buffer setup (`setupMs`, `compileMs`);
- host-to-device input upload;
- kernel launch plus `cl.finish()` synchronization;
- device-to-host output readback;
- total measured transfer + kernel pipeline.

The GPU kernel is asynchronous by design. Calling `cl.finish()` is essential before stopping the host timer; otherwise the timer would mostly measure command submission rather than completed GPU work.

The kernel phase is **host synchronized wall time**, not a GPU hardware-counter event time. The binding's generic `getEventProfilingInfo` route is not assumed to exist. A future adapter can add device event timing without changing the three-way comparison structure.

### Memory metrics and limitations

- Host memory values are Node.js process snapshots before and after each run. A delta can be negative due to garbage collection. The RSS high-water mark is cumulative for the process, not a per-case sampled peak.
- The GPU value is the explicit input + output buffer allocation requested by this program. It is **not** a direct measure of total VRAM currently in use by the driver, desktop, or other processes.
- The logical-byte model estimates 3 RGB channel reads for every neighbour sample plus 4 RGBA output bytes per pixel. It is not physical DRAM traffic, cache misses, or a hardware-counter bandwidth measurement.
- GFLOPS is intentionally omitted. The kernel contains exponentials (`exp`), and calling each such operation a fixed number of FLOPs would imply precision that this source-level benchmark cannot justify.

## 7. Output files and binary artifacts

Every run creates a timestamped subfolder under `benchmark-results/` (or the chosen `--output-dir`). It contains:

| File | Purpose |
|---|---|
| `profile.json` | Complete configuration, hardware, results, per-repetition metrics, validation, and artifact index |
| `profile.csv` | Tabular benchmark summary for Excel, pandas, or plotting |
| `report.md` | Human-readable comparison and caveats |
| `input.rgba` | Raw input image as binary RGBA bytes |
| `output-cpu-sequential.rgba` | Sequential CPU output as raw binary RGBA |
| `output-cpu-parallel.rgba` | Parallel CPU output as raw binary RGBA |
| `output-opencl-gpu.rgba` | GPU output as raw binary RGBA, only if GPU execution succeeds |
| `bilateral_filter.cl` | Exact OpenCL C kernel source compiled for this run |
| `opencl-build.log` | Device/build information, compiler log when exposed, and binary-export status |
| `source/benchmark.mjs` | Snapshot of the host-side benchmark source used for the run |
| `source/cpu-worker.mjs` | Snapshot of the worker source used for the run |
| `*.png` | Convenient viewable copies when `pngjs` is installed |
| `bilateral_filter.opencl.bin` | Compiled program binary only when the installed OpenCL binding exposes a usable program-binary query |

A raw `.rgba` file is a real binary runtime artifact: it stores 4 bytes per pixel in row-major order (R, G, B, A). Use the width and height recorded in `profile.json` when opening it. The kernel source is saved independently so students can inspect the actual GPU program. The binding may not expose compiled OpenCL binary extraction on every version; the profile explicitly records whether that optional export succeeded rather than inventing a binary file.

## 8. Correctness and comparison

The sequential CPU result is the reference. The other paths are compared against it using:

- exact byte equality;
- maximum absolute channel error;
- mean absolute error;
- root-mean-square error;
- PSNR;
- differing-pixel count and percentage;
- a practical maximum-error-of-one check.

Floating-point implementations on CPU and GPU can differ slightly because compilers and devices are allowed to implement floating-point operations differently. A speed result is only useful if its output is also correct within the chosen tolerance.

## 9. Important CUDA/OpenCL distinction

This project runs **OpenCL C kernels**, not CUDA kernels. The conceptual mapping is:

| CUDA term | OpenCL term | Meaning |
|---|---|---|
| CUDA kernel (`__global__`) | OpenCL kernel (`__kernel`) | Device-executed function |
| `threadIdx.x` | `get_local_id(0)` | Work-item's index inside its work-group |
| `blockIdx.x` | `get_group_id(0)` | Work-group index in the NDRange |
| `blockDim.x` | `get_local_size(0)` | Work-items per work-group dimension |
| `blockIdx.x * blockDim.x + threadIdx.x` | `get_global_id(0)` | Global index in a 1-D range |
| CUDA block | OpenCL work-group | Cooperative group of device work-items |
| CUDA thread | OpenCL work-item | Individual logical execution instance |
| `cudaMalloc` | `cl.createBuffer` | Device memory allocation |
| `cudaMemcpy` | `cl.enqueueWriteBuffer` / `cl.enqueueReadBuffer` | Host-device transfer |
| `__shared__` | `__local` | Work-group-shared local memory |
| `__syncthreads()` | `barrier(CLK_LOCAL_MEM_FENCE)` | Work-group synchronization |

CUDA and OpenCL are different platforms with different APIs and compiler toolchains. Conceptual similarity does not make their source code interchangeable.

## 10. Teaching material

See [`docs/teaching-guide.md`](docs/teaching-guide.md) for a lecture sequence, expected observations, GPU execution model, performance pitfalls, suggested experiments, and discussion questions.

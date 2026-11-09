# Teaching Guide: Understanding GPU Acceleration Through a Bilateral Filter

## Lesson goal

Students should finish this lab able to explain not only *whether* a GPU was faster, but *why*, *under which workload*, *at what cost*, and *which measurements support the conclusion*.

The lab compares three implementations of one calculation. This is important: comparing different algorithms would confound the hardware comparison with an algorithmic one.

## 1. The problem: bilateral image filtering

A bilateral filter smooths image noise while attempting to preserve strong edges. For each output pixel, it examines a square neighbourhood. Each neighbour receives a weight derived from:

- spatial distance from the output pixel;
- colour distance from the output pixel.

For a centre pixel `p` and neighbour `q`, the weight is modelled as:

\[
 w(p,q)=\exp\left(-\frac{\|p-q\|^2}{\sigma_s^2}
                  -\frac{\|I(p)-I(q)\|^2}{\sigma_c^2}\right)
\]

The output colour is a weighted average of the neighbour colours.

This calculation is suitable for teaching because each output pixel can be calculated independently. The pixel's neighbours are read-only input data, and each work-item writes exactly one output pixel. No output pixel depends on another output pixel's newly computed result.

### Complexity

For width `W`, height `H`, and radius `r`, each pixel visits `(2r+1)^2` neighbours. The total number of neighbourhood samples is:

\[
  S = W H (2r+1)^2
\]

The lab's single `--workload` argument changes both `W,H` and `r`, so the calculation becomes substantially heavier as the level increases.

## 2. The three execution models

### A. CPU Sequential

One JavaScript thread loops over all rows and columns. It is the reference implementation and provides a baseline for the amount of work performed by one CPU execution thread.

It is not necessarily a perfect representation of all CPU capabilities: modern CPUs may use SIMD/vector instructions internally, caches, branch prediction, and frequency scaling even when the source loop is sequential.

### B. CPU Parallel with worker threads

Node.js worker threads allow JavaScript to execute on multiple OS threads. This lab creates a worker pool and shares the input and output arrays through `SharedArrayBuffer`.

Workers claim row chunks using:

```js
const firstRow = Atomics.add(nextRow, 0, chunkRows);
```

Because the returned ranges do not overlap, workers can write output pixels without locks around individual pixels. The atomic counter is only used to allocate work ranges.

Students can change:

- `--threads`: degree of CPU parallelism;
- `--chunk-rows`: scheduling granularity.

Very small chunks cause more atomic scheduling operations. Very large chunks can reduce scheduling overhead but can make balancing less flexible. In this image filter, rows have roughly equal cost, so static partitioning could also work; dynamic chunks are included to expose scheduling as a tunable parameter.

### C. GPU with OpenCL

The host program (Node.js) performs orchestration:

1. selects an OpenCL device and creates a context and command queue;
2. creates input and output buffers in device memory;
3. compiles the OpenCL C kernel;
4. uploads the input image;
5. launches a 2-D NDRange;
6. waits for the kernel to complete;
7. downloads the output image.

The kernel runs on the device. One work-item calculates one image pixel. `get_global_id(0)` and `get_global_id(1)` identify that pixel's X and Y coordinates.

For the default 16 × 16 local size:

- a work-group contains 256 work-items;
- the work-group grid tiles the global 2-D range;
- each work-item independently calculates one output pixel;
- the bounds check rejects padded work-items outside the real image.

The global range is padded because in OpenCL 1.2, when an explicit local size is supplied, global dimensions are ordinarily chosen as multiples of their matching local dimensions. For an image whose width is not divisible by the local X size, padding plus a bounds check is a safe general pattern.

## 3. CUDA mental model and OpenCL names

Students may know CUDA terminology. Use the following mapping, but make clear that CUDA and OpenCL are separate programming platforms:

| CUDA | OpenCL | Interpretation |
|---|---|---|
| Kernel function | `__kernel` function | Code executed on the device |
| Thread | Work-item | Logical invocation of the kernel |
| Thread block | Work-group | Group of work-items that may cooperate |
| Grid | NDRange | Entire launched index space |
| `threadIdx.x` | `get_local_id(0)` | Local coordinate inside a work-group |
| `blockIdx.x` | `get_group_id(0)` | Work-group coordinate |
| `blockDim.x` | `get_local_size(0)` | Work-group extent |
| `blockIdx.x * blockDim.x + threadIdx.x` | `get_global_id(0)` | Global coordinate in one dimension |
| `cudaMalloc` | `cl.createBuffer` | Device allocation |
| `cudaMemcpy` | enqueue write/read buffer | Transfer between host and device |
| `__shared__` | `__local` | Memory shared by a work-group |
| `__syncthreads()` | `barrier(...)` | Work-group barrier |

A CUDA thread is conceptually similar to an OpenCL work-item; a CUDA block is conceptually similar to an OpenCL work-group. The APIs, language details, runtime behaviour, and binary formats are not interchangeable.

## 4. Why the GPU can win

A GPU contains many execution lanes designed to execute large numbers of similar operations concurrently. The bilateral filter exposes a high number of independent output pixels, and each pixel follows almost the same instruction sequence. This gives the device ample parallel work.

The important factor is not merely that a GPU has more cores. The problem must expose enough independent work, with a suitable access pattern and enough work to amortize dispatch and data movement.

### Data parallelism

Every output pixel is independent. This is an example of data parallelism: the same program is applied to many different data elements.

### Throughput-oriented hardware

CPUs are designed to perform well across many kinds of tasks, including latency-sensitive and branch-heavy workloads. GPUs devote more of their resources to throughput when many work-items are ready to execute.

### Latency hiding

When some GPU work-items wait on memory, the scheduler may run other ready work-items. A large enough set of independent work-items helps keep execution resources occupied.

### Memory coalescing

Adjacent X coordinates access adjacent image pixels. Consecutive work-items in the X dimension therefore tend to access nearby memory addresses. This is a useful pattern for coalescing memory accesses, although the exact transactions depend on the hardware, cache, and compiler.

### Cache reuse

Neighbourhoods overlap: nearby output pixels read many of the same input pixels. GPU caches may reduce external-memory traffic. This example deliberately uses global-memory loads for conceptual clarity; a later optimization exercise could stage image tiles into OpenCL `__local` memory and include a halo region.

## 5. Why the GPU might not win

The GPU path is not just the kernel. It requires host-device transfers and command submission.

For a small workload:

- the kernel may finish very quickly;
- buffer transfers and driver calls may take comparable or greater time;
- the CPU may finish before the GPU path has amortized its setup cost.

For a larger workload, the GPU has more independent calculations over which to amortize fixed overhead.

This is why the lab reports both:

- **kernel phase** — enqueue plus `cl.finish()` synchronization;
- **GPU pipeline** — upload + synchronized kernel + readback.

For a real application, whether transfers belong in the measured time depends on the architecture. If the data is already resident on the GPU for multiple downstream operations, charging every stage for a full round trip would be misleading. If the application uploads an image, filters it once, and downloads it, the end-to-end pipeline is the relevant comparison.

## 6. Work-group size experiments

Try the same workload with different local sizes:

```bash
node benchmark.mjs --workload 3 --local-x 8  --local-y 8
node benchmark.mjs --workload 3 --local-x 16 --local-y 16
node benchmark.mjs --workload 3 --local-x 32 --local-y 8
node benchmark.mjs --workload 3 --local-x 32 --local-y 16
```

Ask students to predict the outcome before running the tests.

Possible observations:

- a larger work-group is not automatically faster;
- the device imposes a maximum work-group size and dimension limits;
- register use and local-memory use can affect occupancy;
- the best size depends on the GPU architecture and kernel resource requirements;
- launch padding may create a small number of extra work-items;
- driver and runtime scheduling can affect measured results.

If a local size is invalid for the selected device, OpenCL should report a launch error. Reduce the size and rerun. The report stores the requested geometry so results can be compared honestly.

## 7. CPU thread-count experiments

Try:

```bash
node benchmark.mjs --workload 4 --threads 1
node benchmark.mjs --workload 4 --threads 2
node benchmark.mjs --workload 4 --threads 4
node benchmark.mjs --workload 4 --threads 8
```

The fastest result may not correspond to the maximum thread count. Additional threads can increase scheduling overhead, contend for memory bandwidth, or compete with other applications. The host CPU may also have fewer physical cores than logical processors.

The report's approximate CPU-core utilization is process CPU time divided by wall time. For example, 350% suggests the process consumed about 3.5 CPU cores on average during that interval. It is a process-level estimate, not a per-core hardware-counter measurement.

## 8. Interpreting the profile

Important values in `profile.json`:

- `benchmarks.cpuSequential.timing.summary.medianMs`
- `benchmarks.cpuParallel.timing.summary.medianMs`
- `benchmarks.openclGpu.timing.pipelineSummary.medianMs`
- `benchmarks.openclGpu.timing.kernelHostSynchronizedMsMean`
- `benchmarks.openclGpu.timing.uploadMsMean`
- `benchmarks.openclGpu.timing.downloadMsMean`
- `benchmarks.openclGpu.launch.globalSize`
- `benchmarks.openclGpu.launch.localSize`
- `benchmarks.openclGpu.launch.workGroupCount`
- `benchmarks.openclGpu.memory.totalDeviceBufferAllocationBytes`
- `validation.*`

Use the median rather than one isolated run when comparing performance. The report also keeps every individual repetition so students can see run-to-run variation. Avoid claiming a speedup from one run if the variation is larger than the difference.

### Amdahl's law connection

Even when the kernel is highly parallel, the total application contains serial work such as file decoding, buffer allocation, compilation, upload, and output encoding. Amdahl's law explains why accelerating only one part of a pipeline has a bounded effect on total runtime.

The `setupMs`, `compileMs`, transfer durations, and kernel duration make those parts visible. The measured pipeline speedup is generally a more conservative application-level result than the kernel-only ratio.

## 9. Binary and compiler artifacts

Every successful run saves:

- the OpenCL C kernel source (`bilateral_filter.cl`);
- the exact host and worker JavaScript source snapshots;
- an OpenCL build log (`opencl-build.log`);
- raw RGBA input/output binary streams;
- a JSON profile and CSV summary.

The raw `.rgba` output files are binary runtime artifacts. They are directly reusable by another program and can be reconstructed as an image using the width and height in `profile.json`.

The OpenCL binding may expose a program-binary query on one version/platform and not another. The program-binary export is attempted only when the relevant binding methods/constants are present. The run log and JSON explicitly state whether an actual compiled program binary was exported. The project never fabricates a `.bin` file when it has only source code.

An OpenCL program binary is implementation-specific: it may be tied to a device vendor, device architecture, driver, or compiler version. It should not be assumed portable between GPUs.

## 10. Suggested 60–90 minute lab sequence

1. **Inspect the host code and kernel (10 min).** Identify which code runs on the CPU and which code is compiled and executed by OpenCL.
2. **Run workload 1 (5 min).** Confirm all three result paths and locate the generated artifacts.
3. **Compare CPU thread counts (10–15 min).** Explain speedup, scheduling overhead, and shared data.
4. **Compare OpenCL local sizes (10–15 min).** Explain work-items, work-groups, NDRange, and padding.
5. **Increase the workload (10 min).** Discuss when the GPU's fixed transfer/launch costs become less significant.
6. **Inspect phase timing (10 min).** Separate compilation, upload, kernel, and download.
7. **Inspect validation and binary artifacts (5–10 min).** Verify that faster results are still numerically equivalent and that source/output files are preserved.
8. **Discuss optimization (10 min).** Propose local-memory tiling, vectorized loads, precomputed spatial weights, or separable approximations. Require students to preserve correctness and benchmark each change.

## 11. Discussion questions

1. Why is one work-item per output pixel a natural mapping for this algorithm?
2. What happens to total work when the filter radius increases from 2 to 4?
3. Why can a GPU kernel be faster while the GPU pipeline is slower?
4. How do `--local-x` and `--local-y` change the number and shape of work-groups?
5. Why are adjacent X coordinates a useful access pattern?
6. How could a tile in `__local` memory reduce repeated global-memory loads?
7. Why does a CPU parallel implementation need coordination even though output pixels are independent?
8. What is the difference between allocated device-buffer size and actual total VRAM usage?
9. Why should a benchmark verify output correctness as well as speed?
10. Why is a compiled OpenCL program binary generally not a portable replacement for kernel source?

## 12. Important limitations to explain explicitly

- `quickStart()` selects the first available OpenCL device; this version does not implement custom platform/device enumeration.
- Device memory usage is reported as this program's buffer allocation, not a direct total-VRAM-used measurement.
- Kernel timing currently uses synchronized host timing. It is not OpenCL event profiling from a hardware clock.
- Memory deltas are before/after snapshots, not peak sampling.
- The logical data-volume model is not a physical memory-bandwidth counter.
- Floating-point CPU/GPU results can differ slightly; inspect the validation section.
- Level 5 is intentionally demanding. On a CPU-only machine, run a lower workload first.

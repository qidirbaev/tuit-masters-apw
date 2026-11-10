# OpenCL vs CUDA: A Comparison

Both platforms let you write data-parallel programs that run thousands of lightweight threads on a GPU. They share the same underlying ideas, so skills transfer in both directions. They differ in **who controls them, how code is compiled, how a launch is written, and what ecosystem surrounds them**.

> **Version note.** Support levels, driver versions and library availability change over time. Treat the "ecosystem" and "platform support" statements as a snapshot and check vendor documentation before making a project decision.

---

## 1. One-page summary

| Aspect | OpenCL | CUDA |
|---|---|---|
| **Owner** | Khronos Group (open standard, many members) | NVIDIA (proprietary) |
| **Hardware** | GPUs from many vendors, CPUs, FPGAs, DSPs | NVIDIA GPUs only |
| **Kernel language** | OpenCL C (C99-based); C++ for OpenCL; SPIR-V as an intermediate form | CUDA C++ (C++ with extensions) |
| **Source model** | Host and device code are **separate** (kernel is usually a string or `.cl` file) | **Single-source**: host and device code in one `.cu` file |
| **Compilation** | Kernel normally compiled **at run time** by the driver | Kernel compiled **ahead of time** by `nvcc` (PTX/SASS embedded); PTX can be JIT-compiled at load time; NVRTC offers run-time compilation |
| **Launch syntax** | Function call: `clEnqueueNDRangeKernel(...)` | `kernel<<<grid, block>>>(args)` |
| **Launch size unit** | **Global size in work-items** (local size is the divisor) | **Grid size in blocks** (block size is the multiplier) |
| **API verbosity** | High: explicit platform, context, queue, program, kernel objects | Lower with the Runtime API; comparable with the Driver API |
| **Device selection** | Platform and device enumeration is mandatory | `cudaSetDevice` (defaults to device 0) |
| **Portability** | Source portable across vendors; **performance is not automatically portable** | Binary/PTX portable across NVIDIA generations; no other vendors |
| **Typical performance on NVIDIA** | Often close with good tuning, but features reach OpenCL later | Reference platform; newest hardware features first |
| **Libraries** | Smaller set (for example CLBlast, clFFT, ViennaCL) | Very large (cuBLAS, cuDNN, cuFFT, cuSPARSE, Thrust/CUB, NCCL, ...) |
| **Tooling** | Vendor-specific profilers and debuggers | Nsight Systems/Compute, `cuda-gdb`, Compute Sanitizer |
| **Learning ecosystem** | Fewer tutorials and books | Large community, documentation and course material |
| **Deep-learning frameworks** | Minor role | Primary GPU back end |

---

## 2. Same ideas, different words

Conceptually the two are nearly isomorphic. The lab's guide already lists the main mapping; this section explains *why* the pieces line up.

| Idea | OpenCL | CUDA |
|---|---|---|
| Small function run by many parallel instances | Kernel run by work-items | Kernel run by threads |
| Instances grouped for cooperation | Work-group | Thread block |
| All groups together | NDRange | Grid |
| Fast scratchpad shared inside a group | `__local` | `__shared__` |
| Group-wide synchronisation | `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` |
| Hardware SIMD bundle | Sub-group (OpenCL 2.1+, extensions) | Warp (32 threads) |
| Processor cluster that runs one group | Compute unit | Streaming multiprocessor (SM) |
| Host-device queue of commands | `cl_command_queue` | Stream |
| Asynchronous command handle | `cl_event` | `cudaEvent_t` / stream ordering |

### The two differences that trip people up

**1. Launch sizes mean different things.**

```
OpenCL: global_size = total work-items          (must be a multiple of local_size in 1.2)
CUDA:   grid        = number of blocks          (total threads = grid * block)
```

To cover a 1000 x 700 image with 16 x 16 work-groups:

```c
// OpenCL: round the *work-item* count up
size_t local[2]  = {16, 16};
size_t global[2] = {1008, 704};          // 63*16 by 44*16
clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
```

```cuda
// CUDA: round the *block* count up
dim3 block(16, 16);
dim3 grid((1000 + 15) / 16, (700 + 15) / 16);   // 63 x 44
bilateral_filter<<<grid, block>>>(...);
```

Both end up with 63 x 44 groups. The lab's `launch.globalSize` corresponds to `grid * block` in CUDA.

**2. The word "local" means opposite things.**

- OpenCL `__local` = fast memory shared by a work-group = CUDA `__shared__`.
- CUDA "local memory" = per-thread memory that spills to slow device DRAM = roughly OpenCL **private** memory that did not fit in registers.

---

## 3. Source structure

### CUDA: one file, one compiler

```cuda
// filter.cu
__global__ void bilateral_filter(const unsigned char* in, unsigned char* out,
                                 int w, int h, int r, float ss, float sc) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    ...
}

int main() {
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
    bilateral_filter<<<grid, block>>>(d_in, d_out, w, h, r, ss, sc);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);
}
```

`nvcc` splits the file: it compiles `__global__` functions to GPU code and hands the rest to the host C++ compiler. The type system is shared, so passing a struct to a kernel is trivial and the compiler type-checks the launch.

### OpenCL: host program plus kernel text

```c
// host.c (no GPU code here)
cl_platform_id plat; clGetPlatformIDs(1, &plat, NULL);
cl_device_id   dev;  clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
cl_context     ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, NULL);
cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, NULL);

cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, NULL);
clBuildProgram(prog, 1, &dev, "", NULL, NULL);
cl_kernel k = clCreateKernel(prog, "bilateral_filter", NULL);

cl_mem d_in  = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  bytes, NULL, NULL);
cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
clEnqueueWriteBuffer(q, d_in, CL_TRUE, 0, bytes, h_in, 0, NULL, NULL);

clSetKernelArg(k, 0, sizeof(cl_mem), &d_in);
clSetKernelArg(k, 1, sizeof(cl_mem), &d_out);
/* ... remaining args ... */
size_t local[2] = {16,16}, global[2] = {1008, 704};
clEnqueueNDRangeKernel(q, k, 2, NULL, global, local, 0, NULL, NULL);
clFinish(q);
clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, bytes, h_out, 0, NULL, NULL);
```

The device code lives in `bilateral_filter.cl` (the lab's file). The compiler cannot check that your host arguments match the kernel signature: mismatches show up as run-time errors such as `CL_INVALID_KERNEL_ARGS` or, worse, silent wrong results.

> **Closest CUDA analogue.** OpenCL's explicit style resembles the CUDA **Driver API** (`cuInit`, `cuCtxCreate`, `cuModuleLoadData`, `cuLaunchKernel`) more than the Runtime API above.

---

## 4. The lab kernel in both languages

**OpenCL C (from the lab):**

```c
__kernel void bilateral_filter(
    __global const uchar *input_rgba, __global uchar *output_rgba,
    const uint width, const uint height, const uint radius,
    const float sigma_space_squared, const float sigma_color_squared)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x >= (int)width || y >= (int)height) return;
    ...
    const int sample_y = clamp(y + dy, 0, (int)height - 1);
    ...
    const float weight = exp(exponent);
    ...
    output_rgba[center_index + 0] = (uchar)clamp(weighted_r / total_weight, 0.0f, 255.0f);
}
```

**Equivalent CUDA C++:**

```cuda
__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return min(max(v, lo), hi);
}

__global__ void bilateral_filter(
    const unsigned char* __restrict__ input_rgba,
    unsigned char* __restrict__ output_rgba,
    unsigned width, unsigned height, unsigned radius,
    float sigma_space_squared, float sigma_color_squared)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (int)width || y >= (int)height) return;
    ...
    const int sample_y = clampi(y + dy, 0, (int)height - 1);
    ...
    const float weight = expf(exponent);
    ...
    output_rgba[center_index + 0] =
        (unsigned char)fminf(fmaxf(weighted_r / total_weight, 0.0f), 255.0f);
}
```

Differences to notice:

| In OpenCL C | In CUDA C++ | Why |
|---|---|---|
| `__kernel`, `__global const uchar *` | `__global__`, `const unsigned char *` | Pointers are not address-space qualified in CUDA; all pointers are generic |
| `get_global_id(0)` | `blockIdx.x * blockDim.x + threadIdx.x` | CUDA has no built-in global ID; you compute it |
| `clamp(int, int, int)` built in | write `min(max(...))` | CUDA has no built-in `clamp` for integers |
| `clamp(float, 0, 255)` | `fminf(fmaxf(v, 0.0f), 255.0f)` | Same reason (CUDA's `__saturatef` only clamps to [0, 1]) |
| `exp(float)` (overloaded) | `expf(float)` | CUDA C++ also supports `exp` overloads; `expf` is the explicit single-precision form |
| `uchar`, `uint` | `unsigned char`, `unsigned int` | OpenCL C defines short aliases |
| `native_exp`, `-cl-fast-relaxed-math` | `__expf`, `--use_fast_math` | Fast, lower-accuracy math options |

The *algorithm* is identical, which is exactly the property the lab depends on: one calculation, several execution models.

---

## 5. Compilation and deployment

| | OpenCL | CUDA |
|---|---|---|
| When is device code compiled? | Normally **on the user's machine at run time** (`clBuildProgram`) | **At build time** by `nvcc`, plus optional run-time paths |
| Intermediate form | Vendor-internal; SPIR-V accepted by OpenCL 2.1+/3.0 implementations that support it | **PTX** (virtual ISA), compiled to **SASS** (real machine code) |
| What you ship | Kernel source (or SPIR-V), or vendor binaries per device | A `fatbin` containing SASS for known architectures plus PTX for forward compatibility |
| Forward compatibility | Source recompiles for new devices automatically | Embedded PTX is JIT-compiled for newer GPUs |
| Build failures show up | At run time, in the program build log | At compile time |
| First-launch cost | Compile time on every start unless binaries are cached | Little, if SASS matches the GPU; JIT cost if only PTX matches |
| Run-time compilation API | `clBuildProgram` (the default path) | NVRTC / Driver API modules |
| Inspectability | Build log, optionally `CL_PROGRAM_BINARIES` (not in every binding) | `cuobjdump`, `nvdisasm`, `--ptx` output |

The lab's `compileMs` is an inherent OpenCL cost; a CUDA program would pay that cost at build time instead.

---

## 6. Memory and data movement

| Capability | OpenCL | CUDA |
|---|---|---|
| Explicit device allocation | `clCreateBuffer` | `cudaMalloc` |
| Explicit copies | `clEnqueueWrite/ReadBuffer`, `clEnqueueCopyBuffer` | `cudaMemcpy`, `cudaMemcpyAsync` |
| Page-locked ("pinned") host memory | `CL_MEM_ALLOC_HOST_PTR` / `CL_MEM_USE_HOST_PTR` plus map/unmap (implementation-dependent behaviour) | `cudaHostAlloc` / `cudaMallocHost` |
| Mapping device memory into host address space | `clEnqueueMapBuffer` | Mapped pinned memory or managed memory |
| Single pointer valid on host and device | **SVM** (OpenCL 2.0+; coarse-grain, fine-grain buffer, fine-grain system) | **Unified Memory** (`cudaMallocManaged`, since CUDA 6) |
| Textures / images | `image2d_t` plus `sampler_t` | Texture objects, surface objects |
| Constant memory | `__constant` (size queried via device info) | `__constant__` (64 KB) |
| Local / shared memory size | Queried: `CL_DEVICE_LOCAL_MEM_SIZE` | Per-SM limit; dynamic shared memory at launch (`<<<g, b, bytes>>>`) |
| Atomics | `atomic_add`, `atomic_cmpxchg` (1.x); C11-style `atomic_fetch_add` (2.0+) | `atomicAdd`, `atomicCAS`, plus cooperative groups |

Practical point: in both platforms a discrete GPU requires **data to cross PCIe**, so the lab's "pipeline vs kernel" distinction applies equally to CUDA.

---

## 7. Execution control and concurrency

| Capability | OpenCL | CUDA |
|---|---|---|
| Ordered work submission | In-order command queue (default) | Default stream |
| Overlap copies and compute | Multiple queues, or out-of-order queue, with events for dependencies | Multiple streams with `cudaMemcpyAsync` |
| Event timing | `CL_QUEUE_PROFILING_ENABLE` + `clGetEventProfilingInfo` | `cudaEventRecord` / `cudaEventElapsedTime` |
| Kernel launching kernels | Device-side enqueue (OpenCL 2.0+, optional in 3.0) | Dynamic parallelism |
| Cross-work-group cooperation | None inside one kernel (except atomics) | Cooperative groups, grid-wide sync (`cudaLaunchCooperativeKernel`) |
| Warp-level operations | Sub-group built-ins (2.1+) or vendor extensions | `__shfl_sync`, `__ballot_sync`, warp reductions |
| Graph capture of repeated work | Not a core feature | CUDA Graphs |
| Tensor-core style matrix hardware | Vendor extensions only | WMMA / MMA / cuBLAS / cuDNN |

NVIDIA-specific hardware features (tensor cores, asynchronous copies, thread-block clusters) usually appear in CUDA first, and OpenCL reaches them, if at all, through extensions.

---

## 8. Portability: what it really means

OpenCL's portability promise is **functional**: a conformant implementation should run the same kernel source and produce comparable results. It does **not** promise that one tuned kernel is fast everywhere.

Reasons performance differs between devices:

- different SIMD widths (32, 64, or other);
- different local-memory sizes and bank structures;
- different register-file sizes and occupancy behaviour;
- CPU devices want vectorised loops with large work-groups, while GPUs want many small work-items;
- implementation quality of each vendor's compiler.

Practical strategies:

1. Query limits with `clGetDeviceInfo` rather than hard-coding them.
2. Expose tuning parameters (like the lab's `--local-x` / `--local-y`).
3. Pass tunables as `-D` build options so the compiler can specialise.
4. Accept that you may need one tuned variant per vendor for the best performance.

CUDA avoids this problem simply by supporting one vendor, at the cost of lock-in.

---

## 9. Ecosystem and tooling

| Area | OpenCL | CUDA |
|---|---|---|
| Linear algebra | CLBlast, clBLAS (legacy), ViennaCL | cuBLAS, cuSOLVER, cuSPARSE |
| FFT | clFFT, vkFFT | cuFFT |
| Deep learning | Limited (some back ends in specific frameworks) | cuDNN, TensorRT, first-class support in major frameworks |
| Parallel algorithms | Boost.Compute, vendor libraries | Thrust, CUB |
| Multi-GPU communication | Manual | NCCL |
| Profiler | Vendor tools (for example AMD, Intel tools), basic event timing | Nsight Systems, Nsight Compute |
| Debugging | Vendor debuggers, printf in kernels (`printf` supported in OpenCL 1.2) | `cuda-gdb`, Compute Sanitizer, `printf` |
| Language bindings | pyopencl, node-opencl bindings, Rust/Java/C# wrappers | PyCUDA, Numba CUDA, CuPy, Java/C# wrappers |
| Learning resources | Fewer; mostly specification-oriented | Extensive books, courses, samples |

---

## 10. Where each fits today

**OpenCL is a good fit when:**

- you must support GPUs from several vendors, or embedded and mobile accelerators;
- the target includes CPUs or FPGAs with the same programming model;
- you need a stable, vendor-neutral standard (for example, in long-lived scientific or industrial software);
- you want to teach the *concepts* of heterogeneous computing without tying students to one vendor (as this lab does).

**CUDA is a good fit when:**

- the target is NVIDIA hardware;
- you rely on its libraries (deep learning, linear algebra, sparse, communication);
- you need the latest hardware features or the best tooling;
- your team benefits from the large community and documentation base.

**Other options worth knowing about:**

| Technology | Relationship |
|---|---|
| **HIP** (AMD ROCm) | CUDA-like syntax that targets AMD GPUs (and NVIDIA through a thin layer); `hipify` tools translate CUDA source |
| **SYCL** | Khronos single-source C++ model; implementations can target OpenCL, CUDA, HIP, Level Zero and CPUs |
| **Vulkan compute / compute shaders** | Graphics-API route to GPU compute, with explicit control |
| **Metal** | Apple's platform; OpenCL is deprecated on macOS |
| **OpenMP offload / OpenACC** | Directive-based offloading, much less code to write |

---

## 11. Common myths

| Myth | Reality |
|---|---|
| "OpenCL is always slower than CUDA." | On the same NVIDIA GPU, well-tuned kernels are often within a small margin. Measured gaps usually come from driver maturity, missed features, or untuned launch parameters rather than from the language model itself. |
| "CUDA kernels can be compiled by an OpenCL driver." | No. The platforms, compilers, runtimes and binary formats differ. Source translation (or a tool such as HIP/SYCL) is required. |
| "OpenCL code is automatically fast on every device." | It is *portable*, not *performance-portable*. |
| "A GPU always beats the CPU." | Not when data volume is small, transfers dominate, or the algorithm is branch-heavy; see the lab's workload levels 1 and 2. |
| "Work-groups can synchronise with each other." | Not within one kernel launch in either platform (CUDA offers cooperative launches as a special case). |

---

## 12. Porting checklist (CUDA to OpenCL, or back)

1. Translate index calculations (`blockIdx * blockDim + threadIdx` vs `get_global_id`).
2. Re-express the launch geometry (grid in **blocks** vs global size in **work-items**).
3. Add address-space qualifiers when going to OpenCL; remove them going to CUDA.
4. Replace built-ins and math functions (see the keyword map).
5. Convert shared-memory declarations and barriers.
6. Replace host-side API calls, error handling, and object lifetimes.
7. Re-validate against a reference output and re-tune work-group sizes; do not assume the optimum carries over.

*Companion files: `01-how-opencl-works.md` and `03-opencl-cuda-keyword-map.md`.*

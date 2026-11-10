# OpenCL <-> CUDA Keyword Map

A lookup reference for translating between OpenCL and CUDA. Every table has the OpenCL term on the left and the CUDA term on the right; the **Notes** column flags where the match is *not* one-to-one.

**Legend:** `=` equivalent, `~` close but differs in behaviour, `!` trap (same word, different meaning, or no equivalent).

Contents:

1. [Concepts and hardware terms](#1-concepts-and-hardware-terms)
2. [Kernel and function qualifiers](#2-kernel-and-function-qualifiers)
3. [Memory spaces and qualifiers](#3-memory-spaces-and-qualifiers)
4. [Index and size built-ins](#4-index-and-size-built-ins)
5. [Synchronisation and memory fences](#5-synchronisation-and-memory-fences)
6. [Atomics](#6-atomics)
7. [Math functions](#7-math-functions)
8. [Data types](#8-data-types)
9. [Warp / sub-group operations](#9-warp--sub-group-operations)
10. [Host API: setup](#10-host-api-setup)
11. [Host API: memory](#11-host-api-memory)
12. [Host API: kernels and launch](#12-host-api-kernels-and-launch)
13. [Host API: queues, streams, events, timing](#13-host-api-queues-streams-events-timing)
14. [Device queries](#14-device-queries)
15. [Error codes](#15-error-codes)
16. [Compiler flags and tools](#16-compiler-flags-and-tools)
17. [Traps](#17-traps)
18. [Worked translation: lab launch code](#18-worked-translation-lab-launch-code)

---

## 1. Concepts and hardware terms

| OpenCL | CUDA | | Notes |
|---|---|---|---|
| Work-item | Thread | = | One logical kernel invocation |
| Work-group | Thread block (block) | = | Group that shares local/shared memory and barriers |
| NDRange | Grid | ~ | Same idea; **sized differently** (see section 4) |
| Global size | `gridDim * blockDim` | ~ | OpenCL specifies total work-items; CUDA specifies blocks |
| Local size | `blockDim` / block shape | = | |
| Number of work-groups | `gridDim` | = | |
| Sub-group (2.1+) | Warp | ~ | Warp is fixed at 32 on NVIDIA; sub-group size is implementation-defined and queryable |
| Wavefront (AMD term) | Warp | ~ | 32 or 64 lanes depending on the GPU |
| Compute Unit (CU) | Streaming Multiprocessor (SM) | ~ | Runs whole work-groups / blocks |
| Processing Element | CUDA core / lane | ~ | |
| Platform | (no equivalent) | ! | CUDA only supports NVIDIA, so there is no platform layer |
| Device | Device (`cudaDeviceProp`) | = | |
| Context | Context (Driver API) / primary context (Runtime API) | ~ | Runtime API creates and manages it implicitly |
| Command queue | Stream | ~ | |
| Event | Event | = | |
| Program | Module (`CUmodule`) | ~ | |
| Kernel object (`cl_kernel`) | Function (`CUfunction`) | ~ | |
| Buffer (`cl_mem`) | Device pointer | ~ | OpenCL buffers are opaque handles, not raw pointers |
| Image / sampler | Texture object / surface object | ~ | |
| SVM (Shared Virtual Memory) | Unified Memory (`cudaMallocManaged`) | ~ | |
| ICD loader | (CUDA driver) | ! | |
| OpenCL C | CUDA C++ | ~ | |
| SPIR-V | PTX | ~ | Portable intermediate forms, though PTX is NVIDIA-only |
| Device binary | SASS / cubin | ~ | |

---

## 2. Kernel and function qualifiers

| OpenCL C | CUDA C++ | | Notes |
|---|---|---|---|
| `__kernel` (or `kernel`) | `__global__` | = | Entry point callable from the host. Must return `void` in both. |
| *(plain function)* | `__device__` | ~ | Helper function called from device code. OpenCL helpers have no qualifier. |
| *(none)* | `__host__` | ! | Host code is a separate program in OpenCL. |
| *(none)* | `__host__ __device__` | ! | No equivalent; duplicate the function or use a shared header with macros. |
| `inline` / `static inline` | `__forceinline__` / `__noinline__` | ~ | |
| `__attribute__((reqd_work_group_size(X,Y,Z)))` | `__launch_bounds__(maxThreads, minBlocks)` | ~ | OpenCL fixes the size exactly; CUDA gives the compiler an upper bound |
| `__attribute__((work_group_size_hint(X,Y,Z)))` | `__launch_bounds__` | ~ | |
| `__attribute__((vec_type_hint(type)))` | (none) | | |
| `restrict` | `__restrict__` | = | |
| `const` | `const` | = | |
| `volatile` | `volatile` | = | |
| `__kernel` arg: `__global T *p` | `T *p` | ~ | CUDA pointers have no address-space qualifier in the type |

---

## 3. Memory spaces and qualifiers

| OpenCL C | CUDA C++ | | Notes |
|---|---|---|---|
| `__global` | *(plain pointer to device memory)* / `__device__` variable | ~ | |
| `__local` | `__shared__` | = | Memory shared by one work-group / block |
| `__constant` | `__constant__` | ~ | CUDA: 64 KB file-scope variable, set from the host with `cudaMemcpyToSymbol`. OpenCL: pointer qualifier; buffer passed as an argument. |
| `__private` (default for locals) | automatic local variables (registers) | = | |
| **(no direct equivalent)** | **"local memory"** | **!** | CUDA "local" memory is per-thread spill space in DRAM; it is **not** OpenCL `__local` |
| `__local` array argument: `clSetKernelArg(k, i, bytes, NULL)` | `extern __shared__ T s[]` with `<<<g, b, bytes>>>` | ~ | Dynamic local/shared memory sizing |
| `__local T tile[16][16];` | `__shared__ T tile[16][16];` | = | Static allocation |
| `__global` + `volatile` | `volatile` | ~ | |
| `__generic` (2.0+) | *(default; all pointers generic)* | ~ | |
| Pipes (2.0+) | (none) | | |
| `image2d_t`, `read_imagef` | `tex2D<T>`, texture objects | ~ | |

Buffer access flags (host side):

| OpenCL | CUDA | Notes |
|---|---|---|
| `CL_MEM_READ_ONLY` | *(no flag; use `const` pointer or `__ldg`)* | OpenCL flag restricts kernel access |
| `CL_MEM_WRITE_ONLY` | *(none)* | |
| `CL_MEM_READ_WRITE` | *(default)* | |
| `CL_MEM_COPY_HOST_PTR` | `cudaMalloc` + `cudaMemcpy` | Initialise at creation |
| `CL_MEM_USE_HOST_PTR` | `cudaHostRegister` / mapped host memory | Use host storage directly |
| `CL_MEM_ALLOC_HOST_PTR` | `cudaHostAlloc` / `cudaMallocHost` | Driver-allocated host-visible memory |

---

## 4. Index and size built-ins

For dimension `d` in `{0,1,2}` (OpenCL) corresponding to `.x`, `.y`, `.z` (CUDA):

| OpenCL C | CUDA C++ | | Notes |
|---|---|---|---|
| `get_local_id(d)` | `threadIdx.x/.y/.z` | = | |
| `get_group_id(d)` | `blockIdx.x/.y/.z` | = | |
| `get_local_size(d)` | `blockDim.x/.y/.z` | = | |
| `get_num_groups(d)` | `gridDim.x/.y/.z` | = | |
| `get_global_id(d)` | `blockIdx.d * blockDim.d + threadIdx.d` | ~ | CUDA has no built-in; also OpenCL result includes any `global_work_offset` |
| `get_global_size(d)` | `gridDim.d * blockDim.d` | ~ | **Total work-items**; not `gridDim` |
| `get_global_offset(d)` | *(none)* | ! | OpenCL can offset the NDRange; CUDA code adds an offset manually |
| `get_work_dim()` | *(none; fixed at compile time by usage)* | | |
| `get_enqueued_local_size(d)` (2.0+) | `blockDim.d` | ~ | Non-uniform work-groups |
| `get_sub_group_id()`, `get_sub_group_local_id()` (2.1+) | `threadIdx.x / warpSize`, `threadIdx.x % warpSize` (1-D blocks) or `__laneid` | ~ | |
| `get_sub_group_size()` | `warpSize` | ~ | |
| Return type `size_t` | `unsigned int` (built-in uint3) | ! | Mind signed/unsigned casts: the lab kernel does `(int)get_global_id(0)` |

Index formulas:

```text
OpenCL:  gid = get_group_id(0) * get_local_size(0) + get_local_id(0)  (+ offset)
CUDA:    gid = blockIdx.x      * blockDim.x         + threadIdx.x
```

Flattening a 2-D index (row-major, as in the lab):

```text
OpenCL:  idx = get_global_id(1) * width + get_global_id(0)
CUDA:    idx = (blockIdx.y * blockDim.y + threadIdx.y) * width
             + (blockIdx.x * blockDim.x + threadIdx.x)
```

---

## 5. Synchronisation and memory fences

| OpenCL C | CUDA C++ | | Notes |
|---|---|---|---|
| `barrier(CLK_LOCAL_MEM_FENCE)` | `__syncthreads()` | ~ | OpenCL's argument also selects what is made visible; CUDA's `__syncthreads()` orders both shared and global writes within the block |
| `barrier(CLK_GLOBAL_MEM_FENCE)` | `__syncthreads()` | ~ | |
| `barrier(CLK_LOCAL_MEM_FENCE \| CLK_GLOBAL_MEM_FENCE)` | `__syncthreads()` | ~ | |
| `work_group_barrier(flags)` (2.0+) | `__syncthreads()` | ~ | |
| `mem_fence(flags)` | `__threadfence_block()` | ~ | |
| `read_mem_fence`, `write_mem_fence` | `__threadfence_block()` | ~ | CUDA has no read-only / write-only fence variants |
| *(global-scope fence)* `mem_fence(CLK_GLOBAL_MEM_FENCE)` | `__threadfence()` | ~ | Device-wide visibility ordering |
| `sub_group_barrier()` (2.1+) | `__syncwarp()` | ~ | |
| `work_group_reduce_add` etc. (2.0+) | `cooperative_groups::reduce`, `__reduce_add_sync` | ~ | |
| `async_work_group_copy` | `cuda::memcpy_async`, `cp.async` | ~ | Async global-to-shared copies |
| `wait_group_events` | `cuda::pipeline` / `__pipeline_wait_prior` | ~ | |
| *(no grid-wide barrier)* | `cooperative_groups::this_grid().sync()` | ! | Needs a cooperative launch in CUDA; not available in OpenCL |

> **Rule shared by both:** every work-item / thread in the group must reach the same barrier. A barrier inside a divergent `if` is undefined behaviour.

---

## 6. Atomics

| OpenCL C (1.x, 32-bit ints) | CUDA C++ | Notes |
|---|---|---|
| `atomic_add(p, v)` | `atomicAdd(p, v)` | Returns the old value in both |
| `atomic_sub` | `atomicSub` | |
| `atomic_xchg` | `atomicExch` | |
| `atomic_inc` / `atomic_dec` | `atomicAdd(p, 1)` / `atomicSub(p, 1)` (CUDA's own `atomicInc`/`atomicDec` have *wrapping* semantics) | **!** Different meaning of `atomicInc` |
| `atomic_cmpxchg` | `atomicCAS` | |
| `atomic_min` / `atomic_max` | `atomicMin` / `atomicMax` | |
| `atomic_and` / `atomic_or` / `atomic_xor` | `atomicAnd` / `atomicOr` / `atomicXor` | |
| `atom_*` (64-bit, extension) | `atomicAdd(unsigned long long*)` etc. | |
| `atomic_fetch_add_explicit(obj, v, order, scope)` (2.0+) | `cuda::atomic_ref<T, scope>::fetch_add`, `atomicAdd_block`, `atomicAdd_system` | ~ |
| `memory_order_*`, `memory_scope_*` | `cuda::memory_order_*`, `cuda::thread_scope_*` | ~ |
| Float atomics: extension only | `atomicAdd(float*, float)` built in | ! |

The lab's worker threads use JavaScript `Atomics.add` to claim row ranges, the CPU-side analogue of `atomic_add` on a global counter.

---

## 7. Math functions

OpenCL built-ins are overloaded on type (`exp` works on `float`, `double`, vectors). CUDA C++ also provides overloads, but C-style names with an `f` suffix are common.

| OpenCL C | CUDA C++ (float) | | Notes |
|---|---|---|---|
| `exp(x)` | `expf(x)` | = | |
| `native_exp(x)` | `__expf(x)` | ~ | Fast, low-accuracy hardware version |
| `exp2`, `exp10` | `exp2f`, `exp10f` | = | |
| `log`, `log2`, `log10` | `logf`, `log2f`, `log10f` | = | |
| `native_log` | `__logf` | ~ | |
| `pow(x, y)` | `powf(x, y)` | = | |
| `native_powr`, `powr` | `__powf` | ~ | |
| `sqrt` / `rsqrt` | `sqrtf` / `rsqrtf` | = | |
| `native_sqrt` / `native_rsqrt` | `__fsqrt_rn` / `__frsqrt_rn` or `rsqrtf` | ~ | |
| `sin`, `cos`, `tan` | `sinf`, `cosf`, `tanf` | = | |
| `native_sin`, `native_cos` | `__sinf`, `__cosf` | ~ | |
| `fabs` | `fabsf` | = | |
| `fmin`, `fmax` | `fminf`, `fmaxf` | = | |
| `min`, `max` (ints and floats) | `min`, `max` (ints), `fminf`, `fmaxf` | ~ | |
| `clamp(x, lo, hi)` | *(none)*: `min(max(x, lo), hi)` | **!** | CUDA has no `clamp` for integers or floats; `__saturatef` clamps only to [0,1] |
| `mix(a, b, t)` | *(none)*: `a + (b - a) * t` or `fmaf` | ! | |
| `step`, `smoothstep` | *(none)* | ! | Write by hand |
| `mad(a, b, c)` | `fmaf(a, b, c)` / `__fmaf_rn` | ~ | OpenCL `mad` may have reduced accuracy; `fma` is exact |
| `fma` | `fmaf` | = | |
| `floor`, `ceil`, `round`, `trunc` | `floorf`, `ceilf`, `roundf`, `truncf` | = | |
| `rint` | `rintf` | = | |
| `convert_uchar_sat(x)` | `__float2uint_rn` with manual saturation | ~ | OpenCL has built-in saturating conversions; CUDA needs explicit clamping |
| `convert_float(x)` | `(float)x` / `__int2float_rn(x)` | ~ | |
| `as_float(u)` (bit reinterpret) | `__uint_as_float(u)` | = | |
| `as_uint(f)` | `__float_as_uint(f)` | = | |
| `mul24(a, b)`, `mad24` | `__mul24(a, b)`, `__umul24` | ~ | Rarely useful on modern hardware |
| `mul_hi` | `__mulhi` / `__umulhi` | = | |
| `popcount` | `__popc` | = | |
| `clz` | `__clz` | = | |
| `printf` (1.2+) | `printf` | = | Buffered; flushed at kernel completion |

Compiler-level fast math:

| OpenCL build option | CUDA nvcc option |
|---|---|
| `-cl-fast-relaxed-math` | `--use_fast_math` |
| `-cl-mad-enable` | `--fmad=true` (default) |
| `-cl-no-signed-zeros` | (included in `--use_fast_math`) |
| `-cl-denorms-are-zero` | `--ftz=true` |
| `-cl-single-precision-constant` | *(use `f` suffix)* |

---

## 8. Data types

| OpenCL C | CUDA C++ | | Notes |
|---|---|---|---|
| `char`, `uchar` | `signed char`, `unsigned char` | = | |
| `short`, `ushort` | `short`, `unsigned short` | = | |
| `int`, `uint` | `int`, `unsigned int` | = | |
| `long`, `ulong` | `long long`, `unsigned long long` | ! | **`long` is 64-bit in OpenCL, but 32-bit on Windows CUDA hosts**; use `long long` |
| `float` | `float` | = | |
| `double` | `double` | ~ | OpenCL: optional (`cl_khr_fp64`); CUDA: supported on all recent hardware |
| `half` | `__half` (`cuda_fp16.h`) | ~ | |
| `size_t` | `size_t` | = | Width differs by device addressing in OpenCL |
| `float2`, `float4`, `uchar4`, ... | `float2`, `float4`, `uchar4`, ... | ~ | OpenCL vector types support swizzles (`.xyzw`, `.s0123`) and operators; CUDA vector types are plain structs |
| `float8`, `float16`, `int8`, ... | *(none)* | ! | CUDA vectors stop at 4 components |
| `v.xy`, `v.s01`, `v.lo`, `v.hi` | `v.x`, `v.y` | ! | CUDA has no swizzles |
| `vload4(i, p)` / `vstore4(v, i, p)` | `reinterpret_cast<float4*>(p)[i]` | ~ | |
| `(float4)(a, b, c, d)` | `make_float4(a, b, c, d)` | ~ | |
| `(uchar)x` cast | `(unsigned char)x` | = | |
| `convert_T[_sat][_rte]()` | cast / intrinsics (`__float2int_rn`) | ~ | |
| `bool` | `bool` | ~ | OpenCL comparisons on vectors return -1/0 per lane |
| `event_t` (in-kernel) | *(none; use cuda::pipeline)* | | |
| `image2d_t`, `sampler_t` | `cudaTextureObject_t` | ~ | |

Host-side type names:

| OpenCL | CUDA |
|---|---|
| `cl_int`, `cl_uint` | `int`, `unsigned` |
| `cl_float` | `float` |
| `cl_mem` | `void*` / `T*` device pointer |
| `cl_context`, `cl_command_queue`, `cl_program`, `cl_kernel`, `cl_event` | `CUcontext`, `cudaStream_t`, `CUmodule`, `CUfunction`, `cudaEvent_t` |
| `size_t[3]` (global/local) | `dim3` |

---

## 9. Warp / sub-group operations

| OpenCL C (2.1+ / `cl_khr_subgroups`) | CUDA C++ | Notes |
|---|---|---|
| `sub_group_broadcast(x, id)` | `__shfl_sync(mask, x, id)` | |
| `sub_group_reduce_add(x)` | `__reduce_add_sync(mask, x)` (ints, newer GPUs) or shuffle loop | |
| `sub_group_scan_inclusive_add(x)` | shuffle-based scan | |
| `sub_group_all(p)` | `__all_sync(mask, p)` | |
| `sub_group_any(p)` | `__any_sync(mask, p)` | |
| `sub_group_shuffle*` (`cl_khr_subgroup_shuffle`) | `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync` | |
| `sub_group_ballot` (`cl_khr_subgroup_ballot`) | `__ballot_sync` | |
| `get_sub_group_size()` | `warpSize` | |

CUDA's `*_sync` functions take an explicit participation **mask**; OpenCL's operate on the whole sub-group.

---

## 10. Host API: setup

| Step | OpenCL (C) | CUDA Runtime | CUDA Driver | Notes |
|---|---|---|---|---|
| Initialise | *(implicit)* | *(implicit)* | `cuInit(0)` | |
| List platforms | `clGetPlatformIDs` | *(none)* | *(none)* | |
| List devices | `clGetDeviceIDs` | `cudaGetDeviceCount` | `cuDeviceGetCount` | |
| Select device | pick `cl_device_id` | `cudaSetDevice(i)` | `cuDeviceGet` | |
| Device properties | `clGetDeviceInfo` | `cudaGetDeviceProperties` / `cudaDeviceGetAttribute` | `cuDeviceGetAttribute` | |
| Create context | `clCreateContext` | *(implicit)* | `cuCtxCreate` | |
| Create queue / stream | `clCreateCommandQueue` (`...WithProperties` in 2.0) | `cudaStreamCreate` | `cuStreamCreate` | |
| Convenience (lab) | `cl.quickStart(true)` | *(default device)* | | Binding-specific helper |
| Release | `clReleaseContext`, `clReleaseCommandQueue` | `cudaStreamDestroy`, `cudaDeviceReset` | `cuCtxDestroy` | OpenCL uses reference counts; CUDA runtime releases at process exit |

---

## 11. Host API: memory

| Action | OpenCL (C) | CUDA Runtime | Notes |
|---|---|---|---|
| Allocate device memory | `clCreateBuffer(ctx, flags, size, host_ptr, &err)` | `cudaMalloc(&ptr, size)` | |
| Free | `clReleaseMemObject` | `cudaFree` | |
| Copy host to device | `clEnqueueWriteBuffer(q, buf, blocking, off, n, src, ...)` | `cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice)` | OpenCL has an offset parameter and queue |
| Copy device to host | `clEnqueueReadBuffer(...)` | `cudaMemcpy(..., cudaMemcpyDeviceToHost)` | |
| Copy device to device | `clEnqueueCopyBuffer` | `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` | |
| Async copy | non-blocking flag + events | `cudaMemcpyAsync(..., stream)` | CUDA needs pinned host memory for true overlap |
| Fill / memset | `clEnqueueFillBuffer` (1.2) | `cudaMemset`, `cudaMemsetAsync` | |
| Rectangular copy | `clEnqueueReadBufferRect` / `WriteBufferRect` | `cudaMemcpy2D` | |
| Map to host | `clEnqueueMapBuffer` / `clEnqueueUnmapMemObject` | `cudaHostGetDevicePointer` (mapped), managed memory | |
| Pinned host memory | `CL_MEM_ALLOC_HOST_PTR` | `cudaMallocHost` / `cudaHostAlloc` | |
| Unified / shared virtual memory | `clSVMAlloc`, `clSVMFree` (2.0) | `cudaMallocManaged` | |
| Sub-buffer | `clCreateSubBuffer` | pointer arithmetic | |
| Image object | `clCreateImage` | `cudaMallocArray` + texture object | |
| Query free memory | *(none standard; vendor extension)* | `cudaMemGetInfo` | The lab reports **allocated** buffer bytes, not free/used VRAM, for this reason |
| Blocking flag | `CL_TRUE` | *(`cudaMemcpy` blocks by default)* | |

---

## 12. Host API: kernels and launch

| Action | OpenCL (C) | CUDA | Notes |
|---|---|---|---|
| Source to program | `clCreateProgramWithSource` | *(compile with `nvcc`)* / NVRTC `nvrtcCreateProgram` | |
| Compile | `clBuildProgram`, `clCompileProgram` + `clLinkProgram` | `nvcc`, `nvrtcCompileProgram` | |
| Binary to program | `clCreateProgramWithBinary` | `cuModuleLoadData`, `cuModuleLoad` | |
| Get binary | `clGetProgramInfo(CL_PROGRAM_BINARIES)` | `cuobjdump`, `nvcc -cubin` | |
| Build log | `clGetProgramBuildInfo(CL_PROGRAM_BUILD_LOG)` | compiler output / `nvrtcGetProgramLog` | Lab writes it to `opencl-build.log` |
| Get kernel by name | `clCreateKernel(prog, "name", &err)` | *(symbol)* / `cuModuleGetFunction` | |
| Set one argument | `clSetKernelArg(k, i, size, &value)` | *(passed in the launch call)* | |
| Local/shared memory arg | `clSetKernelArg(k, i, bytes, NULL)` | third launch parameter `<<<g, b, bytes>>>` | |
| Launch | `clEnqueueNDRangeKernel(q, k, dim, offset, global, local, nwait, wait, &evt)` | `kernel<<<grid, block, shmem, stream>>>(args)` / `cuLaunchKernel` | **global = work-items; grid = blocks** |
| Launch with automatic local size | pass `NULL` for `local_work_size` | *(you must pick a block)* / `cudaOccupancyMaxPotentialBlockSize` | |
| Single work-item task | `clEnqueueTask` (deprecated) | `<<<1, 1>>>` | |
| Wait for queue | `clFinish(q)` | `cudaStreamSynchronize(s)`, `cudaDeviceSynchronize()` | Both are required before timing or reading results (unless a blocking copy follows) |
| Flush without waiting | `clFlush(q)` | *(implicit)* | |
| Kernel work-group limit | `clGetKernelWorkGroupInfo(CL_KERNEL_WORK_GROUP_SIZE)` | `cudaFuncGetAttributes(...).maxThreadsPerBlock` | |
| Preferred multiple | `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` | `warpSize` | |
| Release kernel/program | `clReleaseKernel`, `clReleaseProgram` | `cuModuleUnload` | |

---

## 13. Host API: queues, streams, events, timing

| Action | OpenCL (C) | CUDA | Notes |
|---|---|---|---|
| In-order work queue | default `cl_command_queue` | default / created stream | |
| Out-of-order execution | `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` | multiple streams | |
| Enable hardware timing | `CL_QUEUE_PROFILING_ENABLE` | *(events always timestamp)* | Lab tries `QUEUE_PROFILING_ENABLE` but reports host-synchronised time |
| Event from a command | `cl_event *` last argument of any `clEnqueue*` | `cudaEventRecord(evt, stream)` | |
| Wait on event(s) | `clWaitForEvents` | `cudaEventSynchronize` / `cudaStreamWaitEvent` | |
| Dependency lists | `event_wait_list` parameter | `cudaStreamWaitEvent` | |
| Elapsed time | `clGetEventProfilingInfo(CL_PROFILING_COMMAND_END - ..._START)` (nanoseconds) | `cudaEventElapsedTime(&ms, a, b)` (milliseconds) | |
| Marker / barrier | `clEnqueueMarkerWithWaitList`, `clEnqueueBarrierWithWaitList` | event record / `cudaStreamWaitEvent` | |
| User event | `clCreateUserEvent` | *(none)* | |
| Callback | `clSetEventCallback` | `cudaLaunchHostFunc`, `cudaStreamAddCallback` | |
| Release event | `clReleaseEvent` | `cudaEventDestroy` | |

---

## 14. Device queries

| Property | OpenCL (`clGetDeviceInfo`) | CUDA (`cudaDeviceProp` / attribute) |
|---|---|---|
| Device name | `CL_DEVICE_NAME` | `name` |
| Vendor | `CL_DEVICE_VENDOR` | *(always NVIDIA)* |
| Compute units | `CL_DEVICE_MAX_COMPUTE_UNITS` | `multiProcessorCount` |
| Max work-group / block size | `CL_DEVICE_MAX_WORK_GROUP_SIZE` | `maxThreadsPerBlock` |
| Max size per dimension | `CL_DEVICE_MAX_WORK_ITEM_SIZES` | `maxThreadsDim[3]` |
| Max dimensions | `CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS` | 3 (fixed) |
| Global memory size | `CL_DEVICE_GLOBAL_MEM_SIZE` | `totalGlobalMem` |
| Max single allocation | `CL_DEVICE_MAX_MEM_ALLOC_SIZE` | *(near total memory)* |
| Local/shared memory | `CL_DEVICE_LOCAL_MEM_SIZE` | `sharedMemPerBlock` |
| Constant memory | `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE` | `totalConstMem` |
| Clock rate | `CL_DEVICE_MAX_CLOCK_FREQUENCY` (MHz) | `clockRate` (kHz) |
| Global cache size | `CL_DEVICE_GLOBAL_MEM_CACHE_SIZE` | `l2CacheSize` |
| Warp / sub-group size | `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` or sub-group queries | `warpSize` |
| Double-precision support | `CL_DEVICE_DOUBLE_FP_CONFIG` | `major.minor` compute capability |
| API / version | `CL_DEVICE_VERSION`, `CL_DEVICE_OPENCL_C_VERSION` | `major`, `minor` (compute capability), runtime version |
| Extensions | `CL_DEVICE_EXTENSIONS` | *(compute capability determines features)* |
| Integrated or discrete | `CL_DEVICE_HOST_UNIFIED_MEMORY` | `integrated` |

---

## 15. Error codes

| OpenCL | CUDA | Meaning |
|---|---|---|
| `CL_SUCCESS` (0) | `cudaSuccess` (0) | OK |
| `CL_DEVICE_NOT_FOUND` / `CL_PLATFORM_NOT_FOUND_KHR` | `cudaErrorNoDevice` | No usable device |
| `CL_OUT_OF_HOST_MEMORY` | `cudaErrorMemoryAllocation` | Host allocation failed |
| `CL_MEM_OBJECT_ALLOCATION_FAILURE` | `cudaErrorMemoryAllocation` | Device allocation failed |
| `CL_OUT_OF_RESOURCES` | `cudaErrorLaunchOutOfResources`, `cudaErrorIllegalAddress` | Too many registers, or illegal access in the kernel |
| `CL_BUILD_PROGRAM_FAILURE` | *(nvcc error at build time)* / `NVRTC_ERROR_COMPILATION` | Compile error |
| `CL_INVALID_WORK_GROUP_SIZE` | `cudaErrorInvalidConfiguration` | Bad local size / block size |
| `CL_INVALID_WORK_ITEM_SIZE` | `cudaErrorInvalidConfiguration` | Dimension limit exceeded |
| `CL_INVALID_KERNEL_ARGS` | *(typically a compile error)* | Missing or wrong kernel arguments |
| `CL_INVALID_ARG_SIZE`, `CL_INVALID_ARG_VALUE` | *(compile-time type error)* | Argument mismatch |
| `CL_INVALID_MEM_OBJECT` | `cudaErrorInvalidDevicePointer` | Bad buffer |
| `CL_INVALID_COMMAND_QUEUE` | `cudaErrorInvalidResourceHandle` | Bad queue/stream |
| `CL_INVALID_KERNEL_NAME` | `cudaErrorInvalidDeviceFunction` / `CUDA_ERROR_NOT_FOUND` | Kernel not found |

Error-handling style: OpenCL returns an `int` code or writes through an `errcode_ret` pointer. CUDA returns `cudaError_t` and also keeps a sticky "last error" read with `cudaGetLastError()`, which is necessary after a kernel launch because launches themselves return no value.

---

## 16. Compiler flags and tools

| Purpose | OpenCL | CUDA |
|---|---|---|
| Define a macro | `-D NAME=value` (build options string) | `-DNAME=value` |
| Include path | `-I dir` | `-I dir` |
| Optimisation off | `-cl-opt-disable` | `-O0`, `-G` (debug device code) |
| Strict IEEE | `-cl-fp32-correctly-rounded-divide-sqrt` | `--prec-div=true --prec-sqrt=true` |
| Fast math | `-cl-fast-relaxed-math` | `--use_fast_math` |
| Choose language version | `-cl-std=CL1.2` | `-std=c++17`, `-arch=sm_xx` |
| Target architecture | *(runtime picks device)* | `-arch=sm_xx`, `-gencode` |
| Warnings off | `-w` | `-w` |
| Inspect generated code | vendor tools; `CL_PROGRAM_BINARIES` | `--ptx`, `cuobjdump -sass`, `nvdisasm` |
| Report register use | vendor tools / `clGetKernelWorkGroupInfo` | `--ptxas-options=-v` |
| Profile | vendor profilers / events | `nsys`, `ncu` |
| Debug / memory check | vendor tools | `cuda-gdb`, `compute-sanitizer` |
| Convert one to the other | SYCL / HIP tooling | `hipify-perl`, `hipify-clang` |

---

## 17. Traps

| # | Trap | Explanation |
|---|---|---|
| 1 | **"local" means different things** | OpenCL `__local` = CUDA `__shared__`. CUDA "local memory" = per-thread spill memory (OpenCL private). |
| 2 | **Launch size units** | OpenCL global size counts work-items; CUDA grid counts blocks. Dividing or multiplying by the local size is the most common porting mistake. |
| 3 | **No built-in global ID in CUDA** | Always compute `blockIdx * blockDim + threadIdx` and guard with bounds checks, just as the lab kernel does with `get_global_id`. |
| 4 | **Multiple of local size** | OpenCL 1.2 requires the global size to be an exact multiple of the local size; CUDA does not, because you give only the block count. |
| 5 | **`clamp`, `mix`, `step`, `smoothstep`** | Built in to OpenCL C; absent from CUDA C++. Write helpers. |
| 6 | **`long` width** | OpenCL `long` is always 64-bit; CUDA `long` follows the host ABI (32-bit on Windows). Use `long long`. |
| 7 | **Vector types** | OpenCL has 2/3/4/8/16-component vectors with swizzles; CUDA stops at 4 and has none. |
| 8 | **`atomicInc` / `atomicDec`** | CUDA's versions wrap around at a limit; OpenCL's `atomic_inc` simply adds 1. |
| 9 | **Kernel launches are asynchronous in both** | Time only after `clFinish` / `cudaDeviceSynchronize`, as the lab does. |
| 10 | **Run-time vs build-time compile** | OpenCL kernel errors appear at run time; CUDA's appear in `nvcc`. |
| 11 | **Opaque buffers** | You cannot do pointer arithmetic on a `cl_mem` on the host (use sub-buffers or offsets); CUDA device pointers can be offset directly. |
| 12 | **`const` / address-space pointer types** | OpenCL pointers carry `__global` / `__local` / `__constant` in the type; assigning between spaces is a compile error. |
| 13 | **Fast math differences** | `native_*` and `__*` functions have different accuracy rules; results may differ slightly between platforms (the lab's validation step exists for this reason). |
| 14 | **Barrier divergence** | Both require every work-item/thread in the group to reach the barrier. |
| 15 | **Binary portability** | Neither a `cl_program` binary nor a CUDA `cubin` is portable across vendors; keep the source. |

---

## 18. Worked translation: lab launch code

**Lab (Node.js OpenCL binding):**

```js
const globalSize = [paddedW, paddedH];   // work-items, multiples of local
const localSize  = [16, 16];

cl.enqueueWriteBuffer(queue, deviceInput, true, 0, input.byteLength, input);
cl.enqueueNDRangeKernel(queue, kernel, 2, undefined, globalSize, localSize);
cl.finish(queue);
cl.enqueueReadBuffer(queue, deviceOutput, true, 0, output.byteLength, output);
```

**CUDA Runtime equivalent:**

```cuda
dim3 block(16, 16);
dim3 grid(paddedW / 16, paddedH / 16);   // blocks, NOT work-items
                                         // (or (width + 15) / 16 without pre-padding)

cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
bilateral_filter<<<grid, block>>>(d_in, d_out, width, height, radius, ss, sc);
cudaGetLastError();                      // launch errors
cudaDeviceSynchronize();                 // equivalent of cl.finish(queue)
cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
```

**Mapping of lab report fields:**

| Lab `profile.json` field | OpenCL meaning | CUDA counterpart |
|---|---|---|
| `launch.globalSize` | Padded NDRange in work-items | `gridDim * blockDim` |
| `launch.localSize` | Work-group shape | `blockDim` |
| `launch.workGroupCount` | Number of work-groups | `gridDim.x * gridDim.y` |
| `compileMs` | `clBuildProgram` time | `nvcc` time (build) or NVRTC/JIT time |
| `uploadMsMean` | `clEnqueueWriteBuffer` | `cudaMemcpy` H2D |
| `kernelHostSynchronizedMsMean` | enqueue + `clFinish` | launch + `cudaDeviceSynchronize` |
| `downloadMsMean` | `clEnqueueReadBuffer` | `cudaMemcpy` D2H |
| `memory.totalDeviceBufferAllocationBytes` | Sum of `clCreateBuffer` sizes | Sum of `cudaMalloc` sizes |

---

*Companion files: `01-how-opencl-works.md` and `02-opencl-vs-cuda.md`.*

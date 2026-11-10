# How OpenCL Works

A guide to the OpenCL programming model, using the bilateral-filter lab as the running example.

> **Scope.** The lab targets **OpenCL 1.2** through the Node.js binding `@node-3d/opencl`. Where later versions (2.x, 3.0) behave differently, this guide says so.

---

## 1. What OpenCL is

OpenCL (Open Computing Language) is an **open standard, maintained by the Khronos Group**, for writing programs that run on *heterogeneous* hardware: GPUs, multi-core CPUs, FPGAs, DSPs and other accelerators from many vendors.

OpenCL is **not a library you link against** the way you link a math library. It is a specification with two parts:

| Part | What it is | Where it runs |
|---|---|---|
| **Host API** | A C API (`clCreateBuffer`, `clEnqueueNDRangeKernel`, ...) for finding devices, allocating memory, compiling kernels and launching work | On the CPU, in your application |
| **Kernel language** | *OpenCL C* (a C99-based dialect; C++ for OpenCL and SPIR-V are alternatives) for the functions that run on the device | On the accelerator |

Each hardware vendor ships its own **implementation** inside its driver. Applications find those implementations through the **ICD loader** (Installable Client Driver): a small library (`OpenCL.dll`, `libOpenCL.so`) that reads a registry of installed drivers and forwards each call to the right one.

```
Your program  ->  ICD loader (libOpenCL)  ->  vendor driver  ->  hardware
                                          |->  NVIDIA driver  -> NVIDIA GPU
                                          |->  AMD driver     -> AMD GPU
                                          |->  Intel runtime  -> Intel GPU / CPU
                                          '->  PoCL, etc.     -> CPU
```

This is why the lab's README says that installing the npm package is not enough: the JavaScript binding only talks to an OpenCL runtime that must already be installed with the GPU driver.

---

## 2. Four models in one specification

The OpenCL specification describes four connected models. Learn these and the API stops feeling arbitrary.

1. **Platform model**: what hardware exists and how it is organised.
2. **Execution model**: how kernels are launched and how work is divided.
3. **Memory model**: where data lives and who can see it.
4. **Programming model**: how host and device code are written and built.

---

## 3. Platform model

```
Host (CPU + system RAM)
 '-- Platform (one vendor's OpenCL implementation)
      '-- Device (e.g. one GPU)
           '-- Compute Unit (CU)            <- NVIDIA: SM, AMD: CU, Intel: subslice/EU group
                '-- Processing Element (PE) <- NVIDIA: "CUDA core", the unit that runs one work-item lane
```

| Term | Meaning |
|---|---|
| **Host** | The CPU program that controls everything. In the lab, this is Node.js running `benchmark.mjs`. |
| **Platform** | A vendor's implementation. A machine can expose several (for example NVIDIA plus Intel). |
| **Device** | A processor that can execute kernels. One platform can expose several devices. |
| **Compute Unit** | A group of processing elements with shared resources, such as local memory. A work-group runs entirely on **one** compute unit. |
| **Processing Element** | The smallest execution lane. |

**Lab note:** `cl.quickStart(true)` picks the first platform/device the binding finds. A larger application would enumerate with `clGetPlatformIDs` and `clGetDeviceIDs` and let the user choose.

---

## 4. The host program lifecycle

Almost every OpenCL program follows the same sequence. The table shows the C API call, the matching binding call used in `benchmark.mjs`, and what happens.

| # | Step | C API | Lab (Node binding) | What happens |
|---|---|---|---|---|
| 1 | Pick platform and device | `clGetPlatformIDs`, `clGetDeviceIDs` | `cl.quickStart(true)` | Finds the hardware to use |
| 2 | Create a **context** | `clCreateContext` | returned by `quickStart` | Owns buffers, programs and kernels for the chosen device(s) |
| 3 | Create a **command queue** | `clCreateCommandQueue` | `cl.createCommandQueue(context, device)` | The channel through which the host sends work to the device |
| 4 | Create a **program** from source | `clCreateProgramWithSource` | `cl.createProgramWithSource(context, KERNEL_SOURCE)` | Wraps the kernel text; nothing is compiled yet |
| 5 | **Build** the program | `clBuildProgram` | `cl.buildProgram(program[, options])` | The driver compiles the source *for the target device at run time* |
| 6 | Create a **kernel** object | `clCreateKernel` | `cl.createKernel(program, 'bilateral_filter')` | Selects one `__kernel` function by name |
| 7 | Create **buffers** | `clCreateBuffer` | `cl.createBuffer(context, cl.MEM_READ_ONLY, bytes)` | Allocates device-accessible memory |
| 8 | Set **kernel arguments** | `clSetKernelArg` | `cl.setKernelArg(kernel, i, 'uint', width)` | Binds buffers and scalars to the kernel's parameters |
| 9 | **Upload** data | `clEnqueueWriteBuffer` | `cl.enqueueWriteBuffer(queue, buf, true, 0, n, data)` | Host to device copy |
| 10 | **Launch** the kernel | `clEnqueueNDRangeKernel` | `cl.enqueueNDRangeKernel(queue, kernel, 2, undefined, globalSize, localSize)` | Starts the 2-D index space |
| 11 | **Wait** | `clFinish` | `cl.finish(queue)` | Blocks until everything in the queue has completed |
| 12 | **Download** results | `clEnqueueReadBuffer` | `cl.enqueueReadBuffer(queue, buf, true, 0, n, out)` | Device to host copy |
| 13 | **Release** objects | `clRelease*` | `cl.releaseKernel`, `releaseProgram`, `releaseMemObject`, `releaseCommandQueue`, `releaseContext` | OpenCL objects are reference counted; the host must free them |

The lab's `setupMs` and `compileMs` correspond to steps 1 to 8; upload, kernel and download timings correspond to steps 9 to 12.

> **Binding quirk.** `cl.setKernelArg(kernel, 0, 'uchar*', deviceInput)` takes a *type string* because JavaScript has no static types. The C API instead takes `(kernel, index, size, pointer)`. Check the binding's documentation when you move between versions.

---

## 5. Execution model

### 5.1 NDRange, work-items and work-groups

A kernel launch creates an **NDRange**: an N-dimensional index space (N = 1, 2 or 3). Every point in that space is one **work-item**, and each work-item runs the kernel function once.

The NDRange is divided into equally sized **work-groups**.

```
Global size 48 x 32, local size 16 x 16  ->  3 x 2 = 6 work-groups

   x ->   0..15     16..31    32..47
 y 0..15  [ WG 0,0 ][ WG 1,0 ][ WG 2,0 ]
 y 16..31 [ WG 0,1 ][ WG 1,1 ][ WG 2,1 ]
```

Inside a kernel, each work-item asks "where am I?":

| Function | Returns |
|---|---|
| `get_global_id(d)` | Position in the whole NDRange along dimension `d` |
| `get_local_id(d)` | Position inside its work-group |
| `get_group_id(d)` | Which work-group it belongs to |
| `get_global_size(d)` | Total work-items along `d` |
| `get_local_size(d)` | Work-items per work-group along `d` |
| `get_num_groups(d)` | Number of work-groups along `d` |

The identity that ties them together (with no global offset):

```
get_global_id(d) == get_group_id(d) * get_local_size(d) + get_local_id(d)
```

### 5.2 Why work-groups exist

Work-groups are the **unit of cooperation**:

- All work-items in a group run on the **same compute unit**.
- They can share fast **`__local` memory**.
- They can synchronise with **`barrier()`**.
- **Different work-groups cannot synchronise with each other** inside one kernel launch, and the order in which groups run is undefined. This restriction is what lets the same program scale from a small laptop GPU to a large one: the hardware may run groups one after another or hundreds at once.

The bilateral filter does not need cooperation (each pixel is independent), so the work-group shape only affects scheduling and memory behaviour, not correctness.

### 5.3 Padding in the lab

In OpenCL 1.2, when you specify a local size, every global dimension must be an exact multiple of the matching local dimension. Images are rarely that convenient, so the host rounds up and the kernel guards:

```c
// kernel
if (x >= (int)width || y >= (int)height) return;
```

Worked example (local size 16 x 16 = 256 work-items per group):

| Image | Global size after padding | Work-groups | Launched work-items | Real pixels | Padded (idle) |
|---|---|---|---|---|---|
| 256 x 256 (lab level 1) | 256 x 256 | 16 x 16 = 256 | 65,536 | 65,536 | 0 |
| 768 x 768 (level 3) | 768 x 768 | 48 x 48 = 2,304 | 589,824 | 589,824 | 0 |
| 1280 x 1280 (level 5) | 1280 x 1280 | 80 x 80 = 6,400 | 1,638,400 | 1,638,400 | 0 |
| 1000 x 700 (custom `--input`) | 1008 x 704 | 63 x 44 = 2,772 | 709,632 | 700,000 | 9,632 |

All five preset image sizes are multiples of 16, so with the default 16 x 16 local size no padding occurs. Padding appears with custom `--input` images, or when `--local-x` / `--local-y` do not divide the image dimensions (for example, a 768-wide image with `--local-x 32` still divides evenly, but a 1000-wide image does not).

> **OpenCL 2.0+** adds *non-uniform work-groups*: the last group in each dimension may be smaller, so padding is no longer needed. The guard `if (x >= width ...)` is still a good defensive habit.

### 5.4 What you can and cannot choose for the local size

- The product `local_x * local_y` must not exceed the device's `CL_DEVICE_MAX_WORK_GROUP_SIZE` (commonly 256 to 1024), and each dimension must respect `CL_DEVICE_MAX_WORK_ITEM_SIZES`.
- The kernel itself may have a lower limit (`CL_KERNEL_WORK_GROUP_SIZE`) if it uses many registers.
- Passing `NULL` as the local size lets the driver choose. The lab always passes one explicitly so that students can experiment.
- Violating a limit produces `CL_INVALID_WORK_GROUP_SIZE` at enqueue time.

### 5.5 Hardware execution detail: SIMD groups

Hardware does not run 256 independent instruction streams. It groups work-items into fixed bundles that execute in lock-step: **32 on NVIDIA ("warp")**, **32 or 64 on AMD ("wavefront")**, 8 to 32 on Intel. OpenCL 2.1+ calls this a **sub-group**. Consequences:

- If work-items in one bundle take different branches (*divergence*), the hardware runs both paths with some lanes masked off.
- Memory accesses are most efficient when the lanes of a bundle touch **adjacent addresses** (*coalescing*).
- Local size should usually be a multiple of the bundle size.

In the lab kernel, adjacent `x` values read adjacent pixels, so the inner access pattern is friendly. The loop bounds are identical for every work-item, so there is essentially no divergence, other than at the image border where `clamp` changes the sampled index but not the control flow.

---

## 6. Memory model

OpenCL defines four address spaces:

| Address space | Qualifier | Scope | Speed (typical) | Lab usage |
|---|---|---|---|---|
| **Global** | `__global` | All work-items, and the host through buffers | Slowest, largest (VRAM), cached on most GPUs | `input_rgba`, `output_rgba` |
| **Constant** | `__constant` | All work-items, read-only | Fast when all lanes read the same address | Not used (scalars are passed as arguments) |
| **Local** | `__local` | One work-group | Fast on-chip scratchpad (tens of KB) | Not used (suggested as an optimisation exercise) |
| **Private** | none / `__private` | One work-item | Registers (fastest) | `x`, `y`, `weighted_r`, ... |

```
          +--------------------------------------------+
Host  <-> |  Global / Constant memory  (device DRAM)   |
          +--------------------+-----------------------+
                               |  (cache)
                +--------------+---------------+
                |  Compute Unit                |
                |  +------------------------+  |
                |  | Local memory (per WG)  |  |
                |  +------------------------+  |
                |  Private regs: WI0 WI1 ...   |
                +------------------------------+
```

Key rules:

- **The host never touches device memory directly.** It copies with `clEnqueueWriteBuffer` / `clEnqueueReadBuffer` (or maps with `clEnqueueMapBuffer`). On discrete GPUs these copies cross PCIe, which is why the lab reports upload and download separately.
- **Memory is only guaranteed consistent at specific points.** Within a work-group, `barrier(CLK_LOCAL_MEM_FENCE)` or `barrier(CLK_GLOBAL_MEM_FENCE)` makes writes visible. Across work-groups there is no ordering within a single kernel launch, except through atomics.
- **Buffer flags** describe intended access: `MEM_READ_ONLY`, `MEM_WRITE_ONLY`, `MEM_READ_WRITE`. They are hints to the runtime and are checked against kernel argument qualifiers (`const` / non-`const`).

### Where the lab could improve memory use

Neighbouring output pixels reread almost the same input pixels. A **tiled local-memory** version would:

1. have each work-group cooperatively load a `(16 + 2r) x (16 + 2r)` tile (including the halo) into `__local` memory;
2. call `barrier(CLK_LOCAL_MEM_FENCE)`;
3. compute from the tile instead of from `__global` memory.

That change is the natural next step after the lab, because it exercises every OpenCL memory concept.

---

## 7. How kernels are compiled

OpenCL kernels are normally **compiled at run time**, by the driver, for the device that is actually present.

```
OpenCL C source text
      |  clCreateProgramWithSource
      v
 cl_program (unbuilt)
      |  clBuildProgram(options)         <- driver's compiler runs here
      v
 Device-specific binary (hidden inside the driver)
      |  clCreateKernel(name)
      v
 cl_kernel  ->  clEnqueueNDRangeKernel
```

Implications:

- **Source is portable; binaries are not.** One `.cl` file can run on any conformant device, but a binary produced by `clGetProgramInfo(CL_PROGRAM_BINARIES)` is tied to one vendor, architecture and often one driver version. This is why the lab only saves `bilateral_filter.opencl.bin` if the binding actually exposes the query, and never fabricates one.
- **Compile errors appear at run time.** Read the build log (`clGetProgramBuildInfo(..., CL_PROGRAM_BUILD_LOG)`); the lab writes it to `opencl-build.log`.
- **Build options** (`-cl-fast-relaxed-math`, `-cl-mad-enable`, `-D NAME=value`) change the generated code. `-D` is a convenient way to bake constants such as the radius into the kernel.
- **Compile time is real cost.** The lab measures it as `compileMs` and excludes it from the per-repetition kernel timing.
- Other routes exist: offline compilation, **SPIR-V** binaries (OpenCL 2.1+ / 3.0 optional), and `clCreateProgramWithBinary` to skip recompilation when you cache binaries yourself.

---

## 8. Asynchrony, queues and events

Every `clEnqueue*` call **returns before the work is done**. The command queue is a to-do list for the device.

- By default a queue is **in-order**: commands start in the order they were enqueued.
- A **blocking** flag (`CL_TRUE`, the `true` in `enqueueWriteBuffer(queue, buf, true, ...)`) makes the call wait until that transfer finishes.
- `enqueueNDRangeKernel` has no blocking flag. It returns immediately, so you must synchronise with `clFinish(queue)`, an event wait (`clWaitForEvents`), or a following blocking read.
- **Events** (`cl_event`) represent individual commands. They allow dependencies between commands, out-of-order queues, and *hardware* timing via `CL_QUEUE_PROFILING_ENABLE` plus `clGetEventProfilingInfo`.

The lab's kernel number is `enqueue + cl.finish()` measured with a host clock. That includes driver submission and wake-up latency, so it is slightly larger than the pure GPU execution time that event profiling would report. That limitation is stated explicitly in the lab's report.

If you timed only the `enqueueNDRangeKernel` call, you would measure how long it takes to *submit* the work, often microseconds, and the "GPU" would look implausibly fast.

---

## 9. Reading the lab kernel line by line

```c
__kernel void bilateral_filter(
    __global const uchar *input_rgba,    // read-only pixels in device memory
    __global uchar *output_rgba,         // output pixels in device memory
    const uint width, const uint height, // true image size (not padded)
    const uint radius,                   // filter radius r
    const float sigma_space_squared,     // sigma_s^2
    const float sigma_color_squared)     // sigma_c^2
```

- `__kernel`: callable from the host through `clEnqueueNDRangeKernel`.
- `__global`: pointer into global memory. Pointer address spaces are part of the type in OpenCL C.
- Scalar arguments are passed **by value** at `setKernelArg` time.

```c
const int x = (int)get_global_id(0);
const int y = (int)get_global_id(1);
if (x >= (int)width || y >= (int)height) return;
```

One work-item = one output pixel. The early return discards padded work-items.

```c
for (int dy = -(int)radius; dy <= (int)radius; ++dy) {
    const int sample_y = clamp(y + dy, 0, (int)height - 1);
    for (int dx ...) {
        const int sample_x = clamp(x + dx, 0, (int)width - 1);
```

`clamp` (built in) implements edge replication at the borders. The loop runs `(2r+1)^2` times per work-item, which is exactly the `S = W * H * (2r+1)^2` model from the teaching guide.

```c
const float weight = exp(exponent);
weighted_r += sample_r * weight; ...
```

`exp` is the standard OpenCL built-in. With `-cl-fast-relaxed-math` or `native_exp`, speed would improve at the cost of the CPU/GPU agreement checked in the lab's validation section.

```c
output_rgba[center_index + 0] = (uchar)clamp(weighted_r / total_weight, 0.0f, 255.0f);
```

The conversion truncates, deliberately matching JavaScript's `Uint8Array` behaviour in the CPU versions so results compare byte-for-byte as often as possible.

---

## 10. Common errors and what they mean

| Error / symptom | Typical cause | Fix |
|---|---|---|
| No platforms found / `CL_PLATFORM_NOT_FOUND_KHR` | No ICD or GPU driver with OpenCL installed | Install the vendor's driver or an OpenCL runtime (for example PoCL for CPU) |
| `CL_BUILD_PROGRAM_FAILURE` | Syntax or type error in the kernel | Read `opencl-build.log` |
| `CL_INVALID_WORK_GROUP_SIZE` | Local size too large, not dividing global size (1.2), or above the kernel's limit | Reduce `--local-x` / `--local-y`; query `CL_KERNEL_WORK_GROUP_SIZE` |
| `CL_INVALID_KERNEL_ARGS` | An argument was not set, or set with the wrong type/size | Check every `setKernelArg` index and type |
| `CL_MEM_OBJECT_ALLOCATION_FAILURE` / `CL_OUT_OF_RESOURCES` | Not enough device memory, or an out-of-bounds access inside the kernel | Smaller image; check index arithmetic and padded work-items |
| Kernel "runs" instantly, output is zeros | Reading before `clFinish`, or the kernel returned early for every item | Add synchronisation; verify the bounds check |
| Results differ slightly from CPU | Different floating-point rounding, fused multiply-add, `exp` implementation | Compare with a tolerance (the lab reports max error and PSNR) |

---

## 11. Mental checklist when writing OpenCL

1. **Can the problem be written as "one work-item per output element"?** If yes, start there.
2. **What is the global size, and does it need padding?** Add a bounds guard.
3. **Which data is reused by neighbouring work-items?** That data is a candidate for `__local` memory.
4. **Are adjacent work-items touching adjacent addresses?** If not, restructure.
5. **Is the transfer cost amortised?** Time upload, kernel, and download separately.
6. **Did you synchronise before reading results or stopping timers?**
7. **Did you validate against a reference implementation?**

---

## 12. Further reading

- Khronos OpenCL Registry and specifications: https://www.khronos.org/opencl/
- OpenCL 1.2 quick reference card (all built-ins on two pages)
- *Heterogeneous Computing with OpenCL* (Gaster, Howes, Kaeli, Mistry, Schaa)
- Vendor optimisation guides: NVIDIA OpenCL Best Practices, AMD/Intel OpenCL programming guides

*Companion files: `02-opencl-vs-cuda.md` and `03-opencl-cuda-keyword-map.md`.*

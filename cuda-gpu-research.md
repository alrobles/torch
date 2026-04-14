# Research: How the R `torch` Package Handles CUDA/GPU

> Generated 2026-04-14. Informs the design of a CUDA backend for `alrobles/rxbioclim`.

---

## 1. CUDA Installation & Detection

### Architecture Overview: The Lantern Abstraction Layer

The most important architectural insight is that torch for R uses a **two-library intermediary** called **Lantern** (`src/lantern/`). Rather than directly linking R/Rcpp code to libtorch, torch compiles a shared library (`lantern.so`/`lantern.dll`) that wraps libtorch behind a pure-C function pointer API. This isolation is the cornerstone of how all device dispatch, binary distribution, and versioning works.

```
R code (R7 objects)
  → Rcpp wrappers (src/*.cpp, compiled into torchpkg.so)
  → Lantern C API (function pointers, loaded at runtime via dlopen)
  → libtorch (CPU or CUDA build)
```

### Install-time Detection: Pure R, No configure.ac

There is **no `configure.ac`** file. Instead, the entire CUDA detection pipeline is implemented in **`R/install.R`** and is triggered at package load time (`.onLoad` in `R/package.R`).

The detection priority (lines 426–476 of `R/install.R`) is:

1. **`TORCH_CUDATOOLKIT` env var** — explicit selection of a companion `cuda<ver>` R package (e.g. `cuda12.8`)
2. **`CUDA` env var** — if set to `"cpu"`, forces CPU; if a version string, uses it
3. **Auto-detect installed `cuda*` R packages** (scans `supported_cuda_versions_linux/windows` newest-first)
4. **System CUDA probing** — reads `$CUDA_HOME/version.txt`, queries `nvcc --version`, checks `/usr/local/cuda/bin/nvcc`, and on Windows reads `$CUDA_PATH`

The key function is `cuda_version()` (`R/install.R` line 440), which chains these steps and returns a version string like `"12.8"`, `"cpu"`, or `NULL`. The result feeds `installation_kind()` (line 408) which produces a label like `"cu128"` or `"cpu"`.

**Supported CUDA versions** (as of 2.8.0):

```r
supported_cuda_versions_windows <- c("12.6", "12.8", "12.9")  # install.R:423
supported_cuda_versions_linux   <- c("12.6", "12.8", "12.9")  # install.R:424
```

macOS has **no CUDA support** (`cuda_version()` returns `NULL` for non-Windows/non-Linux, line 444–447).

### Pre-built Binaries: Download from pytorch.org CDN

Torch **ships pre-built CUDA binaries** — no compilation from source at install time. `install_torch()` (line 48) calls `libtorch_url()` (line 268) and `lantern_url()` (line 294) to construct download URLs.

For Linux with CUDA 12.8:
```
# libtorch
https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip
# lantern
https://torch-cdn.mlverse.org/binaries/<sha>/lantern-0.17.0+cu128+x86_64-Linux.zip
```

For CPU-only:
```
https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip
```

The CMake-level logic in `src/lantern/CMakeLists.txt` (lines 16–61) mirrors this exactly — if the `$CUDA` env var is set, it downloads the CUDA-enabled libtorch; otherwise it downloads the CPU build.

### Windows Specifics (`configure.win`)

`configure.win` (line 2) simply prepends Visual Studio generator flags and delegates to the main `configure` script:

```sh
CMAKE_FLAGS="-G\"Visual Studio 17\" $CMAKE_FLAGS" sh ./configure
```

The CMake build additionally handles Windows-specific NVTX header workarounds (lines 69–81 of `CMakeLists.txt`) for CUDA ≥ 12.

### `BUILD_LANTERN` Flag

The `configure` script (lines 13–22) controls whether lantern is built from source or skipped:
- If `BUILD_LANTERN=true`, CMake compiles lantern
- Otherwise, the `dummylantern` target is used (for CRAN installs, where pre-built binaries are assumed to already exist)

---

## 2. Runtime Device Dispatch

### `torch_device()` API

The R-level API is clean and simple (`R/device.R`):

```r
torch_device("cuda")        # current CUDA device
torch_device("cuda:1")      # GPU index 1
torch_device("cpu")         # CPU
torch_device("cuda", 0L)    # explicit index form
```

`Device` is an **R7 class** (a lightweight OOP system defined in `R/R7.R`, not R6) wrapping an external pointer. Its `initialize` parses the `"cuda:1"` colon notation (lines 10–17 of `device.R`), then calls `cpp_torch_device()` (Rcpp, `src/device.cpp` line 17), which calls `lantern_Device()`.

### Moving Tensors Between Devices

The `Tensor$to()` method (`R/tensor.R` line 72) is the primary API:

```r
tensor$to(device = torch_device("cuda"))  # CPU → GPU
tensor$to(device = torch_device("cpu"))   # GPU → CPU
tensor$cuda()                             # shortcut to GPU (line 116)
tensor$cpu()                              # shortcut to CPU (line 127)
```

Under the hood this calls `lantern_Tensor_to(self, options)` (lantern.h line 299), which invokes `torch::Tensor::to()` in libtorch. All actual memory transfer is handled by libtorch.

### Context Managers for Default Device

`local_device()` and `with_device()` (`R/device.R` lines 134–148) allow scoped default device setting using `withr::defer`. Internally, `cpp_set_default_device()` / `cpp_get_current_default_device()` maintain a C-level `torch::Device default_device` global (`src/device.cpp` lines 29–47).

### Error Handling When CUDA Is Not Available

`cuda_is_available()` (`R/cuda.R` line 4) delegates to `cpp_cuda_is_available()` → `lantern_cuda_is_available()` → `torch::cuda::is_available()`. CUDA-only functions (e.g., `cuda_memory_stats`) check availability first and throw a `runtime_error`. At the C level, CUDA-only code is wrapped in `#ifdef __NVCC__` blocks (`src/lantern/src/Cuda.cpp` lines 6–11, 48–56, 69–71, etc.) — without CUDA, those functions throw `std::runtime_error`.

---

## 3. Tensor System & Extensions

### R Object Model: R7 + External Pointers (XPtr)

Tensors are **R7 class** objects (`R/R7.R`) — a custom lightweight OOP system, **not R6**, **not S4**. R7 creates environments with active bindings and method dispatch. The actual tensor data is held in a **typed `XPtr` (external pointer)** (`XPtrTorchTensor`) managed in C++.

The bridge works as follows:
- Rcpp functions return `XPtrTorch*` types (e.g., `XPtrTorchTensor`), defined in `inst/include/`
- These are wrapped into R external pointers (`EXTPTRSXP`)
- R7 tensor objects hold `self$ptr` pointing to this external pointer

### R → Tensor Conversion

`cpp_tensor_from_buffer()` (`src/tensor.cpp` line 70) is the core conversion function:

```cpp
torch::Tensor cpp_tensor_from_buffer(SEXP data, vector<int64_t> shape, XPtrTorchTensorOptions options) {
  return lantern_from_blob(r_dataptr(data), &shape[0], shape.size(), nullptr, 0, options.get());
}
```

`r_dataptr()` (lines 51–67) maps R's SEXP types (`LGLSXP`, `INTSXP`, `REALSXP`, `RAWSXP`, etc.) to raw data pointers, then `torch::from_blob()` creates a tensor view over that memory.

Tensor → R conversion is done by `cpp_buffer_from_tensor()` (line 83), which uses `memcpy` from tensor storage to an R `RAW` vector.

### Memory Management for GPU Tensors

The custom CUDA allocator (`src/lantern/src/AllocatorCuda.cpp`) hooks into `c10::FreeMemoryCallback`:

```cpp
class GarbageCollectorCallback : virtual public c10::FreeMemoryCallback {
  bool Execute() {
    // Calls R GC when GPU memory pressure is high
    switch (should_call_gc()) {
      case CallGC::full: (*call_r_gc)(true); break;
      case CallGC::lite: (*call_r_gc)(false); break;
    }
  }
};
REGISTER_FREE_MEMORY_CALLBACK("garbage_collector_callback", GarbageCollectorCallback)
```

Thresholds are configurable via R options (`torch.cuda_allocator_reserved_rate`, `torch.cuda_allocator_allocated_rate`, `torch.cuda_allocator_allocated_reserved_rate`), set at startup in `package.R` lines 100–103. The callback triggers R's GC when GPU memory is running low, allowing tensor external pointers to be finalized and their GPU memory freed.

---

## 4. Build System Patterns

### CMake Build: Conditional CUDA Compilation

`src/lantern/CMakeLists.txt` controls everything:

- **`$CUDA` env var** determines whether CUDA is enabled (lines 16–23)
- When CUDA is enabled, `enable_language(CUDA)` is called (line 83) and `add_compile_definitions("CUDA<version>")` sets the `__NVCC__` analog used throughout the codebase
- The `.cu` file (`src/Contrib/SortVertices/sort_vert_kernel.cu`) is conditionally included in the source list only when CUDA is enabled (lines 141–148)
- CPU fallback uses `sort_vert_cpu.cpp` instead (lines 163–166)
- `set_source_files_properties(src/Cuda.cpp PROPERTIES COMPILE_DEFINITIONS __NVCC__)` (line 150) enables CUDA-specific `#ifdef __NVCC__` blocks in `Cuda.cpp`

The only `.cu` CUDA kernel file in the project is `sort_vert_kernel.cu` — a vertex sorting helper for bounding box IoU computation. All other GPU operations go through libtorch's existing CUDA implementation.

### No Custom `nvcc` Integration for Core Operations

There are **no custom CUDA kernels for core tensor ops**. The approach is:
- Delegate all GPU math to libtorch's existing CUDA ops
- Write custom CUDA kernels only for contrib operations not in libtorch (like `sort_vert_kernel.cu`)

### Library Renaming Trick

`tools/renamelib.R` (and `tools/renameinit.R`) rename the compiled shared library from `torch.so` → `torchpkg.so` and patch `RcppExports.R` accordingly. This avoids a namespace collision between the R package name (`torch`) and the native shared library name, which would otherwise conflict on macOS.

### Lantern Dynamic Loading

Lantern is loaded at runtime (not link time) via `dlopen` (Unix) or `LoadLibrary` (Windows). The `lantern.h` header defines function pointers (`LANTERN_PTR *`) for each API function. `cpp_lantern_init(path)` (`src/lantern.cpp` line 60) calls `lanternInit()`, which calls `dlopen` on the lantern shared library directory.

This means the R package binary itself has **no compile-time dependency on CUDA** — CUDA availability is determined entirely at runtime by which shared libraries are present in the installation directory.

---

## 5. Cross-Platform Considerations

### Windows

- Uses Visual Studio 17 generator (`configure.win`)
- CUDA detection reads `$CUDA_PATH` env var and queries `nvcc.exe` (lines 520–554 of `install.R`)
- Windows DLLs loaded by prepending lib path to `PATH` (lines 16–17 of `lantern_load.R`)
- CMake adds `-allow-unsupported-compiler` for nvcc (line 66 of `CMakeLists.txt`)
- NVTX headers workaround for CUDA ≥ 12 (lines 69–81 of `CMakeLists.txt`)

### Linux

- CUDA detection reads `$CUDA_HOME`, checks `/usr/local/cuda/version.txt`, queries `nvcc`
- Shared objects loaded as real files; symlinks excluded to avoid double-loading (lines 21–22 of `lantern_load.R`)
- RPATH set to `$ORIGIN` (CMakeLists.txt line 183) so libraries find each other without `LD_LIBRARY_PATH`

### macOS

- **No CUDA support** (macOS returns `NULL` from `cuda_version()`)
- Uses Apple Silicon (arm64) MPS backend via `src/lantern/src/AllocatorMPS.cpp` (conditionally compiled, line 137)
- Downloads macOS-specific libtorch from `github.com/mlverse/libtorch-mac-m1`
- RPATH set to `@loader_path` (CMakeLists.txt line 178)

### Environment Variables Summary

| Variable | Effect |
|---|---|
| `CUDA` | Force CUDA version (e.g. `"12.8"`) or `"cpu"` |
| `TORCH_CUDATOOLKIT` | Select companion `cuda*` R package, or `"FALSE"` to disable |
| `CUDA_HOME` (Linux) | Override CUDA toolkit path |
| `CUDA_PATH` (Windows) | Override CUDA toolkit path |
| `TORCH_HOME` | Override installation directory |
| `TORCH_URL` | Override libtorch download URL |
| `LANTERN_URL` | Override lantern download URL |
| `LANTERN_BASE_URL` | Override CDN base URL for lantern |
| `TORCH_COMMIT_SHA` | Force specific git SHA for lantern download |
| `TORCH_INSTALL` | `0`=disable auto-install, `1`=force install |
| `TORCH_LOAD` | `0`=disable auto-load |
| `TORCH_INSTALL_DEBUG` | `1`=verbose install messages |
| `TORCH_VERIFY_LOAD` | `FALSE`=skip load verification subprocess |
| `TORCH_LOG` | `1`=enable C-level logging |
| `BUILD_LANTERN` | `true`=build lantern from source in configure |

---

## 6. Patterns Applicable to rxbioclim

### Should rxbioclim use libtorch vs raw CUDA kernels?

**Use libtorch as a dependency via the `torch` R package.** The torch R package provides a complete, battle-tested GPU stack. Writing raw CUDA kernels requires a `.cu` compilation pipeline, nvcc integration, cross-platform build complexity, and custom memory management — all of which torch has already solved.

**Concrete recommendation**: rxbioclim should list `torch` as a dependency and use its tensor API directly:

```r
# Raster/bioclim data as torch tensor on GPU
x <- torch_tensor(bioclim_matrix, device = torch_device("cuda"))
result <- x$some_operation()
result_r <- as.matrix(result$cpu())
```

This pattern gives GPU acceleration with zero CUDA kernel code.

### Simplest Path to GPU-accelerated Array Operations

1. **Depend on `torch`** — it handles all CUDA detection, binary install, and device dispatch
2. **Convert input matrices/arrays** with `torch_tensor(x, device = torch_device("cuda"))`
3. **Use torch ops** (matmul, element-wise math, reductions) — CUDA-accelerated automatically
4. **Convert back** with `as.array(tensor$cpu())`

### Key Architectural Insight for rxbioclim

The Lantern pattern (thin C wrapper around a GPU library, loaded at runtime) is the right approach if rxbioclim ever needs its *own* custom CUDA kernels. The pattern would be:

1. Write a small shared library wrapping your CUDA kernels (the "lantern" analog)
2. Build it separately with nvcc enabled
3. Load it at runtime via `dyn.load` conditioned on GPU availability

But **the simpler path is to avoid this entirely** by building on top of torch's existing tensor operations, which cover virtually all array computation patterns needed for bioclimate modeling (broadcasting, reductions, matrix operations, interpolation approximations via differentiable ops).

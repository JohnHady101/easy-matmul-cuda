# Matrix Multiplication in CUDA

Naive dense square matrix multiplication on the GPU: `C = A * B` for `N x N` `int` matrices, with **one CUDA thread computing one output element**.

Single source file: `matmul.cu` (kernel + `main` host driver).

## What the code does

1. Sets `N = 1 << 10` (1024), so all matrices are `1024 x 1024`.
2. Allocates `A`, `B`, `C` with `cudaMallocManaged()` (unified memory, accessible from host and device).
3. Fills `A` and `B` with random ints in `[0, 100)` via `init_matrix()`.
4. Launches the `matmul` kernel with a 2D grid of 2D thread blocks.
5. Each thread computes one dot product (one row of `A` × one column of `B`) and writes one element of `C`.
6. `cudaDeviceSynchronize()` waits for the kernel to finish.

Math per output element:

```text
C[row][col] = sum(i=0..N-1) A[row][i] * B[i][col]
```

Matrices are stored flat in **row-major** order, so element `[r][c]` lives at index `r * N + c`:

```cpp
tmp += a[row * N + i] * b[i * N + col]; // i sweeps the row of A and column of B
c[row * N + col] = tmp;
```

## Build and run

You need an NVIDIA GPU + CUDA toolkit (`nvcc` on `PATH`):

```bash
nvcc matmul.cu -o matmul
./matmul
```

No output is printed by the current `main()` — it just runs the kernel. Add a checksum/print loop over `c` if you want to verify results.

## CUDA execution model in 30 seconds

CUDA launches a kernel as a **Grid** of **Blocks**, each Block containing many **Threads**:

```text
Grid (e.g. 64 x 64 blocks)
┌────────┬────────┬───┐
│ Block  │ Block  │ … │
│(0,0)   │(1,0)   │   │
├────────┼────────┼───┤
│ Block  │ Block  │ … │
│(0,1)   │(1,1)   │   │
└────────┴────────┴───┘

Each Block (e.g. 16 x 16 threads)
┌──┬──┬──┬──┐
│t │t │t │… │  ← threadIdx.x runs along columns (x)
├──┼──┼──┼──┤
│t │t │t │… │
├──┼──┼──┼──┤
│… │… │… │… │  ← threadIdx.y runs along rows (y)
└──┴──┴──┴──┘
```

Built-in variables available in every thread:

| Variable | Meaning |
|---|---|
| `threadIdx.{x,y,z}` | Position of the thread *inside its block*, `0 .. blockDim-1` |
| `blockDim.{x,y,z}` | Size of a block (threads per block per axis) |
| `blockIdx.{x,y,z}` | Position of the block *inside the grid*, `0 .. gridDim-1` |
| `gridDim.{x,y,z}` | Size of the grid (blocks per grid per axis) |

Global identity of a thread = `blockIdx * blockDim + threadIdx` (per axis).

## Block and thread indexing in this repo

Host setup (`matmul.cu:53-59`):

```cpp
int threads = 16;
int blocks = (N + threads - 1) / threads; // ceil(N / 16) = 64 for N=1024

dim3 THREADS(threads, threads); // 16 x 16 = 256 threads per block
dim3 BLOCKS(blocks, blocks);    // 64 x 64 = 4096 blocks in the grid

matmul<<<BLOCKS, THREADS>>>(a, b, c, N);
```

So:

- Threads per block: `16 × 16 = 256`
- Blocks in grid: `64 × 64 = 4096`
- Total threads launched: `4096 × 256 = 1,048,576 = 1024 × 1024 = N²` — exactly one per output element.

Kernel mapping (`matmul.cu:10-13`):

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

- **x-axis → columns, y-axis → rows.** This matches the 2D matrix layout: moving in `+x` moves right across columns, moving in `+y` moves down across rows.
- `blockDim.x = blockDim.y = 16`, so each block owns a `16 × 16` tile of `C`.

### Worked example

Take thread `threadIdx = (5, 3)` inside block `blockIdx = (2, 1)`, with `blockDim = (16, 16)`:

```text
row = 1 * 16 + 3 = 19
col = 2 * 16 + 5 = 37
```

That thread computes `C[19][37]`. Its neighbor `threadIdx = (6, 3)` in the same block computes `C[19][38]` — adjacent threads in `x` write adjacent columns, which gives coalesced writes to `c[row * N + col]`.

Visually, the output matrix `C` (1024×1024) is tiled like this:

```text
C (N x N)
◄───────────── col = blockIdx.x * 16 + threadIdx.x ─────────────►
┌──────────┬──────────┬──────────┐
│ 16x16    │ 16x16    │  …       │ ◄ blockIdx = (0,0), (1,0), …
│ tile     │ tile     │          │   each tile = 1 block
├──────────┼──────────┼──────────┤
│ 16x16    │ 16x16    │  …       │ ◄ blockIdx = (0,1), (1,1), …
│ tile     │ tile     │          │
├──────────┼──────────┼──────────┤
│  …       │  …       │  …       │
└──────────┴──────────┴──────────┘
▲
└─ row = blockIdx.y * 16 + threadIdx.y
```

### Boundary guard

```cpp
if (row < N && col < N) { ... }
```

Needed when `N` is not divisible by the block size. Example: `N = 1000`, `threads = 16` → `blocks = ceil(1000/16) = 63` → grid covers `63*16 = 1008` rows/cols, so the extra 8 rows/cols of threads must exit early instead of reading/writing out of bounds. For `N = 1024` it never triggers (`1024 = 64*16` exactly), but keep it for generality.

## Limitations / natural next steps

- **Naive global-memory access:** each thread reloads its full row of `A` and column of `B` (`2N` reads per thread, `O(N³)` total). A tiled shared-memory kernel reuses each tile `TILE` times and is much faster.
- `B` is read column-wise (`b[i * N + col]` with stride `N`), which does not coalesce well — another reason tiling helps.
- No error checking (`cudaGetLastError`), no result verification, no timing, and `C` is never freed/printed. Good first extensions: add `cudaMemcpy`-free verification against a CPU loop for small `N`, and time the kernel with CUDA events.
- Uses `int` with values `< 100` and `N = 1024`: `100*100*1024 ≈ 1e7`, fits safely in 32-bit `int`.

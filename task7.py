import os

# Elementwise kernels don't use BLAS, but pin it so nothing adds hidden threads.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103052
# Optional overrides (for testing): python task7.py [max_cores] [n_elements]
MAX_CORES = int(sys.argv[1]) if len(sys.argv) > 1 else (os.cpu_count() or 1)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000_000
REPEATS = 5
BLOCK = 65_536            # Kernel B works in 512 KB blocks so temporaries stay in cache
EFFICIENCY_GAP = 0.15     # Kernel A must trail Kernel B by this much to be called "stalled"


def chunk_bounds(n, workers):
    base, extra = divmod(n, workers)
    bounds, start = [], 0
    for i in range(workers):
        end = start + base + (1 if i < extra else 0)
        bounds.append((start, end))
        start = end
    return bounds


def kernel_a(A, B, s, e):
    """Memory-bound: B[i] = A[i]  (pure streaming, almost no arithmetic)."""
    np.copyto(B[s:e], A[s:e])


def kernel_b(A, B, s, e):
    """Compute-bound: B[i] = sin(A[i]) * cos(A[i]) + tan(A[i])."""
    t1 = np.empty(BLOCK)
    t2 = np.empty(BLOCK)
    for bs in range(s, e, BLOCK):
        be = min(bs + BLOCK, e)
        n = be - bs
        a = A[bs:be]
        x, y = t1[:n], t2[:n]
        np.sin(a, out=x)
        np.cos(a, out=y)
        np.multiply(x, y, out=x)
        np.tan(a, out=y)
        np.add(x, y, out=B[bs:be])


def bench(kernel, A, B, workers):
    bounds = chunk_bounds(len(A), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def run():
            futures = [pool.submit(kernel, A, B, s, e) for s, e in bounds]
            for f in futures:
                f.result()
        run()  # warm-up
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            run()
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def verify(name, A, B):
    if name == "A":
        ok = np.array_equal(A, B)
    else:
        ok = True
        for s in (0, len(A) // 2, len(A) - 1000):
            a = A[s:s + 1000]
            ref = np.sin(a) * np.cos(a) + np.tan(a)
            ok = ok and np.allclose(B[s:s + 1000], ref)
    if not ok:
        raise RuntimeError(f"Kernel {name} produced wrong results")


def run_kernel(label, name, kernel, A, B, core_counts):
    rows = []
    for c in core_counts:
        B.fill(0.0)
        rows.append((c, bench(kernel, A, B, c)))
        verify(name, A, B)
    t1 = rows[0][1]
    print(label)
    header = f"{'Cores':>6} | {'Wall Time (ms)':>15} | {'Speedup':>8} | {'Efficiency':>10}"
    print(header)
    print("-" * len(header))
    out = []
    for c, t in rows:
        speedup = t1 / t
        eff = speedup / c
        out.append((c, t, speedup, eff))
        print(f"{c:>6} | {t:>15.2f} | {speedup:>7.2f}x | {eff * 100:>9.1f}%")
    print()
    return out


def main():
    core_counts = sorted({1, min(2, MAX_CORES), MAX_CORES})

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} | Python {platform.python_version()} | NumPy {np.__version__}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()} | Max cores tested: {MAX_CORES}")
    print(f"Array: {N:,} float64 ({N * 8 / 1e6:.0f} MB per array) | seed={STUDENT_ID}")
    print(f"Method: ThreadPoolExecutor, contiguous chunks | median of {REPEATS} runs after warm-up")
    print("Efficiency = Speedup / Cores, Speedup = T(1) / T(p)\n")

    A = np.random.default_rng(STUDENT_ID).random(N)
    B = np.zeros(N)

    ra = run_kernel("Kernel A (memory-bound): B[i] = A[i]", "A", kernel_a, A, B, core_counts)
    rb = run_kernel("Kernel B (compute-bound): B[i] = sin(A[i])*cos(A[i]) + tan(A[i])", "B", kernel_b, A, B, core_counts)

    c_max, _, sp_a, eff_a = ra[-1]
    _, _, sp_b, eff_b = rb[-1]
    print("=" * 72)
    print(f"At {c_max} cores: Kernel A speedup {sp_a:.2f}x (efficiency {eff_a * 100:.1f}%) | "
          f"Kernel B speedup {sp_b:.2f}x (efficiency {eff_b * 100:.1f}%)")
    if c_max == 1:
        print("RESULT: only 1 core available, so no scaling comparison is possible.")
    elif eff_a + EFFICIENCY_GAP < eff_b:
        print("RESULT: Kernel A (vector copy, memory-bound) STALLED in scaling due to memory channel "
              "saturation: extra cores had no more bandwidth to use.")
        print("        Kernel B (sin/cos/tan, compute-bound) kept scaling because its time is spent "
              "in the CPU, not waiting on RAM.")
    else:
        print("RESULT: no clear memory-bandwidth stall on this run: Kernel A's efficiency is not "
              f"clearly below Kernel B's (gap < {EFFICIENCY_GAP * 100:.0f} percentage points).")


if __name__ == "__main__":
    main()

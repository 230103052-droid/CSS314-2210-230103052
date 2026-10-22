import os

# Keep BLAS/OpenMP backends from adding hidden parallelism; we control the workers.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103052
N = 20_000_000
P = 0.95          # parallelizable fraction for Amdahl's Law
REPEATS = 7       # timed runs per worker count (median reported)


def amdahl(p_workers, parallel_fraction=P):
    """Theoretical max speedup: S(p) = 1 / ((1 - P) + P / p)."""
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p_workers)


def chunk_bounds(n, workers):
    """Split [0, n) into `workers` contiguous, near-equal chunks."""
    base, extra = divmod(n, workers)
    bounds, start = [], 0
    for i in range(workers):
        end = start + base + (1 if i < extra else 0)
        bounds.append((start, end))
        start = end
    return bounds


def partial_sum(data, start, end):
    # NumPy releases the GIL inside np.sum, so threads run truly in parallel.
    return float(np.sum(data[start:end]))


def parallel_sum(data, pool, workers):
    futures = [pool.submit(partial_sum, data, s, e) for s, e in chunk_bounds(len(data), workers)]
    return sum(f.result() for f in futures)


def benchmark(data, workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parallel_sum(data, pool, workers)  # warm-up (thread start-up, cache)
        times, result = [], 0.0
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            result = parallel_sum(data, pool, workers)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), result


def main():
    max_cores = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} | Python {platform.python_version()} | NumPy {np.__version__}")
    print(f"Logical cores (os.cpu_count): {max_cores}")
    print(f"Array size: {N:,} float64 (seed={STUDENT_ID}) | Amdahl P = {P}")
    print(f"Method: ThreadPoolExecutor, chunked np.sum | median of {REPEATS} runs after warm-up")
    print(f"Worker counts: {worker_counts}\n")

    data = np.random.default_rng(STUDENT_ID).random(N, dtype=np.float64)
    reference = float(np.sum(data))

    results = []
    for w in worker_counts:
        t_ms, total = benchmark(data, w)
        results.append((w, t_ms, total))

    t1 = results[0][1]  # T(1): 1-worker run
    header = f"{'Cores':>6} | {'Wall Time (ms)':>15} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}"
    print(header)
    print("-" * len(header))
    for w, t_ms, _ in results:
        print(f"{w:>6} | {t_ms:>15.2f} | {t1 / t_ms:>7.2f}x | {amdahl(w):>17.2f}x")

    print(f"\nAmdahl limit as p -> infinity: {1 / (1 - P):.2f}x")
    max_diff = max(abs(total - reference) for _, _, total in results)
    print(f"Serial reference sum: {reference:.6f} | max abs diff across runs: {max_diff:.3e}")


if __name__ == "__main__":
    main()

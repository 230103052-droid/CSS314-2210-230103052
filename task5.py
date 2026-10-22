import os

# Elementwise ops don't use BLAS, but pin it anyway so nothing adds hidden threads.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import math
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103052
A, B = 2.5, 1.0                      # y[i] = A * x[i] + B
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
TRIALS = 7                           # timed trials per measurement (median reported)
MIN_TRIAL_S = 0.02                   # each trial runs enough calls to last >= 20 ms
REFINE_POINTS = 15                   # extra sizes tested inside the crossover bracket
# Optional override: python task5.py 8
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else (os.cpu_count() or 1)


def serial_op(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def chunk_op(x, y, start, end):
    ys = y[start:end]
    np.multiply(x[start:end], A, out=ys)   # NumPy releases the GIL here
    np.add(ys, B, out=ys)


def make_parallel_op(pool):
    def parallel_op(x, y):
        n = len(x)
        w = min(WORKERS, n)
        base, extra = divmod(n, w)
        futures, start = [], 0
        for i in range(w):
            end = start + base + (1 if i < extra else 0)
            futures.append(pool.submit(chunk_op, x, y, start, end))
            start = end
        for f in futures:
            f.result()
    return parallel_op


def time_per_call_us(fn, x, y):
    """Median time of one call in microseconds, over TRIALS trials of many calls each."""
    fn(x, y)  # warm-up
    t0 = time.perf_counter()
    fn(x, y)
    one = max(time.perf_counter() - t0, 1e-7)
    loops = min(200_000, max(1, math.ceil(MIN_TRIAL_S / one)))
    trials = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        for _ in range(loops):
            fn(x, y)
        trials.append((time.perf_counter() - t0) / loops * 1e6)
    return statistics.median(trials)


def measure(n, rng, parallel_op):
    x = rng.random(n)
    y_ser = np.empty(n)
    y_par = np.empty(n)
    t_ser = time_per_call_us(serial_op, x, y_ser)
    t_par = time_per_call_us(parallel_op, x, y_par)
    if not np.allclose(y_ser, y_par):
        raise RuntimeError(f"Parallel result differs from serial at N={n}")
    return n, t_ser, t_par


def first_stable_faster(results):
    """Index of the first size from which parallel stays faster for all larger sizes."""
    faster = [tp < ts for _, ts, tp in results]
    for i in range(len(results)):
        if all(faster[i:]):
            return i
    return None


def print_row(n, ts, tp):
    winner = "parallel" if tp < ts else "serial"
    print(f"{n:>12,} | {ts:>14.2f} | {tp:>14.2f} | {ts / tp:>7.2f}x | {winner}")


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} | Python {platform.python_version()} | NumPy {np.__version__}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()} | Threads used: {WORKERS}")
    print(f"Operation: y[i] = {A} * x[i] + {B} (float64) | method: ThreadPoolExecutor, chunked NumPy ops")
    print(f"Timing: median of {TRIALS} trials, each >= {MIN_TRIAL_S * 1000:.0f} ms of repeated calls (pool created once, dispatch/join timed)\n")

    rng = np.random.default_rng(STUDENT_ID)
    header = f"{'N':>12} | {'Serial (us)':>14} | {'Parallel (us)':>14} | {'Speedup':>8} | Winner"

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        parallel_op = make_parallel_op(pool)

        print("Coarse sweep")
        print(header)
        print("-" * len(header))
        coarse = []
        for n in SIZES:
            r = measure(n, rng, parallel_op)
            coarse.append(r)
            print_row(*r)

        c = first_stable_faster(coarse)
        print()
        if c is None:
            n, ts, tp = coarse[-1]
            print("RESULT: no crossover found. Multi-threading was not consistently faster than serial "
                  f"at any tested size (N={n:,}: speedup {ts / tp:.2f}x).")
            return
        if c == 0:
            print(f"RESULT: multi-threading was faster than serial at every tested size, "
                  f"so the crossover is below N={SIZES[0]:,}.")
            return

        lo, hi = coarse[c - 1][0], coarse[c][0]
        print(f"Coarse bracket: serial wins at N={lo:,}, parallel wins from N={hi:,}. Refining...\n")

        grid = np.geomspace(lo, hi, REFINE_POINTS + 2)
        fine_sizes = sorted({int(round(v)) for v in grid})
        print("Fine sweep inside bracket")
        print(header)
        print("-" * len(header))
        fine = []
        for n in fine_sizes:
            r = measure(n, rng, parallel_op)
            fine.append(r)
            print_row(*r)

        f = first_stable_faster(fine)
        print()
        if f is None or f == 0:
            print(f"RESULT: crossover lies between N={lo:,} (serial faster) and N={hi:,} (parallel faster); "
                  "fine sweep was too noisy to narrow it further.")
        else:
            n_slow, n_fast = fine[f - 1][0], fine[f][0]
            est = int(round(math.sqrt(n_slow * n_fast)))
            print(f"RESULT: multi-threading shifts from SLOWER to FASTER than serial between "
                  f"N={n_slow:,} and N={n_fast:,} (about N ~ {est:,}) on {WORKERS} threads.")


if __name__ == "__main__":
    main()

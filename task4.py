import platform
import os
import queue
import statistics
import sys
import threading
import time

STUDENT_ID = 230103052
N_ITEMS = 1000
WORKERS = 4
REPEATS = 5
# Cost unit in seconds (spec: 0.0001). Optional override: python task4.py 0.001
UNIT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0001


# ---- workload models: cost (seconds) of item i ----
def cost_mod20(i):
    return UNIT * (i % 20)          # as specified


def cost_ramp(i):
    return UNIT * (i // 50)         # extra scenario: cost grows with the index


SCENARIOS = [
    ("Scenario 1 (as specified): cost = UNIT * (i % 20)", cost_mod20),
    ("Scenario 2 (extra): ramp cost = UNIT * (i // 50)", cost_ramp),
]


def chunk_bounds(n, workers):
    base, extra = divmod(n, workers)
    bounds, start = [], 0
    for i in range(workers):
        end = start + base + (1 if i < extra else 0)
        bounds.append((start, end))
        start = end
    return bounds


# ---- strategies: each returns a worker function worker(w, finish) ----
def make_static(cost):
    blocks = chunk_bounds(N_ITEMS, WORKERS)

    def worker(w, finish):
        s, e = blocks[w]
        for i in range(s, e):
            time.sleep(cost(i))
        finish[w] = time.perf_counter()
    return worker


def make_dynamic(cost, chunk):
    q = queue.Queue()
    for s in range(0, N_ITEMS, chunk):
        q.put((s, min(s + chunk, N_ITEMS)))

    def worker(w, finish):
        while True:
            try:
                s, e = q.get_nowait()
            except queue.Empty:
                break
            for i in range(s, e):
                time.sleep(cost(i))
        finish[w] = time.perf_counter()
    return worker


def run_once(worker_fn):
    finish = [0.0] * WORKERS
    barrier = threading.Barrier(WORKERS + 1)

    def target(w):
        barrier.wait()
        worker_fn(w, finish)

    threads = [threading.Thread(target=target, args=(w,)) for w in range(WORKERS)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    total = (time.perf_counter() - t0) * 1000
    rel = [(f - t0) * 1000 for f in finish]
    return total, min(rel), max(rel), max(rel) - min(rel)


def benchmark(make_worker):
    runs = [run_once(make_worker()) for _ in range(REPEATS)]
    return tuple(statistics.median(col) for col in zip(*runs))


def sleep_resolution():
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        time.sleep(UNIT)
    return (time.perf_counter() - t0) / n * 1000


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} | Python {platform.python_version()}")
    print(f"Logical cores: {os.cpu_count()} | Workers: {WORKERS} | Items: {N_ITEMS}")
    print(f"Cost unit: {UNIT} s | Repeats: {REPEATS} (median of each column)")
    print(f"Measured time.sleep({UNIT}) = {sleep_resolution():.3f} ms per call (requested {UNIT * 1000:.3f} ms)\n")

    header = (f"{'Strategy':<28} | {'Total (ms)':>10} | {'First done':>10} | "
              f"{'Last done':>10} | {'Delta (ms)':>10}")

    for title, cost in SCENARIOS:
        costs = [cost(i) for i in range(N_ITEMS)]
        total_work = sum(costs) * 1000
        block_sums = [sum(costs[s:e]) * 1000 for s, e in chunk_bounds(N_ITEMS, WORKERS)]
        print(title)
        print(f"Model: total work = {total_work:.1f} ms | ideal (work/{WORKERS}) = {total_work / WORKERS:.1f} ms | "
              f"static block loads = {[round(b, 1) for b in block_sums]} ms")
        print(header)
        print("-" * len(header))

        strategies = [
            ("A: Static blocks", lambda: make_static(cost)),
            ("B: Dynamic, chunk = 1", lambda: make_dynamic(cost, 1)),
            ("B: Dynamic, chunk = 5", lambda: make_dynamic(cost, 5)),
        ]
        results = {}
        for name, factory in strategies:
            total, first, last, delta = benchmark(factory)
            results[name] = total
            print(f"{name:<28} | {total:>10.2f} | {first:>10.2f} | {last:>10.2f} | {delta:>10.2f}")

        base = results["A: Static blocks"]
        best = min(results["B: Dynamic, chunk = 1"], results["B: Dynamic, chunk = 5"])
        print(f"Best dynamic vs static: {base / best:.2f}x speedup\n")


if __name__ == "__main__":
    main()

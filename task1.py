import os

# Keep any BLAS/OpenMP backend single-threaded so the baseline is strictly serial.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import platform
import subprocess
import time

import numpy as np

STUDENT_ID = 230103052
N = 20_000_000


def cpu_model():
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        elif system == "Windows":
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "name"], text=True
            ).splitlines()
            names = [l.strip() for l in out if l.strip() and l.strip() != "Name"]
            if names:
                return names[0]
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"CPU: {cpu_model()}")
    print(f"Platform: {platform.platform()} | {platform.machine()}")
    print(f"Logical cores (os.cpu_count): {os.cpu_count()}")
    print(f"Python: {platform.python_version()} | NumPy: {np.__version__}")
    print(f"Array size: {N:,} float64 elements (seed={STUDENT_ID})")

    rng = np.random.default_rng(STUDENT_ID)
    t_gen = time.perf_counter()
    data = rng.random(N, dtype=np.float64)
    gen_ms = (time.perf_counter() - t_gen) * 1000
    print(f"Array generation time: {gen_ms:.2f} ms (not included in baseline)")

    # Serial computation: sum and standard deviation (population, ddof=0)
    t0 = time.perf_counter()
    total = float(np.sum(data))
    std = float(np.std(data))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"Sum: {total:.6f}")
    print(f"Standard deviation: {std:.10f}")
    print(f"Serial wall-clock time (sum + std): {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()

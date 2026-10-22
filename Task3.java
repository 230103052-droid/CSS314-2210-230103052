import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;

/**
 * Task 3: cache line contention (false sharing).
 *
 * Every thread i increments counts[i * stride] ITERATIONS times.
 *   - Unpadded: stride = 1  -> neighbouring counters share one 64-byte cache line
 *   - Padded:   stride = 16 -> 128 bytes between counters (own cache line, also safe
 *                              from the adjacent-line prefetcher)
 *
 * Loads/stores use VarHandle "opaque" access so the JIT cannot keep the counter in a
 * register and collapse the loop; both variants execute the same instructions and
 * differ only in memory layout.
 *
 * Usage:  java Task3.java [threads]     (default threads = physical core count)
 */
public class Task3 {
    static final long STUDENT_ID = 230103052L;
    static final long ITERATIONS = 50_000_000L;
    static final long WARMUP_ITERATIONS = 5_000_000L;
    static final int REPEATS = 5;
    static final int UNPADDED_STRIDE = 1;   // 8 bytes between counters
    static final int PADDED_STRIDE = 16;    // 128 bytes between counters

    static final VarHandle ELEM = MethodHandles.arrayElementVarHandle(long[].class);

    /** Runs one benchmark; returns wall-clock time in milliseconds. */
    static double run(int threads, int stride, long iterations) throws InterruptedException {
        final long[] counts = new long[threads * stride + 16];
        final CountDownLatch ready = new CountDownLatch(threads);
        final CountDownLatch go = new CountDownLatch(1);
        Thread[] workers = new Thread[threads];

        for (int i = 0; i < threads; i++) {
            final int idx = i * stride;
            workers[i] = new Thread(() -> {
                ready.countDown();
                try {
                    go.await();
                } catch (InterruptedException e) {
                    return;
                }
                for (long k = 0; k < iterations; k++) {
                    ELEM.setOpaque(counts, idx, (long) ELEM.getOpaque(counts, idx) + 1L);
                }
            });
            workers[i].start();
        }

        ready.await();
        long t0 = System.nanoTime();
        go.countDown();
        for (Thread t : workers) {
            t.join();
        }
        long t1 = System.nanoTime();

        for (int i = 0; i < threads; i++) {
            if (counts[i * stride] != iterations) {
                throw new IllegalStateException("Wrong count at thread " + i);
            }
        }
        return (t1 - t0) / 1_000_000.0;
    }

    static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }

    static double min(double[] v) {
        return Arrays.stream(v).min().orElse(Double.NaN);
    }

    // ---- physical core detection (falls back to logical / 2) ----
    static String exec(String... cmd) {
        try {
            Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            String out = new String(p.getInputStream().readAllBytes()).trim();
            p.waitFor();
            return out;
        } catch (Exception e) {
            return "";
        }
    }

    static int physicalCores(int logical) {
        String os = System.getProperty("os.name").toLowerCase();
        try {
            if (os.contains("win")) {
                String out = exec("powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum");
                int n = Integer.parseInt(out.trim());
                if (n > 0) return n;
            } else if (os.contains("mac")) {
                int n = Integer.parseInt(exec("sysctl", "-n", "hw.physicalcpu").trim());
                if (n > 0) return n;
            } else {
                String out = exec("lscpu", "-p=core,socket");
                long n = out.lines().filter(l -> !l.startsWith("#") && !l.isBlank()).distinct().count();
                if (n > 0) return (int) n;
            }
        } catch (Exception ignored) {
        }
        return Math.max(1, logical / 2);
    }

    public static void main(String[] args) throws Exception {
        int logical = Runtime.getRuntime().availableProcessors();
        int physical = physicalCores(logical);
        int threads = args.length > 0 ? Integer.parseInt(args[0]) : physical;

        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " (" + System.getProperty("os.arch") + ")");
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("Logical cores: " + logical + " | Physical cores: " + physical);
        System.out.println("Threads used: " + threads);
        System.out.printf("Increments per thread: %,d | Repeats: %d (after warm-up)%n", ITERATIONS, REPEATS);
        System.out.println("Unpadded stride: " + UNPADDED_STRIDE + " long (8 B) | Padded stride: "
                + PADDED_STRIDE + " longs (128 B)\n");

        // warm-up so the JIT compiles the loop before timing
        run(threads, UNPADDED_STRIDE, WARMUP_ITERATIONS);
        run(threads, PADDED_STRIDE, WARMUP_ITERATIONS);

        double[] unpadded = new double[REPEATS];
        double[] padded = new double[REPEATS];
        for (int r = 0; r < REPEATS; r++) {
            unpadded[r] = run(threads, UNPADDED_STRIDE, ITERATIONS);
            padded[r] = run(threads, PADDED_STRIDE, ITERATIONS);
            System.out.printf("Run %d: unpadded = %9.2f ms | padded = %9.2f ms%n", r + 1, unpadded[r], padded[r]);
        }

        double mu = median(unpadded), mp = median(padded);
        System.out.println();
        System.out.printf("%-32s | %12s | %12s%n", "Variant", "Median (ms)", "Min (ms)");
        System.out.println("-".repeat(62));
        System.out.printf("%-32s | %12.2f | %12.2f%n", "Unpadded shared (false sharing)", mu, min(unpadded));
        System.out.printf("%-32s | %12.2f | %12.2f%n", "Padded (128 B apart)", mp, min(padded));
        System.out.println();
        System.out.printf("Execution ratio (unpadded / padded): %.2fx slower with false sharing%n", mu / mp);
    }
}

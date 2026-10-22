import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Task 6: synchronization cost and data loss.
 *
 * 4 threads each increment a shared counter 2,000,000 times:
 *   1. Unsynchronized  - volatile int, counter++ (visible to all threads, but ++ is a
 *                        non-atomic read-modify-write -> lost updates). volatile also stops
 *                        the JIT from folding the loop into one addition, which would make
 *                        the timing comparison meaningless.
 *   2. AtomicInteger   - hardware atomic (lock-free CAS / fetch-add)
 *   3. ReentrantLock   - mutual exclusion lock
 *
 * Usage:  java Task6.java
 */
public class Task6 {
    static final long STUDENT_ID = 230103052L;
    static final int THREADS = 4;
    static final int INCREMENTS = 2_000_000;
    static final long EXPECTED = (long) THREADS * INCREMENTS;
    static final int REPEATS = 5;

    static volatile int unsafeCounter = 0;
    static final AtomicInteger atomicCounter = new AtomicInteger(0);
    static int lockedCounter = 0;
    static final ReentrantLock lock = new ReentrantLock();

    /** Runs `work` on THREADS threads started together; returns wall-clock ms. */
    static double timed(Runnable work) throws InterruptedException {
        CountDownLatch ready = new CountDownLatch(THREADS);
        CountDownLatch go = new CountDownLatch(1);
        Thread[] threads = new Thread[THREADS];
        for (int i = 0; i < THREADS; i++) {
            threads[i] = new Thread(() -> {
                ready.countDown();
                try {
                    go.await();
                } catch (InterruptedException e) {
                    return;
                }
                work.run();
            });
            threads[i].start();
        }
        ready.await();
        long t0 = System.nanoTime();
        go.countDown();
        for (Thread t : threads) {
            t.join();
        }
        return (System.nanoTime() - t0) / 1_000_000.0;
    }

    static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s[s.length / 2];
    }

    static double lossPercent(long actual) {
        return (EXPECTED - actual) * 100.0 / EXPECTED;
    }

    public static void main(String[] args) throws Exception {
        System.out.println("Student ID: " + STUDENT_ID);
        System.out.println("OS: " + System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " (" + System.getProperty("os.arch") + ")");
        System.out.println("Java: " + System.getProperty("java.version"));
        System.out.println("Logical cores: " + Runtime.getRuntime().availableProcessors());
        System.out.printf("Threads: %d | Increments per thread: %,d | Expected final value: %,d%n",
                THREADS, INCREMENTS, EXPECTED);
        System.out.println("Runs per variant: " + REPEATS + " (no separate warm-up; every run is shown)\n");

        double[] tUnsafe = new double[REPEATS];
        double[] tAtomic = new double[REPEATS];
        double[] tLock = new double[REPEATS];
        long[] vUnsafe = new long[REPEATS];
        long[] vAtomic = new long[REPEATS];
        long[] vLock = new long[REPEATS];

        for (int r = 0; r < REPEATS; r++) {
            unsafeCounter = 0;
            tUnsafe[r] = timed(() -> {
                for (int i = 0; i < INCREMENTS; i++) {
                    unsafeCounter++;
                }
            });
            vUnsafe[r] = unsafeCounter;

            atomicCounter.set(0);
            tAtomic[r] = timed(() -> {
                for (int i = 0; i < INCREMENTS; i++) {
                    atomicCounter.incrementAndGet();
                }
            });
            vAtomic[r] = atomicCounter.get();

            lockedCounter = 0;
            tLock[r] = timed(() -> {
                for (int i = 0; i < INCREMENTS; i++) {
                    lock.lock();
                    try {
                        lockedCounter++;
                    } finally {
                        lock.unlock();
                    }
                }
            });
            vLock[r] = lockedCounter;
        }

        System.out.println("Unsynchronized counter (volatile int, counter++), all runs:");
        System.out.printf("%-4s | %12s | %12s | %10s | %10s%n", "Run", "Expected", "Actual", "Lost", "Lost %");
        System.out.println("-".repeat(60));
        for (int r = 0; r < REPEATS; r++) {
            System.out.printf("%-4d | %,12d | %,12d | %,10d | %9.2f%%%n",
                    r + 1, EXPECTED, vUnsafe[r], EXPECTED - vUnsafe[r], lossPercent(vUnsafe[r]));
        }

        long worst = Arrays.stream(vUnsafe).min().getAsLong();
        double avgLoss = 0;
        for (long v : vUnsafe) avgLoss += lossPercent(v);
        avgLoss /= REPEATS;
        System.out.printf("%nAverage updates lost (unsynchronized): %.2f%% | worst run: %,d of %,d%n",
                avgLoss, worst, EXPECTED);

        boolean unsafeOk = Arrays.stream(vUnsafe).allMatch(v -> v == EXPECTED);
        boolean atomicOk = Arrays.stream(vAtomic).allMatch(v -> v == EXPECTED);
        boolean lockOk = Arrays.stream(vLock).allMatch(v -> v == EXPECTED);

        double mu = median(tUnsafe), ma = median(tAtomic), ml = median(tLock);
        System.out.println();
        System.out.printf("%-26s | %12s | %-18s%n", "Variant", "Median (ms)", "Correct in all runs");
        System.out.println("-".repeat(62));
        System.out.printf("%-26s | %12.2f | %-18s%n", "Unsynchronized (volatile)", mu, unsafeOk ? "yes (got lucky)" : "NO (data loss)");
        System.out.printf("%-26s | %12.2f | %-18s%n", "AtomicInteger", ma, atomicOk ? "yes" : "NO");
        System.out.printf("%-26s | %12.2f | %-18s%n", "ReentrantLock", ml, lockOk ? "yes" : "NO");

        System.out.println();
        System.out.printf("Synchronization penalty, AtomicInteger vs unsynchronized: %+.2f ms (%.2fx)%n", ma - mu, ma / mu);
        System.out.printf("Synchronization penalty, ReentrantLock vs unsynchronized: %+.2f ms (%.2fx)%n", ml - mu, ml / mu);
        System.out.printf("ReentrantLock vs AtomicInteger: %.2fx%n", ml / ma);
    }
}

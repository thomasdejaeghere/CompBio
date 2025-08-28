import subprocess
import time
import psutil
import statistics

# List of files to benchmark
files = [
    "data_size100_amount10.py",
    "data_size100_amount100.py",
    "data_size300_amount10.py",
    "data_size300_amount100.py",
    "data_size400_amount10.py",
    "data_size400_amount100.py",
]


def run_benchmark(filename, runs=10):
    times = []
    mems = []

    for _ in range(runs):
        start_time = time.perf_counter()
        process = psutil.Popen(
            ["python", filename], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        peak_mem = 0
        while process.is_running():
            try:
                mem = process.memory_info().rss / (1024 * 1024)  # MB
                peak_mem = max(peak_mem, mem)
            except psutil.NoSuchProcess:
                break
            time.sleep(0.01)

        process.wait()
        end_time = time.perf_counter()

        times.append(end_time - start_time)
        mems.append(peak_mem)

    return statistics.mean(times), statistics.mean(mems)


if __name__ == "__main__":
    results = []
    for f in files:
        avg_time, avg_mem = run_benchmark(f, runs=4)
        results.append((f, avg_time, avg_mem))

    print("\nBenchmark Results (averaged over 10 runs):")
    print(f"{'File':<35}{'Time (s)':<15}{'Memory (MB)':<15}")
    print("-" * 65)
    for f, t, m in results:
        print(f"{f:<35}{t:<15.4f}{m:<15.2f}")

import argparse
import glob
import os
import re
import statistics

DIFF_RE = re.compile(r"Diff:\s*(-?[\d.]+)\s*dB")


def load_diffs(path):
    diffs = []
    with open(path) as f:
        for line in f:
            m = DIFF_RE.search(line)
            if m:
                diffs.append(float(m.group(1)))
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Summarize NLOS diff logs collected from the anchor's serial output")
    parser.add_argument("logs", nargs="*", default=None, help="Log files to analyze (default: logs/*.log)")
    args = parser.parse_args()

    paths = args.logs or sorted(
        p for p in glob.glob("logs/*.log") if not os.path.basename(p).startswith("raw_")
    )
    if not paths:
        print("No log files found. Run from the project root, or pass file paths directly.")
        return

    print(f"{'Condition':<28}{'N':>6}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}")
    for path in paths:
        diffs = load_diffs(path)
        name = os.path.splitext(os.path.basename(path))[0]
        if not diffs:
            print(f"{name:<28}  (no [NLOS] lines found)")
            continue
        mean = statistics.mean(diffs)
        std = statistics.pstdev(diffs) if len(diffs) > 1 else 0.0
        print(f"{name:<28}{len(diffs):>6}{mean:>10.2f}{std:>10.2f}{min(diffs):>10.2f}{max(diffs):>10.2f}")


if __name__ == "__main__":
    main()

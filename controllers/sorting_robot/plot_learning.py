"""plot_learning.py — plot learning curves from learning_log_*.csv

Usage:  python plot_learning.py [controller_folder]
Needs:  pip install matplotlib
"""
import csv
import glob
import os
import sys
from collections import deque

import matplotlib.pyplot as plt


def smooth(values, window=10):
    d, out = deque(maxlen=window), []
    for v in values:
        d.append(v)
        out.append(sum(d) / len(d))
    return out


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 \
        else os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(folder, "learning_log_*.csv")))
    if not files:
        print("No learning_log_*.csv files found in", folder)
        return

    plt.figure(figsize=(9, 5))
    for path in files:
        episodes, returns = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                episodes.append(int(row["episode"]))
                returns.append(float(row["return"]))
        name = os.path.basename(path)[len("learning_log_"):-4]
        plt.plot(episodes, returns, alpha=0.25)
        plt.plot(episodes, smooth(returns), lw=2, label=name)

    plt.axhline(0, color="gray", lw=0.8)
    plt.xlabel("episode")
    plt.ylabel("episode return")
    plt.title("Sorting robots — learning curves")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(folder, "learning_curves.png")
    plt.savefig(out, dpi=150)
    print("Saved", out)
    plt.show()


if __name__ == "__main__":
    main()

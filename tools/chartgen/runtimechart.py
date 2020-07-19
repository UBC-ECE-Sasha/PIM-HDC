import argparse
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def split_by_tasklet(
    dpu_time: List[float], nr_tasklets: List[float], nr_dpus: List[float]
) -> Dict[int, Dict[int, float]]:
    times_by_dpu: Dict[int, Dict[int, float]] = {}

    for (dt, nt, nd) in zip(dpu_time, nr_tasklets, nr_dpus):
        dpu_key = int(nd)

        if dpu_key not in times_by_dpu:
            times_by_dpu[dpu_key] = {}

        times_by_dpu[dpu_key][int(nt)] = dt

    return times_by_dpu


def generate_plots(
    host_time: List[float],
    dpu_time: List[float],
    nr_tasklets: List[float],
    nr_dpus: List[float],
) -> Tuple[List[int], Dict[str, List[float]]]:

    average_host = 0
    nr_dpu_uniq = set()
    for ht, nd in zip(host_time, nr_dpus):
        average_host += ht
        nr_dpu_uniq.add(int(nd))

    average_host = average_host / len(host_time)

    times_by_dpu = split_by_tasklet(dpu_time, nr_tasklets, nr_dpus)

    plots: Dict[str, List[float]] = {}

    for k in times_by_dpu:
        for n in times_by_dpu[k]:
            label = f"{n} tasklets"
            if label not in plots:
                plots[label] = []
            plots[label].append(times_by_dpu[k][n])

    list_dpus = list(nr_dpu_uniq)
    list_dpus.sort()

    plots[f"Host (1 core)"] = [average_host for _ in range(0, len(list_dpus))]

    return list_dpus, plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a runtime chart for PIM-HDC")
    parser.add_argument(
        "--csv_file", help="Input CSV data file", required=True, type=str
    )
    parser.add_argument("--xlabel", help="Chart x-axis label", default="DPUs", type=str)
    parser.add_argument(
        "--ylabel", help="Chart y-axis label", default="Time (s)", type=str
    )
    parser.add_argument(
        "--title", help="Chart title", default="PIM-HDC Runtimes", type=str
    )
    parser.add_argument(
        "--output_file", help="Output chart", default="output.png", type=str
    )

    return parser.parse_args()


def main():
    config = parse_args()

    host_time, dpu_time, nr_tasklets, nr_dpus = np.loadtxt(
        config.csv_file, delimiter=",", unpack=True, skiprows=1
    )

    plots = generate_plots(host_time, dpu_time, nr_tasklets, nr_dpus)

    for p in plots[1]:
        plt.plot(plots[0], plots[1][p], label=p)

    plt.xlabel(config.xlabel)
    plt.ylabel(config.ylabel)
    plt.title(config.title)

    plt.legend()

    plt.savefig(config.output_file)


if __name__ == "__main__":
    main()

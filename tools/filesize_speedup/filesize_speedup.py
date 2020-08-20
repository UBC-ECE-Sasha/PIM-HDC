#!/usr/bin/env python3

import argparse
import sys
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# defaults for nice publication ready rendering
from matplotlib.ticker import FormatStrFormatter

fontsize = 12
legend_fontsize = 10

def parse_args() -> argparse.Namespace:
    """
    Parse commandline arguments
    :return: arguments
    """
    parser = argparse.ArgumentParser(description="Generate a bar chart")

    parser.add_argument(
        "--csvfile", help="Input CSV data file", required=True, type=str
    )
    parser.add_argument("--xlabel", help="Chart x-axis label", default="File Size (MB)", type=str)
    parser.add_argument(
        "--ylabel", help="Chart y-axis label", default="Speedup", type=str
    )
    parser.add_argument(
        "--title", help="Chart title", default="PIM-HDC Filesize Speedup", type=str
    )
    parser.add_argument(
        "--outputfile", help="Output chart", default="output.pdf", type=str
    )

    args = parser.parse_args()
    return args


def process(rows) -> dict:
    """

    :returns {
        host: [(x, y)]
        dpu: ([(d, t),x], [y])
    }
    """

    plots = {
        "x": [],
        "y": [],
        "tl": [],
        "dpus": [],
    }

    host = []
    filtered = {

    }

    for r in rows:
        if r[0] == 0:
            host.append((r[2], r[1]))
        else:
            if r[2] not in filtered:
                filtered[r[2]] = (r[2], r[1], r[3], r[4])
            elif r[1] < filtered[r[2]][1]:
                filtered[r[2]] = (r[2], r[1], r[3], r[4])

    dpu = [r for r in filtered.values()]

    host = [x for x in sorted(host, key=lambda v: v[0])]

    for h, d in zip(host, [x for x in sorted(dpu, key=lambda v: v[0])]):
        plots["x"].append(d[0] / (1024*1024))
        plots["y"].append(h[1] / d[1])
        plots["tl"].append(d[2])
        plots["dpus"].append(d[3])

    return plots


def main():
    config = parse_args()

    rows = np.loadtxt(config.csvfile, delimiter=",", skiprows=1)

    plots = process(rows)

    # 6.8 inch high figure, 2.5 inch across (matches column width of paper)
    fig, ax = plt.subplots(figsize=(6.8, 2.5))

    # x-axis labels
    ax.set_xticks(plots["x"])
    ax.set_xlabel('Speedup Over Host Application')
    ax.xaxis.grid(True, linestyle="dotted")

    # add grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")

    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.plot(plots["x"], plots["y"], label="DPU", marker="o")
    ax.plot(plots["x"], [1]*len(plots["y"]), label="Host", linestyle="--")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    topax = ax.secondary_xaxis('top')
    # Put a legend to the right of the current axis and reverse order
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))
    top_xticks = [f"{int(d)}\n({int(t)})" for d, t in zip(plots['dpus'], plots['tl'])]
    topax.set_xticks(plots["x"])
    topax.set_xlabel("DPUs (Tasklets)", fontsize=fontsize)
    topax.set_xticklabels(top_xticks)

    # set up legend
    ax.legend(bbox_to_anchor=(1, 1.04), loc='upper left', fontsize=legend_fontsize)

    plt.xlabel(config.xlabel, fontsize=fontsize)
    plt.ylabel(config.ylabel, fontsize=fontsize)

    plt.savefig(config.outputfile, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()

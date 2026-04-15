#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "scalar_flux_history.csv"
OUTPUT_GIF = SCRIPT_DIR / "flux_evolution.gif"
PLOT_GROUP = 0
FPS = 5
LEVEL_COUNT = 20


def load_scalar_flux_history(path: Path, group: int):
    frames = defaultdict(list)
    times = {}
    x_values = set()
    y_values = set()

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if int(row["group"]) != group:
                continue
            step = int(row["time_step"])
            time = float(row["time"])
            i = int(row["i"])
            j = int(row["j"])
            x = float(row["x_center"])
            y = float(row["y_center"])
            value = float(row["value"])
            frames[step].append((i, j, x, y, value))
            times[step] = time
            x_values.add(x)
            y_values.add(y)

    if not frames:
        raise RuntimeError(f"No rows for group {group} were found in {path}.")

    x_sorted = sorted(x_values)
    y_sorted = sorted(y_values)
    nx = len(x_sorted)
    ny = len(y_sorted)
    x_lookup = {x: k for k, x in enumerate(x_sorted)}
    y_lookup = {y: k for k, y in enumerate(y_sorted)}

    ordered_steps = sorted(frames)
    grids = []
    for step in ordered_steps:
        grid = np.zeros((ny, nx), dtype=float)
        for _, _, x, y, value in frames[step]:
            grid[y_lookup[y], x_lookup[x]] = value
        grids.append(grid)

    return ordered_steps, [times[s] for s in ordered_steps], np.array(x_sorted), np.array(y_sorted), grids


def main():
    steps, times, x, y, grids = load_scalar_flux_history(INPUT_CSV, PLOT_GROUP)
    X, Y = np.meshgrid(x, y)

    vmin = min(float(grid.min()) for grid in grids)
    vmax = max(float(grid.max()) for grid in grids)
    if abs(vmax - vmin) < 1.0e-14:
        vmax = vmin + 1.0
    levels = np.linspace(vmin, vmax, LEVEL_COUNT)

    fig, ax = plt.subplots(figsize=(8, 5))
    contour_holder = {"artist": None}
    colorbar_holder = {"artist": None}

    def draw_frame(frame_idx: int):
        ax.clear()
        contour = ax.contourf(X, Y, grids[frame_idx], levels=levels)
        if colorbar_holder["artist"] is None:
            colorbar_holder["artist"] = fig.colorbar(contour, ax=ax)
            colorbar_holder["artist"].set_label(f"Group {PLOT_GROUP} scalar flux")
        ax.set_title(f"Group {PLOT_GROUP} scalar flux, step={steps[frame_idx]}, time={times[frame_idx]:.6f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        contour_holder["artist"] = contour
        return []

    anim = FuncAnimation(fig, draw_frame, frames=len(grids), interval=200, blit=False)
    anim.save(OUTPUT_GIF, writer=PillowWriter(fps=FPS))
    plt.close(fig)


if __name__ == "__main__":
    main()

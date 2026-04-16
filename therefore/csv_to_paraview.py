#!/usr/bin/env python3
"""
Convert scalar_flux_history.csv into a ParaView-readable VTK XML time series.

Hard-coded inputs/outputs:
- input:  scalar_flux_history.csv in the same directory as this script
- output: paraview_flux/step_XXXX.vtr and paraview_flux/flux_series.pvd

The input CSV is expected to have columns:
    time_step,time,cell,i,j,group,x_center,y_center,value

The output files use a RECTILINEAR_GRID with CELL_DATA arrays named phi_g<group>.
The .pvd file references .vtr XML datasets, which ParaView's PVD reader expects.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = SCRIPT_DIR / "results/scalar_flux_history.csv"
OUTPUT_DIR = SCRIPT_DIR / "results/paraview_flux"
PVD_FILE = OUTPUT_DIR / "flux_series.pvd"
VTK_BASENAME = "step"


class FluxSeries:
    def __init__(self) -> None:
        self.times: Dict[int, float] = {}
        self.groups: List[int] = []
        self.x_centers: List[float] = []
        self.y_centers: List[float] = []
        self.frames: Dict[int, Dict[int, Dict[Tuple[int, int], float]]] = defaultdict(
            lambda: defaultdict(dict)
        )


def sorted_unique(values: List[float]) -> List[float]:
    return sorted(set(values))


def centers_to_edges(centers: List[float]) -> List[float]:
    if len(centers) == 0:
        raise RuntimeError("No coordinates were found.")
    if len(centers) == 1:
        half_width = 0.5
        return [centers[0] - half_width, centers[0] + half_width]

    edges = [0.0] * (len(centers) + 1)
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    for k in range(1, len(centers)):
        edges[k] = 0.5 * (centers[k - 1] + centers[k])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def read_flux_series(path: Path) -> FluxSeries:
    if not path.exists():
        raise RuntimeError(f"Input CSV was not found: {path}")

    series = FluxSeries()
    groups_seen = set()
    x_vals: List[float] = []
    y_vals: List[float] = []

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "time_step",
            "time",
            "i",
            "j",
            "group",
            "x_center",
            "y_center",
            "value",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        for row in reader:
            step = int(row["time_step"])
            time = float(row["time"])
            i = int(row["i"])
            j = int(row["j"])
            group = int(row["group"])
            x = float(row["x_center"])
            y = float(row["y_center"])
            value = float(row["value"])

            series.times[step] = time
            series.frames[step][group][(i, j)] = value
            groups_seen.add(group)
            x_vals.append(x)
            y_vals.append(y)

    if not series.frames:
        raise RuntimeError("No data rows were read from the CSV.")

    series.groups = sorted(groups_seen)
    series.x_centers = sorted_unique(x_vals)
    series.y_centers = sorted_unique(y_vals)
    return series


def fmt(values: List[float]) -> str:
    return " ".join(f"{v:.16e}" for v in values)


def write_rectilinear_vtr(
    vtk_path: Path,
    x_edges: List[float],
    y_edges: List[float],
    step_groups: Dict[int, Dict[Tuple[int, int], float]],
    groups: List[int],
) -> None:
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    with vtk_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write(
            '<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">\n'
        )
        fh.write(f'  <RectilinearGrid WholeExtent="0 {nx} 0 {ny} 0 0">\n')
        fh.write(f'    <Piece Extent="0 {nx} 0 {ny} 0 0">\n')
        fh.write('      <CellData>\n')
        for group in groups:
            values = step_groups.get(group, {})
            cell_values: List[float] = []
            # VTK cell ordering: x index fastest, then y.
            for j in range(ny):
                for i in range(nx):
                    cell_values.append(values.get((i, j), 0.0))
            fh.write(
                f'        <DataArray type="Float64" Name="phi_g{group}" format="ascii">\n'
            )
            fh.write('          ' + fmt(cell_values) + '\n')
            fh.write('        </DataArray>\n')
        fh.write('      </CellData>\n')
        fh.write('      <PointData/>\n')
        fh.write('      <Coordinates>\n')
        fh.write('        <DataArray type="Float64" Name="X_COORDINATES" NumberOfComponents="1" format="ascii">\n')
        fh.write('          ' + fmt(x_edges) + '\n')
        fh.write('        </DataArray>\n')
        fh.write('        <DataArray type="Float64" Name="Y_COORDINATES" NumberOfComponents="1" format="ascii">\n')
        fh.write('          ' + fmt(y_edges) + '\n')
        fh.write('        </DataArray>\n')
        fh.write('        <DataArray type="Float64" Name="Z_COORDINATES" NumberOfComponents="1" format="ascii">\n')
        fh.write('          0.0000000000000000e+00\n')
        fh.write('        </DataArray>\n')
        fh.write('      </Coordinates>\n')
        fh.write('    </Piece>\n')
        fh.write('  </RectilinearGrid>\n')
        fh.write('</VTKFile>\n')


def write_pvd(pvd_path: Path, datasets: List[Tuple[float, str]]) -> None:
    with pvd_path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write('<?xml version="1.0"?>\n')
        fh.write(
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        )
        fh.write('  <Collection>\n')
        for time_value, rel_file in datasets:
            fh.write(
                f'    <DataSet timestep="{time_value:.16e}" group="" part="0" file="{rel_file}"/>\n'
            )
        fh.write('  </Collection>\n')
        fh.write('</VTKFile>\n')


def main() -> None:
    series = read_flux_series(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_edges = centers_to_edges(series.x_centers)
    y_edges = centers_to_edges(series.y_centers)

    datasets: List[Tuple[float, str]] = []
    for step in sorted(series.frames):
        vtk_name = f"{VTK_BASENAME}_{step:04d}.vtr"
        vtk_path = OUTPUT_DIR / vtk_name
        write_rectilinear_vtr(
            vtk_path=vtk_path,
            x_edges=x_edges,
            y_edges=y_edges,
            step_groups=series.frames[step],
            groups=series.groups,
        )
        datasets.append((series.times[step], vtk_name))

    write_pvd(PVD_FILE, datasets)

    print(f"Wrote {len(datasets)} VTK XML time slices to: {OUTPUT_DIR}")
    print(f"Open this file in ParaView: {PVD_FILE}")
    print("Cell-data arrays:", ", ".join(f"phi_g{g}" for g in series.groups))


if __name__ == "__main__":
    main()

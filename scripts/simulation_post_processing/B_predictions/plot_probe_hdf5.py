#!/usr/bin/env python3
"""Inspect and plot packed MS-FLUKSS probe output.

Examples
--------
List stored variables and the probe-grid axes::

    python plot_probe_hdf5.py extracted_probes.h5 --list

Plot density for the outermost, zero-offset probe (the default probe)::

    python plot_probe_hdf5.py extracted_probes.h5 --variable density \
        --output density.png

Plot several probes selected by radius fraction and angular offsets::

    python plot_probe_hdf5.py extracted_probes.h5 --variable density \
        --probe 1.0,0,0 --probe 0.5,-10,10 --show

Axis selections use the nearest stored value.  A flattened probe index can be
selected directly with ``--probe-index``.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


POSITION_FIELDS = {
    "distance": 0,
    "distance_au": 0,
    "longitude": 1,
    "longitude_deg": 1,
    "lon": 1,
    "latitude": 2,
    "latitude_deg": 2,
    "lat": 2,
}


def decode_strings(dataset):
    values = np.asarray(dataset).reshape(-1)
    return [
        value.decode("utf-8").rstrip("\x00")
        if isinstance(value, (bytes, np.bytes_))
        else str(value)
        for value in values
    ]


def parse_probe(value):
    try:
        coordinates = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "probe must be RADIUS_FRACTION,LATITUDE_OFFSET,LONGITUDE_OFFSET"
        ) from exc
    if len(coordinates) != 3:
        raise argparse.ArgumentTypeError(
            "probe must contain exactly three comma-separated values"
        )
    return coordinates


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="packed probe-output HDF5 file")
    parser.add_argument(
        "--variable",
        default="0",
        help="field name or zero-based field index (default: 0)",
    )
    parser.add_argument(
        "--probe",
        action="append",
        type=parse_probe,
        metavar="R,LAT,LON",
        help=(
            "probe-axis coordinates; repeat to plot multiple probes. The "
            "default is the outermost radius and offsets nearest zero"
        ),
    )
    parser.add_argument(
        "--probe-index",
        action="append",
        type=int,
        help="zero-based flattened probe index; repeat for multiple probes",
    )
    parser.add_argument(
        "--x",
        choices=("time", "distance"),
        default="time",
        help="horizontal axis (default: time)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list variables, dimensions, and probe axes, then exit",
    )
    parser.add_argument("--output", type=Path, help="save the figure to this path")
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the interactive plot (may require an X connection)",
    )
    parser.add_argument("--title", help="custom plot title")
    return parser.parse_args()


def require_dataset(h5, name, ndim):
    if name not in h5:
        raise ValueError("{} has no {} dataset".format(h5.filename, name))
    dataset = h5[name]
    if dataset.ndim != ndim:
        raise ValueError(
            "{} must be {}-dimensional; got shape {}".format(
                name, ndim, dataset.shape
            )
        )
    return dataset


def read_axes(h5, probe_count):
    names = (
        "radius_fraction",
        "latitude_offset_deg",
        "longitude_offset_deg",
    )
    if "axes" not in h5 or any(name not in h5["axes"] for name in names):
        return None
    axes = tuple(np.asarray(h5["axes"][name]) for name in names)
    if np.prod([axis.size for axis in axes], dtype=np.int64) != probe_count:
        raise ValueError("the copied probe axes do not match /values")
    return axes


def nearest_index(axis, requested):
    return int(np.nanargmin(np.abs(axis.astype(float) - requested)))


def flat_probe_index(indices, shape):
    return int(np.ravel_multi_index(indices, shape, order="C"))


def select_probes(args, axes, probe_count):
    selected = []
    labels = []

    for index in args.probe_index or []:
        if index < 0 or index >= probe_count:
            raise ValueError(
                "probe index {} is outside [0, {})".format(index, probe_count)
            )
        selected.append(index)
        labels.append("probe {}".format(index))

    if args.probe and axes is None:
        raise ValueError("--probe requires /axes in the packed output")
    if axes is not None:
        shape = tuple(axis.size for axis in axes)
        requested_probes = args.probe
        if not requested_probes and not selected:
            requested_probes = [(float(axes[0][-1]), 0.0, 0.0)]
        for requested in requested_probes or []:
            indices = tuple(
                nearest_index(axis, coordinate)
                for axis, coordinate in zip(axes, requested)
            )
            actual = tuple(float(axis[index]) for axis, index in zip(axes, indices))
            selected.append(flat_probe_index(indices, shape))
            labels.append(
                "r={:g}, lat={:g}°, lon={:g}°".format(*actual)
            )

    if not selected:
        selected = [0]
        labels = ["probe 0"]
    return selected, labels


def resolve_variable(requested, field_names):
    lower_names = [name.lower() for name in field_names]
    if requested.lower() in POSITION_FIELDS:
        index = POSITION_FIELDS[requested.lower()]
        canonical = ("distance_au", "longitude_deg", "latitude_deg")[index]
        return "position", index, canonical
    try:
        index = int(requested)
    except ValueError:
        matches = [i for i, name in enumerate(lower_names) if name == requested.lower()]
        if not matches:
            raise ValueError(
                "unknown variable {!r}; use --list to see field names".format(
                    requested
                )
            )
        index = matches[0]
    if index < 0 or index >= len(field_names):
        raise ValueError(
            "variable index {} is outside [0, {})".format(index, len(field_names))
        )
    return "values", index, field_names[index]


def print_inventory(h5, values, positions, field_names, axes):
    print("File: {}".format(h5.filename))
    print("Records: {:,}".format(values.shape[0]))
    print("Probes: {:,}".format(values.shape[1]))
    print("Variables ({}):".format(values.shape[2]))
    for index, name in enumerate(field_names):
        print("  {:3d}  {}".format(index, name))
    print("Position fields:")
    for index, name in enumerate(("distance_au", "longitude_deg", "latitude_deg")):
        print("  {:3d}  {}".format(index, name))
    if axes is not None:
        labels = ("radius_fraction", "latitude_offset_deg", "longitude_offset_deg")
        print("Probe axes (flattened in this order):")
        for label, axis in zip(labels, axes):
            print(
                "  {}: {} values, {:g} through {:g}".format(
                    label, axis.size, float(np.min(axis)), float(np.max(axis))
                )
            )
    print("Position shape: {}".format(positions.shape))


def main():
    args = parse_args()
    if not args.list and args.output is None and not args.show:
        raise SystemExit("choose --output, --show, or --list")

    with h5py.File(args.input, "r") as h5:
        times = require_dataset(h5, "time", 1)
        values = require_dataset(h5, "values", 3)
        positions = require_dataset(h5, "position", 3)
        found = require_dataset(h5, "found", 2)
        if (
            values.shape[:2] != positions.shape[:2]
            or values.shape[:2] != found.shape
            or values.shape[0] != times.shape[0]
            or positions.shape[2] != 3
        ):
            raise ValueError("time, position, values, and found shapes disagree")

        field_names = decode_strings(require_dataset(h5, "field_name", 1))
        if len(field_names) != values.shape[2]:
            raise ValueError("/field_name does not match the /values fields")
        axes = read_axes(h5, values.shape[1])

        if args.list:
            print_inventory(h5, values, positions, field_names, axes)
            return

        probe_indices, labels = select_probes(args, axes, values.shape[1])
        source, variable_index, variable_name = resolve_variable(
            args.variable, field_names
        )
        time_values = np.asarray(times)
        found_values = np.asarray(found[:, probe_indices], dtype=bool)
        if source == "values":
            y_values = np.asarray(values[:, probe_indices, variable_index])
        else:
            y_values = np.asarray(positions[:, probe_indices, variable_index])
        if args.x == "time":
            x_values = np.repeat(time_values[:, None], len(probe_indices), axis=1)
            x_label = "Time (decimal year)"
        else:
            x_values = np.asarray(positions[:, probe_indices, 0])
            x_label = "Distance (AU)"

    if args.output is not None and not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for column, label in enumerate(labels):
        mask = (
            found_values[:, column]
            & np.isfinite(x_values[:, column])
            & np.isfinite(y_values[:, column])
        )
        axis.plot(
            x_values[mask, column],
            y_values[mask, column],
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=label,
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(variable_name)
    axis.set_title(args.title or "MS-FLUKSS probe: {}".format(variable_name))
    axis.grid(True, alpha=0.25)
    if len(labels) > 1:
        axis.legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180)
        print("Wrote {}".format(args.output.resolve()))
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()

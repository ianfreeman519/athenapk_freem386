# Importing important libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt

import argparse
import re
import matplotlib.animation as animation
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Make a GIF from orszag_tang phdf outputs in a directory."
)
parser.add_argument("directory", help="Directory containing the phdf files")
parser.add_argument(
    "-o", "--output", default="orszag_tang", help="Output GIF filename"
)
args = parser.parse_args()

data_dir = Path(args.directory).expanduser()
files = sorted(data_dir.glob("*.phdf"))
if not files:
    raise FileNotFoundError(f"No .phdf files found in {data_dir}")

n_match = re.search(r"N(\d+)", files[0].name)
n_label = f"N{n_match.group(1)}" if n_match else None

output = Path(args.output)
if not output.is_absolute():
    output = data_dir.parent / output
if n_label:
    output = output.with_name(f"{output.name}_{n_label}")

block = 0


def read_fields(fname):
    with h5py.File(fname, "r") as f:
        pre = f["prim"][block, 4, 0, :, :]
        Bx = f["prim"][block, 5, 0, :, :]
        By = f["prim"][block, 6, 0, :, :]
        Bz = f["prim"][block, 7, 0, :, :]
        t = f["Info"].attrs["Time"]
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    pressure_rot180_abs = np.abs(pre - pre[::-1, ::-1])
    return pre, Bmag, pressure_rot180_abs, t


with h5py.File(files[0], "r") as f:
    x = f["VolumeLocations"]["x"][block, :]
    y = f["VolumeLocations"]["y"][block, :]

vmin_bmag, vmax_bmag = np.inf, -np.inf
vmin_pre, vmax_pre = np.inf, -np.inf
vmin_pressure_rot180_abs, vmax_pressure_rot180_abs = np.inf, -np.inf
for fname in files:
    pre, Bmag, pressure_rot180_abs, _ = read_fields(fname)
    vmin_bmag = min(vmin_bmag, Bmag.min())
    vmax_bmag = max(vmax_bmag, Bmag.max())
    vmin_pre = min(vmin_pre, pre.min())
    vmax_pre = max(vmax_pre, pre.max())
    vmin_pressure_rot180_abs = min(vmin_pressure_rot180_abs, pressure_rot180_abs.min())
    vmax_pressure_rot180_abs = max(vmax_pressure_rot180_abs, pressure_rot180_abs.max())


def make_gif(quantity_name, label, cmap, vmin, vmax, getter, suffix):
    first_pre, first_bmag, first_diff, _ = read_fields(files[0])
    data = getter(first_pre, first_bmag, first_diff)

    # Size the figure to match the data's aspect ratio so the domain isn't stretched
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    fig, ax = plt.subplots(figsize=(6 * x_range / y_range, 6))
    ax.set_aspect("equal")

    im = ax.pcolormesh(x, y, data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    base_title = f"Orszag-Tang Vortex ({n_label})" if n_label else "Orszag-Tang Vortex"

    def update(fname):
        pre, Bmag, pressure_rot180_abs, t = read_fields(fname)
        im.set_array(getter(pre, Bmag, pressure_rot180_abs).ravel())
        ax.set_title(f"{base_title}: {quantity_name}\nt = {t:.3f}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=files, blit=True)
    gif_output = output.with_name(f"{output.name}_{suffix}.gif")
    ani.save(gif_output, writer="pillow", fps=4)
    print(f"saved gif: {gif_output}")
    plt.close(fig)


make_gif("pressure", "pressure", "viridis", vmin_pre, vmax_pre,
         lambda pre, Bmag, pressure_rot180_abs: pre, "pressure")
make_gif("|B|", "|B|", "inferno", vmin_bmag, vmax_bmag,
         lambda pre, Bmag, pressure_rot180_abs: Bmag, "Bmag")
make_gif("|P - rot180(P)|", "|P - rot180(P)|", "magma",
         vmin_pressure_rot180_abs, vmax_pressure_rot180_abs,
         lambda pre, Bmag, pressure_rot180_abs: pressure_rot180_abs,
         "pressure_rot180_abs")

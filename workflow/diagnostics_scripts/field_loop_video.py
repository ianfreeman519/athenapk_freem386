# Importing important libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt

import argparse
import re
import matplotlib.animation as animation
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Make a GIF from field loop phdf outputs in a directory."
)
parser.add_argument("directory", help="Directory containing the phdf files")
parser.add_argument(
    "-o", "--output", default="field_loop", help="Output GIF filename"
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
title = f"Field Loop ({n_label})" if n_label else "Field Loop"


def load_bmag(fname):
    with h5py.File(fname, "r") as f:
        Bx = f["prim"][block, 5, :, :, :]
        By = f["prim"][block, 6, :, :, :]
        Bz = f["prim"][block, 7, :, :, :]
        Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
        t = f["Info"].attrs["Time"]
    return Bmag, t


def get_plane(Bmag, plane):
    if plane == "xy":
        return Bmag[Bmag.shape[0] // 2, :, :]
    if plane == "xz":
        return Bmag[:, Bmag.shape[1] // 2, :]
    if plane == "yz":
        return Bmag[:, :, Bmag.shape[2] // 2]
    raise ValueError(f"Unknown plane: {plane}")


def make_movie(plane, xcoord, ycoord, xlabel, ylabel):
    vmin, vmax = np.inf, -np.inf
    for fname in files:
        Bmag, _ = load_bmag(fname)
        Bslice = get_plane(Bmag, plane)
        vmin = min(vmin, Bslice.min())
        vmax = max(vmax, Bslice.max())

    Bmag0, _ = load_bmag(files[0])
    Bslice0 = get_plane(Bmag0, plane)

    x_range = xcoord.max() - xcoord.min()
    y_range = ycoord.max() - ycoord.min()
    fig, ax = plt.subplots(figsize=(6 * x_range / y_range, 6))
    ax.set_aspect("equal")

    im = ax.pcolormesh(xcoord, ycoord, Bslice0, cmap="inferno", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label="|B|")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def update(fname):
        Bmag, t = load_bmag(fname)
        Bslice = get_plane(Bmag, plane)
        im.set_array(Bslice.ravel())
        ax.set_title(f"{title}: {plane} slice\nt = {t:.3f}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=files, blit=True)
    gif_output = output.with_name(f"{output.name}_Bmag_{plane}.gif")
    ani.save(gif_output, writer="pillow", fps=4)
    print(f"saved gif: {gif_output}")
    plt.close(fig)


with h5py.File(files[0], "r") as f:
    x = f["VolumeLocations"]["x"][block, :]
    y = f["VolumeLocations"]["y"][block, :]
    z = f["VolumeLocations"]["z"][block, :]

make_movie("xy", x, y, "x", "y")

Bmag0, _ = load_bmag(files[0])
if Bmag0.shape[0] > 1:
    make_movie("xz", x, z, "x", "z")
    make_movie("yz", y, z, "y", "z")

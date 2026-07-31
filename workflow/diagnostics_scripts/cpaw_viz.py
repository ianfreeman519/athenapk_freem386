"""Create a Bz animation from single-block CPAW PHDF outputs."""

import argparse
from pathlib import Path

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description="Make a Bz GIF from single-block CPAW PHDF outputs."
)
parser.add_argument("directory", help="Directory containing the PHDF files")
parser.add_argument(
    "-o", "--output", default="cpaw_bz.gif", help="Output GIF filename"
)
parser.add_argument(
    "--bmax",
    type=float,
    default=0.1,
    help="Symmetric Bz color limit (default: 0.1)",
)
parser.add_argument(
    "--fps", type=int, default=8, help="Animation frames per second (default: 8)"
)
parser.add_argument(
    "--label", default=None, help="Optional scenario label to include in the title"
)
args = parser.parse_args()

if args.bmax <= 0.0:
    parser.error("--bmax must be positive")
if args.fps <= 0:
    parser.error("--fps must be positive")

data_dir = Path(args.directory).expanduser()
files = sorted(data_dir.glob("*.phdf"))
if not files:
    raise FileNotFoundError(f"No .phdf files found in {data_dir}")

output = Path(args.output).expanduser()
if not output.is_absolute():
    output = data_dir.parent / output
if output.suffix.lower() != ".gif":
    output = output.with_suffix(".gif")

# CPAW visualization runs use one MeshBlock and one x3 cell. In the primitive
# variable ordering, B1, B2, and B3 are components 5, 6, and 7, respectively.
block = 0
bz_component = 7
x3_index = 0


def read_bz(filename):
    with h5py.File(filename, "r") as f:
        if "prim" not in f:
            raise KeyError(
                f"{filename} does not contain 'prim'; configure the output variables as prim"
            )
        prim = f["prim"]
        if prim.ndim != 5 or prim.shape[1] <= bz_component:
            raise ValueError(
                f"Unexpected prim shape {prim.shape} in {filename}; "
                "expected (block, variable, x3, x2, x1) with Bz at component 7"
            )
        bz = prim[block, bz_component, x3_index, :, :]
        time = f["Info"].attrs["Time"]
    return bz, time


with h5py.File(files[0], "r") as f:
    x1 = f["VolumeLocations"]["x"][block, :]
    x2 = f["VolumeLocations"]["y"][block, :]

bz0, time0 = read_bz(files[0])

# Preserve the physical domain aspect ratio so diagonal propagation is not distorted.
x1_range = x1.max() - x1.min()
x2_range = x2.max() - x2.min()
if x1_range <= 0.0 or x2_range <= 0.0:
    raise ValueError("CPAW Bz visualization requires at least two cells in x1 and x2")

fig, ax = plt.subplots(figsize=(6.0 * x1_range / x2_range, 6.0))
ax.set_aspect("equal")

image = ax.pcolormesh(
    x1,
    x2,
    bz0,
    shading="auto",
    cmap="RdBu_r",
    vmin=-args.bmax,
    vmax=args.bmax,
)
fig.colorbar(image, ax=ax, label=r"$B_z$")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
scenario = f" ({args.label})" if args.label else ""
base_title = rf"Circularly polarized Alfvén wave{scenario}: $B_z$"
title = ax.set_title(base_title + "\n" + rf"$t={time0:.3f}$")


def update(filename):
    bz, time = read_bz(filename)
    image.set_array(bz.ravel())
    title.set_text(base_title + "\n" + rf"$t={time:.3f}$")
    return image, title


movie = animation.FuncAnimation(fig, update, frames=files, blit=True)
movie.save(output, writer="pillow", fps=args.fps)
print(f"saved gif: {output}")
plt.close(fig)

# Importing important libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt

import argparse
import re
import matplotlib.animation as animation
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Make a GIF from current sheet phdf outputs in a directory."
)
parser.add_argument("directory", help="Directory containing the phdf files")
parser.add_argument(
    "-o", "--output", default="current_sheet", help="Output GIF filename"
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

with h5py.File(files[0], "r") as f:
    x = f["VolumeLocations"]["x"][0, :]
    y = f["VolumeLocations"]["y"][0, :]

block=0
# Compute a global vmin/vmax across all frames so the colorbar stays fixed
vmin, vmax = np.inf, -np.inf
for fname in files:
    with h5py.File(fname, "r") as f:
        By = f["prim"][block, 6, 0, :, :]
    vmin = min(vmin, By.min())
    vmax = max(vmax, By.max())


with h5py.File(files[0], "r") as f:
    By = f["prim"][block, 6, 0, :, :]

# Size the figure to match the data's aspect ratio so the domain isn't stretched
x_range = x.max() - x.min()
y_range = y.max() - y.min()
fig, ax = plt.subplots(figsize=(6 * x_range / y_range, 6))
ax.set_aspect("equal")

im = ax.pcolormesh(x, y, By, cmap='jet', vmin=-1.0, vmax=1.0)
fig.colorbar(im, ax=ax, label="By")
ax.set_xlabel("x")
ax.set_ylabel("y")

title = f"Current Sheet By ({n_label})" if n_label else "Current Sheet By"

def update(fname):
    with h5py.File(fname, "r") as f:
        By = f["prim"][block, 6, 0, :, :]
        t = f["Info"].attrs["Time"]
    im.set_array(By.ravel())
    ax.set_title(f"{title}\nt = {t:.3f}")
    return [im]

ani = animation.FuncAnimation(fig, update, frames=files, blit=True)
gif_output = output.with_name(f"{output.name}_By.gif")
# mp4_output = output.with_name(f"{output.name}_Bmag.mp4")
ani.save(gif_output, writer="pillow", fps=4)
print(f"saved gif: {gif_output}")
# ani.save(mp4_output, writer="ffmpeg", fps=8)
# print(f"saved mp4: {mp4_output}")
plt.close(fig)

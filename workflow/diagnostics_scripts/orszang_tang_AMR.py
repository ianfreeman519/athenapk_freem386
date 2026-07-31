import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yt


parser = argparse.ArgumentParser(
    description="Make a pressure movie with AMR meshblock boundaries overlaid."
)
parser.add_argument("phdf_dir", help="Directory containing the Orszag-Tang PHDF files")
parser.add_argument(
    "-o",
    "--output",
    default="orszag_tang_refined.mp4",
    help="Output movie path (default: %(default)s)",
)
parser.add_argument("--fps", type=int, default=8, help="Movie frame rate")
args = parser.parse_args()

phdf_dir = Path(args.phdf_dir).expanduser()
files = sorted(phdf_dir.glob("*.phdf"))

if not files:
    raise FileNotFoundError(f"No PHDF files found in {phdf_dir}")

field = ("gas", "pressure")


def find_global_field_range():
    field_min = np.inf
    field_max = -np.inf

    for fname in files:
        ds = yt.load(str(fname))
        field_data = ds.all_data()[field]
        field_min = min(field_min, float(field_data.min().to_value()))
        field_max = max(field_max, float(field_data.max().to_value()))

    return field_min, field_max


pressure_min, pressure_max = find_global_field_range()
print(f"Using fixed pressure range: [{pressure_min}, {pressure_max}]")


def render_frame(fname):
    ds = yt.load(str(fname))

    slc = yt.SlicePlot(ds, "z", field)
    slc.set_cmap(field, "inferno")
    slc.set_log(field, False)
    slc.set_zlim(field, pressure_min, pressure_max)
    slc.annotate_timestamp()
    slc.annotate_grids()
    slc.render()

    yt_fig = slc.plots[field].figure
    yt_fig.canvas.draw()

    frame = np.asarray(yt_fig.canvas.buffer_rgba()).copy()
    plt.close(yt_fig)

    return frame


first_frame = render_frame(files[0])

fig, ax = plt.subplots(
    figsize=(
        first_frame.shape[1] / 100,
        first_frame.shape[0] / 100,
    )
)
image = ax.imshow(first_frame)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


def update(fname):
    image.set_data(render_frame(fname))
    return (image,)


ani = animation.FuncAnimation(
    fig,
    update,
    frames=files,
    blit=True,
)

output = Path(args.output).expanduser()
output.parent.mkdir(parents=True, exist_ok=True)
ani.save(str(output), writer="ffmpeg", fps=args.fps)

plt.close(fig)
print(f"saved {output}")

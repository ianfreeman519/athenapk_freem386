# Importing important libraries
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import yt

import argparse
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Make field-loop MP4 movies from PHDF outputs in a directory."
)
parser.add_argument("directory", help="Directory containing the phdf files")
parser.add_argument(
    "-o", "--output", default="field_loop", help="Output movie basename"
)
parser.add_argument("--fps", type=int, default=8, help="Movie frame rate")

args = parser.parse_args()

data_dir = Path(args.directory).expanduser()
files = sorted(data_dir.glob("*.phdf"))
if not files:
    raise FileNotFoundError(f"No .phdf files found in {data_dir}")

base_output = Path(args.output)
if not base_output.is_absolute():
    base_output = data_dir.parent / base_output


field = ("gas", "magnetic_field_magnitude")

def find_global_field_range():
    field_min = np.inf
    field_max = -np.inf

    for fname in files:
        ds = yt.load(str(fname))
        field_data = ds.all_data()[field]
        field_min = min(field_min, float(field_data.min().to_value()))
        field_max = max(field_max, float(field_data.max().to_value()))

    return field_min, field_max

Bmag_min, Bmag_max = find_global_field_range()
print(f"Using fixed Bmag range: [{Bmag_min}, {Bmag_max}]")

def render_frame(fname, plane):
    ds = yt.load(str(fname))

    slc = yt.SlicePlot(ds, plane, field)
    slc.set_cmap(field, "inferno")
    slc.set_log(field, False)
    slc.set_zlim(field, Bmag_min, Bmag_max)
    slc.annotate_timestamp()
    slc.annotate_grids(edgecolors='red')
    slc.render()

    yt_fig = slc.plots[field].figure
    yt_fig.canvas.draw()

    frame = np.asarray(yt_fig.canvas.buffer_rgba()).copy()
    plt.close(yt_fig)

    return frame


def make_movie(plane):
    first_frame = render_frame(files[0], plane)

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
        image.set_data(render_frame(fname, plane))
        return (image,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=files,
        blit=True,
    )

    movie_output = base_output.with_name(f"{base_output.name}_Bmag_{plane}.mp4")
    movie_output.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(movie_output), writer="ffmpeg", fps=args.fps)

    plt.close(fig)
    print(f"saved {movie_output}")


first_ds = yt.load(str(files[0]))
planes = ["z"] if int(first_ds.domain_dimensions[2]) == 1 else ["z", "y", "x"]
for plane in planes:
    make_movie(plane)

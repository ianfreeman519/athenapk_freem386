# Importing important libraries
import numpy as np
import h5py
import matplotlib.pyplot as plt
import yt
yt.set_log_level(50)    # Suppress yt warnings - they can be very verbose

import argparse
import re
import matplotlib.animation as animation
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Make magnetic field-line plots from current sheet phdf outputs."
)
parser.add_argument("directory", help="Directory containing the phdf files")
parser.add_argument(
    "-o", "--output", default="current_sheet", help="Output filename prefix"
)
args = parser.parse_args()

data_dir = Path(args.directory).expanduser()

output = Path(args.output)
if not output.is_absolute():
    output = data_dir.parent / output


def require_one(pattern):
    matches = sorted(data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No phdf file matching {pattern!r} found in {data_dir}")
    return matches[0]


init_file = require_one("*00000.phdf")
mid_file = require_one("*00050.phdf")
final_file = require_one("*final.phdf")

snapshots = [
    ("initial", init_file),
    ("middle", mid_file),
    ("final", final_file),
]

field = ("gas", "magnetic_field_y")
stream_x = ("gas", "magnetic_field_x")
stream_y = ("gas", "magnetic_field_y")

for label, fname in snapshots:
    ds = yt.load(str(fname))
    slc = yt.SlicePlot(ds, "z", field)
    slc.set_cmap(field, "jet")
    slc.set_log(field, False)
    slc.set_zlim(field, -1.0, 1.0)
    slc.annotate_streamlines(stream_x, stream_y, color="black", linewidth=1.0)
    slc.annotate_timestamp()

    outname = output.with_name(f"{output.name}_field_lines_{label}.png")
    slc.save(str(outname))

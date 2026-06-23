import os
import numpy as np
import h5py
import yt
import re
from scipy.interpolate import CubicSpline
from pathlib import Path
from typing import Any, Dict
from ipyevents import Event

import ipywidgets as widgets
from IPython.display import display, clear_output

_block_re   = re.compile(r"<\s*([^>]+?)\s*>")     # captures whatever is between '<' and '>'
_comment_re = re.compile(r"#.*$")                 # strip inline comments

# Creating an object to store all the fields so I can call them with field.velocity_x rather than ("gas", "velocity_x")
class Fields:
    """
    All the fields:
    alfven_speed, angular_momentum_magnitude = ('gas', 'alfven_speed'), ('gas', 'angular_momentum_magnitude')
    angular_momentum_x, angular_momentum_y, angular_momentum_z = ('gas', 'angular_momentum_x'), ('gas', 'angular_momentum_y'), ('gas', 'angular_momentum_z')
    averaged_density, baroclinic_vorticity_magnitude = ('gas', 'averaged_density'), ('gas', 'baroclinic_vorticity_magnitude')
    baroclinic_vorticity_x, baroclinic_vorticity_y, baroclinic_vorticity_z = ('gas', 'baroclinic_vorticity_x'), ('gas', 'baroclinic_vorticity_y'), ('gas', 'baroclinic_vorticity_z')
    cell_mass, cell_volume = ('gas', 'cell_mass'), ('gas', 'cell_volume')
    courant_time_step, density = ('gas', 'courant_time_step'), ('gas', 'density')
    density_gradient_magnitude, density_gradient_x, density_gradient_y, density_gradient_z = ('gas', 'density_gradient_magnitude'), ('gas', 'density_gradient_x'), ('gas', 'density_gradient_y'), ('gas', 'density_gradient_z')
    dx_gas, dy_gas, dynamical_time, dz_gas = ('gas', 'dx'), ('gas', 'dy'), ('gas', 'dynamical_time'), ('gas', 'dz')
    jeans_mass, kT, kinetic_energy_density, lorentz_factor = ('gas', 'jeans_mass'), ('gas', 'kT'), ('gas', 'kinetic_energy_density'), ('gas', 'lorentz_factor')
    mach_alfven, mach_number = ('gas', 'mach_alfven'), ('gas', 'mach_number')
    magnetic_energy_density, magnetic_field_los, magnetic_field_magnitude = ('gas', 'magnetic_energy_density'), ('gas', 'magnetic_field_los'), ('gas', 'magnetic_field_magnitude')
    magnetic_field_strength, magnetic_field_x, magnetic_field_y, magnetic_field_z = ('gas', 'magnetic_field_strength'), ('gas', 'magnetic_field_x'), ('gas', 'magnetic_field_y'), ('gas', 'magnetic_field_z')
    magnetic_pressure, mass, mean_molecular_weight = ('gas', 'magnetic_pressure'), ('gas', 'mass'), ('gas', 'mean_molecular_weight')
    momentum_density_x, momentum_density_y, momentum_density_z = ('gas', 'momentum_density_x'), ('gas', 'momentum_density_y'), ('gas', 'momentum_density_z')
    momentum_x, momentum_y, momentum_z = ('gas', 'momentum_x'), ('gas', 'momentum_y'), ('gas', 'momentum_z')
    number_density, plasma_beta = ('gas', 'number_density'), ('gas', 'plasma_beta')
    pressure, pressure_gradient_magnitude, pressure_gradient_x, pressure_gradient_y, pressure_gradient_z = ('gas', 'pressure'), ('gas', 'pressure_gradient_magnitude'), ('gas', 'pressure_gradient_x'), ('gas', 'pressure_gradient_y'), ('gas', 'pressure_gradient_z')
    relative_magnetic_field_x, relative_magnetic_field_y, relative_magnetic_field_z = ('gas', 'relative_magnetic_field_x'), ('gas', 'relative_magnetic_field_y'), ('gas', 'relative_magnetic_field_z')
    relative_velocity_x, relative_velocity_y, relative_velocity_z = ('gas', 'relative_velocity_x'), ('gas', 'relative_velocity_y'), ('gas', 'relative_velocity_z')
    shear, shear_criterion, shear_mach, sound_speed = ('gas', 'shear'), ('gas', 'shear_criterion'), ('gas', 'shear_mach'), ('gas', 'sound_speed')
    specific_angular_momentum_magnitude, specific_angular_momentum_x, specific_angular_momentum_y, specific_angular_momentum_z = ('gas', 'specific_angular_momentum_magnitude'), ('gas', 'specific_angular_momentum_x'), ('gas', 'specific_angular_momentum_y'), ('gas', 'specific_angular_momentum_z')
    specific_thermal_energy, temperature, velocity_los = ('gas', 'specific_thermal_energy'), ('gas', 'temperature'), ('gas', 'velocity_los')
    velocity_magnitude, velocity_x, velocity_y, velocity_z = ('gas', 'velocity_magnitude'), ('gas', 'velocity_x'), ('gas', 'velocity_y'), ('gas', 'velocity_z')
    volume, vorticity_magnitude, vorticity_squared = ('gas', 'volume'), ('gas', 'vorticity_magnitude'), ('gas', 'vorticity_squared')
    vorticity_x, vorticity_y, vorticity_z = ('gas', 'vorticity_x'), ('gas', 'vorticity_y'), ('gas', 'vorticity_z')
    x_gas, y_gas, z_gas = ('gas', 'x'), ('gas', 'y'), ('gas', 'z')
    """
    def __init__(self):
        self.curlBx = ("parthenon", "curlBx")
        self.curlBy = ("parthenon", "curlBy")
        self.curlBz = ("parthenon", "curlBz")
        self.T = ("parthenon", "T")
        self.alfven_speed = ("gas", "alfven_speed")
        self.angular_momentum_magnitude = ("gas", "angular_momentum_magnitude")
        self.angular_momentum_x = ("gas", "angular_momentum_x")
        self.angular_momentum_y = ("gas", "angular_momentum_y")
        self.angular_momentum_z = ("gas", "angular_momentum_z")
        self.averaged_density = ("gas", "averaged_density")
        self.baroclinic_vorticity_magnitude = ("gas", "baroclinic_vorticity_magnitude")
        self.baroclinic_vorticity_x = ("gas", "baroclinic_vorticity_x")
        self.baroclinic_vorticity_y = ("gas", "baroclinic_vorticity_y")
        self.baroclinic_vorticity_z = ("gas", "baroclinic_vorticity_z")
        self.cell_mass = ("gas", "cell_mass")
        self.cell_volume = ("gas", "cell_volume")
        self.courant_time_step = ("gas", "courant_time_step")
        self.density = ("gas", "density")
        self.density_gradient_magnitude = ("gas", "density_gradient_magnitude")
        self.density_gradient_x = ("gas", "density_gradient_x")
        self.density_gradient_y = ("gas", "density_gradient_y")
        self.density_gradient_z = ("gas", "density_gradient_z")
        self.dx_gas = ("gas", "dx")
        self.dy_gas = ("gas", "dy")
        self.dynamical_time = ("gas", "dynamical_time")
        self.dz_gas = ("gas", "dz")
        self.jeans_mass = ('gas', 'jeans_mass')
        self.kT = ('gas', 'kT')
        self.kinetic_energy_density = ('gas', 'kinetic_energy_density')
        self.lorentz_factor = ('gas', 'lorentz_factor')
        self.mach_alfven = ('gas', 'mach_alfven')
        self.mach_number = ('gas', 'mach_number')
        self.magnetic_energy_density = ('gas', 'magnetic_energy_density')
        self.magnetic_field_los = ('gas', 'magnetic_field_los')
        self.magnetic_field_magnitude = ('gas', 'magnetic_field_magnitude')
        self.magnetic_field_strength = ('gas', 'magnetic_field_strength')
        self.magnetic_field_x = ('gas', 'magnetic_field_x')
        self.magnetic_field_y = ('gas', 'magnetic_field_y')
        self.magnetic_field_z = ('gas', 'magnetic_field_z')
        self.magnetic_pressure = ('gas', 'magnetic_pressure')
        self.mean_molecular_weight = ('gas', 'mean_molecular_weight')
        self.momentum_density_x = ('gas', 'momentum_density_x')
        self.momentum_density_y = ('gas', 'momentum_density_y')
        self.momentum_density_z = ('gas', 'momentum_density_z')
        self.number_density = ('gas', 'number_density')
        self.plasma_beta = ('gas', 'plasma_beta')
        self.pressure = ('gas', 'pressure')
        self.pressure_gradient_magnitude = ('gas', 'pressure_gradient_magnitude')
        self.pressure_gradient_x = ('gas', 'pressure_gradient_x')
        self.pressure_gradient_y = ('gas', 'pressure_gradient_y')
        self.pressure_gradient_z = ('gas', 'pressure_gradient_z')
        self.relative_magnetic_field_x = ('gas', 'relative_magnetic_field_x')
        self.relative_magnetic_field_y = ('gas', 'relative_magnetic_field_y')
        self.relative_magnetic_field_z = ('gas', 'relative_magnetic_field_z')
        self.relative_velocity_x = ('gas', 'relative_velocity_x')
        self.relative_velocity_y = ('gas', 'relative_velocity_y')
        self.relative_velocity_z = ('gas', 'relative_velocity_z')
        self.shear = ('gas', 'shear')
        self.shear_criterion = ('gas', 'shear_criterion')
        self.shear_mach = ('gas', 'shear_mach')
        self.sound_speed = ('gas', 'sound_speed')
        self.specific_angular_momentum_magnitude = ('gas', 'specific_angular_momentum_magnitude')
        self.specific_angular_momentum_x = ('gas', 'specific_angular_momentum_x')
        self.specific_angular_momentum_y = ('gas', 'specific_angular_momentum_y')
        self.specific_angular_momentum_z = ('gas', 'specific_angular_momentum_z')
        self.specific_thermal_energy = ('gas', 'specific_thermal_energy')
        self.temperature = ('gas', 'temperature')
        self.velocity_los = ('gas', 'velocity_los')
        self.velocity_magnitude = ('gas', 'velocity_magnitude')
        self.velocity_x = ('gas', 'velocity_x')
        self.velocity_y = ('gas', 'velocity_y')
        self.velocity_z = ('gas', 'velocity_z')
        self.volume = ('gas', 'volume')
        self.vorticity_magnitude = ('gas', 'vorticity_magnitude')
        self.vorticity_squared = ('gas', 'vorticity_squared')
        self.vorticity_x = ('gas', 'vorticity_x')
        self.vorticity_y = ('gas', 'vorticity_y')
        self.vorticity_z = ('gas', 'vorticity_z')
        self.x_gas = ('gas', 'x')
        self.y_gas = ('gas', 'y')
        self.z_gas = ('gas', 'z')
        self.adiabatic_heating_timescale = ('gas', 'adiabatic_heating_timescale')
        self.cooling_timescale = ('gas', 'cooling_timescale')
        self.ohmic_heating_timescale = ('gas', 'ohmic_heating_timescale')
        self.eta = ("parthenon", "eta")
        self.dt_hyp_fms = ("parthenon", "dt_hyp_fms")
        self.dt_hyp_cs = ("parthenon", "dt_hyp_cs")
        self.dt_cool_local = ("parthenon", "dt_cool_local")
        self.dt_heat_local = ("parthenon", "dt_heat_local")
        self.beta = ('parthenon', 'beta')
        self.prim_density =  ('parthenon', 'prim_density')
        self.prim_magnetic_field_1 = ('parthenon', 'prim_magnetic_field_1')
        self.prim_magnetic_field_2 = ('parthenon', 'prim_magnetic_field_2')
        self.prim_magnetic_field_3 = ('parthenon', 'prim_magnetic_field_3')
        self.prim_magnetic_psi = ('parthenon', 'prim_magnetic_psi')
        self.prim_pressure = ('parthenon', 'prim_pressure')
        self.prim_velocity_1 = ('parthenon', 'prim_velocity_1')
        self.prim_velocity_2 = ('parthenon', 'prim_velocity_2')
        self.prim_velocity_3 = ('parthenon', 'prim_velocity_3')
        self.cons_density = ('parthenon', 'cons_density')
        self.cons_magnetic_field_1 = ('parthenon', 'cons_magnetic_field_1')
        self.cons_magnetic_field_2 = ('parthenon', 'cons_magnetic_field_2')
        self.cons_magnetic_field_3 = ('parthenon', 'cons_magnetic_field_3')
        self.cons_magnetic_psi = ('parthenon', 'cons_magnetic_psi')
        self.cons_momentum_density_1 = ('parthenon', 'cons_momentum_density_1')
        self.cons_momentum_density_2 = ('parthenon', 'cons_momentum_density_2')
        self.cons_momentum_density_3 = ('parthenon', 'cons_momentum_density_3')
        self.cons_total_energy_density = ('parthenon', 'cons_total_energy_density')

class AthenaPKSliceExplorer:
    def __init__(
        self,
        directoryDict,
        fields,
        basename="parthenon",
        output_type="out0",
        dictkey="wider_sheet_fixed",
        default_index=0,
    ):
        self.ui = None
        self.key_event = None
        self.directoryDict = directoryDict
        self.fields = fields
        self.basename = basename
        self.output_type = output_type
        self.dictkey = dictkey

        self.fileseries = None
        self.ts = None
        self.ds = None

        self._build_widgets(default_index)

        self._suppress_callbacks = True
        self._load_series(quiet=True)
        self._suppress_callbacks = False

        self._register_callbacks()

    def _on_keypress(self, event):
        """
        Replot on selected keyboard shortcuts.

        Notes:
        - event["key"] is the pressed key, e.g. "p", "Enter"
        - event["shiftKey"] is True when Shift is held
        - event["ctrlKey"], event["altKey"], event["metaKey"] are also available
        """

        key = event.get("key", "")
        shift = event.get("shiftKey", False)

        # Press shift+p to replot
        if key.lower() == "p" and shift:
            self.plot()

        # Press Enter to replot
        elif key == "Enter":
            self.plot()

    def _register_callbacks(self):
        # Clear button callbacks
        self.plot_button._click_handlers.callbacks = []
        self.reload_button._click_handlers.callbacks = []

        self.plot_button.on_click(self._on_plot_clicked)
        self.reload_button.on_click(self._on_reload_clicked)

        # Clear dictkey observers
        for cb in list(self.dictkey_widget._trait_notifiers.get("value", {}).get("change", [])):
            self.dictkey_widget.unobserve(cb, names="value")

        self.dictkey_widget.observe(self._on_dictkey_changed, names="value")

    def _build_widgets(self, default_index):
        self.dictkey_widget = widgets.Dropdown(
            options=list(self.directoryDict.keys()),
            value=self.dictkey,
            description="Run:",
            layout=widgets.Layout(width="350px"),
        )

        self.index_widget = widgets.IntSlider(
            value=default_index,
            min=0,
            max=1,
            step=1,
            description="Index:",
            continuous_update=False,
            layout=widgets.Layout(width="500px"),
        )

        self.axis_widget = widgets.Dropdown(
            options=["x", "y", "z"],
            value="z",
            description="Axis:",
        )

        field_options = {}

        for attr_name in dir(self.fields):
            if attr_name.startswith("_"):
                continue

            field_value = getattr(self.fields, attr_name)

            if (
                isinstance(field_value, tuple)
                and len(field_value) == 2
                and isinstance(field_value[0], str)
                and isinstance(field_value[1], str)
            ):
                display_name = field_value[1]
                field_options[display_name] = field_value

        field_options = dict(sorted(field_options.items()))

        default_field = self.fields.density if "density" in field_options else next(iter(field_options.values()))

        self.cmap_widget = widgets.Text(
            value="plasma",
            description="Cmap:",
        )

        self.field_widget = widgets.Dropdown(
            options=field_options,
            value=default_field,
            description="Field:",
            layout=widgets.Layout(width="350px"),
        )
    

        # Remove unavailable fields from dropdown
        self.field_widget.options = {
            k: v for k, v in self.field_widget.options.items() if v is not None
        }

        self.zoom_widget = widgets.IntSlider(
            value=3,
            min=0,
            max=5,
            step=1,
            description="Zoom exp:",
            continuous_update=False,
        )

        self.annotate_velocity_widget = widgets.Checkbox(
            value=False,
            description="Velocity",
        )

        self.annotate_grids_widget = widgets.Checkbox(
            value=True,
            description="Grids",
        )

        self.Bstreamlines_widget = widgets.Checkbox(
            value=False,
            description="B streams",
        )

        self.timestamp_widget = widgets.Checkbox(
            value=True,
            description="Timestamp",
        )

        self.add_cycle_widget = widgets.Checkbox(
            value=True,
            description="Cycle text",
        )

        self.xcenter_widget = widgets.FloatText(
            value=0.0,
            description="x center:",
            layout=widgets.Layout(width="180px"),
        )

        self.ycenter_widget = widgets.FloatText(
            value=0.0,
            description="y center:",
            layout=widgets.Layout(width="180px"),
        )

        self.zcenter_widget = widgets.FloatText(
            value=0.0,
            description="z center:",
            layout=widgets.Layout(width="180px"),
        )

        self.use_center_widget = widgets.Checkbox(
            value=False,
            description="Use center",
        )

        self.width_x_widget = widgets.FloatText(
            value=5,
            description="width x:",
            layout=widgets.Layout(width="180px"),
        )

        self.width_y_widget = widgets.FloatText(
            value=1,
            description="width y:",
            layout=widgets.Layout(width="180px"),
        )

        self.use_width_widget = widgets.Checkbox(
            value=False,
            description="Use width",
        )

        self.use_limits_widget = widgets.Checkbox(
            value=False,
            description="Set zlim",
        )

        self.zlim_min_widget = widgets.FloatText(
            value=1e-6,
            description="field min:",
            layout=widgets.Layout(width="180px"),
        )

        self.zlim_max_widget = widgets.FloatText(
            value=1e10,
            description="field max:",
            layout=widgets.Layout(width="180px"),
        )
        
        self.log_field_widget = widgets.Checkbox(
            value=True,
            description="Log field",
        )

        self.reload_button = widgets.Button(
            description="Reload series",
            button_style="warning",
        )

        self.plot_button = widgets.Button(
            description="Plot",
            button_style="success",
        )

        self.output = widgets.Output()

    def _load_series(self, *, quiet=False):
        self.dictkey = self.dictkey_widget.value

        self.fileseries = grabFileSeries(
            self.directoryDict[self.dictkey],
            basename=self.basename,
            outputnum=self.output_type,
            extension="phdf",
        )

        self.ts = yt.DatasetSeries(self.fileseries)

        nfiles = len(self.fileseries)

        # Avoid slider trait weirdness while changing max/value.
        self._suppress_callbacks = True
        self.index_widget.max = max(nfiles - 1, 0)

        if self.index_widget.value > self.index_widget.max:
            self.index_widget.value = self.index_widget.max

        self._suppress_callbacks = False

        if quiet:
            return

        self._print_series_info()

    def _print_series_info(self):
        nfiles = len(self.fileseries)

        with self.output:
            clear_output(wait=True)
            print(f"Loaded {nfiles} files for dictkey='{self.dictkey}'")
            if nfiles > 0:
                print(f"First file: {self.fileseries[0]}")
                print(f"Last file:  {self.fileseries[-1]}")

    def _on_dictkey_changed(self, change):
        if getattr(self, "_suppress_callbacks", False):
            return

        if change["old"] == change["new"]:
            return

        self._load_series()

    def _on_reload_clicked(self, button):
        self._load_series()

    def _on_plot_clicked(self, button):
        self.plot()

    def _get_center(self):
        if not self.use_center_widget.value:
            return None

        return (
            self.xcenter_widget.value,
            self.ycenter_widget.value,
            self.zcenter_widget.value,
        )

    def _get_width(self):
        if not self.use_width_widget.value:
            return None

        # This assumes your standard_slice helper accepts width=(wx, wy)
        # in code units or whatever default units it expects.
        return (
            self.width_x_widget.value,
            self.width_y_widget.value,
        )

    def plot(self):
        with self.output:
            clear_output(wait=True)

            index = self.index_widget.value
            print(f"Loading index {index} from run '{self.dictkey_widget.value}'")

            self.ds = self.ts[index]
            try:
                add_timescale_fields(self.ds)
            except:
                print("Fail on adding timescale fields; proceeding without them.")
            field = self.field_widget.value
            axis = self.axis_widget.value
            center = self._get_center()
            width = self._get_width()

            kwargs = dict(
                axis=axis,
                zoom_exp=self.zoom_widget.value,
                annotate_velocity=self.annotate_velocity_widget.value,
                annotate_grids=self.annotate_grids_widget.value,
                Bstreamlines=self.Bstreamlines_widget.value,
                cmap=self.cmap_widget.value,
                annotate_timestamp=self.timestamp_widget.value,
                timestamp_textargs={"color": "white"},
                set_log=self.log_field_widget.value,
                zlim=[None, None],
            )
            
            if self.use_limits_widget.value:
                kwargs["zlim"] = [self.zlim_min_widget.value, self.zlim_max_widget.value]

            if center is not None:
                kwargs["center"] = center

            if width is not None:
                kwargs["width"] = width

            p = standard_slice(
                self.ds,
                field,
                **kwargs,
            )

            if self.add_cycle_widget.value:
                p.annotate_text(
                    [0.03, 0.1],
                    f"cycle={get_cycle(self.ds)}",
                    coord_system="axis",
                    text_args={"color": "white"},
                )

            p.show()

    def show(self):
        controls_top = widgets.HBox([
            self.dictkey_widget,
            self.reload_button,
            self.plot_button,
            self.timestamp_widget,
            self.add_cycle_widget,
        ])

        controls_mid = widgets.HBox([
            self.index_widget,
            self.axis_widget,
            self.field_widget,
            self.cmap_widget,
        ])

        controls_opts = widgets.HBox([
            self.zoom_widget,
            self.annotate_velocity_widget,
            self.annotate_grids_widget,
            self.Bstreamlines_widget,
        ])

        center_controls = widgets.HBox([
            self.use_center_widget,
            self.xcenter_widget,
            self.ycenter_widget,
            self.zcenter_widget,
        ])

        width_controls = widgets.HBox([
            self.use_width_widget,
            self.width_x_widget,
            self.width_y_widget,
        ])

        zlim_controls = widgets.HBox([
            self.use_limits_widget,
            self.zlim_min_widget,
            self.zlim_max_widget,
            self.log_field_widget,
        ])

        self.ui = widgets.VBox([
            controls_top,
            controls_mid,
            controls_opts,
            center_controls,
            width_controls,
            zlim_controls,
            self.output,
        ])

        display(self.ui)
        self._print_series_info()


def get_cycle(input):
    # Handle filename or parameter file. Check if input is string
    if isinstance(input, str):
        name = input
    else:
        try:
            # assume input is a yt dataset:
            name = input.parameter_filename
        except AttributeError:
            raise ValueError("Input must be a filename string or a yt dataset with a parameter_filename attribute.")
    with h5py.File(name, "r") as f:
        cycle = f["Info"].attrs["NCycle"]
    return cycle


def standard_slice(ds, field, 
                   cmap="plasma", 
                   axes_unit="mm", 
                   set_width=None, 
                   set_width_units=None, 
                   annotate_grids=True, 
                   annotate_timestamp=True, 
                   timestamp_format='t = {time:.5f} {units}',
                   timestamp_textargs={"color": "black"},
                   annotate_velocity=True, 
                   Bstreamlines=True, 
                   vstreamlines=False, 
                   streamlinecolor="black",
                   streamlinewidth=0.75,
                   pan=[0,0], 
                   zoom_exp=0, 
                   set_log=True, 
                   zlim=[None, None],
                   axis="z",
                   center=[0,0,0],
                   width=None,
                   width_units="cm",
                   annotate_cycle=False,
                ):
    center = ds.arr(center, "cm")

    if width is None:
        p = yt.SlicePlot(ds, axis, field, center=center)
    else:
        width = ds.arr(width, width_units)
        p = yt.SlicePlot(
            ds,
            axis,
            field,
            center=center,
            width=width,
        )
        
    p.set_origin("native")
    if zoom_exp != 0 and width is not None:
        print("WARN: Both zoom_exp and width are set. Zoom will be applied after width, so the final width is not the user-set width at function call.")
    p.zoom(2**zoom_exp)
    p.pan((pan[0], pan[1]))
    p.set_cmap(field, cmap)
    p.set_axes_unit(axes_unit)

    if set_log: 
        # Before setting log_field, check for negative values:
        ad = ds.all_data()
        if np.any(ad[field[0], field[1]] <= 0):
            print(f"Warning: Field {field} contains non-positive values. Cannot set log scale. Setting to linear scale instead.")
            p.set_log(field, False)
        else:
            p.set_log(field, True)
    else:
        p.set_log(field, False)

    if set_width is not None:
        p.set_width(set_width, set_width_units)
    if annotate_grids:
        p.annotate_grids()
    if annotate_timestamp:
        p.annotate_timestamp(time_format=timestamp_format, text_args=timestamp_textargs)
    if annotate_cycle:
        p.annotate_text([0.03, 0.1], f"cycle={get_cycle(ds)}", coord_system="axis", text_args={"color": "white"})
    if annotate_velocity:
        p.annotate_velocity()
    if Bstreamlines:
        if axis == "z":
            p.annotate_streamlines(("gas", "magnetic_field_x"), ("gas", "magnetic_field_y"), color=streamlinecolor, linewidth=streamlinewidth)
        elif axis == "y":
            p.annotate_streamlines(("gas", "magnetic_field_x"), ("gas", "magnetic_field_z"), color=streamlinecolor, linewidth=streamlinewidth)
        elif axis == "x":
            p.annotate_streamlines(("gas", "magnetic_field_y"), ("gas", "magnetic_field_z"), color=streamlinecolor, linewidth=streamlinewidth)
    if vstreamlines:
        p.annotate_streamlines(("gas", "velocity_x"), ("gas", "velocity_y"), color=streamlinecolor, linewidth=streamlinewidth)
    if zlim != [None, None]:
        p.set_zlim(field, zlim[0], zlim[1])
    if field[1] == "ohmic_over_cooling_timescale_ratio": p.set_colorbar_label(field, r"$\tau_{\mathrm{heating}} / \tau_{\mathrm{cooling}}$")
    if field[1] == "cooling_timescale": p.set_colorbar_label(field, r"$\tau_{\mathrm{cooling}}$ (s)")
    if field[1] == "ohmic_heating_timescale": p.set_colorbar_label(field, r"$\tau_{\mathrm{\eta}}$ (s)")
    if field[1] == "adiabatic_heating_timescale": p.set_colorbar_label(field, r"$\tau_{\Pi} = 1/|\left(\left(\gamma-1\right)\nabla\cdot v\right)|$ (s)")
    if field[1] == "adiabatic_over_cooling_timescale_ratio": p.set_colorbar_label(field, r"$\tau_{\mathrm{\Pi}} / \tau_{\mathrm{cooling}}$")
    if field[1] == "adiabatic_over_ohmic_timescale_ratio": p.set_colorbar_label(field, r"$\tau_{\mathrm{\Pi}} / \tau_{\mathrm{\eta}}$")
    return p

def free_free_loss(T, ni, Zbar=3):
    """
    Here, T is in K, ne is in cm^-3, ni is in cm^-3, and Z is the charge state of the ions. The output power density is in erg cm^-3 s^-1.
    """
    # At first, assume ne = Zbar*ni, which is true for a OCP
    ne = Zbar * ni
    power_density = 1.426e-27 * 1 * Zbar**2 * ne * ni * np.sqrt(T)
    return power_density / ne / ni  # returning power density per ne*ni to generate volumetric cooling tables

def _cooling_timescale(field, data):
    T = data["gas", "temperature"]
    rho = data["gas", "density"]
    P = data["gas", "pressure"]
    gamma = data.ds.gamma
    m_u = data.ds.units.atomic_mass_unit
    A = 27.0
    Zbar = 3.0

    # specific internal energy: erg / g
    e_spec = P / (rho * (gamma - 1.0))

    # Lambda(T) for free-free, with units erg cm^3 / s
    Lambda = 1.426e-27 * Zbar**2 * np.sqrt(T) * data.ds.units.erg * data.ds.units.cm**3 /data.ds.units.s

    # ion and electron densities
    n_i = rho / (A * m_u)
    n_e = Zbar * n_i

    # specific cooling rate: erg / g / s
    dedt_spec = (Lambda * n_e**2) / rho

    return np.abs(e_spec / dedt_spec)



def _ohmic_heating_timescale(field, data):
    curlBz = data["parthenon", "curlBz"]

    jz = (
        curlBz
        * np.sqrt(4 * np.pi)
        * data.ds.units.G
        / data.ds.units.cm
    )

    eta = (
        data["parthenon", "eta"]
        * data.ds.units.cm**2
        / data.ds.units.s
    )

    de_dt = eta * jz**2
    IEN = data["gas", "pressure"] / (data.ds.gamma - 1)

    tau = IEN / de_dt

    curl_floor = 1.0e-300
    bad = (
        (np.abs(curlBz) <= curl_floor)
        | (de_dt <= 0.0)
        | ~np.isfinite(tau)
    )

    return np.where(bad, np.nan * tau.units, tau)

def _adiabatic_heating_timescale(field, data):
    # Uses convention from Sen & Moreno-Insertis eq 12 and 15 where tau_Pi = 1/ abs((gamma-1)*div(v)) 
    # where div(v) is stored in ("parthenon", "divv")
    gm1 = data.ds.gamma - 1
    divv = data["parthenon", "divv"] * data.ds.units.s**(-1)  # convert to 1/s
    tau_Pi = 1 / np.abs(gm1 * divv)
    return tau_Pi

def _ohmic_over_cooling_timescale_ratio(field, data):
    tau_heating = data["gas", "ohmic_heating_timescale"]
    tau_cooling = data["gas", "cooling_timescale"]
    return tau_heating / tau_cooling

def _adiabatic_over_cooling_timescale_ratio(field, data):
    tau_heating = data["gas", "adiabatic_heating_timescale"]
    tau_cooling = data["gas", "cooling_timescale"]
    return tau_heating / tau_cooling

def _adiabatic_over_ohmic_timescale_ratio(field, data):
    tau_adiabatic = data["gas", "adiabatic_heating_timescale"]
    tau_ohmic = data["gas", "ohmic_heating_timescale"]
    return tau_adiabatic / tau_ohmic


def add_timescale_fields(ds):
    ds.add_field(
        ("gas", "adiabatic_heating_timescale"), 
        units=None,
        function=_adiabatic_heating_timescale, 
        sampling_type="cell",
        force_override=True)

    ds.add_field(
        ("gas", "cooling_timescale"),
        units=None,
        function=_cooling_timescale,
        sampling_type="cell",
    )

    ds.add_field(
        ("gas", "ohmic_heating_timescale"),
        units=None,
        function=_ohmic_heating_timescale,
        sampling_type="cell",
    )

    ds.add_field(
        ("gas", "ohmic_over_cooling_timescale_ratio"),
        units=None,
        function=_ohmic_over_cooling_timescale_ratio,
        sampling_type="cell",
    )

    ds.add_field(
        ("gas", "adiabatic_over_cooling_timescale_ratio"),
        units=None,
        function=_adiabatic_over_cooling_timescale_ratio,
        sampling_type="cell",
    )

    ds.add_field(
        ("gas", "adiabatic_over_ohmic_timescale_ratio"),
        units=None,
        function=_adiabatic_over_ohmic_timescale_ratio,
        sampling_type="cell",
    )

def extract_1d_line(hdf5file, line_axis, pos_on_axis, requested_field_name, bounds=[-np.inf, np.inf], suppress_warnings=False, path_elements=False):
    """
    Extracts a 1D line of requested field data from an AthenaPK HDF5 output file along a specified axis at given positions.

    Parameters
    ----------
    hdf5file : str
        Path to the HDF5 file.
    line_axis : str
        Axis along which to extract the line ('x', 'y', or 'z').
    pos_on_axis : list of float
        2d positions in the plane perpendicular to line_axis where the line is extracted.
        For example, if line_axis is 'y', pos_on_axis could be [x0, z0].
    requested_field_name : str
        Name of the field to extract, field must be present in .phdf file
        Current Native (code outputs) options are: 
            "T", "curlBx_gauss", "curlBy_gauss", "curlBz_gauss", "beta", "eta", "prim_density",
            "prim_magnetic_field_1", "prim_magnetic_field_2", "prim_magnetic_field_3",
            "prim_pressure", "prim_velocity_1", "prim_velocity_2", "prim_velocity_3", "prim_magnetic_psi"
            "magnetic_field_1_gauss", "magnetic_field_2_gauss", "magnetic_field_3_gauss",
    bounds : list of float, optional
        [min, max] bounds along the line_axis to include in the output. Default is [-inf, inf].
    suppress_warnings : bool, optional
        If True, suppresses warning messages when no blocks are found at the specified position. Default is False.
    path_elements : bool, optional
        If True, output cell dx, dy, dz along with field data. Default is False.

    Returns
    -------
    pos_1d : ndarray
        1D array of positions along the specified line axis.
    field_1d : ndarray
        1D array of the requested field values along the specified line axis.    
    """
    with h5py.File(hdf5file, 'r') as hdf:
        # keys of APK files: ['Blocks', 'Info', 'Input', 'Levels', 'Locations', 'LogicalLocations', 
            # 'Params', 'VolumeLocations', 'beta', 'curlBx', 'curlBy', 'curlBz', 'eta', 'prim']
        VolumeLocations = hdf['VolumeLocations']
        EdgeLocations = hdf['Locations']
        prim = hdf['prim']
        # to access values in prim, use the following indices:
        prim_density, prim_velocity_1, prim_velocity_2, prim_velocity_3 = 0, 1, 2, 3
        prim_pressure, prim_magnetic_field_1, prim_magnetic_field_2, prim_magnetic_field_3, prim_magnetic_psi = 4, 5, 6, 7, 8

        # Switch case for field selection
        match requested_field_name:
            case "T":
                requested_field = hdf['T'][:, 0, :, :, :]
            case "curlBx_gauss":
                requested_field = hdf['curlBx'][:, 0, :, :, :] * np.sqrt(4*np.pi)    # code units to gauss
            case "curlBy_gauss":
                requested_field = hdf['curlBy'][:, 0, :, :, :] * np.sqrt(4*np.pi)    # code units to gauss
            case "curlBz_gauss": 
                requested_field = hdf['curlBz'][:, 0, :, :, :] * np.sqrt(4*np.pi)    # code units to gauss
            case "beta":
                requested_field = hdf['beta'][:, 0, :, :, :]
            case "eta":
                requested_field = hdf['eta'][:, 0, :, :, :]
            case "prim_density":
                requested_field = prim[:, prim_density, :, :, :]
            case "prim_magnetic_field_1":
                requested_field = prim[:, prim_magnetic_field_1, :, :, :]
            case "prim_magnetic_field_2":
                requested_field = prim[:, prim_magnetic_field_2, :, :, :]
            case "prim_magnetic_field_3":
                requested_field = prim[:, prim_magnetic_field_3, :, :, :]
            case "prim_pressure":
                requested_field = prim[:, prim_pressure, :, :, :]
            case "prim_velocity_1":
                requested_field = prim[:, prim_velocity_1, :, :, :]
            case "prim_velocity_2":
                requested_field = prim[:, prim_velocity_2, :, :, :]
            case "prim_velocity_3":
                requested_field = prim[:, prim_velocity_3, :, :, :]
            case "prim_magnetic_psi":
                requested_field = prim[:, prim_magnetic_psi, :, :, :]
            case "magnetic_field_1_gauss":
                requested_field = prim[:, prim_magnetic_field_1, :, :, :] * np.sqrt(4 * np.pi)    # code units to gauss
            case "magnetic_field_2_gauss":
                requested_field = prim[:, prim_magnetic_field_2, :, :, :] * np.sqrt(4 * np.pi)    # code units to gauss
            case "magnetic_field_3_gauss":
                requested_field = prim[:, prim_magnetic_field_3, :, :, :] * np.sqrt(4 * np.pi)    # code units to gauss
            case _:
                raise ValueError(f"Requested field '{requested_field_name}' not recognized.")
            
        # Getting perpendicular axes to detect when a block intersects the line
        perpendicular_axes = [ax for ax in ['x', 'y', 'z'] if ax != line_axis]
        # Iterate through the blocks, and find the blocks that intersect the chosen line
        blocks_on_line = []
        for b in range(VolumeLocations[line_axis].shape[0]):    # line_axis is just a representative field to get the size
            # store the block's bounding data along line of interest
            x1_b = VolumeLocations[perpendicular_axes[0]][b, :]   # Grab first perpendicular coordinates
            x2_b = VolumeLocations[perpendicular_axes[1]][b, :]   # Grab second perpendicular coordinates
            if min(x1_b) <= pos_on_axis[0] < max(x1_b) and min(x2_b) <= pos_on_axis[1] < max(x2_b):
                blocks_on_line.append(b)
        if len(blocks_on_line) == 0:
            if not suppress_warnings: print(f"No blocks found with cell centers along requested axis {line_axis} at position {pos_on_axis} in file .../{hdf5file[:-40]} for {requested_field_name}. Trying again with cell faces.", flush=True)
            for b in range(EdgeLocations[line_axis].shape[0]):    # line_axis is just a representative field to get the size
                # store the block's bounding data along line of interest
                x1_b = EdgeLocations[perpendicular_axes[0]][b, :]   # Grab first perpendicular coordinates
                x2_b = EdgeLocations[perpendicular_axes[1]][b, :]   # Grab second perpendicular coordinates
                if min(x1_b) <= pos_on_axis[0] < max(x1_b) and min(x2_b) <= pos_on_axis[1] < max(x2_b):
                    blocks_on_line.append(b)
        if len(blocks_on_line) == 0:
            raise ValueError(f"No blocks found with cell centers or faces along requested axis {line_axis} at position {pos_on_axis}in file .../{hdf5file[:-40]} for {requested_field_name}.")
                
        # Sorting the blocks for iteration:
        blocks_on_line = sorted(blocks_on_line, key=lambda b: VolumeLocations[line_axis][b, 0])
        # Calculate the size of the arrays to pass back
        size_1d = len(blocks_on_line) * (VolumeLocations[line_axis].shape[1])

        # preallocate space for position array and desired array
        pos_1d = np.zeros(size_1d)
        field_1d = np.zeros(size_1d)
        mask = np.zeros(size_1d, dtype=bool)        # Sometimes we don't return whole blocks: mask out these values
        # Only preallocate space for path elements if defined above
        if path_elements:   
            dx_1d = np.zeros(size_1d)
            dy_1d = np.zeros(size_1d)
            dz_1d = np.zeros(size_1d)

        # Loop over blocks that intersect the line
        for n, b in enumerate(blocks_on_line):
            # Defining coordinate arrays INSIDE the block
            x1_b = VolumeLocations[line_axis][b, :]                
            x2_b = VolumeLocations[perpendicular_axes[0]][b, :]
            x3_b = VolumeLocations[perpendicular_axes[1]][b, :]
            # Grabbing nearest indices to the requested position on the perpendicular axes
            x2i = find_nearest(x2_b, pos_on_axis[0])[0]
            x3i = find_nearest(x3_b, pos_on_axis[1])[0]
            # Grabbing the field values along the line in this block
            offset = n * (len(x1_b))    # offset where the field values will be placed in the 1D array

            # If path elements are desired, define block-specific dx, dy, dz:
            if path_elements:
                # Each dx, dy, dz is uniform across the block
                dx_b = EdgeLocations['x'][b, 1] - EdgeLocations['x'][b, 0]
                dy_b = EdgeLocations['y'][b, 1] - EdgeLocations['y'][b, 0]
                dz_b = EdgeLocations['z'][b, 1] - EdgeLocations['z'][b, 0]
                
            # Iterate through the cells along the line axis in this block
            for x1i in range(len(x1_b)):
                # Only adding to array if within bounds:
                if bounds[0] <= VolumeLocations[line_axis][b, x1i] <= bounds[1]:
                    match line_axis:
                        case "x":
                            pos_1d[offset + x1i] = VolumeLocations[line_axis][b, x1i]
                            field_1d[offset + x1i] = requested_field[b, x3i, x2i, x1i]
                            mask[offset + x1i] = True
                        case "y":
                            pos_1d[offset + x1i] = VolumeLocations[line_axis][b, x1i]
                            field_1d[offset + x1i] = requested_field[b, x3i, x1i, x2i]
                            mask[offset + x1i] = True
                        case "z":
                            pos_1d[offset + x1i] = VolumeLocations[line_axis][b, x1i]
                            field_1d[offset + x1i] = requested_field[b, x1i, x3i, x2i]
                            mask[offset + x1i] = True
                    # If path elements are desired, store them as well
                    if path_elements:
                        dx_1d[offset + x1i] = dx_b
                        dy_1d[offset + x1i] = dy_b
                        dz_1d[offset + x1i] = dz_b
            # end block loop
        # end hdf5 loop
    # Trim leading and trailing zeros in case no data was added in some blocks
    if path_elements:
        return pos_1d[mask], field_1d[mask], dx_1d[mask], dy_1d[mask], dz_1d[mask]
    else:
        return pos_1d[mask], field_1d[mask]

def find_nearest(array, value):
    """
    Finds nearest value in a 1d array to the input value

    Parameters
    ----------
    array : ndarray
        1D array to search
    value : float
        Value to search for

    Returns
    -------
    idx : int
        Index of nearest value in array
    nearest_value : float
        Nearest value in array

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def collapse_refinement(arr1d, pos1d, epsilon=1e-8):
    """
    Collapse regions of constant values in a refined 1D dataset.

    Parameters
    ----------
    arr1d : ndarray
        1D array of values (e.g., field data).
    pos1d : ndarray
        1D array of corresponding positions.
    epsilon : float
        Threshold for considering two values different.

    Returns
    -------
    fout : list
        Collapsed field values.
    xout : list
        Mean position of each collapsed region.
    lout : list
        Length of each collapsed region in number of original cells.
    """
    if arr1d.shape != pos1d.shape:
        raise ValueError("arr1d and pos1d must have the same shape")

    fout, xout, lout = [], [], []
    istartblock = 0

    for i in range(len(arr1d) - 1):
        if np.abs(arr1d[i] - arr1d[i + 1]) > epsilon:
            fout.append(arr1d[i])
            xout.append(np.mean(pos1d[istartblock:i + 1]))
            lout.append(i + 1 - istartblock)
            istartblock = i + 1

    # Final block
    fout.append(arr1d[-1])
    xout.append(np.mean(pos1d[istartblock:]))
    lout.append(len(arr1d) - istartblock)

    return fout, xout, lout


def derivative1d_on_mesh(arr1d, pos1d, epsilon=1e-8):
    """
    Compute the derivative of a 1D function sampled at unevenly spaced points 
    using a cubic spline over collapsed refinement regions.

    Parameters
    ----------
    arr1d : ndarray
        The array of data values.
    pos1d : ndarray
        The positions corresponding to arr1d.
    epsilon : float, optional
        Tolerance used in collapse_refinement to identify flat regions.

    Returns
    -------
    deriv : ndarray
        The derivative of arr1d evaluated at pos1d.
    """
    fout, xout, lout = collapse_refinement(arr1d, pos1d, epsilon)
    cs = CubicSpline(xout, fout)
    return cs(pos1d, 1)

def dFdx_mesh(arr2d, pos1d, epsilon=1e-8):
    """
    Compute the partial derivative ∂F/∂x across a 2D array of values sampled 
    on a potentially uneven mesh in the x-direction (columns).

    Parameters
    ----------
    arr2d : ndarray
        2D array representing the function values on a 2D mesh.
    pos1d : ndarray
        1D array representing the physical x-positions of the columns.
    epsilon : float, optional
        Tolerance for identifying uniform regions in the refinement collapse.

    Returns
    -------
    dFdx : ndarray
        2D array of partial derivatives with respect to x.
    """
    dFdx = np.zeros_like(arr2d)
    for j in range(len(arr2d[0, :])):
        dFdx[:, j] = derivative1d_on_mesh(arr2d[:, j], pos1d, epsilon)
    return dFdx

def dFdy_mesh(arr2d, pos1d, epsilon=1e-8):
    """
    Compute the partial derivative ∂F/∂y across a 2D array of values sampled 
    on a potentially uneven mesh in the y-direction (rows).

    Parameters
    ----------
    arr2d : ndarray
        2D array representing the function values on a 2D mesh.
    pos1d : ndarray
        1D array representing the physical y-positions of the rows.
    epsilon : float, optional
        Tolerance for identifying uniform regions in the refinement collapse.

    Returns
    -------
    dFdy : ndarray
        2D array of partial derivatives with respect to y.
    """
    dFdy = np.zeros_like(arr2d)
    for i in range(len(arr2d[:, 0])):
        dFdy[i, :] = derivative1d_on_mesh(arr2d[i, :], pos1d, epsilon)
    return dFdy


def _current_eta_z(field, data):
    Jz = data["gas", "velocity_x"]*data["gas", "magnetic_field_y"] - data["gas", "velocity_y"]*data["gas", "magnetic_field_x"]
    return Jz
yt.add_field(
    name=("gas", "current_eta_z"),
    function=_current_eta_z,
    sampling_type="local",
    units="code_magnetic*code_velocity",
    force_override=True,
)

def dFdx_1d_non_U_grid(j, x):
    F = np.zeros_like(j)
    i = 0
    while i < len(F) and i != -1:
        ii = i
        while np.abs(j[ii] - j[i]) < 1e-6:
            ii = np.min([ii + 1, len(j)-1])
            # print(i, ii, np.min([ii + 1, len(j)]))
            f0 = j[i]
            if ii == len(j)-1:
                ii = -1
                break
        f1 = j[ii]
        dx = x[ii] - x[i]
        F[i:ii] = (f1 - f0) / dx
        i = ii
    return F

def dFdy_1d_non_U_grid(j, y):
    F = np.zeros_like(j)
    i = 0
    while i < len(F) and i != -1:
        ii = i
        while np.abs(j[ii] - j[i]) < 1e-6:
            ii = np.min([ii + 1, len(j)-1])
            # print(i, ii, np.min([ii + 1, len(j)]))
            f0 = j[i]
            if ii == len(j)-1:
                ii = -1
                break
        f1 = j[ii]
        dy = y[ii] - y[i]
        F[i:ii] = (f1 - f0) / dy
        i = ii
    return F

def rolling_average(f, a):
    out = np.copy(f)
    for idx, x in enumerate(f[a:-a]):
        i = idx + a
        out[i] = np.mean(f[i-a:i+a])
    return out
        

def div2D(Fx, Fy, dx, dy):
    return dFdx(Fx, dx) + dFdy(Fy, dy)

def dFdx(F, x):
    """
    Compute the derivative of F (2d or 1d) with respect to x, where x is a 1D array
    representing non-uniform grid positions along axis 0 (rows of F).

    Parameters:
    -----------
    F : np.ndarray
        2D array of shape (Nx, Ny)
    x : np.ndarray
        1D array of length Nx, giving grid positions along x-axis

    Returns:
    --------
    dFdx : np.ndarray
        Approximate derivative ∂F/∂x, same shape as F
    """
    was_1d = (F.ndim == 1)
    if was_1d:
        F = F[:, np.newaxis]
    Nx, Ny = F.shape
    dFdx = np.zeros_like(F)

    # Interior points: 3-point non-uniform central difference
    for i in range(1, Nx - 1):
        x0, x1, x2 = x[i-1], x[i], x[i+1]
        f0, f1, f2 = F[i-1], F[i], F[i+1]

        dx0 = x1 - x0
        dx1 = x2 - x1

        # Coefficients derived from Lagrange interpolation polynomial
        a = -dx1 / (dx0 * (dx0 + dx1))
        b = (dx1 - dx0) / (dx0 * dx1)
        c =  dx0 / (dx1 * (dx0 + dx1))

        dFdx[i, :] = a * f0 + b * f1 + c * f2

    # Forward difference at the first point
    dx = x[1] - x[0]
    dFdx[0, :] = (F[1, :] - F[0, :]) / dx

    # Backward difference at the last point
    dx = x[-1] - x[-2]
    dFdx[-1, :] = (F[-1, :] - F[-2, :]) / dx

    if was_1d:
        return dFdx[:, 0]
    else:
        return dFdx
    
    
import numpy as np

def dFdy(F, y):
    """
    Compute the derivative of F with respect to y, where y is a 1D array
    representing non-uniform grid positions along axis 1 (columns of F).

    Parameters:
    -----------
    F : np.ndarray
        2D array of shape (Nx, Ny)
    y : np.ndarray
        1D array of length Ny, giving grid positions along y-axis

    Returns:
    --------
    dFdy : np.ndarray
        Approximate derivative ∂F/∂y, same shape as F
    """
    Nx, Ny = F.shape
    dFdy = np.zeros_like(F)

    # Interior points: 3-point non-uniform central difference
    for j in range(1, Ny - 1):
        y0, y1, y2 = y[j - 1], y[j], y[j + 1]
        dy0 = y1 - y0
        dy1 = y2 - y1
        denom = dy0 * dy1 * (dy0 + dy1)

        a = -(2 * dy1 + dy0) / (dy0 * (dy0 + dy1))
        b = (dy1 - dy0) / (dy0 * dy1)
        c = (2 * dy0 + dy1) / (dy1 * (dy0 + dy1))

        dFdy[:, j] = a * F[:, j - 1] + b * F[:, j] + c * F[:, j + 1]

    # Forward difference at the first column
    dy0 = y[1] - y[0]
    dFdy[:, 0] = (F[:, 1] - F[:, 0]) / dy0

    # Backward difference at the last column
    dy1 = y[-1] - y[-2]
    dFdy[:, -1] = (F[:, -1] - F[:, -2]) / dy1

    return dFdy



def draw_xy_box(p, xmin, xmax, ymin, ymax):
    """Draws a two dimensional box in the xy plane of an xy slice plot p
    
    Args:
        p (yt plot): plot to draw a box on
        xmin, ymin (floats): bottom left corner of box (coord_system="data")
        xmax, ymax (floats): top right corner of box (coord_system="data")
    """
    p.annotate_line((xmin, ymin,0), (xmax, ymin,0), coord_system="data")
    p.annotate_line((xmax, ymin,0), (xmax, ymax,0), coord_system="data")
    p.annotate_line((xmax, ymax,0), (xmin, ymax,0), coord_system="data")
    p.annotate_line((xmin, ymax,0), (xmin, ymin,0), coord_system="data")

def change_in_box(quantity, u, v, dx, dy, dz, dt):
    """Calculate the integral of the flux through a bounding box in the x- and y- direction

    Args:
        quantity (numpy.ndarray): 2D quantity to calculate flux through (e.g. Energy)
        u (numpy.ndarray): 2D x-velocity in the box
        v (numpy.ndarray): 2D y-velocity in the box
        dx (float): grid size in x-direction
        dy (float): grid size in y-direction
        dz (float): grid size in z-direction
        dt (float): timestep
    """
    # Left edge of the box:
    vn1 = u[0,:]    # velocity normal to edge at edge cell
    vn2 = u[0,:]    # velocity normal to edge next to edge
    vn = (vn1 + vn2)/2      # Average (at cell face, one cell inside the box)
    Q1 = quantity[0,:]      # Quantity at edge cell
    Q2 = quantity[1,:]      # Quantity next to edge
    Q = (Q1 + Q2)/2         # Average (at cell face, one cell inside the box)
    da = dy*dz              # da of edge
    flux = np.sum(vn*Q*da)     # quantity change per time into edge
    total_change_left = flux*dt     # total quantity *into* edge
    
    # Bottom edge of the box:
    vn1 = v[:,0]    # velocity normal to edge at edge cell
    vn2 = u[:,1]    # velocity normal to edge next to edge
    vn = (vn1 + vn2)/2      # Average (at cell face, one cell inside the box)
    Q1 = quantity[:,0]      # Quantity at edge cell
    Q2 = quantity[:,1]      # Quantity next to edge
    Q = (Q1 + Q2)/2         # Average (at cell face, one cell inside the box)
    da = dx*dz      # da of edge
    flux = np.sum(vn*Q*da)     # quantity change per time into edge
    total_change_bottom = flux*dt     # total quantity *into* edge
    
    # Top edge of the box:
    vn1 = v[:,-1]    # velocity normal to edge at edge cell
    vn2 = v[:,-2]    # velocity normal to edge next to edge
    vn = (vn1 + vn2)/2      # Average (at cell face, one cell inside the box)
    Q1 = quantity[:,-1]      # Quantity at edge cell
    Q2 = quantity[:,-2]      # Quantity next to edge
    Q = (Q1 + Q2)/2         # Average (at cell face, one cell inside the box)
    da = dx*dz      # da of edge
    flux = np.sum(vn*Q*da)     # quantity change per time into edge
    total_change_top = flux*dt     # total quantity *into* edge
    
    # Right edge of the box:
    vn1 = u[-1,:]    # velocity normal to edge at edge cell
    vn2 = u[-2,:]    # velocity normal to edge next to edge
    vn = (vn1 + vn2)/2      # Average (at cell face, one cell inside the box)
    Q1 = quantity[-1,:]      # Quantity at edge cell
    Q2 = quantity[-2,:]      # Quantity next to edge
    Q = (Q1 + Q2)/2         # Average (at cell face, one cell inside the box)
    da = dy*dz      # da of edge
    flux = np.sum(vn*Q*da)     # quantity change per time into edge
    total_change_right = flux*dt     # total quantity *into* edge
    
    # Notice the sign for the right and top edge, because those da point INTO the box, i.e. -x, -y at right and top
    return total_change_bottom + total_change_left - total_change_right - total_change_top


import h5py

def get_simulation_time(hdf5_file):
    """
    Extracts the simulation time from an Athena++ or AthenaPK HDF5 output file.

    Parameters:
    ----------
    hdf5_file : str
        Path to the HDF5 file.

    Returns:
    -------
    float or None
        The simulation time if found, or None if the time is not available or an error occurs.
    """
    try:
        with h5py.File(hdf5_file, 'r') as hdf:
            # Try root-level attribute (Athena++)
            if 'Time' in hdf.attrs:
                return hdf.attrs['Time']
            # Try 'Info' group attribute (AthenaPK)
            elif 'Info' in hdf and 'Time' in hdf['Info'].attrs:
                return hdf['Info'].attrs['Time']
            else:
                print(f"'Time' attribute not found in root or 'Info' group: {hdf5_file}.")
    except Exception as e:
        print(f"Error while reading HDF5 file '{hdf5_file}': {e}")
    
    return None


def parse_input_file(filename: str | Path, scratchPath="/mnt/gs21/scratch/freem386/", suppress_warnings=True) -> Dict[str, Any]:
    """
    Parse an Athena++/Athena-PK style input file. If a direct path to file is not provided,
    the function will assume filename is a path, and look for file ending in ".in" in that directory.

    Returns
    -------
    dict
        Keys are "<block>_<variable>" (both lower-cased, no spaces),
        values are int, float, or str depending on what parses cleanly.
    """
    params: Dict[str, Any] = {}
    current_block = "global"                      # fallback for lines outside any block
    # Check if filename is a file; if not, look for .in file in the directory
    if not os.path.isfile(filename):
        searchdir = os.path.join(scratchPath, filename)
        for file in os.listdir(searchdir):
            if file.endswith(".in"):
                filename = os.path.join(searchdir, file)
                break
        # Tell the user
        if not suppress_warnings:
            print("Info: No direct input file provided, using", filename, "instead.")
        # If still not a file, raise error
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No input file found at {filename} or in directory.")
    with open(filename, "r") as fp:
        for raw in fp:
            line = _comment_re.sub("", raw).strip()   # drop comments, whitespace
            if not line:
                continue                             # blank line → skip

            # ─── Block header? ─────────────────────────────────────────────
            block_match = _block_re.fullmatch(line)
            if block_match:
                current_block = block_match.group(1).strip().lower()
                continue

            # ─── key = value line? ────────────────────────────────────────
            if "=" not in line:
                continue                             # ignore lines with no '='

            var, val = (x.strip() for x in line.split("=", 1))

            # best-effort numeric conversion
            try:
                # int() will also parse hex/octal if prefixed (0x, 0o, 0b)
                val_parsed: Any = int(val, 0)
            except ValueError:
                try:
                    val_parsed = float(val)
                except ValueError:
                    val_parsed = val                       # keep raw string

            key = f"{current_block}_{var}".lower()
            params[key] = val_parsed

    return params

def grabFileSeries(
    scratchdirectory,
    fn=None,
    basename="output_name",
    f0=0,
    step=1,
    width=5,
    scratchPath="/mnt/gs21/scratch/freem386/",
    outputnum="out2",
    extension="athdf"
):
    """
    Generate a list of file paths for a series of Athena++ .athdf files.

    If `fn` is None, attempts to discover the largest index in the directory by
    matching any file that follows the pattern:
        {basename}.{outputnum}.{index}.athdf

    Parameters
    ----------
    scratchdirectory : str
        Name of the scratch directory where the files are located.
    fn : int or None, optional
        The final index for the series. If None, automatically discovers the
        largest index from existing files in the directory. Default is None.
    basename : str, optional
        The base name of the files (e.g., "mySimulation"). Default is "mySim".
    f0 : int, optional
        The starting index for the series. Default is 0.
    step : int, optional
        The step size between successive indices. Default is 1.
    width : int, optional
        The number of digits to which the index is zero-padded. Default is 5.
    scratchPath : str, optional
        The full path to the scratch directory. Default is "/mnt/gs21/scratch/freem386/".
    outputnum : str, optional
        The output identifier that appears in the file name (e.g., "out2"). Default is "out2".
    extension : str, optional
        the output extension

    Returns
    -------
    list of str
        A list of file paths matching the specified pattern, each ending with "extension".
    """
    
    # If fn is None, find the max index by scanning the directory for matching files
    if fn is None:
        dir_path = os.path.join(scratchPath, scratchdirectory)
        # Regex to match files like basename.out2.00000.athdf (with variable width)
        pattern = rf"^{re.escape(basename)}\.{re.escape(outputnum)}\.(\d+)\.{extension}$"

        max_index = None
        if os.path.isdir(dir_path):
            for fname in os.listdir(dir_path):
                match = re.match(pattern, fname)
                if match:
                    idx = int(match.group(1))
                    if max_index is None or idx > max_index:
                        max_index = idx

        if max_index is None:
            # If no matching files are found, you can decide to raise an error,
            # return an empty list, or default to 0. Here we raise an error:
            raise FileNotFoundError(
                f"No files matching the pattern '{basename}.{outputnum}.*.athdf' were found "
                f"in directory '{dir_path}'. Cannot determine largest index."
            )

        fn = max_index

    # Now generate the list of files from f0 up to and including fn
    files = []
    for f in np.arange(f0, fn + 1, step, dtype=int):
        filename = (
            scratchPath
            + scratchdirectory
            + basename
            + "."
            + outputnum
            + "."
            + str(f).zfill(width)
            + "."
            + extension
        )
        files.append(filename)
    
    # Finally, check if the file with type {basename}.{outputnum}.final.{extension} exists, and if so, add it to the list
    if os.path.isfile(scratchPath + scratchdirectory + basename + "." + outputnum + ".final." + extension):
            files.append(scratchPath + scratchdirectory + basename + "." + outputnum + ".final." + extension)

    return files

import tkinter as tk
from tkinter import ttk

import json
import os
import pickle

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from build_hologram import build_hologram
from show_hologram_on_slm import show_hologram_on_slm, get_blaze_32, set_blaze_32
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------------- SLM / optical parameters (hard-coded) ----------------
Nx, Ny = 512, 512       # SLM pixels
p = 15e-6               # pixel pitch [m]
wavelength = 1064e-9    # wavelength [m]
f = 3.0e-3              # objective focal length [m]

scaling_factor = p / (f * wavelength)

# Hologram parameters defaults
DEFAULTS = {
	"L": 40.0,
	"angle": 0.0,
	"dz": 0.0,            # defocus [µm]; 0 = no Fresnel phase
	"line_offset_x": -20.0,
	"line_offset_y": 0.0,
	"spot_offset_x": 20.0,
	"spot_offset_y": 0.0,
	"astig_vertical": 0.0,
	"astig_oblique": 0.0,
}

A0 = 1.0

# Hard-coded SLM window placement and size (in screen pixels)
SLM_OFFSET_X = 2560  # X-position of the SLM window (e.g. second monitor)
SLM_OFFSET_Y = 0  # Y-position of the SLM window
SLM_WIDTH = Nx
SLM_HEIGHT = Ny


SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "hologram_settings.json")
RNG_STATE_PATH = os.path.join(os.path.dirname(__file__), "rng_state.pkl")


class HologramApp:
	def __init__(self, master):
		self.master = master
		master.title("Line Trap Hologram Controller")

		# Track the current Matplotlib figure used for the SLM
		self.current_fig = None

		# Randomise flag for the line selection mask
		self.randomise = False

		# Load saved RNG flag
		self.load_saved_rng = True

		# Tk variables
		self.vars = {
			name: tk.StringVar(value=str(value)) for name, value in DEFAULTS.items()
		}

		# Map Entry widgets to variable names for nudge handling
		self.entry_to_name = {}

		# Try to load any previously saved settings (vars, randomise, blaze)
		self._load_settings()

		self._build_ui()

		# Make sure randomise button label matches loaded state
		if self.randomise:
			self.randomise_button.config(text="Randomise: ON")
		else:
			self.randomise_button.config(text="Randomise: OFF")
		
		# Make sure load_saved_rng checkbox matches loaded state
		self.load_saved_rng_var.set(self.load_saved_rng)
		# Ensure that closing the main window also closes the SLM figure
		self.master.protocol("WM_DELETE_WINDOW", self.on_close)
		# Build initial hologram
		self.update_hologram()

	def _build_ui(self):
		pad = 5

		row = 0
		ttk.Label(self.master, text="L (µm)").grid(row=row, column=0, sticky="e", padx=pad, pady=pad)
		l_entry = ttk.Entry(self.master, textvariable=self.vars["L"], width=10)
		l_entry.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
		self.entry_to_name[l_entry] = "L"

		row += 1
		ttk.Label(self.master, text="Angle (deg)").grid(row=row, column=0, sticky="e", padx=pad, pady=pad)
		angle_entry = ttk.Entry(self.master, textvariable=self.vars["angle"], width=10)
		angle_entry.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
		self.entry_to_name[angle_entry] = "angle"

		row += 1
		ttk.Label(self.master, text="dz (µm)").grid(row=row, column=0, sticky="e", padx=pad, pady=pad)
		dz_entry = ttk.Entry(self.master, textvariable=self.vars["dz"], width=10)
		dz_entry.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
		self.entry_to_name[dz_entry] = "dz"

		row += 1
		ttk.Label(self.master, text="Line offset X (µm)").grid(row=row, column=0, sticky="e", padx=pad, pady=pad)
		lox_entry = ttk.Entry(self.master, textvariable=self.vars["line_offset_x"], width=10)
		lox_entry.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
		self.entry_to_name[lox_entry] = "line_offset_x"

		row += 1
		ttk.Label(self.master, text="Line offset Y (µm)").grid(row=row, column=0, sticky="e", padx=pad, pady=pad)
		loy_entry = ttk.Entry(self.master, textvariable=self.vars["line_offset_y"], width=10)
		loy_entry.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
		self.entry_to_name[loy_entry] = "line_offset_y"

		row += 1
		ttk.Label(self.master, text="Spot offset X (µm)").grid(row=row, column=0, sticky="e", padx=pad, pady=pad)
		sox_entry = ttk.Entry(self.master, textvariable=self.vars["spot_offset_x"], width=10)
		sox_entry.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
		self.entry_to_name[sox_entry] = "spot_offset_x"

		row += 1
		ttk.Label(self.master, text="Spot offset Y (µm)").grid(row=row, column=0, sticky="e", padx=pad, pady=pad)
		soy_entry = ttk.Entry(self.master, textvariable=self.vars["spot_offset_y"], width=10)
		soy_entry.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
		self.entry_to_name[soy_entry] = "spot_offset_y"

		row += 1
		ttk.Label(self.master, text="astig vertical").grid(row=row, column=0, sticky="e", padx=pad, pady=pad)
		astig_vertical_entry = ttk.Entry(self.master, textvariable=self.vars["astig_vertical"], width=10)
		astig_vertical_entry.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
		self.entry_to_name[astig_vertical_entry] = "astig_vertical"

		row += 1
		ttk.Label(self.master, text="astig oblique").grid(row=row, column=0, sticky="e", padx=pad, pady=pad)
		astig_oblique_entry = ttk.Entry(self.master, textvariable=self.vars["astig_oblique"], width=10)
		astig_oblique_entry.grid(row=row, column=1, sticky="w", padx=pad, pady=pad)
		self.entry_to_name[astig_oblique_entry] = "astig_oblique"

		# ---- Blaze curve editor ----
		row += 1
		blaze_frame = ttk.LabelFrame(self.master, text="Blaze curve")
		blaze_frame.grid(row=row, column=0, columnspan=2, padx=pad, pady=(pad, pad), sticky="nsew")

		# Initialise control points from current blaze_32
		blaze32 = get_blaze_32()
		n_b = len(blaze32)
		samples = np.linspace(0.0, 1.0, n_b)
		self.blaze_ctrl_x = np.array([0.0, 1.0/3.0, 2.0/3.0, 1.0], dtype=float)
		self.blaze_ctrl_y = np.interp(self.blaze_ctrl_x, samples, blaze32)
		self._dragging_blaze_idx = None

		# Create the Matplotlib figure embedded in Tk
		self.blaze_fig = plt.Figure(figsize=(3, 2), dpi=100)
		self.blaze_ax = self.blaze_fig.add_subplot(111)
		curve_x = np.linspace(0.0, 1.0, 128)
		curve_y = self._spline_eval(curve_x, self.blaze_ctrl_x, self.blaze_ctrl_y)
		curve_y = np.clip(curve_y, 0.0, 1.0)
		(self.blaze_line,) = self.blaze_ax.plot(curve_x, curve_y, "-k")
		self.blaze_points = self.blaze_ax.scatter(self.blaze_ctrl_x, self.blaze_ctrl_y, c="r", s=30, zorder=3)
		self.blaze_ax.set_xlim(0.0, 1.0)
		self.blaze_ax.set_ylim(0.0, 1.0)
		self.blaze_ax.set_xlabel("Input (normalised)")
		self.blaze_ax.set_ylabel("Phase (normalised)")
		self.blaze_ax.grid(True, alpha=0.3)

		self.blaze_canvas = FigureCanvasTkAgg(self.blaze_fig, master=blaze_frame)
		self.blaze_canvas.draw()
		self.blaze_canvas.get_tk_widget().pack(fill="both", expand=True)

		# Connect mouse events for dragging the two middle control points
		self.blaze_canvas.mpl_connect("button_press_event", self._on_blaze_press)
		self.blaze_canvas.mpl_connect("button_release_event", self._on_blaze_release)
		self.blaze_canvas.mpl_connect("motion_notify_event", self._on_blaze_motion)

		# Bind updates: when entry loses focus or user presses Return, rebuild hologram
		for entry in (
			l_entry,
			angle_entry,
			dz_entry,
			lox_entry,
			loy_entry,
			sox_entry,
			soy_entry,
			astig_vertical_entry,
			astig_oblique_entry,
		):
			entry.bind("<Return>", lambda event: self.update_hologram())
			entry.bind("<FocusOut>", lambda event: self.update_hologram())
			# Use Up/Down arrow keys to increment/decrement the value
			entry.bind("<Up>", lambda event, e=entry: self._nudge_entry(e, +1))
			entry.bind("<Down>", lambda event, e=entry: self._nudge_entry(e, -1))

		row += 1
		self.randomise_button = ttk.Button(self.master, text="Randomise: OFF", command=self.toggle_randomise)
		self.randomise_button.grid(row=row, column=0, columnspan=2, pady=(pad * 2, pad))

		row += 1
		self.load_saved_rng_var = tk.BooleanVar(value=self.load_saved_rng)
		self.load_saved_rng_checkbox = ttk.Checkbutton(
			self.master,
			text="Load saved RNG on randomize",
			variable=self.load_saved_rng_var,
			command=self._on_load_saved_rng_toggled
		)
		self.load_saved_rng_checkbox.grid(row=row, column=0, columnspan=2, sticky="w", padx=pad, pady=(pad, 0))

		row += 1
		quit_btn = ttk.Button(self.master, text="Quit", command=self.on_close)
		quit_btn.grid(row=row, column=0, columnspan=2, pady=(0, pad))

	def _get_float(self, name):
		value_str = self.vars[name].get().strip()
		try:
			return float(value_str)
		except ValueError:
			# If invalid, revert to previous default and do not update
			self.vars[name].set(str(DEFAULTS[name]))
			return DEFAULTS[name]

	def _nudge_entry(self, entry, direction):
		"""Increment/decrement the entry's float value with arrow keys."""
		# Map entry widget back to its variable name
		name = self.entry_to_name.get(entry)

		# Fallback: if we cannot resolve the name, do nothing
		if name is None:
			value_str = entry.get().strip()
			try:
				value = float(value_str)
			except ValueError:
				return
		else:
			value = self._get_float(name)

		# Choose step sizes
		if name == "angle":
			step = 1.0
		elif name == "dz":
			step = 0.1
		else:
			step = 1.0

		new_value = value + direction * step

		# Update the variable and entry text
		if name is not None:
			self.vars[name].set(str(new_value))
		else:
			entry.delete(0, tk.END)
			entry.insert(0, str(new_value))

		# Immediately update the hologram
		self.update_hologram()

	def _load_settings(self):
		"""Load saved settings (if any) from disk into vars, randomise, blaze."""
		if not os.path.exists(SETTINGS_PATH):
			return
		try:
			with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
				data = json.load(f)
		except Exception:
			return

		# Restore scalar variables
		vars_data = data.get("vars", {})
		for name, value in vars_data.items():
			if name in self.vars:
				try:
					self.vars[name].set(str(float(value)))
				except Exception:
					# Fall back silently to existing value
					pass

		# Restore randomise flag
		if "randomise" in data:
			self.randomise = bool(data["randomise"])

		# Do not restore load_saved_rng from settings.
		# App start default is to load saved RNG.

		# Restore blaze curve if present
		blaze32 = data.get("blaze32")
		if blaze32 is not None:
			try:
				arr = np.asarray(blaze32, dtype=float).ravel()
				if arr.size >= 2:
					set_blaze_32(arr)
			except Exception:
				pass

	def _spline_eval(self, x, xp, yp):
		"""Evaluate a natural cubic spline through (xp, yp) at positions x.

		Simple implementation (small number of points), used to make the
		blaze curve smooth between the four anchor points.
		"""

		xp = np.asarray(xp, dtype=float)
		yp = np.asarray(yp, dtype=float)
		x = np.asarray(x, dtype=float)
		n = xp.size
		if n < 2:
			return np.full_like(x, yp[0] if n == 1 else 0.0, dtype=float)

		# Compute second derivatives (natural spline conditions)
		y2 = np.zeros_like(yp, dtype=float)
		u = np.zeros_like(yp, dtype=float)
		for i in range(1, n - 1):
			sig = (xp[i] - xp[i - 1]) / (xp[i + 1] - xp[i - 1])
			p = sig * y2[i - 1] + 2.0
			y2[i] = (sig - 1.0) / p
			u[i] = (
				(yp[i + 1] - yp[i]) / (xp[i + 1] - xp[i])
				- (yp[i] - yp[i - 1]) / (xp[i] - xp[i - 1])
			)
			u[i] = (6.0 * u[i] / (xp[i + 1] - xp[i - 1]) - sig * u[i - 1]) / p
		for k in range(n - 2, -1, -1):
			y2[k] = y2[k] * y2[k + 1] + u[k]

		# Interpolate
		x_flat = x.ravel()
		y_flat = np.empty_like(x_flat, dtype=float)
		for idx, xv in enumerate(x_flat):
			if xv <= xp[0]:
				klo, khi = 0, 1
			elif xv >= xp[-1]:
				klo, khi = n - 2, n - 1
			else:
				klo, khi = 0, n - 1
				while khi - klo > 1:
					k = (khi + klo) // 2
					if xp[k] > xv:
						khi = k
					else:
						klo = k
			h = xp[khi] - xp[klo]
			if h == 0:
				y_flat[idx] = yp[klo]
				continue
			a = (xp[khi] - xv) / h
			b = (xv - xp[klo]) / h
			y_flat[idx] = (
				a * yp[klo]
				+ b * yp[khi]
				+ ((a**3 - a) * y2[klo] + (b**3 - b) * y2[khi]) * (h * h) / 6.0
			)

		return y_flat.reshape(x.shape)

	def _update_blaze_from_ctrl(self):
		"""Recompute blaze_32 from the control points and refresh hologram."""
		# Build a 32-point blaze curve by linear interpolation of the
		# four control points (two endpoints fixed at x=0,1), using a
		# smooth natural cubic spline.
		s_ctrl = self.blaze_ctrl_x
		y_ctrl = self.blaze_ctrl_y
		samples = np.linspace(0.0, 1.0, 32)
		blaze32 = self._spline_eval(samples, s_ctrl, y_ctrl)
		blaze32 = np.clip(blaze32, 0.0, 1.0)
		set_blaze_32(blaze32)

		# Update the plotted curve and points
		curve_x = np.linspace(0.0, 1.0, 128)
		curve_y = self._spline_eval(curve_x, s_ctrl, y_ctrl)
		curve_y = np.clip(curve_y, 0.0, 1.0)
		self.blaze_line.set_data(curve_x, curve_y)
		self.blaze_points.set_offsets(np.column_stack([s_ctrl, y_ctrl]))
		self.blaze_canvas.draw_idle()

		# Apply the new blaze immediately
		self.update_hologram()

	def _on_blaze_press(self, event):
		"""Start dragging one of the two interior blaze control points."""
		if event.inaxes is not self.blaze_ax:
			return
		if event.xdata is None or event.ydata is None:
			return

		# Only allow dragging of the two interior points (indices 1 and 2)
		click_x = event.xdata
		click_y = event.ydata
		dists = np.hypot(self.blaze_ctrl_x[1:3] - click_x, self.blaze_ctrl_y[1:3] - click_y)
		idx_rel = int(np.argmin(dists))
		if dists[idx_rel] < 0.05:
			self._dragging_blaze_idx = idx_rel + 1

	def _on_blaze_motion(self, event):
		"""While dragging, update the y-value of the selected control point."""
		if self._dragging_blaze_idx is None:
			return
		if event.inaxes is not self.blaze_ax:
			return
		if event.ydata is None:
			return

		# Constrain movement vertically between 0 and 1; x stays fixed
		y = max(0.0, min(1.0, float(event.ydata)))
		self.blaze_ctrl_y[self._dragging_blaze_idx] = y
		self._update_blaze_from_ctrl()

	def _on_blaze_release(self, event):
		"""Stop dragging any control point."""
		self._dragging_blaze_idx = None

	def update_hologram(self):
		"""Recompute the hologram from current UI values and show on SLM."""
		L = self._get_float("L")
		angle = self._get_float("angle")
		dz = self._get_float("dz")
		line_offset_x = self._get_float("line_offset_x")
		line_offset_y = self._get_float("line_offset_y")
		spot_offset_x = self._get_float("spot_offset_x")
		spot_offset_y = self._get_float("spot_offset_y")
		astig_vertical = self._get_float("astig_vertical")
		astig_oblique = self._get_float("astig_oblique")

		# Load saved RNG state if requested and randomisation is enabled
		rng = None
		if self.randomise:
			if self.load_saved_rng:
				rng = self._load_rng_state()
				if rng is None:
					print("Warning: Load saved RNG is enabled but no valid saved state was found. Randomisation skipped.")
					return
			else:
				# Create a fresh RNG and save it immediately for later reuse.
				rng = np.random.default_rng()
				self._save_rng_state(rng)

		holo_total, _ = build_hologram(
			Nx,
			Ny,
			f, 
   			wavelength,
      		p,
			scaling_factor,
			dz,
			L,
			A0,
			angle,
			self.randomise,
			line_offset_x=line_offset_x,
			line_offset_y=line_offset_y,
			spot_offset_x=spot_offset_x,
			spot_offset_y=spot_offset_y,
			astig_vertical=astig_vertical,
			astig_oblique=astig_oblique,
			rng=rng
		)

		# Close previous SLM figure, if any, to avoid accumulating windows
		if self.current_fig is not None:
			try:
				plt.close(self.current_fig)
			except Exception:
				pass

		self.current_fig = show_hologram_on_slm(
			holo_total,
			offset_x=SLM_OFFSET_X,
			offset_y=SLM_OFFSET_Y,
			slm_width=SLM_WIDTH,
			slm_height=SLM_HEIGHT,
		)

	def toggle_randomise(self):
		"""Toggle the randomise flag and update the button label, then rebuild."""
		self.randomise = not self.randomise
		if self.randomise:
			self.randomise_button.config(text="Randomise: ON")
		else:
			self.randomise_button.config(text="Randomise: OFF")

		self.update_hologram()

	def _on_load_saved_rng_toggled(self):
		"""Update the load_saved_rng flag when checkbox is toggled."""
		self.load_saved_rng = self.load_saved_rng_var.get()

	def _load_rng_state(self):
		"""Load RNG state from disk."""
		try:
			with open(RNG_STATE_PATH, "rb") as fp:
				return pickle.load(fp)
		except FileNotFoundError:
			return None
		except Exception as e:
			print(f"Error loading RNG state: {e}")
			return None

	def _save_rng_state(self, rng):
		"""Save the provided RNG state to disk."""
		try:
			with open(RNG_STATE_PATH, "wb") as fp:
				pickle.dump(rng, fp)
		except Exception as e:
			print(f"Error saving RNG state: {e}")
	def on_close(self):
		"""Close the SLM window and then the Tk app."""
		# Persist current settings to disk
		self._save_settings()

		if self.current_fig is not None:
			try:
				plt.close(self.current_fig)
			except Exception:
				pass
			finally:
				self.current_fig = None

		self.master.destroy()

	def _save_settings(self):
		"""Save current vars, randomise flag, load_saved_rng flag, and blaze curve to disk."""
		data = {}
		# Save scalar variables as floats
		vars_data = {}
		for name in self.vars.keys():
			try:
				vars_data[name] = float(self.vars[name].get().strip())
			except Exception:
				# If parsing fails, fall back to DEFAULTS if available
				vars_data[name] = float(DEFAULTS.get(name, 0.0))
		data["vars"] = vars_data

		data["randomise"] = bool(self.randomise)
		data["load_saved_rng"] = bool(self.load_saved_rng)
		# Save the full 32-point blaze curve
		try:
			data["blaze32"] = get_blaze_32().tolist()
		except Exception:
			data["blaze32"] = None

		try:
			with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
				json.dump(data, f, indent=2)
		except Exception:
			# If saving fails, ignore silently; app should still quit cleanly
			pass


def main():
	root = tk.Tk()
	app = HologramApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()


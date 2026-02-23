import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

from build_hologram import build_hologram
from show_hologram_on_slm import show_hologram_on_slm
from fresnel_phase import fresnel_phase


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
}

A0 = 1.0

# Hard-coded SLM window placement and size (in screen pixels)
SLM_OFFSET_X = 0  # X-position of the SLM window (e.g. second monitor)
SLM_OFFSET_Y = 0  # Y-position of the SLM window
SLM_WIDTH = Nx
SLM_HEIGHT = Ny


class HologramApp:
	def __init__(self, master):
		self.master = master
		master.title("Line Trap Hologram Controller")

		# Track the current Matplotlib figure used for the SLM
		self.current_fig = None

		# Randomise flag for the line selection mask
		self.randomise = False

		# Tk variables
		self.vars = {
			name: tk.StringVar(value=str(value)) for name, value in DEFAULTS.items()
		}

		# Map Entry widgets to variable names for nudge handling
		self.entry_to_name = {}

		self._build_ui()
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

		# Bind updates: when entry loses focus or user presses Return, rebuild hologram
		for entry in (l_entry, angle_entry, dz_entry, lox_entry, loy_entry, sox_entry, soy_entry):
			entry.bind("<Return>", lambda event: self.update_hologram())
			entry.bind("<FocusOut>", lambda event: self.update_hologram())
			# Use Up/Down arrow keys to increment/decrement the value
			entry.bind("<Up>", lambda event, e=entry: self._nudge_entry(e, +1))
			entry.bind("<Down>", lambda event, e=entry: self._nudge_entry(e, -1))

		row += 1
		self.randomise_button = ttk.Button(self.master, text="Randomise: OFF", command=self.toggle_randomise)
		self.randomise_button.grid(row=row, column=0, columnspan=2, pady=(pad * 2, pad))

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

	def update_hologram(self):
		"""Recompute the hologram from current UI values and show on SLM."""
		L = self._get_float("L")
		angle = self._get_float("angle")
		dz = self._get_float("dz")
		line_offset_x = self._get_float("line_offset_x")
		line_offset_y = self._get_float("line_offset_y")
		spot_offset_x = self._get_float("spot_offset_x")
		spot_offset_y = self._get_float("spot_offset_y")

		# Compute Fresnel phase (defocus) if needed
		holo_fresnel = None
		if dz != 0.0:
			holo_fresnel = fresnel_phase(Nx, Ny, dz, f, wavelength, p)

		holo_total, _ = build_hologram(
			Nx,
			Ny,
			scaling_factor,
			L,
			A0,
			angle,
			self.randomise,
			line_offset_x=line_offset_x,
			line_offset_y=line_offset_y,
			spot_offset_x=spot_offset_x,
			spot_offset_y=spot_offset_y,
			holo_fresnel=holo_fresnel,
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

	def on_close(self):
		"""Close the SLM window and then the Tk app."""
		if self.current_fig is not None:
			try:
				plt.close(self.current_fig)
			except Exception:
				pass
			finally:
				self.current_fig = None

		self.master.destroy()


def main():
	root = tk.Tk()
	app = HologramApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()


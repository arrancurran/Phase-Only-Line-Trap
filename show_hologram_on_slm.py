import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def show_hologram_on_slm(hologram, offset_x=2560, offset_y=0, slm_width=512, slm_height=512):
    
    blaze_32 = np.array([
    0.0,
    0.0696908,
    0.138666,
    0.206212,
    0.271612,
    0.334152,
    0.393116,
    0.447789,
    0.497457,
    0.541403,
    0.578914,
    0.609577,
    0.634113,
    0.653506,
    0.668742,
    0.680805,
    0.69068,
    0.699352,
    0.707805,
    0.717025,
    0.727995,
    0.741701,
    0.759005,
    0.779875,
    0.803884,
    0.8306,
    0.859595,
    0.890439,
    0.922701,
    0.955953,
    0.989764,
    1.0
])
    blaze = np.interp(np.arange(256), np.linspace(0, 255, len(blaze_32)), blaze_32)

    # Convert phase to 8-bit values and apply blaze
    hologram_uint8 = np.uint8(np.rint(hologram * (255.0 / (2*np.pi))))
    blazed_hologram = blaze[hologram_uint8]
    
    # Create a figure with no axes and matching the slm dimensions in pixels
    fig = plt.figure(figsize=(slm_width, slm_height), dpi=1)
    ax = fig.add_axes([0, 0, 1, 1])
    # Use nearest-neighbor interpolation so each hologram pixel maps to SLM pixels.
    ax.imshow(blazed_hologram, cmap='gray', interpolation='nearest')
    ax.axis('off')
    manager = plt.get_current_fig_manager()
    # Try to handle both TkAgg and Qt backends for window positioning
    try:
        # TkAgg backend (common on Windows)
        window = manager.window
        # Remove title bar and window decorations
        try:
            window.overrideredirect(True)
        except Exception:
            pass
        # Hide the toolbar if present
        try:
            if hasattr(fig.canvas, "toolbar") and fig.canvas.toolbar is not None:
                fig.canvas.toolbar.pack_forget()
        except Exception:
            pass
        # Set explicit window size and position: WIDTHxHEIGHT+X+Y
        window.wm_geometry(f"{slm_width}x{slm_height}+{offset_x}+{offset_y}")
        window.lift()
    except Exception:
        pass  # Not TkAgg or failed to set geometry

    # Ensure the figure window is actually shown (non-blocking) when used
    try:
        fig.canvas.draw_idle()
        plt.show(block=False)
    except Exception:
        pass

    return fig
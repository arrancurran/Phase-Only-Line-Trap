import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

def show_hologram_on_slm(hologram, offset_x=2560, offset_y=0, slm_width=512, slm_height=512):
    
    blaze_32 = np.array([0.000000, 0.057705, 0.114756, 0.170501, 0.224285, 0.275457, 0.323361, 0.367346, 0.406758, 0.440943, 0.469248, 0.491020, 0.505781, 0.514422, 0.518472, 0.519466, 0.518936, 0.518416, 0.519441, 0.523543, 0.532256, 0.547115, 0.569651, 0.600860, 0.640003, 0.685993, 0.737743, 0.794168, 0.854179, 0.916692, 0.980618, 1.000000])
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
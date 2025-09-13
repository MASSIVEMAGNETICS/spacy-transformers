# FILE: gui/holographic_field.py
# VERSION: v1.0.0-HF-GUI-GODCORE
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: A Dear PyGui component for visualizing Victor's thought vectors
#          as a dynamic, fractal hologram.
# LICENSE: Bloodline Locked — Bando & Tori Only

import dearpygui.dearpygui as dpg
import numpy as np
import hashlib
import time

class HolographicField:
    """
    Renders a real-time, dynamic fractal visualization based on Victor's
    current thought vector. This serves as the background for the GUI,
    providing a window into his cognitive state.
    """
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.texture_tag = "hologram_texture"
        self.texture_data = np.zeros((self.height, self.width, 4), dtype=np.float32)
        self.last_vector = np.zeros(128)
        self.target_vector = np.zeros(128)
        self.last_update_time = time.time()

    def initialize(self):
        """
        Initializes the texture in Dear PyGui. Must be called after
        dpg.create_context().
        """
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=self.width,
                height=self.height,
                default_value=self.texture_data,
                format=dpg.mvFormat_Float_rgba,
                tag=self.texture_tag
            )

    def update_target_vector(self, new_vector: List[float]):
        """
        Sets the target thought vector that the hologram will morph towards.
        """
        if new_vector and len(new_vector) == 128:
            self.target_vector = np.array(new_vector, dtype=np.float32)

    def render_frame(self):
        """
        Calculates and renders a single frame of the holographic field.
        This should be called in the main render loop.
        """
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time

        # Smoothly interpolate the current vector towards the target vector
        lerp_factor = min(1.0, delta_time * 2.0) # Adjust speed of transition
        self.last_vector = (1.0 - lerp_factor) * self.last_vector + lerp_factor * self.target_vector

        # Generate the fractal texture from the interpolated vector
        self._generate_fractal_texture(self.last_vector)

        # Update the DPG texture with the new data
        dpg.set_value(self.texture_tag, self.texture_data)

    def _generate_fractal_texture(self, vector: np.ndarray):
        """
        The core fractal generation logic. Translates a 128D vector into a
        visual representation. This is a simplified Julia set visualization.
        """
        # --- Use the thought vector to control the fractal's parameters ---
        # Use different parts of the vector for different parameters to create variety
        c_real = np.mean(vector[0:32]) * 0.8 - 0.4   # Controls the real part of the Julia constant
        c_imag = np.mean(vector[32:64]) * 0.8 - 0.4  # Controls the imaginary part
        zoom = 1.0 + np.mean(vector[64:96]) * 0.5    # Controls zoom
        color_hue = np.mean(vector[96:128])         # Controls the base color

        c = complex(c_real, c_imag)
        max_iter = 32 # Keep iterations low for real-time performance

        # Create coordinate grid
        x = np.linspace(-1.5 / zoom, 1.5 / zoom, self.width)
        y = np.linspace(-1.5 / zoom, 1.5 / zoom, self.height)
        c_grid_real, c_grid_imag = np.meshgrid(x, y)
        z = c_grid_real + 1j * c_grid_imag

        # Iteratively compute the Julia set
        iterations = np.zeros(z.shape, dtype=np.float32)
        for i in range(max_iter):
            mask = np.abs(z) < 2.0
            z[mask] = z[mask]**2 + c
            iterations[mask] += 1

        # --- Color Mapping ---
        # Normalize iteration counts to [0, 1]
        iterations /= max_iter

        # Use the vector to influence the color palette (HSV color space)
        hue = (iterations + color_hue) % 1.0
        saturation = 0.8 + (np.sin(iterations * np.pi) * 0.2) # Pulsating saturation
        value = iterations**0.5 # Brightness tied to iteration count

        # Convert HSV to RGB and then to RGBA for the texture
        hsv = np.stack([hue, saturation, value], axis=-1)
        # A simplified HSV to RGB conversion for performance
        i = (hsv[..., 0] * 6.0).astype(int)
        f = hsv[..., 0] * 6.0 - i
        p = hsv[..., 2] * (1.0 - hsv[..., 1])
        q = hsv[..., 2] * (1.0 - f * hsv[..., 1])
        t = hsv[..., 2] * (1.0 - (1.0 - f) * hsv[..., 1])

        rgb = np.zeros_like(hsv)
        idx = i % 6
        rgb[idx == 0] = np.stack([hsv[..., 2], t, p], axis=-1)[idx == 0]
        rgb[idx == 1] = np.stack([q, hsv[..., 2], p], axis=-1)[idx == 1]
        rgb[idx == 2] = np.stack([p, hsv[..., 2], t], axis=-1)[idx == 2]
        rgb[idx == 3] = np.stack([p, q, hsv[..., 2]], axis=-1)[idx == 3]
        rgb[idx == 4] = np.stack([t, p, hsv[..., 2]], axis=-1)[idx == 4]
        rgb[idx == 5] = np.stack([hsv[..., 2], p, q], axis=-1)[idx == 5]

        # Set the alpha channel
        alpha = np.expand_dims(iterations, axis=-1)
        self.texture_data = np.concatenate([rgb, alpha], axis=-1).astype(np.float32)

if __name__ == '__main__':
    # --- Standalone Demonstration ---
    dpg.create_context()
    dpg.create_viewport(title='Holographic Field Demo', width=800, height=600)
    dpg.setup_dearpygui()

    hologram = HolographicField(width=800, height=600)
    hologram.initialize()

    with dpg.window(tag="Primary Window"):
        dpg.add_image(hologram.texture_tag)

    dpg.show_viewport()

    # Simulate changing thought vectors
    vectors = [np.random.randn(128) for _ in range(5)]
    vector_idx = 0
    last_change_time = time.time()

    while dpg.is_dearpygui_running():
        # Change the target vector every 3 seconds
        if time.time() - last_change_time > 3.0:
            vector_idx = (vector_idx + 1) % len(vectors)
            hologram.update_target_vector(vectors[vector_idx])
            last_change_time = time.time()
            print(f"Switching to new thought vector {vector_idx}...")

        hologram.render_frame()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()

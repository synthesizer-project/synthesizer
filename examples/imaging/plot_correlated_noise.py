"""
Correlated noise example
========================

An example demonstrating how to apply correlated noise to a synthesizer
``Image``. Real observational data contain noise that is spatially correlated
(e.g. due to drizzling, PSF convolution, or detector effects). This example
shows how to model that correlation structure from an observed noise map and
transfer it to a mock image.

The workflow is:

1. Build a simple mock galaxy image (a 2D Gaussian).
2. Construct a synthetic "blank-sky" noise sample that has spatial
   correlations, mimicking the noise pattern of a real detector or
   drizzled mosaic.
3. Apply correlated noise to the mock image via
   ``Image.apply_correlated_noise``.
4. Inspect the results visually and compare the power spectra of the
   original noise template with the generated noise field.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import kpc

from synthesizer.imaging.image import Image

# %%
# Create a mock galaxy image
# --------------------------
# We use a simple 2D Gaussian as a stand-in for a galaxy light profile.

npix = 64
resolution = 0.1 * kpc
fov = npix * resolution

# Pixel coordinate grids centred on zero
x = np.linspace(-npix / 2, npix / 2, npix)
xx, yy = np.meshgrid(x, x)

# Gaussian with sigma = 6 pixels
sigma_pix = 6.0
galaxy_arr = np.exp(-(xx**2 + yy**2) / (2 * sigma_pix**2))

img = Image(resolution=resolution, fov=fov, img=galaxy_arr)

# %%
# Build a correlated noise template
# ----------------------------------
# We simulate a realistic noise map by filtering white noise with a Gaussian
# kernel in Fourier space.  This produces spatially correlated noise that
# resembles drizzled or PSF-correlated detector noise.

rng = np.random.default_rng(0)
white_noise = rng.normal(size=(npix, npix))

# Gaussian smoothing kernel in Fourier space (correlation length ~ 3 px)
corr_length_pix = 3.0
freq = np.fft.fftfreq(npix)
fx, fy = np.meshgrid(freq, freq)
kernel_fft = np.exp(-2 * np.pi**2 * corr_length_pix**2 * (fx**2 + fy**2))

noise_template = np.real(np.fft.ifft2(np.fft.fft2(white_noise) * kernel_fft))
# Normalise so the standard deviation is comparable to the galaxy signal
noise_template *= 0.2 / noise_template.std()

# %%
# Apply correlated noise
# ----------------------
# ``apply_correlated_noise`` estimates the correlation structure from
# ``noise_template`` via its power spectrum and generates a new noise
# realisation with the same statistical properties, which is then added
# to the image.

noisy_img = img.apply_correlated_noise(
    observed_noise_arr=noise_template,
    subtract_mean=True,  # remove any DC offset in the noise template
    correct_periodicity=True,
)

# %%
# Visualise the results
# ---------------------
# We plot the original image, the noise template, the applied noise field,
# and the final noisy image side by side.

fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

vmin_img = galaxy_arr.min()
vmax_img = galaxy_arr.max()

axes[0].imshow(
    galaxy_arr, origin="lower", cmap="inferno", vmin=vmin_img, vmax=vmax_img
)
axes[0].set_title("Mock galaxy (input)")
axes[0].axis("off")

axes[1].imshow(noise_template, origin="lower", cmap="RdBu_r")
axes[1].set_title("Noise template")
axes[1].axis("off")

axes[2].imshow(noisy_img.noise_arr, origin="lower", cmap="RdBu_r")
axes[2].set_title("Generated noise field")
axes[2].axis("off")

axes[3].imshow(
    noisy_img.arr, origin="lower", cmap="inferno", vmin=vmin_img, vmax=vmax_img
)
axes[3].set_title("Noisy image (output)")
axes[3].axis("off")

fig.tight_layout()
plt.show()
plt.close(fig)

# %%
# Compare power spectra
# ---------------------
# The power spectrum of the generated noise field should match that of the
# noise template, confirming that the correlation structure has been correctly
# transferred.


def radial_power_spectrum(arr):
    """Compute the azimuthally averaged power spectrum of a 2D array."""
    ft = np.fft.fft2(arr)
    ps = np.abs(np.fft.fftshift(ft)) ** 2
    n = arr.shape[0]
    cy, cx = n // 2, n // 2
    y, x = np.indices(ps.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    radial_ps = np.bincount(r.ravel(), weights=ps.ravel())
    counts = np.bincount(r.ravel())
    radial_ps = radial_ps[counts > 0] / counts[counts > 0]
    freqs = np.where(counts > 0)[0]
    return freqs, radial_ps


freq_tmpl, ps_tmpl = radial_power_spectrum(noise_template)
freq_gen, ps_gen = radial_power_spectrum(noisy_img.noise_arr)

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(freq_tmpl, ps_tmpl, label="Noise template", lw=2)
ax.loglog(freq_gen, ps_gen, label="Generated noise", lw=2, ls="--")
ax.set_xlabel("Spatial frequency (pixels$^{-1}$)")
ax.set_ylabel("Power")
ax.set_title("Radial power spectrum comparison")
ax.legend()
fig.tight_layout()
plt.show()
plt.close(fig)

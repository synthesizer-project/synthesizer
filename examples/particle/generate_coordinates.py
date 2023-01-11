


from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.particle.image import ImageGenerator


# --- create a set of particles assuming a 3D gaussian

coords = CoordinateGenerator.generate_3D_gaussian(10000)

print(coords)
print(coords.shape)


# --- make a basic (histogram) image from these coordinates


image = ImageGenerator.generate_histogram(coords, size_pixels = 50, pixel_scale = 0.2)


fig, ax = image.make_image_plot(show = True)

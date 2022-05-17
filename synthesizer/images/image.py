""" Definitions for image objects
"""
import numpy as np


class Image:
    """ A class to compute and analyse an image created from simulation
        particle data

        Attributes:
        :attr:

        Returns
        :return:
    """

    # Define slots to reduce memory overhead of this class
    __slots__ = ["res", "width", "img_sum", "npart", "sim_pos",
                 "shifted_sim_pos", "part_val", "pix_pos", "pos_offset",
                 "img"]

    def __init__(self, res, sim_pos, part_val, width):

        # Helpful image metadata
        if isinstance(res, int):
            self.res = (res, res)
        elif isinstance(res, tuple):
            self.res = res
        else:
            raise ValueError("Improper type of res, must be tuple or int")
        self.width = width
        self.img_sum = None

        # Store the particle information
        self.npart = sim_pos.shape[0]
        self.sim_pos = sim_pos
        self.shifted_sim_pos = sim_pos
        self.part_val = part_val
        self.pix_pos = np.zeros(self.sim_pos.shape, dtype=np.float64)

        # Are the positions centered?
        if np.min(sim_pos) < 0:

            # If so compute that offset and shift particles to start at 0
            self.pos_offset = np.min(sim_pos, axis=0)
            self.shifted_sim_pos -= self.pos_offset

        # Set up img object (populated later)
        self.img = np.zeros(self.res, dtype=np.float64)

        # Run instantiation methods
        self.get_pixel_pos()

    def get_pixel_pos(self):
        """ Convert particle positions to the pixel reference frame.
        """

        # Convert sim positions to pixel positions
        self.pix_pos[:, 0] = np.int32(self.shifted_sim_pos[:, 0] / self.width)
        self.pix_pos[:, 1] = np.int32(self.shifted_sim_pos[:, 1] / self.width)
        self.pix_pos[:, 2] = np.int32(self.shifted_sim_pos[:, 2] / self.width)


class NoisyImage(Image):

    # Define slots to reduce memory overhead of this class
    __slots__ = ["pixel_noise", "noisy_img"]

    def __init__(self, res, sim_pos, part_val, width, pixel_noise):

        # Initialise parent
        Image.__init__(self, res, sim_pos, part_val, width)

        # Include noise related attributes
        self.pixel_noise = pixel_noise

        # Set up noisy img
        self.noisy_img = np.zeros(self.res, dtype=np.float64)

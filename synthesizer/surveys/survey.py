""" A script containing the definition of a generic survey calls and prebuilt
    examples
"""
from synthesizer.utils import Singleton


class Survey:
    """ A generic survey object to hold all survey specific attributes
        and methods """

    def __init__(self, filters):

        # Information about the filters
        self.filters = filters

        # Information about the depths
        self.depths = {f: None for f in self.filters}
        self.depth_apertures = {f: None for f in self.filters}

    def compute_pixel_noise(self,):
        pass


class HubbleUDF(Survey, metaclass=Singleton):
    pass

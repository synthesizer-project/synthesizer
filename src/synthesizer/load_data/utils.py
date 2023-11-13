import numpy as np

def get_len(Length):
    """
    Find the beginning and ending indices from a length array
    Args:
        Length (array):
            array of number of particles
    Returns:
        begin (array):
            beginning indices
        end (array):
            ending indices
    """

    begin = np.zeros(len(Length), dtype=np.int64)
    end = np.zeros(len(Length), dtype=np.int64)
    begin[1:] = np.cumsum(Length)[:-1]
    end = np.cumsum(Length)
    return begin, end

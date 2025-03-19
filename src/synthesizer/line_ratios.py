"""A module holding useful line ratios.

This contains both line ratios and diagnostic diagrams for emission line. As
well as including the standard line labels for common lines.

Line ids and specifically the wavelength part here are defined using the cloudy
standard, i.e. using vacuum wavelengths at <200nm and air wavelengths at
>200nm.
"""

# Shorthand for common lines
aliases = {
    "Hb": "H 1 4861.32A",
    "Ha": "H 1 6562.80A",
    "Hg": "H 1 4340.46A",
    "O1": "O 1 6300.30A",
    "O2b": "O 2 3726.03A",
    "O2r": "O 2 3728.81A",
    "O2": "O 2 3726.03A, O 2 3728.81A",
    "O3b": "O 3 4958.91A",
    "O3r": "O 3 5006.84A",
    "O3": "O 3 4958.91A, O 3 5006.84A",
    "Ne3": "Ne 3 3868.76A",
    "N2": "N 2 6583.45A",
    "S2": "S 2 6730.82A, S 2 6716.44A",
}


# Standard names
Ha = aliases["Ha"]
Hb = aliases["Hb"]
O1 = aliases["O1"]
O2b = aliases["O2b"]
O2r = aliases["O2r"]
O2 = aliases["O2"]
O3b = aliases["O3b"]
O3r = aliases["O3r"]
O3 = aliases["O3"]
Ne3 = aliases["Ne3"]
N2 = aliases["N2"]
S2 = aliases["S2"]


# Dictionary of common line labels to use by default
line_labels = {
    "O 2 3726.03A,O 2 3728.81A": "[OII]3726,3729",
    "H 1 4861.32A": r"H\beta",
    "O 3 4958.91A,O 3 5006.84A": "[OIII]4959,5007",
    "H 1 6562.80A": r"H\alpha",
    "O 3 5006.84A": "[OIII]5007",
    "N 2 6583.45A": "[NII]6583",
}


# Define a dictionary to hold line ratios
ratios = {}

# Balmer decrement, should be [2.79--2.86] (Te, ne, dependent)
# for dust free
ratios["BalmerDecrement"] = [Ha, Hb]

# Add reference ratios
ratios["N2"] = [N2, Ha]
ratios["S2"] = [S2, Ha]
ratios["O1"] = [O1, Ha]
ratios["R2"] = [O2b, Hb]
ratios["R3"] = [O3r, Hb]
ratios["R23"] = [O3 + ", " + O2, Hb]
ratios["O32"] = [O3r, O2b]
ratios["Ne3O2"] = [Ne3, O2b]

# Create a tuple of available ratios for importing
available_ratios = tuple(ratios.keys())

# Define a dictionary to hold diagnostic diagrams
diagrams = {}

# Add reference diagrams
diagrams["OHNO"] = [ratios["R3"], [Ne3, O2]]
diagrams["BPT-NII"] = [[N2, Ha], ratios["R3"]]
diagrams["VO78-SII"] = [[S2, Ha], ratios["R3"]]
diagrams["VO78-OI"] = [[O1, Ha], ratios["R3"]]

# Create a tuple of available diagrams for importing
available_diagrams = tuple(diagrams.keys())


def get_bpt_kewley01(logNII_Ha):
    """
    BPT-NII demarcations from Kewley+2001.

    Kewley+03: https://arxiv.org/abs/astro-ph/0106324

    Demarcation defined by:

    log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.47) + 1.19

    Args:
        logNII_Ha (array)
            Array of log([NII]/Halpha) values to give the
            SF-AGN demarcation line

    Returns:
        array
            Corresponding log([OIII]/Hb) ratio array
    """
    return 0.61 / (logNII_Ha - 0.47) + 1.19


def get_bpt_kauffman03(logNII_Ha):
    """
    BPT-NII demarcations from Kauffman+2003.

    Kauffman+03: https://arxiv.org/abs/astro-ph/0304239

    Demarcation defined by:

    log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.05) + 1.3

    Args:
        logNII_Ha (array)
            Array of log([NII]/Halpha) values to give the
            SF-AGN demarcation line

    Returns:
        array
            Corresponding log([OIII]/Hb) ratio array
    """
    return 0.61 / (logNII_Ha - 0.05) + 1.3

"""
A module holding useful line ratios (e.g. R23) and diagrams (pairs of ratios),
e.g. BPT-NII.

Line ids and specifically the wavelength part here are defined using vacuum
wavelengths NOT cloudy standards. There is a parallel module
line_ratios_standard that uses cloudy standard ids.
"""

# shorthand for common lines
O3b = "O 3 4960.29A"
O3r = "O 3 5008.24A"
O3 = [O3b, O3r]
O2b = "O 2 3727.09A"
O2r = "O 2 3729.88A"
O2 = [O2b, O2r]
Ha = "H 1 6564.62A"
Hb = "H 1 4862.69A"
Hg = "H 1 4341.68A"

ratios = {}

# Balmer decrement, should be [2.79--2.86] (Te, ne, dependent)
# for dust free
ratios["BalmerDecrement"] = [[Ha], [Hb]]

# add reference
ratios["N2"] = N2 = [["N 2 6585.27A"], [Ha]]
# add reference
ratios["S2"] = S2 = [["S 2 6732.67A", "S 2 6718.29A"], [Ha]]
ratios["O1"] = O1 = [["O 1 6302.05A"], [Ha]]
ratios["R2"] = [[O2b], [Hb]]
ratios["R3"] = R3 = [[O3r], [Hb]]
ratios["R23"] = [O3 + O2, [Hb]]
ratios["O32"] = [[O3r], [O2b]]
ratios["Ne3O2"] = [["Ne 3 3869.86A"], [O2b]]

# tuple of available ratios
available_ratios = tuple(ratios.keys())

# dictionary of diagrams
diagrams = {}

# add reference
diagrams["OHNO"] = [R3, [["Ne 3 3869.86A"], O2]]

# add reference
diagrams["BPT-NII"] = [N2, R3]

# add reference
# diagrams["VO78-SII"] = [[S2, [Ha]], R3]

# add reference
# diagrams["VO78-OI"] = [[O1, [Ha]], R3]

available_diagrams = tuple(diagrams.keys())

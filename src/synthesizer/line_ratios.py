"""
A module holding useful line ratios (e.g. R23) and diagrams (pairs of ratios),
e.g. BPT-NII.

Line ids and specifically the wavelength part here are defined using the cloudy
standard, i.e. using vacuum wavelengths at <200nm and air wavelengths at
>200nm.

There is a parallel module that vacuum
"""

# shorthand for common lines

# standard names
Ha = "H 1 6562.80A"
Hb = "H 1 4861.32A"
Hg = "H 1 4340.46A"
O1 = ["O 1 6300.30A"]
O2b = "O 2 3726.03A"
O2r = "O 2 3728.81A"
O2 = [O2b, O2r]
O3b = "O 3 4958.91A"
O3r = "O 3 5006.84A"
O3 = [O3b, O3r]
Ne3 = ["Ne 3 3868.76A"]
N2 = ["N 2 6583.45A"]
S2 = ["S 2 6730.82A", "S 2 6716.44A"]

ratios = {}

# Balmer decrement, should be [2.79--2.86] (Te, ne, dependent)
# for dust free
ratios["BalmerDecrement"] = [[Ha], [Hb]]

# add reference
ratios["N2"] = [N2, [Ha]]
# add reference
ratios["S2"] = [S2, [Ha]]
ratios["O1"] = [O1, [Ha]]
ratios["R2"] = [[O2b], [Hb]]
ratios["R3"] = [[O3r], [Hb]]
ratios["R23"] = [O3 + O2, [Hb]]
ratios["O32"] = [[O3r], [O2b]]
ratios["Ne3O2"] = [Ne3, [O2b]]

# tuple of available ratios
available_ratios = tuple(ratios.keys())

# dictionary of diagrams
diagrams = {}

# add reference
diagrams["OHNO"] = [ratios["R3"], [Ne3, O2]]

# add reference
diagrams["BPT-NII"] = [[N2, [Ha]], ratios["R3"]]

# add reference
# diagrams["VO78-SII"] = [[S2, [Ha]], R3]

# add reference
# diagrams["VO78-OI"] = [[O1, [Ha]], R3]

available_diagrams = tuple(diagrams.keys())


class Elements:

    """
    A simple class containing various useful lists and dictionaries.

    Attributes:
        non_metals (list, string)
            A list of elements classified as non-metals.
        metals (list, string)
            A list of elements classified as metals.
        all_elements (list, string)
            A list of all elements, functionally the concatenation of metals
            and non-metals.
        alpha_elements (list, string)
            A list of the elements classified as alpha-elements.
        name (dict, string)
            A dictionary holding the full name of each element.
        A (dict, float)
            Atomic mass of each element (in amus).
    """

    non_metals = [
        "H",
        "He",
    ]

    metals = [
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
    ]

    all_elements = non_metals + metals

    # the alpha process elements
    alpha_elements = [
        "O",
        "Ne",
        "Mg",
        "Si",
        "S",
        "Ar",
        "Ca",
        "Ti",
    ]

    # Name
    name = {}
    name["H"] = "Hydrogen"
    name["He"] = "Helium"
    name["Li"] = "Lithium"
    name["Be"] = "Beryllium"
    name["B"] = "Boron"
    name["C"] = "Carbon"
    name["N"] = "Nitrogen"
    name["O"] = "Oxygen"
    name["F"] = "Fluorine"
    name["Ne"] = "Neon"
    name["Na"] = "Sodium"
    name["Mg"] = "Magnesium"
    name["Al"] = "Aluminium"
    name["Si"] = "Silicon"
    name["P"] = "Phosphorus"
    name["S"] = "Sulphur"
    name["Cl"] = "Chlorine"
    name["Ar"] = "Argon"
    name["K"] = "Potassium"
    name["Ca"] = "Calcium"
    name["Sc"] = "Scandium"
    name["Ti"] = "Titanium"
    name["V"] = "Vanadium"
    name["Cr"] = "Chromium"
    name["Mn"] = "Manganese"
    name["Fe"] = "Iron"
    name["Co"] = "Cobalt"
    name["Ni"] = "Nickel"
    name["Cu"] = "Copper"
    name["Zn"] = "Zinc"

    # mass of elements in amus
    A = {}
    A["H"] = 1.008
    A["He"] = 4.003
    A["Li"] = 6.940
    A["Be"] = 9.012
    A["B"] = 10.81
    A["C"] = 12.011
    A["N"] = 14.007
    A["O"] = 15.999
    A["F"] = 18.998
    A["Ne"] = 20.180
    A["Na"] = 22.990
    A["Mg"] = 24.305
    A["Al"] = 26.982
    A["Si"] = 28.085
    A["P"] = 30.973
    A["S"] = 32.06
    A["Cl"] = 35.45
    A["Ar"] = 39.948
    A["K"] = 39.0983
    A["Ca"] = 40.078
    A["Sc"] = 44.955
    A["Ti"] = 47.867
    A["V"] = 50.9415
    A["Cr"] = 51.9961
    A["Mn"] = 54.938
    A["Fe"] = 55.845
    A["Co"] = 58.933
    A["Ni"] = 58.693
    A["Cu"] = 63.546
    A["Zn"] = 65.38
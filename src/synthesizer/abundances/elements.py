
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
        atomic_mass (dict, float)
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

    # atomic mass of each element elements in amus
    atomic_mass = {}
    atomic_mass["H"] = 1.008
    atomic_mass["He"] = 4.003
    atomic_mass["Li"] = 6.940
    atomic_mass["Be"] = 9.012
    atomic_mass["B"] = 10.81
    atomic_mass["C"] = 12.011
    atomic_mass["N"] = 14.007
    atomic_mass["O"] = 15.999
    atomic_mass["F"] = 18.998
    atomic_mass["Ne"] = 20.180
    atomic_mass["Na"] = 22.990
    atomic_mass["Mg"] = 24.305
    atomic_mass["Al"] = 26.982
    atomic_mass["Si"] = 28.085
    atomic_mass["P"] = 30.973
    atomic_mass["S"] = 32.06
    atomic_mass["Cl"] = 35.45
    atomic_mass["Ar"] = 39.948
    atomic_mass["K"] = 39.0983
    atomic_mass["Ca"] = 40.078
    atomic_mass["Sc"] = 44.955
    atomic_mass["Ti"] = 47.867
    atomic_mass["V"] = 50.9415
    atomic_mass["Cr"] = 51.9961
    atomic_mass["Mn"] = 54.938
    atomic_mass["Fe"] = 55.845
    atomic_mass["Co"] = 58.933
    atomic_mass["Ni"] = 58.693
    atomic_mass["Cu"] = 63.546
    atomic_mass["Zn"] = 65.38

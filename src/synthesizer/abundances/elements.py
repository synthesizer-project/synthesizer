
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
        self.name (dict, string)
            A dictionary holding the full self.name of each element.
        self.atomic_mass (dict, float)
            Atomic mass of each element (in amus).
    """

    def __init__(self):

        self.non_metals = [
            "H",
            "He",
        ]

        self.metals = [
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

        self.all_elements = self.non_metals + self.metals

        # the alpha process elements
        self.alpha_elements = [
            "O",
            "Ne",
            "Mg",
            "Si",
            "S",
            "Ar",
            "Ca",
            "Ti",
        ]

        # self.name
        self.name = {}
        self.name["H"] = "Hydrogen"
        self.name["He"] = "Helium"
        self.name["Li"] = "Lithium"
        self.name["Be"] = "Beryllium"
        self.name["B"] = "Boron"
        self.name["C"] = "Carbon"
        self.name["N"] = "Nitrogen"
        self.name["O"] = "Oxygen"
        self.name["F"] = "Fluorine"
        self.name["Ne"] = "Neon"
        self.name["Na"] = "Sodium"
        self.name["Mg"] = "Magnesium"
        self.name["Al"] = "Aluminium"
        self.name["Si"] = "Silicon"
        self.name["P"] = "Phosphorus"
        self.name["S"] = "Sulphur"
        self.name["Cl"] = "Chlorine"
        self.name["Ar"] = "Argon"
        self.name["K"] = "Potassium"
        self.name["Ca"] = "Calcium"
        self.name["Sc"] = "Scandium"
        self.name["Ti"] = "Titanium"
        self.name["V"] = "Vanadium"
        self.name["Cr"] = "Chromium"
        self.name["Mn"] = "Manganese"
        self.name["Fe"] = "Iron"
        self.name["Co"] = "Cobalt"
        self.name["Ni"] = "Nickel"
        self.name["Cu"] = "Copper"
        self.name["Zn"] = "Zinc"

        # atomic mass of each element elements in amus
        self.atomic_mass = {}
        self.atomic_mass["H"] = 1.008
        self.atomic_mass["He"] = 4.003
        self.atomic_mass["Li"] = 6.940
        self.atomic_mass["Be"] = 9.012
        self.atomic_mass["B"] = 10.81
        self.atomic_mass["C"] = 12.011
        self.atomic_mass["N"] = 14.007
        self.atomic_mass["O"] = 15.999
        self.atomic_mass["F"] = 18.998
        self.atomic_mass["Ne"] = 20.180
        self.atomic_mass["Na"] = 22.990
        self.atomic_mass["Mg"] = 24.305
        self.atomic_mass["Al"] = 26.982
        self.atomic_mass["Si"] = 28.085
        self.atomic_mass["P"] = 30.973
        self.atomic_mass["S"] = 32.06
        self.atomic_mass["Cl"] = 35.45
        self.atomic_mass["Ar"] = 39.948
        self.atomic_mass["K"] = 39.0983
        self.atomic_mass["Ca"] = 40.078
        self.atomic_mass["Sc"] = 44.955
        self.atomic_mass["Ti"] = 47.867
        self.atomic_mass["V"] = 50.9415
        self.atomic_mass["Cr"] = 51.9961
        self.atomic_mass["Mn"] = 54.938
        self.atomic_mass["Fe"] = 55.845
        self.atomic_mass["Co"] = 58.933
        self.atomic_mass["Ni"] = 58.693
        self.atomic_mass["Cu"] = 63.546
        self.atomic_mass["Zn"] = 65.38

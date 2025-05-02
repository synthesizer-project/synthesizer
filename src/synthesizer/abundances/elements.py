"""A module implementing a data class for atomic elements."""

from dataclasses import dataclass, field


@dataclass
class Elements:
    """A data class for atomic elements.

    This is a data class containing a various useful information about atomic
    elements. These include lists of elements classified as non-metals, metals
    , alpha elements, as well as all elements. Also contains a dictionary
    mapping the element identification (e.g. "Fe") to the full name (e.g.
    "iron") and a dictionary containing the atomic mass. These are used in the
    creation of custom abundance patterns.

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
            A dictionary holding the full self.name of each element.
        atomic_mass (dict, float)
            Atomic mass of each element (in amus).
    """

    non_metals: list = field(
        default_factory=lambda: [
            "H",
            "He",
        ]
    )

    metals: list = field(
        default_factory=lambda: [
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
    )

    # the alpha process elements
    alpha_elements: list = field(
        default_factory=lambda: [
            "O",
            "Ne",
            "Mg",
            "Si",
            "S",
            "Ar",
            "Ca",
            "Ti",
        ]
    )

    name: dict = field(
        default_factory=lambda: {
            "H": "hydrogen",
            "He": "helium",
            "Li": "lithium",
            "Be": "beryllium",
            "B": "boron",
            "C": "carbon",
            "N": "nitrogen",
            "O": "oxygen",
            "F": "fluorine",
            "Ne": "neon",
            "Na": "sodium",
            "Mg": "magnesium",
            "Al": "aluminium",
            "Si": "silicon",
            "P": "phosphorus",
            "S": "sulphur",
            "Cl": "chlorine",
            "Ar": "argon",
            "K": "potassium",
            "Ca": "calcium",
            "Sc": "scandium",
            "Ti": "titanium",
            "V": "vanadium",
            "Cr": "chromium",
            "Mn": "manganese",
            "Fe": "iron",
            "Co": "cobalt",
            "Ni": "nickel",
            "Cu": "copper",
            "Zn": "zinc",
        }
    )

    # atomic mass of each element elements in amus
    atomic_mass: dict = field(
        default_factory=lambda: {
            "H": 1.008,
            "He": 4.003,
            "Li": 6.940,
            "Be": 9.012,
            "B": 10.81,
            "C": 12.011,
            "N": 14.007,
            "O": 15.999,
            "F": 18.998,
            "Ne": 20.180,
            "Na": 22.990,
            "Mg": 24.305,
            "Al": 26.982,
            "Si": 28.085,
            "P": 30.973,
            "S": 32.06,
            "Cl": 35.45,
            "Ar": 39.948,
            "K": 39.0983,
            "Ca": 40.078,
            "Sc": 44.955,
            "Ti": 47.867,
            "V": 50.9415,
            "Cr": 51.9961,
            "Mn": 54.938,
            "Fe": 55.845,
            "Co": 58.933,
            "Ni": 58.693,
            "Cu": 63.546,
            "Zn": 65.38,
        }
    )

    def __post_init__(self):
        """Post-initialization processing.

        This method is called after the dataclass is initialized. It creates
        a list of all elements by concatenating the non-metals and metals
        lists. This list is stored in the `all_elements` attribute.
        """
        # create list of all elements
        self.all_elements = self.non_metals + self.metals

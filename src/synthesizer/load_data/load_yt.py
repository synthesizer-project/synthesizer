"""A generic loader for converting yt datasets into Synthesizer galaxies.

This loader is intentionally frontend-agnostic. It accepts a yt dataset path,
an already-loaded yt dataset, a yt data container, or a sequence of yt data
containers. Each container is converted into a
`synthesizer.particle.galaxy.Galaxy` object and, where possible, its stellar,
gas, and black hole data are mapped onto the corresponding Synthesizer
particle components.

Because yt frontends expose a wide variety of field names, the loader uses a
two-stage strategy:

1. It tries to infer sensible defaults from the fields available on the
   dataset.
2. Any ambiguous or frontend-specific mapping can be overridden explicitly via
   `field_map` and `particle_types`.

The loader can also preserve additional yt fields as component attributes so
that simulation-specific metadata remain available after loading.
"""

import os
import re
import warnings
from collections.abc import Iterable, Sequence

import numpy as np
from astropy.cosmology import LambdaCDM
from unyt import Mpc, Msun, deg, km, s, unyt_array, yr
from unyt.exceptions import UnitConversionError, UnitOperationError

from synthesizer.exceptions import InconsistentArguments, UnmetDependency
from synthesizer.load_data.utils import age_lookup_table, lookup_age

try:
    import yt
except ImportError as exc:
    raise UnmetDependency(
        "The `yt` module is required to load yt datasets. Install it via "
        "`pip install yt`."
    ) from exc

from ..particle.blackholes import BlackHoles
from ..particle.galaxy import Galaxy

_VECTOR_PROPERTIES = {"coordinates", "velocities"}

_COMPONENT_ALIASES = {
    "stars": ("star", "stars", "stellar", "stellar_particle", "PartType4"),
    "gas": ("gas", "PartType0"),
    "black_holes": (
        "bh",
        "blackhole",
        "blackholes",
        "black_hole",
        "black_holes",
        "PartType5",
    ),
}

_COMMON_PARTICLE_FIELDS = {
    "coordinates": {
        "vectors": ["particle_position", "Coordinates", "position"],
        "components": [
            (
                "particle_position_x",
                "particle_position_y",
                "particle_position_z",
            ),
            ("position_x", "position_y", "position_z"),
            ("x", "y", "z"),
        ],
    },
    "velocities": {
        "vectors": ["particle_velocity", "Velocity", "velocity"],
        "components": [
            (
                "particle_velocity_x",
                "particle_velocity_y",
                "particle_velocity_z",
            ),
            ("velocity_x", "velocity_y", "velocity_z"),
            ("vx", "vy", "vz"),
        ],
    },
    "smoothing_lengths": [
        "particle_smoothing_length",
        "smoothing_length",
        "SmoothingLength",
        "SubfindHsml",
        "softening_length",
    ],
}

_DEFAULT_FIELD_MAP = {
    "stars": {
        "initial_masses": [
            "particle_initial_mass",
            "initial_mass",
            "InitialMass",
            "birth_mass",
            "creation_mass",
            "initial_stellar_mass",
        ],
        "current_masses": [
            "particle_mass",
            "mass",
            "Mass",
            "Masses",
            "current_mass",
        ],
        "metallicities": [
            "metallicity",
            "particle_metallicity",
            "stellar_metallicity",
            "metallicity_fraction",
            "metallicity_fraction_total",
            "Metallicity",
        ],
        "ages": ["particle_age", "age", "stellar_age"],
        "formation_time": [
            "particle_creation_time",
            "creation_time",
            "birth_time",
            "formation_time",
            "star_formation_time",
            "StellarFormationTime",
            "GFM_StellarFormationTime",
        ],
        "formation_redshift": [
            "particle_creation_redshift",
            "creation_redshift",
            "formation_redshift",
        ],
        "formation_scale_factor": [
            "creation_scale_factor",
            "formation_scale_factor",
        ],
        "s_oxygen": [
            "oxygen_fraction",
            "oxygen_abundance",
            "s_oxygen",
            "oxygen_metallicity",
        ],
        "s_hydrogen": [
            "hydrogen_fraction",
            "hydrogen_abundance",
            "s_hydrogen",
        ],
    },
    "gas": {
        "masses": ["particle_mass", "mass", "Mass", "Masses"],
        "metallicities": [
            "metallicity",
            "particle_metallicity",
            "metallicity_fraction",
            "metal_mass_fraction",
            "Metallicity",
        ],
        "star_forming": [
            "star_forming",
            "is_star_forming",
            "star_formation_mask",
        ],
        "star_formation_rate": [
            "star_formation_rate",
            "sfr",
            "StarFormationRate",
        ],
        "dust_masses": ["dust_mass", "Dust_Masses", "particle_dust_mass"],
        "dust_to_metal_ratio": ["dust_to_metal_ratio", "dtm"],
    },
    "gas_fluid": {
        "masses": [("gas", "cell_mass"), ("gas", "mass")],
        "metallicities": [
            ("gas", "metallicity"),
            ("gas", "metallicity_fraction"),
            ("gas", "metal_mass_fraction"),
        ],
        "star_forming": [
            ("gas", "star_forming"),
            ("gas", "is_star_forming"),
            ("gas", "star_formation_mask"),
        ],
        "star_formation_rate": [
            ("gas", "star_formation_rate"),
            ("gas", "sfr"),
        ],
        "dust_masses": [("gas", "dust_mass")],
        "dust_to_metal_ratio": [("gas", "dust_to_metal_ratio")],
        "coordinates": {
            "vectors": [],
            "components": [("x", "y", "z"), ("px", "py", "pz")],
        },
        "velocities": {
            "vectors": [],
            "components": [
                ("velocity_x", "velocity_y", "velocity_z"),
                ("vx", "vy", "vz"),
            ],
        },
        "smoothing_lengths": [
            ("gas", "smoothing_length"),
            ("gas", "cell_width"),
            ("index", "dx"),
        ],
    },
    "black_holes": {
        "masses": [
            "particle_mass",
            "mass",
            "bh_mass",
            "black_hole_mass",
            "blackhole_mass",
            "subgrid_mass",
            "BlackholeMass",
        ],
        "metallicities": [
            "metallicity",
            "bh_metallicity",
            "black_hole_metallicity",
            "blackhole_metallicity",
        ],
        "accretion_rates": [
            "accretion_rate",
            "bh_accretion_rate",
            "black_hole_accretion_rate",
            "blackhole_accretion_rate",
            "BlackholeAccretionRate",
            "BHAR",
        ],
        "accretion_rates_eddington": [
            "accretion_rate_eddington",
            "eddington_fraction",
            "bh_eddington_fraction",
        ],
        "epsilons": [
            "epsilon",
            "radiative_efficiency",
            "bh_radiative_efficiency",
        ],
        "inclinations": ["inclination", "bh_inclination", "disc_inclination"],
        "spins": ["spin", "bh_spin", "dimensionless_spin"],
    },
}

_INDEX_FIELD_TYPES = {"index", "deposit"}


def _is_dataset(obj):
    """Return `True` if the object looks like a yt dataset."""
    return hasattr(obj, "all_data") and hasattr(obj, "field_list")


def _is_data_container(obj):
    """Return `True` if the object looks like a yt data container."""
    return hasattr(obj, "ds") and hasattr(obj, "__getitem__")


def _ensure_sequence(value):
    """Normalise scalars into a list without splitting strings."""
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _available_fields(ds):
    """Return the union of native and derived fields on a yt dataset."""
    fields = set()
    for attr in ("field_list", "derived_field_list"):
        for field in getattr(ds, attr, []):
            if isinstance(field, tuple) and len(field) == 2:
                fields.add(field)
    return fields


def _native_fields(ds):
    """Return the native fields on a yt dataset."""
    fields = set()
    for field in getattr(ds, "field_list", []):
        if isinstance(field, tuple) and len(field) == 2:
            fields.add(field)
    return fields


def _resolve_container_spec(ds, selector):
    """Resolve a selector into a yt data container."""
    if _is_data_container(selector):
        return selector

    if callable(selector):
        container = selector(ds)
        if not _is_data_container(container):
            raise InconsistentArguments(
                "Callable selectors passed to `load_yt` must return a yt "
                "data container."
            )
        return container

    if not isinstance(selector, dict):
        raise InconsistentArguments(
            "Container selectors must be yt data containers, callables, or "
            "dictionary specifications."
        )

    kind = selector.get("type", selector.get("kind", "all_data"))

    if kind == "all_data":
        return ds.all_data()
    if kind == "sphere":
        return ds.sphere(selector["center"], selector["radius"])
    if kind == "box":
        return ds.box(selector["left_edge"], selector["right_edge"])
    if kind == "region":
        return ds.region(
            selector["center"],
            selector["left_edge"],
            selector["right_edge"],
        )

    raise InconsistentArguments(
        f"Unsupported yt selector type `{kind}` passed to `load_yt`."
    )


def _resolve_dataset_and_containers(data_source, data_containers=None):
    """Resolve user input into a yt dataset and a list of data containers."""
    if data_containers is None:
        if isinstance(data_source, (str, os.PathLike)):
            ds = yt.load(str(data_source))
            return ds, [ds.all_data()]

        if _is_dataset(data_source):
            return data_source, [data_source.all_data()]

        if _is_data_container(data_source):
            return data_source.ds, [data_source]

        containers = (
            list(data_source) if isinstance(data_source, Iterable) else []
        )
        if containers and all(_is_data_container(item) for item in containers):
            ds = containers[0].ds
            if any(container.ds is not ds for container in containers[1:]):
                raise InconsistentArguments(
                    "All yt data containers passed to `load_yt` must belong "
                    "to the same dataset."
                )
            return ds, containers

        raise InconsistentArguments(
            "`data_source` must be a yt dataset path, a loaded yt dataset, "
            "a yt data container, or a sequence of yt data containers."
        )

    if isinstance(data_source, (str, os.PathLike)):
        ds = yt.load(str(data_source))
    elif _is_dataset(data_source):
        ds = data_source
    else:
        raise InconsistentArguments(
            "If `data_containers` is provided then `data_source` must be a "
            "yt dataset path or a loaded yt dataset."
        )

    containers = [
        _resolve_container_spec(ds, selector)
        for selector in _ensure_sequence(data_containers)
    ]

    return ds, containers


def _merge_field_map(field_map):
    """Merge user-provided field overrides with the loader defaults."""
    merged = {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in _DEFAULT_FIELD_MAP.items()
    }

    if field_map is None:
        return merged

    for component, component_map in field_map.items():
        if component not in merged:
            merged[component] = {}
        if not isinstance(component_map, dict):
            raise InconsistentArguments(
                "`field_map` must be a dictionary of component mappings."
            )
        merged[component].update(component_map)

    return merged


def _field_candidates(component_map, property_name):
    """Return a list of field candidates for a component property."""
    value = component_map.get(property_name, [])
    if isinstance(value, list):
        return value
    return [value]


def _vector_candidates(component_map, property_name):
    """Return vector field candidates for a component property."""
    return component_map.get(property_name, {"vectors": [], "components": []})


def _find_vector_fields(available_fields, field_types, vector_map):
    """Find a vector field or a matching x/y/z triplet."""
    vector_field = _find_field(
        available_fields,
        field_types,
        vector_map.get("vectors", []),
    )
    if vector_field is not None:
        return vector_field

    for triplet in vector_map.get("components", []):
        fields = tuple(
            _find_field(available_fields, field_types, candidate)
            for candidate in triplet
        )
        if all(field is not None for field in fields):
            return fields

    return None


def _get_vector(container, field_spec):
    """Return a vector field or a stacked x/y/z triplet."""
    if field_spec is None:
        return None

    if isinstance(field_spec, tuple) and len(field_spec) == 2:
        data = container[field_spec]
        if getattr(data, "ndim", 1) == 2 and data.shape[-1] == 3:
            return data
        raise InconsistentArguments(
            f"Field {field_spec} exists but is not a 3-vector."
        )

    fields = [container[field] for field in field_spec]
    unit = getattr(fields[0], "units", None)
    if unit is None:
        return np.column_stack([np.asarray(f) for f in fields])
    return unyt_array(
        np.column_stack([f.to_value(unit) for f in fields]),
        unit,
    )


def _to_unit(data, unit, reference=None):
    """Convert a yt/unyt array into the requested unit if possible."""
    if data is None:
        return None
    if hasattr(data, "to"):
        try:
            return data.to(unit)
        except UnitConversionError:
            if (
                str(getattr(data, "units", "")) == "dimensionless"
                and reference is not None
                and hasattr(reference, "units")
            ):
                return (np.asarray(data) * reference.units).to(unit)
            raise
    return np.asarray(data) * unit


def _to_dimensionless(data):
    """Convert a field to a plain NumPy array."""
    if data is None:
        return None
    if hasattr(data, "to_value"):
        return np.asarray(data.to_value("dimensionless"))
    return np.asarray(data)


def _sanitize_attr_name(field_name):
    """Convert a yt field name into a safe Python attribute name."""
    attr = re.sub(r"\W+", "_", field_name.strip()).strip("_").lower()
    if not attr or attr[0].isdigit():
        attr = f"yt_{attr}"
    return attr


def _dataset_redshift(ds, redshift):
    """Return the galaxy redshift to use for the current dataset."""
    if redshift is not None:
        return float(redshift)

    dataset_redshift = getattr(ds, "current_redshift", None)
    if dataset_redshift is not None:
        return float(dataset_redshift)

    return 0.0


def _dataset_is_cosmological(ds):
    """Return `True` if the dataset carries cosmological metadata."""
    return bool(getattr(ds, "cosmological_simulation", 0))


def _get_astropy_cosmology(ds):
    """Construct an astropy cosmology from yt dataset metadata.

    Uses the general ``LambdaCDM`` rather than ``FlatLambdaCDM`` because yt
    exposes ``omega_matter`` and ``omega_lambda`` independently and does not
    guarantee a flat universe.
    """
    if not _dataset_is_cosmological(ds):
        return None

    required = ("hubble_constant", "omega_matter", "omega_lambda")
    if any(getattr(ds, attr, None) is None for attr in required):
        return None

    return LambdaCDM(
        H0=float(ds.hubble_constant) * 100.0,
        Om0=float(ds.omega_matter),
        Ode0=float(ds.omega_lambda),
    )


def _resolve_stellar_ages(ds, birth_quantity, mode):
    """Convert a birth quantity to stellar ages using shared lookup utilities.

    Args:
        ds: yt dataset.
        birth_quantity: Field data for the birth quantity.
        mode (str): One of ``"ages"``, ``"formation_time"``,
            ``"formation_redshift"``, or ``"formation_scale_factor"``.

    Returns:
        unyt_array or None: Stellar ages in years, or None if conversion
        is not possible.
    """
    if birth_quantity is None:
        return None

    # Direct age with convertible time units
    if mode == "ages" and hasattr(birth_quantity, "to"):
        try:
            return birth_quantity.to("yr")
        except (UnitConversionError, UnitOperationError):
            pass

    # Formation time with physical time units — subtract from current time
    if mode == "formation_time" and hasattr(birth_quantity, "to"):
        current_time = getattr(ds, "current_time", None)
        if current_time is not None:
            try:
                return (current_time - birth_quantity).to("yr")
            except (UnitConversionError, UnitOperationError):
                pass

    cosmo = _get_astropy_cosmology(ds)
    if cosmo is None:
        return None

    redshift = _dataset_redshift(ds, None)

    # Extract a plain array; skip fields that carry incompatible units
    if hasattr(birth_quantity, "to_value"):
        if str(getattr(birth_quantity, "units", "")) != "dimensionless":
            return None
        values = np.asarray(birth_quantity.to_value("dimensionless"))
    else:
        values = np.asarray(birth_quantity)

    # Map to scale factors for the lookup table
    if mode == "formation_redshift":
        valid = np.isfinite(values) & (values >= 0.0)
        scale_factors = np.where(valid, 1.0 / (1.0 + values), np.nan)
    elif mode in ("formation_scale_factor", "formation_time"):
        # Dimensionless formation_time is interpreted as a scale factor
        valid = np.isfinite(values) & (values > 0.0)
        scale_factors = np.where(valid, values, np.nan)
    else:
        return None  # "ages" with no usable units

    if not np.any(np.isfinite(scale_factors)):
        return None

    scale_factors_lut, ages_lut = age_lookup_table(
        cosmo, redshift=redshift, low_lim=1e-4
    )
    current_age = cosmo.age(redshift)
    formation_ages = lookup_age(scale_factors, scale_factors_lut, ages_lut)
    stellar_ages = current_age - formation_ages

    if hasattr(stellar_ages, "to"):
        return stellar_ages.to("yr")
    return unyt_array(np.asarray(stellar_ages) * 1e9, "yr")


def _mask_component_arrays(kwargs, mask):
    """Apply a particle mask to all array-like entries of matching length."""
    if mask is None:
        return kwargs

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        return kwargs

    masked = {}
    for key, value in kwargs.items():
        if value is None or np.isscalar(value):
            masked[key] = value
            continue

        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0 or shape[0] != mask.size:
            masked[key] = value
            continue

        masked[key] = value[mask]

    return masked


def _infer_particle_type(available_fields, component, candidates):
    """Infer the most likely yt particle type for a component."""
    best_type = None
    best_score = 0
    best_specific_matches = 0
    aliases = {alias.lower() for alias in _COMPONENT_ALIASES[component]}

    field_types = sorted({field_type for field_type, _ in available_fields})
    for field_type in field_types:
        if field_type in _INDEX_FIELD_TYPES:
            continue
        if field_type == "all" and component == "gas":
            continue

        score = 0
        specific_matches = 0
        has_required_fields = component not in {"stars", "black_holes"}
        alias_match = field_type.lower() in aliases
        if alias_match:
            score += 10
            if component == "black_holes":
                has_required_fields = True

        if _find_vector_fields(
            available_fields,
            [field_type],
            _COMMON_PARTICLE_FIELDS["coordinates"],
        ):
            score += 2

        for property_name, property_candidates in candidates.items():
            if property_name in _VECTOR_PROPERTIES:
                continue
            if _find_field(
                available_fields,
                [field_type],
                property_candidates,
            ):
                if (
                    component == "black_holes"
                    and property_name == "metallicities"
                ):
                    continue
                score += 1
                specific_matches += 1
                if component == "stars" and property_name in (
                    "ages",
                    "formation_time",
                    "formation_redshift",
                    "formation_scale_factor",
                ):
                    has_required_fields = True
                if component == "black_holes" and property_name in (
                    "accretion_rates",
                    "accretion_rates_eddington",
                    "epsilons",
                    "inclinations",
                    "spins",
                ):
                    has_required_fields = True

        if not has_required_fields:
            continue

        if score > best_score:
            best_type = field_type
            best_score = score
            best_specific_matches = specific_matches

    if best_specific_matches == 0:
        return None

    return best_type


def _has_fluid_gas(available_fields, component_map):
    """Return `True` if fluid gas fields are available."""
    coords = component_map.get(
        "coordinates", {"vectors": [], "components": []}
    )
    if _find_vector_fields(available_fields, ["index"], coords):
        return True
    if _find_field(available_fields, ["gas"], component_map["masses"]):
        return True
    if ("index", "cell_volume") in available_fields:
        return True
    return False


def _collect_extra_fields(container, field_types, field_pool, consumed_fields):
    """Collect unconsumed yt fields as extra component attributes."""
    extras = {}

    for field in sorted(field_pool):
        field_type, field_name = field
        if field_type not in field_types or field in consumed_fields:
            continue

        attr_name = _sanitize_attr_name(field_name)
        if attr_name in extras:
            attr_name = f"yt_{attr_name}"

        try:
            value = container[field]
        except Exception as exc:
            warnings.warn(
                f"Failed to load extra yt field {field}: {exc}",
                stacklevel=2,
            )
            continue

        if np.isscalar(value):
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) == 0:
            continue

        extras[attr_name] = value

    return extras


def _compute_fluid_cell_masses(container, available_fields):
    """Compute gas cell masses from available yt fluid fields."""
    if ("index", "cell_volume") not in available_fields:
        return None

    density_field = None
    for candidate in ("density", "mass_density"):
        field = ("gas", candidate)
        if field in available_fields:
            density_field = field
            break

    if density_field is None:
        return None

    return container[density_field] * container[("index", "cell_volume")]


def _compute_fluid_smoothing_lengths(container, available_fields):
    """Compute an effective cell smoothing length for fluid gas."""
    if ("index", "dx") in available_fields:
        return container[("index", "dx")]

    if ("index", "cell_volume") in available_fields:
        return container[("index", "cell_volume")] ** (1.0 / 3.0)

    return None


def _coerce_bool_array(data):
    """Convert an array-like object into a boolean NumPy array."""
    if data is None:
        return None
    array = np.asarray(data)
    if array.dtype == bool:
        return array
    return array.astype(bool)


def _build_stellar_component(
    container,
    ds,
    available_fields,
    native_fields,
    component_map,
    particle_type,
    load_extra_fields,
    load_derived_extra_fields,
    metallicity_floor,
):
    """Build stellar particle data from a yt data container."""
    if particle_type is None:
        return None

    consumed_fields = set()

    coords_field = _find_vector_fields(
        available_fields,
        [particle_type],
        _vector_candidates(component_map, "coordinates"),
    )
    velocities_field = _find_vector_fields(
        available_fields,
        [particle_type],
        _vector_candidates(component_map, "velocities"),
    )

    coordinates = _get_vector(container, coords_field)
    velocities = _get_vector(container, velocities_field)

    if coords_field is not None:
        consumed_fields.update(
            (coords_field,)
            if isinstance(coords_field[0], str)
            else coords_field
        )
    if velocities_field is not None:
        consumed_fields.update(
            (velocities_field,)
            if isinstance(velocities_field[0], str)
            else velocities_field
        )

    field = _find_field(
        available_fields,
        [particle_type],
        _field_candidates(component_map, "initial_masses"),
    )
    initial_masses = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)

    field = _find_field(
        available_fields,
        [particle_type],
        _field_candidates(component_map, "current_masses"),
    )
    current_masses = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)

    if initial_masses is None:
        initial_masses = current_masses

    field = _find_field(
        available_fields,
        [particle_type],
        _field_candidates(component_map, "metallicities"),
    )
    metallicities = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)
    if metallicities is None and initial_masses is not None:
        metallicities = np.full(initial_masses.size, metallicity_floor)

    field = _find_field(
        available_fields,
        [particle_type],
        _field_candidates(component_map, "ages"),
    )
    ages = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)
        ages = _resolve_stellar_ages(ds, ages, "ages")

    if ages is None:
        for property_name in (
            "formation_time",
            "formation_redshift",
            "formation_scale_factor",
        ):
            field = _find_field(
                available_fields,
                [particle_type],
                _field_candidates(component_map, property_name),
            )
            if field is None:
                continue
            consumed_fields.add(field)
            ages = _resolve_stellar_ages(ds, container[field], property_name)
            if ages is not None:
                break

    field = _find_field(
        available_fields,
        [particle_type],
        _field_candidates(component_map, "smoothing_lengths"),
    )
    smoothing_lengths = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)

    field = _find_field(
        available_fields,
        [particle_type],
        _field_candidates(component_map, "s_oxygen"),
    )
    s_oxygen = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)

    field = _find_field(
        available_fields,
        [particle_type],
        _field_candidates(component_map, "s_hydrogen"),
    )
    s_hydrogen = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)

    if initial_masses is None or ages is None:
        return None

    kwargs = {
        "initial_masses": _to_unit(initial_masses, Msun),
        "ages": _to_unit(ages, yr),
        "metallicities": _to_dimensionless(metallicities),
        "coordinates": _to_unit(coordinates, Mpc),
        "velocities": _to_unit(velocities, km / s),
        "current_masses": _to_unit(current_masses, Msun),
        "smoothing_lengths": _to_unit(
            smoothing_lengths,
            Mpc,
            reference=coordinates,
        ),
        "s_oxygen": _to_dimensionless(s_oxygen),
        "s_hydrogen": _to_dimensionless(s_hydrogen),
    }

    age_values = np.asarray(kwargs["ages"].to_value("yr"))
    star_mask = np.isfinite(age_values) & (age_values >= 0.0)

    extras = {}
    if load_extra_fields:
        field_pool = (
            available_fields if load_derived_extra_fields else native_fields
        )
        extras = _collect_extra_fields(
            container,
            {particle_type},
            field_pool,
            consumed_fields,
        )

    kwargs.update(extras)

    if np.any(~star_mask):
        kwargs = _mask_component_arrays(kwargs, star_mask)

    return kwargs


def _find_field(available_fields, field_types, *candidate_groups):
    """Find the first matching field across one or more candidate groups."""
    for candidates in candidate_groups:
        field = _find_field_from_group(
            available_fields, field_types, candidates
        )
        if field is not None:
            return field
    return None


def _find_field_from_group(available_fields, field_types, candidates):
    """Find the first matching field in a candidate group."""
    for candidate in _ensure_sequence(candidates):
        if isinstance(candidate, tuple):
            if candidate in available_fields:
                return candidate
            continue

        for field_type in field_types:
            field = (field_type, candidate)
            if field in available_fields:
                return field

    return None


def _build_gas_component(
    container,
    available_fields,
    native_fields,
    component_map,
    fluid_map,
    particle_type,
    load_extra_fields,
    load_derived_extra_fields,
    metallicity_floor,
    dtm,
):
    """Build gas data from particle or fluid yt fields."""
    if particle_type is None:
        return None

    consumed_fields = set()
    field_types = (
        ["gas", "index"] if particle_type == "fluid" else [particle_type]
    )
    mapping = fluid_map if particle_type == "fluid" else component_map

    vector_coordinates = _find_vector_fields(
        available_fields,
        field_types,
        _vector_candidates(mapping, "coordinates"),
    )
    vector_velocities = _find_vector_fields(
        available_fields,
        field_types,
        _vector_candidates(mapping, "velocities"),
    )

    coordinates = _get_vector(container, vector_coordinates)
    velocities = _get_vector(container, vector_velocities)

    if vector_coordinates is not None:
        consumed_fields.update(
            (vector_coordinates,)
            if isinstance(vector_coordinates[0], str)
            else vector_coordinates
        )
    if vector_velocities is not None:
        consumed_fields.update(
            (vector_velocities,)
            if isinstance(vector_velocities[0], str)
            else vector_velocities
        )

    field = _find_field(
        available_fields,
        field_types,
        _field_candidates(mapping, "masses"),
    )
    masses = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)
    if masses is None and particle_type == "fluid":
        masses = _compute_fluid_cell_masses(container, available_fields)
        if masses is not None:
            consumed_fields.add(("index", "cell_volume"))
            for density_field in (("gas", "density"), ("gas", "mass_density")):
                if density_field in available_fields:
                    consumed_fields.add(density_field)
                    break

    field = _find_field(
        available_fields,
        field_types,
        _field_candidates(mapping, "metallicities"),
    )
    metallicities = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)
    if metallicities is None and masses is not None:
        metallicities = np.full(masses.size, metallicity_floor)

    field = _find_field(
        available_fields,
        field_types,
        _field_candidates(mapping, "star_forming"),
    )
    star_forming = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)

    field = _find_field(
        available_fields,
        field_types,
        _field_candidates(mapping, "star_formation_rate"),
    )
    sfr = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)
    if star_forming is None and sfr is not None:
        star_forming = np.asarray(sfr) > 0.0

    field = _find_field(
        available_fields,
        field_types,
        _field_candidates(mapping, "dust_masses"),
    )
    dust_masses = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)

    field = _find_field(
        available_fields,
        field_types,
        _field_candidates(mapping, "dust_to_metal_ratio"),
    )
    dust_to_metal_ratio = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)

    field = _find_field(
        available_fields,
        field_types,
        _field_candidates(mapping, "smoothing_lengths"),
    )
    smoothing_lengths = container[field] if field is not None else None
    if field is not None:
        consumed_fields.add(field)
    if smoothing_lengths is None and particle_type == "fluid":
        smoothing_lengths = _compute_fluid_smoothing_lengths(
            container, available_fields
        )

    if masses is None:
        return None

    kwargs = {
        "masses": _to_unit(masses, Msun),
        "metallicities": _to_dimensionless(metallicities),
        "coordinates": _to_unit(coordinates, Mpc),
        "velocities": _to_unit(velocities, km / s),
        "smoothing_lengths": _to_unit(
            smoothing_lengths,
            Mpc,
            reference=coordinates,
        ),
        "star_forming": _coerce_bool_array(star_forming),
        "dust_masses": _to_unit(dust_masses, Msun),
        "dust_to_metal_ratio": (
            _to_dimensionless(dust_to_metal_ratio)
            if dust_to_metal_ratio is not None
            else dtm
        ),
    }

    extras = {}
    if load_extra_fields:
        field_pool = (
            available_fields if load_derived_extra_fields else native_fields
        )
        extra_types = {"gas"} if particle_type == "fluid" else {particle_type}
        extras = _collect_extra_fields(
            container,
            extra_types,
            field_pool,
            consumed_fields,
        )

    kwargs.update(extras)

    return kwargs


def _build_black_hole_component(
    container,
    available_fields,
    native_fields,
    component_map,
    particle_type,
    load_extra_fields,
    load_derived_extra_fields,
    default_black_hole_metallicity,
):
    """Build black hole particle data from a yt data container."""
    if particle_type is None:
        return None

    consumed_fields = set()

    coords_field = _find_vector_fields(
        available_fields,
        [particle_type],
        _vector_candidates(component_map, "coordinates"),
    )
    velocities_field = _find_vector_fields(
        available_fields,
        [particle_type],
        _vector_candidates(component_map, "velocities"),
    )

    coordinates = _get_vector(container, coords_field)
    velocities = _get_vector(container, velocities_field)

    if coords_field is not None:
        consumed_fields.update(
            (coords_field,)
            if isinstance(coords_field[0], str)
            else coords_field
        )
    if velocities_field is not None:
        consumed_fields.update(
            (velocities_field,)
            if isinstance(velocities_field[0], str)
            else velocities_field
        )

    kwargs = {
        "coordinates": _to_unit(coordinates, Mpc),
        "velocities": _to_unit(velocities, km / s),
    }

    for property_name, unit in (
        ("masses", Msun),
        ("accretion_rates", Msun / yr),
        ("accretion_rates_eddington", None),
        ("epsilons", None),
        ("inclinations", deg),
        ("spins", None),
        ("metallicities", None),
        ("smoothing_lengths", Mpc),
    ):
        field = _find_field(
            available_fields,
            [particle_type],
            _field_candidates(component_map, property_name),
        )
        value = container[field] if field is not None else None
        if field is not None:
            consumed_fields.add(field)

        if (
            property_name == "metallicities"
            and value is None
            and coordinates is not None
        ):
            value = np.full(
                coordinates.shape[0], default_black_hole_metallicity
            )

        if unit is None:
            kwargs[property_name] = _to_dimensionless(value)
        else:
            kwargs[property_name] = _to_unit(
                value,
                unit,
                reference=coordinates
                if property_name == "smoothing_lengths"
                else None,
            )

    if kwargs["masses"] is None:
        return None

    extras = {}
    if load_extra_fields:
        field_pool = (
            available_fields if load_derived_extra_fields else native_fields
        )
        extras = _collect_extra_fields(
            container,
            {particle_type},
            field_pool,
            consumed_fields,
        )

    kwargs.update(extras)

    return kwargs


def _infer_centre(stars, gas, black_holes):
    """Infer a galaxy centre from whichever component is available."""
    weighted_components = (
        ("stars", stars, "current_masses"),
        ("gas", gas, "masses"),
        ("black_holes", black_holes, "masses"),
    )

    for _, component, weight_name in weighted_components:
        if component is None:
            continue

        coordinates = component.get("coordinates")
        weights = component.get(weight_name)
        if coordinates is None or weights is None or coordinates.shape[0] == 0:
            continue

        unit = coordinates.units
        coords = coordinates.to_value(unit)
        w = np.ravel(weights.to_value(weights.units))
        if w.shape[0] == coords.shape[0]:
            return unyt_array(np.average(coords, axis=0, weights=w), unit)
        return unyt_array(np.mean(coords, axis=0), unit)

    for component in (stars, gas, black_holes):
        if component is None:
            continue
        coordinates = component.get("coordinates")
        if coordinates is not None and coordinates.shape[0] > 0:
            unit = coordinates.units
            return unyt_array(
                np.mean(coordinates.to_value(unit), axis=0),
                unit,
            )

    return None


def load_yt(
    data_source,
    data_containers=None,
    particle_types=None,
    field_map=None,
    centres=None,
    redshifts=None,
    dtm=0.3,
    metallicity_floor=1e-5,
    default_black_hole_metallicity=0.012,
    load_extra_fields=True,
    load_derived_extra_fields=False,
    galaxy_name_prefix="yt_galaxy",
    verbose=False,
):
    """Load one or more yt selections into Synthesizer particle galaxies.

    Args:
        data_source:
            A yt dataset path, a loaded yt dataset, a yt data container, or a
            sequence of yt data containers.
        data_containers (optional):
            Optional yt data containers or selector specifications to load from
            a dataset. Supported selector dictionaries are `all_data`,
            `sphere`, `box`, and `region`.
        particle_types (dict, optional):
            Optional overrides for the yt particle type used for each
            component. Expected keys are `stars`, `gas`, and `black_holes`.
            Set `particle_types["gas"] = "fluid"` to force loading gas from
            cell fields rather than a gas particle type.
        field_map (dict, optional):
            Optional per-component field overrides. Each component mapping is a
            dictionary keyed by Synthesizer property name and valued by either
            an exact yt field tuple, a field name, or a list of candidate
            fields. Supported components are `stars`, `gas`, `gas_fluid`, and
            `black_holes`.
        centres (optional):
            An explicit centre or sequence of centres for the output galaxies.
            If omitted then a mass-weighted centre is inferred from the loaded
            components.
        redshifts (optional):
            A scalar redshift or sequence of per-galaxy redshifts. If omitted
            then `ds.current_redshift` is used when available, otherwise 0.
        dtm (float):
            Default dust-to-metals ratio to apply to gas when no dust field is
            available.
        metallicity_floor (float):
            Minimum metallicity used when a stellar or gas metallicity field is
            unavailable.
        default_black_hole_metallicity (float):
            Default black hole metallicity used when the dataset does not
            expose one.
        load_extra_fields (bool):
            If `True`, attach unconsumed yt fields as extra attributes on the
            loaded component objects.
        load_derived_extra_fields (bool):
            If `True`, preserve yt derived fields in addition to native fields
            when `load_extra_fields` is enabled.
        galaxy_name_prefix (str):
            Prefix used when naming the output galaxy objects.
        verbose (bool):
            If `True`, print progress information during loading.

    Returns:
        tuple[list[Galaxy], dict]:
            A list of Synthesizer particle galaxies (one per yt data
            container) and a dictionary reporting the inferred yt particle
            types used for each component::

                {"stars": str, "gas": str, "black_holes": str}
    """
    ds, containers = _resolve_dataset_and_containers(
        data_source, data_containers
    )
    available_fields = _available_fields(ds)
    native_fields = _native_fields(ds)

    if verbose:
        print(f"Loaded yt dataset: {ds}")
        print(f"Loading {len(containers)} container(s)...")

    component_maps = _merge_field_map(field_map)
    particle_types = particle_types or {}

    redshift_values = (
        _ensure_sequence(redshifts) if redshifts is not None else []
    )
    centre_values = _ensure_sequence(centres) if centres is not None else []

    if redshift_values and len(redshift_values) not in (1, len(containers)):
        raise InconsistentArguments(
            "`redshifts` must be either a scalar or match the number of yt "
            "data containers being loaded."
        )
    if centre_values and len(centre_values) not in (1, len(containers)):
        raise InconsistentArguments(
            "`centres` must be either a single centre or match the number of "
            "yt data containers being loaded."
        )

    if "stars" in particle_types:
        star_type = particle_types["stars"]
    else:
        star_type = _infer_particle_type(
            available_fields,
            "stars",
            component_maps["stars"],
        )
        if star_type is None:
            warnings.warn(
                "Could not infer a yt particle type for stars. "
                "Supply `particle_types={'stars': '<type>'}` if the dataset "
                "contains stellar particles.",
                stacklevel=2,
            )

    if "black_holes" in particle_types:
        bh_type = particle_types["black_holes"]
    else:
        bh_type = _infer_particle_type(
            available_fields,
            "black_holes",
            component_maps["black_holes"],
        )

    if "gas" in particle_types:
        gas_type = particle_types["gas"]
    else:
        gas_type = _infer_particle_type(
            available_fields,
            "gas",
            component_maps["gas"],
        )
        if gas_type is None and _has_fluid_gas(
            available_fields,
            component_maps["gas_fluid"],
        ):
            gas_type = "fluid"

    if verbose:
        print(
            f"Inferred particle types — "
            f"stars: {star_type}, gas: {gas_type}, black_holes: {bh_type}"
        )

    field_types = {"stars": star_type, "gas": gas_type, "black_holes": bh_type}

    _common = {
        "coordinates": _COMMON_PARTICLE_FIELDS["coordinates"],
        "velocities": _COMMON_PARTICLE_FIELDS["velocities"],
        "smoothing_lengths": _COMMON_PARTICLE_FIELDS["smoothing_lengths"],
    }
    star_map = {**component_maps["stars"], **_common}
    gas_map = {**component_maps["gas"], **_common}
    bh_map = {**component_maps["black_holes"], **_common}

    galaxies = []

    for index, container in enumerate(containers):
        redshift = _dataset_redshift(
            ds,
            redshift_values[0 if len(redshift_values) == 1 else index]
            if redshift_values
            else None,
        )

        stars = _build_stellar_component(
            container,
            ds,
            available_fields,
            native_fields,
            star_map,
            star_type,
            load_extra_fields,
            load_derived_extra_fields,
            metallicity_floor,
        )

        gas = _build_gas_component(
            container,
            available_fields,
            native_fields,
            gas_map,
            component_maps["gas_fluid"],
            gas_type,
            load_extra_fields,
            load_derived_extra_fields,
            metallicity_floor,
            dtm,
        )

        black_holes = _build_black_hole_component(
            container,
            available_fields,
            native_fields,
            bh_map,
            bh_type,
            load_extra_fields,
            load_derived_extra_fields,
            default_black_hole_metallicity,
        )

        galaxy_centre = None
        if centre_values:
            galaxy_centre = centre_values[
                0 if len(centre_values) == 1 else index
            ]
            galaxy_centre = _to_unit(galaxy_centre, Mpc)
        else:
            galaxy_centre = _infer_centre(stars, gas, black_holes)

        galaxy = Galaxy(
            name=f"{galaxy_name_prefix}_{index}",
            redshift=redshift,
            centre=galaxy_centre,
        )

        if stars is not None:
            galaxy.load_stars(**stars)

        if gas is not None:
            galaxy.load_gas(**gas)

        if black_holes is not None:
            black_hole_kwargs = dict(black_holes)
            galaxy.black_holes = BlackHoles(
                masses=black_hole_kwargs.pop("masses"),
                redshift=redshift,
                centre=galaxy_centre,
                **black_hole_kwargs,
            )

        if verbose:
            print(
                f"  [{index + 1}/{len(containers)}] Loaded galaxy "
                f"{galaxy_name_prefix}_{index} "
                f"(z={redshift:.4f})"
            )

        galaxies.append(galaxy)

    return galaxies, field_types

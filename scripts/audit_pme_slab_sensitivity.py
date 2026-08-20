from pathlib import Path
import csv
import gc
import hashlib
import json

import numpy as np
import openmm
from openmm import unit


# ============================================================
# PURPOSE
# ============================================================
#
# Static PME slab-sensitivity audit.
#
# We use the EXACT 5.5 ns endpoint coordinates and evaluate
# them at several periodic z box lengths:
#
#     12, 16, 20, 24, 30 nm
#
# x/y box vectors, coordinates, force field, cutoff,
# PME tolerance, restraints, walls, etc. remain unchanged.
#
# NO MD.
# NO MINIMIZATION.
# NO SOURCE FILES MODIFIED.
#
# We separate NonbondedForce into:
#
#   group 30 = direct-space nonbonded
#   group 31 = reciprocal-space PME
#
# This lets us see whether the reciprocal PME contribution
# changes significantly with Lz.
# ============================================================


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

SYSTEM_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "pyrene_peg5_graphene_one_sided_opc_supported_pme_production_safe.xml"
)

POSITIONS_NPY = (
    ROOT
    / "parameters"
    / "combined"
    / "pyrene_peg5_graphene_one_sided_opc_production_300K_5500ps_positions_nm.npy"
)

OUTPUT_CSV = (
    ROOT
    / "analysis"
    / "pme_slab_sensitivity_5500ps.csv"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "pme_slab_sensitivity_5500ps.json"
)


# ============================================================
# SYSTEM CONSTANTS
# ============================================================

EXPECTED_PARTICLES = 28224
EXPECTED_VIRTUAL_SITES = 6101

GRAPHENE_C = np.arange(
    0,
    1250,
    dtype=int,
)

GRAPHENE_ALL = np.arange(
    0,
    3750,
    dtype=int,
)

LIGAND = np.arange(
    3750,
    3820,
    dtype=int,
)

# Exact 16 aromatic pyrene atoms in FULL OpenMM indexing.
PYRENE = np.arange(
    3752,
    3768,
    dtype=int,
)

WATER_ALL = np.arange(
    3820,
    EXPECTED_PARTICLES,
    dtype=int,
)


LZ_VALUES_NM = [
    12.0,
    16.0,
    20.0,
    24.0,
    30.0,
]

REFERENCE_LZ_NM = 30.0


DIRECT_GROUP = 30
RECIPROCAL_GROUP = 31


ENERGY_UNIT = (
    unit.kilojoule_per_mole
)

FORCE_UNIT = (
    unit.kilojoule_per_mole
    /
    unit.nanometer
)

CHARGE_UNIT = (
    unit.elementary_charge
)


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def vec3_to_nm(vector):
    """
    Convert an OpenMM box Vec3/Quantity to numpy nm.
    """

    try:
        v = vector.value_in_unit(
            unit.nanometer
        )

        return np.array(
            [
                float(v[0]),
                float(v[1]),
                float(v[2]),
            ],
            dtype=float,
        )

    except AttributeError:

        return np.array(
            [
                float(vector[0]),
                float(vector[1]),
                float(vector[2]),
            ],
            dtype=float,
        )


def make_vec3_nm(array):
    return (
        openmm.Vec3(
            float(array[0]),
            float(array[1]),
            float(array[2]),
        )
        *
        unit.nanometer
    )


def get_energy_and_forces(
    context,
    groups=None,
):
    kwargs = {
        "getEnergy": True,
        "getForces": True,
    }

    if groups is not None:
        kwargs["groups"] = groups

    state = context.getState(
        **kwargs
    )

    energy = float(
        state
        .getPotentialEnergy()
        .value_in_unit(
            ENERGY_UNIT
        )
    )

    forces = np.asarray(
        state
        .getForces(
            asNumpy=True
        )
        .value_in_unit(
            FORCE_UNIT
        ),
        dtype=float,
    )

    return energy, forces


def sum_fz(
    forces,
    indices,
):
    return float(
        np.sum(
            forces[
                indices,
                2,
            ]
        )
    )


def rms_force_difference(
    forces_a,
    forces_b,
):
    diff = (
        forces_a
        -
        forces_b
    )

    magnitude_squared = np.sum(
        diff * diff,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(
                magnitude_squared
            )
        )
    )


def max_force_difference(
    forces_a,
    forces_b,
):
    diff = (
        forces_a
        -
        forces_b
    )

    magnitude = np.sqrt(
        np.sum(
            diff * diff,
            axis=1,
        )
    )

    return float(
        np.max(
            magnitude
        )
    )


def get_platform():
    """
    Prefer CUDA because this is the platform used for
    production and it makes the PME evaluations fast.

    Fall back to CPU if CUDA is unavailable.
    """

    try:
        platform = openmm.Platform.getPlatformByName(
            "CUDA"
        )

        properties = {
            "Precision": "mixed",
        }

        return (
            platform,
            properties,
        )

    except Exception:
        platform = openmm.Platform.getPlatformByName(
            "CPU"
        )

        return (
            platform,
            {},
        )


# ============================================================
# FILE CHECKS
# ============================================================

print()
print("=" * 78)
print("PME SLAB SENSITIVITY AUDIT")
print("=" * 78)
print()

print("System XML:")
print(f"  {SYSTEM_XML}")

print()
print("Endpoint positions:")
print(f"  {POSITIONS_NPY}")
print()


if not SYSTEM_XML.exists():
    raise FileNotFoundError(
        SYSTEM_XML
    )

if not POSITIONS_NPY.exists():
    raise FileNotFoundError(
        POSITIONS_NPY
    )


# ============================================================
# LOAD BASE SYSTEM AND POSITIONS
# ============================================================

xml_text = SYSTEM_XML.read_text(
    encoding="utf-8"
)

base_system = openmm.XmlSerializer.deserialize(
    xml_text
)

positions_nm = np.load(
    POSITIONS_NPY
)


n_particles = (
    base_system.getNumParticles()
)


print(
    f"OpenMM version:       "
    f"{openmm.version.full_version}"
)

print(
    f"Particles:            "
    f"{n_particles}"
)

print(
    f"Coordinate shape:     "
    f"{positions_nm.shape}"
)


if n_particles != EXPECTED_PARTICLES:
    raise RuntimeError(
        f"Expected {EXPECTED_PARTICLES} particles, "
        f"found {n_particles}"
    )


if positions_nm.shape != (
    EXPECTED_PARTICLES,
    3,
):
    raise RuntimeError(
        "Endpoint position array has unexpected shape."
    )


# ============================================================
# VIRTUAL-SITE CHECK
# ============================================================

virtual_sites = [
    i
    for i in range(
        n_particles
    )
    if base_system.isVirtualSite(i)
]


print(
    f"Virtual sites:        "
    f"{len(virtual_sites)}"
)


if len(virtual_sites) != EXPECTED_VIRTUAL_SITES:
    raise RuntimeError(
        f"Expected {EXPECTED_VIRTUAL_SITES} virtual sites, "
        f"found {len(virtual_sites)}"
    )


# ============================================================
# ORIGINAL BOX
# ============================================================

a0, b0, c0 = (
    base_system
    .getDefaultPeriodicBoxVectors()
)

a_nm = vec3_to_nm(a0)
b_nm = vec3_to_nm(b0)
c_nm = vec3_to_nm(c0)

original_lz_nm = float(
    np.linalg.norm(
        c_nm
    )
)


print()
print("Original periodic box:")
print(
    "  a = "
    f"{a_nm}"
)

print(
    "  b = "
    f"{b_nm}"
)

print(
    "  c = "
    f"{c_nm}"
)

print(
    f"  |c| = "
    f"{original_lz_nm:.6f} nm"
)


if abs(
    original_lz_nm
    -
    12.0
) > 1.0e-5:
    raise RuntimeError(
        "Expected original Lz = 12 nm."
    )


c_direction = (
    c_nm
    /
    original_lz_nm
)


# ============================================================
# POSITION SANITY CHECK
# ============================================================

z_min_nm = float(
    np.min(
        positions_nm[:, 2]
    )
)

z_max_nm = float(
    np.max(
        positions_nm[:, 2]
    )
)


print()
print("Coordinate z extent:")
print(
    f"  min z = "
    f"{z_min_nm:.6f} nm"
)

print(
    f"  max z = "
    f"{z_max_nm:.6f} nm"
)


if z_max_nm >= min(
    LZ_VALUES_NM
):
    raise RuntimeError(
        "Particles extend beyond the smallest test box."
    )


# ============================================================
# FIND NONBONDED FORCE
# ============================================================

nonbonded_indices = []

for force_index in range(
    base_system.getNumForces()
):
    force = base_system.getForce(
        force_index
    )

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):
        nonbonded_indices.append(
            force_index
        )


print()
print(
    f"NonbondedForce count: "
    f"{len(nonbonded_indices)}"
)


if len(nonbonded_indices) != 1:
    raise RuntimeError(
        "Expected exactly one NonbondedForce."
    )


nonbonded_index = (
    nonbonded_indices[0]
)

base_nb = base_system.getForce(
    nonbonded_index
)


print(
    f"NonbondedForce index: "
    f"{nonbonded_index}"
)

print(
    f"Nonbonded method:     "
    f"{base_nb.getNonbondedMethod()}"
)

print(
    f"Cutoff:               "
    f"{base_nb.getCutoffDistance()}"
)

print(
    f"Ewald tolerance:      "
    f"{base_nb.getEwaldErrorTolerance():.8g}"
)

print(
    f"LJ dispersion corr.:  "
    f"{base_nb.getUseDispersionCorrection()}"
)


if (
    base_nb.getNonbondedMethod()
    !=
    openmm.NonbondedForce.PME
):
    raise RuntimeError(
        "Expected PME NonbondedForce."
    )


# ============================================================
# CHECK GROUP 30/31 ARE UNUSED
# ============================================================

for force_index in range(
    base_system.getNumForces()
):
    if force_index == nonbonded_index:
        continue

    group = (
        base_system
        .getForce(force_index)
        .getForceGroup()
    )

    if group in (
        DIRECT_GROUP,
        RECIPROCAL_GROUP,
    ):
        raise RuntimeError(
            f"Force group {group} already used by "
            f"force {force_index}."
        )


# ============================================================
# CHARGES + DIPOLE
# ============================================================

charges_e = np.zeros(
    n_particles,
    dtype=float,
)


for i in range(
    n_particles
):
    charge, sigma, epsilon = (
        base_nb
        .getParticleParameters(i)
    )

    charges_e[i] = float(
        charge.value_in_unit(
            CHARGE_UNIT
        )
    )


total_charge_e = float(
    np.sum(
        charges_e
    )
)


# Because the full system is neutral, Mz is origin-independent.
dipole_z_e_nm = float(
    np.sum(
        charges_e
        *
        positions_nm[:, 2]
    )
)

dipole_z_debye = (
    dipole_z_e_nm
    *
    48.0320427
)


print()
print("Charge diagnostics:")

print(
    f"  Total charge: "
    f"{total_charge_e:+.10e} e"
)

print(
    f"  Mz:           "
    f"{dipole_z_e_nm:+.8f} e nm"
)

print(
    f"  Mz:           "
    f"{dipole_z_debye:+.3f} D"
)


if abs(
    total_charge_e
) > 1.0e-5:
    raise RuntimeError(
        "System is not sufficiently neutral for this "
        "simple slab-dipole interpretation."
    )


# ============================================================
# PROVENANCE HASHES
# ============================================================

xml_sha256 = sha256_file(
    SYSTEM_XML
)

positions_sha256 = sha256_file(
    POSITIONS_NPY
)


print()
print("Provenance:")

print(
    f"  XML SHA256:       "
    f"{xml_sha256}"
)

print(
    f"  positions SHA256: "
    f"{positions_sha256}"
)


# ============================================================
# PLATFORM
# ============================================================

platform, platform_properties = (
    get_platform()
)


print()
print(
    f"Platform: "
    f"{platform.getName()}"
)

if platform_properties:
    print(
        f"Properties: "
        f"{platform_properties}"
    )


# ============================================================
# RUN STATIC BOX-HEIGHT AUDIT
# ============================================================

rows = []

reciprocal_forces_by_lz = {}


print()
print("=" * 78)
print("STATIC SINGLE-POINT EVALUATIONS")
print("=" * 78)


for lz_nm in LZ_VALUES_NM:

    print()
    print(
        f"Evaluating Lz = "
        f"{lz_nm:.1f} nm ..."
    )


    # --------------------------------------------------------
    # Fresh System for every box height.
    # --------------------------------------------------------

    system = openmm.XmlSerializer.deserialize(
        xml_text
    )


    nb = system.getForce(
        nonbonded_index
    )


    # --------------------------------------------------------
    # Separate direct and reciprocal PME force groups.
    #
    # Direct-space NonbondedForce = group 30
    # Reciprocal PME              = group 31
    # --------------------------------------------------------

    nb.setForceGroup(
        DIRECT_GROUP
    )

    nb.setReciprocalSpaceForceGroup(
        RECIPROCAL_GROUP
    )


    # --------------------------------------------------------
    # Change ONLY c.
    #
    # a and b are untouched.
    # c direction is untouched.
    # --------------------------------------------------------

    new_c_nm = (
        c_direction
        *
        lz_nm
    )


    system.setDefaultPeriodicBoxVectors(
        make_vec3_nm(
            a_nm
        ),
        make_vec3_nm(
            b_nm
        ),
        make_vec3_nm(
            new_c_nm
        ),
    )


    # --------------------------------------------------------
    # We never integrate, but OpenMM requires an Integrator
    # to construct a Context.
    # --------------------------------------------------------

    integrator = openmm.VerletIntegrator(
        0.001
        *
        unit.picoseconds
    )


    context = openmm.Context(
        system,
        integrator,
        platform,
        platform_properties,
    )


    context.setPositions(
        positions_nm
        *
        unit.nanometer
    )


    # Recompute OPC virtual sites from their parent atoms.
    # This does not alter the source .npy file.
    context.computeVirtualSites()


    # --------------------------------------------------------
    # Confirm actual Context box
    # --------------------------------------------------------

    box_state = context.getState()

    aa, bb, cc = (
        box_state.getPeriodicBoxVectors()
    )

    cc_nm = vec3_to_nm(
        cc
    )

    actual_lz_nm = float(
        np.linalg.norm(
            cc_nm
        )
    )


    if abs(
        actual_lz_nm
        -
        lz_nm
    ) > 1.0e-6:
        raise RuntimeError(
            "Context Lz does not match requested Lz."
        )


    # --------------------------------------------------------
    # Get the coordinates actually used by Context,
    # including recomputed virtual sites.
    # --------------------------------------------------------

    position_state = context.getState(
        getPositions=True
    )

    context_positions_nm = np.asarray(
        position_state
        .getPositions(
            asNumpy=True
        )
        .value_in_unit(
            unit.nanometer
        ),
        dtype=float,
    )


    mz_context_e_nm = float(
        np.sum(
            charges_e
            *
            context_positions_nm[:, 2]
        )
    )


    # --------------------------------------------------------
    # ACTUAL PME PARAMETERS USED BY THIS CONTEXT
    #
    # Important because OpenMM may choose different PME
    # grid dimensions as Lz changes when only the error
    # tolerance is specified.
    # --------------------------------------------------------

    (
        pme_alpha,
        pme_nx,
        pme_ny,
        pme_nz,
    ) = nb.getPMEParametersInContext(
        context
    )

    pme_alpha = float(
        pme_alpha
    )

    pme_nx = int(
        pme_nx
    )

    pme_ny = int(
        pme_ny
    )

    pme_nz = int(
        pme_nz
    )

    print(
        f"  PME actual      = "
        f"alpha={pme_alpha:.8f}  "
        f"grid={pme_nx}x{pme_ny}x{pme_nz}"
    )


    # --------------------------------------------------------
    # ALL FORCES
    # --------------------------------------------------------

    (
        total_energy,
        total_forces,
    ) = get_energy_and_forces(
        context
    )


    # --------------------------------------------------------
    # DIRECT NONBONDED ONLY
    # --------------------------------------------------------

    (
        direct_energy,
        direct_forces,
    ) = get_energy_and_forces(
        context,
        groups={
            DIRECT_GROUP
        },
    )


    # --------------------------------------------------------
    # RECIPROCAL PME ONLY
    # --------------------------------------------------------

    (
        reciprocal_energy,
        reciprocal_forces,
    ) = get_energy_and_forces(
        context,
        groups={
            RECIPROCAL_GROUP
        },
    )


    # --------------------------------------------------------
    # BOTH NONBONDED COMPONENTS
    # --------------------------------------------------------

    (
        nb_energy,
        nb_forces,
    ) = get_energy_and_forces(
        context,
        groups={
            DIRECT_GROUP,
            RECIPROCAL_GROUP,
        },
    )


    # --------------------------------------------------------
    # Consistency check:
    #
    # separate direct + reciprocal should reproduce
    # combined nonbonded energy.
    # --------------------------------------------------------

    energy_split_error = float(
        nb_energy
        -
        (
            direct_energy
            +
            reciprocal_energy
        )
    )


    if abs(
        energy_split_error
    ) > 1.0e-2:
        raise RuntimeError(
            "Nonbonded force-group energy split failed: "
            f"{energy_split_error} kJ/mol"
        )


    # --------------------------------------------------------
    # Store reciprocal forces for later comparison to 30 nm.
    # --------------------------------------------------------

    reciprocal_forces_by_lz[
        lz_nm
    ] = reciprocal_forces.copy()


    # --------------------------------------------------------
    # Net z-force diagnostics
    #
    # Whole-ligand sums are especially useful because
    # internal ligand pair forces cancel in the sum.
    # --------------------------------------------------------

    row = {
        "lz_nm": float(
            lz_nm
        ),

        "total_potential_kJ_mol": float(
            total_energy
        ),

        "nonbonded_total_kJ_mol": float(
            nb_energy
        ),

        "nonbonded_direct_kJ_mol": float(
            direct_energy
        ),

        "pme_reciprocal_kJ_mol": float(
            reciprocal_energy
        ),

        "energy_split_error_kJ_mol": float(
            energy_split_error
        ),

        "dipole_z_e_nm": float(
            mz_context_e_nm
        ),

        "pme_alpha_per_nm": float(
            pme_alpha
        ),

        "pme_nx": int(
            pme_nx
        ),

        "pme_ny": int(
            pme_ny
        ),

        "pme_nz": int(
            pme_nz
        ),

        # -------- whole ligand --------
        "ligand_total_Fz_kJ_mol_nm": sum_fz(
            total_forces,
            LIGAND,
        ),

        "ligand_nonbonded_Fz_kJ_mol_nm": sum_fz(
            nb_forces,
            LIGAND,
        ),

        "ligand_reciprocal_Fz_kJ_mol_nm": sum_fz(
            reciprocal_forces,
            LIGAND,
        ),

        # -------- pyrene aromatic subset --------
        "pyrene_reciprocal_Fz_kJ_mol_nm": sum_fz(
            reciprocal_forces,
            PYRENE,
        ),

        # -------- graphene --------
        "graphene_all_reciprocal_Fz_kJ_mol_nm": sum_fz(
            reciprocal_forces,
            GRAPHENE_ALL,
        ),

        "graphene_carbon_reciprocal_Fz_kJ_mol_nm": sum_fz(
            reciprocal_forces,
            GRAPHENE_C,
        ),

        # -------- water --------
        "water_reciprocal_Fz_kJ_mol_nm": sum_fz(
            reciprocal_forces,
            WATER_ALL,
        ),
    }


    rows.append(
        row
    )


    print(
        f"  total PE       = "
        f"{total_energy: .6f} kJ/mol"
    )

    print(
        f"  NB direct      = "
        f"{direct_energy: .6f} kJ/mol"
    )

    print(
        f"  PME reciprocal = "
        f"{reciprocal_energy: .6f} kJ/mol"
    )

    print(
        f"  ligand PME Fz  = "
        f"{row['ligand_reciprocal_Fz_kJ_mol_nm']:+.6f} "
        f"kJ/mol/nm"
    )


    # Important on CUDA: release each Context before
    # constructing the next one.
    del context
    del integrator
    del system

    gc.collect()


# ============================================================
# COMPARE EVERYTHING TO 30 nm REFERENCE
# ============================================================

reference_row = None

for row in rows:
    if abs(
        row["lz_nm"]
        -
        REFERENCE_LZ_NM
    ) < 1.0e-12:

        reference_row = row
        break


if reference_row is None:
    raise RuntimeError(
        "Reference Lz row not found."
    )


reference_reciprocal_forces = (
    reciprocal_forces_by_lz[
        REFERENCE_LZ_NM
    ]
)


for row in rows:

    lz_nm = row["lz_nm"]

    row[
        "delta_total_PE_vs_30nm_kJ_mol"
    ] = (
        row[
            "total_potential_kJ_mol"
        ]
        -
        reference_row[
            "total_potential_kJ_mol"
        ]
    )


    row[
        "delta_direct_vs_30nm_kJ_mol"
    ] = (
        row[
            "nonbonded_direct_kJ_mol"
        ]
        -
        reference_row[
            "nonbonded_direct_kJ_mol"
        ]
    )


    row[
        "delta_reciprocal_vs_30nm_kJ_mol"
    ] = (
        row[
            "pme_reciprocal_kJ_mol"
        ]
        -
        reference_row[
            "pme_reciprocal_kJ_mol"
        ]
    )


    row[
        "delta_ligand_recip_Fz_vs_30nm_kJ_mol_nm"
    ] = (
        row[
            "ligand_reciprocal_Fz_kJ_mol_nm"
        ]
        -
        reference_row[
            "ligand_reciprocal_Fz_kJ_mol_nm"
        ]
    )


    row[
        "delta_pyrene_recip_Fz_vs_30nm_kJ_mol_nm"
    ] = (
        row[
            "pyrene_reciprocal_Fz_kJ_mol_nm"
        ]
        -
        reference_row[
            "pyrene_reciprocal_Fz_kJ_mol_nm"
        ]
    )


    current_recip_forces = (
        reciprocal_forces_by_lz[
            lz_nm
        ]
    )


    row[
        "rms_particle_recip_force_delta_vs_30nm_kJ_mol_nm"
    ] = rms_force_difference(
        current_recip_forces,
        reference_reciprocal_forces,
    )


    row[
        "max_particle_recip_force_delta_vs_30nm_kJ_mol_nm"
    ] = max_force_difference(
        current_recip_forces,
        reference_reciprocal_forces,
    )


# ============================================================
# SAVE CSV
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fieldnames = list(
    rows[0].keys()
)


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# SAVE JSON
# ============================================================

metadata = {
    "purpose": (
        "Static single-point PME slab-height sensitivity audit "
        "at the exact 5.5 ns endpoint."
    ),

    "no_dynamics_performed": True,

    "source_system_xml": str(
        SYSTEM_XML
    ),

    "source_positions_npy": str(
        POSITIONS_NPY
    ),

    "system_xml_sha256": (
        xml_sha256
    ),

    "positions_sha256": (
        positions_sha256
    ),

    "openmm_version": (
        openmm.version.full_version
    ),

    "platform": (
        platform.getName()
    ),

    "platform_properties": (
        platform_properties
    ),

    "n_particles": int(
        n_particles
    ),

    "n_virtual_sites": int(
        len(
            virtual_sites
        )
    ),

    "total_charge_e": float(
        total_charge_e
    ),

    "dipole_z_e_nm": float(
        dipole_z_e_nm
    ),

    "dipole_z_debye": float(
        dipole_z_debye
    ),

    "original_box_a_nm": (
        a_nm.tolist()
    ),

    "original_box_b_nm": (
        b_nm.tolist()
    ),

    "original_box_c_nm": (
        c_nm.tolist()
    ),

    "tested_lz_nm": (
        LZ_VALUES_NM
    ),

    "reference_lz_nm": (
        REFERENCE_LZ_NM
    ),

    "results": rows,
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        metadata,
        handle,
        indent=2,
    )


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 110)
print("SENSITIVITY RELATIVE TO Lz = 30 nm")
print("=" * 110)

print(
    " Lz(nm) | "
    "alpha      | "
    "PME grid           | "
    "dE_total(kJ/mol) | "
    "dE_PMErecip(kJ/mol) | "
    "Lig PME Fz | "
    "dLig PME Fz | "
    "RMS dF_recip"
)

print("-" * 110)


for row in rows:

    print(
        f" {row['lz_nm']:6.1f} | "
        f"{row['pme_alpha_per_nm']:10.6f} | "
        f"{row['pme_nx']:3d}x"
        f"{row['pme_ny']:3d}x"
        f"{row['pme_nz']:3d} | "
        f"{row['delta_total_PE_vs_30nm_kJ_mol']:16.6f} | "
        f"{row['delta_reciprocal_vs_30nm_kJ_mol']:20.6f} | "
        f"{row['ligand_reciprocal_Fz_kJ_mol_nm']:10.5f} | "
        f"{row['delta_ligand_recip_Fz_vs_30nm_kJ_mol_nm']:12.5f} | "
        f"{row['rms_particle_recip_force_delta_vs_30nm_kJ_mol_nm']:11.5f}"
    )


# ============================================================
# SPECIFIC 12 nm VS 30 nm RESULT
# ============================================================

row_12 = next(
    row
    for row in rows
    if abs(
        row["lz_nm"]
        -
        12.0
    ) < 1.0e-12
)


print()
print("=" * 78)
print("CURRENT 12 nm BOX VS 30 nm REFERENCE")
print("=" * 78)

print(
    f"Delta total potential energy: "
    f"{row_12['delta_total_PE_vs_30nm_kJ_mol']:+.6f} "
    f"kJ/mol"
)

print(
    f"Delta PME reciprocal energy:  "
    f"{row_12['delta_reciprocal_vs_30nm_kJ_mol']:+.6f} "
    f"kJ/mol"
)

print(
    f"Delta ligand PME Fz:          "
    f"{row_12['delta_ligand_recip_Fz_vs_30nm_kJ_mol_nm']:+.6f} "
    f"kJ/mol/nm"
)

print(
    f"Delta pyrene PME Fz:          "
    f"{row_12['delta_pyrene_recip_Fz_vs_30nm_kJ_mol_nm']:+.6f} "
    f"kJ/mol/nm"
)

print(
    f"RMS particle PME force delta: "
    f"{row_12['rms_particle_recip_force_delta_vs_30nm_kJ_mol_nm']:.6f} "
    f"kJ/mol/nm"
)

print(
    f"Max particle PME force delta: "
    f"{row_12['max_particle_recip_force_delta_vs_30nm_kJ_mol_nm']:.6f} "
    f"kJ/mol/nm"
)


print()
print("IMPORTANT:")
print(
    "This audit measures static Lz sensitivity. "
    "It does NOT by itself prove that a 3D PME slab "
    "artifact is absent."
)

print(
    "We will interpret the magnitude and convergence "
    "before deciding whether short MD sensitivity runs "
    "or a formal slab correction are required."
)


print()
print(f"CSV:  {OUTPUT_CSV}")
print(f"JSON: {OUTPUT_JSON}")

print()
print("PME_SLAB_SENSITIVITY_AUDIT_COMPLETE")
print("=" * 78)
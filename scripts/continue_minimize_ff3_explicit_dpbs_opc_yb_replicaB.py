from pathlib import Path
import json

import numpy as np

from openmm import openmm, unit


# ============================================================
# CONTINUED FULL-SYSTEM ENERGY MINIMIZATION
#
# SYSTEM:
#
#   supported IFF graphene
#   +
#   unrestrained Pyrene-PEG5
#   +
#   pruned exact OPC water
#   +
#   explicit 1x DPBS: Na+, K+, Cl-, H2PO4-, HPO4^2-
#   +
#   PME
#   +
#   graphene support
#   +
#   dynamic Yeh-Berkowitz slab correction
#
#   NO runtime water wall
#
# IMPORTANT:
#
#   THIS IS ENERGY MINIMIZATION.
#
#   IT IS NOT MD.
#   NO VELOCITIES ARE ASSIGNED.
#   NO TEMPERATURE IS APPLIED.
#
# ============================================================


# ============================================================
# 1. FILES
# ============================================================

SYSTEM_XML = Path(
    "parameters/combined/"
    "ff3_explicit_dpbs_opc_yb_replicaB.xml"
)

STARTING_POSITIONS = Path(
    "parameters/combined/"
    "ff3_explicit_dpbs_opc_yb_replicaB_minimized_positions_nm.npy"
)

OUTPUT_POSITIONS = Path(
    "parameters/combined/"
    "ff3_explicit_dpbs_opc_yb_replicaB_minimization_continue1_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_replicaB_minimization_continue1.json"
)


# ============================================================
# 2. PARTICLE INDEXING
#
# Read the authoritative particle layout written by the
# explicit-DPBS assembler. Do not duplicate generated indices.
# ============================================================

ASSEMBLY_METADATA = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_replicaB_assembly_check.json"
)

with open(
    ASSEMBLY_METADATA,
    "r",
) as f:
    assembly_metadata = json.load(f)


N_CARBON = 1250
N_GRAPHENE = 3750
N_LIGAND = 70
N_SOLUTE = 3820

SITES_PER_WATER = 4
OPC_O = 0

N_RETAINED_WATERS = int(
    assembly_metadata["opc_waters"]
)

N_RETAINED_WATER_SITES = (
    N_RETAINED_WATERS
    * SITES_PER_WATER
)

N_NA = int(
    assembly_metadata["na_ions"]
)

N_K = int(
    assembly_metadata["k_ions"]
)

N_CL = int(
    assembly_metadata["cl_ions"]
)

N_H2PO4 = int(
    assembly_metadata["h2po4_ions"]
)

N_HPO4 = int(
    assembly_metadata["hpo4_ions"]
)

N_IONS = (
    N_NA
    + N_K
    + N_CL
    + N_H2PO4
    + N_HPO4
)


WATER_START_INDEX = N_SOLUTE

WATER_END_INDEX = (
    WATER_START_INDEX
    + N_RETAINED_WATER_SITES
)

NA_START_INDEX = int(
    assembly_metadata[
        "na_particle_start_index"
    ]
)

NA_END_INDEX = int(
    assembly_metadata[
        "na_particle_end_index_exclusive"
    ]
)

K_START_INDEX = int(
    assembly_metadata[
        "k_particle_start_index"
    ]
)

K_END_INDEX = int(
    assembly_metadata[
        "k_particle_end_index_exclusive"
    ]
)

CL_START_INDEX = int(
    assembly_metadata[
        "cl_particle_start_index"
    ]
)

CL_END_INDEX = int(
    assembly_metadata[
        "cl_particle_end_index_exclusive"
    ]
)

H2PO4_START_INDEX = int(
    assembly_metadata[
        "h2po4_particle_start_index"
    ]
)

H2PO4_END_INDEX = int(
    assembly_metadata[
        "h2po4_particle_end_index_exclusive"
    ]
)

HPO4_START_INDEX = int(
    assembly_metadata[
        "hpo4_particle_start_index"
    ]
)

HPO4_END_INDEX = int(
    assembly_metadata[
        "hpo4_particle_end_index_exclusive"
    ]
)

N_TOTAL_EXPECTED = int(
    assembly_metadata["total_particles"]
)


if WATER_END_INDEX != NA_START_INDEX:
    raise RuntimeError(
        "Water/Na particle boundary mismatch."
    )

if NA_END_INDEX != K_START_INDEX:
    raise RuntimeError(
        "Na/K particle boundary mismatch."
    )

if K_END_INDEX != CL_START_INDEX:
    raise RuntimeError(
        "K/Cl particle boundary mismatch."
    )

if CL_END_INDEX != H2PO4_START_INDEX:
    raise RuntimeError(
        "Cl/H2PO4 particle boundary mismatch."
    )

if H2PO4_END_INDEX != HPO4_START_INDEX:
    raise RuntimeError(
        "H2PO4/HPO4 particle boundary mismatch."
    )

if HPO4_END_INDEX != N_TOTAL_EXPECTED:
    raise RuntimeError(
        "Final DPBS particle boundary mismatch."
    )

if NA_END_INDEX - NA_START_INDEX != N_NA:
    raise RuntimeError(
        "Na particle count mismatch."
    )

if K_END_INDEX - K_START_INDEX != N_K:
    raise RuntimeError(
        "K particle count mismatch."
    )

if CL_END_INDEX - CL_START_INDEX != N_CL:
    raise RuntimeError(
        "Cl particle count mismatch."
    )

if (
    H2PO4_END_INDEX
    - H2PO4_START_INDEX
    != 7 * N_H2PO4
):
    raise RuntimeError(
        "H2PO4 particle count mismatch."
    )

if (
    HPO4_END_INDEX
    - HPO4_START_INDEX
    != 6 * N_HPO4
):
    raise RuntimeError(
        "HPO4 particle count mismatch."
    )


# ============================================================
# 4. MINIMIZATION SETTINGS
#
# OpenMM minimizer convergence is based on RMS force.
#
# 10 kJ/(mol nm) is OpenMM's standard default tolerance.
#
# We use a finite iteration cap on this first run so that
# anything unexpected stops cleanly for inspection.
# ============================================================

MINIMIZATION_TOLERANCE = 10.0
MAX_ITERATIONS = 5000

CONSTRAINT_TOLERANCE = 1.0e-6


# ============================================================
# 5. VERIFY INPUTS
# ============================================================

for path in [
    SYSTEM_XML,
    STARTING_POSITIONS,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# 6. LOAD SYSTEM
# ============================================================

print()
print("=" * 72)
print("SUPPORTED SOLVATED ENERGY MINIMIZATION")
print("=" * 72)


with open(
    SYSTEM_XML,
    "r",
) as f:

    system = (
        openmm.XmlSerializer.deserialize(
            f.read()
        )
    )


positions = np.load(
    STARTING_POSITIONS
)


n_particles = (
    system.getNumParticles()
)


if positions.shape != (
    n_particles,
    3,
):

    raise RuntimeError(
        "Position count does not match System. "
        f"System={n_particles}, "
        f"positions={positions.shape}"
    )


if not np.isfinite(
    positions
).all():

    raise RuntimeError(
        "Starting positions contain NaN/Inf."
    )


if n_particles != N_TOTAL_EXPECTED:

    raise RuntimeError(
        "Explicit-DPBS particle count mismatch. "
        f"Expected {N_TOTAL_EXPECTED}, "
        f"found {n_particles}."
    )


n_waters = N_RETAINED_WATERS
water_sites = N_RETAINED_WATER_SITES


print()
print("SYSTEM COUNTS")
print("-------------")

print(
    "Graphene particles:",
    N_GRAPHENE,
)

print(
    "Pyrene-PEG5 particles:",
    N_LIGAND,
)

print(
    "OPC waters:",
    n_waters,
)

print(
    "OPC sites:",
    water_sites,
)

print(
    "Na+ ions:",
    N_NA,
)

print(
    "K+ ions:",
    N_K,
)

print(
    "Cl- ions:",
    N_CL,
)

print(
    "H2PO4- ions:",
    N_H2PO4,
)

print(
    "HPO4^2- ions:",
    N_HPO4,
)

print(
    "Total particles:",
    n_particles,
)

print(
    "Constraints:",
    system.getNumConstraints(),
)

print(
    "Forces:",
    system.getNumForces(),
)


# ============================================================
# 7. VERIFY EXPLICIT-DPBS COUNT CONSISTENCY
# ============================================================

if n_waters <= 0:

    raise RuntimeError(
        "No retained OPC waters were found."
    )

if water_sites != (
    n_waters
    * SITES_PER_WATER
):

    raise RuntimeError(
        "Retained OPC water-site count is inconsistent."
    )

if (
    WATER_START_INDEX != N_SOLUTE
    or
    WATER_END_INDEX - WATER_START_INDEX
    != water_sites
):

    raise RuntimeError(
        "Retained OPC particle range is inconsistent."
    )

if (
    NA_END_INDEX - NA_START_INDEX
    != N_NA
):

    raise RuntimeError(
        "Na particle count is inconsistent."
    )

if (
    K_END_INDEX - K_START_INDEX
    != N_K
):

    raise RuntimeError(
        "K particle count is inconsistent."
    )

if (
    CL_END_INDEX - CL_START_INDEX
    != N_CL
):

    raise RuntimeError(
        "Cl particle count is inconsistent."
    )

if (
    H2PO4_END_INDEX - H2PO4_START_INDEX
    != 7 * N_H2PO4
):

    raise RuntimeError(
        "H2PO4 particle count is inconsistent."
    )

if (
    HPO4_END_INDEX - HPO4_START_INDEX
    != 6 * N_HPO4
):

    raise RuntimeError(
        "HPO4 particle count is inconsistent."
    )


# ============================================================
# 8. VIRTUAL SITE CHECK
# ============================================================

virtual_sites = [
    i
    for i in range(
        n_particles
    )
    if system.isVirtualSite(i)
]


print(
    "Virtual sites:",
    len(
        virtual_sites
    ),
)


if len(
    virtual_sites
) != n_waters:

    raise RuntimeError(
        "Expected one virtual M site per OPC water."
    )


# ============================================================
# 9. FORCE GROUP LABELS
#
# These match the assembled System.
# ============================================================

GROUP_LABELS = {

    0:
        "Graphene bonds",

    1:
        "Graphene angles",

    2:
        "Graphene out-of-plane",

    3:
        "Pyrene-PEG5 bonds",

    4:
        "Pyrene-PEG5 angles",

    5:
        "Graphene support",

    7:
        "Pyrene-PEG5 torsions",

    8:
        "Unified nonbonded",

    9:
        "OPC bond terms",

    10:
        "OPC angle terms",

    11:
        "DPBS phosphate bonds",

    12:
        "DPBS phosphate angles",

    13:
        "DPBS phosphate torsions",

    29:
        "Yeh-Berkowitz slab correction",
}


# ============================================================
# 9A. VERIFY FF-3 HAMILTONIAN
#
# Refuse to minimize unless the corrected wall-free FF-3
# Hamiltonian is actually present.
# ============================================================

support_matches = []
yb_matches = []
wall_like_forces = []


for force_index in range(
    system.getNumForces()
):

    force = system.getForce(
        force_index
    )

    force_name = force.getName()
    force_group = force.getForceGroup()

    if (
        isinstance(
            force,
            openmm.CustomExternalForce,
        )
        and
        force_name
        ==
        "GrapheneSupportZRestraint_global_z0"
    ):

        support_matches.append(
            (
                force_index,
                force,
            )
        )

    if (
        isinstance(
            force,
            openmm.CustomCVForce,
        )
        and
        force_name
        ==
        "Yeh-Berkowitz slab correction"
    ):

        yb_matches.append(
            (
                force_index,
                force,
            )
        )

    if "wall" in force_name.lower():

        wall_like_forces.append(
            (
                force_index,
                force_name,
            )
        )


if len(
    support_matches
) != 1:

    raise RuntimeError(
        "Expected exactly one validated global-z0 graphene "
        "support force; found "
        f"{len(support_matches)}."
    )


support_index, support_force = (
    support_matches[0]
)


if support_force.getForceGroup() != 5:

    raise RuntimeError(
        "Graphene support is not in force group 5."
    )


if support_force.getNumParticles() != N_CARBON:

    raise RuntimeError(
        "Graphene support does not act on exactly the "
        f"{N_CARBON} carbon cores."
    )


if support_force.getEnergyFunction() != (
    "0.5*kz*periodicdistance(0,0,z,0,0,z0)^2"
):

    raise RuntimeError(
        "Graphene support expression does not match "
        "the validated global-z0 form."
    )


support_globals = {
    support_force.getGlobalParameterName(i):
    support_force.getGlobalParameterDefaultValue(i)

    for i in range(
        support_force.getNumGlobalParameters()
    )
}


if "kz" not in support_globals:

    raise RuntimeError(
        "Graphene support is missing global parameter kz."
    )


if "z0" not in support_globals:

    raise RuntimeError(
        "Graphene support is missing global parameter z0."
    )


if len(
    yb_matches
) != 1:

    raise RuntimeError(
        "Expected exactly one Yeh-Berkowitz CustomCVForce; "
        f"found {len(yb_matches)}."
    )


yb_index, yb_force = (
    yb_matches[0]
)


if yb_force.getForceGroup() != 29:

    raise RuntimeError(
        "Yeh-Berkowitz correction is not in force group 29."
    )


if yb_force.getEnergyFunction() != (
    "yb_coeff*Mz^2"
):

    raise RuntimeError(
        "Unexpected Yeh-Berkowitz energy expression."
    )


yb_globals = {
    yb_force.getGlobalParameterName(i):
    yb_force.getGlobalParameterDefaultValue(i)

    for i in range(
        yb_force.getNumGlobalParameters()
    )
}


if "yb_coeff" not in yb_globals:

    raise RuntimeError(
        "Yeh-Berkowitz force is missing yb_coeff."
    )


if not np.isfinite(
    yb_globals[
        "yb_coeff"
    ]
):

    raise RuntimeError(
        "Yeh-Berkowitz coefficient is non-finite."
    )


if yb_globals[
    "yb_coeff"
] <= 0.0:

    raise RuntimeError(
        "Yeh-Berkowitz coefficient must be positive."
    )


if wall_like_forces:

    raise RuntimeError(
        "Wall-like runtime force unexpectedly present: "
        f"{wall_like_forces}"
    )


print()
print("FF-3 HAMILTONIAN CHECK")
print("----------------------")

print(
    "Graphene support:",
    "PASS",
)

print(
    "  force index:",
    support_index,
)

print(
    "  force group:",
    support_force.getForceGroup(),
)

print(
    "  supported carbons:",
    support_force.getNumParticles(),
)

print(
    "  kz:",
    support_globals["kz"],
)

print(
    "  z0:",
    support_globals["z0"],
    "nm",
)

print(
    "Yeh-Berkowitz:",
    "PASS",
)

print(
    "  force index:",
    yb_index,
)

print(
    "  force group:",
    yb_force.getForceGroup(),
)

print(
    "  yb_coeff:",
    yb_globals["yb_coeff"],
)

print(
    "Runtime water wall:",
    "ABSENT",
)




# ============================================================
# 10. PARTICLE IDENTITY HELPER
# ============================================================

def describe_particle(index):

    index = int(
        index
    )

    if index < N_CARBON:

        return (
            f"graphene carbon core {index}"
        )

    if index < (
        2 * N_CARBON
    ):

        return (
            "graphene upper pi site "
            f"{index - N_CARBON}"
        )

    if index < N_GRAPHENE:

        return (
            "graphene lower pi site "
            f"{index - 2 * N_CARBON}"
        )

    if index < N_SOLUTE:

        return (
            "Pyrene-PEG5 atom "
            f"{index - N_GRAPHENE}"
        )

    if index < WATER_END_INDEX:

        local = (
            index
            - WATER_START_INDEX
        )

        water = (
            local
            // SITES_PER_WATER
        )

        site = (
            local
            % SITES_PER_WATER
        )

        names = {
            0: "O",
            1: "H1",
            2: "H2",
            3: "M",
        }

        return (
            f"OPC retained water {water} "
            f"site {names[site]}"
        )

    if index < NA_END_INDEX:

        return (
            "Na+ ion "
            f"{index - NA_START_INDEX}"
        )

    if index < K_END_INDEX:

        return (
            "K+ ion "
            f"{index - K_START_INDEX}"
        )

    if index < CL_END_INDEX:

        return (
            "Cl- ion "
            f"{index - CL_START_INDEX}"
        )

    if index < H2PO4_END_INDEX:

        local = index - H2PO4_START_INDEX
        molecule = local // 7
        atom = local % 7

        return (
            f"H2PO4- ion {molecule} "
            f"local atom {atom}"
        )

    if index < HPO4_END_INDEX:

        local = index - HPO4_START_INDEX
        molecule = local // 6
        atom = local % 6

        return (
            f"HPO4^2- ion {molecule} "
            f"local atom {atom}"
        )

    return (
        f"unknown particle {index}"
    )


# ============================================================
# 11. MASSIVE-PARTICLE MASK
#
# Ignore virtual M sites for primary force metrics.
# ============================================================

masses = np.asarray(
    [
        system
        .getParticleMass(i)
        .value_in_unit(
            unit.dalton
        )

        for i in range(
            n_particles
        )
    ]
)


massive_mask = (
    masses > 0.0
)


# ============================================================
# 12. OPENMM CONTEXT
# ============================================================

integrator = (
    openmm.VerletIntegrator(
        0.001
        * unit.picoseconds
    )
)


integrator.setConstraintTolerance(
    CONSTRAINT_TOLERANCE
)


try:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CUDA"
        )
    )

except Exception:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CPU"
        )
    )


properties = {}


if platform.getName() == "CUDA":

    properties = {
        "Precision": "mixed",
    }


context = openmm.Context(
    system,
    integrator,
    platform,
    properties,
)


context.setPositions(
    positions
    * unit.nanometer
)


context.applyConstraints(
    CONSTRAINT_TOLERANCE
)

context.computeVirtualSites()


print()
print(
    "OpenMM platform:",
    platform.getName(),
)


# ============================================================
# 13. STATE ANALYSIS FUNCTION
# ============================================================

def analyze_state(
    label,
):

    state = context.getState(
        getEnergy=True,
        getForces=True,
        getPositions=True,
    )

    energy = (
        state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    forces = (
        state
        .getForces(
            asNumpy=True
        )
        .value_in_unit(
            unit.kilojoule_per_mole
            / unit.nanometer
        )
    )

    current_positions = (
        state
        .getPositions(
            asNumpy=True
        )
        .value_in_unit(
            unit.nanometer
        )
    )

    if not np.isfinite(
        energy
    ):

        raise RuntimeError(
            f"{label} energy is NaN/Inf."
        )

    if not np.isfinite(
        forces
    ).all():

        raise RuntimeError(
            f"{label} forces contain NaN/Inf."
        )

    if not np.isfinite(
        current_positions
    ).all():

        raise RuntimeError(
            f"{label} positions contain NaN/Inf."
        )

    magnitudes = np.linalg.norm(
        forces,
        axis=1,
    )

    massive_magnitudes = (
        magnitudes[
            massive_mask
        ]
    )

    massive_indices = np.where(
        massive_mask
    )[0]

    local_max = int(
        np.argmax(
            massive_magnitudes
        )
    )

    max_index = int(
        massive_indices[
            local_max
        ]
    )

    max_force = float(
        massive_magnitudes[
            local_max
        ]
    )

    # RMS of Cartesian force components for massive particles.
    rms_force_component = float(
        np.sqrt(
            np.mean(
                forces[
                    massive_mask
                ]
                ** 2
            )
        )
    )

    graphene_max = float(
        np.max(
            magnitudes[
                :N_GRAPHENE
            ]
        )
    )

    ligand_max = float(
        np.max(
            magnitudes[
                N_GRAPHENE:
                N_SOLUTE
            ]
        )
    )

    water_massive_indices = np.where(
        massive_mask[
            WATER_START_INDEX:
            WATER_END_INDEX
        ]
    )[0] + WATER_START_INDEX

    water_max = float(
        np.max(
            magnitudes[
                water_massive_indices
            ]
        )
    )

    na_indices = np.arange(
        NA_START_INDEX,
        NA_END_INDEX,
        dtype=int,
    )

    k_indices = np.arange(
        K_START_INDEX,
        K_END_INDEX,
        dtype=int,
    )

    cl_indices = np.arange(
        CL_START_INDEX,
        CL_END_INDEX,
        dtype=int,
    )

    h2po4_indices = np.arange(
        H2PO4_START_INDEX,
        H2PO4_END_INDEX,
        dtype=int,
    )

    hpo4_indices = np.arange(
        HPO4_START_INDEX,
        HPO4_END_INDEX,
        dtype=int,
    )

    na_max = float(
        np.max(magnitudes[na_indices])
    )

    k_max = float(
        np.max(magnitudes[k_indices])
    )

    cl_max = float(
        np.max(magnitudes[cl_indices])
    )

    h2po4_max = float(
        np.max(magnitudes[h2po4_indices])
    )

    hpo4_max = float(
        np.max(magnitudes[hpo4_indices])
    )

    oxygen_indices = (
        WATER_START_INDEX
        +
        np.arange(n_waters)
        * SITES_PER_WATER
    )

    oxygen_z = current_positions[
        oxygen_indices,
        2,
    ]

    na_z = current_positions[
        na_indices,
        2,
    ]

    k_z = current_positions[
        k_indices,
        2,
    ]

    cl_z = current_positions[
        cl_indices,
        2,
    ]

    h2po4_z = current_positions[
        h2po4_indices,
        2,
    ]

    hpo4_z = current_positions[
        hpo4_indices,
        2,
    ]

    graphene_z = current_positions[
        :N_CARBON,
        2,
    ]

    graphene_z_max = float(
        np.max(graphene_z)
    )

    waters_below_graphene = int(
        np.sum(
            oxygen_z
            < graphene_z_max
        )
    )

    na_below_graphene = int(
        np.sum(
            na_z
            < graphene_z_max
        )
    )

    k_below_graphene = int(
        np.sum(
            k_z
            < graphene_z_max
        )
    )

    cl_below_graphene = int(
        np.sum(
            cl_z
            < graphene_z_max
        )
    )

    h2po4_atoms_below_graphene = int(
        np.sum(
            h2po4_z
            < graphene_z_max
        )
    )

    hpo4_atoms_below_graphene = int(
        np.sum(
            hpo4_z
            < graphene_z_max
        )
    )

    lowest_water_oxygen_gap = float(
        np.min(oxygen_z)
        - graphene_z_max
    )

    lowest_na_gap = float(
        np.min(na_z)
        - graphene_z_max
    )

    lowest_k_gap = float(
        np.min(k_z)
        - graphene_z_max
    )

    lowest_cl_gap = float(
        np.min(cl_z)
        - graphene_z_max
    )

    lowest_h2po4_atom_gap = float(
        np.min(h2po4_z)
        - graphene_z_max
    )

    lowest_hpo4_atom_gap = float(
        np.min(hpo4_z)
        - graphene_z_max
    )


    # --------------------------------------------------------
    # INDEPENDENT CONSTRAINT AUDIT
    #
    # Measure every constrained distance directly from the
    # current coordinates using the triclinic minimum image.
    # --------------------------------------------------------

    periodic_box_nm = (
        state
        .getPeriodicBoxVectors(
            asNumpy=True
        )
        .value_in_unit(
            unit.nanometer
        )
    )

    inverse_box = np.linalg.inv(
        periodic_box_nm
    )


    constraint_abs_errors_nm = []
    constraint_rel_errors = []

    worst_constraint_index = -1
    worst_constraint_particles = (
        -1,
        -1,
    )

    worst_constraint_target_nm = 0.0
    worst_constraint_actual_nm = 0.0


    for constraint_index in range(
        system.getNumConstraints()
    ):

        particle1, particle2, target_distance = (
            system.getConstraintParameters(
                constraint_index
            )
        )

        particle1 = int(
            particle1
        )

        particle2 = int(
            particle2
        )

        target_nm = float(
            target_distance.value_in_unit(
                unit.nanometer
            )
        )

        displacement = (
            current_positions[
                particle2
            ]
            -
            current_positions[
                particle1
            ]
        )

        fractional = (
            displacement
            @
            inverse_box
        )

        fractional -= np.round(
            fractional
        )

        minimum_image_displacement = (
            fractional
            @
            periodic_box_nm
        )

        actual_nm = float(
            np.linalg.norm(
                minimum_image_displacement
            )
        )

        abs_error_nm = abs(
            actual_nm
            -
            target_nm
        )

        if target_nm > 0.0:

            rel_error = (
                abs_error_nm
                /
                target_nm
            )

        else:

            rel_error = 0.0


        constraint_abs_errors_nm.append(
            abs_error_nm
        )

        constraint_rel_errors.append(
            rel_error
        )


        if (
            worst_constraint_index < 0
            or
            abs_error_nm
            >
            constraint_abs_errors_nm[
                worst_constraint_index
            ]
        ):

            worst_constraint_index = (
                constraint_index
            )

            worst_constraint_particles = (
                particle1,
                particle2,
            )

            worst_constraint_target_nm = (
                target_nm
            )

            worst_constraint_actual_nm = (
                actual_nm
            )


    if constraint_abs_errors_nm:

        max_constraint_abs_error_nm = float(
            np.max(
                constraint_abs_errors_nm
            )
        )

        max_constraint_rel_error = float(
            np.max(
                constraint_rel_errors
            )
        )

    else:

        max_constraint_abs_error_nm = 0.0
        max_constraint_rel_error = 0.0


    return {

        "label":
            label,

        "energy_kj_mol":
            float(
                energy
            ),

        "max_force_kj_mol_nm":
            max_force,

        "max_force_particle":
            max_index,

        "max_force_identity":
            describe_particle(
                max_index
            ),

        "rms_force_component_kj_mol_nm":
            rms_force_component,

        "graphene_max_force_kj_mol_nm":
            graphene_max,

        "ligand_max_force_kj_mol_nm":
            ligand_max,

        "water_max_force_kj_mol_nm":
            water_max,

        "na_max_force_kj_mol_nm":
            na_max,

        "k_max_force_kj_mol_nm":
            k_max,

        "cl_max_force_kj_mol_nm":
            cl_max,

        "h2po4_max_force_kj_mol_nm":
            h2po4_max,

        "hpo4_max_force_kj_mol_nm":
            hpo4_max,

        "oxygen_z_min_nm":
            float(
                np.min(
                    oxygen_z
                )
            ),

        "oxygen_z_max_nm":
            float(
                np.max(
                    oxygen_z
                )
            ),

        "na_z_min_nm":
            float(
                np.min(
                    na_z
                )
            ),

        "na_z_max_nm":
            float(
                np.max(
                    na_z
                )
            ),

        "k_z_min_nm":
            float(np.min(k_z)),

        "k_z_max_nm":
            float(np.max(k_z)),

        "cl_z_min_nm":
            float(
                np.min(
                    cl_z
                )
            ),

        "cl_z_max_nm":
            float(
                np.max(
                    cl_z
                )
            ),

        "h2po4_z_min_nm":
            float(np.min(h2po4_z)),

        "h2po4_z_max_nm":
            float(np.max(h2po4_z)),

        "hpo4_z_min_nm":
            float(np.min(hpo4_z)),

        "hpo4_z_max_nm":
            float(np.max(hpo4_z)),

        "waters_below_graphene":
            waters_below_graphene,

        "na_below_graphene":
            na_below_graphene,

        "k_below_graphene":
            k_below_graphene,

        "cl_below_graphene":
            cl_below_graphene,

        "h2po4_atoms_below_graphene":
            h2po4_atoms_below_graphene,

        "hpo4_atoms_below_graphene":
            hpo4_atoms_below_graphene,

        "lowest_water_oxygen_gap_nm":
            lowest_water_oxygen_gap,

        "lowest_na_gap_nm":
            lowest_na_gap,

        "lowest_k_gap_nm":
            lowest_k_gap,

        "lowest_cl_gap_nm":
            lowest_cl_gap,

        "lowest_h2po4_atom_gap_nm":
            lowest_h2po4_atom_gap,

        "lowest_hpo4_atom_gap_nm":
            lowest_hpo4_atom_gap,

        "max_constraint_abs_error_nm":
            max_constraint_abs_error_nm,

        "max_constraint_rel_error":
            max_constraint_rel_error,

        "worst_constraint_index":
            worst_constraint_index,

        "worst_constraint_particles":
            worst_constraint_particles,

        "worst_constraint_target_nm":
            worst_constraint_target_nm,

        "worst_constraint_actual_nm":
            worst_constraint_actual_nm,

        "graphene_carbon_z_min_nm":
            float(
                np.min(
                    graphene_z
                )
            ),

        "graphene_carbon_z_max_nm":
            float(
                np.max(
                    graphene_z
                )
            ),

        "positions":
            current_positions,

        "forces":
            forces,

        "force_magnitudes":
            magnitudes,
    }


# ============================================================
# 14. ENERGY BY FORCE GROUP
# ============================================================

def group_energies():

    results = {}

    for group, name in sorted(
        GROUP_LABELS.items()
    ):

        state = context.getState(

            getEnergy=True,

            groups=(
                1
                << group
            ),
        )

        energy = (
            state
            .getPotentialEnergy()
            .value_in_unit(
                unit.kilojoule_per_mole
            )
        )

        results[
            name
        ] = float(
            energy
        )

    return results


# ============================================================
# 15. STARTING STATE
# ============================================================

print()
print("=" * 72)
print("BEFORE MINIMIZATION")
print("=" * 72)


before = analyze_state(
    "before"
)


before_groups = (
    group_energies()
)


print()
print(
    f"Potential energy: "
    f"{before['energy_kj_mol']:.6f} kJ/mol"
)

print(
    f"RMS force component: "
    f"{before['rms_force_component_kj_mol_nm']:.6f} "
    f"kJ/(mol nm)"
)

print(
    f"Maximum force: "
    f"{before['max_force_kj_mol_nm']:.6f} "
    f"kJ/(mol nm)"
)

print(
    "Maximum-force particle:",
    before[
        "max_force_identity"
    ],
)


print()
print("MAX FORCE BY COMPONENT")
print("----------------------")

print(
    f"Graphene:    "
    f"{before['graphene_max_force_kj_mol_nm']:.6f}"
)

print(
    f"Pyrene-PEG5: "
    f"{before['ligand_max_force_kj_mol_nm']:.6f}"
)

print(
    f"OPC water:   "
    f"{before['water_max_force_kj_mol_nm']:.6f}"
)


print()
print("ENERGY BY FORCE GROUP")
print("---------------------")


for name, value in (
    before_groups.items()
):

    print(
        f"{name:<28s} "
        f"{value:>18.6f} kJ/mol"
    )


# ============================================================
# 16. MINIMIZE
#
# This changes positions only.
#
# No time integration occurs.
# ============================================================

# ============================================================
# OPENMM NATIVE MINIMIZATION REPORTER
#
# Diagnostic only.  It records OpenMM's L-BFGS progress and
# never stops or otherwise alters the minimization.
# ============================================================

class ConvergenceReporter(
    openmm.MinimizationReporter
):

    def __init__(self):
        super().__init__()

        self.records = []
        self.calls = 0
        self.last_iteration = None
        self.lbfgs_cycles = 0

    def report(
        self,
        iteration,
        x,
        grad,
        args,
    ):

        self.calls += 1

        # OpenMM can restart the L-BFGS iteration counter while
        # changing its temporary constraint-restraint strength.
        if (
            self.last_iteration is not None
            and
            iteration < self.last_iteration
        ):
            self.lbfgs_cycles += 1

        self.last_iteration = iteration

        grad_array = np.asarray(
            grad,
            dtype=float,
        )

        gradient_rms = float(
            np.sqrt(
                np.mean(
                    grad_array ** 2
                )
            )
        )

        record = {
            "report_call":
                self.calls,

            "iteration":
                int(iteration),

            "objective_gradient_rms":
                gradient_rms,

            "system_energy_kj_mol":
                float(
                    args["system energy"]
                ),

            "restraint_energy_kj_mol":
                float(
                    args["restraint energy"]
                ),

            "restraint_strength_kj_mol_nm2":
                float(
                    args["restraint strength"]
                ),

            "max_constraint_error":
                float(
                    args["max constraint error"]
                ),
        }

        self.records.append(
            record
        )

        if (
            self.calls == 1
            or
            self.calls % 100 == 0
            or
            gradient_rms < 20.0
        ):

            print(
                f"call={self.calls:5d}  "
                f"iter={iteration:5d}  "
                f"grad_RMS={gradient_rms:12.5f}  "
                f"E={record['system_energy_kj_mol']:14.3f}  "
                f"restraint_E="
                f"{record['restraint_energy_kj_mol']:11.6f}  "
                f"constraint_err="
                f"{record['max_constraint_error']:.3e}"
            )

        # Reporter is observational only.
        return False


reporter = ConvergenceReporter()


print()
print("=" * 72)
print("RUNNING L-BFGS ENERGY MINIMIZATION")
print("=" * 72)

print()
print(
    f"Tolerance: "
    f"{MINIMIZATION_TOLERANCE:.3f} "
    f"kJ/(mol nm)"
)

print(
    "Maximum iterations:",
    MAX_ITERATIONS,
)


openmm.LocalEnergyMinimizer.minimize(

    context,

    tolerance=(
        MINIMIZATION_TOLERANCE
    ),

    maxIterations=(
        MAX_ITERATIONS
    ),

    reporter=reporter,
)


context.computeVirtualSites()


# ============================================================
# 17. FINAL STATE
# ============================================================

print()
print("=" * 72)
print("AFTER MINIMIZATION")
print("=" * 72)


after = analyze_state(
    "after"
)


after_groups = (
    group_energies()
)


print()
print(
    f"Potential energy: "
    f"{after['energy_kj_mol']:.6f} kJ/mol"
)

print(
    f"Energy change: "
    f"{after['energy_kj_mol'] - before['energy_kj_mol']:.6f} "
    f"kJ/mol"
)

print(
    f"RMS force component: "
    f"{after['rms_force_component_kj_mol_nm']:.6f} "
    f"kJ/(mol nm)"
)

print(
    f"Maximum force: "
    f"{after['max_force_kj_mol_nm']:.6f} "
    f"kJ/(mol nm)"
)

print(
    "Maximum-force particle:",
    after[
        "max_force_identity"
    ],
)


print()
print("MAX FORCE BY COMPONENT")
print("----------------------")

print(
    f"Graphene:    "
    f"{after['graphene_max_force_kj_mol_nm']:.6f}"
)

print(
    f"Pyrene-PEG5: "
    f"{after['ligand_max_force_kj_mol_nm']:.6f}"
)

print(
    f"OPC water:   "
    f"{after['water_max_force_kj_mol_nm']:.6f}"
)


print()
print("ENERGY BY FORCE GROUP")
print("---------------------")


for name, value in (
    after_groups.items()
):

    print(
        f"{name:<28s} "
        f"{value:>18.6f} kJ/mol"
    )


# ============================================================
# 18. DISPLACEMENTS
# ============================================================

before_positions = (
    before[
        "positions"
    ]
)

after_positions = (
    after[
        "positions"
    ]
)


displacements = np.linalg.norm(

    after_positions
    -
    before_positions,

    axis=1,
)


graphene_displacement = float(
    np.max(
        displacements[
            :N_GRAPHENE
        ]
    )
)


ligand_displacement = float(
    np.max(
        displacements[
            N_GRAPHENE:
            N_SOLUTE
        ]
    )
)


water_massive_indices = np.where(
    massive_mask[
        WATER_START_INDEX:
        WATER_END_INDEX
    ]
)[0] + WATER_START_INDEX


water_displacement = float(
    np.max(
        displacements[
            water_massive_indices
        ]
    )
)


na_indices = np.arange(
    NA_START_INDEX,
    NA_END_INDEX,
    dtype=int,
)

k_indices = np.arange(
    K_START_INDEX,
    K_END_INDEX,
    dtype=int,
)

cl_indices = np.arange(
    CL_START_INDEX,
    CL_END_INDEX,
    dtype=int,
)

h2po4_indices = np.arange(
    H2PO4_START_INDEX,
    H2PO4_END_INDEX,
    dtype=int,
)

hpo4_indices = np.arange(
    HPO4_START_INDEX,
    HPO4_END_INDEX,
    dtype=int,
)


na_displacement = float(
    np.max(
        displacements[
            na_indices
        ]
    )
)

k_displacement = float(
    np.max(
        displacements[
            k_indices
        ]
    )
)

cl_displacement = float(
    np.max(
        displacements[
            cl_indices
        ]
    )
)

h2po4_displacement = float(
    np.max(
        displacements[
            h2po4_indices
        ]
    )
)

hpo4_displacement = float(
    np.max(
        displacements[
            hpo4_indices
        ]
    )
)


print()
print("MAXIMUM POSITION CHANGE")
print("-----------------------")

print(
    f"Graphene:      "
    f"{graphene_displacement:.6f} nm"
)

print(
    f"Pyrene-PEG5:   "
    f"{ligand_displacement:.6f} nm"
)

print(
    f"Water atoms:   "
    f"{water_displacement:.6f} nm"
)

print(
    f"Na+ ions:      "
    f"{na_displacement:.6f} nm"
)

print(
    f"K+ ions:       "
    f"{k_displacement:.6f} nm"
)

print(
    f"Cl- ions:      "
    f"{cl_displacement:.6f} nm"
)

print(
    f"H2PO4- atoms:  "
    f"{h2po4_displacement:.6f} nm"
)

print(
    f"HPO4^2- atoms: "
    f"{hpo4_displacement:.6f} nm"
)


# ============================================================
# 19. SUPPORTED WALL-FREE GEOMETRY
# ============================================================

print()
print("SUPPORTED GEOMETRY AFTER MINIMIZATION")
print("-------------------------------------")

print(
    "Graphene carbon z range: "
    f"{after['graphene_carbon_z_min_nm']:.6f} "
    f"to "
    f"{after['graphene_carbon_z_max_nm']:.6f} nm"
)

print(
    "Water oxygen z range:    "
    f"{after['oxygen_z_min_nm']:.6f} "
    f"to "
    f"{after['oxygen_z_max_nm']:.6f} nm"
)

print(
    "Na+ z range:             "
    f"{after['na_z_min_nm']:.6f} "
    f"to "
    f"{after['na_z_max_nm']:.6f} nm"
)

print(
    "K+ z range:              "
    f"{after['k_z_min_nm']:.6f} "
    f"to "
    f"{after['k_z_max_nm']:.6f} nm"
)

print(
    "Cl- z range:             "
    f"{after['cl_z_min_nm']:.6f} "
    f"to "
    f"{after['cl_z_max_nm']:.6f} nm"
)

print(
    "H2PO4- atom z range:     "
    f"{after['h2po4_z_min_nm']:.6f} "
    f"to "
    f"{after['h2po4_z_max_nm']:.6f} nm"
)

print(
    "HPO4^2- atom z range:    "
    f"{after['hpo4_z_min_nm']:.6f} "
    f"to "
    f"{after['hpo4_z_max_nm']:.6f} nm"
)

print(
    "Water oxygens below graphene:",
    after["waters_below_graphene"],
)

print(
    "Na+ ions below graphene:",
    after["na_below_graphene"],
)

print(
    "K+ ions below graphene:",
    after["k_below_graphene"],
)

print(
    "Cl- ions below graphene:",
    after["cl_below_graphene"],
)

print(
    "H2PO4- atoms below graphene:",
    after["h2po4_atoms_below_graphene"],
)

print(
    "HPO4^2- atoms below graphene:",
    after["hpo4_atoms_below_graphene"],
)

print(
    f"Lowest water-O gap above graphene: "
    f"{after['lowest_water_oxygen_gap_nm']:.8f} nm"
)

print(
    f"Lowest Na+ gap above graphene: "
    f"{after['lowest_na_gap_nm']:.8f} nm"
)

print(
    f"Lowest K+ gap above graphene: "
    f"{after['lowest_k_gap_nm']:.8f} nm"
)

print(
    f"Lowest Cl- gap above graphene: "
    f"{after['lowest_cl_gap_nm']:.8f} nm"
)

print(
    f"Lowest H2PO4- atom gap above graphene: "
    f"{after['lowest_h2po4_atom_gap_nm']:.8f} nm"
)

print(
    f"Lowest HPO4^2- atom gap above graphene: "
    f"{after['lowest_hpo4_atom_gap_nm']:.8f} nm"
)


# ============================================================
# 20. TOP 10 RESIDUAL FORCES
# ============================================================

massive_indices = np.where(
    massive_mask
)[0]


order = massive_indices[

    np.argsort(

        after[
            "force_magnitudes"
        ][
            massive_indices
        ]

    )[::-1]
]


top_indices = order[:10]


top_records = []


print()
print("TOP 10 RESIDUAL FORCES")
print("----------------------")


for rank, index in enumerate(
    top_indices,
    start=1,
):

    magnitude = float(
        after[
            "force_magnitudes"
        ][
            index
        ]
    )

    identity = (
        describe_particle(
            index
        )
    )

    print(
        f"{rank:2d}. "
        f"particle {int(index):5d} | "
        f"{magnitude:14.6f} | "
        f"{identity}"
    )

    top_records.append(
        {
            "rank":
                rank,

            "particle":
                int(
                    index
                ),

            "force_kj_mol_nm":
                magnitude,

            "identity":
                identity,
        }
    )


# ============================================================
# 21. CLASSIFY RESULT
# ============================================================

energy_decreased = (
    after[
        "energy_kj_mol"
    ]
    <
    before[
        "energy_kj_mol"
    ]
)


if len(reporter.records) == 0:

    raise RuntimeError(
        "MinimizationReporter received no records."
    )


last_reporter_record = (
    reporter.records[-1]
)


minimum_reported_gradient_rms = float(
    min(
        record[
            "objective_gradient_rms"
        ]
        for record in reporter.records
    )
)


openmm_gradient_converged = (
    last_reporter_record[
        "objective_gradient_rms"
    ]
    <=
    MINIMIZATION_TOLERANCE
    * 1.05
)


constraints_satisfied = (
    after[
        "max_constraint_rel_error"
    ]
    <=
    CONSTRAINT_TOLERANCE
)


one_sided_geometry_preserved = (
    after[
        "waters_below_graphene"
    ]
    == 0
    and
    after[
        "na_below_graphene"
    ]
    == 0
    and
    after[
        "k_below_graphene"
    ]
    == 0
    and
    after[
        "cl_below_graphene"
    ]
    == 0
    and
    after[
        "h2po4_atoms_below_graphene"
    ]
    == 0
    and
    after[
        "hpo4_atoms_below_graphene"
    ]
    == 0
)


if not energy_decreased:

    status = (
        "FAIL_ENERGY_DID_NOT_DECREASE"
    )


elif not openmm_gradient_converged:

    status = (
        "FAIL_OPENMM_GRADIENT_NOT_CONVERGED"
    )


elif not constraints_satisfied:

    status = (
        "FAIL_CONSTRAINT_TOLERANCE"
    )


elif not one_sided_geometry_preserved:

    status = (
        "FAIL_AQUEOUS_SPECIES_BELOW_GRAPHENE"
    )


else:

    status = (
        "MINIMIZATION_PASS"
    )


print()
print("=" * 72)
print("MINIMIZATION RESULT")
print("=" * 72)

print()
print(
    status
)

print()
print("ACCEPTANCE CHECKS")
print("-----------------")

print(
    "Energy decreased:",
    "PASS" if energy_decreased else "FAIL",
)

print(
    "OpenMM final objective-gradient RMS:",
    "PASS" if openmm_gradient_converged else "FAIL",
    (
        f"({last_reporter_record['objective_gradient_rms']:.6f} "
        f"<= {MINIMIZATION_TOLERANCE * 1.05:.6f})"
    ),
)

print(
    "Minimum reported objective-gradient RMS:",
    f"{minimum_reported_gradient_rms:.6f} kJ/(mol nm)",
)

print(
    "OpenMM reporter calls:",
    reporter.calls,
)

print(
    "Detected L-BFGS constraint-restraint cycles:",
    reporter.lbfgs_cycles + 1,
)

print(
    "Constraint relative error <= tolerance:",
    "PASS" if constraints_satisfied else "FAIL",
    f"({after['max_constraint_rel_error']:.3e} "
    f"<= {CONSTRAINT_TOLERANCE:.3e})",
)

print(
    "One-sided aqueous geometry preserved:",
    "PASS" if one_sided_geometry_preserved else "FAIL",
    (
        f"(water={after['waters_below_graphene']}, "
        f"Na+={after['na_below_graphene']}, "
        f"K+={after['k_below_graphene']}, "
        f"Cl-={after['cl_below_graphene']}, "
        f"H2PO4- atoms={after['h2po4_atoms_below_graphene']}, "
        f"HPO4^2- atoms={after['hpo4_atoms_below_graphene']})"
    ),
)

print(
    "Residual max force (diagnostic only):",
    f"{after['max_force_kj_mol_nm']:.6f} kJ/(mol nm)",
)


# ============================================================
# 22. SAVE MINIMIZED POSITIONS
# ============================================================

OUTPUT_POSITIONS.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_POSITIONS,
    after_positions,
)


# ============================================================
# 23. SAVE METADATA
# ============================================================

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "status":
        status,

    "energy_decreased":
        bool(
            energy_decreased
        ),

    "openmm_gradient_converged":
        bool(
            openmm_gradient_converged
        ),

    "final_openmm_objective_gradient_rms_kj_mol_nm":
        last_reporter_record[
            "objective_gradient_rms"
        ],

    "minimum_reported_objective_gradient_rms_kj_mol_nm":
        minimum_reported_gradient_rms,

    "openmm_reporter_calls":
        reporter.calls,

    "lbfgs_constraint_restraint_cycles":
        reporter.lbfgs_cycles + 1,

    "constraints_satisfied":
        bool(
            constraints_satisfied
        ),

    "one_sided_geometry_preserved":
        bool(
            one_sided_geometry_preserved
        ),

    "platform":
        platform.getName(),

    "total_particles":
        n_particles,

    "electrolyte_model":
        "1x Dulbecco PBS without CaCl2 or MgCl2 in OPC water",

    "opc_waters":
        int(
            n_waters
        ),

    "na_ions":
        N_NA,

    "k_ions":
        N_K,

    "cl_ions":
        N_CL,

    "h2po4_ions":
        N_H2PO4,

    "hpo4_ions":
        N_HPO4,

    "total_ions":
        N_IONS,

    "minimization_tolerance_kj_mol_nm":
        MINIMIZATION_TOLERANCE,

    "max_iterations":
        MAX_ITERATIONS,

    "constraint_tolerance":
        CONSTRAINT_TOLERANCE,

    "before_energy_kj_mol":
        before[
            "energy_kj_mol"
        ],

    "after_energy_kj_mol":
        after[
            "energy_kj_mol"
        ],

    "energy_change_kj_mol":
        (
            after[
                "energy_kj_mol"
            ]
            -
            before[
                "energy_kj_mol"
            ]
        ),

    "before_rms_force_component_kj_mol_nm":
        before[
            "rms_force_component_kj_mol_nm"
        ],

    "after_rms_force_component_kj_mol_nm":
        after[
            "rms_force_component_kj_mol_nm"
        ],

    "after_max_constraint_abs_error_nm":
        after[
            "max_constraint_abs_error_nm"
        ],

    "after_max_constraint_rel_error":
        after[
            "max_constraint_rel_error"
        ],

    "after_worst_constraint_index":
        after[
            "worst_constraint_index"
        ],

    "after_worst_constraint_particles":
        [
            int(x)
            for x in after[
                "worst_constraint_particles"
            ]
        ],

    "after_worst_constraint_target_nm":
        after[
            "worst_constraint_target_nm"
        ],

    "after_worst_constraint_actual_nm":
        after[
            "worst_constraint_actual_nm"
        ],

    "before_max_force_kj_mol_nm":
        before[
            "max_force_kj_mol_nm"
        ],

    "after_max_force_kj_mol_nm":
        after[
            "max_force_kj_mol_nm"
        ],

    "after_max_force_identity":
        after[
            "max_force_identity"
        ],

    "before_max_graphene_force_kj_mol_nm":
        before[
            "graphene_max_force_kj_mol_nm"
        ],

    "after_max_graphene_force_kj_mol_nm":
        after[
            "graphene_max_force_kj_mol_nm"
        ],

    "before_max_ligand_force_kj_mol_nm":
        before[
            "ligand_max_force_kj_mol_nm"
        ],

    "after_max_ligand_force_kj_mol_nm":
        after[
            "ligand_max_force_kj_mol_nm"
        ],

    "before_max_water_force_kj_mol_nm":
        before[
            "water_max_force_kj_mol_nm"
        ],

    "after_max_water_force_kj_mol_nm":
        after[
            "water_max_force_kj_mol_nm"
        ],

    "before_max_na_force_kj_mol_nm":
        before[
            "na_max_force_kj_mol_nm"
        ],

    "after_max_na_force_kj_mol_nm":
        after[
            "na_max_force_kj_mol_nm"
        ],

    "before_max_k_force_kj_mol_nm":
        before[
            "k_max_force_kj_mol_nm"
        ],

    "after_max_k_force_kj_mol_nm":
        after[
            "k_max_force_kj_mol_nm"
        ],

    "before_max_cl_force_kj_mol_nm":
        before[
            "cl_max_force_kj_mol_nm"
        ],

    "after_max_cl_force_kj_mol_nm":
        after[
            "cl_max_force_kj_mol_nm"
        ],

    "before_max_h2po4_force_kj_mol_nm":
        before[
            "h2po4_max_force_kj_mol_nm"
        ],

    "after_max_h2po4_force_kj_mol_nm":
        after[
            "h2po4_max_force_kj_mol_nm"
        ],

    "before_max_hpo4_force_kj_mol_nm":
        before[
            "hpo4_max_force_kj_mol_nm"
        ],

    "after_max_hpo4_force_kj_mol_nm":
        after[
            "hpo4_max_force_kj_mol_nm"
        ],

    "graphene_max_displacement_nm":
        graphene_displacement,

    "ligand_max_displacement_nm":
        ligand_displacement,

    "water_max_displacement_nm":
        water_displacement,

    "na_max_displacement_nm":
        na_displacement,

    "k_max_displacement_nm":
        k_displacement,

    "cl_max_displacement_nm":
        cl_displacement,

    "h2po4_max_displacement_nm":
        h2po4_displacement,

    "hpo4_max_displacement_nm":
        hpo4_displacement,

    "after_graphene_carbon_z_min_nm":
        after[
            "graphene_carbon_z_min_nm"
        ],

    "after_graphene_carbon_z_max_nm":
        after[
            "graphene_carbon_z_max_nm"
        ],

    "after_water_oxygen_z_min_nm":
        after[
            "oxygen_z_min_nm"
        ],

    "after_water_oxygen_z_max_nm":
        after[
            "oxygen_z_max_nm"
        ],

    "after_na_z_min_nm":
        after[
            "na_z_min_nm"
        ],

    "after_na_z_max_nm":
        after[
            "na_z_max_nm"
        ],

    "after_k_z_min_nm":
        after[
            "k_z_min_nm"
        ],

    "after_k_z_max_nm":
        after[
            "k_z_max_nm"
        ],

    "after_cl_z_min_nm":
        after[
            "cl_z_min_nm"
        ],

    "after_cl_z_max_nm":
        after[
            "cl_z_max_nm"
        ],

    "after_h2po4_z_min_nm":
        after[
            "h2po4_z_min_nm"
        ],

    "after_h2po4_z_max_nm":
        after[
            "h2po4_z_max_nm"
        ],

    "after_hpo4_z_min_nm":
        after[
            "hpo4_z_min_nm"
        ],

    "after_hpo4_z_max_nm":
        after[
            "hpo4_z_max_nm"
        ],

    "waters_below_graphene_after":
        after[
            "waters_below_graphene"
        ],

    "na_below_graphene_after":
        after[
            "na_below_graphene"
        ],

    "k_below_graphene_after":
        after[
            "k_below_graphene"
        ],

    "cl_below_graphene_after":
        after[
            "cl_below_graphene"
        ],

    "h2po4_atoms_below_graphene_after":
        after[
            "h2po4_atoms_below_graphene"
        ],

    "hpo4_atoms_below_graphene_after":
        after[
            "hpo4_atoms_below_graphene"
        ],

    "lowest_water_oxygen_gap_nm_after":
        after[
            "lowest_water_oxygen_gap_nm"
        ],

    "lowest_na_gap_nm_after":
        after[
            "lowest_na_gap_nm"
        ],

    "lowest_k_gap_nm_after":
        after[
            "lowest_k_gap_nm"
        ],

    "lowest_cl_gap_nm_after":
        after[
            "lowest_cl_gap_nm"
        ],

    "lowest_h2po4_atom_gap_nm_after":
        after[
            "lowest_h2po4_atom_gap_nm"
        ],

    "lowest_hpo4_atom_gap_nm_after":
        after[
            "lowest_hpo4_atom_gap_nm"
        ],

    "before_energy_groups_kj_mol":
        before_groups,

    "after_energy_groups_kj_mol":
        after_groups,

    "top_10_residual_forces":
        top_records,

    "md_run":
        False,

    "velocities_assigned":
        False,

    "temperature_applied":
        False,

    "pyrene_peg5_restrained":
        False,
}


with open(
    OUTPUT_METADATA,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# 24. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "Minimized positions:",
    OUTPUT_POSITIONS,
)

print(
    "Metadata:",
    OUTPUT_METADATA,
)


print()
print("=" * 72)
print("ENERGY MINIMIZATION: COMPLETE")
print("=" * 72)

print()
print(
    "No molecular dynamics has been run."
)

print(
    "No velocities were assigned."
)

print(
    "No temperature was applied."
)

print(
    "Pyrene-PEG5 remained unrestrained."
)

print()
print(
    "Do not begin heating until the "
    "minimization output has been inspected."
)

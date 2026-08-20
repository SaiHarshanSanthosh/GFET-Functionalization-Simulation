from pathlib import Path
import json
import math

import numpy as np

from openmm import openmm, unit, app
from openmm.app import element


# ============================================================
# CLEAN ONE-SIDED OPC WATER REBUILD
#
# PURPOSE
# -------
# Rebuild the aqueous slab from scratch because the previous
# saved water box contains true local water-water overlaps.
#
# Strategy:
#
#   1. Generate a fresh periodic TIP4P-Ew water packing in the
#      ACTUAL skewed graphene x/y cell.
#
#   2. Use a 6.7 nm solvent-slab z dimension:
#
#          0.0 ... 6.7 nm
#
#   3. Shift the slab upward by 0.1 nm:
#
#          0.1 ... 6.8 nm
#
#   4. Preserve every generated oxygen and orientation.
#
#   5. Rebuild H/H/M to exact OpenMM OPC geometry.
#
#   6. Remove whole waters whose oxygen overlaps any
#      LJ-active graphene/Pyrene-PEG5 particle.
#
#   7. Explicitly check water-water O-O distances.
#
# NO MD.
# NO MINIMIZATION.
# NO FORCE-FIELD PARAMETERS ARE MODIFIED.
# ============================================================


# ============================================================
# 1. CONSTANTS
# ============================================================

N_CARBON = 1250
N_GRAPHENE = 3750
N_LIGAND = 70
N_SOLUTE = 3820

SITES_PER_WATER = 4

O = 0
H1 = 1
H2 = 2
M = 3


LOWER_BOUNDARY_NM = 0.100
UPPER_BOUNDARY_NM = 6.800

SLAB_HEIGHT_NM = (
    UPPER_BOUNDARY_NM
    - LOWER_BOUNDARY_NM
)

FINAL_BOX_Z_NM = 12.0


# Hard sanity threshold.
#
# We do NOT expect a properly generated liquid-water box to
# contain the catastrophic ~0.18 nm O-O contacts seen before.
#
# If any retained O-O pair is below 0.24 nm, stop.
#
MIN_ALLOWED_OO_NM = 0.240


# ============================================================
# 2. INPUT FILES
# ============================================================

SUPPORTED_SOLUTE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_supported_positions_nm.npy"
)

GRAPHENE_BOX = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_periodic_box_vectors_nm.npy"
)

BASE_SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_vacuum_assembly_nocutoff.xml"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

OUTPUT_WATER = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_clean_exact_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "clean_one_sided_opc_water_rebuild.json"
)


# ============================================================
# 4. VERIFY INPUTS
# ============================================================

for path in [
    SUPPORTED_SOLUTE,
    GRAPHENE_BOX,
    BASE_SYSTEM_XML,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )


# ============================================================
# 5. LOAD SOLUTE + GRAPHENE PERIODIC CELL
# ============================================================

solute_positions = np.load(
    SUPPORTED_SOLUTE
)

original_box = np.load(
    GRAPHENE_BOX
)


if solute_positions.shape != (
    N_SOLUTE,
    3,
):

    raise RuntimeError(
        "Unexpected supported-solute shape: "
        f"{solute_positions.shape}"
    )


if original_box.shape != (
    3,
    3,
):

    raise RuntimeError(
        "Unexpected periodic-box shape: "
        f"{original_box.shape}"
    )


if not np.isfinite(
    solute_positions
).all():

    raise RuntimeError(
        "Solute positions contain NaN/Inf."
    )


# Preserve graphene a/b vectors.
a_nm = (
    original_box[0]
    .astype(float)
)

b_nm = (
    original_box[1]
    .astype(float)
)


# Temporary solvent-generation box.
#
# Same x/y graphene unit cell, but only 6.7 nm tall.
c_slab_nm = np.asarray(
    [
        0.0,
        0.0,
        SLAB_HEIGHT_NM,
    ],
    dtype=float,
)


slab_box = np.vstack(
    [
        a_nm,
        b_nm,
        c_slab_nm,
    ]
)


# Final MD box.
c_final_nm = np.asarray(
    [
        0.0,
        0.0,
        FINAL_BOX_Z_NM,
    ],
    dtype=float,
)


final_box = np.vstack(
    [
        a_nm,
        b_nm,
        c_final_nm,
    ]
)


print()
print("=" * 72)
print("CLEAN ONE-SIDED OPC WATER REBUILD")
print("=" * 72)

print()
print("GRAPHENE X/Y CELL")
print("-----------------")

print(
    "a:",
    a_nm,
)

print(
    "b:",
    b_nm,
)

print()
print(
    f"Temporary solvent z: "
    f"{SLAB_HEIGHT_NM:.3f} nm"
)

print(
    f"Final MD z:          "
    f"{FINAL_BOX_Z_NM:.3f} nm"
)


# ============================================================
# 6. GENERATE FRESH PERIODIC WATER PACKING
#
# OpenMM addSolvent does not expose OPC as a direct packing
# model. We therefore use the TIP4P-Ew water packing template
# ONLY to obtain a physically sensible distribution of oxygen
# positions and molecular orientations.
#
# We convert every molecule to exact OPC geometry afterward.
# ============================================================

print()
print("=" * 72)
print("GENERATING FRESH PERIODIC WATER PACKING")
print("=" * 72)


empty_topology = app.Topology()

empty_positions = unit.Quantity(
    [],
    unit.nanometer,
)


modeller = app.Modeller(
    empty_topology,
    empty_positions,
)


packing_forcefield = app.ForceField(
    "amber19/tip4pew.xml"
)


a_vec = (
    openmm.Vec3(
        *a_nm
    )
    * unit.nanometer
)

b_vec = (
    openmm.Vec3(
        *b_nm
    )
    * unit.nanometer
)

c_vec = (
    openmm.Vec3(
        *c_slab_nm
    )
    * unit.nanometer
)


modeller.addSolvent(

    packing_forcefield,

    model="tip4pew",

    boxVectors=(
        a_vec,
        b_vec,
        c_vec,
    ),

    neutralize=False,

    ionicStrength=(
        0.0
        * unit.molar
    ),
)


generated_positions = (
    modeller.positions
)


generated_positions_nm = np.asarray(
    [
        [
            p.x,
            p.y,
            p.z,
        ]

        for p in (
            generated_positions
            .value_in_unit(
                unit.nanometer
            )
        )
    ],
    dtype=float,
)


generated_residues = list(
    modeller.topology.residues()
)


print()
print(
    "Generated residues:",
    len(
        generated_residues
    ),
)

print(
    "Generated particles:",
    len(
        generated_positions_nm
    ),
)


# ============================================================
# 7. EXTRACT EACH WATER BY RESIDUE
#
# Do NOT blindly reshape the global coordinate array.
#
# We explicitly use the topology residue/atom indices so every
# O/H/H/M stays attached to the correct molecule.
# ============================================================

raw_waters = []


for residue_index, residue in enumerate(
    generated_residues
):

    atoms = list(
        residue.atoms()
    )

    if residue.name not in {
        "HOH",
        "WAT",
    }:

        raise RuntimeError(
            "Unexpected residue in water-only box: "
            f"{residue.name}"
        )

    name_to_atom = {
        atom.name:
            atom

        for atom in atoms
    }

    required_names = {
        "O",
        "H1",
        "H2",
        "M",
    }

    if set(
        name_to_atom.keys()
    ) != required_names:

        raise RuntimeError(
            f"Unexpected water atom names in residue "
            f"{residue_index}: "
            f"{sorted(name_to_atom.keys())}"
        )

    coords = np.empty(
        (
            4,
            3,
        ),
        dtype=float,
    )

    coords[O] = (
        generated_positions_nm[
            name_to_atom[
                "O"
            ].index
        ]
    )

    coords[H1] = (
        generated_positions_nm[
            name_to_atom[
                "H1"
            ].index
        ]
    )

    coords[H2] = (
        generated_positions_nm[
            name_to_atom[
                "H2"
            ].index
        ]
    )

    coords[M] = (
        generated_positions_nm[
            name_to_atom[
                "M"
            ].index
        ]
    )

    raw_waters.append(
        coords
    )


raw_waters = np.asarray(
    raw_waters,
    dtype=float,
)


if raw_waters.ndim != 3:

    raise RuntimeError(
        "Water extraction failed."
    )


if raw_waters.shape[1:] != (
    4,
    3,
):

    raise RuntimeError(
        "Unexpected extracted water shape: "
        f"{raw_waters.shape}"
    )


n_generated = (
    raw_waters.shape[0]
)


print()
print("TOPOLOGY-SAFE WATER EXTRACTION")
print("------------------------------")

print(
    "Waters:",
    n_generated,
)

print(
    "Shape:",
    raw_waters.shape,
)


# ============================================================
# 8. PERIODIC HELPERS
# ============================================================

slab_inverse = np.linalg.inv(
    slab_box
)


final_inverse = np.linalg.inv(
    final_box
)


def minimum_image(
    displacement,
    box,
    inverse_box,
):

    fractional = (
        displacement
        @ inverse_box
    )

    fractional -= np.round(
        fractional
    )

    return (
        fractional
        @ box
    )


# ============================================================
# 9. CANONICALIZE EACH GENERATED WATER
#
# Move the oxygen into the primary slab cell while preserving
# the local periodic O-H/H/M geometry of the molecule.
#
# This avoids splitting waters that happen to cross a box edge.
# ============================================================

canonical = np.empty_like(
    raw_waters
)


for i in range(
    n_generated
):

    molecule = (
        raw_waters[i]
    )

    oxygen = (
        molecule[O]
    )

    oxygen_frac = (
        oxygen
        @ slab_inverse
    )

    oxygen_frac -= np.floor(
        oxygen_frac
    )

    oxygen_wrapped = (
        oxygen_frac
        @ slab_box
    )

    canonical[
        i,
        O,
        :
    ] = oxygen_wrapped

    for site in [
        H1,
        H2,
        M,
    ]:

        displacement = (
            molecule[site]
            - oxygen
        )

        local = minimum_image(

            displacement,

            slab_box,

            slab_inverse,
        )

        canonical[
            i,
            site,
            :
        ] = (
            oxygen_wrapped
            + local
        )


# Shift whole aqueous slab upward.
canonical[
    :,
    :,
    2
] += LOWER_BOUNDARY_NM


# ============================================================
# 10. VERIFY OXYGEN SLAB
# ============================================================

generated_oxygen_z = (
    canonical[
        :,
        O,
        2
    ]
)


print()
print("GENERATED OXYGEN Z RANGE")
print("------------------------")

print(
    f"Minimum: "
    f"{generated_oxygen_z.min():.6f} nm"
)

print(
    f"Maximum: "
    f"{generated_oxygen_z.max():.6f} nm"
)


if generated_oxygen_z.min() < (
    LOWER_BOUNDARY_NM
    - 1e-8
):

    raise RuntimeError(
        "Generated water lies below lower boundary."
    )


if generated_oxygen_z.max() > (
    UPPER_BOUNDARY_NM
    + 1e-8
):

    raise RuntimeError(
        "Generated water lies above upper boundary."
    )


# ============================================================
# 11. GET EXACT OPC GEOMETRY DIRECTLY FROM OPENMM
# ============================================================

print()
print("=" * 72)
print("READING EXACT OPC GEOMETRY")
print("=" * 72)


reference_topology = app.Topology()

chain = (
    reference_topology
    .addChain()
)

residue = (
    reference_topology
    .addResidue(
        "HOH",
        chain,
    )
)


atom_o = (
    reference_topology
    .addAtom(
        "O",
        element.oxygen,
        residue,
    )
)

atom_h1 = (
    reference_topology
    .addAtom(
        "H1",
        element.hydrogen,
        residue,
    )
)

atom_h2 = (
    reference_topology
    .addAtom(
        "H2",
        element.hydrogen,
        residue,
    )
)


reference_topology.addBond(
    atom_o,
    atom_h1,
)

reference_topology.addBond(
    atom_o,
    atom_h2,
)


reference_positions = unit.Quantity(
    [
        openmm.Vec3(
            *canonical[
                0,
                O,
                :
            ]
        ),

        openmm.Vec3(
            *canonical[
                0,
                H1,
                :
            ]
        ),

        openmm.Vec3(
            *canonical[
                0,
                H2,
                :
            ]
        ),
    ],

    unit.nanometer,
)


opc_forcefield = app.ForceField(
    "amber19/opc.xml"
)


reference_modeller = app.Modeller(
    reference_topology,
    reference_positions,
)


reference_modeller.addExtraParticles(
    opc_forcefield
)


reference_system = (
    opc_forcefield.createSystem(

        reference_modeller.topology,

        nonbondedMethod=app.NoCutoff,

        constraints=None,

        rigidWater=True,

        removeCMMotion=False,
    )
)


if (
    reference_system
    .getNumParticles()
    != 4
):

    raise RuntimeError(
        "OPC reference water is not four-site."
    )


# ============================================================
# 12. READ OPC CONSTRAINT DISTANCES
# ============================================================

constraint_distances = {}


for constraint_index in range(
    reference_system
    .getNumConstraints()
):

    p1, p2, distance = (
        reference_system
        .getConstraintParameters(
            constraint_index
        )
    )

    pair = tuple(
        sorted(
            (
                int(p1),
                int(p2),
            )
        )
    )

    constraint_distances[
        pair
    ] = float(
        distance.value_in_unit(
            unit.nanometer
        )
    )


R_OH1 = (
    constraint_distances[
        (O, H1)
    ]
)

R_OH2 = (
    constraint_distances[
        (O, H2)
    ]
)

R_HH = (
    constraint_distances[
        (H1, H2)
    ]
)


if not np.isclose(
    R_OH1,
    R_OH2,
    atol=1e-12,
):

    raise RuntimeError(
        "OPC O-H constraints differ unexpectedly."
    )


R_OH = (
    0.5
    * (
        R_OH1
        + R_OH2
    )
)


cos_theta = (
    (
        2.0 * R_OH**2
        - R_HH**2
    )
    /
    (
        2.0 * R_OH**2
    )
)


cos_theta = np.clip(
    cos_theta,
    -1.0,
    1.0,
)


THETA_RAD = float(
    np.arccos(
        cos_theta
    )
)


HALF_THETA = (
    0.5
    * THETA_RAD
)


print()
print(
    f"OPC O-H: "
    f"{R_OH:.10f} nm"
)

print(
    f"OPC H-H: "
    f"{R_HH:.10f} nm"
)

print(
    f"OPC H-O-H: "
    f"{np.degrees(THETA_RAD):.8f} degrees"
)


# ============================================================
# 13. READ OPC M-SITE RULE
# ============================================================

if not (
    reference_system
    .isVirtualSite(M)
):

    raise RuntimeError(
        "OPC M site is not virtual."
    )


virtual_site = (
    reference_system
    .getVirtualSite(M)
)


if not isinstance(
    virtual_site,
    openmm.ThreeParticleAverageSite,
):

    raise RuntimeError(
        "Unexpected OPC M-site type."
    )


parents = [
    int(
        virtual_site
        .getParticle(i)
    )

    for i in range(
        virtual_site
        .getNumParticles()
    )
]


weights = [
    float(
        virtual_site
        .getWeight(i)
    )

    for i in range(
        virtual_site
        .getNumParticles()
    )
]


if parents != [
    O,
    H1,
    H2,
]:

    raise RuntimeError(
        "Unexpected OPC M-site parents."
    )


W_O = weights[0]
W_H1 = weights[1]
W_H2 = weights[2]


print(
    "M-site weights:",
    weights,
)


# ============================================================
# 14. CONVERT FRESH WATER PACKING TO EXACT OPC
#
# Oxygen positions stay fixed.
#
# The TIP4P-Ew molecular orientation supplies the initial
# water orientation only.
# ============================================================

oxygen = (
    canonical[
        :,
        O,
        :
    ].copy()
)


v1 = (
    canonical[
        :,
        H1,
        :
    ]
    - oxygen
)


v2 = (
    canonical[
        :,
        H2,
        :
    ]
    - oxygen
)


r1 = np.linalg.norm(
    v1,
    axis=1,
)

r2 = np.linalg.norm(
    v2,
    axis=1,
)


u1 = (
    v1
    / r1[:, None]
)

u2 = (
    v2
    / r2[:, None]
)


bisector = (
    u1
    + u2
)


side = (
    u1
    - u2
)


bisector_norm = np.linalg.norm(
    bisector,
    axis=1,
)

side_norm = np.linalg.norm(
    side,
    axis=1,
)


if np.any(
    bisector_norm < 1e-8
):

    raise RuntimeError(
        "Undefined water bisector."
    )


if np.any(
    side_norm < 1e-8
):

    raise RuntimeError(
        "Undefined H1/H2 orientation."
    )


bisector /= (
    bisector_norm[:, None]
)

side /= (
    side_norm[:, None]
)


cos_half = math.cos(
    HALF_THETA
)

sin_half = math.sin(
    HALF_THETA
)


new_v1 = (
    R_OH
    * (
        cos_half
        * bisector
        +
        sin_half
        * side
    )
)


new_v2 = (
    R_OH
    * (
        cos_half
        * bisector
        -
        sin_half
        * side
    )
)


exact_opc = np.empty_like(
    canonical
)


exact_opc[
    :,
    O,
    :
] = oxygen


exact_opc[
    :,
    H1,
    :
] = (
    oxygen
    + new_v1
)


exact_opc[
    :,
    H2,
    :
] = (
    oxygen
    + new_v2
)


exact_opc[
    :,
    M,
    :
] = (
    W_O
    * exact_opc[
        :,
        O,
        :
    ]
    +
    W_H1
    * exact_opc[
        :,
        H1,
        :
    ]
    +
    W_H2
    * exact_opc[
        :,
        H2,
        :
    ]
)


# ============================================================
# 15. VERIFY EXACT OPC INTERNAL GEOMETRY
# ============================================================

check_oh1 = np.linalg.norm(

    exact_opc[
        :,
        H1,
        :
    ]
    -
    exact_opc[
        :,
        O,
        :
    ],

    axis=1,
)


check_oh2 = np.linalg.norm(

    exact_opc[
        :,
        H2,
        :
    ]
    -
    exact_opc[
        :,
        O,
        :
    ],

    axis=1,
)


check_hh = np.linalg.norm(

    exact_opc[
        :,
        H1,
        :
    ]
    -
    exact_opc[
        :,
        H2,
        :
    ],

    axis=1,
)


max_internal_error = float(
    max(
        np.max(
            np.abs(
                check_oh1
                - R_OH
            )
        ),

        np.max(
            np.abs(
                check_oh2
                - R_OH
            )
        ),

        np.max(
            np.abs(
                check_hh
                - R_HH
            )
        ),
    )
)


print()
print("EXACT OPC CONVERSION")
print("--------------------")

print(
    f"Maximum rigid-geometry error: "
    f"{max_internal_error:.3e} nm"
)


if max_internal_error > 1e-10:

    raise RuntimeError(
        "Exact OPC conversion failed."
    )


# ============================================================
# 16. LOAD SOLUTE NONBONDED PARAMETERS
# ============================================================

with open(
    BASE_SYSTEM_XML,
    "r",
) as f:

    base_system = (
        openmm.XmlSerializer.deserialize(
            f.read()
        )
    )


nb_forces = [
    force

    for force in (
        base_system.getForces()
    )

    if isinstance(
        force,
        openmm.NonbondedForce,
    )
]


if len(
    nb_forces
) != 1:

    raise RuntimeError(
        "Expected one solute NonbondedForce."
    )


nb = nb_forces[0]


if nb.getNumParticles() != (
    N_SOLUTE
):

    raise RuntimeError(
        "Solute NonbondedForce count mismatch."
    )


# ============================================================
# 17. GET OPC OXYGEN LJ PARAMETERS
# ============================================================

opc_nb_forces = [
    force

    for force in (
        reference_system
        .getForces()
    )

    if isinstance(
        force,
        openmm.NonbondedForce,
    )
]


if len(
    opc_nb_forces
) != 1:

    raise RuntimeError(
        "Expected one OPC reference NonbondedForce."
    )


opc_nb = opc_nb_forces[0]


_, sigma_o, epsilon_o = (
    opc_nb.getParticleParameters(
        O
    )
)


SIGMA_O_NM = float(
    sigma_o.value_in_unit(
        unit.nanometer
    )
)


EPSILON_O_KJ = float(
    epsilon_o.value_in_unit(
        unit.kilojoule_per_mole
    )
)


# ============================================================
# 18. FIND LJ-ACTIVE SOLUTE PARTICLES
# ============================================================

active_indices = []
active_sigmas = []


for i in range(
    N_SOLUTE
):

    charge, sigma, epsilon = (
        nb.getParticleParameters(i)
    )

    epsilon_kj = float(
        epsilon.value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    if epsilon_kj <= 1e-12:
        continue

    sigma_nm = float(
        sigma.value_in_unit(
            unit.nanometer
        )
    )

    active_indices.append(
        i
    )

    active_sigmas.append(
        sigma_nm
    )


active_indices = np.asarray(
    active_indices,
    dtype=int,
)

active_sigmas = np.asarray(
    active_sigmas,
    dtype=float,
)


active_positions = (
    solute_positions[
        active_indices
    ]
)


print()
print("LJ-ACTIVE SOLUTE")
print("----------------")

print(
    "Particles:",
    len(
        active_indices
    ),
)


# ============================================================
# 19. PAIRWISE SOLUTE-WATER EXCLUSION DISTANCE
#
# OpenMM uses Lorentz-Berthelot sigma mixing:
#
#       sigma_ij = (sigma_i + sigma_j)/2
#
# The LJ minimum is:
#
#       r_min = 2^(1/6) * sigma_ij
#
# Remove a water when its oxygen starts inside that minimum
# distance from any LJ-active solute site.
# ============================================================

mixed_sigma = (
    0.5
    * (
        active_sigmas
        + SIGMA_O_NM
    )
)


pair_rmin = (
    (2.0 ** (1.0 / 6.0))
    * mixed_sigma
)


# ============================================================
# 20. FILTER WATERS AGAINST SOLUTE
# ============================================================

oxygen_positions = (
    exact_opc[
        :,
        O,
        :
    ]
)


keep_mask = np.ones(
    n_generated,
    dtype=bool,
)


minimum_clearance = np.full(
    n_generated,
    np.inf,
    dtype=float,
)


BLOCK_SIZE = 128


print()
print("=" * 72)
print("FILTERING SOLUTE-WATER OVERLAPS")
print("=" * 72)


for start in range(
    0,
    n_generated,
    BLOCK_SIZE,
):

    stop = min(
        start + BLOCK_SIZE,
        n_generated,
    )

    block = (
        oxygen_positions[
            start:stop
        ]
    )

    displacement = (
        block[
            :,
            None,
            :
        ]
        -
        active_positions[
            None,
            :,
            :
        ]
    )

    displacement = minimum_image(

        displacement,

        final_box,

        final_inverse,
    )

    distances = np.linalg.norm(
        displacement,
        axis=2,
    )

    clearance = (
        distances
        -
        pair_rmin[
            None,
            :
        ]
    )

    block_minimum = np.min(
        clearance,
        axis=1,
    )

    minimum_clearance[
        start:stop
    ] = block_minimum

    keep_mask[
        start:stop
    ] = (
        block_minimum
        >= 0.0
    )


n_removed_solute = int(
    np.sum(
        ~keep_mask
    )
)


retained = (
    exact_opc[
        keep_mask
    ]
)


retained_clearance = (
    minimum_clearance[
        keep_mask
    ]
)


print()
print(
    "Generated waters:",
    n_generated,
)

print(
    "Removed for solute overlap:",
    n_removed_solute,
)

print(
    "Retained waters:",
    len(
        retained
    ),
)


if len(
    retained
) == 0:

    raise RuntimeError(
        "All generated waters were removed."
    )


print(
    f"Minimum retained solute LJ clearance: "
    f"{retained_clearance.min():.8f} nm"
)


# ============================================================
# 21. VERIFY RETAINED OXYGEN BOUNDARIES
# ============================================================

retained_oxygen = (
    retained[
        :,
        O,
        :
    ]
)


retained_z = (
    retained_oxygen[
        :,
        2
    ]
)


print()
print("RETAINED WATER OXYGEN RANGE")
print("---------------------------")

print(
    f"Minimum z: "
    f"{retained_z.min():.6f} nm"
)

print(
    f"Maximum z: "
    f"{retained_z.max():.6f} nm"
)


if retained_z.min() < (
    LOWER_BOUNDARY_NM
    - 1e-8
):

    raise RuntimeError(
        "Retained oxygen below lower boundary."
    )


if retained_z.max() > (
    UPPER_BOUNDARY_NM
    + 1e-8
):

    raise RuntimeError(
        "Retained oxygen above upper boundary."
    )


# ============================================================
# 22. WATER-WATER NEAREST O-O SCAN
# ============================================================

n_retained = len(
    retained
)


nearest_distance = np.full(
    n_retained,
    np.inf,
    dtype=float,
)


nearest_neighbor = np.full(
    n_retained,
    -1,
    dtype=int,
)


print()
print("=" * 72)
print("VALIDATING WATER-WATER PACKING")
print("=" * 72)


for start in range(
    0,
    n_retained,
    BLOCK_SIZE,
):

    stop = min(
        start + BLOCK_SIZE,
        n_retained,
    )

    block = (
        retained_oxygen[
            start:stop
        ]
    )

    displacement = (
        block[
            :,
            None,
            :
        ]
        -
        retained_oxygen[
            None,
            :,
            :
        ]
    )

    displacement = minimum_image(

        displacement,

        final_box,

        final_inverse,
    )

    distance_squared = np.sum(
        displacement
        * displacement,
        axis=2,
    )

    for local_i, global_i in enumerate(
        range(
            start,
            stop,
        )
    ):

        distance_squared[
            local_i,
            global_i
        ] = np.inf

    local_neighbor = np.argmin(
        distance_squared,
        axis=1,
    )

    local_distance = np.sqrt(

        distance_squared[
            np.arange(
                stop - start
            ),
            local_neighbor,
        ]
    )

    nearest_neighbor[
        start:stop
    ] = (
        local_neighbor
    )

    nearest_distance[
        start:stop
    ] = (
        local_distance
    )


closest_water = int(
    np.argmin(
        nearest_distance
    )
)


closest_neighbor = int(
    nearest_neighbor[
        closest_water
    ]
)


closest_distance = float(
    nearest_distance[
        closest_water
    ]
)


print()
print("CLOSEST RETAINED O-O PAIR")
print("-------------------------")

print(
    "Water A:",
    closest_water,
)

print(
    "Water B:",
    closest_neighbor,
)

print(
    f"O-O distance: "
    f"{closest_distance:.8f} nm"
)


thresholds = [
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
]


threshold_counts = {}


print()
print("NEAREST-NEIGHBOR COUNTS")
print("-----------------------")


for threshold in thresholds:

    count = int(
        np.sum(
            nearest_distance
            < threshold
        )
    )

    threshold_counts[
        f"{threshold:.2f}"
    ] = count

    print(
        f"Nearest O-O < {threshold:.2f} nm: "
        f"{count} waters"
    )


# ============================================================
# 23. HARD FAIL ON SEVERE WATER-WATER PACKING
# ============================================================

if closest_distance < (
    MIN_ALLOWED_OO_NM
):

    raise RuntimeError(
        "Fresh solvent packing still contains an "
        "unacceptable O-O contact. "
        f"Closest = {closest_distance:.8f} nm; "
        f"required >= {MIN_ALLOWED_OO_NM:.3f} nm."
    )


print()
print(
    "Water-water packing: PASS"
)


# ============================================================
# 24. OPC M-SITE VALIDATION
# ============================================================

expected_m = (
    W_O
    * retained[
        :,
        O,
        :
    ]
    +
    W_H1
    * retained[
        :,
        H1,
        :
    ]
    +
    W_H2
    * retained[
        :,
        H2,
        :
    ]
)


m_error = np.linalg.norm(

    retained[
        :,
        M,
        :
    ]
    -
    expected_m,

    axis=1,
)


max_m_error = float(
    np.max(
        m_error
    )
)


print()
print("OPC M-SITE VALIDATION")
print("---------------------")

print(
    f"Maximum formula error: "
    f"{max_m_error:.3e} nm"
)


if max_m_error > 1e-12:

    raise RuntimeError(
        "OPC M-site validation failed."
    )


# ============================================================
# 25. EFFECTIVE WATER VOLUME
#
# Approximate from number of waters using bulk water number
# density ~33.367 molecules / nm^3.
#
# Used only as a bookkeeping estimate.
# ============================================================

BULK_WATER_NUMBER_DENSITY = (
    33.367
)


effective_volume_nm3 = (
    n_retained
    / BULK_WATER_NUMBER_DENSITY
)


print()
print("EFFECTIVE WATER VOLUME")
print("----------------------")

print(
    f"From retained water count: "
    f"{effective_volume_nm3:.3f} nm^3"
)


# ============================================================
# 26. SAVE NEW CLEAN WATER FILE
#
# Preserve all older checkpoints.
# ============================================================

OUTPUT_WATER.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_WATER,
    retained,
)


# ============================================================
# 27. SAVE METADATA
# ============================================================

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "generation_method":
        "OpenMM Modeller.addSolvent TIP4P-Ew packing "
        "in graphene triclinic cell, followed by exact "
        "OPC internal geometry conversion",

    "temporary_water_model_for_packing":
        "TIP4P-Ew",

    "production_water_model":
        "OPC",

    "generated_waters":
        int(
            n_generated
        ),

    "removed_for_solute_lj_overlap":
        int(
            n_removed_solute
        ),

    "retained_waters":
        int(
            n_retained
        ),

    "sites_per_water":
        4,

    "total_retained_sites":
        int(
            4
            * n_retained
        ),

    "graphene_a_nm":
        a_nm.tolist(),

    "graphene_b_nm":
        b_nm.tolist(),

    "temporary_slab_height_nm":
        float(
            SLAB_HEIGHT_NM
        ),

    "lower_water_boundary_nm":
        float(
            LOWER_BOUNDARY_NM
        ),

    "upper_water_boundary_nm":
        float(
            UPPER_BOUNDARY_NM
        ),

    "final_box_z_nm":
        float(
            FINAL_BOX_Z_NM
        ),

    "exact_opc_oh_nm":
        float(
            R_OH
        ),

    "exact_opc_hh_nm":
        float(
            R_HH
        ),

    "exact_opc_hoh_angle_deg":
        float(
            np.degrees(
                THETA_RAD
            )
        ),

    "opc_m_site_weights":
        weights,

    "minimum_retained_solute_lj_clearance_nm":
        float(
            retained_clearance.min()
        ),

    "closest_retained_oo_distance_nm":
        float(
            closest_distance
        ),

    "nearest_oo_threshold_counts":
        threshold_counts,

    "minimum_allowed_oo_nm":
        float(
            MIN_ALLOWED_OO_NM
        ),

    "effective_water_volume_nm3":
        float(
            effective_volume_nm3
        ),

    "md_run":
        False,

    "minimization_run":
        False,

    "force_field_parameters_modified":
        False,

    "status":
        "PASS",
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
# 28. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "Clean exact OPC positions:",
    OUTPUT_WATER,
)

print(
    "Metadata:",
    OUTPUT_METADATA,
)


print()
print("=" * 72)
print("CLEAN ONE-SIDED OPC WATER REBUILD: PASS")
print("=" * 72)

print()
print(
    "Fresh water packing generated in the actual "
    "triclinic graphene x/y cell."
)

print(
    "All retained waters use exact OpenMM OPC geometry."
)

print(
    "No severe water-water overlaps remain."
)

print(
    "No MD has been run."
)

print(
    "No minimization has been run."
)

print(
    "Older water checkpoints were NOT overwritten."
)

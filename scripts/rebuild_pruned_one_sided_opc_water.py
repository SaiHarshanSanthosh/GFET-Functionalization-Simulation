from pathlib import Path
import json
import math

import numpy as np

from openmm import openmm, unit, app
from openmm.app import element


# ============================================================
# PRUNED CLEAN ONE-SIDED OPC WATER BUILD
#
# PURPOSE
# -------
# Build fresh solvent, convert it to exact OPC geometry,
# remove solute overlaps, then remove the SMALLEST practical
# set of waters needed to eliminate severe O-O contacts.
#
# IMPORTANT
# ---------
# This is still INITIAL-CONDITION CONSTRUCTION.
#
# NO MD.
# NO MINIMIZATION.
# NO FORCE-FIELD PARAMETERS ARE CHANGED.
#
# Severe water-water contact threshold:
#
#       O-O < 0.240 nm
#
# Contacts above this threshold will be left for normal
# energy minimization to relax.
# ============================================================


# ============================================================
# 1. SYSTEM CONSTANTS
# ============================================================

N_GRAPHENE = 3750
N_LIGAND = 70
N_SOLUTE = 3820

O = 0
H1 = 1
H2 = 2
M = 3

SITES_PER_WATER = 4


# ============================================================
# 2. GEOMETRY
# ============================================================

LOWER_WATER_BOUNDARY_NM = 0.100
UPPER_WATER_BOUNDARY_NM = 6.800

WATER_SLAB_HEIGHT_NM = (
    UPPER_WATER_BOUNDARY_NM
    - LOWER_WATER_BOUNDARY_NM
)

FINAL_BOX_Z_NM = 12.0


# ============================================================
# 3. PRUNING POLICY
#
# We only remove catastrophic packing contacts.
#
# 0.24 nm is still shorter than ordinary liquid-water O-O
# separation, but it eliminates the huge repulsive contacts
# that were producing forces ~1e5 kJ/(mol nm).
#
# Remaining ordinary strain will be handled by minimization.
# ============================================================

CLASH_THRESHOLD_NM = 0.240

MAX_PRUNE_FRACTION = 0.025


# Exact minimum vertex cover is used for small disconnected
# clash clusters.
#
# Extremely large unexpected clusters fall back to a
# deterministic greedy cover.
EXACT_COMPONENT_MAX_NODES = 30


# ============================================================
# 4. FILES
# ============================================================

SOLUTE_POSITIONS_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_supported_positions_nm.npy"
)

BOX_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_periodic_box_vectors_nm.npy"
)

BASE_SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_vacuum_assembly_nocutoff.xml"
)


OUTPUT_WATER = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_pruned_exact_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "pruned_one_sided_opc_water_build.json"
)


# ============================================================
# 5. VERIFY INPUT FILES
# ============================================================

for path in [
    SOLUTE_POSITIONS_FILE,
    BOX_FILE,
    BASE_SYSTEM_XML,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# 6. LOAD SOLUTE + BOX
# ============================================================

solute_positions = np.load(
    SOLUTE_POSITIONS_FILE
)

input_box = np.load(
    BOX_FILE
)


if solute_positions.shape != (
    N_SOLUTE,
    3,
):

    raise RuntimeError(
        "Unexpected solute shape: "
        f"{solute_positions.shape}"
    )


if input_box.shape != (
    3,
    3,
):

    raise RuntimeError(
        "Unexpected box shape: "
        f"{input_box.shape}"
    )


if not np.isfinite(
    solute_positions
).all():

    raise RuntimeError(
        "Solute coordinates contain NaN/Inf."
    )


a_nm = np.asarray(
    input_box[0],
    dtype=float,
)

b_nm = np.asarray(
    input_box[1],
    dtype=float,
)


slab_box = np.vstack(
    [
        a_nm,

        b_nm,

        np.asarray(
            [
                0.0,
                0.0,
                WATER_SLAB_HEIGHT_NM,
            ]
        ),
    ]
)


final_box = np.vstack(
    [
        a_nm,

        b_nm,

        np.asarray(
            [
                0.0,
                0.0,
                FINAL_BOX_Z_NM,
            ]
        ),
    ]
)


slab_inverse = np.linalg.inv(
    slab_box
)

final_inverse = np.linalg.inv(
    final_box
)


print()
print("=" * 72)
print("PRUNED CLEAN ONE-SIDED OPC WATER BUILD")
print("=" * 72)

print()
print("GRAPHENE PERIODIC CELL")
print("----------------------")

print(
    "a:",
    a_nm,
)

print(
    "b:",
    b_nm,
)

print(
    f"Water slab: "
    f"{LOWER_WATER_BOUNDARY_NM:.3f} "
    f"to "
    f"{UPPER_WATER_BOUNDARY_NM:.3f} nm"
)

print(
    f"Final z box: "
    f"{FINAL_BOX_Z_NM:.3f} nm"
)


# ============================================================
# 7. PERIODIC HELPERS
#
# For translating individual molecules we use fractional
# wrapping.
#
# For very short O-O contact detection we test neighboring
# x/y lattice images explicitly, which is safer for this
# oblique graphene cell.
# ============================================================

def minimum_image_simple(
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


def exact_short_xy_distance_squared(
    block,
    targets,
    box,
    inverse_box,
):
    """
    Return minimum squared distances under x/y PBC.

    For the <0.24 nm clash problem, z periodic wrapping cannot
    create a short contact because the final z cell is 12 nm
    while the aqueous slab occupies only 0.1-6.8 nm.

    We explicitly test 3x3 neighboring x/y lattice images.
    """

    displacement = (
        block[:, None, :]
        -
        targets[None, :, :]
    )

    fractional = (
        displacement
        @ inverse_box
    )

    base_x = np.round(
        fractional[:, :, 0]
    )

    base_y = np.round(
        fractional[:, :, 1]
    )

    best_squared = np.full(
        (
            block.shape[0],
            targets.shape[0],
        ),
        np.inf,
        dtype=float,
    )

    for dx in (-1.0, 0.0, 1.0):

        for dy in (-1.0, 0.0, 1.0):

            wrapped = (
                fractional.copy()
            )

            wrapped[:, :, 0] -= (
                base_x
                + dx
            )

            wrapped[:, :, 1] -= (
                base_y
                + dy
            )

            # Do NOT wrap z for our one-sided 12 nm box.

            cart = (
                wrapped
                @ box
            )

            squared = np.sum(
                cart
                * cart,
                axis=2,
            )

            best_squared = np.minimum(
                best_squared,
                squared,
            )

    return best_squared


# ============================================================
# 8. GENERATE FRESH TIP4P-EW PACKING
#
# TIP4P-Ew is used ONLY as a packing/orientation template.
#
# Every generated molecule will be converted to exact OPC
# geometry before being saved.
# ============================================================

print()
print("=" * 72)
print("GENERATING FRESH WATER PACKING")
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


modeller.addSolvent(

    packing_forcefield,

    model="tip4pew",

    boxVectors=(

        openmm.Vec3(
            *a_nm
        ) * unit.nanometer,

        openmm.Vec3(
            *b_nm
        ) * unit.nanometer,

        openmm.Vec3(
            0.0,
            0.0,
            WATER_SLAB_HEIGHT_NM,
        ) * unit.nanometer,
    ),

    neutralize=False,

    ionicStrength=(
        0.0
        * unit.molar
    ),
)


generated_positions_nm = np.asarray(

    [
        [
            p.x,
            p.y,
            p.z,
        ]

        for p in (
            modeller.positions
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
# 9. EXTRACT WATER MOLECULES SAFELY
# ============================================================

waters = []


for residue_index, residue in enumerate(
    generated_residues
):

    atoms = list(
        residue.atoms()
    )

    atom_by_name = {
        atom.name:
            atom

        for atom in atoms
    }

    required = {
        "O",
        "H1",
        "H2",
        "M",
    }

    if set(
        atom_by_name.keys()
    ) != required:

        raise RuntimeError(
            "Unexpected water topology in residue "
            f"{residue_index}: "
            f"{sorted(atom_by_name.keys())}"
        )

    molecule = np.empty(
        (
            4,
            3,
        ),
        dtype=float,
    )

    molecule[O] = (
        generated_positions_nm[
            atom_by_name[
                "O"
            ].index
        ]
    )

    molecule[H1] = (
        generated_positions_nm[
            atom_by_name[
                "H1"
            ].index
        ]
    )

    molecule[H2] = (
        generated_positions_nm[
            atom_by_name[
                "H2"
            ].index
        ]
    )

    molecule[M] = (
        generated_positions_nm[
            atom_by_name[
                "M"
            ].index
        ]
    )

    waters.append(
        molecule
    )


waters = np.asarray(
    waters,
    dtype=float,
)


if waters.shape[1:] != (
    4,
    3,
):

    raise RuntimeError(
        "Water extraction failed: "
        f"{waters.shape}"
    )


n_generated = (
    waters.shape[0]
)


print()
print(
    "Extracted waters:",
    n_generated,
)


# ============================================================
# 10. CANONICALIZE WHOLE MOLECULES
#
# Put O into the primary slab cell.
#
# Preserve each site's local displacement from O.
# ============================================================

canonical = np.empty_like(
    waters
)


for i in range(
    n_generated
):

    old_o = (
        waters[
            i,
            O,
            :
        ]
    )

    fractional_o = (
        old_o
        @ slab_inverse
    )

    fractional_o -= np.floor(
        fractional_o
    )

    new_o = (
        fractional_o
        @ slab_box
    )

    canonical[
        i,
        O,
        :
    ] = new_o

    for site in (
        H1,
        H2,
        M,
    ):

        local = minimum_image_simple(

            waters[
                i,
                site,
                :
            ]
            - old_o,

            slab_box,

            slab_inverse,
        )

        canonical[
            i,
            site,
            :
        ] = (
            new_o
            + local
        )


# Shift whole aqueous region upward.
canonical[:, :, 2] += (
    LOWER_WATER_BOUNDARY_NM
)


# ============================================================
# 11. READ EXACT OPC GEOMETRY FROM OPENMM
# ============================================================

print()
print("=" * 72)
print("READING EXACT OPC DEFINITION")
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


# ============================================================
# 12. OPC CONSTRAINTS
# ============================================================

constraint_distances = {}


for i in range(
    reference_system.getNumConstraints()
):

    p1, p2, distance = (
        reference_system
        .getConstraintParameters(i)
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
        "OPC O-H distances disagree."
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


THETA_RAD = float(
    np.arccos(
        np.clip(
            cos_theta,
            -1.0,
            1.0,
        )
    )
)


HALF_THETA = (
    THETA_RAD
    / 2.0
)


print()
print(
    f"O-H: "
    f"{R_OH:.10f} nm"
)

print(
    f"H-H: "
    f"{R_HH:.10f} nm"
)

print(
    f"H-O-H: "
    f"{np.degrees(THETA_RAD):.8f} degrees"
)


# ============================================================
# 13. OPC M VIRTUAL SITE
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
        "Unexpected OPC virtual-site type."
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
# 14. CONVERT ALL WATERS TO EXACT OPC
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


bisector /= (
    np.linalg.norm(
        bisector,
        axis=1,
    )[:, None]
)


side /= (
    np.linalg.norm(
        side,
        axis=1,
    )[:, None]
)


new_v1 = (
    R_OH
    * (
        math.cos(
            HALF_THETA
        )
        * bisector

        +

        math.sin(
            HALF_THETA
        )
        * side
    )
)


new_v2 = (
    R_OH
    * (
        math.cos(
            HALF_THETA
        )
        * bisector

        -

        math.sin(
            HALF_THETA
        )
        * side
    )
)


opc_waters = np.empty_like(
    canonical
)


opc_waters[
    :,
    O,
    :
] = oxygen


opc_waters[
    :,
    H1,
    :
] = (
    oxygen
    + new_v1
)


opc_waters[
    :,
    H2,
    :
] = (
    oxygen
    + new_v2
)


opc_waters[
    :,
    M,
    :
] = (
    W_O
    * opc_waters[
        :,
        O,
        :
    ]

    +

    W_H1
    * opc_waters[
        :,
        H1,
        :
    ]

    +

    W_H2
    * opc_waters[
        :,
        H2,
        :
    ]
)


# ============================================================
# 15. VERIFY OPC GEOMETRY
# ============================================================

oh1 = np.linalg.norm(
    opc_waters[:, H1, :]
    -
    opc_waters[:, O, :],
    axis=1,
)

oh2 = np.linalg.norm(
    opc_waters[:, H2, :]
    -
    opc_waters[:, O, :],
    axis=1,
)

hh = np.linalg.norm(
    opc_waters[:, H1, :]
    -
    opc_waters[:, H2, :],
    axis=1,
)


geometry_error = max(

    np.max(
        np.abs(
            oh1 - R_OH
        )
    ),

    np.max(
        np.abs(
            oh2 - R_OH
        )
    ),

    np.max(
        np.abs(
            hh - R_HH
        )
    ),
)


print()
print(
    f"Maximum OPC geometry error: "
    f"{geometry_error:.3e} nm"
)


if geometry_error > 1e-10:

    raise RuntimeError(
        "OPC conversion failed."
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

    for force
    in base_system.getForces()

    if isinstance(
        force,
        openmm.NonbondedForce,
    )
]


if len(
    nb_forces
) != 1:

    raise RuntimeError(
        "Expected one base NonbondedForce."
    )


base_nb = nb_forces[0]


# ============================================================
# 17. OPC OXYGEN LJ PARAMETER
# ============================================================

opc_nb_forces = [
    force

    for force
    in reference_system.getForces()

    if isinstance(
        force,
        openmm.NonbondedForce,
    )
]


if len(
    opc_nb_forces
) != 1:

    raise RuntimeError(
        "Expected one OPC NonbondedForce."
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


# ============================================================
# 18. FIND LJ-ACTIVE SOLUTE SITES
# ============================================================

solute_lj_indices = []

solute_lj_sigmas = []


for i in range(
    N_SOLUTE
):

    _, sigma, epsilon = (
        base_nb
        .getParticleParameters(i)
    )

    epsilon_value = float(
        epsilon.value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    if epsilon_value <= 1e-12:
        continue

    solute_lj_indices.append(
        i
    )

    solute_lj_sigmas.append(

        float(
            sigma.value_in_unit(
                unit.nanometer
            )
        )
    )


solute_lj_indices = np.asarray(
    solute_lj_indices,
    dtype=int,
)


solute_lj_sigmas = np.asarray(
    solute_lj_sigmas,
    dtype=float,
)


solute_lj_positions = (
    solute_positions[
        solute_lj_indices
    ]
)


mixed_sigma = (
    0.5
    * (
        solute_lj_sigmas
        + SIGMA_O_NM
    )
)


solute_water_rmin = (
    (2.0 ** (1.0 / 6.0))
    * mixed_sigma
)


print()
print("=" * 72)
print("SOLUTE-WATER FILTER")
print("=" * 72)

print()
print(
    "LJ-active solute particles:",
    len(
        solute_lj_indices
    ),
)


# ============================================================
# 19. REMOVE SOLUTE OVERLAPS
# ============================================================

all_oxygen = (
    opc_waters[
        :,
        O,
        :
    ]
)


n_all = (
    len(
        all_oxygen
    )
)


minimum_solute_clearance = np.full(
    n_all,
    np.inf,
)


BLOCK_SIZE = 96


for start in range(
    0,
    n_all,
    BLOCK_SIZE,
):

    stop = min(
        start + BLOCK_SIZE,
        n_all,
    )

    block = (
        all_oxygen[
            start:stop
        ]
    )

    # Simple minimum image is sufficient here because the
    # original solute and aqueous region occupy the same
    # supported cell and we only need LJ-overlap rejection.
    displacement = (
        block[:, None, :]
        -
        solute_lj_positions[
            None,
            :,
            :
        ]
    )

    displacement = minimum_image_simple(

        displacement,

        final_box,

        final_inverse,
    )

    distance = np.linalg.norm(
        displacement,
        axis=2,
    )

    clearance = (
        distance
        -
        solute_water_rmin[
            None,
            :
        ]
    )

    minimum_solute_clearance[
        start:stop
    ] = (
        np.min(
            clearance,
            axis=1,
        )
    )


solute_keep = (
    minimum_solute_clearance
    >= 0.0
)


after_solute = (
    opc_waters[
        solute_keep
    ]
)


after_solute_clearance = (
    minimum_solute_clearance[
        solute_keep
    ]
)


n_removed_solute = int(
    np.sum(
        ~solute_keep
    )
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
    "Remaining before O-O pruning:",
    len(
        after_solute
    ),
)

print(
    f"Minimum retained solute clearance: "
    f"{after_solute_clearance.min():.8f} nm"
)


# ============================================================
# 20. BUILD WATER-WATER CLASH GRAPH
#
# Each water is a graph vertex.
#
# If two oxygens are closer than 0.240 nm, connect them
# with an edge.
#
# We then find a small vertex cover:
#
#   removing those vertices removes every bad pair.
# ============================================================

print()
print("=" * 72)
print("BUILDING SEVERE O-O CLASH GRAPH")
print("=" * 72)


oxygen = (
    after_solute[
        :,
        O,
        :
    ]
)


n_water = (
    len(
        oxygen
    )
)


clash_edges = []


for start in range(
    0,
    n_water,
    BLOCK_SIZE,
):

    stop = min(
        start + BLOCK_SIZE,
        n_water,
    )

    block = (
        oxygen[
            start:stop
        ]
    )

    distance_squared = (
        exact_short_xy_distance_squared(

            block,

            oxygen,

            final_box,

            final_inverse,
        )
    )

    threshold_squared = (
        CLASH_THRESHOLD_NM**2
    )

    for local_i, global_i in enumerate(
        range(
            start,
            stop,
        )
    ):

        # Only record j > i to avoid duplicates/self edges.
        candidates = np.where(
            (
                distance_squared[
                    local_i
                ]
                < threshold_squared
            )
            &
            (
                np.arange(
                    n_water
                )
                > global_i
            )
        )[0]

        for j in candidates:

            clash_edges.append(
                (
                    int(
                        global_i
                    ),

                    int(
                        j
                    ),
                )
            )


print()
print(
    "Severe O-O edges:",
    len(
        clash_edges
    ),
)


clash_nodes = set()


for u, v in clash_edges:

    clash_nodes.add(
        u
    )

    clash_nodes.add(
        v
    )


print(
    "Waters involved in severe contacts:",
    len(
        clash_nodes
    ),
)


# ============================================================
# 21. CONNECTED COMPONENTS
# ============================================================

adjacency = {
    node:
        set()

    for node in clash_nodes
}


for u, v in clash_edges:

    adjacency[u].add(
        v
    )

    adjacency[v].add(
        u
    )


components = []

unvisited = set(
    clash_nodes
)


while unvisited:

    seed = min(
        unvisited
    )

    stack = [
        seed
    ]

    component = set()

    while stack:

        node = stack.pop()

        if node in component:
            continue

        component.add(
            node
        )

        for neighbor in (
            adjacency[
                node
            ]
        ):

            if neighbor not in component:

                stack.append(
                    neighbor
                )

    unvisited -= (
        component
    )

    components.append(
        component
    )


components.sort(
    key=lambda c:
        (
            -len(c),
            min(c),
        )
)


print(
    "Connected clash clusters:",
    len(
        components
    ),
)


if components:

    print(
        "Largest clash cluster:",
        max(
            len(c)
            for c in components
        ),
        "waters",
    )


# ============================================================
# 22. COMPONENT EDGE HELPER
# ============================================================

def component_edges(
    component,
):

    component = set(
        component
    )

    return frozenset(

        (
            u,
            v
        )

        for u, v
        in clash_edges

        if (
            u in component
            and
            v in component
        )
    )


# ============================================================
# 23. GREEDY VERTEX COVER FALLBACK
# ============================================================

def greedy_vertex_cover(
    edges,
    clearance,
):

    remaining = set(
        edges
    )

    chosen = set()

    while remaining:

        degree = {}

        for u, v in remaining:

            degree[u] = (
                degree.get(
                    u,
                    0,
                )
                + 1
            )

            degree[v] = (
                degree.get(
                    v,
                    0,
                )
                + 1
            )

        # Prefer:
        #
        #   1. water participating in most clashes
        #   2. water closer to the solute
        #   3. smaller index for determinism
        node = min(

            degree,

            key=lambda x:
                (
                    -degree[x],
                    clearance[x],
                    x,
                ),
        )

        chosen.add(
            node
        )

        remaining = {
            edge

            for edge
            in remaining

            if node not in edge
        }

    return chosen


# ============================================================
# 24. EXACT MINIMUM VERTEX COVER FOR SMALL COMPONENTS
#
# Each bad O-O pair is an edge.
#
# At least one endpoint of every edge must be removed.
#
# Branch on one edge:
#
#      remove u
#       OR
#      remove v
#
# For sparse small clusters this gives the true minimum number
# of waters removed.
# ============================================================

def exact_minimum_vertex_cover(
    edges,
    clearance,
):

    edges = frozenset(
        edges
    )

    if not edges:

        return set()

    greedy_initial = (
        greedy_vertex_cover(
            edges,
            clearance,
        )
    )

    best = set(
        greedy_initial
    )

    best_score = float(
        np.sum(
            clearance[
                list(
                    best
                )
            ]
        )
    )

    def matching_lower_bound(
        remaining_edges,
    ):

        used = set()

        count = 0

        for u, v in sorted(
            remaining_edges
        ):

            if (
                u not in used
                and
                v not in used
            ):

                used.add(
                    u
                )

                used.add(
                    v
                )

                count += 1

        return count

    def recurse(
        remaining_edges,
        chosen,
        chosen_score,
    ):

        nonlocal best
        nonlocal best_score

        if not remaining_edges:

            if (
                len(chosen)
                < len(best)
            ):

                best = set(
                    chosen
                )

                best_score = float(
                    chosen_score
                )

            elif (
                len(chosen)
                == len(best)
                and
                chosen_score
                < best_score
            ):

                best = set(
                    chosen
                )

                best_score = float(
                    chosen_score
                )

            return

        if len(
            chosen
        ) >= len(
            best
        ):

            return

        lower_bound = (
            matching_lower_bound(
                remaining_edges
            )
        )

        if (
            len(chosen)
            + lower_bound
            > len(best)
        ):

            return

        # Pick an edge whose endpoints have high degree.
        degree = {}

        for u, v in remaining_edges:

            degree[u] = (
                degree.get(
                    u,
                    0,
                )
                + 1
            )

            degree[v] = (
                degree.get(
                    v,
                    0,
                )
                + 1
            )

        u, v = max(

            remaining_edges,

            key=lambda edge:
                (
                    degree[
                        edge[0]
                    ]
                    +
                    degree[
                        edge[1]
                    ]
                ),
        )

        # Try the more useful removal first:
        # high degree, then closer to solute.
        branch_nodes = sorted(

            (
                u,
                v,
            ),

            key=lambda node:
                (
                    -degree[
                        node
                    ],
                    clearance[
                        node
                    ],
                    node,
            ),
        )

        for node in branch_nodes:

            new_edges = frozenset(

                edge

                for edge
                in remaining_edges

                if node not in edge
            )

            recurse(

                new_edges,

                chosen
                | {
                    node
                },

                chosen_score
                + float(
                    clearance[
                        node
                    ]
                ),
            )

    recurse(
        edges,
        set(),
        0.0,
    )

    return best


# ============================================================
# 25. SOLVE PRUNING SET
# ============================================================

waters_to_remove = set()

n_exact_components = 0
n_greedy_components = 0


for component_index, component in enumerate(
    components,
    start=1,
):

    edges = (
        component_edges(
            component
        )
    )

    if len(
        component
    ) <= EXACT_COMPONENT_MAX_NODES:

        cover = (
            exact_minimum_vertex_cover(

                edges,

                after_solute_clearance,
            )
        )

        method = "exact"

        n_exact_components += 1

    else:

        cover = (
            greedy_vertex_cover(

                edges,

                after_solute_clearance,
            )
        )

        method = "greedy"

        n_greedy_components += 1

    waters_to_remove |= (
        cover
    )

    print(
        f"Cluster {component_index:3d}: "
        f"nodes={len(component):3d} "
        f"edges={len(edges):3d} "
        f"remove={len(cover):3d} "
        f"method={method}"
    )


# ============================================================
# 26. APPLY PRUNING
# ============================================================

remove_indices = np.asarray(
    sorted(
        waters_to_remove
    ),
    dtype=int,
)


prune_mask = np.ones(
    n_water,
    dtype=bool,
)


if len(
    remove_indices
) > 0:

    prune_mask[
        remove_indices
    ] = False


final_waters = (
    after_solute[
        prune_mask
    ]
)


n_pruned = (
    len(
        remove_indices
    )
)


prune_fraction = (
    n_pruned
    / n_water
    if n_water > 0
    else 0.0
)


print()
print("=" * 72)
print("PRUNING SUMMARY")
print("=" * 72)

print()
print(
    "Waters before O-O pruning:",
    n_water,
)

print(
    "Waters removed for severe O-O contacts:",
    n_pruned,
)

print(
    f"Pruned fraction: "
    f"{100.0 * prune_fraction:.3f}%"
)

print(
    "Final retained waters:",
    len(
        final_waters
    ),
)


if prune_fraction > (
    MAX_PRUNE_FRACTION
):

    raise RuntimeError(
        "Pruning would remove too much solvent: "
        f"{100.0 * prune_fraction:.3f}% > "
        f"{100.0 * MAX_PRUNE_FRACTION:.3f}%."
    )


# ============================================================
# 27. FINAL O-O VALIDATION
# ============================================================

print()
print("=" * 72)
print("FINAL WATER-WATER VALIDATION")
print("=" * 72)


final_oxygen = (
    final_waters[
        :,
        O,
        :
    ]
)


n_final = len(
    final_oxygen
)


nearest_distance = np.full(
    n_final,
    np.inf,
)


nearest_neighbor = np.full(
    n_final,
    -1,
    dtype=int,
)


for start in range(
    0,
    n_final,
    BLOCK_SIZE,
):

    stop = min(
        start + BLOCK_SIZE,
        n_final,
    )

    block = (
        final_oxygen[
            start:stop
        ]
    )

    distance_squared = (
        exact_short_xy_distance_squared(

            block,

            final_oxygen,

            final_box,

            final_inverse,
        )
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

    neighbor = np.argmin(
        distance_squared,
        axis=1,
    )

    distance = np.sqrt(

        distance_squared[
            np.arange(
                stop - start
            ),
            neighbor,
        ]
    )

    nearest_distance[
        start:stop
    ] = distance

    nearest_neighbor[
        start:stop
    ] = neighbor


closest_index = int(
    np.argmin(
        nearest_distance
    )
)


closest_neighbor = int(
    nearest_neighbor[
        closest_index
    ]
)


closest_distance = float(
    nearest_distance[
        closest_index
    ]
)


print()
print(
    "Closest water A:",
    closest_index,
)

print(
    "Closest water B:",
    closest_neighbor,
)

print(
    f"Closest O-O distance: "
    f"{closest_distance:.8f} nm"
)


threshold_counts = {}


for threshold in [
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
]:

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


if closest_distance < (
    CLASH_THRESHOLD_NM
    - 1e-10
):

    raise RuntimeError(
        "Pruning failed to eliminate all severe "
        "water-water contacts."
    )


print()
print(
    "Severe O-O contacts: NONE"
)


# ============================================================
# 28. FINAL OXYGEN Z RANGE
# ============================================================

oxygen_z = (
    final_oxygen[
        :,
        2
    ]
)


print()
print("FINAL WATER OXYGEN Z RANGE")
print("--------------------------")

print(
    f"Minimum: "
    f"{oxygen_z.min():.6f} nm"
)

print(
    f"Maximum: "
    f"{oxygen_z.max():.6f} nm"
)


if oxygen_z.min() < (
    LOWER_WATER_BOUNDARY_NM
    - 1e-8
):

    raise RuntimeError(
        "Water oxygen below lower boundary."
    )


if oxygen_z.max() > (
    UPPER_WATER_BOUNDARY_NM
    + 1e-8
):

    raise RuntimeError(
        "Water oxygen above upper boundary."
    )


# ============================================================
# 29. FINAL M-SITE CHECK
# ============================================================

expected_m = (

    W_O
    * final_waters[
        :,
        O,
        :
    ]

    +

    W_H1
    * final_waters[
        :,
        H1,
        :
    ]

    +

    W_H2
    * final_waters[
        :,
        H2,
        :
    ]
)


m_error = np.linalg.norm(

    final_waters[
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
print(
    f"Maximum OPC M-site formula error: "
    f"{max_m_error:.3e} nm"
)


if max_m_error > 1e-12:

    raise RuntimeError(
        "M-site validation failed."
    )


# ============================================================
# 30. WATER COUNT / DENSITY BOOKKEEPING
# ============================================================

BULK_WATER_NUMBER_DENSITY = 33.367


effective_volume_nm3 = (
    len(
        final_waters
    )
    / BULK_WATER_NUMBER_DENSITY
)


print()
print("WATER COUNT BOOKKEEPING")
print("-----------------------")

print(
    "Final waters:",
    len(
        final_waters
    ),
)

print(
    "Final OPC sites:",
    4
    * len(
        final_waters
    ),
)

print(
    f"Bulk-density-equivalent volume: "
    f"{effective_volume_nm3:.3f} nm^3"
)


# ============================================================
# 31. SAVE
# ============================================================

OUTPUT_WATER.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_WATER,
    final_waters,
)


OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "status":
        "PASS",

    "generation_template":
        "TIP4P-Ew",

    "production_water_model":
        "OPC",

    "generated_waters":
        int(
            n_generated
        ),

    "removed_for_solute_overlap":
        int(
            n_removed_solute
        ),

    "waters_before_oo_pruning":
        int(
            n_water
        ),

    "severe_oo_threshold_nm":
        float(
            CLASH_THRESHOLD_NM
        ),

    "severe_clash_edges":
        int(
            len(
                clash_edges
            )
        ),

    "waters_in_severe_clash_graph":
        int(
            len(
                clash_nodes
            )
        ),

    "clash_components":
        int(
            len(
                components
            )
        ),

    "exact_components":
        int(
            n_exact_components
        ),

    "greedy_components":
        int(
            n_greedy_components
        ),

    "waters_removed_for_oo_clashes":
        int(
            n_pruned
        ),

    "prune_fraction":
        float(
            prune_fraction
        ),

    "final_waters":
        int(
            len(
                final_waters
            )
        ),

    "final_opc_sites":
        int(
            4
            * len(
                final_waters
            )
        ),

    "closest_final_oo_nm":
        float(
            closest_distance
        ),

    "nearest_oo_threshold_counts":
        threshold_counts,

    "oxygen_z_min_nm":
        float(
            oxygen_z.min()
        ),

    "oxygen_z_max_nm":
        float(
            oxygen_z.max()
        ),

    "maximum_m_site_error_nm":
        float(
            max_m_error
        ),

    "effective_volume_nm3":
        float(
            effective_volume_nm3
        ),

    "md_run":
        False,

    "minimization_run":
        False,

    "force_field_parameters_modified":
        False,

    "old_water_files_overwritten":
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
# 32. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "Pruned exact OPC positions:",
    OUTPUT_WATER,
)

print(
    "Metadata:",
    OUTPUT_METADATA,
)


print()
print("=" * 72)
print("PRUNED ONE-SIDED OPC WATER BUILD: PASS")
print("=" * 72)

print()
print(
    "All waters use exact OpenMM OPC geometry."
)

print(
    "All severe O-O contacts below 0.240 nm were removed."
)

print(
    "No MD has been run."
)

print(
    "No minimization has been run."
)

print(
    "Graphene and Pyrene-PEG5 were not modified."
)

print(
    "Older solvent checkpoints were not overwritten."
)

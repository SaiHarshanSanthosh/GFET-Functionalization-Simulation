from pathlib import Path
import json

import numpy as np

from openmm import openmm, unit


# ============================================================
# STARTING WATER-CLASH DIAGNOSTIC
#
# PURPOSE
# -------
# Diagnose the very large starting forces observed on a small
# number of OPC water oxygen atoms.
#
# This script checks:
#
#   1. nearest water O-O distances
#   2. nearest water O -> solute LJ-active distances
#   3. the specific waters appearing in the top-force list
#   4. whether the huge forces occur in close water-water pairs
#
# NO MD.
# NO MINIMIZATION.
# NO COORDINATES ARE MODIFIED.
# ============================================================


# ============================================================
# 1. SYSTEM CONSTANTS
# ============================================================

N_CARBON = 1250
N_GRAPHENE = 3750
N_LIGAND = 70
N_SOLUTE = 3820

N_WATERS = 6196
SITES_PER_WATER = 4

N_WATER_SITES = (
    N_WATERS
    * SITES_PER_WATER
)

N_TOTAL = (
    N_SOLUTE
    + N_WATER_SITES
)

OPC_O = 0


# ============================================================
# 2. INPUT FILES
# ============================================================

SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_supported_pme.xml"
)

POSITIONS_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_supported_positions_nm.npy"
)

ENERGY_METADATA = Path(
    "analysis/"
    "full_supported_solvated_energy_check.json"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "starting_water_clash_diagnostic.json"
)


# ============================================================
# 3. VERIFY INPUT FILES
# ============================================================

for path in [
    SYSTEM_XML,
    POSITIONS_FILE,
    ENERGY_METADATA,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )


# ============================================================
# 4. LOAD SYSTEM
# ============================================================

print()
print("=" * 72)
print("STARTING WATER-CLASH DIAGNOSTIC")
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
    POSITIONS_FILE
)


with open(
    ENERGY_METADATA,
    "r",
) as f:

    energy_metadata = json.load(
        f
    )


print()
print("SYSTEM")
print("------")

print(
    "Particles:",
    system.getNumParticles(),
)

print(
    "Position shape:",
    positions.shape,
)


if system.getNumParticles() != N_TOTAL:

    raise RuntimeError(
        f"Expected {N_TOTAL} particles, "
        f"found {system.getNumParticles()}."
    )


if positions.shape != (
    N_TOTAL,
    3,
):

    raise RuntimeError(
        "Unexpected coordinate shape: "
        f"{positions.shape}"
    )


if not np.isfinite(
    positions
).all():

    raise RuntimeError(
        "Positions contain NaN or Inf."
    )


# ============================================================
# 5. PERIODIC BOX
# ============================================================

a, b, c = (
    system.getDefaultPeriodicBoxVectors()
)


def vec_to_nm(vec):

    return np.asarray(
        vec.value_in_unit(
            unit.nanometer
        ),
        dtype=float,
    )


box = np.vstack(
    [
        vec_to_nm(a),
        vec_to_nm(b),
        vec_to_nm(c),
    ]
)


inverse_box = np.linalg.inv(
    box
)


print()
print("PERIODIC BOX")
print("------------")

print("a:", box[0])
print("b:", box[1])
print("c:", box[2])


# ============================================================
# 6. MINIMUM-IMAGE FUNCTION
#
# OpenMM box vectors are stored as row vectors here:
#
#     cartesian = fractional @ box
#
# Convert displacement into fractional coordinates,
# wrap to the nearest periodic image, then convert back.
# ============================================================

def minimum_image(displacements):

    fractional = (
        displacements
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
# 7. LOCATE NONBONDED FORCE
# ============================================================

nb_forces = [
    force
    for force
    in system.getForces()
    if isinstance(
        force,
        openmm.NonbondedForce,
    )
]


if len(nb_forces) != 1:

    raise RuntimeError(
        "Expected exactly one NonbondedForce, "
        f"found {len(nb_forces)}."
    )


nb = nb_forces[0]


# ============================================================
# 8. WATER OXYGEN INDICES
# ============================================================

water_oxygen_indices = (
    N_SOLUTE
    +
    np.arange(
        N_WATERS,
        dtype=int,
    )
    * SITES_PER_WATER
    +
    OPC_O
)


oxygen_positions = (
    positions[
        water_oxygen_indices
    ]
)


if oxygen_positions.shape != (
    N_WATERS,
    3,
):

    raise RuntimeError(
        "Water oxygen extraction failed."
    )


# ============================================================
# 9. READ OPC OXYGEN LJ PARAMETERS
# ============================================================

first_oxygen_global = int(
    water_oxygen_indices[0]
)


q_o, sigma_o, epsilon_o = (
    nb.getParticleParameters(
        first_oxygen_global
    )
)


q_o_e = (
    q_o.value_in_unit(
        unit.elementary_charge
    )
)

sigma_o_nm = (
    sigma_o.value_in_unit(
        unit.nanometer
    )
)

epsilon_o_kj = (
    epsilon_o.value_in_unit(
        unit.kilojoule_per_mole
    )
)


# For identical LJ particles:
#
# minimum of LJ potential occurs at:
#
# r_min = 2^(1/6) * sigma
#
oo_lj_minimum_nm = (
    (2.0 ** (1.0 / 6.0))
    * sigma_o_nm
)


print()
print("OPC OXYGEN NONBONDED PARAMETERS")
print("-------------------------------")

print(
    f"Charge:  {q_o_e:+.8f} e"
)

print(
    f"Sigma:   {sigma_o_nm:.8f} nm"
)

print(
    f"Epsilon: {epsilon_o_kj:.8f} kJ/mol"
)

print(
    f"O-O LJ minimum distance: "
    f"{oo_lj_minimum_nm:.8f} nm"
)


# ============================================================
# 10. FIND NEAREST WATER OXYGEN FOR EVERY WATER
#
# We process in blocks so memory usage remains reasonable.
# ============================================================

BLOCK_SIZE = 128


nearest_oo_distance = np.full(
    N_WATERS,
    np.inf,
    dtype=float,
)


nearest_oo_neighbor = np.full(
    N_WATERS,
    -1,
    dtype=int,
)


print()
print("SCANNING WATER-WATER CONTACTS")
print("-----------------------------")


for start in range(
    0,
    N_WATERS,
    BLOCK_SIZE,
):

    stop = min(
        start + BLOCK_SIZE,
        N_WATERS,
    )

    block = (
        oxygen_positions[
            start:stop
        ]
    )

    displacement = (
        block[:, None, :]
        -
        oxygen_positions[
            None,
            :,
            :
        ]
    )

    displacement = minimum_image(
        displacement
    )

    distance_squared = np.sum(
        displacement
        * displacement,
        axis=2,
    )

    # Prevent each oxygen from selecting itself.
    for local_index, global_water in enumerate(
        range(
            start,
            stop,
        )
    ):

        distance_squared[
            local_index,
            global_water
        ] = np.inf

    local_nearest = np.argmin(
        distance_squared,
        axis=1,
    )

    local_distance = np.sqrt(

        distance_squared[
            np.arange(
                stop - start
            ),
            local_nearest,
        ]
    )

    nearest_oo_neighbor[
        start:stop
    ] = local_nearest

    nearest_oo_distance[
        start:stop
    ] = local_distance


# ============================================================
# 11. GLOBAL CLOSEST WATER-WATER CONTACT
# ============================================================

closest_water = int(
    np.argmin(
        nearest_oo_distance
    )
)


closest_neighbor = int(
    nearest_oo_neighbor[
        closest_water
    ]
)


closest_oo_distance = float(
    nearest_oo_distance[
        closest_water
    ]
)


closest_ratio = (
    closest_oo_distance
    / oo_lj_minimum_nm
)


print()
print("CLOSEST WATER-WATER CONTACT")
print("---------------------------")

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
    f"{closest_oo_distance:.8f} nm"
)

print(
    f"Relative to LJ minimum: "
    f"{closest_ratio:.4f} x r_min"
)


# ============================================================
# 12. APPROXIMATE LJ ENERGY/FORCE FOR CLOSEST O-O PAIR
#
# OPC oxygen has q=0, so direct O-O interaction is LJ only.
# This is useful for diagnosing whether the giant starting
# forces are explained by steric overlap.
# ============================================================

r = closest_oo_distance

sr6 = (
    sigma_o_nm
    / r
) ** 6


sr12 = (
    sr6
    * sr6
)


closest_oo_lj_energy = (
    4.0
    * epsilon_o_kj
    * (
        sr12
        - sr6
    )
)


closest_oo_lj_force = abs(

    (
        24.0
        * epsilon_o_kj
        / r
    )

    * (
        2.0 * sr12
        - sr6
    )
)


print(
    f"Approx O-O LJ energy: "
    f"{closest_oo_lj_energy:.3f} kJ/mol"
)

print(
    f"Approx O-O LJ force:  "
    f"{closest_oo_lj_force:.3f} "
    f"kJ/(mol nm)"
)


# ============================================================
# 13. COUNT WATERS WITH VERY CLOSE O-O NEIGHBORS
# ============================================================

oo_thresholds = [
    0.18,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
]


print()
print("WATER-WATER NEAREST-NEIGHBOR COUNTS")
print("-----------------------------------")


oo_counts = {}


for threshold in oo_thresholds:

    count = int(
        np.sum(
            nearest_oo_distance
            < threshold
        )
    )

    oo_counts[
        f"{threshold:.2f}"
    ] = count

    print(
        f"Nearest O-O < {threshold:.2f} nm: "
        f"{count} waters"
    )


# ============================================================
# 14. TEN CLOSEST UNIQUE WATER-WATER PAIRS
#
# Each water reports one nearest neighbor.
# Canonicalize pairs to avoid printing the same pair twice.
# ============================================================

pair_map = {}


for water_index in range(
    N_WATERS
):

    neighbor = int(
        nearest_oo_neighbor[
            water_index
        ]
    )

    pair = tuple(
        sorted(
            (
                water_index,
                neighbor,
            )
        )
    )

    distance = float(
        nearest_oo_distance[
            water_index
        ]
    )

    if (
        pair not in pair_map
        or distance < pair_map[pair]
    ):

        pair_map[
            pair
        ] = distance


closest_pairs = sorted(

    pair_map.items(),

    key=lambda item:
        item[1],

)[:10]


print()
print("10 CLOSEST UNIQUE WATER-WATER PAIRS")
print("-----------------------------------")


closest_pair_records = []


for rank, (
    pair,
    distance,
) in enumerate(
    closest_pairs,
    start=1,
):

    w1, w2 = pair

    ratio = (
        distance
        / oo_lj_minimum_nm
    )

    print(
        f"{rank:2d}. "
        f"waters {w1:4d} / {w2:4d} | "
        f"{distance:.8f} nm | "
        f"{ratio:.4f} x r_min"
    )

    closest_pair_records.append(
        {
            "rank": rank,
            "water_1": int(w1),
            "water_2": int(w2),
            "oo_distance_nm": float(
                distance
            ),
            "relative_to_oo_lj_minimum": float(
                ratio
            ),
        }
    )


# ============================================================
# 15. FIND SOLUTE PARTICLES WITH ACTIVE LJ TERMS
#
# Ignore particles with epsilon=0 because they cannot produce
# the giant short-range Lennard-Jones repulsion by themselves.
# ============================================================

solute_lj_indices = []


for particle in range(
    N_SOLUTE
):

    charge, sigma, epsilon = (
        nb.getParticleParameters(
            particle
        )
    )

    epsilon_kj = (
        epsilon.value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    if epsilon_kj > 1e-12:

        solute_lj_indices.append(
            particle
        )


solute_lj_indices = np.asarray(
    solute_lj_indices,
    dtype=int,
)


solute_lj_positions = (
    positions[
        solute_lj_indices
    ]
)


print()
print("LJ-ACTIVE SOLUTE PARTICLES")
print("--------------------------")

print(
    "Count:",
    len(
        solute_lj_indices
    ),
)


# ============================================================
# 16. NEAREST LJ-ACTIVE SOLUTE PARTICLE TO EACH WATER OXYGEN
# ============================================================

nearest_solute_distance = np.full(
    N_WATERS,
    np.inf,
    dtype=float,
)


nearest_solute_particle = np.full(
    N_WATERS,
    -1,
    dtype=int,
)


print()
print("SCANNING WATER-SOLUTE CONTACTS")
print("------------------------------")


for start in range(
    0,
    N_WATERS,
    BLOCK_SIZE,
):

    stop = min(
        start + BLOCK_SIZE,
        N_WATERS,
    )

    block = (
        oxygen_positions[
            start:stop
        ]
    )

    displacement = (
        block[:, None, :]
        -
        solute_lj_positions[
            None,
            :,
            :
        ]
    )

    displacement = minimum_image(
        displacement
    )

    distance_squared = np.sum(
        displacement
        * displacement,
        axis=2,
    )

    local_nearest = np.argmin(
        distance_squared,
        axis=1,
    )

    local_distance = np.sqrt(

        distance_squared[
            np.arange(
                stop - start
            ),
            local_nearest,
        ]
    )

    nearest_solute_distance[
        start:stop
    ] = local_distance

    nearest_solute_particle[
        start:stop
    ] = (

        solute_lj_indices[
            local_nearest
        ]
    )


# ============================================================
# 17. DESCRIBE SOLUTE PARTICLE
# ============================================================

def describe_solute_particle(
    index
):

    index = int(index)

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

    return (
        "Pyrene-PEG5 atom "
        f"{index - N_GRAPHENE}"
    )


# ============================================================
# 18. GLOBAL CLOSEST WATER-SOLUTE LJ CONTACT
# ============================================================

closest_solute_water = int(
    np.argmin(
        nearest_solute_distance
    )
)


closest_solute_particle = int(

    nearest_solute_particle[
        closest_solute_water
    ]
)


closest_solute_distance = float(

    nearest_solute_distance[
        closest_solute_water
    ]
)


print()
print("CLOSEST WATER -> LJ-ACTIVE SOLUTE CONTACT")
print("------------------------------------------")

print(
    "Water:",
    closest_solute_water,
)

print(
    "Solute particle:",
    closest_solute_particle,
)

print(
    "Identity:",
    describe_solute_particle(
        closest_solute_particle
    ),
)

print(
    f"Distance: "
    f"{closest_solute_distance:.8f} nm"
)


# ============================================================
# 19. WATER-SOLUTE CLOSE-CONTACT COUNTS
# ============================================================

solute_thresholds = [
    0.18,
    0.20,
    0.22,
    0.24,
    0.26,
    0.28,
    0.30,
]


print()
print("WATER-SOLUTE NEAREST-CONTACT COUNTS")
print("-----------------------------------")


solute_counts = {}


for threshold in solute_thresholds:

    count = int(
        np.sum(
            nearest_solute_distance
            < threshold
        )
    )

    solute_counts[
        f"{threshold:.2f}"
    ] = count

    print(
        f"Nearest O-solute < {threshold:.2f} nm: "
        f"{count} waters"
    )


# ============================================================
# 20. INSPECT WATERS FROM PREVIOUS TOP-FORCE LIST
# ============================================================

print()
print("=" * 72)
print("PREVIOUS TOP-FORCE WATER DIAGNOSTIC")
print("=" * 72)


top_force_records = (
    energy_metadata.get(
        "top_10_forces",
        []
    )
)


top_water_diagnostics = []


for record in top_force_records:

    description = str(
        record.get(
            "description",
            ""
        )
    )

    if not description.startswith(
        "OPC water "
    ):

        continue

    try:

        water_index = int(
            description.split()[2]
        )

    except Exception:

        continue

    force_value = float(
        record.get(
            "force_kj_mol_nm",
            np.nan,
        )
    )

    nearest_water = int(
        nearest_oo_neighbor[
            water_index
        ]
    )

    oo_distance = float(
        nearest_oo_distance[
            water_index
        ]
    )

    solute_particle = int(
        nearest_solute_particle[
            water_index
        ]
    )

    solute_distance = float(
        nearest_solute_distance[
            water_index
        ]
    )

    print()
    print(
        f"Water {water_index}"
    )

    print(
        f"  starting force: "
        f"{force_value:.3f} kJ/(mol nm)"
    )

    print(
        f"  nearest water: "
        f"{nearest_water}"
    )

    print(
        f"  O-O distance: "
        f"{oo_distance:.8f} nm"
    )

    print(
        f"  O-O / LJ minimum: "
        f"{oo_distance / oo_lj_minimum_nm:.4f}"
    )

    print(
        f"  nearest LJ solute: "
        f"{solute_particle} "
        f"({describe_solute_particle(solute_particle)})"
    )

    print(
        f"  O-solute distance: "
        f"{solute_distance:.8f} nm"
    )

    top_water_diagnostics.append(
        {
            "water": water_index,
            "starting_force_kj_mol_nm":
                force_value,
            "nearest_water":
                nearest_water,
            "nearest_oo_distance_nm":
                oo_distance,
            "nearest_oo_relative_to_lj_minimum":
                float(
                    oo_distance
                    / oo_lj_minimum_nm
                ),
            "nearest_lj_solute_particle":
                solute_particle,
            "nearest_lj_solute_identity":
                describe_solute_particle(
                    solute_particle
                ),
            "nearest_lj_solute_distance_nm":
                solute_distance,
        }
    )


# ============================================================
# 21. CLASSIFY
# ============================================================

if closest_oo_distance < 0.20:

    status = (
        "SEVERE_WATER_WATER_OVERLAP"
    )


elif closest_oo_distance < 0.24:

    status = (
        "STRONG_WATER_WATER_CLASH"
    )


elif closest_solute_distance < 0.20:

    status = (
        "SEVERE_WATER_SOLUTE_OVERLAP"
    )


elif closest_solute_distance < 0.24:

    status = (
        "STRONG_WATER_SOLUTE_CLASH"
    )


else:

    status = (
        "NO_EXTREME_GEOMETRIC_OVERLAP_FOUND"
    )


print()
print("=" * 72)
print("DIAGNOSTIC RESULT")
print("=" * 72)

print()
print(status)


# ============================================================
# 22. SAVE METADATA
# ============================================================

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "status":
        status,

    "n_waters":
        N_WATERS,

    "opc_oxygen_charge_e":
        float(
            q_o_e
        ),

    "opc_oxygen_sigma_nm":
        float(
            sigma_o_nm
        ),

    "opc_oxygen_epsilon_kj_mol":
        float(
            epsilon_o_kj
        ),

    "oo_lj_minimum_nm":
        float(
            oo_lj_minimum_nm
        ),

    "closest_oo_water_1":
        closest_water,

    "closest_oo_water_2":
        closest_neighbor,

    "closest_oo_distance_nm":
        closest_oo_distance,

    "closest_oo_relative_to_lj_minimum":
        float(
            closest_ratio
        ),

    "closest_oo_lj_energy_kj_mol":
        float(
            closest_oo_lj_energy
        ),

    "closest_oo_lj_force_kj_mol_nm":
        float(
            closest_oo_lj_force
        ),

    "water_water_threshold_counts":
        oo_counts,

    "closest_water_pairs":
        closest_pair_records,

    "lj_active_solute_particles":
        int(
            len(
                solute_lj_indices
            )
        ),

    "closest_solute_water":
        closest_solute_water,

    "closest_solute_particle":
        closest_solute_particle,

    "closest_solute_identity":
        describe_solute_particle(
            closest_solute_particle
        ),

    "closest_solute_distance_nm":
        closest_solute_distance,

    "water_solute_threshold_counts":
        solute_counts,

    "top_force_water_diagnostics":
        top_water_diagnostics,

    "md_run":
        False,

    "minimization_run":
        False,

    "coordinates_modified":
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
# 23. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "Metadata:",
    OUTPUT_METADATA,
)


print()
print("=" * 72)
print("STARTING WATER-CLASH DIAGNOSTIC: COMPLETE")
print("=" * 72)

print()
print(
    "No MD has been run."
)

print(
    "No minimization has been run."
)

print(
    "No coordinates were modified."
)

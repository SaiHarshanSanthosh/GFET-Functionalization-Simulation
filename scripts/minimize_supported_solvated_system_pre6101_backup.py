from pathlib import Path
import json

import numpy as np

from openmm import openmm, unit


# ============================================================
# FIRST FULL-SYSTEM ENERGY MINIMIZATION
#
# SYSTEM:
#
#   supported IFF graphene
#   +
#   unrestrained Pyrene-PEG5
#   +
#   pruned exact OPC water
#   +
#   PME
#   +
#   graphene support
#   +
#   one-sided water wall
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
    "pyrene_peg5_graphene_one_sided_opc_supported_pme.xml"
)

STARTING_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_supported_positions_nm.npy"
)

OUTPUT_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_minimized_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "supported_solvated_minimization.json"
)


# ============================================================
# 2. PARTICLE INDEXING
# ============================================================

N_CARBON = 1250
N_GRAPHENE = 3750
N_LIGAND = 70
N_SOLUTE = 3820

SITES_PER_WATER = 4
OPC_O = 0


# ============================================================
# 3. SLAB GEOMETRY
# ============================================================

LOWER_WATER_BOUNDARY_NM = 0.100
UPPER_WATER_BOUNDARY_NM = 6.800


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
MAX_ITERATIONS = 2000

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


water_sites = (
    n_particles
    - N_SOLUTE
)


if water_sites <= 0:

    raise RuntimeError(
        "No water particles found."
    )


if (
    water_sites
    % SITES_PER_WATER
) != 0:

    raise RuntimeError(
        "Water particle count is not divisible by 4."
    )


n_waters = (
    water_sites
    // SITES_PER_WATER
)


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
# 7. VERIFY EXPECTED WATER COUNT
# ============================================================

if n_waters != 6134:

    raise RuntimeError(
        "Expected 6134 pruned OPC waters; "
        f"found {n_waters}."
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

    6:
        "Water wall",

    7:
        "Pyrene-PEG5 torsions",

    8:
        "Unified nonbonded",

    9:
        "OPC bond terms",

    10:
        "OPC angle terms",
}


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

    local = (
        index
        - N_SOLUTE
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
        f"OPC water {water} "
        f"site {names[site]}"
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

    water_massive_mask = (
        massive_mask[
            N_SOLUTE:
        ]
    )

    water_max = float(
        np.max(
            magnitudes[
                N_SOLUTE:
            ][
                water_massive_mask
            ]
        )
    )

    oxygen_indices = (
        N_SOLUTE
        +
        np.arange(
            n_waters
        )
        * 4
    )

    oxygen_z = (
        current_positions[
            oxygen_indices,
            2,
        ]
    )

    lower_violation = np.maximum(
        LOWER_WATER_BOUNDARY_NM
        - oxygen_z,
        0.0,
    )

    upper_violation = np.maximum(
        oxygen_z
        - UPPER_WATER_BOUNDARY_NM,
        0.0,
    )

    maximum_wall_violation = float(
        max(
            np.max(
                lower_violation
            ),
            np.max(
                upper_violation
            ),
        )
    )

    outside_waters = int(
        np.sum(
            (
                oxygen_z
                < LOWER_WATER_BOUNDARY_NM
            )
            |
            (
                oxygen_z
                > UPPER_WATER_BOUNDARY_NM
            )
        )
    )

    graphene_z = (
        current_positions[
            :N_CARBON,
            2,
        ]
    )

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

        "waters_outside_wall":
            outside_waters,

        "maximum_wall_violation_nm":
            maximum_wall_violation,

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
        N_SOLUTE:
    ]
)[0] + N_SOLUTE


water_displacement = float(
    np.max(
        displacements[
            water_massive_indices
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


# ============================================================
# 19. GRAPHENE SUPPORT / WATER WALL
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
    "Waters outside nominal wall:",
    after[
        "waters_outside_wall"
    ],
)

print(
    f"Maximum wall violation: "
    f"{after['maximum_wall_violation_nm']:.8f} nm"
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


if not energy_decreased:

    status = (
        "FAIL_ENERGY_DID_NOT_DECREASE"
    )


elif after[
    "max_force_kj_mol_nm"
] > 5000.0:

    status = (
        "LARGE_RESIDUAL_FORCE"
    )


elif after[
    "max_force_kj_mol_nm"
] > 1000.0:

    status = (
        "MINIMIZED_BUT_INSPECT_RESIDUAL_FORCES"
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

    "platform":
        platform.getName(),

    "total_particles":
        n_particles,

    "opc_waters":
        int(
            n_waters
        ),

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

    "graphene_max_displacement_nm":
        graphene_displacement,

    "ligand_max_displacement_nm":
        ligand_displacement,

    "water_max_displacement_nm":
        water_displacement,

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

    "waters_outside_nominal_wall":
        after[
            "waters_outside_wall"
        ],

    "maximum_wall_violation_nm":
        after[
            "maximum_wall_violation_nm"
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

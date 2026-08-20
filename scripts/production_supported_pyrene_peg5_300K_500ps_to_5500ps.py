from pathlib import Path
import csv
import json
import math
import os
import sys
import time

import numpy as np

import openmm as mm
from openmm import app, unit


# ============================================================
# PATHS
# ============================================================

SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_supported_pme.xml"
)

START_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "equilibrated_300K_500ps_positions_nm.npy"
)

START_VELOCITIES = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "equilibrated_300K_500ps_velocities_nm_per_ps.npy"
)

SAFE_SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "supported_pme_production_safe.xml"
)

FINAL_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "production_300K_5500ps_positions_nm.npy"
)

FINAL_VELOCITIES = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "production_300K_5500ps_velocities_nm_per_ps.npy"
)

FINAL_CHECKPOINT = Path(
    "checkpoints/"
    "pyrene_peg5_graphene_production_300K_5500ps.chk"
)

RECOVERY_CHECKPOINT = Path(
    "checkpoints/"
    "pyrene_peg5_graphene_RECOVERY_300K_500to5500ps.chk"
)

LOG_CSV = Path(
    "analysis/"
    "production_300K_500to5500ps_log.csv"
)

METADATA_JSON = Path(
    "analysis/"
    "production_300K_5500ps.json"
)

TRAJECTORY_DCD = Path(
    "trajectories/"
    "pyrene_peg5_graphene_production_300K_500to5500ps_visible.dcd"
)


for directory in [
    FINAL_POSITIONS.parent,
    FINAL_CHECKPOINT.parent,
    LOG_CSV.parent,
    TRAJECTORY_DCD.parent,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PROTOCOL
# ============================================================

TEMPERATURE_K = 300.0

FRICTION_PER_PS = 1.0

TIMESTEP_PS = 0.001

START_TIME_PS = 500.0

PRODUCTION_TIME_PS = 5000.0

END_TIME_PS = (
    START_TIME_PS
    +
    PRODUCTION_TIME_PS
)

REPORT_INTERVAL_PS = 2.0

CHECKPOINT_INTERVAL_PS = 10.0

CONSOLE_INTERVAL_PS = 20.0

RANDOM_SEED = 20260819


TOTAL_STEPS = int(
    round(
        PRODUCTION_TIME_PS
        /
        TIMESTEP_PS
    )
)

REPORT_STEPS = int(
    round(
        REPORT_INTERVAL_PS
        /
        TIMESTEP_PS
    )
)

CHECKPOINT_STEPS = int(
    round(
        CHECKPOINT_INTERVAL_PS
        /
        TIMESTEP_PS
    )
)

CONSOLE_EVERY_REPORTS = int(
    round(
        CONSOLE_INTERVAL_PS
        /
        REPORT_INTERVAL_PS
    )
)

START_STEP = int(
    round(
        START_TIME_PS
        /
        TIMESTEP_PS
    )
)


if TOTAL_STEPS % REPORT_STEPS != 0:
    raise RuntimeError(
        "Production length must be divisible "
        "by reporting interval."
    )

if CHECKPOINT_STEPS % REPORT_STEPS != 0:
    raise RuntimeError(
        "Checkpoint interval must be divisible "
        "by reporting interval."
    )


N_REPORTS = (
    TOTAL_STEPS
    //
    REPORT_STEPS
)


# ============================================================
# EXPECTED SYSTEM INDEXING
# ============================================================

N_GRAPHENE_CARBONS = 1250

N_GRAPHENE_TOTAL = 3750

LIGAND_START = 3750

LIGAND_END = 3820

N_SOLUTE = 3820

N_WATERS = 6101

WATER_SITES = 4

EXPECTED_PARTICLES = (
    N_SOLUTE
    +
    N_WATERS * WATER_SITES
)

EXPECTED_VISIBLE_ATOMS = (
    N_GRAPHENE_CARBONS
    +
    (LIGAND_END - LIGAND_START)
    +
    N_WATERS * 3
)


# Exact aromatic atoms:
# ligand-local 2..17
PYRENE_LOCAL = np.arange(
    2,
    18,
    dtype=int,
)

PYRENE_GLOBAL = (
    LIGAND_START
    +
    PYRENE_LOCAL
)


WATER_O_INDICES = np.asarray(
    [
        N_SOLUTE
        +
        WATER_SITES * i
        for i in range(
            N_WATERS
        )
    ],
    dtype=int,
)


# Visible DCD:
#
# graphene carbon cores
# ligand all 70 atoms
# OPC O/H/H only
#
# π pseudo-particles omitted.
# OPC M sites omitted.

VISIBLE_INDICES = []

VISIBLE_INDICES.extend(
    range(
        0,
        N_GRAPHENE_CARBONS,
    )
)

VISIBLE_INDICES.extend(
    range(
        LIGAND_START,
        LIGAND_END,
    )
)

for water in range(
    N_WATERS
):

    base = (
        N_SOLUTE
        +
        WATER_SITES * water
    )

    VISIBLE_INDICES.extend(
        [
            base,
            base + 1,
            base + 2,
        ]
    )


if len(VISIBLE_INDICES) != EXPECTED_VISIBLE_ATOMS:
    raise RuntimeError(
        "Visible atom count construction failed: "
        f"{len(VISIBLE_INDICES)} vs "
        f"{EXPECTED_VISIBLE_ATOMS}"
    )


# ============================================================
# WATER WALLS
# ============================================================

LOWER_WALL_NM = 0.100

UPPER_WALL_NM = 6.800


# ============================================================
# INPUT CHECKS
# ============================================================

for path in [
    SYSTEM_XML,
    START_POSITIONS,
    START_VELOCITIES,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required input:\n{path}"
        )


# ============================================================
# LOAD ORIGINAL SYSTEM
# ============================================================

system = mm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text()
)


if system.getNumParticles() != EXPECTED_PARTICLES:

    raise RuntimeError(
        "Particle count mismatch: "
        f"{system.getNumParticles()} vs "
        f"{EXPECTED_PARTICLES}"
    )


# ============================================================
# IMPORTANT GPU SAFETY PATCH
#
# The graphene support force previously used:
#
#   global kz
#   per-particle z0
#
# Every graphene carbon has the SAME z0 = 0.5 nm.
#
# We verify that explicitly and then replace it with:
#
#   global kz
#   global z0
#
# This is mathematically identical for our flat support plane,
# while removing the per-particle parameter from the
# CustomExternalForce.
#
# If anything differs from our expected setup, ABORT.
# ============================================================

def patch_graphene_support_force(
    system,
):

    candidates = []

    for force_index in range(
        system.getNumForces()
    ):

        force = system.getForce(
            force_index
        )

        if (
            isinstance(
                force,
                mm.CustomExternalForce,
            )
            and
            force.getForceGroup() == 5
        ):

            candidates.append(
                (
                    force_index,
                    force,
                )
            )


    if len(candidates) != 1:

        raise RuntimeError(
            "Expected exactly one group-5 "
            "CustomExternalForce for graphene support; "
            f"found {len(candidates)}."
        )


    force_index, old_force = (
        candidates[0]
    )


    print()
    print(
        "Graphene support restraint found:"
    )

    print(
        f"  force index: "
        f"{force_index}"
    )

    print(
        f"  force group: "
        f"{old_force.getForceGroup()}"
    )

    print(
        f"  particles: "
        f"{old_force.getNumParticles()}"
    )

    print(
        f"  expression: "
        f"{old_force.getEnergyFunction()}"
    )


    if old_force.getNumParticles() != N_GRAPHENE_CARBONS:

        raise RuntimeError(
            "Graphene support does not contain "
            "exactly 1250 particles."
        )


    per_names = [
        old_force.getPerParticleParameterName(i)
        for i in range(
            old_force.getNumPerParticleParameters()
        )
    ]


    if per_names != ["z0"]:

        raise RuntimeError(
            "Expected graphene support per-particle "
            f"parameters ['z0'], found {per_names}."
        )


    global_values = {}

    for i in range(
        old_force.getNumGlobalParameters()
    ):

        name = (
            old_force
            .getGlobalParameterName(i)
        )

        value = (
            old_force
            .getGlobalParameterDefaultValue(i)
        )

        global_values[
            name
        ] = value


    if "kz" not in global_values:

        raise RuntimeError(
            "Graphene support force has no "
            "global parameter named 'kz'."
        )


    particle_indices = []

    z0_values = []


    for i in range(
        old_force.getNumParticles()
    ):

        (
            particle_index,
            parameters,
        ) = old_force.getParticleParameters(
            i
        )

        particle_indices.append(
            int(
                particle_index
            )
        )

        z0_values.append(
            float(
                parameters[0]
            )
        )


    if sorted(
        particle_indices
    ) != list(
        range(
            N_GRAPHENE_CARBONS
        )
    ):

        raise RuntimeError(
            "Group-5 support is not applied "
            "exactly to graphene carbon indices "
            "0-1249."
        )


    z0_values = np.asarray(
        z0_values,
        dtype=float,
    )


    z0_min = float(
        z0_values.min()
    )

    z0_max = float(
        z0_values.max()
    )

    z0_mean = float(
        z0_values.mean()
    )


    print(
        f"  kz: "
        f"{global_values['kz']}"
    )

    print(
        f"  z0 range: "
        f"{z0_min:.12f} -> "
        f"{z0_max:.12f} nm"
    )


    if (
        z0_max
        -
        z0_min
    ) > 1.0e-10:

        raise RuntimeError(
            "Graphene z0 values are not identical. "
            "Refusing to replace the per-particle "
            "restraint."
        )


    new_force = mm.CustomExternalForce(
        "0.5*kz*periodicdistance("
        "0,0,z,0,0,z0)^2"
    )


    new_force.addGlobalParameter(
        "kz",
        global_values["kz"],
    )

    new_force.addGlobalParameter(
        "z0",
        z0_mean,
    )


    for particle_index in (
        particle_indices
    ):

        new_force.addParticle(
            particle_index,
            [],
        )


    new_force.setForceGroup(
        old_force.getForceGroup()
    )

    try:

        new_force.setName(
            old_force.getName()
            +
            "_global_z0"
        )

    except Exception:

        pass


    system.removeForce(
        force_index
    )

    system.addForce(
        new_force
    )


    print(
        "PASS: graphene support restraint "
        "rewritten with GLOBAL z0"
    )

    print(
        "      Mathematical target plane "
        "and kz are unchanged."
    )


patch_graphene_support_force(
    system
)


# ============================================================
# CHECK FOR ANY OTHER PER-PARTICLE CUSTOM EXTERNAL FORCES
#
# For CUDA production we do not want to proceed blindly if
# another CustomExternalForce still carries per-particle data.
# ============================================================

remaining_external_pp = []


for force_index in range(
    system.getNumForces()
):

    force = system.getForce(
        force_index
    )

    if isinstance(
        force,
        mm.CustomExternalForce,
    ):

        npp = (
            force
            .getNumPerParticleParameters()
        )

        if npp > 0:

            names = [
                force.getPerParticleParameterName(i)
                for i in range(npp)
            ]

            remaining_external_pp.append(
                {
                    "force_index": force_index,
                    "group": force.getForceGroup(),
                    "parameters": names,
                    "expression": force.getEnergyFunction(),
                }
            )


if remaining_external_pp:

    print()
    print(
        "Remaining CustomExternalForce "
        "per-particle parameters:"
    )

    for item in (
        remaining_external_pp
    ):

        print(
            item
        )

    raise RuntimeError(
        "CUDA production aborted because another "
        "CustomExternalForce still has "
        "per-particle parameters."
    )


print()
print(
    "PASS: no CustomExternalForce "
    "per-particle parameters remain"
)


# ============================================================
# SAVE EXACT PRODUCTION SYSTEM
# ============================================================

SAFE_SYSTEM_XML.write_text(
    mm.XmlSerializer.serialize(
        system
    )
)


# ============================================================
# LOAD 500 ps STATE
# ============================================================

positions_nm = np.load(
    START_POSITIONS
)

velocities_nm_ps = np.load(
    START_VELOCITIES
)


if positions_nm.shape != (
    EXPECTED_PARTICLES,
    3,
):

    raise RuntimeError(
        "Starting position shape mismatch: "
        f"{positions_nm.shape}"
    )


if velocities_nm_ps.shape != (
    EXPECTED_PARTICLES,
    3,
):

    raise RuntimeError(
        "Starting velocity shape mismatch: "
        f"{velocities_nm_ps.shape}"
    )


# ============================================================
# DOF FOR TEMPERATURE REPORTING
# ============================================================

n_massive = 0


for i in range(
    system.getNumParticles()
):

    mass = (
        system.getParticleMass(i)
        .value_in_unit(
            unit.dalton
        )
    )

    if mass > 0.0:

        n_massive += 1


n_constraints = (
    system.getNumConstraints()
)


has_cm_remover = any(
    isinstance(
        system.getForce(i),
        mm.CMMotionRemover,
    )
    for i in range(
        system.getNumForces()
    )
)


degrees_of_freedom = (
    3 * n_massive
    -
    n_constraints
    -
    (
        3
        if has_cm_remover
        else 0
    )
)


R_KJ_MOL_K = (
    0.00831446261815324
)


print()
print(
    f"Massive particles: "
    f"{n_massive}"
)

print(
    f"Constraints: "
    f"{n_constraints}"
)

print(
    f"CMMotionRemover: "
    f"{has_cm_remover}"
)

print(
    f"Estimated DOF: "
    f"{degrees_of_freedom}"
)


# ============================================================
# DUMMY FULL TOPOLOGY FOR DCDREPORTER
#
# The Simulation topology contains one dummy atom for each
# OpenMM particle.
#
# DCDReporter(atomSubset=...) then writes exactly the:
#
#   1250 graphene carbons
#     70 ligand atoms
#  18303 water O/H/H atoms
#
# = 19623 visible atoms
# ============================================================

topology = app.Topology()

chain = topology.addChain()

residue = topology.addResidue(
    "SYS",
    chain,
)


for i in range(
    EXPECTED_PARTICLES
):

    topology.addAtom(
        "X",
        None,
        residue,
    )


topology.setPeriodicBoxVectors(
    system.getDefaultPeriodicBoxVectors()
)


if topology.getNumAtoms() != EXPECTED_PARTICLES:

    raise RuntimeError(
        "Dummy topology atom count mismatch."
    )


# ============================================================
# INTEGRATOR
# ============================================================

integrator = mm.LangevinMiddleIntegrator(
    TEMPERATURE_K
    *
    unit.kelvin,

    FRICTION_PER_PS
    /
    unit.picosecond,

    TIMESTEP_PS
    *
    unit.picoseconds,
)


integrator.setRandomNumberSeed(
    RANDOM_SEED
)


# ============================================================
# CUDA PLATFORM
# ============================================================

platform_names = [
    mm.Platform.getPlatform(i).getName()
    for i in range(
        mm.Platform.getNumPlatforms()
    )
]


if "CUDA" not in platform_names:

    raise RuntimeError(
        "CUDA platform is not available."
    )


platform = (
    mm.Platform
    .getPlatformByName(
        "CUDA"
    )
)


properties = {
    "Precision": "mixed",
}


# ============================================================
# SIMULATION
# ============================================================

simulation = app.Simulation(
    topology,
    system,
    integrator,
    platform,
    properties,
)


simulation.context.setPositions(
    positions_nm
    *
    unit.nanometer
)


simulation.context.setVelocities(
    velocities_nm_ps
    *
    (
        unit.nanometer
        /
        unit.picosecond
    )
)


simulation.context.setTime(
    START_TIME_PS
    *
    unit.picosecond
)


simulation.currentStep = (
    START_STEP
)


# ============================================================
# VISIBLE DCD
# ============================================================

if TRAJECTORY_DCD.exists():

    TRAJECTORY_DCD.unlink()


simulation.reporters.append(
    app.DCDReporter(
        str(
            TRAJECTORY_DCD
        ),
        REPORT_STEPS,
        append=False,
        enforcePeriodicBox=False,
        atomSubset=VISIBLE_INDICES,
    )
)


# ============================================================
# HELPERS
# ============================================================

def box_numpy(
    state,
):

    return np.asarray(
        state
        .getPeriodicBoxVectors(
            asNumpy=True
        )
        .value_in_unit(
            unit.nanometer
        ),
        dtype=float,
    )


def unwrap_cluster(
    xyz,
    box,
):
    """
    Make one small molecule whole across the periodic box.
    """

    xyz = np.asarray(
        xyz,
        dtype=float,
    )

    reference = xyz[0]

    inverse_box = np.linalg.inv(
        box
    )

    output = np.zeros_like(
        xyz
    )

    output[0] = reference


    for i in range(
        1,
        len(xyz),
    ):

        displacement = (
            xyz[i]
            -
            reference
        )

        fractional = (
            displacement
            @
            inverse_box
        )

        fractional -= np.round(
            fractional
        )

        output[i] = (
            reference
            +
            fractional
            @
            box
        )


    return output


def calculate_tilt_deg(
    xyz,
    box,
):
    """
    0 degrees = pyrene plane parallel to graphene/xy.
    """

    pyrene = xyz[
        PYRENE_GLOBAL
    ]

    pyrene = unwrap_cluster(
        pyrene,
        box,
    )

    centered = (
        pyrene
        -
        pyrene.mean(
            axis=0
        )
    )


    covariance = (
        centered.T
        @
        centered
    )


    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
    )


    normal = eigenvectors[
        :,
        int(
            np.argmin(
                eigenvalues
            )
        ),
    ]


    nz = float(
        np.clip(
            abs(
                normal[2]
            ),
            0.0,
            1.0,
        )
    )


    return float(
        np.degrees(
            np.arccos(
                nz
            )
        )
    )


def state_metrics(
    state,
):

    xyz = np.asarray(
        state
        .getPositions(
            asNumpy=True
        )
        .value_in_unit(
            unit.nanometer
        ),
        dtype=float,
    )


    box = box_numpy(
        state
    )


    potential = (
        state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )


    kinetic = (
        state
        .getKineticEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )


    temperature = (
        2.0
        *
        kinetic
        /
        (
            degrees_of_freedom
            *
            R_KJ_MOL_K
        )
    )


    graphene_z = xyz[
        :N_GRAPHENE_CARBONS,
        2
    ]


    graphene_mean_z = float(
        graphene_z.mean()
    )


    pyrene_mean_z = float(
        xyz[
            PYRENE_GLOBAL,
            2
        ].mean()
    )


    pyrene_height = (
        pyrene_mean_z
        -
        graphene_mean_z
    )


    tilt_deg = calculate_tilt_deg(
        xyz,
        box,
    )


    water_o_z = xyz[
        WATER_O_INDICES,
        2
    ]


    water_min = float(
        water_o_z.min()
    )

    water_max = float(
        water_o_z.max()
    )


    waters_below_graphene = int(
        np.count_nonzero(
            water_o_z
            <
            graphene_mean_z
        )
    )


    lower_wall_violation = float(
        max(
            0.0,
            LOWER_WALL_NM
            -
            water_min,
        )
    )


    upper_wall_violation = float(
        max(
            0.0,
            water_max
            -
            UPPER_WALL_NM,
        )
    )


    time_ps = (
        state
        .getTime()
        .value_in_unit(
            unit.picosecond
        )
    )


    return {
        "time_ps": float(
            time_ps
        ),
        "step": int(
            simulation.currentStep
        ),
        "temperature_K": float(
            temperature
        ),
        "potential_kJ_mol": float(
            potential
        ),
        "kinetic_kJ_mol": float(
            kinetic
        ),
        "pyrene_height_nm": float(
            pyrene_height
        ),
        "pyrene_tilt_deg": float(
            tilt_deg
        ),
        "graphene_z_mean_nm": float(
            graphene_mean_z
        ),
        "graphene_z_min_nm": float(
            graphene_z.min()
        ),
        "graphene_z_max_nm": float(
            graphene_z.max()
        ),
        "water_O_min_nm": float(
            water_min
        ),
        "water_O_max_nm": float(
            water_max
        ),
        "waters_below_graphene": int(
            waters_below_graphene
        ),
        "lower_wall_violation_nm": float(
            lower_wall_violation
        ),
        "upper_wall_violation_nm": float(
            upper_wall_violation
        ),
    }


def atomic_checkpoint(
    simulation,
    destination,
):

    destination = Path(
        destination
    )

    temporary = destination.with_suffix(
        destination.suffix
        +
        ".tmp"
    )


    checkpoint_bytes = (
        simulation
        .context
        .createCheckpoint()
    )


    temporary.write_bytes(
        checkpoint_bytes
    )


    os.replace(
        temporary,
        destination,
    )


def validate_metrics(
    metrics,
):

    float_keys = [
        "time_ps",
        "temperature_K",
        "potential_kJ_mol",
        "kinetic_kJ_mol",
        "pyrene_height_nm",
        "pyrene_tilt_deg",
        "graphene_z_mean_nm",
        "water_O_min_nm",
        "water_O_max_nm",
    ]


    for key in float_keys:

        if not math.isfinite(
            metrics[key]
        ):

            raise RuntimeError(
                "Non-finite value detected: "
                f"{key} = "
                f"{metrics[key]}"
            )


# ============================================================
# INITIAL STATE
# ============================================================

initial_state = (
    simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
    )
)


initial_metrics = state_metrics(
    initial_state
)


validate_metrics(
    initial_metrics
)


print()
print("=" * 76)
print("5 NS PRODUCTION MD")
print("=" * 76)

print(
    f"OpenMM version:          "
    f"{mm.__version__}"
)

print(
    f"Platform:                "
    f"{platform.getName()}"
)

print(
    f"Particles:               "
    f"{system.getNumParticles()}"
)

print(
    f"Visible DCD atoms:       "
    f"{len(VISIBLE_INDICES)}"
)

print(
    f"Waters:                  "
    f"{N_WATERS}"
)

print()

print(
    f"Temperature:             "
    f"{TEMPERATURE_K:.1f} K"
)

print(
    f"Ensemble:                NVT"
)

print(
    f"Integrator:              "
    f"LangevinMiddle"
)

print(
    f"Friction:                "
    f"{FRICTION_PER_PS:.1f} /ps"
)

print(
    f"Timestep:                "
    f"{TIMESTEP_PS * 1000:.1f} fs"
)

print(
    f"Starting time:           "
    f"{START_TIME_PS:.1f} ps"
)

print(
    f"Production length:       "
    f"{PRODUCTION_TIME_PS:.1f} ps"
)

print(
    f"Final time:              "
    f"{END_TIME_PS:.1f} ps"
)

print(
    f"Total integration steps: "
    f"{TOTAL_STEPS}"
)

print(
    f"Trajectory interval:     "
    f"{REPORT_INTERVAL_PS:.1f} ps"
)

print(
    f"Recovery checkpoint:     "
    f"every {CHECKPOINT_INTERVAL_PS:.1f} ps"
)

print()

print(
    "No barostat"
)

print(
    "No ions"
)

print(
    "No ligand restraint"
)

print(
    "Graphene z support ON"
)

print(
    "Water slab walls ON"
)

print()

print(
    "INITIAL 500 ps STATE"
)

print(
    f"  Temperature: "
    f"{initial_metrics['temperature_K']:.3f} K"
)

print(
    f"  Potential:   "
    f"{initial_metrics['potential_kJ_mol']:.3f} kJ/mol"
)

print(
    f"  Pyrene h:    "
    f"{initial_metrics['pyrene_height_nm']:.6f} nm"
)

print(
    f"  Pyrene tilt: "
    f"{initial_metrics['pyrene_tilt_deg']:.3f} deg"
)

print(
    f"  Water O:     "
    f"{initial_metrics['water_O_min_nm']:.6f} -> "
    f"{initial_metrics['water_O_max_nm']:.6f} nm"
)

print(
    f"  Below sheet: "
    f"{initial_metrics['waters_below_graphene']}"
)

print("=" * 76)
print()


# ============================================================
# LOG
# ============================================================

fieldnames = [
    "time_ps",
    "step",
    "temperature_K",
    "potential_kJ_mol",
    "kinetic_kJ_mol",
    "pyrene_height_nm",
    "pyrene_tilt_deg",
    "graphene_z_mean_nm",
    "graphene_z_min_nm",
    "graphene_z_max_nm",
    "water_O_min_nm",
    "water_O_max_nm",
    "waters_below_graphene",
    "lower_wall_violation_nm",
    "upper_wall_violation_nm",
]


records = [
    initial_metrics
]


start_wallclock = time.time()


with LOG_CSV.open(
    "w",
    newline="",
) as log_file:

    writer = csv.DictWriter(
        log_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerow(
        initial_metrics
    )

    log_file.flush()


    try:

        for report_index in range(
            1,
            N_REPORTS + 1,
        ):

            simulation.step(
                REPORT_STEPS
            )


            state = (
                simulation.context.getState(
                    getPositions=True,
                    getEnergy=True,
                )
            )


            metrics = state_metrics(
                state
            )


            validate_metrics(
                metrics
            )


            writer.writerow(
                metrics
            )

            log_file.flush()

            records.append(
                metrics
            )


            steps_from_start = (
                simulation.currentStep
                -
                START_STEP
            )


            # --------------------------------------------
            # Crash recovery checkpoint every 10 ps
            # --------------------------------------------

            if (
                steps_from_start
                %
                CHECKPOINT_STEPS
                ==
                0
            ):

                atomic_checkpoint(
                    simulation,
                    RECOVERY_CHECKPOINT,
                )


            # --------------------------------------------
            # Console every 20 ps
            # --------------------------------------------

            if (
                report_index
                %
                CONSOLE_EVERY_REPORTS
                ==
                0
            ):

                elapsed = (
                    time.time()
                    -
                    start_wallclock
                )

                ns_completed = (
                    (
                        metrics["time_ps"]
                        -
                        START_TIME_PS
                    )
                    /
                    1000.0
                )


                print(
                    f"t={metrics['time_ps']:8.1f} ps   "
                    f"T={metrics['temperature_K']:7.2f} K   "
                    f"h={metrics['pyrene_height_nm'] * 10:6.3f} A   "
                    f"tilt={metrics['pyrene_tilt_deg']:6.2f} deg   "
                    f"water={metrics['water_O_min_nm']:.3f}"
                    f"->{metrics['water_O_max_nm']:.3f} nm   "
                    f"below={metrics['waters_below_graphene']}   "
                    f"wall+={metrics['upper_wall_violation_nm']:.4f} nm   "
                    f"completed={ns_completed:.3f} ns"
                )


    except BaseException as exc:

        print()
        print("=" * 76)
        print("PRODUCTION RUN INTERRUPTED")
        print("=" * 76)

        print(
            f"Exception: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )


        try:

            atomic_checkpoint(
                simulation,
                RECOVERY_CHECKPOINT,
            )

            print(
                "Emergency recovery checkpoint saved:"
            )

            print(
                RECOVERY_CHECKPOINT
            )

        except Exception as checkpoint_exc:

            print(
                "WARNING: emergency checkpoint "
                "could not be saved:"
            )

            print(
                checkpoint_exc
            )


        raise


# ============================================================
# FINAL STATE
# ============================================================

final_state = (
    simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
    )
)


final_metrics = state_metrics(
    final_state
)


validate_metrics(
    final_metrics
)


final_positions_nm = (
    final_state
    .getPositions(
        asNumpy=True
    )
    .value_in_unit(
        unit.nanometer
    )
)


final_velocities_nm_ps = (
    final_state
    .getVelocities(
        asNumpy=True
    )
    .value_in_unit(
        unit.nanometer
        /
        unit.picosecond
    )
)


np.save(
    FINAL_POSITIONS,
    np.asarray(
        final_positions_nm,
        dtype=float,
    ),
)


np.save(
    FINAL_VELOCITIES,
    np.asarray(
        final_velocities_nm_ps,
        dtype=float,
    ),
)


atomic_checkpoint(
    simulation,
    FINAL_CHECKPOINT,
)


atomic_checkpoint(
    simulation,
    RECOVERY_CHECKPOINT,
)


# ============================================================
# SUMMARY
# ============================================================

times = np.asarray(
    [
        r["time_ps"]
        for r in records
    ],
    dtype=float,
)

temperatures = np.asarray(
    [
        r["temperature_K"]
        for r in records
    ],
    dtype=float,
)

heights_nm = np.asarray(
    [
        r["pyrene_height_nm"]
        for r in records
    ],
    dtype=float,
)

tilts_deg = np.asarray(
    [
        r["pyrene_tilt_deg"]
        for r in records
    ],
    dtype=float,
)

below_counts = np.asarray(
    [
        r["waters_below_graphene"]
        for r in records
    ],
    dtype=int,
)

upper_violations = np.asarray(
    [
        r["upper_wall_violation_nm"]
        for r in records
    ],
    dtype=float,
)


# Last 500 ps summary
summary_start_ps = (
    END_TIME_PS
    -
    500.0
)

mask_last_500 = (
    times
    >=
    summary_start_ps
)


mean_T_last500 = float(
    temperatures[
        mask_last_500
    ].mean()
)

std_T_last500 = float(
    temperatures[
        mask_last_500
    ].std()
)

mean_height_last500_nm = float(
    heights_nm[
        mask_last_500
    ].mean()
)

std_height_last500_nm = float(
    heights_nm[
        mask_last_500
    ].std()
)

min_height_last500_nm = float(
    heights_nm[
        mask_last_500
    ].min()
)

max_height_last500_nm = float(
    heights_nm[
        mask_last_500
    ].max()
)

mean_tilt_last500 = float(
    tilts_deg[
        mask_last_500
    ].mean()
)

max_tilt_last500 = float(
    tilts_deg[
        mask_last_500
    ].max()
)


metadata = {
    "protocol": {
        "ensemble": "NVT",
        "temperature_K": TEMPERATURE_K,
        "friction_per_ps": FRICTION_PER_PS,
        "timestep_ps": TIMESTEP_PS,
        "start_time_ps": START_TIME_PS,
        "production_time_ps": PRODUCTION_TIME_PS,
        "end_time_ps": END_TIME_PS,
        "report_interval_ps": REPORT_INTERVAL_PS,
        "checkpoint_interval_ps": CHECKPOINT_INTERVAL_PS,
        "random_seed": RANDOM_SEED,
        "barostat": False,
        "ions": False,
        "ligand_restraint": False,
        "graphene_support": True,
        "water_walls": True,
    },

    "system": {
        "particles": system.getNumParticles(),
        "massive_particles": n_massive,
        "constraints": n_constraints,
        "degrees_of_freedom": degrees_of_freedom,
        "waters": N_WATERS,
        "visible_dcd_atoms": len(
            VISIBLE_INDICES
        ),
    },

    "gpu_safety": {
        "graphene_support_rewritten_to_global_z0": True,
        "remaining_custom_external_per_particle_parameters": 0,
        "safe_system_xml": str(
            SAFE_SYSTEM_XML
        ),
    },

    "final": final_metrics,

    "last_500ps": {
        "temperature_mean_K": mean_T_last500,
        "temperature_std_K": std_T_last500,

        "pyrene_height_mean_nm": mean_height_last500_nm,
        "pyrene_height_std_nm": std_height_last500_nm,
        "pyrene_height_min_nm": min_height_last500_nm,
        "pyrene_height_max_nm": max_height_last500_nm,

        "pyrene_tilt_mean_deg": mean_tilt_last500,
        "pyrene_tilt_max_deg": max_tilt_last500,

        "max_waters_below_graphene": int(
            below_counts[
                mask_last_500
            ].max()
        ),

        "max_upper_wall_violation_nm": float(
            upper_violations[
                mask_last_500
            ].max()
        ),
    },

    "outputs": {
        "positions": str(
            FINAL_POSITIONS
        ),
        "velocities": str(
            FINAL_VELOCITIES
        ),
        "checkpoint": str(
            FINAL_CHECKPOINT
        ),
        "recovery_checkpoint": str(
            RECOVERY_CHECKPOINT
        ),
        "trajectory": str(
            TRAJECTORY_DCD
        ),
        "log": str(
            LOG_CSV
        ),
        "metadata": str(
            METADATA_JSON
        ),
    },
}


METADATA_JSON.write_text(
    json.dumps(
        metadata,
        indent=2,
    )
)


wallclock_seconds = (
    time.time()
    -
    start_wallclock
)


print()
print("=" * 76)
print("5 NS PRODUCTION COMPLETE")
print("=" * 76)

print(
    f"Final time: "
    f"{final_metrics['time_ps']:.3f} ps"
)

print(
    f"Final temperature: "
    f"{final_metrics['temperature_K']:.3f} K"
)

print()

print(
    "LAST 500 ps:"
)

print(
    f"  Temperature: "
    f"{mean_T_last500:.3f} +/- "
    f"{std_T_last500:.3f} K"
)

print(
    f"  Pyrene height: "
    f"{mean_height_last500_nm:.6f} +/- "
    f"{std_height_last500_nm:.6f} nm"
)

print(
    f"                 "
    f"{mean_height_last500_nm * 10:.3f} +/- "
    f"{std_height_last500_nm * 10:.3f} A"
)

print(
    f"  Height range: "
    f"{min_height_last500_nm * 10:.3f} -> "
    f"{max_height_last500_nm * 10:.3f} A"
)

print(
    f"  Mean tilt: "
    f"{mean_tilt_last500:.3f} deg"
)

print(
    f"  Max tilt:  "
    f"{max_tilt_last500:.3f} deg"
)

print(
    f"  Max waters below graphene: "
    f"{int(below_counts[mask_last_500].max())}"
)

print(
    f"  Max upper-wall violation: "
    f"{float(upper_violations[mask_last_500].max()):.6f} nm"
)

print()

print(
    f"Wall-clock runtime: "
    f"{wallclock_seconds / 3600.0:.3f} h"
)

print()

print(
    "OUTPUTS"
)

print(
    f"  Positions:  "
    f"{FINAL_POSITIONS}"
)

print(
    f"  Velocities: "
    f"{FINAL_VELOCITIES}"
)

print(
    f"  Checkpoint: "
    f"{FINAL_CHECKPOINT}"
)

print(
    f"  Recovery:   "
    f"{RECOVERY_CHECKPOINT}"
)

print(
    f"  DCD:        "
    f"{TRAJECTORY_DCD}"
)

print(
    f"  CSV:        "
    f"{LOG_CSV}"
)

print(
    f"  JSON:       "
    f"{METADATA_JSON}"
)

print(
    f"  Safe XML:   "
    f"{SAFE_SYSTEM_XML}"
)

print()
print(
    "PRODUCTION_300K_5NS_PASS"
)
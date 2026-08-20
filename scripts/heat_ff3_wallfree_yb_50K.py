from pathlib import Path
import csv
import json
import math

import numpy as np

from openmm import openmm, unit


# ============================================================
# STAGED NVT HEATING — STAGE 1: 50 K
#
# INPUT:
#   Corrected FF-3 minimized wall-free/YB system.
#
# THIS STAGE:
#   NVT
#   50 K
#   20 ps
#   1 fs timestep
#   LangevinMiddleIntegrator
#
# IMPORTANT:
#   - Graphene support remains active.
#   - Dynamic Yeh-Berkowitz slab correction remains active.
#   - No runtime water wall is permitted.
#   - Pyrene-PEG5 remains completely unrestrained.
#   - No ions are added.
#   - No barostat is added.
#   - This is preparation, NOT production MD.
# ============================================================


# ============================================================
# 1. SYSTEM INDEXING
# ============================================================

N_GRAPHENE_CARBONS = 1250
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70
N_SOLUTE = 3820

SITES_PER_WATER = 4

LIGAND_START = N_GRAPHENE_TOTAL
LIGAND_STOP = N_SOLUTE


# ============================================================
# 2. HEATING PARAMETERS
# ============================================================

TARGET_TEMPERATURE_K = 50.0

FRICTION_PER_PS = 1.0

TIMESTEP_FS = 1.0
TIMESTEP_PS = TIMESTEP_FS / 1000.0

TOTAL_TIME_PS = 20.0

TOTAL_STEPS = int(
    round(
        TOTAL_TIME_PS
        / TIMESTEP_PS
    )
)

REPORT_INTERVAL_PS = 1.0

REPORT_STEPS = int(
    round(
        REPORT_INTERVAL_PS
        / TIMESTEP_PS
    )
)

CONSTRAINT_TOLERANCE = 1.0e-6

LANGEVIN_SEED = 20260818
VELOCITY_SEED = 20260819


# ============================================================
# 3. GEOMETRIC MONITORING
#
# FF-3 is wall-free. No lower/upper water-wall coordinates
# are defined or permitted.
# ============================================================


# ============================================================
# 4. INPUT FILES
# ============================================================

SYSTEM_XML = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb.xml"
)

POSITIONS_FILE = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb_"
    "minimized_positions_nm.npy"
)

LIGAND_MOL2 = Path(
    "structures/"
    "pyrene_peg5_propargyl.mol2"
)


# ============================================================
# 5. OUTPUT FILES
# ============================================================

OUTPUT_POSITIONS = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb_"
    "heated_050K_positions_nm.npy"
)

OUTPUT_VELOCITIES = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb_"
    "heated_050K_velocities_nm_per_ps.npy"
)

OUTPUT_CHECKPOINT = Path(
    "checkpoints/"
    "ff3_wallfree_yb_heating_050K.chk"
)

OUTPUT_LOG = Path(
    "analysis/"
    "ff3_wallfree_yb_heating_050K_log.csv"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "ff3_wallfree_yb_heating_050K.json"
)


# ============================================================
# 6. VERIFY INPUT FILES
# ============================================================

for path in [
    SYSTEM_XML,
    POSITIONS_FILE,
    LIGAND_MOL2,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# 7. LOAD SYSTEM
# ============================================================

with open(
    SYSTEM_XML,
    "r",
) as f:

    system = (
        openmm.XmlSerializer.deserialize(
            f.read()
        )
    )


positions_nm = np.load(
    POSITIONS_FILE
)


n_particles = (
    system.getNumParticles()
)


if positions_nm.shape != (
    n_particles,
    3,
):

    raise RuntimeError(
        "Coordinate/System mismatch: "
        f"positions={positions_nm.shape}, "
        f"particles={n_particles}"
    )


water_sites = (
    n_particles
    - N_SOLUTE
)


if (
    water_sites
    % SITES_PER_WATER
) != 0:

    raise RuntimeError(
        "Water site count is not divisible by 4."
    )


n_waters = (
    water_sites
    // SITES_PER_WATER
)


if n_waters <= 0:

    raise RuntimeError(
        "No OPC waters were detected."
    )


# ============================================================
# 8. STRICT FF-3 HAMILTONIAN PREFLIGHT
#
# This gate executes before any Context is created and before
# any molecular dynamics can occur.
# ============================================================

SUPPORT_FORCE_NAME = (
    "GrapheneSupportZRestraint_global_z0"
)

SUPPORT_FORCE_GROUP = 5

SUPPORT_EXPRESSION = (
    "0.5*kz*periodicdistance(0,0,z,0,0,z0)^2"
)

SUPPORT_KZ = 1000.0
SUPPORT_Z0_NM = 0.5


YB_FORCE_NAME = (
    "Yeh-Berkowitz slab correction"
)

YB_FORCE_GROUP = 29

YB_EXPRESSION = (
    "yb_coeff*Mz^2"
)

K_E = 138.935456


support_matches = []
yb_matches = []

custom_external_forces = []

wall_like_forces = []
unexpected_restraints = []


for force_index in range(
    system.getNumForces()
):

    force = system.getForce(
        force_index
    )

    force_name = (
        force.getName()
        or
        ""
    )

    force_name_lower = (
        force_name.lower()
    )


    if "wall" in force_name_lower:

        wall_like_forces.append(
            (
                force_index,
                force_name,
            )
        )


    if (
        "restraint"
        in force_name_lower
        and
        force_name
        !=
        SUPPORT_FORCE_NAME
    ):

        unexpected_restraints.append(
            (
                force_index,
                force_name,
            )
        )


    if isinstance(
        force,
        openmm.CustomExternalForce,
    ):

        custom_external_forces.append(
            (
                force_index,
                force,
            )
        )

        if force_name == SUPPORT_FORCE_NAME:

            support_matches.append(
                (
                    force_index,
                    force,
                )
            )


    if isinstance(
        force,
        openmm.CustomCVForce,
    ):

        if force_name == YB_FORCE_NAME:

            yb_matches.append(
                (
                    force_index,
                    force,
                )
            )


if wall_like_forces:

    raise RuntimeError(
        "Runtime water-wall-like force detected: "
        f"{wall_like_forces}"
    )


if unexpected_restraints:

    raise RuntimeError(
        "Unexpected restraint-like force detected: "
        f"{unexpected_restraints}"
    )


if len(
    custom_external_forces
) != 1:

    raise RuntimeError(
        "Expected exactly one CustomExternalForce "
        "(graphene support only); found "
        f"{len(custom_external_forces)}."
    )


if len(
    support_matches
) != 1:

    raise RuntimeError(
        "Expected exactly one validated graphene "
        "support force; found "
        f"{len(support_matches)}."
    )


support_index, support_force = (
    support_matches[0]
)


if support_force.getForceGroup() != (
    SUPPORT_FORCE_GROUP
):

    raise RuntimeError(
        "Graphene support force-group mismatch: "
        f"{support_force.getForceGroup()}"
    )


if support_force.getEnergyFunction() != (
    SUPPORT_EXPRESSION
):

    raise RuntimeError(
        "Graphene support expression mismatch: "
        f"{support_force.getEnergyFunction()}"
    )


if support_force.getNumParticles() != (
    N_GRAPHENE_CARBONS
):

    raise RuntimeError(
        "Graphene support particle-count mismatch: "
        f"{support_force.getNumParticles()}"
    )


supported_particles = []


for i in range(
    support_force.getNumParticles()
):

    particle_index, parameters = (
        support_force.getParticleParameters(
            i
        )
    )

    supported_particles.append(
        int(
            particle_index
        )
    )


if sorted(
    supported_particles
) != list(
    range(
        N_GRAPHENE_CARBONS
    )
):

    raise RuntimeError(
        "Graphene support is not applied exactly "
        "to carbon particles 0..1249."
    )


support_globals = {}


for i in range(
    support_force.getNumGlobalParameters()
):

    name = (
        support_force
        .getGlobalParameterName(
            i
        )
    )

    value = float(
        support_force
        .getGlobalParameterDefaultValue(
            i
        )
    )

    support_globals[
        name
    ] = value


if set(
    support_globals
) != {
    "kz",
    "z0",
}:

    raise RuntimeError(
        "Unexpected graphene-support globals: "
        f"{support_globals}"
    )


if not math.isclose(
    support_globals["kz"],
    SUPPORT_KZ,
    rel_tol=0.0,
    abs_tol=1.0e-12,
):

    raise RuntimeError(
        "Graphene support kz mismatch: "
        f"{support_globals['kz']}"
    )


if not math.isclose(
    support_globals["z0"],
    SUPPORT_Z0_NM,
    rel_tol=0.0,
    abs_tol=1.0e-12,
):

    raise RuntimeError(
        "Graphene support z0 mismatch: "
        f"{support_globals['z0']}"
    )


if len(
    yb_matches
) != 1:

    raise RuntimeError(
        "Expected exactly one Yeh-Berkowitz "
        "correction; found "
        f"{len(yb_matches)}."
    )


yb_index, yb_force = (
    yb_matches[0]
)


if yb_force.getForceGroup() != (
    YB_FORCE_GROUP
):

    raise RuntimeError(
        "Yeh-Berkowitz force-group mismatch: "
        f"{yb_force.getForceGroup()}"
    )


if yb_force.getEnergyFunction() != (
    YB_EXPRESSION
):

    raise RuntimeError(
        "Yeh-Berkowitz expression mismatch: "
        f"{yb_force.getEnergyFunction()}"
    )


yb_globals = {}


for i in range(
    yb_force.getNumGlobalParameters()
):

    name = (
        yb_force
        .getGlobalParameterName(
            i
        )
    )

    value = float(
        yb_force
        .getGlobalParameterDefaultValue(
            i
        )
    )

    yb_globals[
        name
    ] = value


if set(
    yb_globals
) != {
    "yb_coeff",
}:

    raise RuntimeError(
        "Unexpected Yeh-Berkowitz globals: "
        f"{yb_globals}"
    )


yb_coeff = (
    yb_globals[
        "yb_coeff"
    ]
)


if (
    not math.isfinite(
        yb_coeff
    )
    or
    yb_coeff <= 0.0
):

    raise RuntimeError(
        "Invalid Yeh-Berkowitz coefficient: "
        f"{yb_coeff}"
    )


box_vectors = (
    system.getDefaultPeriodicBoxVectors()
)


box_nm = np.asarray(
    [
        [
            float(
                component.value_in_unit(
                    unit.nanometer
                )
            )
            for component in vector
        ]
        for vector in box_vectors
    ],
    dtype=float,
)


box_volume_nm3 = float(
    abs(
        np.linalg.det(
            box_nm
        )
    )
)


if (
    not math.isfinite(
        box_volume_nm3
    )
    or
    box_volume_nm3 <= 0.0
):

    raise RuntimeError(
        "Invalid periodic box volume: "
        f"{box_volume_nm3}"
    )


expected_yb_coeff = (
    2.0
    *
    math.pi
    *
    K_E
    /
    box_volume_nm3
)


if not math.isclose(
    yb_coeff,
    expected_yb_coeff,
    rel_tol=1.0e-10,
    abs_tol=1.0e-12,
):

    raise RuntimeError(
        "Yeh-Berkowitz coefficient does not match "
        "the current box volume: "
        f"stored={yb_coeff}, "
        f"expected={expected_yb_coeff}"
    )


print()
print("=" * 78)
print("FF-3 HAMILTONIAN PREFLIGHT")
print("=" * 78)

print(
    "Graphene support: PASS"
)

print(
    f"  force index: {support_index}"
)

print(
    f"  force group: "
    f"{support_force.getForceGroup()}"
)

print(
    f"  supported carbons: "
    f"{support_force.getNumParticles()}"
)

print(
    f"  kz: "
    f"{support_globals['kz']}"
)

print(
    f"  z0: "
    f"{support_globals['z0']} nm"
)


print(
    "Yeh-Berkowitz: PASS"
)

print(
    f"  force index: {yb_index}"
)

print(
    f"  force group: "
    f"{yb_force.getForceGroup()}"
)

print(
    f"  box volume: "
    f"{box_volume_nm3:.12f} nm^3"
)

print(
    f"  yb_coeff: "
    f"{yb_coeff:.12f}"
)

print(
    "Runtime water wall: ABSENT"
)

print(
    "Additional positional restraints: ABSENT"
)


# ============================================================
# 9. IDENTIFY PYRENE / AROMATIC ATOMS FROM MOL2
# ============================================================

mol2_lines = (
    LIGAND_MOL2
    .read_text()
    .splitlines()
)


atom_section = None
bond_section = None


for i, line in enumerate(
    mol2_lines
):

    stripped = (
        line.strip()
    )

    if stripped == "@<TRIPOS>ATOM":

        atom_section = (
            i + 1
        )

    elif stripped == "@<TRIPOS>BOND":

        bond_section = (
            i + 1
        )


if atom_section is None:

    raise RuntimeError(
        "MOL2 ATOM section not found."
    )


if bond_section is None:

    raise RuntimeError(
        "MOL2 BOND section not found."
    )


ligand_atom_types = []


for line in mol2_lines[
    atom_section:
]:

    if line.startswith(
        "@<TRIPOS>"
    ):

        break

    if not line.strip():

        continue

    fields = line.split()

    if len(fields) < 6:

        continue

    ligand_atom_types.append(
        fields[5].lower()
    )


if len(
    ligand_atom_types
) != N_LIGAND:

    raise RuntimeError(
        "Unexpected ligand atom count in MOL2: "
        f"{len(ligand_atom_types)}"
    )


aromatic_atoms = set()


for line in mol2_lines[
    bond_section:
]:

    if line.startswith(
        "@<TRIPOS>"
    ):

        break

    if not line.strip():

        continue

    fields = line.split()

    if len(fields) < 4:

        continue

    atom_1 = int(
        fields[1]
    ) - 1

    atom_2 = int(
        fields[2]
    ) - 1

    bond_type = (
        fields[3]
        .lower()
    )

    if bond_type in {
        "ar",
        "aro",
    }:

        aromatic_atoms.add(
            atom_1
        )

        aromatic_atoms.add(
            atom_2
        )


# GAFF-style aromatic atom-type fallback.
for i, atom_type in enumerate(
    ligand_atom_types
):

    if atom_type in {
        "ca",
        "cp",
        "cq",
        "cc",
        "cd",
    }:

        aromatic_atoms.add(
            i
        )


aromatic_atoms = sorted(
    aromatic_atoms
)


if len(
    aromatic_atoms
) != 16:

    raise RuntimeError(
        "Expected 16 aromatic/pyrene atoms, "
        f"found {len(aromatic_atoms)}."
    )


aromatic_atoms = np.asarray(
    aromatic_atoms,
    dtype=int,
)


# ============================================================
# 9. DETERMINE DEGREES OF FREEDOM
#
# Only massive particles contribute kinetic DOF.
# Distance constraints remove one DOF each.
# ============================================================

n_massive_particles = 0


for i in range(
    n_particles
):

    mass = (
        system
        .getParticleMass(i)
        .value_in_unit(
            unit.dalton
        )
    )

    if mass > 0.0:

        n_massive_particles += 1


degrees_of_freedom = (
    3
    * n_massive_particles
    -
    system.getNumConstraints()
)


has_cm_remover = False


for force in system.getForces():

    if (
        force.__class__.__name__
        ==
        "CMMotionRemover"
    ):

        has_cm_remover = True

        degrees_of_freedom -= 3


if degrees_of_freedom <= 0:

    raise RuntimeError(
        "Invalid degrees of freedom."
    )


# Gas constant in kJ/(mol K)
R_KJ_PER_MOL_K = (
    0.00831446261815324
)


# ============================================================
# 10. CREATE 50 K LANGEVIN INTEGRATOR
# ============================================================

integrator = (
    openmm.LangevinMiddleIntegrator(

        TARGET_TEMPERATURE_K
        * unit.kelvin,

        FRICTION_PER_PS
        / unit.picosecond,

        TIMESTEP_PS
        * unit.picoseconds,
    )
)


integrator.setConstraintTolerance(
    CONSTRAINT_TOLERANCE
)


integrator.setRandomNumberSeed(
    LANGEVIN_SEED
)


# ============================================================
# 11. CREATE CUDA CONTEXT
# ============================================================

platform = (
    openmm.Platform
    .getPlatformByName(
        "CUDA"
    )
)


try:

    context = openmm.Context(

        system,
        integrator,
        platform,

        {
            "Precision":
                "mixed"
        },
    )

    cuda_precision = (
        "mixed"
    )


except Exception:

    context = openmm.Context(
        system,
        integrator,
        platform,
    )

    cuda_precision = (
        "platform default"
    )


# ============================================================
# 12. LOAD MINIMIZED POSITIONS
# ============================================================

context.setPositions(
    positions_nm
    * unit.nanometer
)


# Enforce rigid-water constraints and recompute virtual sites.
context.applyConstraints(
    CONSTRAINT_TOLERANCE
)

context.computeVirtualSites()


# ============================================================
# 13. ASSIGN INITIAL 50 K VELOCITIES
#
# This occurs ONLY for the first heating stage.
# Later stages must continue the existing velocities.
# ============================================================

context.setVelocitiesToTemperature(

    TARGET_TEMPERATURE_K
    * unit.kelvin,

    VELOCITY_SEED,
)


context.applyVelocityConstraints(
    CONSTRAINT_TOLERANCE
)


context.setTime(
    0.0
    * unit.picoseconds
)


# ============================================================
# 14. HELPERS
# ============================================================

def state_arrays_and_metrics():

    state = context.getState(

        positions=True,
        velocities=True,
        energy=True,
        enforcePeriodicBox=False,
    )

    positions = np.asarray(

        state
        .getPositions(
            asNumpy=True
        )
        .value_in_unit(
            unit.nanometer
        ),

        dtype=float,
    )

    velocities = np.asarray(

        state
        .getVelocities(
            asNumpy=True
        )
        .value_in_unit(
            unit.nanometer
            /
            unit.picosecond
        ),

        dtype=float,
    )

    potential_energy = float(

        state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    kinetic_energy = float(

        state
        .getKineticEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    temperature = (

        2.0
        * kinetic_energy

        /

        (
            degrees_of_freedom
            *
            R_KJ_PER_MOL_K
        )
    )

    time_ps = float(

        state
        .getTime()
        .value_in_unit(
            unit.picosecond
        )
    )

    graphene_z = (
        positions[
            :N_GRAPHENE_CARBONS,
            2
        ]
    )

    graphene_z_min = float(
        graphene_z.min()
    )

    graphene_z_max = float(
        graphene_z.max()
    )

    graphene_z_mean = float(
        graphene_z.mean()
    )

    ligand = (
        positions[
            LIGAND_START:
            LIGAND_STOP
        ]
    )

    ligand_z_min = float(
        ligand[
            :,
            2
        ].min()
    )

    ligand_z_max = float(
        ligand[
            :,
            2
        ].max()
    )

    pyrene_z_mean = float(

        ligand[
            aromatic_atoms,
            2
        ].mean()
    )

    pyrene_height = (
        pyrene_z_mean
        -
        graphene_z_mean
    )

    oxygen_indices = (

        N_SOLUTE
        +
        4
        * np.arange(
            n_waters
        )
    )

    oxygen_z = (
        positions[
            oxygen_indices,
            2
        ]
    )

    water_z_min = float(
        oxygen_z.min()
    )

    water_z_max = float(
        oxygen_z.max()
    )

    water_below_graphene = int(

        np.sum(
            oxygen_z
            <
            graphene_z_max
        )
    )

    total_energy = (
        potential_energy
        +
        kinetic_energy
    )

    metrics = {

        "time_ps":
            time_ps,

        "temperature_K":
            temperature,

        "potential_kJ_per_mol":
            potential_energy,

        "kinetic_kJ_per_mol":
            kinetic_energy,

        "total_kJ_per_mol":
            total_energy,

        "graphene_z_min_nm":
            graphene_z_min,

        "graphene_z_max_nm":
            graphene_z_max,

        "graphene_z_mean_nm":
            graphene_z_mean,

        "ligand_z_min_nm":
            ligand_z_min,

        "ligand_z_max_nm":
            ligand_z_max,

        "pyrene_z_mean_nm":
            pyrene_z_mean,

        "pyrene_height_above_graphene_nm":
            pyrene_height,

        "water_z_min_nm":
            water_z_min,

        "water_z_max_nm":
            water_z_max,

        "water_below_graphene":
            water_below_graphene,
    }

    return (
        state,
        positions,
        velocities,
        metrics,
    )


def check_finite(
    metrics,
):

    for key in [
        "temperature_K",
        "potential_kJ_per_mol",
        "kinetic_kJ_per_mol",
        "total_kJ_per_mol",
        "pyrene_height_above_graphene_nm",
        "water_z_min_nm",
        "water_z_max_nm",
    ]:

        if not math.isfinite(
            metrics[
                key
            ]
        ):

            raise RuntimeError(
                "Non-finite MD value detected: "
                f"{key}={metrics[key]}"
            )


# ============================================================
# 15. HEADER
# ============================================================

print()
print("=" * 78)
print("STAGED NVT HEATING — STAGE 1")
print("=" * 78)

print()
print("SYSTEM")
print("------")

print(
    "Particles:",
    n_particles,
)

print(
    "Massive particles:",
    n_massive_particles,
)

print(
    "OPC waters:",
    n_waters,
)

print(
    "Constraints:",
    system.getNumConstraints(),
)

print(
    "Degrees of freedom:",
    degrees_of_freedom,
)

print(
    "CMMotionRemover:",
    "YES" if has_cm_remover else "NO",
)

print(
    "Platform:",
    platform.getName(),
)

print(
    "CUDA precision:",
    cuda_precision,
)


print()
print("HEATING PROTOCOL")
print("----------------")

print(
    f"Target temperature: "
    f"{TARGET_TEMPERATURE_K:.1f} K"
)

print(
    f"Integrator: "
    f"LangevinMiddleIntegrator"
)

print(
    f"Friction: "
    f"{FRICTION_PER_PS:.3f} /ps"
)

print(
    f"Timestep: "
    f"{TIMESTEP_FS:.3f} fs"
)

print(
    f"Total time: "
    f"{TOTAL_TIME_PS:.1f} ps"
)

print(
    f"Total steps: "
    f"{TOTAL_STEPS}"
)

print(
    f"Report interval: "
    f"{REPORT_INTERVAL_PS:.1f} ps"
)

print(
    "Ensemble: NVT"
)

print(
    "Barostat: NONE"
)

print(
    "Added ions: NONE"
)

print(
    "Pyrene-PEG5 restraint: NONE"
)


# ============================================================
# 16. INITIAL STATE AFTER VELOCITY ASSIGNMENT
# ============================================================

(
    state,
    current_positions,
    current_velocities,
    initial_metrics,
) = state_arrays_and_metrics()


check_finite(
    initial_metrics
)


print()
print("=" * 78)
print("INITIAL 50 K STATE")
print("=" * 78)

print()

print(
    f"Temperature: "
    f"{initial_metrics['temperature_K']:.3f} K"
)

print(
    f"Potential energy: "
    f"{initial_metrics['potential_kJ_per_mol']:.3f} kJ/mol"
)

print(
    f"Kinetic energy: "
    f"{initial_metrics['kinetic_kJ_per_mol']:.3f} kJ/mol"
)

print(
    f"Pyrene height above graphene: "
    f"{initial_metrics['pyrene_height_above_graphene_nm']:.6f} nm"
)

print(
    f"Water O z range: "
    f"{initial_metrics['water_z_min_nm']:.6f} "
    f"to "
    f"{initial_metrics['water_z_max_nm']:.6f} nm"
)

print(
    "Water O below graphene:",
    initial_metrics[
        "water_below_graphene"
    ],
)


# ============================================================
# 17. RUN 20 ps AT 50 K
# ============================================================

print()
print("=" * 78)
print("RUNNING 50 K NVT")
print("=" * 78)

print()
print(
    " time(ps)    T(K)         "
    "Potential(kJ/mol)   "
    "Pyrene-h(nm)   "
    "Water-z(nm)          "
    "below-G"
)

print(
    "-" * 112
)


log_rows = []


def append_metrics(
    metrics,
):

    log_rows.append(
        metrics.copy()
    )


append_metrics(
    initial_metrics
)


print(
    f"{initial_metrics['time_ps']:9.3f} "
    f"{initial_metrics['temperature_K']:8.3f} "
    f"{initial_metrics['potential_kJ_per_mol']:18.3f} "
    f"{initial_metrics['pyrene_height_above_graphene_nm']:13.6f} "
    f"{initial_metrics['water_z_min_nm']:7.4f}-"
    f"{initial_metrics['water_z_max_nm']:7.4f} "
    f"{initial_metrics['water_below_graphene']:8d}"
)


steps_completed = 0


while steps_completed < (
    TOTAL_STEPS
):

    steps_this_block = min(

        REPORT_STEPS,

        TOTAL_STEPS
        -
        steps_completed,
    )

    integrator.step(
        steps_this_block
    )

    steps_completed += (
        steps_this_block
    )

    (
        state,
        current_positions,
        current_velocities,
        metrics,
    ) = state_arrays_and_metrics()

    check_finite(
        metrics
    )

    append_metrics(
        metrics
    )

    print(
        f"{metrics['time_ps']:9.3f} "
        f"{metrics['temperature_K']:8.3f} "
        f"{metrics['potential_kJ_per_mol']:18.3f} "
        f"{metrics['pyrene_height_above_graphene_nm']:13.6f} "
        f"{metrics['water_z_min_nm']:7.4f}-"
        f"{metrics['water_z_max_nm']:7.4f} "
        f"{metrics['water_below_graphene']:8d}"
    )


# ============================================================
# 18. FINAL STATE
# ============================================================

(
    final_state,
    final_positions,
    final_velocities,
    final_metrics,
) = state_arrays_and_metrics()


check_finite(
    final_metrics
)


# ============================================================
# 19. SAVE COORDINATES + VELOCITIES
# ============================================================

OUTPUT_POSITIONS.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_POSITIONS,
    final_positions,
)


np.save(
    OUTPUT_VELOCITIES,
    final_velocities,
)


# ============================================================
# 20. SAVE OPENMM CHECKPOINT
#
# This preserves velocities and thermostat/random state for
# exact continuation to the next heating stage when possible.
# ============================================================

OUTPUT_CHECKPOINT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


checkpoint = (
    context.createCheckpoint()
)


if isinstance(
    checkpoint,
    str,
):

    checkpoint = (
        checkpoint.encode(
            "latin1"
        )
    )


OUTPUT_CHECKPOINT.write_bytes(
    checkpoint
)


# ============================================================
# 21. SAVE CSV LOG
# ============================================================

OUTPUT_LOG.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fieldnames = list(
    log_rows[0].keys()
)


with open(
    OUTPUT_LOG,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        log_rows
    )


# ============================================================
# 22. FINAL CLASSIFICATION
# ============================================================

temperature_ok = (

    35.0
    <=
    final_metrics[
        "temperature_K"
    ]
    <=
    65.0
)


one_sided_ok = (

    final_metrics[
        "water_below_graphene"
    ]
    ==
    0
)


if (
    temperature_ok
    and
    one_sided_ok
):

    status = (
        "HEATING_050K_PASS"
    )


else:

    status = (
        "HEATING_050K_COMPLETE_INSPECT_WARNINGS"
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

    "stage":
        "50 K",

    "ensemble":
        "NVT",

    "target_temperature_K":
        TARGET_TEMPERATURE_K,

    "friction_per_ps":
        FRICTION_PER_PS,

    "timestep_fs":
        TIMESTEP_FS,

    "total_time_ps":
        TOTAL_TIME_PS,

    "total_steps":
        TOTAL_STEPS,

    "n_particles":
        n_particles,

    "n_waters":
        n_waters,

    "n_constraints":
        system.getNumConstraints(),

    "degrees_of_freedom":
        degrees_of_freedom,

    "cuda_precision":
        cuda_precision,

    "langevin_seed":
        LANGEVIN_SEED,

    "velocity_seed":
        VELOCITY_SEED,

    "ions_added":
        False,

    "barostat_present":
        False,

    "ligand_restrained":
        False,

    "initial":
        initial_metrics,

    "final":
        final_metrics,

    "checks":
        {
            "temperature_ok":
                temperature_ok,

            "one_sided_water_ok":
                one_sided_ok,
        },

    "positions_file":
        str(
            OUTPUT_POSITIONS
        ),

    "velocities_file":
        str(
            OUTPUT_VELOCITIES
        ),

    "checkpoint_file":
        str(
            OUTPUT_CHECKPOINT
        ),

    "log_file":
        str(
            OUTPUT_LOG
        ),
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
print("=" * 78)
print("50 K HEATING SUMMARY")
print("=" * 78)

print()

print(
    f"Final temperature: "
    f"{final_metrics['temperature_K']:.3f} K"
)

print(
    f"Final potential energy: "
    f"{final_metrics['potential_kJ_per_mol']:.3f} kJ/mol"
)

print(
    f"Final kinetic energy: "
    f"{final_metrics['kinetic_kJ_per_mol']:.3f} kJ/mol"
)

print()
print(
    f"Graphene z range: "
    f"{final_metrics['graphene_z_min_nm']:.6f} "
    f"to "
    f"{final_metrics['graphene_z_max_nm']:.6f} nm"
)

print(
    f"Pyrene mean height above graphene: "
    f"{final_metrics['pyrene_height_above_graphene_nm']:.6f} nm"
)

print(
    f"Ligand z range: "
    f"{final_metrics['ligand_z_min_nm']:.6f} "
    f"to "
    f"{final_metrics['ligand_z_max_nm']:.6f} nm"
)

print()
print(
    f"Water oxygen z range: "
    f"{final_metrics['water_z_min_nm']:.6f} "
    f"to "
    f"{final_metrics['water_z_max_nm']:.6f} nm"
)

print(
    "Waters below graphene:",
    final_metrics[
        "water_below_graphene"
    ],
)

print()
print("=" * 78)
print("RESULT")
print("=" * 78)

print()
print(
    status
)


print()
print("=" * 78)
print("SAVED")
print("=" * 78)

print(
    "Positions:",
    OUTPUT_POSITIONS
)

print(
    "Velocities:",
    OUTPUT_VELOCITIES
)

print(
    "Checkpoint:",
    OUTPUT_CHECKPOINT
)

print(
    "Log:",
    OUTPUT_LOG
)

print(
    "Metadata:",
    OUTPUT_METADATA
)


print()
print("=" * 78)
print("50 K NVT HEATING: COMPLETE")
print("=" * 78)

print()
print(
    "This was heating/equilibration preparation, "
    "not production MD."
)

print(
    "No ions were added."
)

print(
    "Pyrene-PEG5 remained unrestrained."
)

print(
    "Do not start 100 K until this output is inspected."
)

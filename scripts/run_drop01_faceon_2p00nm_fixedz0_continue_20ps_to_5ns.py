from pathlib import Path
import csv
import json
import math

import numpy as np

from openmm import openmm, unit
from openmm.app import DCDFile, Topology, element


# ============================================================
# DROP-01 CONTINUATION - EXACT CHECKPOINT 20 ps -> 5 ns
#
# INPUT:
#   Drop-01 face-on pyrene-PEG5 starting 2.00 nm above graphene.
#   6032 OPC waters after initial-condition overlap pruning.
#   No prior velocities are continued.
#
# THIS STAGE:
#   Unbiased NVT Drop-01 sampling
#   300 K
#   20 ps safety segment of a predetermined 5 ns trajectory
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
#   - This is the first segment of the actual Drop-01 trajectory.
#   - No minimization, pulling, gravity, adsorption force, or ligand restraint.
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

TERMINAL_C36_LOCAL_INDEX = 36


# ============================================================
# 2. DROP-01 PARAMETERS
# ============================================================

TARGET_TEMPERATURE_K = 300.0

FRICTION_PER_PS = 1.0

TIMESTEP_FS = 1.0
TIMESTEP_PS = TIMESTEP_FS / 1000.0

TOTAL_TIME_PS = 4980.0

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

LANGEVIN_SEED = 20260828

VELOCITY_SEED = 20260827

# Actual VMD trajectory: one frame every 1 ps.
TRAJECTORY_INTERVAL_PS = 1.0

TRAJECTORY_STEPS = int(
    round(
        TRAJECTORY_INTERVAL_PS
        / TIMESTEP_PS
    )
)

if TRAJECTORY_STEPS % REPORT_STEPS != 0:
    raise RuntimeError(
        "Trajectory interval must be an integer multiple "
        "of the scalar report interval."
    )



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
    "drop01_faceon_2p00nm_fixedz0_wallfree_opc_yb.xml"
)

POSITIONS_FILE = Path(
    "parameters/combined/"
    "drop01_faceon_2p00nm_fixedz0_stageA_20ps_positions_nm.npy"
)

VELOCITIES_VALIDATION_FILE = Path(
    "parameters/combined/"
    "drop01_faceon_2p00nm_fixedz0_stageA_20ps_velocities_nm_per_ps.npy"
)

INPUT_CHECKPOINT = Path(
    "checkpoints/"
    "drop01_faceon_2p00nm_fixedz0_stageA_20ps.chk"
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
    "drop01_faceon_2p00nm_fixedz0_5ns_positions_nm.npy"
)

OUTPUT_VELOCITIES = Path(
    "parameters/combined/"
    "drop01_faceon_2p00nm_fixedz0_5ns_velocities_nm_per_ps.npy"
)

OUTPUT_CHECKPOINT = Path(
    "checkpoints/"
    "drop01_faceon_2p00nm_fixedz0_5ns.chk"
)

OUTPUT_LOG = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_20ps_to_5ns_log.csv"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_5ns.json"
)

OUTPUT_DCD = Path(
    "trajectories/"
    "drop01_faceon_2p00nm_fixedz0_graphene_ligand.dcd"
)

RESCUE_CHECKPOINT = Path(
    "checkpoints/"
    "drop01_faceon_2p00nm_fixedz0_20ps_to_5ns_rescue.chk"
)

RESCUE_INTERVAL_PS = 100.0

RESCUE_STEPS = int(
    round(
        RESCUE_INTERVAL_PS
        / TIMESTEP_PS
    )
)

TRAJECTORY_REFERENCE_PDB = Path(
    "visualization/"
    "drop01_faceon_2p00nm_initial.pdb"
)


# ============================================================
# 6. VERIFY INPUT FILES
# ============================================================

for path in [
    SYSTEM_XML,
    POSITIONS_FILE,
    VELOCITIES_VALIDATION_FILE,
    INPUT_CHECKPOINT,
    LIGAND_MOL2,
    TRAJECTORY_REFERENCE_PDB,
    OUTPUT_DCD,
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

validation_velocities_nm_per_ps = np.load(
    VELOCITIES_VALIDATION_FILE
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
# 10. CREATE 300 K EQUILIBRATION INTEGRATOR
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
# 12-13. LOAD EXACT DROP-01 STAGE-A CHECKPOINT
#
# CRITICAL:
#   - no setPositions()
#   - no velocity regeneration
#   - no setTime()
#   - no constraint projection after load
#
# The checkpoint defines the exact continuation state.
# ============================================================

checkpoint_bytes = INPUT_CHECKPOINT.read_bytes()

context.loadCheckpoint(
    checkpoint_bytes
)

checkpoint_state = context.getState(
    positions=True,
    velocities=True,
    energy=True,
    enforcePeriodicBox=False,
)

checkpoint_time_ps = float(
    checkpoint_state
    .getTime()
    .value_in_unit(
        unit.picoseconds
    )
)

if abs(
    checkpoint_time_ps
    -
    20.0
) > 1.0e-6:

    raise RuntimeError(
        "Checkpoint does not represent exactly 20 ps: "
        f"{checkpoint_time_ps:.12f} ps"
    )


checkpoint_positions_nm = np.asarray(
    checkpoint_state
    .getPositions(
        asNumpy=True
    )
    .value_in_unit(
        unit.nanometer
    ),
    dtype=float,
)

checkpoint_velocities_nm_per_ps = np.asarray(
    checkpoint_state
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


if checkpoint_positions_nm.shape != positions_nm.shape:

    raise RuntimeError(
        "Checkpoint/saved Stage-A position shapes differ."
    )

if (
    checkpoint_velocities_nm_per_ps.shape
    !=
    validation_velocities_nm_per_ps.shape
):

    raise RuntimeError(
        "Checkpoint/saved Stage-A velocity shapes differ."
    )


max_checkpoint_position_difference_nm = float(
    np.max(
        np.abs(
            checkpoint_positions_nm
            -
            positions_nm
        )
    )
)

max_checkpoint_velocity_difference_nm_per_ps = float(
    np.max(
        np.abs(
            checkpoint_velocities_nm_per_ps
            -
            validation_velocities_nm_per_ps
        )
    )
)


if max_checkpoint_position_difference_nm > 1.0e-6:

    raise RuntimeError(
        "Checkpoint positions do not reproduce Stage-A endpoint: "
        f"{max_checkpoint_position_difference_nm:.12e} nm"
    )

if max_checkpoint_velocity_difference_nm_per_ps > 1.0e-6:

    raise RuntimeError(
        "Checkpoint velocities do not reproduce Stage-A endpoint: "
        f"{max_checkpoint_velocity_difference_nm_per_ps:.12e} nm/ps"
    )


print()
print("=" * 78)
print("EXACT CHECKPOINT CONTINUATION")
print("=" * 78)
print(
    f"Checkpoint time: "
    f"{checkpoint_time_ps:.6f} ps"
)
print(
    "Max checkpoint-position difference:",
    f"{max_checkpoint_position_difference_nm:.3e} nm"
)
print(
    "Max checkpoint-velocity difference:",
    f"{max_checkpoint_velocity_difference_nm_per_ps:.3e} nm/ps"
)
print(
    "Velocity regeneration: NONE"
)
print(
    "Position reset: NONE"
)
print(
    "Time reset: NONE"
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


    pyrene_xyz = (
        ligand[
            aromatic_atoms,
            :
        ]
    )

    pyrene_center = (
        pyrene_xyz.mean(
            axis=0
        )
    )

    terminal_c36_xyz = (
        ligand[
            TERMINAL_C36_LOCAL_INDEX
        ]
    )

    terminal_c36_height = float(
        terminal_c36_xyz[
            2
        ]
        -
        graphene_z_mean
    )

    pyrene_to_terminal_c36 = float(
        np.linalg.norm(
            terminal_c36_xyz
            -
            pyrene_center
        )
    )

    ligand_vertical_span = float(
        ligand_z_max
        -
        ligand_z_min
    )


    pyrene_centered = (
        pyrene_xyz
        -
        pyrene_center
    )

    (
        _,
        _,
        pyrene_vh,
    ) = np.linalg.svd(
        pyrene_centered,
        full_matrices=False,
    )

    pyrene_normal = (
        pyrene_vh[
            -1,
            :
        ]
    )

    pyrene_normal_norm = float(
        np.linalg.norm(
            pyrene_normal
        )
    )

    if (
        not math.isfinite(
            pyrene_normal_norm
        )
        or
        pyrene_normal_norm
        <=
        1.0e-12
    ):
        raise RuntimeError(
            "Could not determine finite pyrene plane normal."
        )

    pyrene_normal = (
        pyrene_normal
        /
        pyrene_normal_norm
    )

    pyrene_cos_to_z = float(
        np.clip(
            abs(
                pyrene_normal[
                    2
                ]
            ),
            0.0,
            1.0,
        )
    )

    pyrene_tilt_deg = float(
        np.degrees(
            np.arccos(
                pyrene_cos_to_z
            )
        )
    )

    pyrene_plane_offsets = (
        pyrene_centered
        @
        pyrene_normal
    )

    pyrene_plane_rms_nm = float(
        np.sqrt(
            np.mean(
                pyrene_plane_offsets
                *
                pyrene_plane_offsets
            )
        )
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

        "pyrene_tilt_from_graphene_plane_deg":
            pyrene_tilt_deg,

        "pyrene_plane_rms_nm":
            pyrene_plane_rms_nm,

        "terminal_C36_height_above_graphene_nm":
            terminal_c36_height,

        "pyrene_center_to_terminal_C36_nm":
            pyrene_to_terminal_c36,

        "ligand_vertical_span_nm":
            ligand_vertical_span,

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
        "pyrene_tilt_from_graphene_plane_deg",
        "pyrene_plane_rms_nm",
        "terminal_C36_height_above_graphene_nm",
        "pyrene_center_to_terminal_C36_nm",
        "ligand_vertical_span_nm",
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
print("300 K NVT EQUILIBRATION")
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
print("INITIAL STATE ENTERING 300 K EQUILIBRATION")
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
    f"Pyrene tilt from graphene plane: "
    f"{initial_metrics['pyrene_tilt_from_graphene_plane_deg']:.3f} deg"
)

print(
    f"Pyrene plane RMS: "
    f"{initial_metrics['pyrene_plane_rms_nm']:.6f} nm"
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


print()
print("DROP-01 LINKER STATE AT CONTINUATION START")
print("-----------------------------")
print(
    f"Terminal C36 height: "
    f"{initial_metrics['terminal_C36_height_above_graphene_nm']:.6f} nm"
)
print(
    f"Pyrene center -> C36: "
    f"{initial_metrics['pyrene_center_to_terminal_C36_nm']:.6f} nm"
)
print(
    f"Ligand vertical span: "
    f"{initial_metrics['ligand_vertical_span_nm']:.6f} nm"
)


# ============================================================
# 17. CONTINUE DROP-01: 20 ps -> 5000 ps
# ============================================================

print()
print("=" * 78)
print("RUNNING DROP-01 CONTINUATION: 20 -> 5000 ps")
print("=" * 78)

print()
print(
    " time(ps)    T(K)         "
    "Potential(kJ/mol)   "
    "Pyrene-h(nm)   "
    "Tilt(deg)   "
    "Water-z(nm)          "
    "below-G"
)

print(
    "-" * 112
)


# ------------------------------------------------------------
# VMD trajectory subset:
#   graphene carbon cores 0..1249
#   ligand 3750..3819
#
# Waters and graphene pi particles remain fully simulated but
# are omitted from the visualization trajectory.
# ------------------------------------------------------------

trajectory_atom_indices = np.asarray(
    list(
        range(
            N_GRAPHENE_CARBONS
        )
    )
    +
    list(
        range(
            LIGAND_START,
            LIGAND_STOP,
        )
    ),
    dtype=int,
)

if trajectory_atom_indices.size != 1320:
    raise RuntimeError(
        "Expected exactly 1320 trajectory atoms."
    )


trajectory_topology = Topology()

trajectory_topology.setPeriodicBoxVectors(
    system.getDefaultPeriodicBoxVectors()
)

traj_chain = trajectory_topology.addChain()
traj_residue = trajectory_topology.addResidue(
    "DROP",
    traj_chain,
)

for _ in trajectory_atom_indices:

    trajectory_topology.addAtom(
        "X",
        element.carbon,
        traj_residue,
    )


# ------------------------------------------------------------
# Append to the existing scientific master trajectory.
#
# Stage A already contains:
#   t = 0, 1, ..., 20 ps  => 21 DCD models
#
# The first continuation frame must therefore be 21 ps.
# ------------------------------------------------------------

import struct

with open(
    OUTPUT_DCD,
    "rb",
) as f:

    header_magic = f.read(
        8
    )

    if (
        len(header_magic) != 8
        or
        header_magic[4:8] != b"CORD"
    ):

        raise RuntimeError(
            "Existing Drop-01 DCD has an invalid header."
        )

    existing_dcd_models = struct.unpack(
        "<i",
        f.read(4),
    )[0]


if existing_dcd_models != 21:

    raise RuntimeError(
        "Expected exactly 21 existing DCD models "
        "(0 through 20 ps), found "
        f"{existing_dcd_models}. "
        "Refusing to append."
    )


dcd_handle = open(
    OUTPUT_DCD,
    "r+b",
)

dcd = DCDFile(
    dcd_handle,
    trajectory_topology,
    TIMESTEP_FS
    * unit.femtoseconds,
    firstStep=0,
    interval=TRAJECTORY_STEPS,
    append=True,
)


print(
    "Existing master DCD models:",
    existing_dcd_models,
)
print(
    "First new trajectory frame will be: 21 ps"
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
    f"{initial_metrics['pyrene_tilt_from_graphene_plane_deg']:9.3f} "
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

    if (
        steps_completed
        %
        TRAJECTORY_STEPS
    ) == 0:

        dcd.writeModel(
            unit.Quantity(
                current_positions[
                    trajectory_atom_indices
                ],
                unit.nanometer,
            ),
            periodicBoxVectors=(
                state
                .getPeriodicBoxVectors()
            ),
        )


    if (
        steps_completed
        %
        RESCUE_STEPS
    ) == 0:

        RESCUE_CHECKPOINT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rescue_bytes = (
            context.createCheckpoint()
        )

        if isinstance(
            rescue_bytes,
            str,
        ):
            rescue_bytes = rescue_bytes.encode(
                "latin1"
            )

        rescue_tmp = (
            RESCUE_CHECKPOINT
            .with_suffix(
                ".tmp"
            )
        )

        rescue_tmp.write_bytes(
            rescue_bytes
        )

        rescue_tmp.replace(
            RESCUE_CHECKPOINT
        )


    print(
        f"{metrics['time_ps']:9.3f} "
        f"{metrics['temperature_K']:8.3f} "
        f"{metrics['potential_kJ_per_mol']:18.3f} "
        f"{metrics['pyrene_height_above_graphene_nm']:13.6f} "
        f"{metrics['pyrene_tilt_from_graphene_plane_deg']:9.3f} "
        f"{metrics['water_z_min_nm']:7.4f}-"
        f"{metrics['water_z_max_nm']:7.4f} "
        f"{metrics['water_below_graphene']:8d}"
    )


dcd_handle.flush()
dcd_handle.close()


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

    285.0
    <=
    final_metrics[
        "temperature_K"
    ]
    <=
    315.0
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
        "DROP01_5NS_CONTINUATION_PASS"
    )


else:

    status = (
        "DROP01_5NS_CONTINUATION_INSPECT_WARNINGS"
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
        "Drop-01 unbiased face-on 2.00 nm, exact checkpoint continuation 20-5000 ps",

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

    "fresh_velocities_generated_at_t0":
        True,

    "minimization_performed":
        False,

    "pulling_force_present":
        False,

    "gravity_present":
        False,

    "trajectory_interval_ps":
        TRAJECTORY_INTERVAL_PS,

    "trajectory_atom_count":
        int(
            trajectory_atom_indices.size
        ),

    "trajectory_file":
        str(
            OUTPUT_DCD
        ),

    "trajectory_reference_pdb":
        str(
            TRAJECTORY_REFERENCE_PDB
        ),

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
print("DROP-01 5 ns CONTINUATION SUMMARY")
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
    f"Pyrene tilt from graphene plane: "
    f"{final_metrics['pyrene_tilt_from_graphene_plane_deg']:.3f} deg"
)

print(
    f"Pyrene plane RMS: "
    f"{final_metrics['pyrene_plane_rms_nm']:.6f} nm"
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
print("DROP-01 5 ns CONTINUATION: COMPLETE")
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
    "Do not continue beyond 20 ps until this Stage A output is inspected."
)

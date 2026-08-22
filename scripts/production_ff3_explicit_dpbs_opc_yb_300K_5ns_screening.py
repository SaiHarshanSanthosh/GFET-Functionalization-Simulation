from pathlib import Path
import csv
import json
import math

import numpy as np

from openmm import openmm, unit


# ============================================================
# 5 ns 300 K NVT PRODUCTION MD
#
# INPUT:
#   Corrected FF-3 minimized wall-free/YB system.
#
# THIS STAGE:
#   NVT
#   300 K
#   5000 ps production
#   1 fs timestep
#   LangevinMiddleIntegrator
#
# IMPORTANT:
#   - Graphene support remains active.
#   - Dynamic Yeh-Berkowitz slab correction remains active.
#   - No runtime water wall is permitted.
#   - Pyrene-PEG5 remains completely unrestrained.
#   - Na+ and Cl- are inherited from the validated assembly.
#   - No ions are added during heating.
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

TARGET_TEMPERATURE_K = 300.0

FRICTION_PER_PS = 1.0

TIMESTEP_FS = 1.0
TIMESTEP_PS = TIMESTEP_FS / 1000.0

TOTAL_TIME_PS = 5000.0

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

LANGEVIN_SEED = 20260827


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
    "ff3_explicit_dpbs_opc_yb.xml"
)

POSITIONS_FILE = Path(
    "parameters/combined/"
    "ff3_explicit_dpbs_opc_yb_"
    "equilibrated_300K_continue500ps_positions_nm.npy"
)

VELOCITIES_FILE = Path(
    "parameters/combined/"
    "ff3_explicit_dpbs_opc_yb_"
    "equilibrated_300K_continue500ps_velocities_nm_per_ps.npy"
)

LIGAND_MOL2 = Path(
    "structures/"
    "pyrene_peg5_propargyl.mol2"
)

ASSEMBLY_METADATA = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_assembly_check.json"
)

PRIOR_STAGE_METADATA = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_equilibration_300K_continue500ps.json"
)

INPUT_CHECKPOINT = Path(
    "checkpoints/"
    "ff3_explicit_dpbs_opc_yb_equilibration_300K_continue500ps.chk"
)


# ============================================================
# 5. OUTPUT FILES
# ============================================================

OUTPUT_POSITIONS = Path(
    "parameters/combined/"
    "ff3_explicit_dpbs_opc_yb_"
    "production_300K_5ns_positions_nm.npy"
)

OUTPUT_VELOCITIES = Path(
    "parameters/combined/"
    "ff3_explicit_dpbs_opc_yb_"
    "production_300K_5ns_velocities_nm_per_ps.npy"
)

OUTPUT_CHECKPOINT = Path(
    "checkpoints/"
    "ff3_explicit_dpbs_opc_yb_production_300K_5ns.chk"
)

OUTPUT_LOG = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_production_300K_5ns_log.csv"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_production_300K_5ns.json"
)

SCREENING_TRAJECTORY = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_production_300K_5ns_"
    "screening_trajectory_float32.npy"
)

SCREENING_METADATA = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_production_300K_5ns_"
    "screening_trajectory.json"
)

RESCUE_CHECKPOINT = Path(
    "checkpoints/"
    "ff3_explicit_dpbs_opc_yb_production_300K_5ns_rescue.chk"
)

RESCUE_PROGRESS = Path(
    "analysis/"
    "ff3_explicit_dpbs_opc_yb_production_300K_5ns_"
    "rescue_progress.json"
)

RESCUE_INTERVAL_PS = 100.0

RESCUE_STEPS = int(
    round(
        RESCUE_INTERVAL_PS
        / TIMESTEP_PS
    )
)

if RESCUE_STEPS % REPORT_STEPS != 0:
    raise RuntimeError(
        "Rescue checkpoint interval must be divisible "
        "by the reporting interval."
    )


# ============================================================
# 6. REFUSE TO OVERWRITE EXISTING OUTPUTS
# ============================================================

for output_path in [
    OUTPUT_POSITIONS,
    OUTPUT_VELOCITIES,
    OUTPUT_CHECKPOINT,
    OUTPUT_LOG,
    OUTPUT_METADATA,
    SCREENING_TRAJECTORY,
    SCREENING_METADATA,
    RESCUE_CHECKPOINT,
    RESCUE_PROGRESS,
]:

    if output_path.exists():

        raise RuntimeError(
            "Refusing to overwrite existing output: "
            f"{output_path}"
        )


# ============================================================
# 7. VERIFY INPUT FILES
# ============================================================

for path in [
    SYSTEM_XML,
    POSITIONS_FILE,
    VELOCITIES_FILE,
    LIGAND_MOL2,
    ASSEMBLY_METADATA,
    PRIOR_STAGE_METADATA,
    INPUT_CHECKPOINT,
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

velocities_nm_per_ps = np.load(
    VELOCITIES_FILE
)


with open(
    ASSEMBLY_METADATA,
    "r",
) as f:

    assembly_metadata = json.load(f)


with open(
    PRIOR_STAGE_METADATA,
    "r",
) as f:

    prior_stage_metadata = json.load(f)


if prior_stage_metadata.get(
    "status"
) != "EQUILIBRATION_300K_CONTINUE500PS_PASS":

    raise RuntimeError(
        "300 K continuation requires a successful "
        "EQUILIBRATION_300K_CONTINUE500PS_PASS input stage. "
        f"Found: {prior_stage_metadata.get('status')}"
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


metadata_graphene_particles = int(
    assembly_metadata["graphene_particles"]
)

metadata_ligand_particles = int(
    assembly_metadata["pyrene_peg5_particles"]
)

n_waters = int(
    assembly_metadata["opc_waters"]
)

metadata_sites_per_water = int(
    assembly_metadata["opc_sites_per_water"]
)

n_na = int(assembly_metadata["na_ions"])
n_k = int(assembly_metadata["k_ions"])
n_cl = int(assembly_metadata["cl_ions"])
n_h2po4 = int(assembly_metadata["h2po4_ions"])
n_hpo4 = int(assembly_metadata["hpo4_ions"])

n_ions = int(
    assembly_metadata["total_ions"]
)

na_start_index = int(
    assembly_metadata["na_particle_start_index"]
)
na_end_index = int(
    assembly_metadata["na_particle_end_index_exclusive"]
)

k_start_index = int(
    assembly_metadata["k_particle_start_index"]
)
k_end_index = int(
    assembly_metadata["k_particle_end_index_exclusive"]
)

cl_start_index = int(
    assembly_metadata["cl_particle_start_index"]
)
cl_end_index = int(
    assembly_metadata["cl_particle_end_index_exclusive"]
)

h2po4_start_index = int(
    assembly_metadata["h2po4_particle_start_index"]
)
h2po4_end_index = int(
    assembly_metadata["h2po4_particle_end_index_exclusive"]
)

hpo4_start_index = int(
    assembly_metadata["hpo4_particle_start_index"]
)
hpo4_end_index = int(
    assembly_metadata["hpo4_particle_end_index_exclusive"]
)

metadata_total_particles = int(
    assembly_metadata["total_particles"]
)

if metadata_graphene_particles != N_GRAPHENE_TOTAL:
    raise RuntimeError(
        "Graphene particle count disagrees with assembly metadata."
    )

if metadata_ligand_particles != N_LIGAND:
    raise RuntimeError(
        "Ligand particle count disagrees with assembly metadata."
    )

if metadata_sites_per_water != SITES_PER_WATER:
    raise RuntimeError(
        "OPC sites-per-water disagrees with assembly metadata."
    )

if n_ions != (
    n_na + n_k + n_cl + n_h2po4 + n_hpo4
):
    raise RuntimeError(
        "Explicit-DPBS ion-count metadata is internally inconsistent."
    )

water_start_index = N_SOLUTE

water_sites = (
    n_waters
    * metadata_sites_per_water
)

water_end_index = (
    water_start_index
    + water_sites
)

if water_end_index != na_start_index:
    raise RuntimeError(
        "Water/Na+ particle boundary mismatch."
    )

if na_end_index - na_start_index != n_na:
    raise RuntimeError(
        "Na+ particle range/count mismatch."
    )

if na_end_index != k_start_index:
    raise RuntimeError(
        "Na+/K+ particle boundary mismatch."
    )

if k_end_index - k_start_index != n_k:
    raise RuntimeError(
        "K+ particle range/count mismatch."
    )

if k_end_index != cl_start_index:
    raise RuntimeError(
        "K+/Cl- particle boundary mismatch."
    )

if cl_end_index - cl_start_index != n_cl:
    raise RuntimeError(
        "Cl- particle range/count mismatch."
    )

if cl_end_index != h2po4_start_index:
    raise RuntimeError(
        "Cl-/H2PO4- particle boundary mismatch."
    )

if (
    h2po4_end_index
    - h2po4_start_index
    != 7 * n_h2po4
):
    raise RuntimeError(
        "H2PO4- particle range/count mismatch."
    )

if h2po4_end_index != hpo4_start_index:
    raise RuntimeError(
        "H2PO4-/HPO4^2- particle boundary mismatch."
    )

if (
    hpo4_end_index
    - hpo4_start_index
    != 6 * n_hpo4
):
    raise RuntimeError(
        "HPO4^2- particle range/count mismatch."
    )

if hpo4_end_index != metadata_total_particles:
    raise RuntimeError(
        "Final explicit-DPBS particle endpoint disagrees with metadata."
    )

if n_particles != metadata_total_particles:
    raise RuntimeError(
        "System particle count disagrees with assembly metadata. "
        f"System={n_particles}, "
        f"metadata={metadata_total_particles}."
    )

if not bool(
    assembly_metadata["ions_added"]
):
    raise RuntimeError(
        "Assembly metadata says ions were not added."
    )


if not bool(
    assembly_metadata[
        "yeh_berkowitz_enabled"
    ]
):

    raise RuntimeError(
        "Assembly metadata says Yeh-Berkowitz "
        "is not enabled."
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
# 10. CREATE 300 K LANGEVIN INTEGRATOR
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
# 12. LOAD EXACT EQUILIBRATION CHECKPOINT
#
# Scientific cumulative equilibration before production:
#     200 + 300 + 500 ps = 1000 ps
#
# The final stage itself reset the OpenMM clock before running
# 500 ps, so the checkpoint clock may be ~500 ps. We preserve
# whatever exact time exists in the checkpoint.
# ============================================================

checkpoint_bytes = INPUT_CHECKPOINT.read_bytes()

context.loadCheckpoint(
    checkpoint_bytes
)


# ============================================================
# 13. VERIFY EXACT CHECKPOINT CONTINUITY
#
# Saved NPY arrays are comparison references ONLY.
# They never initialize or alter the production Context.
# ============================================================

checkpoint_state = context.getState(
    positions=True,
    velocities=True,
    energy=True,
    enforcePeriodicBox=False,
)

checkpoint_positions_nm = np.asarray(
    checkpoint_state
    .getPositions(asNumpy=True)
    .value_in_unit(unit.nanometer),
    dtype=float,
)

checkpoint_velocities_nm_per_ps = np.asarray(
    checkpoint_state
    .getVelocities(asNumpy=True)
    .value_in_unit(
        unit.nanometer
        / unit.picosecond
    ),
    dtype=float,
)

checkpoint_time_ps = float(
    checkpoint_state
    .getTime()
    .value_in_unit(
        unit.picoseconds
    )
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
            velocities_nm_per_ps
        )
    )
)

if (
    max_checkpoint_position_difference_nm
    >
    1.0e-6
):
    raise RuntimeError(
        "Checkpoint position mismatch: "
        f"{max_checkpoint_position_difference_nm:.9e} nm"
    )

if (
    max_checkpoint_velocity_difference_nm_per_ps
    >
    1.0e-6
):
    raise RuntimeError(
        "Checkpoint velocity mismatch: "
        f"{max_checkpoint_velocity_difference_nm_per_ps:.9e} nm/ps"
    )

if not np.isfinite(
    checkpoint_time_ps
):
    raise RuntimeError(
        "Checkpoint time is non-finite."
    )

print()
print("=" * 78)
print("EXACT EQUILIBRATION CHECKPOINT")
print("=" * 78)

print(
    f"Checkpoint time: "
    f"{checkpoint_time_ps:.12f} ps"
)

print(
    "Max position difference: "
    f"{max_checkpoint_position_difference_nm:.3e} nm"
)

print(
    "Max velocity difference: "
    f"{max_checkpoint_velocity_difference_nm_per_ps:.3e} nm/ps"
)

print("Position reset: NONE")
print("Velocity reset/regeneration: NONE")
print("Time reset: NONE")
print("Constraint reprojection after checkpoint load: NONE")
print("Minimization: NONE")
print()

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
        water_start_index
        +
        metadata_sites_per_water
        * np.arange(n_waters)
    )

    oxygen_z = positions[
        oxygen_indices,
        2
    ]

    na_z = positions[
        na_start_index:na_end_index,
        2
    ]

    k_z = positions[
        k_start_index:k_end_index,
        2
    ]

    cl_z = positions[
        cl_start_index:cl_end_index,
        2
    ]

    h2po4_z = positions[
        h2po4_start_index:h2po4_end_index,
        2
    ]

    hpo4_z = positions[
        hpo4_start_index:hpo4_end_index,
        2
    ]

    water_z_min = float(oxygen_z.min())
    water_z_max = float(oxygen_z.max())

    na_z_min = float(na_z.min())
    na_z_max = float(na_z.max())

    k_z_min = float(k_z.min())
    k_z_max = float(k_z.max())

    cl_z_min = float(cl_z.min())
    cl_z_max = float(cl_z.max())

    h2po4_z_min = float(h2po4_z.min())
    h2po4_z_max = float(h2po4_z.max())

    hpo4_z_min = float(hpo4_z.min())
    hpo4_z_max = float(hpo4_z.max())

    water_below_graphene = int(
        np.sum(oxygen_z < graphene_z_max)
    )

    na_below_graphene = int(
        np.sum(na_z < graphene_z_max)
    )

    k_below_graphene = int(
        np.sum(k_z < graphene_z_max)
    )

    cl_below_graphene = int(
        np.sum(cl_z < graphene_z_max)
    )

    h2po4_below_graphene = int(
        np.sum(h2po4_z < graphene_z_max)
    )

    hpo4_below_graphene = int(
        np.sum(hpo4_z < graphene_z_max)
    )

    lowest_water_gap_nm = float(
        water_z_min - graphene_z_max
    )

    lowest_na_gap_nm = float(
        na_z_min - graphene_z_max
    )

    lowest_k_gap_nm = float(
        k_z_min - graphene_z_max
    )

    lowest_cl_gap_nm = float(
        cl_z_min - graphene_z_max
    )

    lowest_h2po4_gap_nm = float(
        h2po4_z_min - graphene_z_max
    )

    lowest_hpo4_gap_nm = float(
        hpo4_z_min - graphene_z_max
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

        "na_z_min_nm":
            na_z_min,

        "na_z_max_nm":
            na_z_max,

        "k_z_min_nm":
            k_z_min,

        "k_z_max_nm":
            k_z_max,

        "cl_z_min_nm":
            cl_z_min,

        "cl_z_max_nm":
            cl_z_max,

        "h2po4_z_min_nm":
            h2po4_z_min,

        "h2po4_z_max_nm":
            h2po4_z_max,

        "hpo4_z_min_nm":
            hpo4_z_min,

        "hpo4_z_max_nm":
            hpo4_z_max,

        "na_below_graphene":
            na_below_graphene,

        "k_below_graphene":
            k_below_graphene,

        "cl_below_graphene":
            cl_below_graphene,

        "h2po4_atoms_below_graphene":
            h2po4_below_graphene,

        "hpo4_atoms_below_graphene":
            hpo4_below_graphene,

        "lowest_water_gap_nm":
            lowest_water_gap_nm,

        "lowest_na_gap_nm":
            lowest_na_gap_nm,

        "lowest_k_gap_nm":
            lowest_k_gap_nm,

        "lowest_cl_gap_nm":
            lowest_cl_gap_nm,

        "lowest_h2po4_gap_nm":
            lowest_h2po4_gap_nm,

        "lowest_hpo4_gap_nm":
            lowest_hpo4_gap_nm,
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
        "na_z_min_nm",
        "na_z_max_nm",
        "k_z_min_nm",
        "k_z_max_nm",
        "cl_z_min_nm",
        "cl_z_max_nm",
        "h2po4_z_min_nm",
        "h2po4_z_max_nm",
        "hpo4_z_min_nm",
        "hpo4_z_max_nm",
        "lowest_water_gap_nm",
        "lowest_na_gap_nm",
        "lowest_k_gap_nm",
        "lowest_cl_gap_nm",
        "lowest_h2po4_gap_nm",
        "lowest_hpo4_gap_nm",
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
# 14B. COMPACT SCREENING-TRAJECTORY LAYOUT
# ============================================================

SCREENING_N_FRAMES = (
    TOTAL_STEPS
    // REPORT_STEPS
    + 1
)

if TOTAL_STEPS % REPORT_STEPS != 0:
    raise RuntimeError(
        "Production duration must be exactly divisible "
        "by report interval."
    )


SCREENING_LAYOUT = {}

screening_cursor = 0


def reserve_screening_block(
    name,
    width,
    shape,
    description,
):
    global screening_cursor

    start = screening_cursor
    stop = start + int(width)

    SCREENING_LAYOUT[name] = {
        "start_column_inclusive":
            start,
        "end_column_exclusive":
            stop,
        "shape":
            list(shape),
        "description":
            description,
    }

    screening_cursor = stop


reserve_screening_block(
    "production_elapsed_ps",
    1,
    [1],
    "Elapsed time since beginning of production MD.",
)

reserve_screening_block(
    "openmm_context_time_ps",
    1,
    [1],
    "Exact OpenMM Context clock preserved from checkpoint.",
)

reserve_screening_block(
    "graphene_z_summary_nm",
    3,
    [3],
    "Graphene carbon mean, minimum, maximum z.",
)

reserve_screening_block(
    "ligand_xyz_nm",
    N_LIGAND * 3,
    [N_LIGAND, 3],
    "Pyrene-PEG5 particle Cartesian coordinates.",
)

reserve_screening_block(
    "na_xyz_nm",
    n_na * 3,
    [n_na, 3],
    "Na+ Cartesian coordinates.",
)

reserve_screening_block(
    "k_xyz_nm",
    n_k * 3,
    [n_k, 3],
    "K+ Cartesian coordinates.",
)

reserve_screening_block(
    "cl_xyz_nm",
    n_cl * 3,
    [n_cl, 3],
    "Cl- Cartesian coordinates.",
)

reserve_screening_block(
    "h2po4_xyz_nm",
    (
        h2po4_end_index
        -
        h2po4_start_index
    ) * 3,
    [
        h2po4_end_index
        -
        h2po4_start_index,
        3,
    ],
    "All H2PO4- particle coordinates.",
)

reserve_screening_block(
    "hpo4_xyz_nm",
    (
        hpo4_end_index
        -
        hpo4_start_index
    ) * 3,
    [
        hpo4_end_index
        -
        hpo4_start_index,
        3,
    ],
    "All HPO4^2- particle coordinates.",
)


SCREENING_WIDTH = screening_cursor


def pack_screening_frame(
    positions,
    production_elapsed_ps,
    context_time_ps,
):

    graphene_z = positions[
        :N_GRAPHENE_CARBONS,
        2,
    ]

    pieces = [
        np.asarray(
            [
                production_elapsed_ps,
                context_time_ps,
                float(np.mean(graphene_z)),
                float(np.min(graphene_z)),
                float(np.max(graphene_z)),
            ],
            dtype=np.float32,
        ),

        np.asarray(
            positions[
                LIGAND_START:
                LIGAND_STOP
            ],
            dtype=np.float32,
        ).reshape(-1),

        np.asarray(
            positions[
                na_start_index:
                na_end_index
            ],
            dtype=np.float32,
        ).reshape(-1),

        np.asarray(
            positions[
                k_start_index:
                k_end_index
            ],
            dtype=np.float32,
        ).reshape(-1),

        np.asarray(
            positions[
                cl_start_index:
                cl_end_index
            ],
            dtype=np.float32,
        ).reshape(-1),

        np.asarray(
            positions[
                h2po4_start_index:
                h2po4_end_index
            ],
            dtype=np.float32,
        ).reshape(-1),

        np.asarray(
            positions[
                hpo4_start_index:
                hpo4_end_index
            ],
            dtype=np.float32,
        ).reshape(-1),
    ]

    row = np.concatenate(
        pieces
    )

    if row.shape != (
        SCREENING_WIDTH,
    ):
        raise RuntimeError(
            "Screening trajectory row width mismatch: "
            f"got {row.shape}, "
            f"expected {(SCREENING_WIDTH,)}"
        )

    if not np.all(
        np.isfinite(row)
    ):
        raise RuntimeError(
            "Non-finite coordinate detected in "
            "screening trajectory frame."
        )

    return row



# ============================================================
# 15. HEADER
# ============================================================

print()
print("=" * 78)
print("5 ns 300 K NVT PRODUCTION MD")
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
print("PRODUCTION PROTOCOL")
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
# 16. INITIAL STATE FROM EXACT EQUILIBRATION CHECKPOINT
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
print("INITIAL STATE ENTERING 5 ns PRODUCTION")
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
# 16B. INITIALIZE COMPACT SCREENING TRAJECTORY
#
# This is a disk-backed NumPy array:
#
#   shape = (5001, SCREENING_WIDTH)
#   dtype = float32
#
# It is written one row at a time and flushed every 100 ps.
# ============================================================

SCREENING_TRAJECTORY.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RESCUE_CHECKPOINT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RESCUE_PROGRESS.parent.mkdir(
    parents=True,
    exist_ok=True,
)

screening_trajectory = (
    np.lib.format.open_memmap(
        SCREENING_TRAJECTORY,
        mode="w+",
        dtype=np.float32,
        shape=(
            SCREENING_N_FRAMES,
            SCREENING_WIDTH,
        ),
    )
)

screening_frames_written = 0


def write_screening_frame(
    positions,
    production_elapsed_ps,
    context_time_ps,
):
    global screening_frames_written

    if (
        screening_frames_written
        >=
        SCREENING_N_FRAMES
    ):
        raise RuntimeError(
            "Attempted to write too many "
            "screening frames."
        )

    screening_trajectory[
        screening_frames_written,
        :
    ] = pack_screening_frame(
        positions,
        production_elapsed_ps,
        context_time_ps,
    )

    screening_frames_written += 1


def atomic_checkpoint(
    output_path,
):

    checkpoint_data = (
        context.createCheckpoint()
    )

    if isinstance(
        checkpoint_data,
        str,
    ):
        checkpoint_data = (
            checkpoint_data.encode(
                "latin1"
            )
        )

    temporary_path = Path(
        str(output_path)
        +
        ".tmp"
    )

    temporary_path.write_bytes(
        checkpoint_data
    )

    temporary_path.replace(
        output_path
    )


def write_rescue_progress(
    production_elapsed_ps,
    context_time_ps,
):

    progress = {
        "status":
            "PRODUCTION_RESCUE_CHECKPOINT",

        "production_elapsed_ps":
            float(
                production_elapsed_ps
            ),

        "openmm_context_time_ps":
            float(
                context_time_ps
            ),

        "frames_written":
            int(
                screening_frames_written
            ),

        "screening_trajectory":
            str(
                SCREENING_TRAJECTORY
            ),

        "rescue_checkpoint":
            str(
                RESCUE_CHECKPOINT
            ),
    }

    temporary_path = Path(
        str(
            RESCUE_PROGRESS
        )
        +
        ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            progress,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_path.replace(
        RESCUE_PROGRESS
    )


write_screening_frame(
    current_positions,
    0.0,
    initial_metrics[
        "time_ps"
    ],
)

print()
print(
    "Compact screening trajectory:"
)

print(
    f"  frames: {SCREENING_N_FRAMES}"
)

print(
    f"  columns/frame: {SCREENING_WIDTH}"
)

print(
    "  dtype: float32"
)

print(
    f"  output: {SCREENING_TRAJECTORY}"
)

print(
    f"  rescue interval: "
    f"{RESCUE_INTERVAL_PS:.1f} ps"
)

print()


# ============================================================
# 17. RUN 5 ns PRODUCTION MD AT 300 K
# ============================================================

print()
print("=" * 78)
print("RUNNING 300 K NVT PRODUCTION MD")
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

    production_elapsed_ps = (
        steps_completed
        *
        TIMESTEP_PS
    )

    write_screening_frame(
        current_positions,
        production_elapsed_ps,
        metrics["time_ps"],
    )

    if (
        steps_completed
        %
        RESCUE_STEPS
        ==
        0
    ):

        screening_trajectory.flush()

        atomic_checkpoint(
            RESCUE_CHECKPOINT
        )

        write_rescue_progress(
            production_elapsed_ps,
            metrics["time_ps"],
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
# 17B. FINALIZE COMPACT SCREENING TRAJECTORY
# ============================================================

screening_trajectory.flush()

if (
    screening_frames_written
    !=
    SCREENING_N_FRAMES
):
    raise RuntimeError(
        "Incorrect number of screening frames: "
        f"wrote {screening_frames_written}, "
        f"expected {SCREENING_N_FRAMES}"
    )


screening_metadata = {
    "format":
        "NumPy .npy matrix",

    "dtype":
        "float32",

    "units":
        {
            "time":
                "ps",
            "coordinates":
                "nm",
        },

    "source_checkpoint":
        str(
            INPUT_CHECKPOINT
        ),

    "scientific_equilibration_before_production_ps":
        1000.0,

    "source_checkpoint_openmm_time_ps":
        checkpoint_time_ps,

    "production_duration_ps":
        TOTAL_TIME_PS,

    "frame_interval_ps":
        REPORT_INTERVAL_PS,

    "n_frames":
        SCREENING_N_FRAMES,

    "n_columns":
        SCREENING_WIDTH,

    "matrix_shape":
        [
            SCREENING_N_FRAMES,
            SCREENING_WIDTH,
        ],

    "layout":
        SCREENING_LAYOUT,

    "species_counts":
        {
            "na":
                n_na,
            "k":
                n_k,
            "cl":
                n_cl,
            "h2po4_molecules":
                n_h2po4,
            "hpo4_molecules":
                n_hpo4,
        },

    "box_vectors_nm":
        box_nm.tolist(),

    "phosphate_note":
        (
            "All phosphate particle coordinates are retained. "
            "The phosphorus reference particle will be "
            "identified from molecular topology during "
            "post-processing rather than assumed here."
        ),

    "trajectory_file":
        str(
            SCREENING_TRAJECTORY
        ),

    "rescue_checkpoint_file":
        str(
            RESCUE_CHECKPOINT
        ),

    "rescue_progress_file":
        str(
            RESCUE_PROGRESS
        ),
}


SCREENING_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)

SCREENING_METADATA.write_text(
    json.dumps(
        screening_metadata,
        indent=2,
    ),
    encoding="utf-8",
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
# future exact continuation of the production trajectory.
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


one_sided_water_ok = (
    final_metrics["water_below_graphene"] == 0
)

one_sided_na_ok = (
    final_metrics["na_below_graphene"] == 0
)

one_sided_k_ok = (
    final_metrics["k_below_graphene"] == 0
)

one_sided_cl_ok = (
    final_metrics["cl_below_graphene"] == 0
)

one_sided_h2po4_ok = (
    final_metrics["h2po4_atoms_below_graphene"] == 0
)

one_sided_hpo4_ok = (
    final_metrics["hpo4_atoms_below_graphene"] == 0
)

one_sided_ok = (
    one_sided_water_ok
    and one_sided_na_ok
    and one_sided_k_ok
    and one_sided_cl_ok
    and one_sided_h2po4_ok
    and one_sided_hpo4_ok
)

if (
    temperature_ok
    and one_sided_ok
):
    status = "PRODUCTION_300K_5NS_PASS"
else:
    status = "PRODUCTION_300K_5NS_COMPLETE_INSPECT_WARNINGS"


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
        "5 ns 300 K NVT production",

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

    "n_na":
        n_na,

    "n_k":
        n_k,

    "n_cl":
        n_cl,

    "n_h2po4":
        n_h2po4,

    "n_hpo4":
        n_hpo4,

    "n_ions":
        n_ions,

    "n_constraints":
        system.getNumConstraints(),

    "degrees_of_freedom":
        degrees_of_freedom,

    "cuda_precision":
        cuda_precision,

    "langevin_seed":
        LANGEVIN_SEED,

    "ions_present":
        True,

    "ions_added_during_heating":
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
                one_sided_water_ok,

            "one_sided_na_ok":
                one_sided_na_ok,

            "one_sided_k_ok":
                one_sided_k_ok,

            "one_sided_cl_ok":
                one_sided_cl_ok,

            "one_sided_h2po4_ok":
                one_sided_h2po4_ok,

            "one_sided_hpo4_ok":
                one_sided_hpo4_ok,

            "one_sided_aqueous_species_ok":
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

    "screening_trajectory_file":
        str(
            SCREENING_TRAJECTORY
        ),

    "screening_metadata_file":
        str(
            SCREENING_METADATA
        ),

    "rescue_checkpoint_file":
        str(
            RESCUE_CHECKPOINT
        ),

    "rescue_progress_file":
        str(
            RESCUE_PROGRESS
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
print("5 ns 300 K NVT PRODUCTION SUMMARY")
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
print("5 ns 300 K NVT PRODUCTION: COMPLETE")
print("=" * 78)

print()
print(
    "This was unbiased 5 ns production MD following "
    "1 ns cumulative equilibration."
)

print(
    "No ions were added."
)

print(
    "Pyrene-PEG5 remained unrestrained."
)

print(
    "Do not start production MD until this output is inspected."
)

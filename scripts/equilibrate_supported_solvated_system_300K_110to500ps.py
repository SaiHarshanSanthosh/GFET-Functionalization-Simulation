from pathlib import Path
import csv
import json
import math

import numpy as np

from openmm import app, openmm, unit


# ============================================================
# 300 K NVT EQUILIBRATION - 110 TO 500 ps
#
# INPUT:
#   Completed 300 K heating stage.
#
# PROTOCOL:
#   NVT
#   300 K
#   100 ps
#   1 fs timestep
#   LangevinMiddleIntegrator
#
# IMPORTANT:
#   - Continue positions and velocities from 300 K heating.
#   - Do NOT randomize velocities.
#   - Graphene support remains ON.
#   - Water walls remain ON.
#   - Pyrene-PEG5 remains completely unrestrained.
#   - No ions.
#   - No barostat.
#   - This is equilibration, NOT production MD.
# ============================================================


# ============================================================
# 1. SYSTEM INDEXING
# ============================================================

N_GRAPHENE_CARBONS = 1250
N_GRAPHENE_TOTAL = 3750

N_LIGAND = 70

N_SOLUTE = 3820

N_WATERS = 6101
SITES_PER_WATER = 4

LIGAND_START = N_GRAPHENE_TOTAL
LIGAND_STOP = N_SOLUTE


# ============================================================
# 2. SIMULATION PARAMETERS
# ============================================================

TARGET_TEMPERATURE_K = 300.0

FRICTION_PER_PS = 1.0

TIMESTEP_FS = 1.0
TIMESTEP_PS = TIMESTEP_FS / 1000.0

TOTAL_TIME_PS = 390.0

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


RECOVERY_INTERVAL_PS = 5.0

RECOVERY_STEPS = int(
    round(
        RECOVERY_INTERVAL_PS
        / TIMESTEP_PS
    )
)

RECOVERY_CHECKPOINT = Path(
    "checkpoints/"
    "pyrene_peg5_graphene_RECOVERY_300K_110to500ps.chk"
)

CONSTRAINT_TOLERANCE = 1.0e-6

EQUILIBRATION_SEED = 20260821


# ============================================================
# 3. WALL POSITIONS
# ============================================================

LOWER_WATER_WALL_NM = 0.100
UPPER_WATER_WALL_NM = 6.800


# ============================================================
# 4. INPUT FILES
# ============================================================

SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_supported_pme.xml"
)

POSITIONS_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "diagnostic_300K_110ps_positions_nm.npy"
)

VELOCITIES_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "diagnostic_300K_110ps_velocities_nm_per_ps.npy"
)

LIGAND_MOL2 = Path(
    "structures/"
    "pyrene_peg5_propargyl.mol2"
)

VMD_TOPOLOGY_PDB = Path(
    "structures/"
    "pyrene_peg5_graphene_minimized_vmd_whole_with_water_6101.pdb"
)


# ============================================================
# 5. OUTPUT FILES
# ============================================================

OUTPUT_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "equilibrated_300K_500ps_positions_nm.npy"
)

OUTPUT_VELOCITIES = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "equilibrated_300K_500ps_velocities_nm_per_ps.npy"
)

OUTPUT_CHECKPOINT = Path(
    "checkpoints/"
    "pyrene_peg5_graphene_equilibration_300K_500ps.chk"
)

OUTPUT_LOG = Path(
    "analysis/"
    "equilibration_300K_110to500ps_log.csv"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "equilibration_300K_500ps.json"
)

OUTPUT_TRAJECTORY = Path(
    "trajectories/"
    "pyrene_peg5_graphene_equilibration_300K_110to500ps_visible.dcd"
)


# ============================================================
# 6. VERIFY INPUT FILES
# ============================================================

for path in [
    SYSTEM_XML,
    POSITIONS_FILE,
    VELOCITIES_FILE,
    LIGAND_MOL2,
    VMD_TOPOLOGY_PDB,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# 7. LOAD SYSTEM / STATE
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


n_particles = (
    system.getNumParticles()
)


expected_particles = (
    N_SOLUTE
    +
    N_WATERS * SITES_PER_WATER
)


if n_particles != expected_particles:

    raise RuntimeError(
        "Unexpected system particle count: "
        f"{n_particles}, expected {expected_particles}"
    )


if positions_nm.shape != (
    n_particles,
    3,
):

    raise RuntimeError(
        f"Position shape mismatch: {positions_nm.shape}"
    )


if velocities_nm_per_ps.shape != (
    n_particles,
    3,
):

    raise RuntimeError(
        f"Velocity shape mismatch: {velocities_nm_per_ps.shape}"
    )


if not np.isfinite(
    positions_nm
).all():

    raise RuntimeError(
        "Positions contain NaN/Inf."
    )


if not np.isfinite(
    velocities_nm_per_ps
).all():

    raise RuntimeError(
        "Velocities contain NaN/Inf."
    )


# ============================================================
# 8. IDENTIFY PYRENE AROMATIC ATOMS
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

    text = line.strip()

    if text == "@<TRIPOS>ATOM":

        atom_section = i + 1

    elif text == "@<TRIPOS>BOND":

        bond_section = i + 1


if atom_section is None or bond_section is None:

    raise RuntimeError(
        "Could not find MOL2 ATOM/BOND sections."
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

    if len(fields) >= 6:

        ligand_atom_types.append(
            fields[5].lower()
        )


if len(
    ligand_atom_types
) != N_LIGAND:

    raise RuntimeError(
        "Unexpected MOL2 ligand atom count: "
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


aromatic_atoms = np.asarray(
    sorted(
        aromatic_atoms
    ),
    dtype=int,
)


if len(
    aromatic_atoms
) != 16:

    raise RuntimeError(
        "Expected 16 pyrene aromatic atoms, "
        f"found {len(aromatic_atoms)}"
    )


# ============================================================
# 9. DEGREES OF FREEDOM
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
    3 * n_massive_particles
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


R_KJ_PER_MOL_K = (
    0.00831446261815324
)


# ============================================================
# 10. VMD VISIBLE ATOM INDICES
#
# Order MUST match the existing VMD PDB:
#
# 1250 real graphene carbons
# 70 ligand atoms
# 6101 waters x O/H/H
#
# Hidden:
# 2500 graphene pi pseudo-sites
# 6101 OPC M sites
# ============================================================

visible_indices = []


visible_indices.extend(
    range(
        N_GRAPHENE_CARBONS
    )
)


visible_indices.extend(
    range(
        LIGAND_START,
        LIGAND_STOP,
    )
)


for water_index in range(
    N_WATERS
):

    base = (
        N_SOLUTE
        +
        4 * water_index
    )

    visible_indices.extend(
        [
            base,
            base + 1,
            base + 2,
        ]
    )


visible_indices = np.asarray(
    visible_indices,
    dtype=int,
)


expected_visible_atoms = (
    N_GRAPHENE_CARBONS
    +
    N_LIGAND
    +
    3 * N_WATERS
)


if len(
    visible_indices
) != expected_visible_atoms:

    raise RuntimeError(
        "Visible trajectory index count mismatch."
    )


# Build a lightweight topology solely for writing the DCD.
#
# Do NOT parse the visualization PDB with OpenMM here.
# The visualization PDB intentionally contains many repeated
# atom names (for example graphene carbon atoms all named C),
# which OpenMM's PDB parser can interpret as duplicate atoms.
#
# DCD stores coordinates, not chemical atom names.  The only
# critical requirement here is that this topology contains
# exactly the same number of atoms, in the same order, as the
# visible coordinate array written below.

dcd_topology = app.Topology()


dcd_topology.setPeriodicBoxVectors(
    system.getDefaultPeriodicBoxVectors()
)


dcd_chain = (
    dcd_topology.addChain(
        "V"
    )
)


dcd_residue = (
    dcd_topology.addResidue(
        "VIS",
        dcd_chain,
        "1",
    )
)


for atom_index in range(
    expected_visible_atoms
):

    dcd_topology.addAtom(

        f"A{atom_index + 1}",

        None,

        dcd_residue,

        str(
            atom_index + 1
        ),
    )


vmd_atom_count = (
    dcd_topology.getNumAtoms()
)


if vmd_atom_count != (
    expected_visible_atoms
):

    raise RuntimeError(
        "DCD topology atom count mismatch: "
        f"{vmd_atom_count} vs "
        f"{expected_visible_atoms}"
    )


print(
    "DCD trajectory topology atoms:",
    vmd_atom_count,
)


# ============================================================
# 11. CREATE INTEGRATOR
#
# LangevinMiddle keeps the system coupled to a 300 K bath.
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
    EQUILIBRATION_SEED
)


# ============================================================
# 12. CREATE CUDA CONTEXT
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

    cuda_precision = "mixed"


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
# 13. CONTINUE 300 K STATE
# ============================================================

context.setPositions(
    positions_nm
    * unit.nanometer
)


context.applyConstraints(
    CONSTRAINT_TOLERANCE
)


context.setVelocities(
    velocities_nm_per_ps
    * unit.nanometer
    / unit.picosecond
)


context.applyVelocityConstraints(
    CONSTRAINT_TOLERANCE
)


context.computeVirtualSites()


context.setTime(
    110.0
    * unit.picoseconds
)


# ============================================================
# 14. STATIC WATER OXYGEN INDICES
# ============================================================

oxygen_indices = (
    N_SOLUTE
    +
    4 * np.arange(
        N_WATERS
    )
)


# ============================================================
# 15. STATE / METRICS
# ============================================================

def get_state_and_metrics():

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
            graphene_z_mean
        )
    )

    water_above_wall = int(

        np.sum(
            oxygen_z
            >
            UPPER_WATER_WALL_NM
        )
    )

    water_below_wall = int(

        np.sum(
            oxygen_z
            <
            LOWER_WATER_WALL_NM
        )
    )

    upper_wall_violation = max(

        0.0,

        water_z_max
        -
        UPPER_WATER_WALL_NM,
    )

    lower_wall_violation = max(

        0.0,

        LOWER_WATER_WALL_NM
        -
        water_z_min,
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
            potential_energy
            +
            kinetic_energy,

        "graphene_z_min_nm":
            graphene_z_min,

        "graphene_z_max_nm":
            graphene_z_max,

        "graphene_z_mean_nm":
            graphene_z_mean,

        "pyrene_height_above_graphene_nm":
            pyrene_height,

        "ligand_z_min_nm":
            float(
                ligand[:, 2].min()
            ),

        "ligand_z_max_nm":
            float(
                ligand[:, 2].max()
            ),

        "water_z_min_nm":
            water_z_min,

        "water_z_max_nm":
            water_z_max,

        "water_below_graphene":
            water_below_graphene,

        "water_above_upper_wall":
            water_above_wall,

        "water_below_lower_wall":
            water_below_wall,

        "upper_wall_violation_nm":
            upper_wall_violation,

        "lower_wall_violation_nm":
            lower_wall_violation,
    }

    for key, value in metrics.items():

        if isinstance(
            value,
            float,
        ):

            if not math.isfinite(
                value
            ):

                raise RuntimeError(
                    "Non-finite value detected: "
                    f"{key}={value}"
                )

    return (
        state,
        positions,
        velocities,
        metrics,
    )


# ============================================================
# 16. HEADER
# ============================================================

print()
print("=" * 78)
print("300 K NVT EQUILIBRATION - 110 TO 500 ps")
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
    N_WATERS,
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

print(
    "Visible VMD atoms:",
    expected_visible_atoms,
)


print()
print("PROTOCOL")
print("--------")

print(
    f"Temperature: "
    f"{TARGET_TEMPERATURE_K:.1f} K"
)

print(
    "Ensemble: NVT"
)

print(
    "Integrator: LangevinMiddleIntegrator"
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
    f"Trajectory interval: "
    f"{REPORT_INTERVAL_PS:.1f} ps"
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
# 17. INITIAL STATE
# ============================================================

(
    state,
    current_positions,
    current_velocities,
    initial_metrics,
) = get_state_and_metrics()


print()
print("INITIAL CONTINUATION STATE")
print("--------------------------")

print(
    f"Temperature: "
    f"{initial_metrics['temperature_K']:.3f} K"
)

print(
    f"Potential energy: "
    f"{initial_metrics['potential_kJ_per_mol']:.3f} kJ/mol"
)

print(
    f"Pyrene height: "
    f"{initial_metrics['pyrene_height_above_graphene_nm']:.6f} nm"
)

print(
    f"Water O z range: "
    f"{initial_metrics['water_z_min_nm']:.6f} "
    f"to "
    f"{initial_metrics['water_z_max_nm']:.6f} nm"
)

print(
    "Waters below graphene:",
    initial_metrics[
        "water_below_graphene"
    ],
)


# ============================================================
# 18. PREPARE OUTPUT DIRECTORIES
# ============================================================

for path in [
    OUTPUT_POSITIONS,
    OUTPUT_VELOCITIES,
    OUTPUT_CHECKPOINT,
    OUTPUT_LOG,
    OUTPUT_METADATA,
    OUTPUT_TRAJECTORY,
    RECOVERY_CHECKPOINT,
]:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 19. RUN 100 ps AND WRITE VMD DCD
# ============================================================

log_rows = []


log_rows.append(
    initial_metrics.copy()
)


print()
print("=" * 78)
print("RUNNING 300 K NVT EQUILIBRATION")
print("=" * 78)

print()

print(
    " time(ps)    T(K)       "
    "Potential(kJ/mol)   "
    "Pyrene-h(nm)   "
    "Water-z(nm)          "
    "below-G  above-wall"
)

print(
    "-" * 108
)


print(
    f"{initial_metrics['time_ps']:9.3f} "
    f"{initial_metrics['temperature_K']:8.3f} "
    f"{initial_metrics['potential_kJ_per_mol']:18.3f} "
    f"{initial_metrics['pyrene_height_above_graphene_nm']:13.6f} "
    f"{initial_metrics['water_z_min_nm']:7.4f}-"
    f"{initial_metrics['water_z_max_nm']:7.4f} "
    f"{initial_metrics['water_below_graphene']:8d} "
    f"{initial_metrics['water_above_upper_wall']:11d}"
)


with open(
    OUTPUT_TRAJECTORY,
    "wb",
) as trajectory_file:

    dcd = app.DCDFile(

        trajectory_file,

        dcd_topology,

        TIMESTEP_FS
        * unit.femtoseconds,

        firstStep=110000,

        interval=REPORT_STEPS,
    )

    dcd.writeModel(

        current_positions[
            visible_indices
        ]
        * unit.nanometer,

        periodicBoxVectors=(
            state.getPeriodicBoxVectors()
        ),
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
        ) = get_state_and_metrics()

        log_rows.append(
            metrics.copy()
        )

        dcd.writeModel(

            current_positions[
                visible_indices
            ]
            * unit.nanometer,

            periodicBoxVectors=(
                state.getPeriodicBoxVectors()
            ),
        )


        if (
            steps_completed
            % RECOVERY_STEPS
            ==
            0
        ):

            recovery_data = (
                context.createCheckpoint()
            )

            if isinstance(
                recovery_data,
                str,
            ):
                recovery_data = (
                    recovery_data.encode(
                        "latin1"
                    )
                )

            recovery_tmp = Path(
                str(RECOVERY_CHECKPOINT)
                + ".tmp"
            )

            recovery_tmp.write_bytes(
                recovery_data
            )

            recovery_tmp.replace(
                RECOVERY_CHECKPOINT
            )

            trajectory_file.flush()

            print(
                "  >>> RECOVERY CHECKPOINT:",
                f"{metrics['time_ps']:.1f} ps"
            )

        print(
            f"{metrics['time_ps']:9.3f} "
            f"{metrics['temperature_K']:8.3f} "
            f"{metrics['potential_kJ_per_mol']:18.3f} "
            f"{metrics['pyrene_height_above_graphene_nm']:13.6f} "
            f"{metrics['water_z_min_nm']:7.4f}-"
            f"{metrics['water_z_max_nm']:7.4f} "
            f"{metrics['water_below_graphene']:8d} "
            f"{metrics['water_above_upper_wall']:11d}"
        )


# ============================================================
# 20. FINAL STATE
# ============================================================

(
    final_state,
    final_positions,
    final_velocities,
    final_metrics,
) = get_state_and_metrics()


# ============================================================
# 21. SAVE FINAL STATE
# ============================================================

np.save(
    OUTPUT_POSITIONS,
    final_positions,
)


np.save(
    OUTPUT_VELOCITIES,
    final_velocities,
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
# 22. SAVE CSV
# ============================================================

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
# 23. EQUILIBRATION SUMMARY STATISTICS
# ============================================================

half_index = (
    len(log_rows)
    // 2
)


last_half = (
    log_rows[
        half_index:
    ]
)


last_half_temperatures = np.asarray(
    [
        row[
            "temperature_K"
        ]
        for row in last_half
    ]
)


last_half_pyrene = np.asarray(
    [
        row[
            "pyrene_height_above_graphene_nm"
        ]
        for row in last_half
    ]
)


last_half_potential = np.asarray(
    [
        row[
            "potential_kJ_per_mol"
        ]
        for row in last_half
    ]
)


mean_temperature_last_half = float(
    last_half_temperatures.mean()
)


std_temperature_last_half = float(
    last_half_temperatures.std()
)


mean_pyrene_last_half = float(
    last_half_pyrene.mean()
)


std_pyrene_last_half = float(
    last_half_pyrene.std()
)


min_pyrene_last_half = float(
    last_half_pyrene.min()
)


max_pyrene_last_half = float(
    last_half_pyrene.max()
)


mean_potential_last_half = float(
    last_half_potential.mean()
)


max_waters_below_graphene = max(
    row[
        "water_below_graphene"
    ]
    for row in log_rows
)


max_upper_wall_violation = max(
    row[
        "upper_wall_violation_nm"
    ]
    for row in log_rows
)


max_lower_wall_violation = max(
    row[
        "lower_wall_violation_nm"
    ]
    for row in log_rows
)


# ============================================================
# 24. STABILITY CLASSIFICATION
#
# PASS means this 100 ps equilibration run was stable.
# It does NOT automatically mean production can begin.
# We will inspect the trends afterward.
# ============================================================

temperature_ok = (
    290.0
    <=
    mean_temperature_last_half
    <=
    310.0
)


one_sided_ok = (
    max_waters_below_graphene
    ==
    0
)


walls_ok = (
    max_upper_wall_violation
    <
    0.050

    and

    max_lower_wall_violation
    <
    0.050
)


if (
    temperature_ok
    and
    one_sided_ok
    and
    walls_ok
):

    status = (
        "EQUILIBRATION_300K_500PS_PASS"
    )

else:

    status = (
        "EQUILIBRATION_300K_500PS_COMPLETE_INSPECT_WARNINGS"
    )


# ============================================================
# 25. SAVE METADATA
# ============================================================

metadata = {

    "status":
        status,

    "ensemble":
        "NVT",

    "target_temperature_K":
        TARGET_TEMPERATURE_K,

    "total_time_ps":
        TOTAL_TIME_PS,

    "timestep_fs":
        TIMESTEP_FS,

    "total_steps":
        TOTAL_STEPS,

    "report_interval_ps":
        REPORT_INTERVAL_PS,

    "n_particles":
        n_particles,

    "n_waters":
        N_WATERS,

    "visible_trajectory_atoms":
        expected_visible_atoms,

    "friction_per_ps":
        FRICTION_PER_PS,

    "equilibration_seed":
        EQUILIBRATION_SEED,

    "barostat":
        False,

    "ions_added":
        False,

    "ligand_restrained":
        False,

    "initial":
        initial_metrics,

    "final":
        final_metrics,

    "last_50ps":
        {

            "mean_temperature_K":
                mean_temperature_last_half,

            "std_temperature_K":
                std_temperature_last_half,

            "mean_pyrene_height_nm":
                mean_pyrene_last_half,

            "std_pyrene_height_nm":
                std_pyrene_last_half,

            "min_pyrene_height_nm":
                min_pyrene_last_half,

            "max_pyrene_height_nm":
                max_pyrene_last_half,

            "mean_potential_kJ_per_mol":
                mean_potential_last_half,
        },

    "checks":
        {

            "temperature_ok":
                temperature_ok,

            "one_sided_water_ok":
                one_sided_ok,

            "walls_ok":
                walls_ok,

            "maximum_waters_below_graphene":
                int(
                    max_waters_below_graphene
                ),

            "maximum_upper_wall_violation_nm":
                float(
                    max_upper_wall_violation
                ),

            "maximum_lower_wall_violation_nm":
                float(
                    max_lower_wall_violation
                ),
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

    "trajectory_file":
        str(
            OUTPUT_TRAJECTORY
        ),

    "trajectory_topology_pdb":
        str(
            VMD_TOPOLOGY_PDB
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
# 26. FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("300 K EQUILIBRATION SUMMARY - TOTAL 500 ps")
print("=" * 78)

print()

print(
    f"Final temperature: "
    f"{final_metrics['temperature_K']:.3f} K"
)

print(
    f"Mean temperature, last 50 ps: "
    f"{mean_temperature_last_half:.3f} "
    f"+/- "
    f"{std_temperature_last_half:.3f} K"
)

print()

print(
    f"Final pyrene height: "
    f"{final_metrics['pyrene_height_above_graphene_nm']:.6f} nm"
)

print(
    f"Mean pyrene height, last 50 ps: "
    f"{mean_pyrene_last_half:.6f} "
    f"+/- "
    f"{std_pyrene_last_half:.6f} nm"
)

print(
    f"Pyrene range, last 50 ps: "
    f"{min_pyrene_last_half:.6f} "
    f"to "
    f"{max_pyrene_last_half:.6f} nm"
)

print()

print(
    f"Final graphene z range: "
    f"{final_metrics['graphene_z_min_nm']:.6f} "
    f"to "
    f"{final_metrics['graphene_z_max_nm']:.6f} nm"
)

print(
    f"Final water O z range: "
    f"{final_metrics['water_z_min_nm']:.6f} "
    f"to "
    f"{final_metrics['water_z_max_nm']:.6f} nm"
)

print(
    "Maximum waters below graphene:",
    max_waters_below_graphene,
)

print(
    f"Maximum upper wall violation: "
    f"{max_upper_wall_violation:.6f} nm"
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

print(
    "VMD trajectory:",
    OUTPUT_TRAJECTORY
)

print(
    "VMD topology:",
    VMD_TOPOLOGY_PDB
)

print()
print("=" * 78)
print("300 K NVT EQUILIBRATION: COMPLETE")
print("=" * 78)

print()
print(
    "This was equilibration, not production MD."
)

print(
    "No ions were added."
)

print(
    "Pyrene-PEG5 remained unrestrained."
)

print(
    "Inspect the full 500 ps equilibration before starting production."
)

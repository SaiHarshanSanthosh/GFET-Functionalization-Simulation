from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
from openmm import openmm, unit


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

SYSTEM_XML = (
    ROOT / "parameters" / "combined" /
    "ff3_explicit_dpbs_opc_yb_replicaB.xml"
)

POSITIONS_FILE = (
    ROOT / "parameters" / "combined" /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "production_300K_10ns_positions_nm.npy"
)

VELOCITIES_FILE = (
    ROOT / "parameters" / "combined" /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "production_300K_10ns_velocities_nm_per_ps.npy"
)

ASSEMBLY_FILE = (
    ROOT / "analysis" /
    "ff3_explicit_dpbs_opc_yb_replicaB_assembly_check.json"
)

DESIGN_FILE = (
    ROOT / "analysis" /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "na_screening_probe_design.json"
)


# ============================================================
# CONSTANTS
# ============================================================

N_GRAPHENE_CARBONS = 1250
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70

TIMESTEP_FS = 1.0
TIMESTEP_PS = TIMESTEP_FS / 1000.0

TEMPERATURE_K = 300.0
FRICTION_PER_PS = 1.0

CONSTRAINT_TOLERANCE = 1.0e-6

FORCE_GROUP = 30

PULL_UPDATE_PS = 1.0
MONITOR_PS = 10.0

COMPACT_INTERVAL_PS = 1.0
WATER_INTERVAL_PS = 10.0

OPC_STORED_SITES = 3

SEEDS = {
    "probe_0p5nm": 2026082501,
    "probe_1p0nm": 2026082502,
    "probe_1p5nm": 2026082503,
    "probe_2p0nm": 2026082504,
    "probe_3p0nm": 2026082505,
    "bulk_reference": 2026082506,
}


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--branch",
    required=True,
    choices=list(SEEDS.keys()),
)

args = parser.parse_args()

BRANCH = args.branch


# ============================================================
# INPUT VALIDATION
# ============================================================

for path in [
    SYSTEM_XML,
    POSITIONS_FILE,
    VELOCITIES_FILE,
    ASSEMBLY_FILE,
    DESIGN_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input missing: {path}"
        )


assembly = json.loads(
    ASSEMBLY_FILE.read_text(
        encoding="utf-8"
    )
)

design = json.loads(
    DESIGN_FILE.read_text(
        encoding="utf-8"
    )
)


branch_matches = [
    x
    for x in design["branches"]
    if x["name"] == BRANCH
]


if len(branch_matches) != 1:

    raise RuntimeError(
        f"Could not uniquely find branch {BRANCH}"
    )


branch = branch_matches[0]

TARGET_HEIGHT_NM = float(
    branch[
        "target_height_above_graphene_mean_nm"
    ]
)

PULL_PS = float(
    branch["pull_time_ps"]
)

SETTLE_PS = float(
    branch["settle_time_ps"]
)

PRODUCTION_PS = float(
    branch["production_time_ps"]
)

K_RESTRAINT = float(
    branch[
        "restraint_k_kJ_mol_nm2"
    ]
)

PROBE_INDEX = int(
    branch[
        "probe_particle_index"
    ]
)

PROBE_NA_LOCAL_INDEX = int(
    branch[
        "probe_na_local_index"
    ]
)

LANGEVIN_SEED = int(
    SEEDS[BRANCH]
)


# ============================================================
# OUTPUTS
# ============================================================

PREFIX = (
    f"ff3_explicit_dpbs_opc_yb_replicaB_"
    f"na_screening_{BRANCH}"
)

ANALYSIS = ROOT / "analysis"
PARAMETERS = ROOT / "parameters" / "combined"
CHECKPOINTS = ROOT / "checkpoints"

OUTPUT_LOG = (
    ANALYSIS /
    f"{PREFIX}_log.csv"
)

OUTPUT_METADATA = (
    ANALYSIS /
    f"{PREFIX}.json"
)

OUTPUT_POSITIONS = (
    PARAMETERS /
    f"{PREFIX}_final_positions_nm.npy"
)

OUTPUT_VELOCITIES = (
    PARAMETERS /
    f"{PREFIX}_final_velocities_nm_per_ps.npy"
)

OUTPUT_CHECKPOINT = (
    CHECKPOINTS /
    f"{PREFIX}_final.chk"
)

COMPACT_TRAJECTORY = (
    ANALYSIS /
    f"{PREFIX}_production_compact_1ps_float32.npy"
)

COMPACT_METADATA = (
    ANALYSIS /
    f"{PREFIX}_production_compact_1ps.json"
)

WATER_TRAJECTORY = (
    ANALYSIS /
    f"{PREFIX}_production_opc_ohh_10ps_float32.npy"
)

WATER_METADATA = (
    ANALYSIS /
    f"{PREFIX}_production_opc_ohh_10ps.json"
)


for path in [
    OUTPUT_LOG,
    OUTPUT_METADATA,
    OUTPUT_POSITIONS,
    OUTPUT_VELOCITIES,
    OUTPUT_CHECKPOINT,
    COMPACT_TRAJECTORY,
    COMPACT_METADATA,
    WATER_TRAJECTORY,
    WATER_METADATA,
]:

    if path.exists():

        raise RuntimeError(
            "Refusing to overwrite existing output: "
            f"{path}"
        )


ANALYSIS.mkdir(
    parents=True,
    exist_ok=True,
)

PARAMETERS.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINTS.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD SYSTEM + STARTING STATE
# ============================================================

with SYSTEM_XML.open(
    "r",
    encoding="utf-8",
) as f:

    system = openmm.XmlSerializer.deserialize(
        f.read()
    )


positions_nm = np.load(
    POSITIONS_FILE
)

velocities_nm_ps = np.load(
    VELOCITIES_FILE
)


n_particles = (
    system.getNumParticles()
)


if positions_nm.shape != (
    n_particles,
    3,
):

    raise RuntimeError(
        "Position/System mismatch."
    )


if velocities_nm_ps.shape != (
    n_particles,
    3,
):

    raise RuntimeError(
        "Velocity/System mismatch."
    )


# ============================================================
# STRICT ORIGINAL-HAMILTONIAN CHECKS
# ============================================================

force_names = [
    system.getForce(i).getName() or ""
    for i in range(
        system.getNumForces()
    )
]


if not any(
    "Yeh-Berkowitz" in name
    for name in force_names
):

    raise RuntimeError(
        "Yeh-Berkowitz correction not found."
    )


if not any(
    "GrapheneSupport" in name
    for name in force_names
):

    raise RuntimeError(
        "Graphene support force not found."
    )


for i in range(
    system.getNumForces()
):

    force = system.getForce(i)

    if isinstance(
        force,
        openmm.MonteCarloBarostat,
    ):

        raise RuntimeError(
            "Unexpected barostat found."
        )


used_groups = [
    system.getForce(i).getForceGroup()
    for i in range(
        system.getNumForces()
    )
]


if FORCE_GROUP in used_groups:

    raise RuntimeError(
        f"Force group {FORCE_GROUP} "
        "already occupied."
    )


n_original_forces = (
    system.getNumForces()
)


# ============================================================
# PARTICLE INDEX RANGES
# ============================================================

n_waters = int(
    assembly["opc_waters"]
)

sites_per_water = int(
    assembly["opc_sites_per_water"]
)


if sites_per_water != 4:

    raise RuntimeError(
        "Expected four-site OPC water."
    )


ligand_start = N_GRAPHENE_TOTAL
ligand_stop = (
    ligand_start
    +
    N_LIGAND
)


water_start = ligand_stop

water_stop = (
    water_start
    +
    n_waters
    *
    sites_per_water
)


na_start = int(
    assembly[
        "na_particle_start_index"
    ]
)

na_stop = int(
    assembly[
        "na_particle_end_index_exclusive"
    ]
)

k_start = int(
    assembly[
        "k_particle_start_index"
    ]
)

k_stop = int(
    assembly[
        "k_particle_end_index_exclusive"
    ]
)

cl_start = int(
    assembly[
        "cl_particle_start_index"
    ]
)

cl_stop = int(
    assembly[
        "cl_particle_end_index_exclusive"
    ]
)

h2_start = int(
    assembly[
        "h2po4_particle_start_index"
    ]
)

h2_stop = int(
    assembly[
        "h2po4_particle_end_index_exclusive"
    ]
)

hp_start = int(
    assembly[
        "hpo4_particle_start_index"
    ]
)

hp_stop = int(
    assembly[
        "hpo4_particle_end_index_exclusive"
    ]
)


n_na = int(
    assembly["na_ions"]
)

n_k = int(
    assembly["k_ions"]
)

n_cl = int(
    assembly["cl_ions"]
)

n_h2 = int(
    assembly["h2po4_ions"]
)

n_hp = int(
    assembly["hpo4_ions"]
)


if water_stop != na_start:

    raise RuntimeError(
        "Water/Na particle boundary mismatch."
    )


if (
    PROBE_INDEX
    !=
    na_start
    +
    PROBE_NA_LOCAL_INDEX
):

    raise RuntimeError(
        "Probe index does not match Na local index."
    )


# ============================================================
# OPC VIRTUAL-SITE RULE
# ============================================================

first_water = water_start

virtual_site = system.getVirtualSite(
    first_water + 3
)


if not isinstance(
    virtual_site,
    openmm.ThreeParticleAverageSite,
):

    raise RuntimeError(
        "OPC M site is not a "
        "ThreeParticleAverageSite."
    )


wO = float(
    virtual_site.getWeight(0)
)

wH1 = float(
    virtual_site.getWeight(1)
)

wH2 = float(
    virtual_site.getWeight(2)
)


# ============================================================
# ADD PROBE z RESTRAINT
# ============================================================

graphene_initial_mean_z = float(
    np.mean(
        positions_nm[
            :N_GRAPHENE_CARBONS,
            2,
        ]
    )
)

probe_initial_z = float(
    positions_nm[
        PROBE_INDEX,
        2,
    ]
)

INITIAL_HEIGHT_NM = (
    probe_initial_z
    -
    graphene_initial_mean_z
)


restraint = openmm.CustomCentroidBondForce(
    2,
    "0.5*k_probe*(z1-z2-z_target)^2"
)

restraint.setName(
    "NaScreeningProbeZRestraint"
)

restraint.setForceGroup(
    FORCE_GROUP
)

restraint.addGlobalParameter(
    "k_probe",
    K_RESTRAINT
)

restraint.addGlobalParameter(
    "z_target",
    INITIAL_HEIGHT_NM
)


probe_group = restraint.addGroup(
    [PROBE_INDEX],
    [1.0],
)

graphene_group = restraint.addGroup(
    list(
        range(
            N_GRAPHENE_CARBONS
        )
    ),
    [1.0] * N_GRAPHENE_CARBONS,
)


restraint.addBond(
    [
        probe_group,
        graphene_group,
    ],
    [],
)


system.addForce(
    restraint
)


if system.getNumForces() != (
    n_original_forces + 1
):

    raise RuntimeError(
        "Unexpected force-count change."
    )


# ============================================================
# INTEGRATOR / CUDA CONTEXT
# ============================================================

integrator = (
    openmm.LangevinMiddleIntegrator(
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
)


integrator.setConstraintTolerance(
    CONSTRAINT_TOLERANCE
)

integrator.setRandomNumberSeed(
    LANGEVIN_SEED
)


platform = (
    openmm.Platform.getPlatformByName(
        "CUDA"
    )
)


context = openmm.Context(
    system,
    integrator,
    platform,
    {
        "Precision": "mixed",
    },
)


# A modified System cannot safely load the old checkpoint.
# Each branch therefore starts from the exact frozen final
# coordinates and velocities of Replica B 10 ns.
context.setPositions(
    positions_nm
    *
    unit.nanometer
)

context.setVelocities(
    velocities_nm_ps
    *
    unit.nanometer
    /
    unit.picosecond
)


context.computeVirtualSites()


# ============================================================
# STEP COUNTS
# ============================================================

def ps_to_steps(value_ps):

    steps = int(
        round(
            value_ps
            /
            TIMESTEP_PS
        )
    )

    if not math.isclose(
        steps
        *
        TIMESTEP_PS,
        value_ps,
        abs_tol=1e-9,
    ):

        raise RuntimeError(
            f"Time {value_ps} ps does "
            "not map exactly to steps."
        )

    return steps


PULL_STEPS = ps_to_steps(
    PULL_PS
)

SETTLE_STEPS = ps_to_steps(
    SETTLE_PS
)

PRODUCTION_STEPS = ps_to_steps(
    PRODUCTION_PS
)

PULL_UPDATE_STEPS = ps_to_steps(
    PULL_UPDATE_PS
)

MONITOR_STEPS = ps_to_steps(
    MONITOR_PS
)

COMPACT_STEPS = ps_to_steps(
    COMPACT_INTERVAL_PS
)

WATER_STEPS = ps_to_steps(
    WATER_INTERVAL_PS
)


if PULL_STEPS % PULL_UPDATE_STEPS != 0:

    raise RuntimeError(
        "Pull duration not divisible "
        "by pull update interval."
    )


if PRODUCTION_STEPS % COMPACT_STEPS != 0:

    raise RuntimeError(
        "Production not divisible "
        "by compact interval."
    )


if PRODUCTION_STEPS % WATER_STEPS != 0:

    raise RuntimeError(
        "Production not divisible "
        "by water interval."
    )


N_COMPACT_FRAMES = (
    PRODUCTION_STEPS
    //
    COMPACT_STEPS
    +
    1
)

N_WATER_FRAMES = (
    PRODUCTION_STEPS
    //
    WATER_STEPS
    +
    1
)


# ============================================================
# COMPACT TRAJECTORY LAYOUT
# ============================================================

compact_layout = {}

column = 0


def add_layout(
    name,
    shape,
):

    global column

    width = int(
        np.prod(shape)
    )

    compact_layout[name] = {
        "start_column_inclusive":
            column,

        "end_column_exclusive":
            column + width,

        "shape":
            list(shape),
    }

    column += width


add_layout(
    "production_elapsed_ps",
    [1],
)

add_layout(
    "context_time_ps",
    [1],
)

add_layout(
    "graphene_z_summary_nm",
    [3],
)

add_layout(
    "ligand_xyz_nm",
    [N_LIGAND, 3],
)

add_layout(
    "na_xyz_nm",
    [n_na, 3],
)

add_layout(
    "k_xyz_nm",
    [n_k, 3],
)

add_layout(
    "cl_xyz_nm",
    [n_cl, 3],
)

add_layout(
    "h2po4_xyz_nm",
    [
        7 * n_h2,
        3,
    ],
)

add_layout(
    "hpo4_xyz_nm",
    [
        6 * n_hp,
        3,
    ],
)


COMPACT_WIDTH = column


compact = np.lib.format.open_memmap(
    COMPACT_TRAJECTORY,
    mode="w+",
    dtype=np.float32,
    shape=(
        N_COMPACT_FRAMES,
        COMPACT_WIDTH,
    ),
)


water_out = np.lib.format.open_memmap(
    WATER_TRAJECTORY,
    mode="w+",
    dtype=np.float32,
    shape=(
        N_WATER_FRAMES,
        n_waters,
        OPC_STORED_SITES,
        3,
    ),
)


# ============================================================
# LOGGING HELPERS
# ============================================================

OUTPUT_LOG.write_text(
    (
        "stage,stage_elapsed_ps,"
        "context_time_ps,"
        "target_height_nm,"
        "actual_height_mean_nm,"
        "actual_gap_max_nm,"
        "restraint_energy_kJ_per_mol,"
        "temperature_K\n"
    ),
    encoding="utf-8",
)


def state_arrays(
    get_velocities=False,
    get_energy=False,
):

    state = context.getState(
        getPositions=True,
        getVelocities=get_velocities,
        getEnergy=get_energy,
    )

    xyz = np.asarray(
        state.getPositions(
            asNumpy=True
        ).value_in_unit(
            unit.nanometer
        ),
        dtype=np.float64,
    )

    vel = None

    if get_velocities:

        vel = np.asarray(
            state.getVelocities(
                asNumpy=True
            ).value_in_unit(
                unit.nanometer
                /
                unit.picosecond
            ),
            dtype=np.float64,
        )

    return state, xyz, vel


def geometry_from_xyz(
    xyz,
):

    graphene_z = (
        xyz[
            :N_GRAPHENE_CARBONS,
            2,
        ]
    )

    gmean = float(
        np.mean(
            graphene_z
        )
    )

    gmin = float(
        np.min(
            graphene_z
        )
    )

    gmax = float(
        np.max(
            graphene_z
        )
    )

    probe_z = float(
        xyz[
            PROBE_INDEX,
            2,
        ]
    )

    return {
        "gmean": gmean,
        "gmin": gmin,
        "gmax": gmax,
        "probe_height_mean":
            probe_z - gmean,
        "probe_gap_max":
            probe_z - gmax,
    }


def restraint_energy():

    state = context.getState(
        getEnergy=True,
        groups={FORCE_GROUP},
    )

    return float(
        state.getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )


def temperature_from_state(
    state,
):

    kinetic = (
        state.getKineticEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    # Only used as a monitoring quantity.
    # Same DOF used in existing production workflow.
    dof = 233928

    R = (
        unit.MOLAR_GAS_CONSTANT_R
        .value_in_unit(
            unit.kilojoule_per_mole
            /
            unit.kelvin
        )
    )

    return float(
        2.0
        *
        kinetic
        /
        (
            dof
            *
            R
        )
    )


def monitor(
    stage,
    stage_elapsed_ps,
):

    state, xyz, _ = (
        state_arrays(
            get_energy=True
        )
    )

    geo = geometry_from_xyz(
        xyz
    )

    target = float(
        context.getParameter(
            "z_target"
        )
    )

    rE = restraint_energy()

    temp = temperature_from_state(
        state
    )

    context_time = (
        state.getTime()
        .value_in_unit(
            unit.picoseconds
        )
    )

    line = (
        f"{stage},"
        f"{stage_elapsed_ps:.3f},"
        f"{context_time:.3f},"
        f"{target:.6f},"
        f"{geo['probe_height_mean']:.6f},"
        f"{geo['probe_gap_max']:.6f},"
        f"{rE:.6f},"
        f"{temp:.3f}\n"
    )

    with OUTPUT_LOG.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(line)

    print(
        f"{stage:10s} "
        f"{stage_elapsed_ps:8.1f} ps | "
        f"target {target:6.3f} nm | "
        f"actual {geo['probe_height_mean']:6.3f} nm | "
        f"R {rE:8.3f} kJ/mol | "
        f"T {temp:7.2f} K",
        flush=True,
    )


# ============================================================
# STAGE 1 — GENTLE PULL
# ============================================================

print("=" * 78)
print("REPLICA B Na+ DIRECT SCREENING BRANCH")
print("=" * 78)

print()
print(f"Branch:             {BRANCH}")
print(f"Probe particle:     {PROBE_INDEX}")
print(
    f"Initial height:     "
    f"{INITIAL_HEIGHT_NM:.6f} nm"
)
print(
    f"Target height:      "
    f"{TARGET_HEIGHT_NM:.6f} nm"
)
print(f"Pull:               {PULL_PS:.1f} ps")
print(f"Settle:             {SETTLE_PS:.1f} ps")
print(
    f"Production:         "
    f"{PRODUCTION_PS:.1f} ps"
)
print(
    f"Langevin seed:      "
    f"{LANGEVIN_SEED}"
)
print()


pull_chunks = (
    PULL_STEPS
    //
    PULL_UPDATE_STEPS
)


for chunk in range(
    pull_chunks
):

    fraction = (
        (chunk + 1)
        /
        pull_chunks
    )

    target = (
        INITIAL_HEIGHT_NM
        +
        fraction
        *
        (
            TARGET_HEIGHT_NM
            -
            INITIAL_HEIGHT_NM
        )
    )

    context.setParameter(
        "z_target",
        target
    )

    integrator.step(
        PULL_UPDATE_STEPS
    )

    elapsed_ps = (
        (chunk + 1)
        *
        PULL_UPDATE_PS
    )

    if (
        chunk == 0
        or
        (chunk + 1)
        %
        int(
            MONITOR_PS
            /
            PULL_UPDATE_PS
        )
        ==
        0
        or
        chunk
        ==
        pull_chunks - 1
    ):

        monitor(
            "pull",
            elapsed_ps,
        )


# Ensure exact final target.
context.setParameter(
    "z_target",
    TARGET_HEIGHT_NM
)


# ============================================================
# STAGE 2 — SETTLE
# ============================================================

settle_chunks = (
    SETTLE_STEPS
    //
    MONITOR_STEPS
)


for chunk in range(
    settle_chunks
):

    integrator.step(
        MONITOR_STEPS
    )

    monitor(
        "settle",
        (chunk + 1)
        *
        MONITOR_PS,
    )


settle_remainder = (
    SETTLE_STEPS
    %
    MONITOR_STEPS
)


if settle_remainder:

    integrator.step(
        settle_remainder
    )


# ============================================================
# PRODUCTION WRITERS
# ============================================================

def write_compact(
    frame_index,
    production_elapsed_ps,
    state,
    xyz,
):

    row = compact[
        frame_index
    ]

    def put(
        name,
        values,
    ):

        info = compact_layout[name]

        a = int(
            info[
                "start_column_inclusive"
            ]
        )

        b = int(
            info[
                "end_column_exclusive"
            ]
        )

        row[a:b] = np.asarray(
            values,
            dtype=np.float32,
        ).ravel()


    geo = geometry_from_xyz(
        xyz
    )

    context_time = float(
        state.getTime()
        .value_in_unit(
            unit.picoseconds
        )
    )


    put(
        "production_elapsed_ps",
        [production_elapsed_ps],
    )

    put(
        "context_time_ps",
        [context_time],
    )

    put(
        "graphene_z_summary_nm",
        [
            geo["gmean"],
            geo["gmin"],
            geo["gmax"],
        ],
    )

    put(
        "ligand_xyz_nm",
        xyz[
            ligand_start:
            ligand_stop
        ],
    )

    put(
        "na_xyz_nm",
        xyz[
            na_start:
            na_stop
        ],
    )

    put(
        "k_xyz_nm",
        xyz[
            k_start:
            k_stop
        ],
    )

    put(
        "cl_xyz_nm",
        xyz[
            cl_start:
            cl_stop
        ],
    )

    put(
        "h2po4_xyz_nm",
        xyz[
            h2_start:
            h2_stop
        ],
    )

    put(
        "hpo4_xyz_nm",
        xyz[
            hp_start:
            hp_stop
        ],
    )


def write_water(
    frame_index,
    xyz,
):

    all_water = (
        xyz[
            water_start:
            water_stop
        ]
        .reshape(
            n_waters,
            4,
            3,
        )
    )

    water_out[
        frame_index
    ] = (
        all_water[
            :,
            :3,
            :
        ]
        .astype(
            np.float32,
            copy=False,
        )
    )


# ============================================================
# STAGE 3 — PRODUCTION
# ============================================================

print()
print(
    "Starting production recording...",
    flush=True,
)


compact_frame = 0
water_frame = 0


state, xyz, _ = state_arrays(
    get_energy=True
)


write_compact(
    compact_frame,
    0.0,
    state,
    xyz,
)

write_water(
    water_frame,
    xyz,
)


compact_frame += 1
water_frame += 1


production_chunks = (
    PRODUCTION_STEPS
    //
    COMPACT_STEPS
)


wall_start = time.perf_counter()


for chunk in range(
    production_chunks
):

    integrator.step(
        COMPACT_STEPS
    )

    elapsed_ps = (
        (chunk + 1)
        *
        COMPACT_INTERVAL_PS
    )

    state, xyz, _ = state_arrays(
        get_energy=True
    )


    write_compact(
        compact_frame,
        elapsed_ps,
        state,
        xyz,
    )

    compact_frame += 1


    if (
        (chunk + 1)
        %
        int(
            WATER_INTERVAL_PS
            /
            COMPACT_INTERVAL_PS
        )
        ==
        0
    ):

        write_water(
            water_frame,
            xyz,
        )

        water_frame += 1


    if (
        chunk == 0
        or
        (chunk + 1)
        %
        int(
            100.0
            /
            COMPACT_INTERVAL_PS
        )
        ==
        0
        or
        chunk
        ==
        production_chunks - 1
    ):

        monitor(
            "production",
            elapsed_ps,
        )

        compact.flush()
        water_out.flush()


wall_seconds = (
    time.perf_counter()
    -
    wall_start
)


# ============================================================
# FINAL VALIDATION / STATE
# ============================================================

compact.flush()
water_out.flush()


if compact_frame != N_COMPACT_FRAMES:

    raise RuntimeError(
        f"Compact frame mismatch: "
        f"{compact_frame} vs "
        f"{N_COMPACT_FRAMES}"
    )


if water_frame != N_WATER_FRAMES:

    raise RuntimeError(
        f"Water frame mismatch: "
        f"{water_frame} vs "
        f"{N_WATER_FRAMES}"
    )


final_state, final_xyz, final_vel = (
    state_arrays(
        get_velocities=True,
        get_energy=True,
    )
)


if not np.all(
    np.isfinite(
        final_xyz
    )
):

    raise RuntimeError(
        "Non-finite final positions."
    )


if not np.all(
    np.isfinite(
        final_vel
    )
):

    raise RuntimeError(
        "Non-finite final velocities."
    )


np.save(
    OUTPUT_POSITIONS,
    final_xyz,
)

np.save(
    OUTPUT_VELOCITIES,
    final_vel,
)


OUTPUT_CHECKPOINT.write_bytes(
    context.createCheckpoint()
)


final_geo = geometry_from_xyz(
    final_xyz
)


# ============================================================
# METADATA
# ============================================================

box_vectors = (
    system.getDefaultPeriodicBoxVectors()
)

box_nm = np.asarray(
    [
        [
            v.x,
            v.y,
            v.z,
        ]
        for v in box_vectors
    ],
    dtype=float,
)


compact_meta = {
    "format":
        "NumPy .npy matrix",

    "dtype":
        "float32",

    "units":
        {
            "time": "ps",
            "coordinates": "nm",
        },

    "branch":
        BRANCH,

    "probe_particle_index":
        PROBE_INDEX,

    "probe_na_local_index":
        PROBE_NA_LOCAL_INDEX,

    "target_height_above_graphene_mean_nm":
        TARGET_HEIGHT_NM,

    "frame_interval_ps":
        COMPACT_INTERVAL_PS,

    "n_frames":
        N_COMPACT_FRAMES,

    "n_columns":
        COMPACT_WIDTH,

    "matrix_shape":
        [
            N_COMPACT_FRAMES,
            COMPACT_WIDTH,
        ],

    "layout":
        compact_layout,

    "box_vectors_nm":
        box_nm.tolist(),

    "trajectory_file":
        str(
            COMPACT_TRAJECTORY
        ),
}


COMPACT_METADATA.write_text(
    json.dumps(
        compact_meta,
        indent=2,
    ),
    encoding="utf-8",
)


water_meta = {
    "format":
        "NumPy .npy 4D matrix",

    "dtype":
        "float32",

    "units":
        {
            "time": "ps",
            "coordinates": "nm",
        },

    "branch":
        BRANCH,

    "probe_particle_index":
        PROBE_INDEX,

    "target_height_above_graphene_mean_nm":
        TARGET_HEIGHT_NM,

    "frame_interval_ps":
        WATER_INTERVAL_PS,

    "n_frames":
        N_WATER_FRAMES,

    "n_waters":
        n_waters,

    "stored_sites_per_water":
        3,

    "stored_site_order":
        [
            "O",
            "H1",
            "H2",
        ],

    "full_opc_site_order":
        [
            "O",
            "H1",
            "H2",
            "M",
        ],

    "matrix_shape":
        [
            N_WATER_FRAMES,
            n_waters,
            3,
            3,
        ],

    "m_site_reconstruction":
        {
            "equation":
                (
                    "rM = wO*rO + "
                    "wH1*rH1 + "
                    "wH2*rH2"
                ),

            "wO": wO,
            "wH1": wH1,
            "wH2": wH2,
        },

    "box_vectors_nm":
        box_nm.tolist(),

    "trajectory_file":
        str(
            WATER_TRAJECTORY
        ),

    "companion_compact_trajectory":
        str(
            COMPACT_TRAJECTORY
        ),
}


WATER_METADATA.write_text(
    json.dumps(
        water_meta,
        indent=2,
    ),
    encoding="utf-8",
)


metadata = {
    "status":
        "PASS",

    "branch":
        BRANCH,

    "experiment":
        (
            "Direct electrolyte-screening branch "
            "using one existing restrained Na+ probe."
        ),

    "source_state":
        (
            "Replica B 10 ns frozen "
            "positions and velocities"
        ),

    "probe_particle_index":
        PROBE_INDEX,

    "probe_na_local_index":
        PROBE_NA_LOCAL_INDEX,

    "initial_height_above_graphene_mean_nm":
        INITIAL_HEIGHT_NM,

    "target_height_above_graphene_mean_nm":
        TARGET_HEIGHT_NM,

    "final_height_above_graphene_mean_nm":
        final_geo[
            "probe_height_mean"
        ],

    "final_gap_above_graphene_max_nm":
        final_geo[
            "probe_gap_max"
        ],

    "restraint_k_kJ_mol_nm2":
        K_RESTRAINT,

    "probe_restraint_force_group":
        FORCE_GROUP,

    "pull_ps":
        PULL_PS,

    "settle_ps":
        SETTLE_PS,

    "production_ps":
        PRODUCTION_PS,

    "temperature_K":
        TEMPERATURE_K,

    "friction_per_ps":
        FRICTION_PER_PS,

    "timestep_fs":
        TIMESTEP_FS,

    "langevin_seed":
        LANGEVIN_SEED,

    "production_wall_seconds":
        wall_seconds,

    "production_ns_per_hour":
        (
            (
                PRODUCTION_PS
                /
                1000.0
            )
            /
            (
                wall_seconds
                /
                3600.0
            )
        ),

    "compact_frames":
        N_COMPACT_FRAMES,

    "water_frames":
        N_WATER_FRAMES,

    "output_checkpoint":
        str(
            OUTPUT_CHECKPOINT
        ),

    "compact_trajectory":
        str(
            COMPACT_TRAJECTORY
        ),

    "water_trajectory":
        str(
            WATER_TRAJECTORY
        ),
}


OUTPUT_METADATA.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 78)
print("SCREENING BRANCH COMPLETE")
print("=" * 78)

print()
print(
    f"Branch:              "
    f"{BRANCH}"
)

print(
    f"Target height:       "
    f"{TARGET_HEIGHT_NM:.6f} nm"
)

print(
    f"Final actual height: "
    f"{final_geo['probe_height_mean']:.6f} nm"
)

print(
    f"Final gap from max:  "
    f"{final_geo['probe_gap_max']:.6f} nm"
)

print()
print(
    f"Compact frames:      "
    f"{compact_frame}"
)

print(
    f"Water frames:        "
    f"{water_frame}"
)

print(
    f"Production speed:    "
    f"{metadata['production_ns_per_hour']:.3f} ns/hour"
)

print()
print(
    "No particles or charges were added."
)

print(
    "Original FF-3 System modified only "
    "by the Na+ probe z-restraint."
)

print()
print(
    f"{BRANCH}: PASS"
)

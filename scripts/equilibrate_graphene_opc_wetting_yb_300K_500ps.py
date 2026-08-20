from pathlib import Path
import csv
import json
import time

import numpy as np

import openmm
from openmm import unit


# ============================================================
# FF-1B
# GRAPHENE / OPC WETTING PILOT
#
# 500 ps NVT equilibration at 300 K
#
# - bare supported graphene
# - cylindrical OPC droplet
# - validated dynamic Yeh-Berkowitz correction
# - no water walls
# - no barostat
# - no velocity reinitialization
#
# Oxygen coordinates are saved every 2 ps for later
# cylindrical contact-angle analysis.
# ============================================================


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

SYSTEM_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_yb.xml"
)

INPUT_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_yb_heated_300K_positions_nm.npy"
)

INPUT_VELOCITIES = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_yb_heated_300K_velocities_nm_per_ps.npy"
)

OUTPUT_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_yb_equilibrated_300K_500ps_positions_nm.npy"
)

OUTPUT_VELOCITIES = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_yb_equilibrated_300K_500ps_velocities_nm_per_ps.npy"
)

OUTPUT_LOG = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_yb_300K_500ps_log.csv"
)

OUTPUT_OXYGEN_TRAJ = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_yb_300K_500ps_oxygen_trajectory.npz"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_yb_300K_500ps.json"
)

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "graphene_opc_wetting_yb_300K_500ps.chk"
)


# ============================================================
# SETTINGS
# ============================================================

TEMPERATURE_K = 300.0

FRICTION_PER_PS = 1.0

TIMESTEP_PS = 0.001

TOTAL_PS = 500.0

TOTAL_STEPS = int(
    TOTAL_PS
    /
    TIMESTEP_PS
)

REPORT_PS = 2.0

REPORT_STEPS = int(
    REPORT_PS
    /
    TIMESTEP_PS
)

CHECKPOINT_PS = 10.0

CHECKPOINT_STEPS = int(
    CHECKPOINT_PS
    /
    TIMESTEP_PS
)

SEED = 20260820

N_GRAPHENE = 3750
N_CARBON = 1250

YB_FORCE_GROUP = 29


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 80)
print("GRAPHENE / OPC WETTING YB 300 K EQUILIBRATION")
print("=" * 80)


system = openmm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text(
        encoding="utf-8"
    )
)

positions_nm = np.load(
    INPUT_POSITIONS
)

velocities_nm_ps = np.load(
    INPUT_VELOCITIES
)


n_particles = (
    system.getNumParticles()
)


if positions_nm.shape != (
    n_particles,
    3,
):
    raise RuntimeError(
        "Position shape mismatch."
    )


if velocities_nm_ps.shape != (
    n_particles,
    3,
):
    raise RuntimeError(
        "Velocity shape mismatch."
    )


n_virtual = sum(
    system.isVirtualSite(i)
    for i in range(
        n_particles
    )
)

n_massive = (
    n_particles
    -
    n_virtual
)

dof = (
    3 * n_massive
    -
    system.getNumConstraints()
)


print()
print(
    f"Particles:       {n_particles}"
)

print(
    f"Massive:         {n_massive}"
)

print(
    f"Virtual sites:   {n_virtual}"
)

print(
    f"Constraints:     {system.getNumConstraints()}"
)

print(
    f"Approximate DOF: {dof}"
)

print(
    f"Temperature:     {TEMPERATURE_K:.1f} K"
)

print(
    f"Length:          {TOTAL_PS:.1f} ps"
)

print(
    f"Timestep:        {1000*TIMESTEP_PS:.1f} fs"
)

print(
    f"Report interval: {REPORT_PS:.1f} ps"
)


# ============================================================
# VERIFY YB
# ============================================================

yb_found = False


for i in range(
    system.getNumForces()
):

    force = system.getForce(i)

    if (
        isinstance(
            force,
            openmm.CustomCVForce,
        )
        and
        force.getForceGroup()
        ==
        YB_FORCE_GROUP
    ):
        yb_found = True


if not yb_found:
    raise RuntimeError(
        "Yeh-Berkowitz force missing."
    )


print(
    "Yeh-Berkowitz correction: FOUND"
)


# ============================================================
# CONTEXT
# ============================================================

integrator = openmm.LangevinMiddleIntegrator(
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


integrator.setConstraintTolerance(
    1.0e-6
)

integrator.setRandomNumberSeed(
    SEED
)


platform = (
    openmm.Platform
    .getPlatformByName(
        "CUDA"
    )
)


context = openmm.Context(
    system,
    integrator,
    platform,
    {
        "Precision": "mixed"
    },
)


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


print(
    "Platform:        CUDA mixed"
)

print(
    "Velocities:      CONTINUED from heating"
)


# ============================================================
# SELECTIONS
# ============================================================

oxygen_indices = np.arange(
    N_GRAPHENE,
    n_particles,
    4,
    dtype=int,
)


if len(
    oxygen_indices
) != n_virtual:

    raise RuntimeError(
        "OPC oxygen indexing does not match water count."
    )


# ============================================================
# STORAGE
# ============================================================

rows = []

trajectory_times_ps = []

oxygen_frames = []

carbon_planes = []


def evaluate(
    time_ps,
):

    context.computeVirtualSites()


    state = context.getState(
        getEnergy=True,
        getPositions=True,
    )


    potential = float(
        state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )


    kinetic = float(
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
            dof
            *
            0.00831446261815324
        )
    )


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


    carbon_plane = float(
        np.mean(
            xyz[
                :N_CARBON,
                2,
            ]
        )
    )


    O_xyz = (
        xyz[
            oxygen_indices
        ]
    )


    heights = (
        O_xyz[:, 2]
        -
        carbon_plane
    )


    yb_state = context.getState(
        getEnergy=True,
        groups=(
            1 << YB_FORCE_GROUP
        ),
    )


    yb_energy = float(
        yb_state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )


    below_graphene = int(
        np.sum(
            heights < 0.0
        )
    )


    result = {
        "time_ps": float(
            time_ps
        ),

        "temperature_K": float(
            temperature
        ),

        "potential_energy_kJ_mol": float(
            potential
        ),

        "kinetic_energy_kJ_mol": float(
            kinetic
        ),

        "yb_energy_kJ_mol": float(
            yb_energy
        ),

        "carbon_plane_nm": float(
            carbon_plane
        ),

        "lowest_O_height_nm": float(
            np.min(
                heights
            )
        ),

        "highest_O_height_nm": float(
            np.max(
                heights
            )
        ),

        "mean_O_height_nm": float(
            np.mean(
                heights
            )
        ),

        "waters_below_graphene": (
            below_graphene
        ),
    }


    return (
        result,
        O_xyz,
        carbon_plane,
    )


# ============================================================
# INITIAL FRAME
# ============================================================

result, O_xyz, carbon_plane = evaluate(
    0.0
)

rows.append(
    result
)

trajectory_times_ps.append(
    0.0
)

oxygen_frames.append(
    O_xyz.astype(
        np.float32
    )
)

carbon_planes.append(
    carbon_plane
)


# ============================================================
# RUN
# ============================================================

print()
print("=" * 80)
print("RUNNING 500 ps EQUILIBRATION")
print("=" * 80)


start_wall = time.time()


current_step = 0


while current_step < TOTAL_STEPS:

    next_report_step = min(
        current_step
        +
        REPORT_STEPS,

        TOTAL_STEPS,
    )


    integrator.step(
        next_report_step
        -
        current_step
    )


    current_step = (
        next_report_step
    )


    time_ps = (
        current_step
        *
        TIMESTEP_PS
    )


    result, O_xyz, carbon_plane = evaluate(
        time_ps
    )


    rows.append(
        result
    )


    trajectory_times_ps.append(
        time_ps
    )


    oxygen_frames.append(
        O_xyz.astype(
            np.float32
        )
    )


    carbon_planes.append(
        carbon_plane
    )


    if (
        current_step
        %
        CHECKPOINT_STEPS
        ==
        0
    ):

        CHECKPOINT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        CHECKPOINT.write_bytes(
            context.createCheckpoint()
        )


    if (
        current_step
        %
        int(
            100.0
            /
            TIMESTEP_PS
        )
        ==
        0
    ):

        print(
            f"{time_ps:7.1f} ps  "
            f"T={result['temperature_K']:7.2f} K  "
            f"PE={result['potential_energy_kJ_mol']:11.1f}  "
            f"YB={result['yb_energy_kJ_mol']:8.4f}  "
            f"Omin={result['lowest_O_height_nm']:6.3f}  "
            f"Omax={result['highest_O_height_nm']:6.3f}  "
            f"below={result['waters_below_graphene']}"
        )


    if not np.isfinite(
        result[
            "potential_energy_kJ_mol"
        ]
    ):

        raise RuntimeError(
            "Non-finite potential energy."
        )


    if (
        result[
            "waters_below_graphene"
        ]
        >
        0
    ):

        raise RuntimeError(
            "Water crossed below graphene."
        )


    if (
        result[
            "lowest_O_height_nm"
        ]
        <
        0.18
    ):

        raise RuntimeError(
            "Water oxygen approached graphene "
            "unphysically closely."
        )


elapsed_s = (
    time.time()
    -
    start_wall
)


# ============================================================
# FINAL STATE
# ============================================================

context.computeVirtualSites()


final_state = context.getState(
    getPositions=True,
    getVelocities=True,
)


final_positions = np.asarray(
    final_state
    .getPositions(
        asNumpy=True
    )
    .value_in_unit(
        unit.nanometer
    ),
    dtype=float,
)


final_velocities = np.asarray(
    final_state
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


np.save(
    OUTPUT_POSITIONS,
    final_positions
)


np.save(
    OUTPUT_VELOCITIES,
    final_velocities
)


CHECKPOINT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT.write_bytes(
    context.createCheckpoint()
)


# ============================================================
# SAVE OXYGEN TRAJECTORY
# ============================================================

oxygen_frames = np.stack(
    oxygen_frames,
    axis=0,
)


trajectory_times_ps = np.asarray(
    trajectory_times_ps,
    dtype=float,
)


carbon_planes = np.asarray(
    carbon_planes,
    dtype=float,
)


np.savez_compressed(
    OUTPUT_OXYGEN_TRAJ,
    times_ps=trajectory_times_ps,
    oxygen_positions_nm=oxygen_frames,
    carbon_plane_nm=carbon_planes,
)


# ============================================================
# CSV
# ============================================================

OUTPUT_LOG.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_LOG,
    "w",
    newline="",
    encoding="utf-8",
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# LAST 200 ps SUMMARY
# ============================================================

analysis_rows = [
    row
    for row in rows
    if row[
        "time_ps"
    ] >= 300.0
]


temperature_last = np.asarray(
    [
        row[
            "temperature_K"
        ]
        for row in analysis_rows
    ]
)


pe_last = np.asarray(
    [
        row[
            "potential_energy_kJ_mol"
        ]
        for row in analysis_rows
    ]
)


yb_last = np.asarray(
    [
        row[
            "yb_energy_kJ_mol"
        ]
        for row in analysis_rows
    ]
)


lowest_last = np.asarray(
    [
        row[
            "lowest_O_height_nm"
        ]
        for row in analysis_rows
    ]
)


highest_last = np.asarray(
    [
        row[
            "highest_O_height_nm"
        ]
        for row in analysis_rows
    ]
)


max_below = max(
    row[
        "waters_below_graphene"
    ]
    for row in rows
)


metadata = {
    "system_xml": str(
        SYSTEM_XML
    ),

    "length_ps": (
        TOTAL_PS
    ),

    "temperature_K": (
        TEMPERATURE_K
    ),

    "timestep_fs": (
        TIMESTEP_PS
        *
        1000.0
    ),

    "friction_per_ps": (
        FRICTION_PER_PS
    ),

    "seed": (
        SEED
    ),

    "report_interval_ps": (
        REPORT_PS
    ),

    "last_200ps": {
        "mean_temperature_K": float(
            np.mean(
                temperature_last
            )
        ),

        "sd_temperature_K": float(
            np.std(
                temperature_last
            )
        ),

        "mean_PE_kJ_mol": float(
            np.mean(
                pe_last
            )
        ),

        "sd_PE_kJ_mol": float(
            np.std(
                pe_last
            )
        ),

        "mean_YB_energy_kJ_mol": float(
            np.mean(
                yb_last
            )
        ),

        "max_YB_energy_kJ_mol": float(
            np.max(
                yb_last
            )
        ),

        "mean_lowest_O_height_nm": float(
            np.mean(
                lowest_last
            )
        ),

        "minimum_O_height_nm": float(
            np.min(
                lowest_last
            )
        ),

        "mean_highest_O_height_nm": float(
            np.mean(
                highest_last
            )
        ),

        "maximum_highest_O_height_nm": float(
            np.max(
                highest_last
            )
        ),
    },

    "max_waters_below_graphene": int(
        max_below
    ),

    "wall_time_s": float(
        elapsed_s
    ),

    "output_positions": str(
        OUTPUT_POSITIONS
    ),

    "output_velocities": str(
        OUTPUT_VELOCITIES
    ),

    "oxygen_trajectory": str(
        OUTPUT_OXYGEN_TRAJ
    ),

    "checkpoint": str(
        CHECKPOINT
    ),
}


OUTPUT_JSON.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 80)
print("500 ps EQUILIBRATION SUMMARY")
print("=" * 80)

print(
    f"Last 200 ps T:       "
    f"{np.mean(temperature_last):.3f} "
    f"+/- "
    f"{np.std(temperature_last):.3f} K"
)

print(
    f"Last 200 ps PE:      "
    f"{np.mean(pe_last):.3f} "
    f"+/- "
    f"{np.std(pe_last):.3f} kJ/mol"
)

print(
    f"Last 200 ps YB E:    "
    f"{np.mean(yb_last):.6f} kJ/mol "
    f"(max {np.max(yb_last):.6f})"
)

print(
    f"Lowest O, mean:      "
    f"{np.mean(lowest_last):.6f} nm"
)

print(
    f"Lowest O, minimum:   "
    f"{np.min(lowest_last):.6f} nm"
)

print(
    f"Highest O, mean:     "
    f"{np.mean(highest_last):.6f} nm"
)

print(
    f"Waters below sheet:  "
    f"{max_below}"
)

print(
    f"O trajectory frames: "
    f"{oxygen_frames.shape[0]}"
)

print(
    f"Wall time:           "
    f"{elapsed_s:.2f} s"
)

print()
print(
    f"O trajectory:"
    f"\n  {OUTPUT_OXYGEN_TRAJ}"
)

print(
    f"Checkpoint:"
    f"\n  {CHECKPOINT}"
)

print()
print(
    "GRAPHENE_OPC_WETTING_YB_300K_500PS_PASS"
)

print("=" * 80)
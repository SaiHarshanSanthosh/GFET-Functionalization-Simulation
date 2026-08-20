from pathlib import Path
import csv
import json
import time

import numpy as np

import openmm
from openmm import unit


# ============================================================
# FF-1B
# GRAPHENE / OPC WETTING PILOT HEATING
#
# Temperatures:
#   50 -> 100 -> 150 -> 200 -> 250 -> 300 K
#
# 20 ps per stage
# 1 fs timestep
#
# Dynamic Yeh-Berkowitz correction is already contained
# in the System XML.
# ============================================================


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

SYSTEM_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_25x50_yb.xml"
)

INPUT_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_25x50_yb_minimized_converged_positions_nm.npy"
)

OUTPUT_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_25x50_yb_heated_300K_positions_nm.npy"
)

OUTPUT_VELOCITIES = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_25x50_yb_heated_300K_velocities_nm_per_ps.npy"
)

OUTPUT_CSV = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_25x50_yb_heating.csv"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_25x50_yb_heating.json"
)

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "graphene_opc_wetting_25x50_yb_heated_300K.chk"
)


# ============================================================
# SETTINGS
# ============================================================

TEMPERATURES_K = [
    50.0,
    100.0,
    150.0,
    200.0,
    250.0,
    300.0,
]

STAGE_PS = 20.0

TIMESTEP_PS = 0.001

STEPS_PER_STAGE = int(
    STAGE_PS
    /
    TIMESTEP_PS
)

REPORT_EVERY_STEPS = 1000

FRICTION_PER_PS = 1.0

RANDOM_SEED = 20260819

N_GRAPHENE = 7500
N_CARBON = 2500

YB_FORCE_GROUP = 29


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 80)
print("GRAPHENE / OPC WETTING YB HEATING")
print("=" * 80)


system = openmm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text(
        encoding="utf-8"
    )
)

positions_nm = np.load(
    INPUT_POSITIONS
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


n_virtual = sum(
    system.isVirtualSite(i)
    for i in range(
        n_particles
    )
)


print()
print(
    f"Particles:       {n_particles}"
)

print(
    f"Virtual sites:   {n_virtual}"
)

print(
    f"Heating stages:  {TEMPERATURES_K}"
)

print(
    f"Stage length:    {STAGE_PS:.1f} ps"
)

print(
    f"Total heating:   "
    f"{STAGE_PS * len(TEMPERATURES_K):.1f} ps"
)

print(
    f"Timestep:        "
    f"{1000*TIMESTEP_PS:.1f} fs"
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
        "Yeh-Berkowitz correction missing."
    )


print(
    "Yeh-Berkowitz correction: FOUND"
)


# ============================================================
# INTEGRATOR / CONTEXT
# ============================================================

integrator = openmm.LangevinMiddleIntegrator(
    TEMPERATURES_K[0]
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
    RANDOM_SEED
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

context.computeVirtualSites()


context.setVelocitiesToTemperature(
    TEMPERATURES_K[0]
    *
    unit.kelvin,

    RANDOM_SEED,
)


print(
    "Platform:        CUDA mixed"
)


# ============================================================
# DEGREES OF FREEDOM
#
# Massive coordinates
# minus constraints.
#
# We are interested mainly in temperature stability rather
# than claiming an exact thermodynamic DOF decomposition.
# ============================================================

n_massive = sum(
    not system.isVirtualSite(i)
    for i in range(
        n_particles
    )
)


dof = (
    3 * n_massive
    -
    system.getNumConstraints()
)


print(
    f"Approximate DOF: {dof}"
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


# ============================================================
# LOGGING
# ============================================================

rows = []


def evaluate(
    target_temperature,
    stage_index,
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


    oxygen_heights = (
        xyz[
            oxygen_indices,
            2,
        ]
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


    return {
        "stage": int(
            stage_index
        ),

        "target_temperature_K": float(
            target_temperature
        ),

        "measured_temperature_K": float(
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

        "lowest_water_O_height_nm": float(
            np.min(
                oxygen_heights
            )
        ),

        "highest_water_O_height_nm": float(
            np.max(
                oxygen_heights
            )
        ),
    }


# ============================================================
# HEAT
# ============================================================

start_wall = time.time()


for stage_index, temperature_K in enumerate(
    TEMPERATURES_K,
    start=1,
):

    print()
    print(
        "-" * 80
    )

    print(
        f"Stage {stage_index}/"
        f"{len(TEMPERATURES_K)}: "
        f"{temperature_K:.0f} K"
    )

    print(
        "-" * 80
    )


    integrator.setTemperature(
        temperature_K
        *
        unit.kelvin
    )


    # Reinitialize velocities at the beginning of each
    # temperature stage so the ramp is controlled and does
    # not depend on the preceding instantaneous velocity
    # distribution.
    context.setVelocitiesToTemperature(
        temperature_K
        *
        unit.kelvin,

        RANDOM_SEED
        +
        stage_index,
    )


    stage_temperatures = []


    for step_start in range(
        0,
        STEPS_PER_STAGE,
        REPORT_EVERY_STEPS,
    ):

        nstep = min(
            REPORT_EVERY_STEPS,
            STEPS_PER_STAGE
            -
            step_start
        )


        integrator.step(
            nstep
        )


        result = evaluate(
            temperature_K,
            stage_index,
        )


        stage_temperatures.append(
            result[
                "measured_temperature_K"
            ]
        )


        rows.append(
            result
        )


    final = rows[-1]


    print(
        f"Measured T:       "
        f"{np.mean(stage_temperatures):.2f} "
        f"+/- "
        f"{np.std(stage_temperatures):.2f} K"
    )

    print(
        f"Final PE:         "
        f"{final['potential_energy_kJ_mol']:.3f} "
        f"kJ/mol"
    )

    print(
        f"Final YB E:       "
        f"{final['yb_energy_kJ_mol']:.6f} "
        f"kJ/mol"
    )

    print(
        f"Lowest O height:  "
        f"{final['lowest_water_O_height_nm']:.4f} "
        f"nm"
    )

    print(
        f"Highest O height: "
        f"{final['highest_water_O_height_nm']:.4f} "
        f"nm"
    )


    if not np.isfinite(
        final[
            "potential_energy_kJ_mol"
        ]
    ):
        raise RuntimeError(
            "Non-finite energy during heating."
        )


    if (
        final[
            "lowest_water_O_height_nm"
        ]
        <
        0.15
    ):
        raise RuntimeError(
            "Water oxygen approached graphene "
            "unphysically closely."
        )


# ============================================================
# FINAL STATE
# ============================================================

context.computeVirtualSites()


final_state = context.getState(
    getPositions=True,
    getVelocities=True,
    getEnergy=True,
)


final_positions_nm = np.asarray(
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
    final_positions_nm,
)


np.save(
    OUTPUT_VELOCITIES,
    final_velocities,
)


CHECKPOINT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT.write_bytes(
    context.createCheckpoint()
)


# ============================================================
# CSV
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_CSV,
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
# FINAL SUMMARY
# ============================================================

final_300 = [
    row
    for row in rows
    if row[
        "target_temperature_K"
    ] == 300.0
]


temperatures_300 = np.asarray(
    [
        row[
            "measured_temperature_K"
        ]
        for row in final_300
    ],
    dtype=float,
)


elapsed = (
    time.time()
    -
    start_wall
)


metadata = {
    "system_xml": str(
        SYSTEM_XML
    ),

    "input_positions": str(
        INPUT_POSITIONS
    ),

    "temperatures_K": (
        TEMPERATURES_K
    ),

    "stage_ps": (
        STAGE_PS
    ),

    "total_heating_ps": (
        STAGE_PS
        *
        len(
            TEMPERATURES_K
        )
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
        RANDOM_SEED
    ),

    "final_300K": {
        "mean_temperature_K": float(
            np.mean(
                temperatures_300
            )
        ),

        "sd_temperature_K": float(
            np.std(
                temperatures_300
            )
        ),

        "potential_energy_kJ_mol": (
            rows[-1][
                "potential_energy_kJ_mol"
            ]
        ),

        "yb_energy_kJ_mol": (
            rows[-1][
                "yb_energy_kJ_mol"
            ]
        ),

        "lowest_water_O_height_nm": (
            rows[-1][
                "lowest_water_O_height_nm"
            ]
        ),

        "highest_water_O_height_nm": (
            rows[-1][
                "highest_water_O_height_nm"
            ]
        ),
    },

    "wall_time_s": float(
        elapsed
    ),

    "output_positions": str(
        OUTPUT_POSITIONS
    ),

    "output_velocities": str(
        OUTPUT_VELOCITIES
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


print()
print("=" * 80)
print("HEATING SUMMARY")
print("=" * 80)

print(
    f"300 K stage T: "
    f"{np.mean(temperatures_300):.3f} "
    f"+/- "
    f"{np.std(temperatures_300):.3f} K"
)

print(
    f"Final potential E: "
    f"{rows[-1]['potential_energy_kJ_mol']:.3f} "
    f"kJ/mol"
)

print(
    f"Final YB E: "
    f"{rows[-1]['yb_energy_kJ_mol']:.6f} "
    f"kJ/mol"
)

print(
    f"Lowest O height: "
    f"{rows[-1]['lowest_water_O_height_nm']:.6f} nm"
)

print(
    f"Highest O height: "
    f"{rows[-1]['highest_water_O_height_nm']:.6f} nm"
)

print(
    f"Wall time: "
    f"{elapsed:.2f} s"
)

print()
print(
    f"Positions:"
    f"\n  {OUTPUT_POSITIONS}"
)

print(
    f"Velocities:"
    f"\n  {OUTPUT_VELOCITIES}"
)

print(
    f"Checkpoint:"
    f"\n  {CHECKPOINT}"
)

print(
    f"CSV:"
    f"\n  {OUTPUT_CSV}"
)

print(
    f"JSON:"
    f"\n  {OUTPUT_JSON}"
)

print()
print(
    "GRAPHENE_OPC_WETTING_YB_HEATING_PASS"
)

print("=" * 80)
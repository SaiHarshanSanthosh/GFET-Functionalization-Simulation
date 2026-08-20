from pathlib import Path
import json
import time

import numpy as np

import openmm
from openmm import unit


# ============================================================
# FF-1B
# MINIMIZE GRAPHENE / OPC WETTING PILOT
#
# Uses:
#   - bare supported graphene
#   - cylindrical OPC droplet
#   - validated dynamic Yeh-Berkowitz correction
#
# NO heating.
# NO MD.
#
# Minimization convergence is evaluated using:
#   - energy change
#   - RMS force on massive particles
#   - max force on massive particles
#   - explicit constraint errors
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
    / "graphene_opc_wetting_cylinder_25x50_positions_nm.npy"
)

OUTPUT_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_25x50_yb_minimized_positions_nm.npy"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_25x50_yb_minimization.json"
)


# ============================================================
# SETTINGS
# ============================================================

MINIMIZATION_TOLERANCE = 10.0
MAX_ITERATIONS = 5000

CONSTRAINT_TOLERANCE = 1.0e-6

YB_FORCE_GROUP = 29


# ============================================================
# HELPERS
# ============================================================

def force_metrics(
    forces,
    massive_mask,
):

    massive_forces = (
        forces[
            massive_mask
        ]
    )

    magnitude = np.sqrt(
        np.sum(
            massive_forces**2,
            axis=1,
        )
    )

    rms_components = float(
        np.sqrt(
            np.mean(
                massive_forces**2
            )
        )
    )

    rms_magnitude = float(
        np.sqrt(
            np.mean(
                magnitude**2
            )
        )
    )

    max_magnitude = float(
        np.max(
            magnitude
        )
    )

    return {
        "rms_components_kJ_mol_nm": (
            rms_components
        ),

        "rms_magnitude_kJ_mol_nm": (
            rms_magnitude
        ),

        "max_magnitude_kJ_mol_nm": (
            max_magnitude
        ),
    }


def constraint_metrics(
    system,
    positions_nm,
):

    absolute_errors = []
    relative_errors = []


    for i in range(
        system.getNumConstraints()
    ):

        p1, p2, target = (
            system.getConstraintParameters(i)
        )

        p1 = int(p1)
        p2 = int(p2)

        target_nm = float(
            target.value_in_unit(
                unit.nanometer
            )
        )


        actual_nm = float(
            np.linalg.norm(
                positions_nm[p1]
                -
                positions_nm[p2]
            )
        )


        abs_error = abs(
            actual_nm
            -
            target_nm
        )


        relative_error = (
            abs_error
            /
            target_nm
        )


        absolute_errors.append(
            abs_error
        )

        relative_errors.append(
            relative_error
        )


    return {
        "max_absolute_error_nm": float(
            np.max(
                absolute_errors
            )
        ),

        "rms_absolute_error_nm": float(
            np.sqrt(
                np.mean(
                    np.asarray(
                        absolute_errors
                    )**2
                )
            )
        ),

        "max_relative_error": float(
            np.max(
                relative_errors
            )
        ),
    }


def get_state_metrics(
    context,
    system,
    massive_mask,
):

    context.computeVirtualSites()


    state = context.getState(
        getEnergy=True,
        getForces=True,
        getPositions=True,
    )


    energy = float(
        state
        .getPotentialEnergy()
        .value_in_unit(
            unit.kilojoule_per_mole
        )
    )


    forces = np.asarray(
        state
        .getForces(
            asNumpy=True
        )
        .value_in_unit(
            unit.kilojoule_per_mole
            /
            unit.nanometer
        ),
        dtype=float,
    )


    positions_nm = np.asarray(
        state
        .getPositions(
            asNumpy=True
        )
        .value_in_unit(
            unit.nanometer
        ),
        dtype=float,
    )


    fm = force_metrics(
        forces,
        massive_mask,
    )


    cm = constraint_metrics(
        system,
        positions_nm,
    )


    return (
        energy,
        forces,
        positions_nm,
        fm,
        cm,
    )


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 80)
print("GRAPHENE / OPC WETTING YB MINIMIZATION")
print("=" * 80)


if not SYSTEM_XML.exists():
    raise FileNotFoundError(
        SYSTEM_XML
    )


if not INPUT_POSITIONS.exists():
    raise FileNotFoundError(
        INPUT_POSITIONS
    )


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
        "Coordinate shape does not match System."
    )


virtual_mask = np.array(
    [
        system.isVirtualSite(i)
        for i in range(
            n_particles
        )
    ],
    dtype=bool,
)


massive_mask = ~virtual_mask


n_virtual = int(
    np.sum(
        virtual_mask
    )
)

n_massive = int(
    np.sum(
        massive_mask
    )
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
    f"Constraints:     "
    f"{system.getNumConstraints()}"
)

print(
    f"Min tolerance:   "
    f"{MINIMIZATION_TOLERANCE:.3f} "
    f"kJ/mol/nm"
)

print(
    f"Max iterations:  "
    f"{MAX_ITERATIONS}"
)

print(
    f"Constraint tol:  "
    f"{CONSTRAINT_TOLERANCE:.1e}"
)


# ============================================================
# VERIFY YB FORCE EXISTS
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
        "Validated Yeh-Berkowitz CustomCVForce "
        "was not found."
    )


print()
print(
    "Yeh-Berkowitz force group 29: FOUND"
)


# ============================================================
# CONTEXT
# ============================================================

try:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CUDA"
        )
    )

    properties = {
        "Precision": "mixed"
    }


except Exception:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CPU"
        )
    )

    properties = {}


integrator = openmm.VerletIntegrator(
    0.001
    *
    unit.picoseconds
)


integrator.setConstraintTolerance(
    CONSTRAINT_TOLERANCE
)


context = openmm.Context(
    system,
    integrator,
    platform,
    properties,
)


context.setPositions(
    positions_nm
    *
    unit.nanometer
)


context.computeVirtualSites()


print(
    f"Platform:        "
    f"{platform.getName()}"
)


# ============================================================
# BEFORE
# ============================================================

(
    initial_energy,
    initial_forces,
    initial_realized_positions,
    initial_force_metrics,
    initial_constraint_metrics,
) = get_state_metrics(
    context,
    system,
    massive_mask,
)


print()
print("=" * 80)
print("BEFORE MINIMIZATION")
print("=" * 80)

print(
    f"Potential energy:        "
    f"{initial_energy:.6f} kJ/mol"
)

print(
    f"RMS force components:    "
    f"{initial_force_metrics['rms_components_kJ_mol_nm']:.6f} "
    f"kJ/mol/nm"
)

print(
    f"RMS |F|:                 "
    f"{initial_force_metrics['rms_magnitude_kJ_mol_nm']:.6f} "
    f"kJ/mol/nm"
)

print(
    f"Max |F|:                 "
    f"{initial_force_metrics['max_magnitude_kJ_mol_nm']:.6f} "
    f"kJ/mol/nm"
)

print(
    f"Max constraint abs err:  "
    f"{initial_constraint_metrics['max_absolute_error_nm']:.3e} nm"
)

print(
    f"Max constraint rel err:  "
    f"{initial_constraint_metrics['max_relative_error']:.3e}"
)


# ============================================================
# MINIMIZE
# ============================================================

print()
print("=" * 80)
print("RUNNING L-BFGS MINIMIZATION")
print("=" * 80)

start_time = time.time()


openmm.LocalEnergyMinimizer.minimize(
    context,
    tolerance=MINIMIZATION_TOLERANCE,
    maxIterations=MAX_ITERATIONS,
)


elapsed_s = (
    time.time()
    -
    start_time
)


print(
    f"Wall time: "
    f"{elapsed_s:.3f} s"
)


# ============================================================
# AFTER
# ============================================================

(
    final_energy,
    final_forces,
    final_positions_nm,
    final_force_metrics,
    final_constraint_metrics,
) = get_state_metrics(
    context,
    system,
    massive_mask,
)


energy_change = (
    final_energy
    -
    initial_energy
)


print()
print("=" * 80)
print("AFTER MINIMIZATION")
print("=" * 80)

print(
    f"Potential energy:        "
    f"{final_energy:.6f} kJ/mol"
)

print(
    f"Energy change:           "
    f"{energy_change:.6f} kJ/mol"
)

print(
    f"RMS force components:    "
    f"{final_force_metrics['rms_components_kJ_mol_nm']:.6f} "
    f"kJ/mol/nm"
)

print(
    f"RMS |F|:                 "
    f"{final_force_metrics['rms_magnitude_kJ_mol_nm']:.6f} "
    f"kJ/mol/nm"
)

print(
    f"Max |F|:                 "
    f"{final_force_metrics['max_magnitude_kJ_mol_nm']:.6f} "
    f"kJ/mol/nm"
)

print(
    f"Max constraint abs err:  "
    f"{final_constraint_metrics['max_absolute_error_nm']:.3e} nm"
)

print(
    f"Max constraint rel err:  "
    f"{final_constraint_metrics['max_relative_error']:.3e}"
)


# ============================================================
# YB ENERGY AFTER MINIMIZATION
# ============================================================

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


print()
print(
    f"Final YB correction E:   "
    f"{yb_energy:.9f} kJ/mol"
)


# ============================================================
# WATER / GRAPHENE EXTENTS
# ============================================================

N_GRAPHENE = 7500

oxygen_indices = np.arange(
    N_GRAPHENE,
    n_particles,
    4,
    dtype=int,
)


oxygen_positions = (
    final_positions_nm[
        oxygen_indices
    ]
)


carbon_plane = float(
    np.mean(
        final_positions_nm[
            :2500,
            2,
        ]
    )
)


minimum_O_height = float(
    np.min(
        oxygen_positions[:, 2]
        -
        carbon_plane
    )
)


maximum_O_height = float(
    np.max(
        oxygen_positions[:, 2]
        -
        carbon_plane
    )
)


print()
print(
    f"Final carbon plane:      "
    f"{carbon_plane:.6f} nm"
)

print(
    f"Lowest O above plane:    "
    f"{minimum_O_height:.6f} nm"
)

print(
    f"Highest O above plane:   "
    f"{maximum_O_height:.6f} nm"
)


if minimum_O_height < 0.15:

    raise RuntimeError(
        "Water oxygen approached graphene "
        "unphysically closely."
    )


# ============================================================
# FINITE CHECK
# ============================================================

if not np.isfinite(
    final_energy
):

    raise RuntimeError(
        "Final energy is non-finite."
    )


if not np.all(
    np.isfinite(
        final_forces
    )
):

    raise RuntimeError(
        "Final force array is non-finite."
    )


if not np.all(
    np.isfinite(
        final_positions_nm
    )
):

    raise RuntimeError(
        "Final coordinates are non-finite."
    )


# ============================================================
# SAVE
# ============================================================

np.save(
    OUTPUT_POSITIONS,
    final_positions_nm,
)


results = {
    "system_xml": str(
        SYSTEM_XML
    ),

    "input_positions": str(
        INPUT_POSITIONS
    ),

    "output_positions": str(
        OUTPUT_POSITIONS
    ),

    "platform": (
        platform.getName()
    ),

    "particles": n_particles,

    "massive_particles": (
        n_massive
    ),

    "virtual_sites": (
        n_virtual
    ),

    "constraints": (
        system.getNumConstraints()
    ),

    "minimization": {
        "tolerance_kJ_mol_nm": (
            MINIMIZATION_TOLERANCE
        ),

        "max_iterations": (
            MAX_ITERATIONS
        ),

        "constraint_tolerance": (
            CONSTRAINT_TOLERANCE
        ),

        "wall_time_s": (
            elapsed_s
        ),
    },

    "initial": {
        "potential_energy_kJ_mol": (
            initial_energy
        ),

        **initial_force_metrics,

        "constraints": (
            initial_constraint_metrics
        ),
    },

    "final": {
        "potential_energy_kJ_mol": (
            final_energy
        ),

        "energy_change_kJ_mol": (
            energy_change
        ),

        **final_force_metrics,

        "constraints": (
            final_constraint_metrics
        ),

        "yb_energy_kJ_mol": (
            yb_energy
        ),

        "carbon_plane_nm": (
            carbon_plane
        ),

        "minimum_O_height_nm": (
            minimum_O_height
        ),

        "maximum_O_height_nm": (
            maximum_O_height
        ),
    },

    "note": (
        "Energy minimization only. "
        "No heating or molecular dynamics."
    ),
}


OUTPUT_JSON.write_text(
    json.dumps(
        results,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(
    f"Minimized positions:"
    f"\n  {OUTPUT_POSITIONS}"
)

print(
    f"Metadata:"
    f"\n  {OUTPUT_JSON}"
)

print()
print(
    "NO HEATING OR MD HAS BEEN RUN."
)

print()
print(
    "GRAPHENE_OPC_WETTING_YB_MINIMIZATION_COMPLETE"
)

print("=" * 80)
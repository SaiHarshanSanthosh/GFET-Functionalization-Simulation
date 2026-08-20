from pathlib import Path
import json

import numpy as np

from openmm import openmm, unit


# ============================================================
# CONTINUE MINIMIZATION WITH OPENMM MINIMIZATION REPORTER
#
# PURPOSE
# -------
# Start from the first minimized structure and determine:
#
#   - whether L-BFGS actually converges
#   - objective-gradient RMS during minimization
#   - system energy
#   - temporary restraint energy
#   - maximum rigid-constraint error
#   - number of minimizer iterations
#
# NO MD.
# NO VELOCITIES.
# NO TEMPERATURE.
# ============================================================


SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_supported_pme.xml"
)

STARTING_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_minimized_positions_nm.npy"
)

OUTPUT_POSITIONS = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_minimized_converged_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "supported_solvated_minimization_convergence.json"
)


N_SOLUTE = 3820
N_GRAPHENE = 3750
N_CARBON = 1250

TOLERANCE = 10.0

# Give it substantially more room, while retaining a finite
# diagnostic checkpoint.
MAX_ITERATIONS = 5000

CONSTRAINT_TOLERANCE = 1.0e-6


# ============================================================
# LOAD
# ============================================================

for path in [
    SYSTEM_XML,
    STARTING_POSITIONS,
]:
    if not path.exists():
        raise FileNotFoundError(path)


with open(SYSTEM_XML, "r") as f:
    system = openmm.XmlSerializer.deserialize(f.read())


positions = np.load(
    STARTING_POSITIONS
)


n_particles = system.getNumParticles()


if positions.shape != (n_particles, 3):
    raise RuntimeError(
        f"Position mismatch: {positions.shape} vs {n_particles}"
    )


n_water_sites = (
    n_particles
    - N_SOLUTE
)


if n_water_sites % 4 != 0:
    raise RuntimeError(
        "Water site count is not divisible by 4."
    )


n_waters = (
    n_water_sites
    // 4
)


print()
print("=" * 72)
print("MINIMIZATION CONVERGENCE CHECK")
print("=" * 72)

print()
print("Particles:", n_particles)
print("OPC waters:", n_waters)
print("Constraints:", system.getNumConstraints())


# ============================================================
# CONTEXT
# ============================================================

integrator = openmm.VerletIntegrator(
    0.001 * unit.picoseconds
)

integrator.setConstraintTolerance(
    CONSTRAINT_TOLERANCE
)


try:
    platform = openmm.Platform.getPlatformByName(
        "CUDA"
    )
    properties = {
        "Precision": "mixed"
    }

except Exception:
    platform = openmm.Platform.getPlatformByName(
        "CPU"
    )
    properties = {}


context = openmm.Context(
    system,
    integrator,
    platform,
    properties,
)


context.setPositions(
    positions * unit.nanometer
)


context.applyConstraints(
    CONSTRAINT_TOLERANCE
)

context.computeVirtualSites()


print(
    "Platform:",
    platform.getName()
)


# ============================================================
# STARTING STATE
# ============================================================

state = context.getState(
    getEnergy=True,
    getPositions=True,
)


starting_energy = (
    state.getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


print()
print("STARTING ENERGY")
print("---------------")

print(
    f"{starting_energy:.6f} kJ/mol"
)


# ============================================================
# REPORTER
# ============================================================

class ConvergenceReporter(
    openmm.MinimizationReporter
):

    def __init__(self):
        super().__init__()

        self.records = []

        self.calls = 0

        self.last_iteration = None

        self.lbfgs_cycles = 0

    def report(
        self,
        iteration,
        x,
        grad,
        args,
    ):

        self.calls += 1

        # OpenMM resets iteration to zero when it changes
        # the constraint-restraint strength.
        if (
            self.last_iteration is not None
            and iteration < self.last_iteration
        ):
            self.lbfgs_cycles += 1

        self.last_iteration = iteration

        grad_array = np.asarray(
            grad,
            dtype=float,
        )

        gradient_rms = float(
            np.sqrt(
                np.mean(
                    grad_array ** 2
                )
            )
        )

        system_energy = float(
            args[
                "system energy"
            ]
        )

        restraint_energy = float(
            args[
                "restraint energy"
            ]
        )

        restraint_strength = float(
            args[
                "restraint strength"
            ]
        )

        max_constraint_error = float(
            args[
                "max constraint error"
            ]
        )

        record = {
            "report_call":
                self.calls,

            "iteration":
                int(iteration),

            "objective_gradient_rms":
                gradient_rms,

            "system_energy_kj_mol":
                system_energy,

            "restraint_energy_kj_mol":
                restraint_energy,

            "restraint_strength_kj_mol_nm2":
                restraint_strength,

            "max_constraint_error":
                max_constraint_error,
        }

        self.records.append(
            record
        )

        # Avoid thousands of console lines.
        if (
            self.calls == 1
            or self.calls % 100 == 0
            or gradient_rms < 20.0
        ):

            print(
                f"call={self.calls:5d}  "
                f"iter={iteration:5d}  "
                f"grad_RMS={gradient_rms:12.5f}  "
                f"E={system_energy:14.3f}  "
                f"restraint_E={restraint_energy:11.6f}  "
                f"constraint_err={max_constraint_error:.3e}"
            )

        # Never stop early ourselves.
        return False


reporter = ConvergenceReporter()


# ============================================================
# MINIMIZATION
# ============================================================

print()
print("=" * 72)
print("CONTINUING L-BFGS MINIMIZATION")
print("=" * 72)

print()
print(
    f"Tolerance: {TOLERANCE:.3f} kJ/(mol nm)"
)

print(
    "Maximum iterations:",
    MAX_ITERATIONS,
)


openmm.LocalEnergyMinimizer.minimize(
    context,
    tolerance=TOLERANCE,
    maxIterations=MAX_ITERATIONS,
    reporter=reporter,
)


context.computeVirtualSites()


# ============================================================
# FINAL STATE
# ============================================================

state = context.getState(
    getEnergy=True,
    getPositions=True,
    getForces=True,
)


final_energy = (
    state.getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


final_positions = (
    state.getPositions(
        asNumpy=True
    )
    .value_in_unit(
        unit.nanometer
    )
)


forces = (
    state.getForces(
        asNumpy=True
    )
    .value_in_unit(
        unit.kilojoule_per_mole
        / unit.nanometer
    )
)


force_magnitudes = np.linalg.norm(
    forces,
    axis=1,
)


# ============================================================
# CONSTRAINT ERROR CHECK
# ============================================================

max_constraint_relative_error = 0.0
max_constraint_absolute_error_nm = 0.0


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
            final_positions[p1]
            -
            final_positions[p2]
        )
    )

    absolute_error = abs(
        actual_nm
        - target_nm
    )

    relative_error = (
        absolute_error
        / target_nm
    )

    max_constraint_absolute_error_nm = max(
        max_constraint_absolute_error_nm,
        absolute_error,
    )

    max_constraint_relative_error = max(
        max_constraint_relative_error,
        relative_error,
    )


# ============================================================
# GEOMETRY
# ============================================================

oxygen_indices = (
    N_SOLUTE
    +
    np.arange(
        n_waters
    )
    * 4
)


oxygen_z = (
    final_positions[
        oxygen_indices,
        2
    ]
)


graphene_z = (
    final_positions[
        :N_CARBON,
        2
    ]
)


# ============================================================
# REPORTER SUMMARY
# ============================================================

if len(
    reporter.records
) == 0:

    raise RuntimeError(
        "MinimizationReporter received no records."
    )


last_record = (
    reporter.records[-1]
)


minimum_reported_gradient_rms = float(
    min(
        r[
            "objective_gradient_rms"
        ]
        for r in reporter.records
    )
)


print()
print("=" * 72)
print("CONVERGENCE SUMMARY")
print("=" * 72)

print()
print(
    "Reporter calls:",
    reporter.calls,
)

print(
    "Detected L-BFGS restraint cycles:",
    reporter.lbfgs_cycles + 1,
)

print(
    f"Starting energy: "
    f"{starting_energy:.6f} kJ/mol"
)

print(
    f"Final energy:    "
    f"{final_energy:.6f} kJ/mol"
)

print(
    f"Energy change:   "
    f"{final_energy - starting_energy:.6f} kJ/mol"
)


print()
print(
    f"Last objective-gradient RMS: "
    f"{last_record['objective_gradient_rms']:.6f} "
    f"kJ/(mol nm)"
)

print(
    f"Minimum reported gradient RMS: "
    f"{minimum_reported_gradient_rms:.6f} "
    f"kJ/(mol nm)"
)

print(
    f"Reporter max constraint error: "
    f"{last_record['max_constraint_error']:.3e}"
)

print(
    f"Direct max constraint relative error: "
    f"{max_constraint_relative_error:.3e}"
)

print(
    f"Direct max constraint absolute error: "
    f"{max_constraint_absolute_error_nm:.3e} nm"
)


print()
print("RAW PHYSICAL FORCE CHECK")
print("------------------------")

print(
    f"Maximum raw force: "
    f"{np.max(force_magnitudes):.6f} "
    f"kJ/(mol nm)"
)

print(
    f"Graphene max raw force: "
    f"{np.max(force_magnitudes[:N_GRAPHENE]):.6f}"
)

print(
    f"Pyrene-PEG5 max raw force: "
    f"{np.max(force_magnitudes[N_GRAPHENE:N_SOLUTE]):.6f}"
)

print(
    f"Water max raw force: "
    f"{np.max(force_magnitudes[N_SOLUTE:]):.6f}"
)


print()
print("SUPPORTED GEOMETRY")
print("------------------")

print(
    f"Graphene carbon z: "
    f"{graphene_z.min():.6f} "
    f"to "
    f"{graphene_z.max():.6f} nm"
)

print(
    f"Water oxygen z: "
    f"{oxygen_z.min():.6f} "
    f"to "
    f"{oxygen_z.max():.6f} nm"
)


# ============================================================
# CLASSIFICATION
# ============================================================

gradient_pass = (
    minimum_reported_gradient_rms
    <= TOLERANCE
    * 1.05
)


constraint_pass = (
    max_constraint_relative_error
    <= CONSTRAINT_TOLERANCE
    * 2.0
)


energy_pass = (
    final_energy
    <= starting_energy
    + 1.0
)


if (
    gradient_pass
    and
    constraint_pass
    and
    energy_pass
):

    status = (
        "MINIMIZATION_CONVERGENCE_PASS"
    )


elif (
    constraint_pass
    and
    energy_pass
):

    status = (
        "MINIMIZATION_STABLE_BUT_NOT_FULLY_CONVERGED"
    )


else:

    status = (
        "MINIMIZATION_REQUIRES_INSPECTION"
    )


print()
print("=" * 72)
print("RESULT")
print("=" * 72)

print()
print(
    status
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_POSITIONS.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_POSITIONS,
    final_positions,
)


OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {
    "status":
        status,

    "platform":
        platform.getName(),

    "reporter_calls":
        reporter.calls,

    "lbfgs_restraint_cycles":
        reporter.lbfgs_cycles + 1,

    "tolerance_kj_mol_nm":
        TOLERANCE,

    "max_iterations":
        MAX_ITERATIONS,

    "starting_energy_kj_mol":
        float(
            starting_energy
        ),

    "final_energy_kj_mol":
        float(
            final_energy
        ),

    "energy_change_kj_mol":
        float(
            final_energy
            - starting_energy
        ),

    "last_objective_gradient_rms":
        float(
            last_record[
                "objective_gradient_rms"
            ]
        ),

    "minimum_objective_gradient_rms":
        minimum_reported_gradient_rms,

    "last_reported_constraint_error":
        float(
            last_record[
                "max_constraint_error"
            ]
        ),

    "direct_max_constraint_relative_error":
        float(
            max_constraint_relative_error
        ),

    "direct_max_constraint_absolute_error_nm":
        float(
            max_constraint_absolute_error_nm
        ),

    "maximum_raw_force_kj_mol_nm":
        float(
            np.max(
                force_magnitudes
            )
        ),

    "reporter_records":
        reporter.records,

    "md_run":
        False,

    "velocities_assigned":
        False,

    "temperature_applied":
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


print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "Positions:",
    OUTPUT_POSITIONS,
)

print(
    "Metadata:",
    OUTPUT_METADATA,
)


print()
print("=" * 72)
print("MINIMIZATION CONVERGENCE CHECK: COMPLETE")
print("=" * 72)

print()
print(
    "No MD was run."
)

print(
    "No velocities were assigned."
)

print(
    "No temperature was applied."
)

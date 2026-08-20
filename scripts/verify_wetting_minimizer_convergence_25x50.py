from pathlib import Path
import numpy as np
import openmm
from openmm import unit

ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

SYSTEM_XML = (
    ROOT / "parameters" / "combined"
    / "graphene_opc_wetting_cylinder_25x50_yb.xml"
)

INPUT_POSITIONS = (
    ROOT / "parameters" / "combined"
    / "graphene_opc_wetting_cylinder_25x50_yb_minimized_positions_nm.npy"
)

OUTPUT_POSITIONS = (
    ROOT / "parameters" / "combined"
    / "graphene_opc_wetting_cylinder_25x50_yb_minimized_converged_positions_nm.npy"
)


class Reporter(openmm.MinimizationReporter):

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last = None
        self.last_iteration = None

    def report(
        self,
        iteration,
        x,
        grad,
        args,
    ):
        self.calls += 1

        grad = np.asarray(
            grad,
            dtype=float,
        )

        rms_grad = float(
            np.sqrt(
                np.mean(
                    grad * grad
                )
            )
        )

        self.last = {
            "rms_objective_gradient": rms_grad,
            "system_energy": float(
                args["system energy"]
            ),
            "restraint_energy": float(
                args["restraint energy"]
            ),
            "restraint_strength": float(
                args["restraint strength"]
            ),
            "max_constraint_error": float(
                args["max constraint error"]
            ),
        }

        self.last_iteration = int(
            iteration
        )

        if (
            self.calls <= 5
            or
            self.calls % 100 == 0
        ):
            print(
                f"call={self.calls:5d}  "
                f"iter={iteration:5d}  "
                f"RMSgrad={rms_grad:12.6f}  "
                f"E={args['system energy']:14.3f}  "
                f"Erest={args['restraint energy']:12.6f}  "
                f"maxC={args['max constraint error']:.3e}"
            )

        return False


print()
print("=" * 80)
print("WETTING MINIMIZER CONVERGENCE VERIFICATION")
print("=" * 80)


system = openmm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text(
        encoding="utf-8"
    )
)

positions = np.load(
    INPUT_POSITIONS
)


platform = openmm.Platform.getPlatformByName(
    "CUDA"
)

integrator = openmm.VerletIntegrator(
    0.001 * unit.picoseconds
)

integrator.setConstraintTolerance(
    1.0e-6
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
    positions * unit.nanometer
)

context.computeVirtualSites()


reporter = Reporter()


print()
print(
    "Continuing minimization with maxIterations=0 "
    "(run until OpenMM declares convergence)..."
)
print()


openmm.LocalEnergyMinimizer.minimize(
    context,
    tolerance=10.0,
    maxIterations=0,
    reporter=reporter,
)


state = context.getState(
    getPositions=True,
    getEnergy=True,
)


final_positions = np.asarray(
    state.getPositions(
        asNumpy=True
    ).value_in_unit(
        unit.nanometer
    ),
    dtype=float,
)


final_energy = float(
    state.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
)


np.save(
    OUTPUT_POSITIONS,
    final_positions,
)


print()
print("=" * 80)
print("FINAL MINIMIZER REPORT")
print("=" * 80)

print(
    f"Reporter calls:          "
    f"{reporter.calls}"
)

print(
    f"Last L-BFGS iteration:   "
    f"{reporter.last_iteration}"
)

if reporter.last is not None:

    print(
        f"Last RMS objective grad: "
        f"{reporter.last['rms_objective_gradient']:.6f} "
        f"kJ/mol/nm"
    )

    print(
        f"Last restraint energy:   "
        f"{reporter.last['restraint_energy']:.9f} kJ/mol"
    )

    print(
        f"Last max constraint err: "
        f"{reporter.last['max_constraint_error']:.3e}"
    )

print(
    f"Final potential energy:  "
    f"{final_energy:.6f} kJ/mol"
)

print()
print(
    f"Saved:"
    f"\n  {OUTPUT_POSITIONS}"
)

print()
print(
    "OPENMM_MINIMIZER_RETURNED_WITH_MAXITERATIONS_ZERO"
)

print("=" * 80)
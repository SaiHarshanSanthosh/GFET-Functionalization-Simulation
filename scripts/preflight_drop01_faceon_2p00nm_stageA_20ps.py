from pathlib import Path

DROP = Path(
    "scripts/run_drop01_faceon_2p00nm_stageA_20ps.py"
)

source = DROP.read_text()

marker = """# ============================================================
# 17. RUN DROP-01 STAGE A: 0 -> 20 ps
# ============================================================
"""

if source.count(marker) != 1:
    raise RuntimeError(
        "Could not uniquely locate the Drop-01 MD boundary."
    )

prefix = source.split(marker, 1)[0]


# ============================================================
# SAFETY GATES
# ============================================================

if "integrator.step(" in prefix:
    raise RuntimeError(
        "SAFETY FAILURE: integrator.step() occurs before MD boundary."
    )

if prefix.count(
    "context.setVelocitiesToTemperature("
) != 1:

    raise RuntimeError(
        "SAFETY FAILURE: expected exactly one t=0 velocity initialization."
    )

if "context.setTime(\n    0.0" not in prefix:
    raise RuntimeError(
        "SAFETY FAILURE: Drop-01 does not begin at t=0."
    )

if "LocalEnergyMinimizer.minimize(" in prefix:
    raise RuntimeError(
        "SAFETY FAILURE: minimization exists before Drop-01 release."
    )

if "simulation.minimizeEnergy(" in prefix:
    raise RuntimeError(
        "SAFETY FAILURE: minimization exists before Drop-01 release."
    )

if "DCDFile(" in prefix:
    raise RuntimeError(
        "SAFETY FAILURE: trajectory output begins before MD boundary."
    )


print()
print("=" * 78)
print("STATIC DROP-01 ZERO-STEP PREFLIGHT")
print("=" * 78)
print("Initial velocities WILL be generated with the locked Drop-01 seed.")
print("No molecular-dynamics steps will be executed.")
print("No DCD trajectory will be written.")
print()


exec(
    compile(
        prefix,
        str(DROP),
        "exec",
    ),
    {
        "__name__": "__main__",
        "__file__": str(DROP),
    },
)


print()
print("=" * 78)
print("STATIC DROP-01 PREFLIGHT COMPLETE")
print("=" * 78)
print("MD steps executed: 0")
print("Drop-01 ligand has NOT been released.")

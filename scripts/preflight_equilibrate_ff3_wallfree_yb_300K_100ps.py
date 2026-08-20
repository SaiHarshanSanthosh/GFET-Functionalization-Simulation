from pathlib import Path

EQUIL = Path(
    "scripts/equilibrate_ff3_wallfree_yb_300K_100ps.py"
)

source = EQUIL.read_text()

marker = """# ============================================================
# 17. RUN 100 ps EQUILIBRATION AT 300 K
# ============================================================
"""

if source.count(marker) != 1:
    raise RuntimeError(
        "Could not uniquely locate the equilibration MD boundary."
    )

prefix = source.split(marker, 1)[0]

if "integrator.step(" in prefix:
    raise RuntimeError(
        "SAFETY FAILURE: integrator.step() occurs before boundary."
    )

if "setVelocitiesToTemperature" in prefix:
    raise RuntimeError(
        "SAFETY FAILURE: velocities would be re-randomized."
    )

if "context.setVelocities(" not in prefix:
    raise RuntimeError(
        "Saved 300 K velocities are not loaded."
    )

print()
print("=" * 78)
print("STATIC FF-3 300 K EQUILIBRATION PREFLIGHT")
print("=" * 78)
print("No molecular-dynamics steps will be executed.")
print()

exec(
    compile(
        prefix,
        str(EQUIL),
        "exec",
    ),
    {
        "__name__": "__main__",
        "__file__": str(EQUIL),
    },
)

print()
print("=" * 78)
print("STATIC EQUILIBRATION PREFLIGHT COMPLETE")
print("=" * 78)
print("No molecular-dynamics steps were executed.")

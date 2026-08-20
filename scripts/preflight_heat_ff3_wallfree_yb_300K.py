from pathlib import Path

HEATER = Path(
    "scripts/heat_ff3_wallfree_yb_300K.py"
)

source = HEATER.read_text()

marker = """# ============================================================
# 17. RUN 20 ps AT 300 K
# ============================================================
"""

if source.count(marker) != 1:
    raise RuntimeError(
        "Could not uniquely locate the 300 K MD boundary."
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
        "Saved continuation velocities are not loaded."
    )

print()
print("=" * 78)
print("STATIC FF-3 300 K CONTINUATION PREFLIGHT")
print("=" * 78)
print("No molecular-dynamics steps will be executed.")
print()

exec(
    compile(
        prefix,
        str(HEATER),
        "exec",
    ),
    {
        "__name__": "__main__",
        "__file__": str(HEATER),
    },
)

print()
print("=" * 78)
print("STATIC 300 K PREFLIGHT COMPLETE")
print("=" * 78)
print("No molecular-dynamics steps were executed.")

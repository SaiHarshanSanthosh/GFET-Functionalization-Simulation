from pathlib import Path

HEATER = Path(
    "scripts/heat_ff3_wallfree_yb_50K.py"
)

source = HEATER.read_text()

marker = """# ============================================================
# 17. RUN 20 ps AT 50 K
# ============================================================
"""

if source.count(marker) != 1:
    raise RuntimeError(
        "Could not uniquely locate the MD-run boundary."
    )

prefix = source.split(
    marker,
    1,
)[0]

if "integrator.step(" in prefix:
    raise RuntimeError(
        "Safety failure: integrator.step() appears "
        "before the intended preflight boundary."
    )

if "setVelocitiesToTemperature" not in prefix:
    raise RuntimeError(
        "Expected initial velocity assignment "
        "was not found in the preflight section."
    )

print()
print("=" * 78)
print("STATIC FF-3 50 K HEATING PREFLIGHT")
print("=" * 78)
print(
    "Executing heater only through the initial-state audit."
)
print(
    "No integrator.step() call is present in the executed code."
)
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
print("STATIC PREFLIGHT COMPLETE")
print("=" * 78)
print("No molecular-dynamics steps were executed.")

from pathlib import Path
import struct
import math

RUNNER = Path(
    "scripts/"
    "run_drop01_faceon_2p00nm_fixedz0_continue_20ps_to_5ns.py"
)

source = RUNNER.read_text()

boundary = """# ============================================================
# 17. CONTINUE DROP-01: 20 ps -> 5000 ps
# ============================================================
"""

if source.count(boundary) != 1:
    raise RuntimeError(
        "Could not uniquely identify continuation MD boundary."
    )

prefix = source.split(
    boundary,
    1,
)[0]


# ============================================================
# STATIC ZERO-STEP SAFETY CHECK
# ============================================================

required = [
    "context.loadCheckpoint(",
    "checkpoint_time_ps",
    "TOTAL_TIME_PS = 4980.0",
    "REPORT_INTERVAL_PS = 1.0",
]

for token in required:

    if token not in prefix:
        raise RuntimeError(
            f"Required preflight token absent: {token}"
        )


for forbidden in [
    "integrator.step(",
    "DCDFile(",
    'open(\n    OUTPUT_DCD,\n    "r+b"',
    "context.setPositions(",
    "context.setVelocitiesToTemperature(",
    "context.setTime(",
    "context.applyConstraints(",
    "context.applyVelocityConstraints(",
    "LocalEnergyMinimizer.minimize(",
    "simulation.minimizeEnergy(",
]:

    if forbidden in prefix:
        raise RuntimeError(
            "ZERO-STEP SAFETY FAILURE: "
            f"{forbidden}"
        )


print()
print("=" * 78)
print("DROP-01 20 ps -> 5 ns ZERO-STEP CONTINUATION PREFLIGHT")
print("=" * 78)
print("Checkpoint will be loaded.")
print("MD steps executed: 0")
print("DCD opened for writing: NO")
print("Coordinates modified: NO")
print("Velocities regenerated: NO")
print()


# Execute only through checkpoint validation,
# stopping before the continuation/run section.
ns = {
    "__name__": "__main__",
    "__file__": str(RUNNER),
}

exec(
    compile(
        prefix,
        str(RUNNER),
        "exec",
    ),
    ns,
)


# ============================================================
# CHECKPOINT STATE ASSERTIONS
# ============================================================

checkpoint_time_ps = float(
    ns[
        "checkpoint_time_ps"
    ]
)

position_diff = float(
    ns[
        "max_checkpoint_position_difference_nm"
    ]
)

velocity_diff = float(
    ns[
        "max_checkpoint_velocity_difference_nm_per_ps"
    ]
)

metrics = ns[
    "initial_metrics"
]


if abs(
    checkpoint_time_ps
    -
    20.0
) > 1.0e-6:

    raise RuntimeError(
        "Checkpoint time is not exactly 20 ps."
    )


if position_diff > 1.0e-6:

    raise RuntimeError(
        "Checkpoint position mismatch."
    )


if velocity_diff > 1.0e-6:

    raise RuntimeError(
        "Checkpoint velocity mismatch."
    )


if (
    metrics[
        "water_below_graphene"
    ]
    !=
    0
):

    raise RuntimeError(
        "Checkpoint contains water below graphene."
    )


for key in [
    "temperature_K",
    "potential_kJ_per_mol",
    "pyrene_height_above_graphene_nm",
    "pyrene_tilt_from_graphene_plane_deg",
]:

    if not math.isfinite(
        float(
            metrics[key]
        )
    ):

        raise RuntimeError(
            f"Non-finite checkpoint metric: {key}"
        )


# ============================================================
# READ-ONLY MASTER DCD HEADER CHECK
# ============================================================

dcd_path = ns[
    "OUTPUT_DCD"
]

with open(
    dcd_path,
    "rb",
) as f:

    magic = f.read(
        8
    )

    if (
        len(magic) != 8
        or
        magic[4:8] != b"CORD"
    ):

        raise RuntimeError(
            "Master DCD header invalid."
        )

    model_count = struct.unpack(
        "<i",
        f.read(4),
    )[0]

    f.seek(
        92
    )

    comments_bytes = struct.unpack(
        "<i",
        f.read(4),
    )[0]

    f.seek(
        104
        +
        comments_bytes
    )

    dcd_atom_count = struct.unpack(
        "<i",
        f.read(4),
    )[0]


if model_count != 21:

    raise RuntimeError(
        "Master DCD should contain exactly "
        f"21 models; found {model_count}."
    )


if dcd_atom_count != 1320:

    raise RuntimeError(
        "Master DCD should contain exactly "
        f"1320 atoms; found {dcd_atom_count}."
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("CONTINUATION CHECKPOINT VERIFICATION")
print("=" * 78)

print(
    f"Checkpoint time: "
    f"{checkpoint_time_ps:.12f} ps"
)

print(
    "Max position difference:",
    f"{position_diff:.3e} nm"
)

print(
    "Max velocity difference:",
    f"{velocity_diff:.3e} nm/ps"
)

print(
    f"Temperature: "
    f"{metrics['temperature_K']:.3f} K"
)

print(
    f"Potential energy: "
    f"{metrics['potential_kJ_per_mol']:.3f} kJ/mol"
)

print(
    f"Pyrene height: "
    f"{metrics['pyrene_height_above_graphene_nm']:.6f} nm"
)

print(
    f"Pyrene tilt: "
    f"{metrics['pyrene_tilt_from_graphene_plane_deg']:.3f} deg"
)

print(
    "Waters below graphene:",
    metrics[
        "water_below_graphene"
    ],
)

print()
print("MASTER DCD")
print("----------")

print(
    "Existing models:",
    model_count
)

print(
    "Atoms/model:",
    dcd_atom_count
)

print(
    "Expected next frame:",
    "21 ps"
)


print()
print("=" * 78)
print("DROP01 CONTINUATION ZERO-STEP PREFLIGHT: PASS")
print("=" * 78)
print("MD steps executed: 0")
print("Master DCD modified: NO")
print("Ready for exact 20 ps -> 5 ns continuation.")

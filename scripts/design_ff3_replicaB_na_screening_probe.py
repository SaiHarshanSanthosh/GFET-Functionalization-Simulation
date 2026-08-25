from pathlib import Path
import json
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "ff3_explicit_dpbs_opc_yb_replicaB_"
      "production_300K_10ns_positions_nm.npy"
)

ASSEMBLY = (
    ROOT
    / "analysis"
    / "ff3_explicit_dpbs_opc_yb_replicaB_assembly_check.json"
)

OUT = (
    ROOT
    / "analysis"
    / "ff3_explicit_dpbs_opc_yb_replicaB_"
      "na_screening_probe_design.json"
)


N_GRAPHENE_CARBONS = 1250

# Same physical Na+ particle will be used in every branch.
# Picking it near the middle of our target range minimizes
# the amount of steered motion needed during branch setup.
SELECTION_TARGET_GAP_NM = 2.0

TARGET_HEIGHTS_NM = [
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
]

# Initial pilot protocol per branch.
PULL_TIME_PS = 500.0
SETTLE_TIME_PS = 500.0
PRODUCTION_TIME_PS = 1000.0

# Harmonic z restraint.
RESTRAINT_K_KJ_MOL_NM2 = 500.0


for path in [
    POSITIONS,
    ASSEMBLY,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}"
        )


positions = np.load(
    POSITIONS
)

assembly = json.loads(
    ASSEMBLY.read_text(
        encoding="utf-8"
    )
)


na_start = int(
    assembly[
        "na_particle_start_index"
    ]
)

na_stop = int(
    assembly[
        "na_particle_end_index_exclusive"
    ]
)


n_na = (
    na_stop
    -
    na_start
)


if n_na != int(
    assembly["na_ions"]
):
    raise RuntimeError(
        "Na particle count mismatch."
    )


graphene_z = (
    positions[
        :N_GRAPHENE_CARBONS,
        2
    ]
)


graphene_mean_z = float(
    np.mean(graphene_z)
)

graphene_min_z = float(
    np.min(graphene_z)
)

graphene_max_z = float(
    np.max(graphene_z)
)


na_positions = (
    positions[
        na_start:
        na_stop
    ]
)


# Use the same convention as our previous interfacial
# analyses for reporting candidate locations:
# gap above frame's maximum graphene-carbon z.
na_gap_from_max = (
    na_positions[:, 2]
    -
    graphene_max_z
)


# Select one existing Na+ already near 2 nm.
local_index = int(
    np.argmin(
        np.abs(
            na_gap_from_max
            -
            SELECTION_TARGET_GAP_NM
        )
    )
)


particle_index = (
    na_start
    +
    local_index
)


selected_xyz = (
    na_positions[
        local_index
    ]
)


selected_gap_max = float(
    na_gap_from_max[
        local_index
    ]
)

selected_gap_mean = float(
    selected_xyz[2]
    -
    graphene_mean_z
)


# Branch restraint will use graphene mean-z as the
# dynamic reference through a CustomCentroidBondForce.
#
# This means target height is:
#
#     z_probe - z_graphene_centroid = target_height
#
# Later analysis can still report actual mean heights
# relative to the framewise maximum graphene-carbon z.
branches = []


for target in TARGET_HEIGHTS_NM:

    branches.append(
        {
            "name":
                (
                    "bulk_reference"
                    if target == 5.0
                    else
                    f"probe_{target:.1f}nm"
                    .replace(".", "p")
                ),

            "target_height_above_graphene_mean_nm":
                target,

            "probe_particle_index":
                particle_index,

            "probe_na_local_index":
                local_index,

            "pull_time_ps":
                PULL_TIME_PS,

            "settle_time_ps":
                SETTLE_TIME_PS,

            "production_time_ps":
                PRODUCTION_TIME_PS,

            "restraint_k_kJ_mol_nm2":
                RESTRAINT_K_KJ_MOL_NM2,
        }
    )


summary = {
    "status":
        "PASS",

    "purpose":
        (
            "Direct molecular screening experiment using "
            "one existing Na+ ion as a restrained +1e probe. "
            "No particles or net charge are added."
        ),

    "source_state":
        str(POSITIONS),

    "selection_target_gap_from_graphene_max_nm":
        SELECTION_TARGET_GAP_NM,

    "graphene_final_frame_nm":
        {
            "mean_z":
                graphene_mean_z,

            "min_z":
                graphene_min_z,

            "max_z":
                graphene_max_z,
        },

    "probe":
        {
            "na_local_index":
                local_index,

            "system_particle_index":
                particle_index,

            "initial_xyz_nm":
                selected_xyz.tolist(),

            "initial_gap_from_graphene_max_nm":
                selected_gap_max,

            "initial_gap_from_graphene_mean_nm":
                selected_gap_mean,
        },

    "restraint_definition":
        (
            "Harmonic restraint on "
            "(z_probe - z_graphene_centroid - target_height), "
            "implemented with CustomCentroidBondForce."
        ),

    "targets_nm":
        TARGET_HEIGHTS_NM,

    "protocol":
        {
            "pull_ps":
                PULL_TIME_PS,

            "settle_ps":
                SETTLE_TIME_PS,

            "initial_production_ps":
                PRODUCTION_TIME_PS,

            "water_save_interval_ps":
                10.0,

            "compact_save_interval_ps":
                1.0,

            "note":
                (
                    "Initial pilot only. Extend production "
                    "only if block convergence requires it."
                ),
        },

    "branches":
        branches,
}


OUT.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 76)
print("REPLICA B Na+ SCREENING-PROBE DESIGN")
print("=" * 76)

print()
print(
    f"Na+ count: {n_na}"
)

print(
    f"Graphene mean z: "
    f"{graphene_mean_z:.6f} nm"
)

print(
    f"Graphene max z:  "
    f"{graphene_max_z:.6f} nm"
)

print()
print("Selected existing Na+ probe:")

print(
    f"  Na local index:      "
    f"{local_index}"
)

print(
    f"  System particle:     "
    f"{particle_index}"
)

print(
    f"  xyz:                 "
    f"{selected_xyz}"
)

print(
    f"  gap from max C z:    "
    f"{selected_gap_max:.6f} nm"
)

print(
    f"  gap from mean C z:   "
    f"{selected_gap_mean:.6f} nm"
)

print()
print("Planned branches:")

for branch in branches:

    print(
        f"  {branch['name']:18s} "
        f"target="
        f"{branch['target_height_above_graphene_mean_nm']:.1f} nm"
    )

print()
print(
    "No charge or particle is added."
)

print(
    "Every branch uses the SAME Na+ particle."
)

print()
print("Saved:", OUT)

print()
print(
    "Na+ SCREENING-PROBE DESIGN: PASS"
)

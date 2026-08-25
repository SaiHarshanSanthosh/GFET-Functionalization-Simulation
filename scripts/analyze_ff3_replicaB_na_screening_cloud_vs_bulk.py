from pathlib import Path
import csv
import json
import math

import numpy as np
import matplotlib.pyplot as plt


A = Path("analysis")

INPUT_CSV = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_cloud.csv"
)

INPUT_JSON = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_cloud.json"
)

OUT_CSV = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_cloud_vs_bulk.csv"
)

OUT_JSON = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_cloud_vs_bulk.json"
)

OUT_PNG = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_cloud_vs_bulk.png"
)


BRANCHES = [
    "probe_0p5nm",
    "probe_1p0nm",
    "probe_1p5nm",
    "probe_2p0nm",
    "probe_3p0nm",
]

TARGETS = {
    "probe_0p5nm": 0.5,
    "probe_1p0nm": 1.0,
    "probe_1p5nm": 1.5,
    "probe_2p0nm": 2.0,
    "probe_3p0nm": 3.0,
}

REFERENCE = "bulk_reference"

REPORT_RADII = [
    0.3,
    0.5,
    0.7,
    1.0,
    1.5,
    2.0,
]


# ============================================================
# LOAD FULL MEAN CURVES
# ============================================================

rows = []

with INPUT_CSV.open(
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for r in reader:

        rows.append({
            "branch":
                r["branch"],

            "target_nm":
                float(r["target_nm"]),

            "radius_nm":
                float(r["radius_nm"]),

            "environment_charge_e":
                float(
                    r[
                        "environment_charge_e"
                    ]
                ),

            "effective_enclosed_charge_e":
                float(
                    r[
                        "effective_enclosed_charge_e"
                    ]
                ),

            "water_charge_e":
                float(
                    r[
                        "water_charge_e"
                    ]
                ),

            "ion_charge_e":
                float(
                    r[
                        "ion_charge_e"
                    ]
                ),

            "ligand_charge_e":
                float(
                    r[
                        "ligand_charge_e"
                    ]
                ),
        })


def branch_rows(name):

    x = [
        r
        for r in rows
        if r["branch"] == name
    ]

    x.sort(
        key=lambda z: z["radius_nm"]
    )

    return x


ref_rows = branch_rows(
    REFERENCE
)

ref_by_radius = {
    round(
        r["radius_nm"],
        6
    ): r

    for r in ref_rows
}


# ============================================================
# LOAD SAVED BLOCK SD VALUES
# ============================================================

summary_json = json.loads(
    INPUT_JSON.read_text(
        encoding="utf-8"
    )
)


summary_by_branch = {
    x["branch"]: x
    for x in summary_json["summary"]
}


# ============================================================
# BULK-SUBTRACTED CURVES
# ============================================================

delta_rows = []

for branch in BRANCHES:

    for r in branch_rows(branch):

        key = round(
            r["radius_nm"],
            6
        )

        ref = ref_by_radius[key]

        delta_rows.append({
            "branch":
                branch,

            "target_nm":
                TARGETS[branch],

            "radius_nm":
                r["radius_nm"],

            "delta_environment_charge_e":
                (
                    r["environment_charge_e"]
                    -
                    ref[
                        "environment_charge_e"
                    ]
                ),

            "delta_water_charge_e":
                (
                    r["water_charge_e"]
                    -
                    ref[
                        "water_charge_e"
                    ]
                ),

            "delta_ion_charge_e":
                (
                    r["ion_charge_e"]
                    -
                    ref[
                        "ion_charge_e"
                    ]
                ),

            "delta_ligand_charge_e":
                (
                    r["ligand_charge_e"]
                    -
                    ref[
                        "ligand_charge_e"
                    ]
                ),
        })


# ============================================================
# REPORT TABLE
# ============================================================

print("=" * 88)
print("Na+ SCREENING CLOUD — DIFFERENCE FROM 5 nm BULK REFERENCE")
print("=" * 88)

print()
print(
    "Delta Q = Q_environment(near graphene) "
    "- Q_environment(5 nm reference)"
)

print()
print(
    "  Delta Q > 0 : less negative screening than bulk"
)

print(
    "  Delta Q < 0 : more negative screening than bulk"
)

print(
    "  Delta Q ~ 0 : bulk-like screening"
)

print()


header = (
    f"{'branch':16s} "
    f"{'radius':>7s} "
    f"{'deltaQ':>10s} "
    f"{'approx SD':>11s} "
    f"{'water':>10s} "
    f"{'ions':>10s}"
)

print(header)
print("-" * len(header))


report = []


for branch in BRANCHES:

    bsum = summary_by_branch[
        branch
    ]

    rsum = summary_by_branch[
        REFERENCE
    ]


    for radius in REPORT_RADII:

        nearest = min(
            [
                x
                for x in delta_rows
                if x["branch"] == branch
            ],
            key=lambda x:
                abs(
                    x["radius_nm"]
                    -
                    radius
                )
        )

        binfo = (
            bsum["radii_nm"][
                str(radius)
            ]
        )

        rinfo = (
            rsum["radii_nm"][
                str(radius)
            ]
        )

        # Independent trajectories:
        # combine block-scale fluctuations in quadrature.
        approx_sd = math.sqrt(
            float(
                binfo[
                    "block_sd_e"
                ]
            ) ** 2
            +
            float(
                rinfo[
                    "block_sd_e"
                ]
            ) ** 2
        )


        item = {
            "branch":
                branch,

            "target_nm":
                TARGETS[branch],

            "radius_nm":
                radius,

            "delta_environment_charge_e":
                nearest[
                    "delta_environment_charge_e"
                ],

            "approx_combined_block_sd_e":
                approx_sd,

            "delta_water_charge_e":
                nearest[
                    "delta_water_charge_e"
                ],

            "delta_ion_charge_e":
                nearest[
                    "delta_ion_charge_e"
                ],
        }

        report.append(
            item
        )


        print(
            f"{branch:16s} "
            f"{radius:7.1f} "
            f"{item['delta_environment_charge_e']:+10.3f} "
            f"{approx_sd:11.3f} "
            f"{item['delta_water_charge_e']:+10.3f} "
            f"{item['delta_ion_charge_e']:+10.3f}"
        )


# ============================================================
# SIMPLE "BULK-LIKE" CHECK
# ============================================================

print()
print("=" * 88)
print("SUMMARY AT 0.7 nm")
print("=" * 88)
print()

for x in report:

    if abs(
        x["radius_nm"]
        -
        0.7
    ) > 1e-9:

        continue

    delta = x[
        "delta_environment_charge_e"
    ]

    sd = x[
        "approx_combined_block_sd_e"
    ]

    ratio = (
        abs(delta)
        /
        sd
        if sd > 0
        else float("inf")
    )

    print(
        f"{x['target_nm']:3.1f} nm probe: "
        f"Delta Q = {delta:+.3f} e, "
        f"block-scale |Delta|/SD = {ratio:.2f}"
    )


# ============================================================
# SAVE FULL CSV
# ============================================================

with OUT_CSV.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "branch",
        "target_nm",
        "radius_nm",
        "delta_environment_charge_e",
        "delta_water_charge_e",
        "delta_ion_charge_e",
        "delta_ligand_charge_e",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        delta_rows
    )


# ============================================================
# SAVE JSON
# ============================================================

OUT_JSON.write_text(
    json.dumps(
        {
            "reference_branch":
                REFERENCE,

            "definition":
                (
                    "Delta Q = environment charge "
                    "for near-graphene probe minus "
                    "environment charge for 5 nm "
                    "bulk-reference probe."
                ),

            "interpretation":
                {
                    "positive":
                        (
                            "less negative local "
                            "screening than bulk"
                        ),

                    "negative":
                        (
                            "more negative local "
                            "screening than bulk"
                        ),

                    "zero":
                        "bulk-like screening",
                },

            "report":
                report,
        },
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.0, 5.2)
)


for branch in BRANCHES:

    br = [
        x
        for x in delta_rows
        if x["branch"] == branch
    ]

    rr = np.array([
        x["radius_nm"]
        for x in br
    ])

    dq = np.array([
        x[
            "delta_environment_charge_e"
        ]
        for x in br
    ])

    ax.plot(
        rr,
        dq,
        label=(
            f"{TARGETS[branch]:g} nm"
        )
    )


ax.axhline(
    0.0,
    linewidth=1
)

ax.set_xlim(
    0.0,
    2.0
)

ax.set_xlabel(
    "Distance from restrained Na+ (nm)"
)

ax.set_ylabel(
    "Difference from 5 nm bulk screening cloud (e)"
)

ax.set_title(
    "Graphene-associated change in Na+ screening cloud"
)

ax.legend(
    title="Probe height"
)

fig.tight_layout()

fig.savefig(
    OUT_PNG,
    dpi=200
)

plt.close(fig)


print()
print(f"CSV:  {OUT_CSV}")
print(f"JSON: {OUT_JSON}")
print(f"Plot: {OUT_PNG}")
print()
print("BULK-SUBTRACTED SCREENING ANALYSIS COMPLETE")

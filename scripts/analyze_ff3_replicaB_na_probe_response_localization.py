from pathlib import Path
import runpy
import json
import numpy as np
import matplotlib.pyplot as plt

A = Path("analysis")

BASE = Path(
    "scripts/"
    "analyze_ff3_replicaB_na_probe_planar_interfacial_response.py"
)

OUT_CSV = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "na_probe_response_localization.csv"
)

OUT_JSON = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "na_probe_response_localization.json"
)

OUT_PNG = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "na_probe_response_localization.png"
)

# Reuse the already validated trajectory parsing / OPC reconstruction.
g = runpy.run_path(str(BASE))

data = g["data"]
blocks = g["blocks"]
CENTERS = g["CENTERS"]
DZ = g["DZ"]
AREA = g["AREA_NM2"]
BRANCHES = g["BRANCHES"]
REFERENCE = g["REFERENCE"]

ref = data[REFERENCE]

ZTOTAL = 4.0

def delta_rho(branch, key, ids=None):

    b = data[branch][key]
    r = ref[key]

    if ids is None:
        bm = np.mean(b, axis=0)
        rm = np.mean(r, axis=0)
    else:
        bm = np.mean(b[ids], axis=0)
        rm = np.mean(r[ids], axis=0)

    return (
        (bm - rm)
        /
        (AREA * DZ)
    )


def magnitude(profile, zmax):

    mask = (
        (CENTERS >= 0.0)
        &
        (CENTERS < zmax)
    )

    return float(
        np.sum(
            np.abs(profile[mask])
        )
        *
        DZ
    )


def localization_distance(profile, fraction):

    mask = (
        (CENTERS >= 0.0)
        &
        (CENTERS < ZTOTAL)
    )

    z = CENTERS[mask]
    weight = (
        np.abs(profile[mask])
        *
        DZ
    )

    total = float(
        np.sum(weight)
    )

    if total <= 0:
        return float("nan")

    cumulative = np.cumsum(weight)

    j = int(
        np.searchsorted(
            cumulative,
            fraction * total
        )
    )

    j = min(
        j,
        len(z) - 1
    )

    return float(z[j])


results = []

print()
print("=" * 108)
print("GRAPHENE-REFERENCED ELECTROLYTE RESPONSE LOCALIZATION")
print("=" * 108)

print()
print(
    "R = integral of |Delta rho_environment(z)| dz"
)

print(
    "This measures rearrangement magnitude even when "
    "positive and negative layers cancel in net charge."
)

print()

print(
    f"{'branch':16s} "
    f"{'R 0-1':>10s} "
    f"{'R 0-2':>10s} "
    f"{'block SD':>10s} "
    f"{'% <1nm':>9s} "
    f"{'% <2nm':>9s} "
    f"{'z50':>7s} "
    f"{'z90':>7s} "
    f"{'water R2':>10s} "
    f"{'ion R2':>10s}"
)

print("-" * 108)

for branch, target in BRANCHES:

    if branch == REFERENCE:
        continue

    env = delta_rho(
        branch,
        "environment_hist_e"
    )

    water = delta_rho(
        branch,
        "water_hist_e"
    )

    ions = delta_rho(
        branch,
        "ion_hist_e"
    )

    r1 = magnitude(
        env,
        1.0
    )

    r2 = magnitude(
        env,
        2.0
    )

    r4 = magnitude(
        env,
        ZTOTAL
    )

    water_r2 = magnitude(
        water,
        2.0
    )

    ion_r2 = magnitude(
        ions,
        2.0
    )

    frac1 = (
        r1 / r4
        if r4 > 0
        else float("nan")
    )

    frac2 = (
        r2 / r4
        if r4 > 0
        else float("nan")
    )

    z50 = localization_distance(
        env,
        0.50
    )

    z90 = localization_distance(
        env,
        0.90
    )

    block_r2 = np.array([
        magnitude(
            delta_rho(
                branch,
                "environment_hist_e",
                ids
            ),
            2.0
        )
        for ids in blocks
    ])

    block_sd = float(
        np.std(
            block_r2,
            ddof=1
        )
    )

    result = {
        "branch": branch,
        "target_height_nm": target,
        "redistribution_0to1_e_per_nm2": r1,
        "redistribution_0to2_e_per_nm2": r2,
        "redistribution_0to4_e_per_nm2": r4,
        "redistribution_0to2_block_sd_e_per_nm2": block_sd,
        "fraction_within_1nm": frac1,
        "fraction_within_2nm": frac2,
        "z50_nm": z50,
        "z90_nm": z90,
        "water_redistribution_0to2_e_per_nm2": water_r2,
        "ion_redistribution_0to2_e_per_nm2": ion_r2,
        "block_0to2_e_per_nm2": block_r2.tolist(),
    }

    results.append(result)

    print(
        f"{branch:16s} "
        f"{r1:10.4f} "
        f"{r2:10.4f} "
        f"{block_sd:10.4f} "
        f"{100*frac1:8.1f}% "
        f"{100*frac2:8.1f}% "
        f"{z50:7.3f} "
        f"{z90:7.3f} "
        f"{water_r2:10.4f} "
        f"{ion_r2:10.4f}"
    )


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

header = (
    "branch,target_height_nm,"
    "redistribution_0to1_e_per_nm2,"
    "redistribution_0to2_e_per_nm2,"
    "redistribution_0to4_e_per_nm2,"
    "redistribution_0to2_block_sd_e_per_nm2,"
    "fraction_within_1nm,"
    "fraction_within_2nm,"
    "z50_nm,z90_nm,"
    "water_redistribution_0to2_e_per_nm2,"
    "ion_redistribution_0to2_e_per_nm2"
)

lines = [header]

for x in results:
    lines.append(
        ",".join([
            x["branch"],
            f"{x['target_height_nm']:.6f}",
            f"{x['redistribution_0to1_e_per_nm2']:.12f}",
            f"{x['redistribution_0to2_e_per_nm2']:.12f}",
            f"{x['redistribution_0to4_e_per_nm2']:.12f}",
            f"{x['redistribution_0to2_block_sd_e_per_nm2']:.12f}",
            f"{x['fraction_within_1nm']:.12f}",
            f"{x['fraction_within_2nm']:.12f}",
            f"{x['z50_nm']:.9f}",
            f"{x['z90_nm']:.9f}",
            f"{x['water_redistribution_0to2_e_per_nm2']:.12f}",
            f"{x['ion_redistribution_0to2_e_per_nm2']:.12f}",
        ])
    )

OUT_CSV.write_text(
    "\n".join(lines) + "\n"
)


OUT_JSON.write_text(
    json.dumps(
        {
            "definition": (
                "Localization of absolute graphene-referenced "
                "environmental charge redistribution."
            ),
            "reference_branch": REFERENCE,
            "normalization_height_nm": ZTOTAL,
            "results": results,
        },
        indent=2
    )
)


# ------------------------------------------------------------
# CUMULATIVE LOCALIZATION PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8.0, 5.2)
)

for branch, target in BRANCHES:

    if branch == REFERENCE:
        continue

    env = delta_rho(
        branch,
        "environment_hist_e"
    )

    mask = (
        (CENTERS >= 0.0)
        &
        (CENTERS < ZTOTAL)
    )

    z = CENTERS[mask]

    weight = (
        np.abs(env[mask])
        *
        DZ
    )

    total = np.sum(weight)

    if total > 0:
        cumulative = (
            np.cumsum(weight)
            /
            total
        )
    else:
        cumulative = (
            np.zeros_like(weight)
        )

    ax.plot(
        z,
        cumulative,
        label=f"{target:g} nm"
    )

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1
)

ax.axhline(
    0.9,
    linestyle="--",
    linewidth=1
)

ax.axvline(
    1.0,
    linestyle="--",
    linewidth=1
)

ax.axvline(
    2.0,
    linestyle="--",
    linewidth=1
)

ax.set_xlim(
    0.0,
    4.0
)

ax.set_ylim(
    0.0,
    1.02
)

ax.set_xlabel(
    "Height above graphene (nm)"
)

ax.set_ylabel(
    "Fraction of observed charge redistribution"
)

ax.set_title(
    "Spatial localization of electrolyte response"
)

ax.legend(
    title="Probe target"
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
print("LOCALIZATION ANALYSIS COMPLETE")

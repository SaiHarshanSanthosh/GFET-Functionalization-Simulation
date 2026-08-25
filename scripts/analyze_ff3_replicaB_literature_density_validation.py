from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FILES
# ============================================================

A = Path("analysis")

COMPACT_NPY = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "production_300K_10ns_screening_trajectory_float32.npy"
)

COMPACT_JSON = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "production_300K_10ns_screening_trajectory.json"
)

WATER_NPY = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "production_300K_10ns_opc_ohh_10ps_float32.npy"
)

OUT_CSV = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "literature_density_profile_validation.csv"
)

OUT_JSON = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "literature_density_profile_validation.json"
)

OUT_PNG = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "literature_density_profile_validation.png"
)

# ============================================================
# HISTOGRAM SETTINGS
# ============================================================

DZ = 0.01          # nm = 0.1 Angstrom
ZMAX = 4.0         # nm

EDGES = np.arange(
    0.0,
    ZMAX + DZ,
    DZ
)

Z = 0.5 * (
    EDGES[:-1] +
    EDGES[1:]
)

AREA_NM2 = 32.755246

# ============================================================
# LITERATURE VALUES
# ============================================================

# 2023 Nanoscale Advances, additive FF
LIT_WATER1 = 0.330
LIT_WATER2 = 0.610
LIT_CL = 0.640
LIT_NA_INNER = 0.480

# Na+ additive solvent-separated PMF minimum
LIT_NA_OUTER_PMF = 0.775

# 2026 JACS oxygen-density first minimum
LIT_WATER_MIN_LOW = 0.45
LIT_WATER_MIN_HIGH = 0.47

# ============================================================
# LOAD
# ============================================================

for p in [
    COMPACT_NPY,
    COMPACT_JSON,
    WATER_NPY,
]:
    if not p.exists():
        raise FileNotFoundError(p)

meta = json.loads(
    COMPACT_JSON.read_text()
)

compact = np.load(
    COMPACT_NPY,
    mmap_mode="r"
)

water = np.load(
    WATER_NPY,
    mmap_mode="r"
)

layout = meta["layout"]


def get_slice(name):

    x = layout[name]

    return slice(
        int(
            x[
                "start_column_inclusive"
            ]
        ),
        int(
            x[
                "end_column_exclusive"
            ]
        )
    )


# Water is every 10 ps.
# Compact trajectory is every 1 ps.
compact_ids = np.arange(
    0,
    compact.shape[0],
    10,
    dtype=int
)

if len(compact_ids) != water.shape[0]:
    raise RuntimeError(
        "Water/compact frame alignment mismatch."
    )


# ============================================================
# EXTRACT GRAPHENE / IONS
# ============================================================

g = compact[
    compact_ids,
    get_slice(
        "graphene_z_summary_nm"
    )
].astype(np.float64)

# [mean, min, max]
graphene_mean_z = g[:, 0]


na = compact[
    compact_ids,
    get_slice(
        "na_xyz_nm"
    )
].reshape(
    len(compact_ids),
    -1,
    3
).astype(np.float64)


cl = compact[
    compact_ids,
    get_slice(
        "cl_xyz_nm"
    )
].reshape(
    len(compact_ids),
    -1,
    3
).astype(np.float64)


n_frames = len(compact_ids)
n_bins = len(Z)

print("=" * 84)
print("GRAPHENE / ELECTROLYTE LITERATURE DENSITY VALIDATION")
print("=" * 84)

print()
print(f"Frames:             {n_frames}")
print(f"Water molecules:    {water.shape[1]}")
print(f"Na+ ions:           {na.shape[1]}")
print(f"Cl- ions:           {cl.shape[1]}")
print(f"Area:               {AREA_NM2:.6f} nm^2")
print(f"Bin width:          {DZ:.3f} nm")
print(
    "Surface reference: framewise mean "
    "graphene carbon plane"
)
print()


# ============================================================
# ACCUMULATE FULL + FIVE 2-ns BLOCKS
# ============================================================

blocks = np.array_split(
    np.arange(n_frames),
    5
)

block_of_frame = np.empty(
    n_frames,
    dtype=int
)

for b, ids in enumerate(blocks):
    block_of_frame[ids] = b


full_water = np.zeros(n_bins)
full_na = np.zeros(n_bins)
full_cl = np.zeros(n_bins)

block_water = np.zeros(
    (5, n_bins)
)

block_na = np.zeros(
    (5, n_bins)
)

block_cl = np.zeros(
    (5, n_bins)
)


for f in range(n_frames):

    surface = graphene_mean_z[f]

    # OPC oxygen only
    z_water = (
        np.asarray(
            water[f, :, 0, 2],
            dtype=np.float64
        )
        -
        surface
    )

    z_na = (
        na[f, :, 2]
        -
        surface
    )

    z_cl = (
        cl[f, :, 2]
        -
        surface
    )

    hw = np.histogram(
        z_water,
        bins=EDGES
    )[0]

    hn = np.histogram(
        z_na,
        bins=EDGES
    )[0]

    hc = np.histogram(
        z_cl,
        bins=EDGES
    )[0]

    full_water += hw
    full_na += hn
    full_cl += hc

    b = block_of_frame[f]

    block_water[b] += hw
    block_na[b] += hn
    block_cl[b] += hc


# ============================================================
# NUMBER DENSITIES
#
# count / volume -> nm^-3
# /1000 -> Angstrom^-3
# ============================================================

def to_density(hist, frames):

    rho_nm3 = (
        hist
        /
        (
            frames
            *
            AREA_NM2
            *
            DZ
        )
    )

    rho_A3 = (
        rho_nm3
        /
        1000.0
    )

    return rho_nm3, rho_A3


water_nm3, water_A3 = to_density(
    full_water,
    n_frames
)

na_nm3, na_A3 = to_density(
    full_na,
    n_frames
)

cl_nm3, cl_A3 = to_density(
    full_cl,
    n_frames
)


block_water_A3 = []
block_na_A3 = []
block_cl_A3 = []

for b in range(5):

    nf = len(blocks[b])

    block_water_A3.append(
        to_density(
            block_water[b],
            nf
        )[1]
    )

    block_na_A3.append(
        to_density(
            block_na[b],
            nf
        )[1]
    )

    block_cl_A3.append(
        to_density(
            block_cl[b],
            nf
        )[1]
    )


block_water_A3 = np.asarray(
    block_water_A3
)

block_na_A3 = np.asarray(
    block_na_A3
)

block_cl_A3 = np.asarray(
    block_cl_A3
)


# ============================================================
# SMOOTHING / FEATURE DETECTION
# ============================================================

kernel = np.array(
    [1, 2, 3, 4, 5, 4, 3, 2, 1],
    dtype=float
)

kernel /= kernel.sum()


def smooth(y):

    return np.convolve(
        y,
        kernel,
        mode="same"
    )


def max_position(
    y,
    lo,
    hi
):

    ys = smooth(y)

    mask = (
        (Z >= lo)
        &
        (Z <= hi)
    )

    ids = np.where(mask)[0]

    j = ids[
        np.argmax(
            ys[ids]
        )
    ]

    return float(Z[j])


def min_between(
    y,
    z1,
    z2
):

    ys = smooth(y)

    mask = (
        (Z > z1)
        &
        (Z < z2)
    )

    ids = np.where(mask)[0]

    j = ids[
        np.argmin(
            ys[ids]
        )
    ]

    return float(Z[j])


def feature_set(
    w,
    n,
    c
):

    water1 = max_position(
        w,
        0.20,
        0.45
    )

    water2 = max_position(
        w,
        0.45,
        0.85
    )

    water_min = min_between(
        w,
        water1,
        water2
    )

    # Inner/direct Na feature
    na_inner = max_position(
        n,
        0.30,
        0.58
    )

    # Outer/solvent-separated Na feature
    na_outer = max_position(
        n,
        0.58,
        1.05
    )

    cl_peak = max_position(
        c,
        0.25,
        0.90
    )

    return {
        "water_peak1_nm":
            water1,

        "water_first_minimum_nm":
            water_min,

        "water_peak2_nm":
            water2,

        "na_inner_feature_nm":
            na_inner,

        "na_outer_feature_nm":
            na_outer,

        "cl_peak_nm":
            cl_peak,
    }


features = feature_set(
    water_A3,
    na_A3,
    cl_A3
)


block_features = []

for b in range(5):

    block_features.append(
        feature_set(
            block_water_A3[b],
            block_na_A3[b],
            block_cl_A3[b]
        )
    )


# ============================================================
# BLOCK UNCERTAINTIES
# ============================================================

feature_sd = {}

for key in features:

    vals = np.array([
        x[key]
        for x in block_features
    ])

    feature_sd[key] = float(
        np.std(
            vals,
            ddof=1
        )
    )


# ============================================================
# BULK CONCENTRATIONS / ENRICHMENT
# ============================================================

bulk_mask = (
    (Z >= 2.0)
    &
    (Z <= 4.0)
)


def bulk_info(
    rho_nm3,
    feature_position
):

    bulk = float(
        np.mean(
            rho_nm3[
                bulk_mask
            ]
        )
    )

    # 1 M = 0.602214076 particles/nm^3
    molar = (
        bulk
        /
        0.602214076
    )

    j = int(
        np.argmin(
            np.abs(
                Z -
                feature_position
            )
        )
    )

    peak = float(
        smooth(
            rho_nm3
        )[j]
    )

    enhancement = (
        peak / bulk
        if bulk > 0
        else float("nan")
    )

    return bulk, molar, enhancement


water_bulk = bulk_info(
    water_nm3,
    features[
        "water_peak1_nm"
    ]
)

na_bulk = bulk_info(
    na_nm3,
    features[
        "na_outer_feature_nm"
    ]
)

cl_bulk = bulk_info(
    cl_nm3,
    features[
        "cl_peak_nm"
    ]
)


# ============================================================
# PRINT COMPARISON
# ============================================================

print("=" * 84)
print("STRUCTURAL FEATURES")
print("=" * 84)
print()

print(
    f"{'metric':30s} "
    f"{'ours':>10s} "
    f"{'block SD':>10s} "
    f"{'literature':>18s}"
)

print("-" * 74)


comparison = [
    (
        "Water O peak 1",
        "water_peak1_nm",
        LIT_WATER1,
        "2023 additive"
    ),

    (
        "Water first minimum",
        "water_first_minimum_nm",
        0.46,
        "2026 ~0.45-0.47"
    ),

    (
        "Water O peak 2",
        "water_peak2_nm",
        LIT_WATER2,
        "2023 additive"
    ),

    (
        "Na+ inner feature",
        "na_inner_feature_nm",
        LIT_NA_INNER,
        "2023 additive"
    ),

    (
        "Na+ outer feature",
        "na_outer_feature_nm",
        LIT_NA_OUTER_PMF,
        "2023 PMF minimum"
    ),

    (
        "Cl- dominant peak",
        "cl_peak_nm",
        LIT_CL,
        "2023 additive"
    ),
]


comparison_json = []


for label, key, lit, note in comparison:

    ours = features[key]
    sd = feature_sd[key]

    print(
        f"{label:30s} "
        f"{ours:10.3f} "
        f"{sd:10.3f} "
        f"{lit:9.3f} nm  {note}"
    )

    comparison_json.append({
        "metric": label,
        "our_position_nm": ours,
        "our_block_sd_nm": sd,
        "literature_position_nm": lit,
        "difference_nm": ours - lit,
        "literature_note": note,
    })


print()
print("=" * 84)
print("BULK CONCENTRATION / INTERFACIAL ENRICHMENT")
print("=" * 84)
print()

print(
    f"Water bulk: "
    f"{water_bulk[1]:.2f} M"
)

print(
    f"Na+ bulk:   "
    f"{na_bulk[1]:.4f} M | "
    f"outer-peak enrichment "
    f"{na_bulk[2]:.2f}x"
)

print(
    f"Cl- bulk:   "
    f"{cl_bulk[1]:.4f} M | "
    f"peak enrichment "
    f"{cl_bulk[2]:.2f}x"
)


# ============================================================
# CSV
# ============================================================

lines = [
    (
        "z_nm,"
        "water_O_density_A^-3,"
        "Na_density_A^-3,"
        "Cl_density_A^-3"
    )
]

for i in range(len(Z)):

    lines.append(
        f"{Z[i]:.6f},"
        f"{water_A3[i]:.12e},"
        f"{na_A3[i]:.12e},"
        f"{cl_A3[i]:.12e}"
    )

OUT_CSV.write_text(
    "\n".join(lines) + "\n"
)


# ============================================================
# JSON
# ============================================================

OUT_JSON.write_text(
    json.dumps(
        {
            "surface_reference":
                (
                    "framewise mean graphene "
                    "carbon z plane"
                ),

            "bin_width_nm":
                DZ,

            "n_frames":
                n_frames,

            "features":
                features,

            "feature_block_sd_nm":
                feature_sd,

            "literature_comparison":
                comparison_json,

            "bulk_concentrations_M":
                {
                    "water":
                        water_bulk[1],

                    "Na":
                        na_bulk[1],

                    "Cl":
                        cl_bulk[1],
                },

            "interfacial_peak_enrichment":
                {
                    "Na_outer":
                        na_bulk[2],

                    "Cl":
                        cl_bulk[2],
                },

            "literature_context":
                {
                    "2023_Nanoscale_Advances":
                        (
                            "1 M simple salts; additive "
                            "and Drude polarizable FFs."
                        ),

                    "2026_JACS":
                        (
                            "Graphene-NaCl interface; "
                            "ML-MD / VSFG comparison."
                        ),

                    "warning":
                        (
                            "Compare peak positions and "
                            "layering qualitatively; "
                            "our DPBS concentration and "
                            "force field differ."
                        ),
                },
        },
        indent=2
    )
)


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(8.2, 9.5),
    sharex=True
)


# Water
axes[0].plot(
    Z,
    smooth(water_A3),
    label="Replica B OPC water O"
)

axes[0].axvline(
    LIT_WATER1,
    linestyle="--",
    label="2023 additive peak 1"
)

axes[0].axvline(
    LIT_WATER2,
    linestyle="--",
    label="2023 additive peak 2"
)

axes[0].axvspan(
    LIT_WATER_MIN_LOW,
    LIT_WATER_MIN_HIGH,
    alpha=0.15,
    label="2026 first-minimum range"
)

axes[0].set_ylabel(
    "Water O density (A^-3)"
)

axes[0].legend()


# Na
axes[1].plot(
    Z,
    smooth(na_A3),
    label="Replica B Na+"
)

axes[1].axvline(
    LIT_NA_INNER,
    linestyle="--",
    label="2023 inner feature"
)

axes[1].axvline(
    LIT_NA_OUTER_PMF,
    linestyle="--",
    label="2023 outer PMF minimum"
)

axes[1].set_ylabel(
    "Na+ density (A^-3)"
)

axes[1].legend()


# Cl
axes[2].plot(
    Z,
    smooth(cl_A3),
    label="Replica B Cl-"
)

axes[2].axvline(
    LIT_CL,
    linestyle="--",
    label="2023 additive Cl- peak"
)

axes[2].set_ylabel(
    "Cl- density (A^-3)"
)

axes[2].set_xlabel(
    "Distance above graphene carbon plane (nm)"
)

axes[2].legend()


for ax in axes:

    ax.set_xlim(
        0.0,
        1.5
    )

    ax.grid(
        alpha=0.2
    )


fig.suptitle(
    "Graphene-electrolyte molecular layering: "
    "Replica B vs literature"
)

fig.tight_layout()

fig.savefig(
    OUT_PNG,
    dpi=220
)

plt.close(fig)


print()
print(f"CSV:  {OUT_CSV}")
print(f"JSON: {OUT_JSON}")
print(f"Plot: {OUT_PNG}")
print()
print("LITERATURE DENSITY VALIDATION COMPLETE")

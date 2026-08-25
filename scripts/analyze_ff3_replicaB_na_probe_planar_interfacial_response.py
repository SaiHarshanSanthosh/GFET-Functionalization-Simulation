from pathlib import Path
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from openmm import openmm, unit


# ============================================================
# PATHS / CONSTANTS
# ============================================================

ROOT = Path(".")
A = ROOT / "analysis"

SYSTEM_XML = (
    ROOT / "parameters" / "combined" /
    "ff3_explicit_dpbs_opc_yb_replicaB.xml"
)

ASSEMBLY_JSON = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_assembly_check.json"
)

OUT_JSON = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "na_probe_planar_interfacial_response.json"
)

OUT_CSV = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "na_probe_planar_interfacial_response.csv"
)

OUT_PROFILE_CSV = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "na_probe_delta_charge_density_profiles.csv"
)

OUT_PROFILE_PNG = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "na_probe_delta_charge_density_profiles.png"
)

OUT_CUMULATIVE_PNG = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_"
    "na_probe_cumulative_interfacial_charge.png"
)


BRANCHES = [
    ("probe_0p5nm", 0.5),
    ("probe_1p0nm", 1.0),
    ("probe_1p5nm", 1.5),
    ("probe_2p0nm", 2.0),
    ("probe_3p0nm", 3.0),
    ("bulk_reference", 5.0),
]

REFERENCE = "bulk_reference"

PROBE_LOCAL = 55

N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70

DZ = 0.05
ZMAX = 6.0

EDGES = np.arange(
    0.0,
    ZMAX + DZ,
    DZ
)

CENTERS = (
    0.5 *
    (
        EDGES[:-1]
        +
        EDGES[1:]
    )
)

SLABS = [
    (0.0, 1.0),
    (0.0, 2.0),
]

E_PER_NM2_TO_C_PER_M2 = (
    1.602176634e-19
    /
    1.0e-18
)


# ============================================================
# SYSTEM / CHARGES
# ============================================================

system = openmm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text()
)

assembly = json.loads(
    ASSEMBLY_JSON.read_text()
)

nb = next(
    f for f in system.getForces()
    if isinstance(
        f,
        openmm.NonbondedForce
    )
)

q = np.array([
    nb.getParticleParameters(i)[0]
    .value_in_unit(
        unit.elementary_charge
    )

    for i in range(
        system.getNumParticles()
    )
], dtype=np.float64)


box = np.array([
    [
        v.x,
        v.y,
        v.z
    ]

    for v in
    system.getDefaultPeriodicBoxVectors()
], dtype=np.float64)


AREA_NM2 = float(
    np.linalg.norm(
        np.cross(
            box[0],
            box[1]
        )
    )
)


# ============================================================
# INDEX RANGES
# ============================================================

lig0 = N_GRAPHENE_TOTAL
lig1 = lig0 + N_LIGAND

n_water = int(
    assembly["opc_waters"]
)

water0 = lig1
water1 = (
    water0
    +
    4 * n_water
)

na0 = int(
    assembly[
        "na_particle_start_index"
    ]
)

na1 = int(
    assembly[
        "na_particle_end_index_exclusive"
    ]
)

k0 = int(
    assembly[
        "k_particle_start_index"
    ]
)

k1 = int(
    assembly[
        "k_particle_end_index_exclusive"
    ]
)

cl0 = int(
    assembly[
        "cl_particle_start_index"
    ]
)

cl1 = int(
    assembly[
        "cl_particle_end_index_exclusive"
    ]
)

h20 = int(
    assembly[
        "h2po4_particle_start_index"
    ]
)

h21 = int(
    assembly[
        "h2po4_particle_end_index_exclusive"
    ]
)

hp0 = int(
    assembly[
        "hpo4_particle_start_index"
    ]
)

hp1 = int(
    assembly[
        "hpo4_particle_end_index_exclusive"
    ]
)


if water1 != na0:
    raise RuntimeError(
        "Water/Na boundary mismatch."
    )


q_lig = q[lig0:lig1]
q_na = q[na0:na1]
q_k = q[k0:k1]
q_cl = q[cl0:cl1]
q_h2 = q[h20:h21]
q_hp = q[hp0:hp1]

q_probe = float(
    q_na[PROBE_LOCAL]
)


# ============================================================
# OPC M-SITE RECONSTRUCTION
# ============================================================

vs = system.getVirtualSite(
    water0 + 3
)

if not isinstance(
    vs,
    openmm.ThreeParticleAverageSite
):
    raise RuntimeError(
        "Unexpected OPC virtual site."
    )

wO = float(
    vs.getWeight(0)
)

wH1 = float(
    vs.getWeight(1)
)

wH2 = float(
    vs.getWeight(2)
)

q_water = (
    q[water0:water1]
    .reshape(
        n_water,
        4
    )
)


# ============================================================
# HELPERS
# ============================================================

def layout_slice(
    meta,
    name
):
    info = meta["layout"][name]

    return slice(
        int(
            info[
                "start_column_inclusive"
            ]
        ),
        int(
            info[
                "end_column_exclusive"
            ]
        )
    )


def charge_histogram(
    z,
    charges
):
    return np.histogram(
        z,
        bins=EDGES,
        weights=charges
    )[0]


def analyze_branch(
    branch
):

    prefix = (
        "ff3_explicit_dpbs_opc_yb_replicaB_"
        f"na_screening_{branch}_production"
    )

    meta = json.loads(
        (
            A /
            f"{prefix}_compact_1ps.json"
        ).read_text()
    )

    compact = np.load(
        A /
        f"{prefix}_compact_1ps_float32.npy",
        mmap_mode="r"
    )

    water = np.load(
        A /
        f"{prefix}_opc_ohh_10ps_float32.npy",
        mmap_mode="r"
    )


    # Water saved at:
    # 0, 10, 20, ... 1000 ps
    ids = np.arange(
        0,
        1001,
        10,
        dtype=int
    )

    if len(ids) != water.shape[0]:
        raise RuntimeError(
            f"{branch}: water alignment error."
        )


    # graphene_z_summary:
    # [mean, min, max]
    g = compact[
        ids,
        layout_slice(
            meta,
            "graphene_z_summary_nm"
        )
    ].astype(np.float64)

    graphene_max = g[:, 2]


    lig = compact[
        ids,
        layout_slice(
            meta,
            "ligand_xyz_nm"
        )
    ].reshape(
        len(ids),
        len(q_lig),
        3
    ).astype(np.float64)


    na = compact[
        ids,
        layout_slice(
            meta,
            "na_xyz_nm"
        )
    ].reshape(
        len(ids),
        len(q_na),
        3
    ).astype(np.float64)


    kk = compact[
        ids,
        layout_slice(
            meta,
            "k_xyz_nm"
        )
    ].reshape(
        len(ids),
        len(q_k),
        3
    ).astype(np.float64)


    cl = compact[
        ids,
        layout_slice(
            meta,
            "cl_xyz_nm"
        )
    ].reshape(
        len(ids),
        len(q_cl),
        3
    ).astype(np.float64)


    h2 = compact[
        ids,
        layout_slice(
            meta,
            "h2po4_xyz_nm"
        )
    ].reshape(
        len(ids),
        len(q_h2),
        3
    ).astype(np.float64)


    hp = compact[
        ids,
        layout_slice(
            meta,
            "hpo4_xyz_nm"
        )
    ].reshape(
        len(ids),
        len(q_hp),
        3
    ).astype(np.float64)


    n_frames = len(ids)
    n_bins = len(CENTERS)


    water_hist = np.zeros(
        (
            n_frames,
            n_bins
        ),
        dtype=np.float64
    )

    ion_hist = np.zeros_like(
        water_hist
    )

    ligand_hist = np.zeros_like(
        water_hist
    )

    probe_hist = np.zeros_like(
        water_hist
    )


    na_mask = np.ones(
        len(q_na),
        dtype=bool
    )

    na_mask[
        PROBE_LOCAL
    ] = False


    probe_height = np.zeros(
        n_frames,
        dtype=np.float64
    )


    for f in range(
        n_frames
    ):

        surface = float(
            graphene_max[f]
        )


        # ----------------------------------------------------
        # WATER
        # ----------------------------------------------------

        ohh = np.asarray(
            water[f],
            dtype=np.float64
        )

        zO = (
            ohh[:, 0, 2]
            -
            surface
        )

        zH1 = (
            ohh[:, 1, 2]
            -
            surface
        )

        zH2 = (
            ohh[:, 2, 2]
            -
            surface
        )

        zM = (
            wO * ohh[:, 0, 2]
            +
            wH1 * ohh[:, 1, 2]
            +
            wH2 * ohh[:, 2, 2]
            -
            surface
        )


        water_hist[f] += (
            charge_histogram(
                zO,
                q_water[:, 0]
            )
        )

        water_hist[f] += (
            charge_histogram(
                zH1,
                q_water[:, 1]
            )
        )

        water_hist[f] += (
            charge_histogram(
                zH2,
                q_water[:, 2]
            )
        )

        water_hist[f] += (
            charge_histogram(
                zM,
                q_water[:, 3]
            )
        )


        # ----------------------------------------------------
        # LIGAND
        # ----------------------------------------------------

        ligand_hist[f] += (
            charge_histogram(
                lig[f, :, 2]
                -
                surface,

                q_lig
            )
        )


        # ----------------------------------------------------
        # IONS — PROBE EXCLUDED
        # ----------------------------------------------------

        ion_hist[f] += (
            charge_histogram(
                na[
                    f,
                    na_mask,
                    2
                ]
                -
                surface,

                q_na[
                    na_mask
                ]
            )
        )

        ion_hist[f] += (
            charge_histogram(
                kk[f, :, 2]
                -
                surface,

                q_k
            )
        )

        ion_hist[f] += (
            charge_histogram(
                cl[f, :, 2]
                -
                surface,

                q_cl
            )
        )

        ion_hist[f] += (
            charge_histogram(
                h2[f, :, 2]
                -
                surface,

                q_h2
            )
        )

        ion_hist[f] += (
            charge_histogram(
                hp[f, :, 2]
                -
                surface,

                q_hp
            )
        )


        # ----------------------------------------------------
        # RESTRAINED Na+ PROBE ITSELF
        # ----------------------------------------------------

        probe_height[f] = (
            na[
                f,
                PROBE_LOCAL,
                2
            ]
            -
            surface
        )

        if (
            0.0
            <=
            probe_height[f]
            <
            ZMAX
        ):

            j = int(
                probe_height[f]
                /
                DZ
            )

            probe_hist[
                f,
                j
            ] += q_probe


    env_hist = (
        water_hist
        +
        ion_hist
        +
        ligand_hist
    )

    net_hist = (
        env_hist
        +
        probe_hist
    )


    return {
        "probe_height_nm":
            probe_height,

        "water_hist_e":
            water_hist,

        "ion_hist_e":
            ion_hist,

        "ligand_hist_e":
            ligand_hist,

        "environment_hist_e":
            env_hist,

        "probe_hist_e":
            probe_hist,

        "net_hist_e":
            net_hist,
    }


# ============================================================
# RUN SIX BRANCHES
# ============================================================

print("=" * 88)
print("PLANAR GRAPHENE-REFERENCED INTERFACIAL RESPONSE")
print("=" * 88)

print()
print(
    f"Graphene area: {AREA_NM2:.6f} nm^2"
)

print(
    f"Bin width:     {DZ:.3f} nm"
)

print(
    "Reference surface: framewise maximum "
    "graphene carbon z"
)

print(
    "Probe excluded from environmental "
    "screening quantity."
)

print()


data = {}

for branch, target in BRANCHES:

    print(
        f"Analyzing {branch} ...",
        flush=True
    )

    data[branch] = (
        analyze_branch(
            branch
        )
    )


ref = data[
    REFERENCE
]


# ============================================================
# BLOCKS
# ============================================================

blocks = np.array_split(
    np.arange(101),
    5
)


# ============================================================
# PROFILE DIFFERENCES
# ============================================================

profiles = {}

for branch, target in BRANCHES:

    if branch == REFERENCE:
        continue

    profiles[branch] = {}

    for key in [
        "water_hist_e",
        "ion_hist_e",
        "ligand_hist_e",
        "environment_hist_e",
        "probe_hist_e",
        "net_hist_e",
    ]:

        branch_mean = np.mean(
            data[branch][key],
            axis=0
        )

        ref_mean = np.mean(
            ref[key],
            axis=0
        )

        delta_hist = (
            branch_mean
            -
            ref_mean
        )

        # Convert bin charge to
        # planar volume charge density:
        # e / nm^3
        delta_rho = (
            delta_hist
            /
            (
                AREA_NM2
                *
                DZ
            )
        )

        profiles[
            branch
        ][key] = (
            delta_rho
        )


# ============================================================
# INTEGRATED SLAB METRICS
# ============================================================

metrics = []


def slab_charge(
    hist,
    z0,
    z1
):

    mask = (
        (CENTERS >= z0)
        &
        (CENTERS < z1)
    )

    return np.sum(
        hist[:, mask],
        axis=1
    )


print()
print("=" * 88)
print("INTERFACIAL CHARGE METRICS")
print("=" * 88)

print()

print(
    "Delta sigma_env = screening/environment "
    "change relative to 5 nm reference."
)

print(
    "Delta sigma_net = probe + environment "
    "change relative to 5 nm reference."
)

print()

print(
    f"{'branch':16s} "
    f"{'slab':>7s} "
    f"{'env e/nm2':>14s} "
    f"{'env SD':>10s} "
    f"{'probe':>10s} "
    f"{'net e/nm2':>14s} "
    f"{'net SD':>10s} "
    f"{'SNR':>7s}"
)

print("-" * 100)


for branch, target in BRANCHES:

    if branch == REFERENCE:
        continue


    for z0, z1 in SLABS:

        b_env = slab_charge(
            data[branch][
                "environment_hist_e"
            ],
            z0,
            z1
        )

        r_env = slab_charge(
            ref[
                "environment_hist_e"
            ],
            z0,
            z1
        )

        b_probe = slab_charge(
            data[branch][
                "probe_hist_e"
            ],
            z0,
            z1
        )

        r_probe = slab_charge(
            ref[
                "probe_hist_e"
            ],
            z0,
            z1
        )

        b_net = (
            b_env
            +
            b_probe
        )

        r_net = (
            r_env
            +
            r_probe
        )


        env_delta_frame = (
            b_env
            -
            r_env
        )

        probe_delta_frame = (
            b_probe
            -
            r_probe
        )

        net_delta_frame = (
            b_net
            -
            r_net
        )


        env_sigma = float(
            np.mean(
                env_delta_frame
            )
            /
            AREA_NM2
        )

        probe_sigma = float(
            np.mean(
                probe_delta_frame
            )
            /
            AREA_NM2
        )

        net_sigma = float(
            np.mean(
                net_delta_frame
            )
            /
            AREA_NM2
        )


        env_blocks = np.array([
            np.mean(
                env_delta_frame[
                    ids
                ]
            )
            /
            AREA_NM2

            for ids in blocks
        ])

        net_blocks = np.array([
            np.mean(
                net_delta_frame[
                    ids
                ]
            )
            /
            AREA_NM2

            for ids in blocks
        ])


        env_sd = float(
            np.std(
                env_blocks,
                ddof=1
            )
        )

        net_sd = float(
            np.std(
                net_blocks,
                ddof=1
            )
        )


        snr = (
            abs(net_sigma)
            /
            net_sd
            if net_sd > 0
            else float("inf")
        )


        metrics.append({
            "branch":
                branch,

            "target_height_nm":
                target,

            "mean_probe_height_above_graphene_max_nm":
                float(
                    np.mean(
                        data[branch][
                            "probe_height_nm"
                        ]
                    )
                ),

            "slab_start_nm":
                z0,

            "slab_end_nm":
                z1,

            "delta_sigma_environment_e_per_nm2":
                env_sigma,

            "delta_sigma_environment_block_sd_e_per_nm2":
                env_sd,

            "delta_sigma_probe_e_per_nm2":
                probe_sigma,

            "delta_sigma_net_e_per_nm2":
                net_sigma,

            "delta_sigma_net_block_sd_e_per_nm2":
                net_sd,

            "delta_sigma_net_C_per_m2":
                (
                    net_sigma
                    *
                    E_PER_NM2_TO_C_PER_M2
                ),

            "net_signal_to_block_sd":
                snr,

            "environment_block_values_e_per_nm2":
                env_blocks.tolist(),

            "net_block_values_e_per_nm2":
                net_blocks.tolist(),
        })


        slab_name = (
            f"{z0:.0f}-{z1:.0f}"
        )


        print(
            f"{branch:16s} "
            f"{slab_name:>7s} "
            f"{env_sigma:+14.6f} "
            f"{env_sd:10.6f} "
            f"{probe_sigma:+10.6f} "
            f"{net_sigma:+14.6f} "
            f"{net_sd:10.6f} "
            f"{snr:7.2f}"
        )


# ============================================================
# COMPONENT BREAKDOWN FOR 0-1 nm
# ============================================================

print()
print("=" * 88)
print("0-1 nm COMPONENT BREAKDOWN")
print("=" * 88)

print()

print(
    f"{'branch':16s} "
    f"{'water':>12s} "
    f"{'ions':>12s} "
    f"{'ligand':>12s} "
    f"{'probe':>12s} "
    f"{'net':>12s}"
)

print("-" * 82)


component_summary = []


for branch, target in BRANCHES:

    if branch == REFERENCE:
        continue


    z0 = 0.0
    z1 = 1.0


    vals = {}

    for name, key in [
        (
            "water",
            "water_hist_e"
        ),
        (
            "ions",
            "ion_hist_e"
        ),
        (
            "ligand",
            "ligand_hist_e"
        ),
        (
            "probe",
            "probe_hist_e"
        ),
    ]:

        b = slab_charge(
            data[branch][key],
            z0,
            z1
        )

        r = slab_charge(
            ref[key],
            z0,
            z1
        )

        vals[name] = float(
            np.mean(
                b - r
            )
            /
            AREA_NM2
        )


    net = (
        vals["water"]
        +
        vals["ions"]
        +
        vals["ligand"]
        +
        vals["probe"]
    )


    component_summary.append({
        "branch":
            branch,

        "target_height_nm":
            target,

        "water_delta_sigma_e_per_nm2":
            vals["water"],

        "ion_delta_sigma_e_per_nm2":
            vals["ions"],

        "ligand_delta_sigma_e_per_nm2":
            vals["ligand"],

        "probe_delta_sigma_e_per_nm2":
            vals["probe"],

        "net_delta_sigma_e_per_nm2":
            net,
    })


    print(
        f"{branch:16s} "
        f"{vals['water']:+12.6f} "
        f"{vals['ions']:+12.6f} "
        f"{vals['ligand']:+12.6f} "
        f"{vals['probe']:+12.6f} "
        f"{net:+12.6f}"
    )


# ============================================================
# SAVE METRIC CSV
# ============================================================

header = (
    "branch,target_height_nm,"
    "mean_probe_height_above_graphene_max_nm,"
    "slab_start_nm,slab_end_nm,"
    "delta_sigma_environment_e_per_nm2,"
    "delta_sigma_environment_block_sd_e_per_nm2,"
    "delta_sigma_probe_e_per_nm2,"
    "delta_sigma_net_e_per_nm2,"
    "delta_sigma_net_block_sd_e_per_nm2,"
    "delta_sigma_net_C_per_m2,"
    "net_signal_to_block_sd"
)

lines = [
    header
]

for m in metrics:

    lines.append(
        ",".join([
            m["branch"],
            f"{m['target_height_nm']:.6f}",
            f"{m['mean_probe_height_above_graphene_max_nm']:.9f}",
            f"{m['slab_start_nm']:.6f}",
            f"{m['slab_end_nm']:.6f}",
            f"{m['delta_sigma_environment_e_per_nm2']:.12f}",
            f"{m['delta_sigma_environment_block_sd_e_per_nm2']:.12f}",
            f"{m['delta_sigma_probe_e_per_nm2']:.12f}",
            f"{m['delta_sigma_net_e_per_nm2']:.12f}",
            f"{m['delta_sigma_net_block_sd_e_per_nm2']:.12f}",
            f"{m['delta_sigma_net_C_per_m2']:.12f}",
            f"{m['net_signal_to_block_sd']:.6f}",
        ])
    )

OUT_CSV.write_text(
    "\n".join(lines)
    +
    "\n"
)


# ============================================================
# SAVE PROFILE CSV
# ============================================================

profile_lines = [
    (
        "branch,target_height_nm,z_nm,"
        "delta_rho_water_e_per_nm3,"
        "delta_rho_ions_e_per_nm3,"
        "delta_rho_ligand_e_per_nm3,"
        "delta_rho_environment_e_per_nm3,"
        "delta_rho_probe_e_per_nm3,"
        "delta_rho_net_e_per_nm3"
    )
]


for branch, target in BRANCHES:

    if branch == REFERENCE:
        continue

    p = profiles[
        branch
    ]

    for j, z in enumerate(
        CENTERS
    ):

        profile_lines.append(
            (
                f"{branch},"
                f"{target:.6f},"
                f"{z:.6f},"
                f"{p['water_hist_e'][j]:.12f},"
                f"{p['ion_hist_e'][j]:.12f},"
                f"{p['ligand_hist_e'][j]:.12f},"
                f"{p['environment_hist_e'][j]:.12f},"
                f"{p['probe_hist_e'][j]:.12f},"
                f"{p['net_hist_e'][j]:.12f}"
            )
        )


OUT_PROFILE_CSV.write_text(
    "\n".join(
        profile_lines
    )
    +
    "\n"
)


# ============================================================
# SAVE JSON
# ============================================================

OUT_JSON.write_text(
    json.dumps(
        {
            "description":
                (
                    "Graphene-referenced planar "
                    "electrolyte response to a "
                    "restrained +1e Na probe."
                ),

            "reference_branch":
                REFERENCE,

            "surface_reference":
                (
                    "framewise maximum graphene "
                    "carbon z"
                ),

            "area_nm2":
                AREA_NM2,

            "bin_width_nm":
                DZ,

            "probe_charge_e":
                q_probe,

            "environment_definition":
                (
                    "water + all non-probe ions "
                    "+ ligand"
                ),

            "metrics":
                metrics,

            "component_summary_0to1nm":
                component_summary,
        },
        indent=2
    )
)


# ============================================================
# PLOT 1 — DELTA ENVIRONMENT CHARGE DENSITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.2, 5.2)
)


for branch, target in BRANCHES:

    if branch == REFERENCE:
        continue

    ax.plot(
        CENTERS,
        profiles[
            branch
        ][
            "environment_hist_e"
        ],
        label=f"{target:g} nm"
    )


ax.axhline(
    0.0,
    linewidth=1
)

ax.axvline(
    1.0,
    linewidth=1,
    linestyle="--"
)

ax.axvline(
    2.0,
    linewidth=1,
    linestyle="--"
)

ax.set_xlim(
    0.0,
    4.0
)

ax.set_xlabel(
    "Height above graphene surface (nm)"
)

ax.set_ylabel(
    "Delta environment charge density (e/nm^3)"
)

ax.set_title(
    "Probe-induced electrolyte charge redistribution"
)

ax.legend(
    title="Probe target"
)

fig.tight_layout()

fig.savefig(
    OUT_PROFILE_PNG,
    dpi=200
)

plt.close(fig)


# ============================================================
# PLOT 2 — CUMULATIVE NET INTERFACIAL CHARGE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.2, 5.2)
)


for branch, target in BRANCHES:

    if branch == REFERENCE:
        continue

    delta_net_density = (
        profiles[
            branch
        ][
            "net_hist_e"
        ]
    )

    cumulative_sigma = np.cumsum(
        delta_net_density
        *
        DZ
    )

    ax.plot(
        CENTERS,
        cumulative_sigma,
        label=f"{target:g} nm"
    )


ax.axhline(
    0.0,
    linewidth=1
)

ax.axvline(
    1.0,
    linewidth=1,
    linestyle="--"
)

ax.axvline(
    2.0,
    linewidth=1,
    linestyle="--"
)

ax.set_xlim(
    0.0,
    4.0
)

ax.set_xlabel(
    "Height above graphene surface (nm)"
)

ax.set_ylabel(
    "Cumulative net charge change (e/nm^2)"
)

ax.set_title(
    "Net probe + electrolyte charge presented above graphene"
)

ax.legend(
    title="Probe target"
)

fig.tight_layout()

fig.savefig(
    OUT_CUMULATIVE_PNG,
    dpi=200
)

plt.close(fig)


# ============================================================
# FINAL
# ============================================================

print()
print(
    f"Metrics CSV:   {OUT_CSV}"
)

print(
    f"Profile CSV:   {OUT_PROFILE_CSV}"
)

print(
    f"JSON:          {OUT_JSON}"
)

print(
    f"Profile plot:  {OUT_PROFILE_PNG}"
)

print(
    f"Cumulative:    {OUT_CUMULATIVE_PNG}"
)

print()
print(
    "PLANAR INTERFACIAL RESPONSE ANALYSIS COMPLETE"
)

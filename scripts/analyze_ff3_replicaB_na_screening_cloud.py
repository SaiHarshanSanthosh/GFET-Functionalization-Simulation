from pathlib import Path
import json
import numpy as np
from openmm import openmm, unit

ROOT = Path(".")
A = ROOT / "analysis"

SYSTEM_XML = ROOT / "parameters" / "combined" / "ff3_explicit_dpbs_opc_yb_replicaB.xml"
ASSEMBLY_JSON = A / "ff3_explicit_dpbs_opc_yb_replicaB_assembly_check.json"

OUT_CSV = A / "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_cloud.csv"
OUT_JSON = A / "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_cloud.json"
OUT_PNG = A / "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_cloud.png"

BRANCHES = [
    ("probe_0p5nm", 0.5),
    ("probe_1p0nm", 1.0),
    ("probe_1p5nm", 1.5),
    ("probe_2p0nm", 2.0),
    ("probe_3p0nm", 3.0),
    ("bulk_reference", 5.0),
]

PROBE_LOCAL = 55
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70

RMAX = 2.5
DR = 0.02
EDGES = np.arange(0.0, RMAX + DR, DR)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])

REPORT_RADII = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]


# ------------------------------------------------------------
# SYSTEM / CHARGES
# ------------------------------------------------------------

system = openmm.XmlSerializer.deserialize(
    SYSTEM_XML.read_text()
)

assembly = json.loads(
    ASSEMBLY_JSON.read_text()
)

nb = next(
    f for f in system.getForces()
    if isinstance(f, openmm.NonbondedForce)
)

q = np.array([
    nb.getParticleParameters(i)[0].value_in_unit(
        unit.elementary_charge
    )
    for i in range(system.getNumParticles())
], dtype=np.float64)

box = np.array([
    [v.x, v.y, v.z]
    for v in system.getDefaultPeriodicBoxVectors()
], dtype=np.float64)

# 2D triclinic cell for x/y minimum image.
B = np.array([
    [box[0,0], box[1,0]],
    [box[0,1], box[1,1]],
], dtype=np.float64)

B_inv = np.linalg.inv(B)


# ------------------------------------------------------------
# INDEX RANGES
# ------------------------------------------------------------

lig0 = N_GRAPHENE_TOTAL
lig1 = lig0 + N_LIGAND

n_water = int(assembly["opc_waters"])
water0 = lig1
water1 = water0 + 4 * n_water

na0 = int(assembly["na_particle_start_index"])
na1 = int(assembly["na_particle_end_index_exclusive"])

k0 = int(assembly["k_particle_start_index"])
k1 = int(assembly["k_particle_end_index_exclusive"])

cl0 = int(assembly["cl_particle_start_index"])
cl1 = int(assembly["cl_particle_end_index_exclusive"])

h20 = int(assembly["h2po4_particle_start_index"])
h21 = int(assembly["h2po4_particle_end_index_exclusive"])

hp0 = int(assembly["hpo4_particle_start_index"])
hp1 = int(assembly["hpo4_particle_end_index_exclusive"])

if water1 != na0:
    raise RuntimeError("Water/Na index mismatch")


# ------------------------------------------------------------
# OPC M SITE
# ------------------------------------------------------------

vs = system.getVirtualSite(water0 + 3)

wO = float(vs.getWeight(0))
wH1 = float(vs.getWeight(1))
wH2 = float(vs.getWeight(2))

q_water = q[water0:water1].reshape(n_water, 4)

q_lig = q[lig0:lig1]
q_na = q[na0:na1]
q_k = q[k0:k1]
q_cl = q[cl0:cl1]
q_h2 = q[h20:h21]
q_hp = q[hp0:hp1]


def sl(meta, name):
    x = meta["layout"][name]
    return slice(
        int(x["start_column_inclusive"]),
        int(x["end_column_exclusive"])
    )


def mic_distances(points, center):
    """
    Minimum image in graphene x/y plane.
    No z wrapping is needed for r <= 2.5 nm
    around probes at 0.5-5 nm.
    """

    d = np.asarray(points, dtype=np.float64) - center

    frac = d[:, :2] @ B_inv.T
    frac -= np.rint(frac)

    d[:, :2] = frac @ B.T

    return np.sqrt(
        np.sum(d * d, axis=1)
    )


def add_hist(hist, coords, charges, probe):
    r = mic_distances(coords, probe)

    hist += np.histogram(
        r,
        bins=EDGES,
        weights=charges
    )[0]


def analyze_branch(branch):

    prefix = (
        "ff3_explicit_dpbs_opc_yb_replicaB_"
        f"na_screening_{branch}_production"
    )

    meta = json.loads(
        (A / f"{prefix}_compact_1ps.json").read_text()
    )

    compact = np.load(
        A / f"{prefix}_compact_1ps_float32.npy",
        mmap_mode="r"
    )

    water = np.load(
        A / f"{prefix}_opc_ohh_10ps_float32.npy",
        mmap_mode="r"
    )

    ids = np.arange(0, 1001, 10)

    if len(ids) != water.shape[0]:
        raise RuntimeError(f"{branch}: water alignment error")

    lig = compact[
        ids, sl(meta, "ligand_xyz_nm")
    ].reshape(len(ids), len(q_lig), 3)

    na = compact[
        ids, sl(meta, "na_xyz_nm")
    ].reshape(len(ids), len(q_na), 3)

    kk = compact[
        ids, sl(meta, "k_xyz_nm")
    ].reshape(len(ids), len(q_k), 3)

    cl = compact[
        ids, sl(meta, "cl_xyz_nm")
    ].reshape(len(ids), len(q_cl), 3)

    h2 = compact[
        ids, sl(meta, "h2po4_xyz_nm")
    ].reshape(len(ids), len(q_h2), 3)

    hp = compact[
        ids, sl(meta, "hpo4_xyz_nm")
    ].reshape(len(ids), len(q_hp), 3)

    frame_total = []
    frame_water = []
    frame_ions = []
    frame_lig = []

    for f in range(len(ids)):

        probe = np.asarray(
            na[f, PROBE_LOCAL],
            dtype=np.float64
        )

        hw = np.zeros(len(CENTERS))
        hi = np.zeros(len(CENTERS))
        hl = np.zeros(len(CENTERS))

        # -------------------------
        # Water: O, H1, H2, M
        # -------------------------

        ohh = np.asarray(
            water[f],
            dtype=np.float64
        )

        M = (
            wO  * ohh[:,0,:]
            + wH1 * ohh[:,1,:]
            + wH2 * ohh[:,2,:]
        )

        water_xyz = np.concatenate(
            [
                ohh[:,0,:],
                ohh[:,1,:],
                ohh[:,2,:],
                M,
            ],
            axis=0
        )

        water_q = np.concatenate(
            [
                q_water[:,0],
                q_water[:,1],
                q_water[:,2],
                q_water[:,3],
            ]
        )

        add_hist(
            hw,
            water_xyz,
            water_q,
            probe
        )

        # -------------------------
        # Other ions
        # -------------------------

        na_mask = np.ones(
            len(q_na),
            dtype=bool
        )

        na_mask[PROBE_LOCAL] = False

        add_hist(
            hi,
            na[f, na_mask],
            q_na[na_mask],
            probe
        )

        add_hist(
            hi,
            kk[f],
            q_k,
            probe
        )

        add_hist(
            hi,
            cl[f],
            q_cl,
            probe
        )

        add_hist(
            hi,
            h2[f],
            q_h2,
            probe
        )

        add_hist(
            hi,
            hp[f],
            q_hp,
            probe
        )

        # -------------------------
        # Ligand
        # -------------------------

        add_hist(
            hl,
            lig[f],
            q_lig,
            probe
        )

        frame_water.append(
            np.cumsum(hw)
        )

        frame_ions.append(
            np.cumsum(hi)
        )

        frame_lig.append(
            np.cumsum(hl)
        )

        frame_total.append(
            np.cumsum(
                hw + hi + hl
            )
        )

    return {
        "total": np.asarray(frame_total),
        "water": np.asarray(frame_water),
        "ions": np.asarray(frame_ions),
        "ligand": np.asarray(frame_lig),
    }


# ------------------------------------------------------------
# ANALYZE
# ------------------------------------------------------------

results = {}

print("=" * 78)
print("Na+ LOCAL SCREENING CLOUD")
print("=" * 78)
print()

for branch, target in BRANCHES:

    print(
        f"Analyzing {branch} ...",
        flush=True
    )

    results[branch] = analyze_branch(branch)


# ------------------------------------------------------------
# REPORT TABLE
# ------------------------------------------------------------

def radius_index(r):
    return int(
        np.argmin(
            np.abs(CENTERS - r)
        )
    )


print()
print(
    "Numbers below are mean ENVIRONMENT charge "
    "inside a sphere around the +1e Na+."
)
print(
    "Perfect neutralization would approach -1 e."
)
print()

header = "branch".ljust(16)

for r in REPORT_RADII:
    header += f" Q<{r:.1f}nm".rjust(11)

print(header)
print("-" * len(header))

summary = []

for branch, target in BRANCHES:

    arr = results[branch]["total"]

    mean_curve = np.mean(
        arr,
        axis=0
    )

    row = {
        "branch": branch,
        "target_nm": target,
        "radii_nm": {},
    }

    line = branch.ljust(16)

    for r in REPORT_RADII:

        j = radius_index(r)

        value = float(
            mean_curve[j]
        )

        blocks = np.array_split(
            np.arange(arr.shape[0]),
            5
        )

        block_vals = np.array([
            np.mean(
                arr[idx, j]
            )
            for idx in blocks
        ])

        sd = float(
            np.std(
                block_vals,
                ddof=1
            )
        )

        row["radii_nm"][str(r)] = {
            "mean_environment_charge_e":
                value,
            "block_sd_e":
                sd,
            "effective_enclosed_charge_e":
                1.0 + value,
        }

        line += f" {value:+10.3f}"

    print(line)
    summary.append(row)


# ------------------------------------------------------------
# COMPONENT TABLE AT 0.7 AND 1.0 nm
# ------------------------------------------------------------

print()
print("COMPONENTS")
print()

print(
    f"{'branch':16s} "
    f"{'r':>5s} "
    f"{'water':>10s} "
    f"{'ions':>10s} "
    f"{'ligand':>10s} "
    f"{'total env':>11s} "
    f"{'q effective':>12s}"
)

print("-" * 82)

for branch, target in BRANCHES:

    for r in [0.7, 1.0]:

        j = radius_index(r)

        qw = float(
            np.mean(
                results[branch]["water"][:, j]
            )
        )

        qi = float(
            np.mean(
                results[branch]["ions"][:, j]
            )
        )

        ql = float(
            np.mean(
                results[branch]["ligand"][:, j]
            )
        )

        qt = qw + qi + ql

        print(
            f"{branch:16s} "
            f"{r:5.1f} "
            f"{qw:+10.3f} "
            f"{qi:+10.3f} "
            f"{ql:+10.3f} "
            f"{qt:+11.3f} "
            f"{1.0+qt:+12.3f}"
        )


# ------------------------------------------------------------
# SAVE CSV
# ------------------------------------------------------------

lines = [
    "branch,target_nm,radius_nm,"
    "environment_charge_e,"
    "effective_enclosed_charge_e,"
    "water_charge_e,ion_charge_e,ligand_charge_e"
]

for branch, target in BRANCHES:

    mt = np.mean(
        results[branch]["total"],
        axis=0
    )

    mw = np.mean(
        results[branch]["water"],
        axis=0
    )

    mi = np.mean(
        results[branch]["ions"],
        axis=0
    )

    ml = np.mean(
        results[branch]["ligand"],
        axis=0
    )

    for j, r in enumerate(CENTERS):

        lines.append(
            f"{branch},{target:.6f},{r:.6f},"
            f"{mt[j]:.9f},{1.0+mt[j]:.9f},"
            f"{mw[j]:.9f},{mi[j]:.9f},{ml[j]:.9f}"
        )

OUT_CSV.write_text(
    "\n".join(lines) + "\n"
)


# ------------------------------------------------------------
# SAVE JSON
# ------------------------------------------------------------

OUT_JSON.write_text(
    json.dumps(
        {
            "description":
                "Cumulative mobile/environment charge around restrained Na+ probe.",
            "probe_charge_e": 1.0,
            "rmax_nm": RMAX,
            "dr_nm": DR,
            "summary": summary,
        },
        indent=2
    )
)


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    figsize=(8.0, 5.2)
)

for branch, target in BRANCHES:

    mean_curve = np.mean(
        results[branch]["total"],
        axis=0
    )

    ax.plot(
        CENTERS,
        1.0 + mean_curve,
        label=f"{target:g} nm"
    )

ax.axhline(
    0.0,
    linewidth=1
)

ax.axhline(
    1.0,
    linewidth=1,
    linestyle="--"
)

ax.set_xlim(
    0.0,
    2.0
)

ax.set_xlabel(
    "Distance from restrained Na+ (nm)"
)

ax.set_ylabel(
    "Effective enclosed charge, +1e + environment (e)"
)

ax.set_title(
    "Local screening cloud around Na+"
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
print("LOCAL SCREENING CLOUD ANALYSIS COMPLETE")

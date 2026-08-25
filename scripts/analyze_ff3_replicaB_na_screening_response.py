from pathlib import Path
import json
import numpy as np
from openmm import openmm, unit

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

OUT_CSV = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_response.csv"
)

OUT_JSON = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_response.json"
)

OUT_PNG = (
    A /
    "ff3_explicit_dpbs_opc_yb_replicaB_na_screening_response.png"
)

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

EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19


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

area_nm2 = float(
    np.linalg.norm(
        np.cross(box[0], box[1])
    )
)


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
    raise RuntimeError("Water/Na boundary mismatch")


# ------------------------------------------------------------
# OPC M-SITE GEOMETRY
# ------------------------------------------------------------

vs = system.getVirtualSite(water0 + 3)

if not isinstance(vs, openmm.ThreeParticleAverageSite):
    raise RuntimeError("Unexpected OPC virtual site")

wO = float(vs.getWeight(0))
wH1 = float(vs.getWeight(1))
wH2 = float(vs.getWeight(2))


q_lig = q[lig0:lig1]
q_water = q[water0:water1].reshape(n_water, 4)
q_na = q[na0:na1]
q_k = q[k0:k1]
q_cl = q[cl0:cl1]
q_h2 = q[h20:h21]
q_hp = q[hp0:hp1]

solution_charge = float(q[lig0:].sum())

probe_charge = float(q_na[PROBE_LOCAL])


print("=" * 78)
print("DIRECT Na+ SCREENING RESPONSE ANALYSIS")
print("=" * 78)
print()
print(f"Area:                     {area_nm2:.6f} nm^2")
print(f"Solution charge:          {solution_charge:+.12e} e")
print(f"Probe charge:             {probe_charge:+.6f} e")
print(f"OPC M weights:            {wO:.12f}, {wH1:.12f}, {wH2:.12f}")
print()


# mV per (e nm)
#
# phi(0)-phi(top) = - Mz / (eps0*A)
#
MV_PER_E_NM = (
    -E_CHARGE * 1.0e12 /
    (EPS0 * area_nm2)
)

print(
    f"Planar potential factor:  "
    f"{MV_PER_E_NM:.6f} mV per e*nm"
)
print()


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def sl(meta, name):
    info = meta["layout"][name]
    return slice(
        int(info["start_column_inclusive"]),
        int(info["end_column_exclusive"])
    )


def load_branch(branch):

    prefix = (
        "ff3_explicit_dpbs_opc_yb_replicaB_"
        f"na_screening_{branch}_production"
    )

    compact_json = (
        A / f"{prefix}_compact_1ps.json"
    )

    compact_npy = (
        A / f"{prefix}_compact_1ps_float32.npy"
    )

    water_npy = (
        A / f"{prefix}_opc_ohh_10ps_float32.npy"
    )

    meta = json.loads(
        compact_json.read_text()
    )

    compact = np.load(
        compact_npy,
        mmap_mode="r"
    )

    water = np.load(
        water_npy,
        mmap_mode="r"
    )

    if compact.shape[0] != 1001:
        raise RuntimeError(
            f"{branch}: expected 1001 compact frames"
        )

    if water.shape[0] != 101:
        raise RuntimeError(
            f"{branch}: expected 101 water frames"
        )

    # Water frames are 0,10,...1000 ps.
    ci = np.arange(
        0,
        compact.shape[0],
        10,
        dtype=int
    )

    if len(ci) != water.shape[0]:
        raise RuntimeError(
            f"{branch}: trajectory alignment mismatch"
        )

    g = compact[
        ci,
        sl(meta, "graphene_z_summary_nm")
    ][:, 0].astype(np.float64)

    lig = compact[
        ci,
        sl(meta, "ligand_xyz_nm")
    ].reshape(
        len(ci),
        len(q_lig),
        3
    ).astype(np.float64)

    na = compact[
        ci,
        sl(meta, "na_xyz_nm")
    ].reshape(
        len(ci),
        len(q_na),
        3
    ).astype(np.float64)

    kk = compact[
        ci,
        sl(meta, "k_xyz_nm")
    ].reshape(
        len(ci),
        len(q_k),
        3
    ).astype(np.float64)

    cl = compact[
        ci,
        sl(meta, "cl_xyz_nm")
    ].reshape(
        len(ci),
        len(q_cl),
        3
    ).astype(np.float64)

    h2 = compact[
        ci,
        sl(meta, "h2po4_xyz_nm")
    ].reshape(
        len(ci),
        len(q_h2),
        3
    ).astype(np.float64)

    hp = compact[
        ci,
        sl(meta, "hpo4_xyz_nm")
    ].reshape(
        len(ci),
        len(q_hp),
        3
    ).astype(np.float64)


    mz = np.zeros(
        len(ci),
        dtype=np.float64
    )


    # Non-water solution species.
    mz += np.einsum(
        "fi,i->f",
        lig[:, :, 2] - g[:, None],
        q_lig
    )

    mz += np.einsum(
        "fi,i->f",
        na[:, :, 2] - g[:, None],
        q_na
    )

    mz += np.einsum(
        "fi,i->f",
        kk[:, :, 2] - g[:, None],
        q_k
    )

    mz += np.einsum(
        "fi,i->f",
        cl[:, :, 2] - g[:, None],
        q_cl
    )

    mz += np.einsum(
        "fi,i->f",
        h2[:, :, 2] - g[:, None],
        q_h2
    )

    mz += np.einsum(
        "fi,i->f",
        hp[:, :, 2] - g[:, None],
        q_hp
    )


    # Explicit OPC water, including reconstructed M site.
    for f in range(len(ci)):

        z = (
            np.asarray(
                water[f, :, :, 2],
                dtype=np.float64
            )
            -
            g[f]
        )

        zM = (
            wO * z[:, 0]
            +
            wH1 * z[:, 1]
            +
            wH2 * z[:, 2]
        )

        mz[f] += np.sum(
            q_water[:, 0] * z[:, 0]
            +
            q_water[:, 1] * z[:, 1]
            +
            q_water[:, 2] * z[:, 2]
            +
            q_water[:, 3] * zM
        )


    probe_height = (
        na[:, PROBE_LOCAL, 2]
        -
        g
    )


    return {
        "mz": mz,
        "probe_height": probe_height,
    }


# ------------------------------------------------------------
# LOAD ALL SIX
# ------------------------------------------------------------

data = {}

for branch, target in BRANCHES:

    print(f"Loading {branch} ...", flush=True)

    data[branch] = load_branch(
        branch
    )


print()
print("All six branches loaded.")
print()


# ------------------------------------------------------------
# REFERENCE
# ------------------------------------------------------------

ref_name = "bulk_reference"

ref = data[ref_name]

ref_mz_mean = float(
    np.mean(ref["mz"])
)

ref_h_mean = float(
    np.mean(ref["probe_height"])
)


# Five approximately 200-ps blocks.
blocks = np.array_split(
    np.arange(101),
    5
)


# ------------------------------------------------------------
# RESPONSE TABLE
# ------------------------------------------------------------

rows = []

print(
    f"{'branch':16s} "
    f"{'h mean':>8s} "
    f"{'h SD':>8s} "
    f"{'total mV':>11s} "
    f"{'block SD':>10s} "
    f"{'direct mV':>11s} "
    f"{'env mV':>11s}"
)

print("-" * 86)


for branch, target in BRANCHES:

    d = data[branch]

    hmean = float(
        np.mean(
            d["probe_height"]
        )
    )

    hsd = float(
        np.std(
            d["probe_height"],
            ddof=1
        )
    )

    mzmean = float(
        np.mean(
            d["mz"]
        )
    )

    d_mz = (
        mzmean
        -
        ref_mz_mean
    )

    total_mv = (
        MV_PER_E_NM
        *
        d_mz
    )

    direct_d_mz = (
        probe_charge
        *
        (
            hmean
            -
            ref_h_mean
        )
    )

    direct_mv = (
        MV_PER_E_NM
        *
        direct_d_mz
    )

    env_mv = (
        total_mv
        -
        direct_mv
    )


    block_mv = []

    for ids in blocks:

        dm = (
            np.mean(
                d["mz"][ids]
            )
            -
            np.mean(
                ref["mz"][ids]
            )
        )

        block_mv.append(
            MV_PER_E_NM
            *
            dm
        )


    block_mv = np.asarray(
        block_mv,
        dtype=float
    )

    block_sd = float(
        np.std(
            block_mv,
            ddof=1
        )
    )


    rows.append({
        "branch": branch,
        "target_nm": target,
        "mean_height_nm": hmean,
        "height_sd_nm": hsd,
        "mean_solution_Mz_e_nm": mzmean,
        "delta_Mz_vs_5nm_e_nm": d_mz,
        "delta_phi_total_mV": total_mv,
        "delta_phi_block_sd_mV": block_sd,
        "delta_phi_direct_probe_mV": direct_mv,
        "delta_phi_environment_mV": env_mv,
        "block_delta_phi_mV": block_mv.tolist(),
    })


    print(
        f"{branch:16s} "
        f"{hmean:8.4f} "
        f"{hsd:8.4f} "
        f"{total_mv:11.3f} "
        f"{block_sd:10.3f} "
        f"{direct_mv:11.3f} "
        f"{env_mv:11.3f}"
    )


# ------------------------------------------------------------
# EXPONENTIAL FIT
#
# y(h) = A [ exp(-h/lambda) - exp(-href/lambda) ]
# ------------------------------------------------------------

fit_rows = [
    r
    for r in rows
    if r["branch"] != ref_name
]

h = np.array(
    [
        r["mean_height_nm"]
        for r in fit_rows
    ],
    dtype=float
)

y = np.array(
    [
        r["delta_phi_total_mV"]
        for r in fit_rows
    ],
    dtype=float
)


def fit_decay(h, y, href):

    lambdas = np.linspace(
        0.10,
        5.00,
        4901
    )

    best = None

    for lam in lambdas:

        x = (
            np.exp(-h / lam)
            -
            np.exp(-href / lam)
        )

        denom = float(
            np.dot(x, x)
        )

        if denom <= 0:
            continue

        amp = float(
            np.dot(x, y)
            /
            denom
        )

        pred = amp * x

        sse = float(
            np.sum(
                (y - pred) ** 2
            )
        )

        if (
            best is None
            or
            sse < best["sse"]
        ):
            best = {
                "lambda_nm": float(lam),
                "amplitude_mV": amp,
                "sse": sse,
                "prediction": pred,
            }


    ybar = float(
        np.mean(y)
    )

    sst = float(
        np.sum(
            (y - ybar) ** 2
        )
    )

    if sst > 0:
        r2 = (
            1.0
            -
            best["sse"]
            /
            sst
        )
    else:
        r2 = float("nan")

    best["r2"] = float(r2)

    return best


fit = fit_decay(
    h,
    y,
    ref_h_mean
)


# ------------------------------------------------------------
# BLOCK FITS
# ------------------------------------------------------------

block_fits = []

for b in range(5):

    hb = np.array(
        [
            np.mean(
                data[r["branch"]][
                    "probe_height"
                ][blocks[b]]
            )
            for r in fit_rows
        ],
        dtype=float
    )

    href_b = float(
        np.mean(
            ref[
                "probe_height"
            ][blocks[b]]
        )
    )

    yb = np.array(
        [
            r[
                "block_delta_phi_mV"
            ][b]
            for r in fit_rows
        ],
        dtype=float
    )

    fb = fit_decay(
        hb,
        yb,
        href_b
    )

    block_fits.append(
        fb
    )


block_lambdas = np.array(
    [
        x["lambda_nm"]
        for x in block_fits
    ],
    dtype=float
)


# ------------------------------------------------------------
# PRINT FIT
# ------------------------------------------------------------

print()
print("=" * 78)
print("SCREENING DECAY FIT")
print("=" * 78)
print()

print(
    f"5 nm reference mean height: "
    f"{ref_h_mean:.6f} nm"
)

print(
    f"Best lambda_eff:             "
    f"{fit['lambda_nm']:.6f} nm"
)

print(
    f"Amplitude:                   "
    f"{fit['amplitude_mV']:.6f} mV"
)

print(
    f"R^2:                         "
    f"{fit['r2']:.6f}"
)

print()

print(
    "200-ps block lambda_eff values:"
)

for i, fb in enumerate(block_fits):
    print(
        f"  block {i+1}: "
        f"{fb['lambda_nm']:.6f} nm "
        f"(R^2 {fb['r2']:.4f})"
    )

print()

print(
    f"Block lambda mean:           "
    f"{block_lambdas.mean():.6f} nm"
)

print(
    f"Block lambda SD:             "
    f"{block_lambdas.std(ddof=1):.6f} nm"
)

print()

print(
    "Independent bulk Debye result "
    "from Replica B: ~0.7055 nm"
)


boundary_hit = (
    fit["lambda_nm"] <= 0.101
    or
    fit["lambda_nm"] >= 4.999
)

if boundary_hit or fit["r2"] < 0.80:
    quality = (
        "WEAK / NOT YET A DEFENSIBLE "
        "SINGLE-EXPONENTIAL SCREENING LENGTH"
    )
else:
    quality = (
        "PROVISIONAL SINGLE-EXPONENTIAL "
        "SCREENING SIGNAL"
    )

print()
print(f"Fit assessment: {quality}")


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

header = (
    "branch,target_nm,mean_height_nm,height_sd_nm,"
    "mean_solution_Mz_e_nm,delta_Mz_vs_5nm_e_nm,"
    "delta_phi_total_mV,delta_phi_block_sd_mV,"
    "delta_phi_direct_probe_mV,"
    "delta_phi_environment_mV"
)

lines = [header]

for r in rows:

    lines.append(
        ",".join([
            r["branch"],
            f"{r['target_nm']:.6f}",
            f"{r['mean_height_nm']:.9f}",
            f"{r['height_sd_nm']:.9f}",
            f"{r['mean_solution_Mz_e_nm']:.12f}",
            f"{r['delta_Mz_vs_5nm_e_nm']:.12f}",
            f"{r['delta_phi_total_mV']:.9f}",
            f"{r['delta_phi_block_sd_mV']:.9f}",
            f"{r['delta_phi_direct_probe_mV']:.9f}",
            f"{r['delta_phi_environment_mV']:.9f}",
        ])
    )

OUT_CSV.write_text(
    "\n".join(lines) + "\n"
)


# ------------------------------------------------------------
# JSON
# ------------------------------------------------------------

result = {
    "method": (
        "Planar-averaged solution potential "
        "from framewise charge dipole moment"
    ),
    "electrostatic_relation": (
        "phi_graphene_minus_vacuum = "
        "-Mz/(epsilon0*A)"
    ),
    "explicit_water_relative_permittivity_used": 1.0,
    "area_nm2": area_nm2,
    "solution_net_charge_e": solution_charge,
    "probe_charge_e": probe_charge,
    "reference_branch": ref_name,
    "reference_mean_height_nm": ref_h_mean,
    "rows": rows,
    "fit": {
        "model": (
            "A*(exp(-h/lambda)-"
            "exp(-href/lambda))"
        ),
        "lambda_eff_nm": fit["lambda_nm"],
        "amplitude_mV": fit["amplitude_mV"],
        "r2": fit["r2"],
        "assessment": quality,
    },
    "block_lambda_eff_nm":
        block_lambdas.tolist(),
    "block_lambda_mean_nm":
        float(block_lambdas.mean()),
    "block_lambda_sd_nm":
        float(block_lambdas.std(ddof=1)),
    "independent_bulk_debye_nm": 0.705505,
}

OUT_JSON.write_text(
    json.dumps(
        result,
        indent=2
    )
)


# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

try:
    import matplotlib.pyplot as plt

    hh = np.linspace(
        min(h) * 0.9,
        ref_h_mean,
        300
    )

    yy = (
        fit["amplitude_mV"]
        *
        (
            np.exp(
                -hh / fit["lambda_nm"]
            )
            -
            np.exp(
                -ref_h_mean
                /
                fit["lambda_nm"]
            )
        )
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 5.0)
    )

    total_x = np.array([
        r["mean_height_nm"]
        for r in rows
    ])

    total_y = np.array([
        r["delta_phi_total_mV"]
        for r in rows
    ])

    total_err = np.array([
        r["delta_phi_block_sd_mV"]
        for r in rows
    ])

    direct_y = np.array([
        r["delta_phi_direct_probe_mV"]
        for r in rows
    ])

    env_y = np.array([
        r["delta_phi_environment_mV"]
        for r in rows
    ])

    ax.errorbar(
        total_x,
        total_y,
        yerr=total_err,
        marker="o",
        linestyle="none",
        capsize=3,
        label="Total screened response"
    )

    ax.plot(
        total_x,
        direct_y,
        marker="o",
        label="Direct probe contribution"
    )

    ax.plot(
        total_x,
        env_y,
        marker="o",
        label="Environment response"
    )

    ax.plot(
        hh,
        yy,
        label=(
            f"Exponential fit "
            f"(lambda={fit['lambda_nm']:.3f} nm)"
        )
    )

    ax.axhline(
        0.0,
        linewidth=1
    )

    ax.set_xlabel(
        "Na+ probe height above graphene (nm)"
    )

    ax.set_ylabel(
        "Potential response vs 5 nm reference (mV)"
    )

    ax.set_title(
        "Explicit-DPBS screening response"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT_PNG,
        dpi=200
    )

    plt.close(fig)

    print()
    print(f"Plot: {OUT_PNG}")

except Exception as exc:
    print()
    print(
        "Plot generation failed, "
        f"numerical analysis still saved: {exc}"
    )


print(f"CSV:  {OUT_CSV}")
print(f"JSON: {OUT_JSON}")
print()
print("ANALYSIS COMPLETE")

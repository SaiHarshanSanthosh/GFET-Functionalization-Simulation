from pathlib import Path
import json
import math

import numpy as np
from openmm import XmlSerializer, openmm, unit


# ============================================================
# DROP-01 STAGE-A 20 ps LIGAND GEOMETRY AUDIT
#
# READ ONLY:
#   - no Context
#   - no MD
#   - no minimization
#   - no coordinate modification
# ============================================================

N_GRAPHENE_CARBONS = 1250
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70
N_SOLUTE = 3820

LIGAND_START = N_GRAPHENE_TOTAL
LIGAND_STOP = N_SOLUTE

C35 = 35
C36 = 36
H69 = 69


SYSTEM_XML = Path(
    "parameters/combined/"
    "drop01_faceon_2p00nm_fixedz0_wallfree_opc_yb.xml"
)

INITIAL_POSITIONS = Path(
    "parameters/combined/"
    "drop01_faceon_2p00nm_fixedz0_wallfree_opc_yb_positions_nm.npy"
)

FINAL_POSITIONS = Path(
    "parameters/combined/"
    "drop01_faceon_2p00nm_fixedz0_stageA_20ps_positions_nm.npy"
)

MOL2 = Path(
    "structures/"
    "pyrene_peg5_propargyl.mol2"
)

OUTPUT = Path(
    "analysis/"
    "drop01_faceon_2p00nm_fixedz0_stageA_20ps_"
    "ligand_geometry_audit.json"
)


# ============================================================
# 1. LOAD
# ============================================================

system = XmlSerializer.deserialize(
    SYSTEM_XML.read_text()
)

xyz0 = np.load(
    INITIAL_POSITIONS
)

xyz1 = np.load(
    FINAL_POSITIONS
)


if xyz0.shape != xyz1.shape:
    raise RuntimeError(
        "Initial/final coordinate shapes differ."
    )

if xyz1.shape[0] != system.getNumParticles():
    raise RuntimeError(
        "Coordinate/System particle-count mismatch."
    )


# ============================================================
# 2. MOL2 ATOM NAMES + PYRENE AROMATIC ATOMS
# ============================================================

lines = MOL2.read_text().splitlines()

atom_start = None
bond_start = None

for i, line in enumerate(lines):

    if line.strip() == "@<TRIPOS>ATOM":
        atom_start = i + 1

    elif line.strip() == "@<TRIPOS>BOND":
        bond_start = i + 1


if atom_start is None or bond_start is None:
    raise RuntimeError(
        "Could not locate MOL2 sections."
    )


atom_names = []
atom_types = []

for line in lines[atom_start:]:

    if line.startswith("@<TRIPOS>"):
        break

    if not line.strip():
        continue

    f = line.split()

    atom_names.append(
        f[1]
    )

    atom_types.append(
        f[5].lower()
    )


if len(atom_names) != N_LIGAND:
    raise RuntimeError(
        f"Expected {N_LIGAND} ligand atoms; "
        f"found {len(atom_names)}."
    )


aromatic = set()

for line in lines[bond_start:]:

    if line.startswith("@<TRIPOS>"):
        break

    if not line.strip():
        continue

    f = line.split()

    if len(f) < 4:
        continue

    a = int(f[1]) - 1
    b = int(f[2]) - 1
    bt = f[3].lower()

    if bt in {
        "ar",
        "aro",
    }:
        aromatic.add(a)
        aromatic.add(b)


for i, t in enumerate(atom_types):

    if t in {
        "ca",
        "cp",
        "cq",
        "cc",
        "cd",
    }:
        aromatic.add(i)


aromatic = np.asarray(
    sorted(aromatic),
    dtype=int,
)


if len(aromatic) != 16:
    raise RuntimeError(
        f"Expected 16 pyrene aromatic atoms; "
        f"found {len(aromatic)}."
    )


# ============================================================
# 3. PERIODIC BOX / MINIMUM IMAGE
# ============================================================

a, b, c = (
    system.getDefaultPeriodicBoxVectors()
)


def vec_nm(v):

    return np.asarray(
        [
            v[0],
            v[1],
            v[2],
        ],
        dtype=float,
    )


box = np.vstack(
    [
        vec_nm(a),
        vec_nm(b),
        vec_nm(c),
    ]
)

inv_box = np.linalg.inv(
    box
)


def minimum_image(delta):

    frac = (
        delta
        @
        inv_box
    )

    frac -= np.round(
        frac
    )

    return (
        frac
        @
        box
    )


def distance_nm(p1, p2):

    d = minimum_image(
        p1 - p2
    )

    return float(
        np.linalg.norm(
            d
        )
    )


def angle_deg(a, b, c):

    # Angle a-b-c, with b as vertex.
    v1 = minimum_image(
        a - b
    )

    v2 = minimum_image(
        c - b
    )

    cosine = float(
        np.dot(v1, v2)
        /
        (
            np.linalg.norm(v1)
            *
            np.linalg.norm(v2)
        )
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


# ============================================================
# 4. FIND GAFF2 LIGAND BOND FORCE
# ============================================================

ligand_bond_force = None

for force in system.getForces():

    if (
        isinstance(
            force,
            openmm.HarmonicBondForce
        )
        and
        force.getName()
        ==
        "GAFF2 ligand bonds"
    ):

        ligand_bond_force = force
        break


if ligand_bond_force is None:
    raise RuntimeError(
        "Could not find GAFF2 ligand bond force."
    )


# ============================================================
# 5. AUDIT EVERY GAFF2 LIGAND BOND
# ============================================================

bond_rows = []


for term_i in range(
    ligand_bond_force.getNumBonds()
):

    i, j, r0, k = (
        ligand_bond_force.getBondParameters(
            term_i
        )
    )

    i = int(i)
    j = int(j)

    if not (
        LIGAND_START <= i < LIGAND_STOP
        and
        LIGAND_START <= j < LIGAND_STOP
    ):
        continue

    li = i - LIGAND_START
    lj = j - LIGAND_START

    r0_nm = float(
        r0.value_in_unit(
            unit.nanometer
        )
    )

    d0 = distance_nm(
        xyz0[i],
        xyz0[j],
    )

    d1 = distance_nm(
        xyz1[i],
        xyz1[j],
    )

    abs_dev = abs(
        d1 - r0_nm
    )

    rel_dev = (
        abs_dev
        /
        r0_nm
    )

    # Deliberately lenient structural-integrity flag.
    #
    # We flag only if the final bond differs from the FF
    # equilibrium value by BOTH a meaningful absolute amount
    # and >15%.
    severe = bool(
        abs_dev > 0.020
        and
        rel_dev > 0.15
    )

    bond_rows.append(
        {
            "term":
                term_i,

            "local_i":
                li,

            "local_j":
                lj,

            "name_i":
                atom_names[li],

            "name_j":
                atom_names[lj],

            "equilibrium_nm":
                r0_nm,

            "initial_nm":
                d0,

            "final_nm":
                d1,

            "absolute_deviation_from_r0_nm":
                abs_dev,

            "relative_deviation_from_r0":
                rel_dev,

            "severe_flag":
                severe,
        }
    )


if not bond_rows:
    raise RuntimeError(
        "No ligand bonds were found."
    )


bond_rows_sorted = sorted(
    bond_rows,
    key=lambda x:
        x[
            "absolute_deviation_from_r0_nm"
        ],
    reverse=True,
)


severe_bonds = [
    row
    for row in bond_rows
    if row["severe_flag"]
]


# ============================================================
# 6. GRAPHENE PLANE + PYRENE GEOMETRY
# ============================================================

def fit_plane(points):

    center = points.mean(
        axis=0
    )

    centered = (
        points
        -
        center
    )

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    normal = vh[-1]

    normal /= np.linalg.norm(
        normal
    )

    if normal[2] < 0:
        normal *= -1.0

    return center, normal


def ligand_geometry(xyz):

    graphene = (
        xyz[
            :N_GRAPHENE_CARBONS
        ]
    )

    ligand = (
        xyz[
            LIGAND_START:
            LIGAND_STOP
        ]
    )

    g_center, g_normal = fit_plane(
        graphene
    )

    pyrene = (
        ligand[
            aromatic
        ]
    )

    p_center, p_normal = fit_plane(
        pyrene
    )

    plane_offsets = (
        pyrene
        -
        p_center
    ) @ p_normal

    plane_rms = float(
        np.sqrt(
            np.mean(
                plane_offsets ** 2
            )
        )
    )

    cosang = float(
        np.clip(
            abs(
                np.dot(
                    g_normal,
                    p_normal,
                )
            ),
            0.0,
            1.0,
        )
    )

    tilt = float(
        np.degrees(
            np.arccos(
                cosang
            )
        )
    )

    heights = (
        ligand
        -
        g_center
    ) @ g_normal

    pyrene_height = float(
        np.dot(
            p_center
            -
            g_center,
            g_normal,
        )
    )

    c36_height = float(
        heights[C36]
    )

    pyrene_to_c36 = float(
        distance_nm(
            p_center,
            ligand[C36],
        )
    )

    return {
        "pyrene_height_nm":
            pyrene_height,

        "pyrene_tilt_deg":
            tilt,

        "pyrene_plane_rms_nm":
            plane_rms,

        "ligand_min_height_nm":
            float(
                heights.min()
            ),

        "ligand_max_height_nm":
            float(
                heights.max()
            ),

        "ligand_vertical_span_nm":
            float(
                heights.max()
                -
                heights.min()
            ),

        "terminal_C36_height_nm":
            c36_height,

        "pyrene_center_to_C36_nm":
            pyrene_to_c36,

        "C35_C36_nm":
            distance_nm(
                ligand[C35],
                ligand[C36],
            ),

        "C36_H69_nm":
            distance_nm(
                ligand[C36],
                ligand[H69],
            ),

        "C35_C36_H69_deg":
            angle_deg(
                ligand[C35],
                ligand[C36],
                ligand[H69],
            ),
    }


geom0 = ligand_geometry(
    xyz0
)

geom1 = ligand_geometry(
    xyz1
)


# ============================================================
# 7. PREDEFINED STRUCTURAL-INTEGRITY CHECKS
# ============================================================

pyrene_planar_ok = (
    geom1[
        "pyrene_plane_rms_nm"
    ]
    <
    0.020
)

alkyne_linear_ok = (
    geom1[
        "C35_C36_H69_deg"
    ]
    >
    160.0
)

no_severe_bond_stretch = (
    len(
        severe_bonds
    )
    ==
    0
)

ligand_above_graphene = (
    geom1[
        "ligand_min_height_nm"
    ]
    >
    0.0
)


overall_pass = all(
    [
        pyrene_planar_ok,
        alkyne_linear_ok,
        no_severe_bond_stretch,
        ligand_above_graphene,
    ]
)


# ============================================================
# 8. REPORT
# ============================================================

print()
print("=" * 78)
print("DROP-01 STAGE-A 20 ps LIGAND GEOMETRY AUDIT")
print("=" * 78)

print()
print("ALL GAFF2 LIGAND BONDS")
print("----------------------")
print(
    "Bond terms checked:",
    len(
        bond_rows
    )
)
print(
    "Severe bond-stretch flags:",
    len(
        severe_bonds
    )
)

print()
print("TOP 10 FINAL BOND DEVIATIONS FROM FF EQUILIBRIUM")
print("------------------------------------------------")

for rank, row in enumerate(
    bond_rows_sorted[:10],
    start=1,
):

    print(
        f"{rank:2d}. "
        f"{row['local_i']:2d}:{row['name_i']:<5s} - "
        f"{row['local_j']:2d}:{row['name_j']:<5s} | "
        f"r0={row['equilibrium_nm']:.5f} nm  "
        f"final={row['final_nm']:.5f} nm  "
        f"|dr|={row['absolute_deviation_from_r0_nm']:.5f} nm  "
        f"rel={100.0 * row['relative_deviation_from_r0']:.2f}%"
    )


print()
print("PYRENE")
print("------")
print(
    f"Height: "
    f"{geom0['pyrene_height_nm']:.6f} -> "
    f"{geom1['pyrene_height_nm']:.6f} nm"
)
print(
    f"Tilt: "
    f"{geom0['pyrene_tilt_deg']:.3f} -> "
    f"{geom1['pyrene_tilt_deg']:.3f} deg"
)
print(
    f"Plane RMS: "
    f"{geom0['pyrene_plane_rms_nm']:.6f} -> "
    f"{geom1['pyrene_plane_rms_nm']:.6f} nm"
)


print()
print("TERMINAL ALKYNE")
print("----------------")
print(
    f"C35-C36: "
    f"{geom0['C35_C36_nm']:.6f} -> "
    f"{geom1['C35_C36_nm']:.6f} nm"
)
print(
    f"C36-H69: "
    f"{geom0['C36_H69_nm']:.6f} -> "
    f"{geom1['C36_H69_nm']:.6f} nm"
)
print(
    f"C35-C36-H69: "
    f"{geom0['C35_C36_H69_deg']:.3f} -> "
    f"{geom1['C35_C36_H69_deg']:.3f} deg"
)


print()
print("LINKER / SURFACE GEOMETRY")
print("-------------------------")
print(
    f"C36 height: "
    f"{geom0['terminal_C36_height_nm']:.6f} -> "
    f"{geom1['terminal_C36_height_nm']:.6f} nm"
)
print(
    f"Pyrene center -> C36: "
    f"{geom0['pyrene_center_to_C36_nm']:.6f} -> "
    f"{geom1['pyrene_center_to_C36_nm']:.6f} nm"
)
print(
    f"Ligand vertical span: "
    f"{geom0['ligand_vertical_span_nm']:.6f} -> "
    f"{geom1['ligand_vertical_span_nm']:.6f} nm"
)
print(
    f"Closest ligand atom height: "
    f"{geom1['ligand_min_height_nm']:.6f} nm"
)


print()
print("INTEGRITY CHECKS")
print("----------------")
print(
    "No severe bond stretching:",
    "PASS"
    if no_severe_bond_stretch
    else "FAIL",
)
print(
    "Pyrene planarity (<0.020 nm RMS):",
    "PASS"
    if pyrene_planar_ok
    else "FAIL",
)
print(
    "Terminal alkyne linearity (>160 deg):",
    "PASS"
    if alkyne_linear_ok
    else "FAIL",
)
print(
    "Ligand remains above graphene:",
    "PASS"
    if ligand_above_graphene
    else "FAIL",
)


status = (
    "DROP01_STAGEA_20PS_LIGAND_GEOMETRY_PASS"
    if overall_pass
    else
    "DROP01_STAGEA_20PS_LIGAND_GEOMETRY_INSPECT"
)


print()
print("=" * 78)
print(status)
print("=" * 78)


# ============================================================
# 9. SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

metadata = {
    "status":
        status,

    "bond_terms_checked":
        len(
            bond_rows
        ),

    "severe_bond_flags":
        severe_bonds,

    "top_10_bond_deviations":
        bond_rows_sorted[:10],

    "initial_geometry":
        geom0,

    "final_geometry":
        geom1,

    "checks":
        {
            "no_severe_bond_stretch":
                no_severe_bond_stretch,

            "pyrene_planarity_ok":
                pyrene_planar_ok,

            "terminal_alkyne_linearity_ok":
                alkyne_linear_ok,

            "ligand_above_graphene":
                ligand_above_graphene,
        },
}

OUTPUT.write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    +
    "\n"
)

print()
print(
    "Saved:",
    OUTPUT
)
print(
    "No MD or coordinate modification was performed."
)

from pathlib import Path
import json
import numpy as np

START = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb_minimized_positions_nm.npy"
)

MINIMIZED = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb_"
    "equilibrated_300K_600ps_total_positions_nm.npy"
)

MOL2 = Path(
    "structures/pyrene_peg5_propargyl.mol2"
)

OUTPUT = Path(
    "analysis/"
    "ff3_300K_equilibrated_ligand_geometry_audit.json"
)

N_GRAPHENE_CARBONS = 1250
LIGAND_START = 3750
N_LIGAND = 70
LIGAND_STOP = (
    LIGAND_START
    +
    N_LIGAND
)

ALKYNE_C1 = 35
ALKYNE_C2 = 36
TERMINAL_H = 69


# ============================================================
# PARSE LIGAND MOL2
# ============================================================

lines = MOL2.read_text().splitlines()

atom_section = None
bond_section = None

for i, line in enumerate(lines):

    text = line.strip()

    if text == "@<TRIPOS>ATOM":
        atom_section = i + 1

    elif text == "@<TRIPOS>BOND":
        bond_section = i + 1


if atom_section is None or bond_section is None:
    raise RuntimeError(
        "Could not find MOL2 ATOM/BOND sections."
    )


atom_types = []

for line in lines[
    atom_section:
]:

    if line.startswith("@<TRIPOS>"):
        break

    if not line.strip():
        continue

    fields = line.split()

    atom_types.append(
        fields[5].lower()
    )


if len(atom_types) != N_LIGAND:
    raise RuntimeError(
        f"Expected {N_LIGAND} ligand atoms, "
        f"found {len(atom_types)}."
    )


aromatic_atoms = set()
bonds = []


for line in lines[
    bond_section:
]:

    if line.startswith("@<TRIPOS>"):
        break

    if not line.strip():
        continue

    fields = line.split()

    if len(fields) < 4:
        continue

    a = int(fields[1]) - 1
    b = int(fields[2]) - 1
    bond_type = fields[3].lower()

    bonds.append(
        (
            a,
            b,
            bond_type,
        )
    )

    if bond_type in {
        "ar",
        "aro",
    }:

        aromatic_atoms.add(a)
        aromatic_atoms.add(b)


for i, atom_type in enumerate(
    atom_types
):

    if atom_type in {
        "ca",
        "cp",
        "cq",
        "cc",
        "cd",
    }:

        aromatic_atoms.add(i)


aromatic_atoms = np.asarray(
    sorted(aromatic_atoms),
    dtype=int,
)


if len(aromatic_atoms) != 16:
    raise RuntimeError(
        "Expected exactly 16 pyrene aromatic atoms, "
        f"found {len(aromatic_atoms)}."
    )


triple_pairs = {
    tuple(
        sorted(
            (
                a,
                b,
            )
        )
    )
    for a, b, bond_type in bonds
    if bond_type in {
        "3",
        "triple",
    }
}


if (
    tuple(
        sorted(
            (
                ALKYNE_C1,
                ALKYNE_C2,
            )
        )
    )
    not in triple_pairs
):

    raise RuntimeError(
        "Expected local atoms 35-36 to be "
        "the MOL2 triple bond."
    )


# ============================================================
# GEOMETRY HELPERS
# ============================================================

def fitted_plane(points):

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

    if normal[2] < 0.0:
        normal = -normal

    signed = (
        centered
        @
        normal
    )

    rms = float(
        np.sqrt(
            np.mean(
                signed**2
            )
        )
    )

    return (
        center,
        normal,
        rms,
    )


def angle_degrees(
    vector_a,
    vector_b,
):

    a = (
        vector_a
        /
        np.linalg.norm(
            vector_a
        )
    )

    b = (
        vector_b
        /
        np.linalg.norm(
            vector_b
        )
    )

    cosine = float(
        np.clip(
            np.dot(
                a,
                b,
            ),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def bond_angle_degrees(
    p1,
    p2,
    p3,
):

    v1 = (
        p1
        -
        p2
    )

    v2 = (
        p3
        -
        p2
    )

    return angle_degrees(
        v1,
        v2,
    )


# ============================================================
# ANALYZE ONE STRUCTURE
# ============================================================

def analyze(
    label,
    xyz,
):

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise RuntimeError(
            f"{label}: unexpected coordinate shape "
            f"{xyz.shape}"
        )

    if xyz.shape[0] < LIGAND_STOP:
        raise RuntimeError(
            f"{label}: too few particles."
        )

    if not np.isfinite(xyz).all():
        raise RuntimeError(
            f"{label}: NaN/Inf coordinates."
        )


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

    pyrene = (
        ligand[
            aromatic_atoms
        ]
    )


    (
        graphene_center,
        graphene_normal,
        graphene_plane_rms,
    ) = fitted_plane(
        graphene
    )

    (
        pyrene_center,
        pyrene_normal,
        pyrene_plane_rms,
    ) = fitted_plane(
        pyrene
    )


    # Pyrene-plane normal relative to graphene-plane normal.
    # Parallel planes -> 0 degrees.
    raw_normal_angle = angle_degrees(
        graphene_normal,
        pyrene_normal,
    )

    pyrene_tilt = min(
        raw_normal_angle,
        180.0
        -
        raw_normal_angle,
    )


    # Signed perpendicular distance between the pyrene center
    # and fitted graphene plane.
    pyrene_height = float(
        np.dot(
            pyrene_center
            -
            graphene_center,
            graphene_normal,
        )
    )


    c1 = ligand[
        ALKYNE_C1
    ]

    c2 = ligand[
        ALKYNE_C2
    ]

    terminal_h = ligand[
        TERMINAL_H
    ]


    c1_height = float(
        np.dot(
            c1
            -
            graphene_center,
            graphene_normal,
        )
    )

    c2_height = float(
        np.dot(
            c2
            -
            graphene_center,
            graphene_normal,
        )
    )

    h_height = float(
        np.dot(
            terminal_h
            -
            graphene_center,
            graphene_normal,
        )
    )


    alkyne_bond_length = float(
        np.linalg.norm(
            c2
            -
            c1
        )
    )

    alkyne_terminal_angle = (
        bond_angle_degrees(
            c1,
            c2,
            terminal_h,
        )
    )

    pyrene_to_terminal_c = float(
        np.linalg.norm(
            c2
            -
            pyrene_center
        )
    )


    ligand_heights = (
        (
            ligand
            -
            graphene_center
        )
        @
        graphene_normal
    )


    return {

        "label":
            label,

        "graphene_plane_rms_nm":
            graphene_plane_rms,

        "graphene_normal":
            graphene_normal.tolist(),

        "pyrene_plane_rms_nm":
            pyrene_plane_rms,

        "pyrene_tilt_deg":
            pyrene_tilt,

        "pyrene_height_nm":
            pyrene_height,

        "alkyne_C35_height_nm":
            c1_height,

        "alkyne_C36_height_nm":
            c2_height,

        "terminal_H69_height_nm":
            h_height,

        "alkyne_C35_C36_bond_nm":
            alkyne_bond_length,

        "alkyne_C35_C36_H69_angle_deg":
            alkyne_terminal_angle,

        "pyrene_center_to_terminal_C36_nm":
            pyrene_to_terminal_c,

        "ligand_min_height_nm":
            float(
                ligand_heights.min()
            ),

        "ligand_max_height_nm":
            float(
                ligand_heights.max()
            ),

        "ligand_vertical_span_nm":
            float(
                ligand_heights.max()
                -
                ligand_heights.min()
            ),
    }


# ============================================================
# LOAD AND COMPARE
# ============================================================

start_xyz = np.load(
    START
)

minimized_xyz = np.load(
    MINIMIZED
)


before = analyze(
    "assembled_start",
    start_xyz,
)

after = analyze(
    "minimized",
    minimized_xyz,
)


changes = {

    "pyrene_height_change_nm":
        (
            after[
                "pyrene_height_nm"
            ]
            -
            before[
                "pyrene_height_nm"
            ]
        ),

    "pyrene_tilt_change_deg":
        (
            after[
                "pyrene_tilt_deg"
            ]
            -
            before[
                "pyrene_tilt_deg"
            ]
        ),

    "pyrene_plane_rms_change_nm":
        (
            after[
                "pyrene_plane_rms_nm"
            ]
            -
            before[
                "pyrene_plane_rms_nm"
            ]
        ),

    "terminal_C36_height_change_nm":
        (
            after[
                "alkyne_C36_height_nm"
            ]
            -
            before[
                "alkyne_C36_height_nm"
            ]
        ),

    "terminal_H69_height_change_nm":
        (
            after[
                "terminal_H69_height_nm"
            ]
            -
            before[
                "terminal_H69_height_nm"
            ]
        ),

    "pyrene_to_terminal_C36_change_nm":
        (
            after[
                "pyrene_center_to_terminal_C36_nm"
            ]
            -
            before[
                "pyrene_center_to_terminal_C36_nm"
            ]
        ),

    "ligand_vertical_span_change_nm":
        (
            after[
                "ligand_vertical_span_nm"
            ]
            -
            before[
                "ligand_vertical_span_nm"
            ]
        ),
}


result = {

    "aromatic_local_indices":
        aromatic_atoms.tolist(),

    "terminal_group_local_indices": {
        "alkyne_C35":
            ALKYNE_C1,

        "alkyne_C36":
            ALKYNE_C2,

        "terminal_H69":
            TERMINAL_H,
    },

    "before":
        before,

    "after":
        after,

    "changes":
        changes,
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    json.dumps(
        result,
        indent=2,
    )
)


print()
print("=" * 72)
print("FF-3 POST-MINIMIZATION LIGAND GEOMETRY AUDIT")
print("=" * 72)

print()
print("PYRENE")
print("------")

print(
    f"Height:    "
    f"{before['pyrene_height_nm']:.6f}"
    f" -> "
    f"{after['pyrene_height_nm']:.6f} nm"
)

print(
    f"Tilt:      "
    f"{before['pyrene_tilt_deg']:.6f}"
    f" -> "
    f"{after['pyrene_tilt_deg']:.6f} deg"
)

print(
    f"Plane RMS: "
    f"{before['pyrene_plane_rms_nm']:.6f}"
    f" -> "
    f"{after['pyrene_plane_rms_nm']:.6f} nm"
)


print()
print("TERMINAL ALKYNE")
print("---------------")

print(
    f"C35 height: "
    f"{before['alkyne_C35_height_nm']:.6f}"
    f" -> "
    f"{after['alkyne_C35_height_nm']:.6f} nm"
)

print(
    f"C36 height: "
    f"{before['alkyne_C36_height_nm']:.6f}"
    f" -> "
    f"{after['alkyne_C36_height_nm']:.6f} nm"
)

print(
    f"H69 height: "
    f"{before['terminal_H69_height_nm']:.6f}"
    f" -> "
    f"{after['terminal_H69_height_nm']:.6f} nm"
)

print(
    f"C35-C36:    "
    f"{before['alkyne_C35_C36_bond_nm']:.6f}"
    f" -> "
    f"{after['alkyne_C35_C36_bond_nm']:.6f} nm"
)

print(
    f"C35-C36-H69 angle: "
    f"{before['alkyne_C35_C36_H69_angle_deg']:.3f}"
    f" -> "
    f"{after['alkyne_C35_C36_H69_angle_deg']:.3f} deg"
)


print()
print("LINKER")
print("------")

print(
    f"Pyrene center -> terminal C36: "
    f"{before['pyrene_center_to_terminal_C36_nm']:.6f}"
    f" -> "
    f"{after['pyrene_center_to_terminal_C36_nm']:.6f} nm"
)

print(
    f"Vertical span: "
    f"{before['ligand_vertical_span_nm']:.6f}"
    f" -> "
    f"{after['ligand_vertical_span_nm']:.6f} nm"
)

print(
    f"Min height: "
    f"{after['ligand_min_height_nm']:.6f} nm"
)

print(
    f"Max height: "
    f"{after['ligand_max_height_nm']:.6f} nm"
)


print()
print("SAVED")
print("-----")
print(OUTPUT)

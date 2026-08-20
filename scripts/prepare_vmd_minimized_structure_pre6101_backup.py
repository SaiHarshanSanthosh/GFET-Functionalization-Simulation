from pathlib import Path
from collections import deque
import json

import numpy as np

from openmm import openmm, unit


# ============================================================
# PREPARE CONVERGED MINIMIZED STRUCTURE FOR VMD
#
# PURPOSE
# -------
# OpenMM stores coordinates inside a periodic simulation box.
# A bonded molecule can therefore look "split" across periodic
# boundaries even when none of its chemical bonds are broken.
#
# This script:
#
#   1. Loads the converged minimized coordinates.
#   2. Reads Pyrene-PEG5 connectivity from MOL2.
#   3. Reconstructs the ligand as one continuous molecule.
#   4. Checks ligand bond lengths.
#   5. Places the whole ligand near the central graphene image.
#   6. Exports clean visualization-only PDB files for VMD.
#
# IMPORTANT:
# ---------
# The actual OpenMM simulation coordinates are NOT modified.
# These outputs are visualization-only.
#
# NO MD.
# NO MINIMIZATION.
# NO HEATING.
# ============================================================


# ============================================================
# 1. SYSTEM INDEXING
# ============================================================

N_GRAPHENE_CARBONS = 1250
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70
N_SOLUTE = 3820

LIGAND_START = N_GRAPHENE_TOTAL
LIGAND_STOP = N_SOLUTE

SITES_PER_WATER = 4


# ============================================================
# 2. INPUT FILES
# ============================================================

SYSTEM_XML = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_supported_pme.xml"
)

POSITIONS_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_one_sided_opc_"
    "minimized_converged_positions_nm.npy"
)

LIGAND_MOL2 = Path(
    "structures/"
    "pyrene_peg5_propargyl.mol2"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

DRY_OUTPUT = Path(
    "structures/"
    "pyrene_peg5_graphene_minimized_vmd_whole.pdb"
)

FULL_OUTPUT = Path(
    "structures/"
    "pyrene_peg5_graphene_minimized_vmd_whole_with_water.pdb"
)

METADATA_OUTPUT = Path(
    "analysis/"
    "vmd_minimized_structure_preparation.json"
)


# ============================================================
# 4. VERIFY INPUTS
# ============================================================

for path in [
    SYSTEM_XML,
    POSITIONS_FILE,
    LIGAND_MOL2,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# 5. LOAD OPENMM SYSTEM
# ============================================================

with open(
    SYSTEM_XML,
    "r",
) as f:

    system = (
        openmm.XmlSerializer.deserialize(
            f.read()
        )
    )


positions_nm = np.load(
    POSITIONS_FILE
)


n_particles = (
    system.getNumParticles()
)


if positions_nm.shape != (
    n_particles,
    3,
):

    raise RuntimeError(
        "System/coordinate mismatch: "
        f"{n_particles} particles vs "
        f"{positions_nm.shape}"
    )


if not np.isfinite(
    positions_nm
).all():

    raise RuntimeError(
        "Coordinates contain NaN/Inf."
    )


water_sites = (
    n_particles
    - N_SOLUTE
)


if (
    water_sites
    % SITES_PER_WATER
) != 0:

    raise RuntimeError(
        "Water site count is not divisible by four."
    )


n_waters = (
    water_sites
    // SITES_PER_WATER
)


print()
print("=" * 72)
print("VMD MINIMIZED STRUCTURE PREPARATION")
print("=" * 72)

print()
print("SYSTEM")
print("------")

print(
    "Total simulation particles:",
    n_particles,
)

print(
    "Graphene carbons:",
    N_GRAPHENE_CARBONS,
)

print(
    "Pyrene-PEG5 atoms:",
    N_LIGAND,
)

print(
    "OPC waters:",
    n_waters,
)


# ============================================================
# 6. READ PERIODIC BOX
# ============================================================

a, b, c = (
    system.getDefaultPeriodicBoxVectors()
)


def vector_to_nm(v):

    return np.asarray(
        v.value_in_unit(
            unit.nanometer
        ),
        dtype=float,
    )


a_nm = vector_to_nm(a)
b_nm = vector_to_nm(b)
c_nm = vector_to_nm(c)


box = np.vstack(
    [
        a_nm,
        b_nm,
        c_nm,
    ]
)


box_inverse = np.linalg.inv(
    box
)


print()
print("PERIODIC BOX")
print("------------")

print(
    "a:",
    a_nm,
)

print(
    "b:",
    b_nm,
)

print(
    "c:",
    c_nm,
)


# ============================================================
# 7. MINIMUM-IMAGE HELPER
# ============================================================

def minimum_image_displacement(
    displacement,
):
    """
    Convert a displacement to its nearest periodic image.

    Positions and box vectors are represented as row vectors.
    """

    fractional = (
        displacement
        @ box_inverse
    )

    fractional -= np.round(
        fractional
    )

    return (
        fractional
        @ box
    )


# ============================================================
# 8. PARSE MOL2 ATOMS + BONDS
# ============================================================

mol2_lines = (
    LIGAND_MOL2
    .read_text()
    .splitlines()
)


atom_section = None
bond_section = None


for i, line in enumerate(
    mol2_lines
):

    if line.strip() == "@<TRIPOS>ATOM":

        atom_section = (
            i + 1
        )

    elif line.strip() == "@<TRIPOS>BOND":

        bond_section = (
            i + 1
        )


if atom_section is None:

    raise RuntimeError(
        "MOL2 ATOM section not found."
    )


if bond_section is None:

    raise RuntimeError(
        "MOL2 BOND section not found."
    )


# ============================================================
# 9. LIGAND ATOMS
# ============================================================

ligand_atoms = []


for line in mol2_lines[
    atom_section:
]:

    if line.startswith(
        "@<TRIPOS>"
    ):

        break

    if not line.strip():

        continue

    fields = line.split()

    if len(fields) < 6:

        continue

    atom_id = int(
        fields[0]
    )

    atom_name = (
        fields[1]
    )

    atom_type = (
        fields[5]
    )

    name_upper = (
        atom_name.upper()
    )

    if name_upper.startswith("H"):

        element = "H"

    elif name_upper.startswith("C"):

        element = "C"

    elif name_upper.startswith("N"):

        element = "N"

    elif name_upper.startswith("O"):

        element = "O"

    else:

        type_upper = (
            atom_type.upper()
        )

        if type_upper.startswith("H"):
            element = "H"

        elif type_upper.startswith("C"):
            element = "C"

        elif type_upper.startswith("N"):
            element = "N"

        elif type_upper.startswith("O"):
            element = "O"

        else:

            raise RuntimeError(
                "Could not determine element for "
                f"{atom_name} / {atom_type}"
            )

    ligand_atoms.append(
        {
            "id":
                atom_id,

            "name":
                atom_name,

            "type":
                atom_type,

            "element":
                element,
        }
    )


if len(
    ligand_atoms
) != N_LIGAND:

    raise RuntimeError(
        "Expected 70 ligand atoms, "
        f"found {len(ligand_atoms)}."
    )


# ============================================================
# 10. LIGAND BONDS
# ============================================================

ligand_bonds = []


for line in mol2_lines[
    bond_section:
]:

    if line.startswith(
        "@<TRIPOS>"
    ):

        break

    if not line.strip():

        continue

    fields = line.split()

    if len(fields) < 4:

        continue

    bond_id = int(
        fields[0]
    )

    atom_1 = int(
        fields[1]
    ) - 1

    atom_2 = int(
        fields[2]
    ) - 1

    bond_type = (
        fields[3]
    )

    ligand_bonds.append(
        {
            "id":
                bond_id,

            "a":
                atom_1,

            "b":
                atom_2,

            "type":
                bond_type,
        }
    )


print()
print("LIGAND TOPOLOGY")
print("----------------")

print(
    "Atoms:",
    len(
        ligand_atoms
    ),
)

print(
    "Bonds:",
    len(
        ligand_bonds
    ),
)


# ============================================================
# 11. BUILD LIGAND BOND GRAPH
# ============================================================

adjacency = {
    i:
        []

    for i in range(
        N_LIGAND
            )
}


for bond in ligand_bonds:

    i = bond[
        "a"
    ]

    j = bond[
        "b"
    ]

    adjacency[
        i
    ].append(
        j
    )

    adjacency[
        j
    ].append(
        i
    )


# ============================================================
# 12. ORIGINAL LIGAND COORDINATES
# ============================================================

ligand_original = (
    positions_nm[
        LIGAND_START:
        LIGAND_STOP
    ].copy()
)


# ============================================================
# 13. ORIGINAL DIRECT BOND DISTANCES
#
# These are what VMD effectively sees if the molecule happens
# to be split across periodic boundaries.
# ============================================================

original_direct_bond_lengths = []


for bond in ligand_bonds:

    i = bond[
        "a"
    ]

    j = bond[
        "b"
    ]

    distance = float(
        np.linalg.norm(
            ligand_original[j]
            -
            ligand_original[i]
        )
    )

    original_direct_bond_lengths.append(
        distance
    )


# ============================================================
# 14. TRUE PERIODIC BOND DISTANCES
# ============================================================

periodic_bond_lengths = []


for bond in ligand_bonds:

    i = bond[
        "a"
    ]

    j = bond[
        "b"
    ]

    displacement = (
        ligand_original[j]
        -
        ligand_original[i]
    )

    displacement = (
        minimum_image_displacement(
            displacement
        )
    )

    distance = float(
        np.linalg.norm(
            displacement
        )
    )

    periodic_bond_lengths.append(
        distance
    )


# ============================================================
# 15. RECONSTRUCT WHOLE LIGAND
#
# Traverse the chemical bond graph.
#
# For each bonded neighbor, use the nearest periodic copy of
# the bond displacement.
# ============================================================

ligand_unwrapped = np.full(
    (
        N_LIGAND,
        3,
    ),
    np.nan,
    dtype=float,
)


visited = np.zeros(
    N_LIGAND,
    dtype=bool,
)


root = 0


ligand_unwrapped[
    root
] = (
    ligand_original[
        root
    ]
)


visited[
    root
] = True


queue = deque(
    [
        root
    ]
)


while queue:

    current = (
        queue.popleft()
    )

    for neighbor in (
        adjacency[
            current
        ]
    ):

        displacement = (
            ligand_original[
                neighbor
            ]
            -
            ligand_original[
                current
            ]
        )

        displacement = (
            minimum_image_displacement(
                displacement
            )
        )

        proposed = (
            ligand_unwrapped[
                current
            ]
            +
            displacement
        )

        if not visited[
            neighbor
        ]:

            ligand_unwrapped[
                neighbor
            ] = proposed

            visited[
                neighbor
            ] = True

            queue.append(
                neighbor
            )

        else:

            disagreement = float(
                np.linalg.norm(
                    ligand_unwrapped[
                        neighbor
                    ]
                    -
                    proposed
                )
            )

            if disagreement > 1.0e-5:

                raise RuntimeError(
                    "Ligand unwrapping produced inconsistent "
                    "coordinates around a bond cycle. "
                    f"Atoms {current + 1}/{neighbor + 1}, "
                    f"disagreement={disagreement:.8f} nm."
                )


if not np.all(
    visited
):

    missing = np.where(
        ~visited
    )[0]

    raise RuntimeError(
        "Ligand bond graph is disconnected. "
        f"Unvisited atoms: {missing.tolist()}"
    )


# ============================================================
# 16. VERIFY UNWRAPPED BOND LENGTHS
# ============================================================

unwrapped_bond_lengths = []


for bond in ligand_bonds:

    i = bond[
        "a"
    ]

    j = bond[
        "b"
    ]

    distance = float(
        np.linalg.norm(
            ligand_unwrapped[j]
            -
            ligand_unwrapped[i]
        )
    )

    unwrapped_bond_lengths.append(
        distance
    )


periodic_bond_lengths = np.asarray(
    periodic_bond_lengths
)


unwrapped_bond_lengths = np.asarray(
    unwrapped_bond_lengths
)


bond_reconstruction_error = float(
    np.max(
        np.abs(
            unwrapped_bond_lengths
            -
            periodic_bond_lengths
        )
    )
)


if bond_reconstruction_error > 1.0e-6:

    raise RuntimeError(
        "Unwrapped ligand bond lengths do not match "
        "periodic bond lengths."
    )


# ============================================================
# 17. DETECT AROMATIC/PYRENE ATOMS
#
# First use MOL2 aromatic bonds.
#
# This is for visualization placement/diagnostics only.
# ============================================================

aromatic_atoms = set()


for bond in ligand_bonds:

    bond_type = str(
        bond[
            "type"
        ]
    ).lower()

    if bond_type in {
        "ar",
        "aro",
    }:

        aromatic_atoms.add(
            bond[
                "a"
            ]
        )

        aromatic_atoms.add(
            bond[
                "b"
            ]
        )


# If the MOL2 has GAFF-style atom types instead of aromatic
# bond labels, also look for common aromatic carbon types.
for i, atom in enumerate(
    ligand_atoms
):

    atom_type = (
        atom[
            "type"
        ].lower()
    )

    if atom_type in {
        "ca",
        "cp",
        "cq",
        "cc",
        "cd",
    }:

        if atom[
            "element"
        ] == "C":

            aromatic_atoms.add(
                i
            )


aromatic_atoms = sorted(
    aromatic_atoms
)


# ============================================================
# 18. GRAPHENE GEOMETRY
# ============================================================

graphene = (
    positions_nm[
        :N_GRAPHENE_CARBONS
    ]
)


graphene_center = np.mean(
    graphene,
    axis=0,
)


graphene_z_min = float(
    graphene[
        :,
        2
    ].min()
)

graphene_z_max = float(
    graphene[
        :,
        2
    ].max()
)

graphene_z_mean = float(
    graphene[
        :,
        2
    ].mean()
)


# ============================================================
# 19. CHOOSE LIGAND ANCHOR
#
# Prefer aromatic/pyrene atoms.
# Otherwise use all ligand heavy atoms.
# ============================================================

if len(
    aromatic_atoms
) >= 8:

    anchor_indices = np.asarray(
        aromatic_atoms,
        dtype=int,
    )

    anchor_description = (
        f"aromatic atoms ({len(aromatic_atoms)})"
    )


else:

    anchor_indices = np.asarray(
        [
            i
            for i, atom
            in enumerate(
                ligand_atoms
            )
            if atom[
                "element"
            ] != "H"
        ],
        dtype=int,
    )

    anchor_description = (
        "all ligand heavy atoms"
    )


ligand_anchor = np.mean(
    ligand_unwrapped[
        anchor_indices
    ],
    axis=0,
)


# ============================================================
# 20. CHOOSE BEST X/Y PERIODIC IMAGE
#
# Only translate by whole lattice vectors a and b.
#
# z is deliberately unchanged.
# ============================================================

best_shift = np.zeros(
    3,
    dtype=float,
)

best_nm = (
    0,
    0,
)

best_xy_distance = np.inf


for na in range(
    -3,
    4,
):

    for nb in range(
        -3,
        4,
    ):

        shift = (
            na * a_nm
            +
            nb * b_nm
        )

        candidate_anchor = (
            ligand_anchor
            +
            shift
        )

        xy_distance = float(
            np.linalg.norm(
                candidate_anchor[
                    :2
                ]
                -
                graphene_center[
                    :2
                ]
            )
        )

        if xy_distance < (
            best_xy_distance
        ):

            best_xy_distance = (
                xy_distance
            )

            best_shift = (
                shift
            )

            best_nm = (
                na,
                nb,
            )


ligand_vmd = (
    ligand_unwrapped
    +
    best_shift
)


# ============================================================
# 21. LIGAND/PYRENE Z DIAGNOSTICS
# ============================================================

ligand_z_min = float(
    ligand_vmd[
        :,
        2
    ].min()
)

ligand_z_max = float(
    ligand_vmd[
        :,
        2
    ].max()
)


if len(
    aromatic_atoms
) > 0:

    aromatic_z = (
        ligand_vmd[
            aromatic_atoms,
            2
        ]
    )

    aromatic_z_min = float(
        aromatic_z.min()
    )

    aromatic_z_max = float(
        aromatic_z.max()
    )

    aromatic_z_mean = float(
        aromatic_z.mean()
    )

    aromatic_height_above_graphene = (
        aromatic_z_mean
        -
        graphene_z_mean
    )


else:

    aromatic_z_min = None
    aromatic_z_max = None
    aromatic_z_mean = None
    aromatic_height_above_graphene = None


print()
print("=" * 72)
print("PERIODIC-WRAPPING DIAGNOSTIC")
print("=" * 72)

print()
print("ORIGINAL DIRECT BONDS")
print("---------------------")

print(
    f"Minimum: "
    f"{min(original_direct_bond_lengths):.6f} nm"
)

print(
    f"Maximum: "
    f"{max(original_direct_bond_lengths):.6f} nm"
)


print()
print("TRUE PERIODIC BONDS")
print("-------------------")

print(
    f"Minimum: "
    f"{periodic_bond_lengths.min():.6f} nm"
)

print(
    f"Maximum: "
    f"{periodic_bond_lengths.max():.6f} nm"
)


print()
print("UNWRAPPED BONDS")
print("----------------")

print(
    f"Minimum: "
    f"{unwrapped_bond_lengths.min():.6f} nm"
)

print(
    f"Maximum: "
    f"{unwrapped_bond_lengths.max():.6f} nm"
)

print(
    f"Maximum reconstruction error: "
    f"{bond_reconstruction_error:.3e} nm"
)


# ============================================================
# 22. IDENTIFY SUSPICIOUS DIRECT BONDS
#
# A covalent bond appearing >0.25 nm in direct coordinates is
# likely split across a periodic image.
# ============================================================

suspicious_direct_bonds = []


for bond_index, bond in enumerate(
    ligand_bonds
):

    direct_length = float(
        original_direct_bond_lengths[
            bond_index
        ]
    )

    periodic_length = float(
        periodic_bond_lengths[
            bond_index
        ]
    )

    if direct_length > 0.25:

        record = {
            "atom_1":
                int(
                    bond[
                        "a"
                    ]
                    + 1
                ),

            "atom_2":
                int(
                    bond[
                        "b"
                    ]
                    + 1
                ),

            "direct_length_nm":
                direct_length,

            "periodic_length_nm":
                periodic_length,
        }

        suspicious_direct_bonds.append(
            record
        )


print()
print("PERIODICALLY SPLIT DISPLAY BONDS")
print("--------------------------------")

print(
    "Count:",
    len(
        suspicious_direct_bonds
    ),
)


for record in (
    suspicious_direct_bonds[
        :10
    ]
):

    print(
        f"atoms "
        f"{record['atom_1']:2d} / "
        f"{record['atom_2']:2d} | "
        f"direct="
        f"{record['direct_length_nm']:.6f} nm | "
        f"periodic="
        f"{record['periodic_length_nm']:.6f} nm"
    )


# ============================================================
# 23. GEOMETRY REPORT
# ============================================================

print()
print("=" * 72)
print("MINIMIZED GEOMETRY")
print("=" * 72)

print()
print("GRAPHENE")
print("--------")

print(
    f"z minimum: "
    f"{graphene_z_min:.6f} nm"
)

print(
    f"z maximum: "
    f"{graphene_z_max:.6f} nm"
)

print(
    f"z mean:    "
    f"{graphene_z_mean:.6f} nm"
)


print()
print("PYRENE-PEG5")
print("-----------")

print(
    f"Whole ligand z minimum: "
    f"{ligand_z_min:.6f} nm"
)

print(
    f"Whole ligand z maximum: "
    f"{ligand_z_max:.6f} nm"
)

print(
    "Placement anchor:",
    anchor_description,
)

print(
    "Periodic image translation:",
    f"{best_nm[0]}*a + {best_nm[1]}*b",
)


if aromatic_z_mean is not None:

    print()
    print("AROMATIC / PYRENE GROUP")
    print("-----------------------")

    print(
        "Detected aromatic atoms:",
        len(
            aromatic_atoms
        ),
    )

    print(
        f"z minimum: "
        f"{aromatic_z_min:.6f} nm"
    )

    print(
        f"z maximum: "
        f"{aromatic_z_max:.6f} nm"
    )

    print(
        f"z mean:    "
        f"{aromatic_z_mean:.6f} nm"
    )

    print(
        f"Mean height above graphene: "
        f"{aromatic_height_above_graphene:.6f} nm"
    )


# ============================================================
# 24. PREPARE VISUALIZATION COORDINATES
# ============================================================

display_positions = (
    positions_nm.copy()
)


# Replace ONLY ligand coordinates in the visualization copy.
display_positions[
    LIGAND_START:
    LIGAND_STOP
] = (
    ligand_vmd
)


# ============================================================
# 25. PDB CELL PARAMETERS
# ============================================================

def vector_length(
    vector,
):

    return float(
        np.linalg.norm(
            vector
        )
    )


def vector_angle(
    v1,
    v2,
):

    cosine = (
        np.dot(
            v1,
            v2
        )
        /
        (
            np.linalg.norm(
                v1
            )
            *
            np.linalg.norm(
                v2
            )
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


cell_a_A = (
    10.0
    * vector_length(
        a_nm
    )
)

cell_b_A = (
    10.0
    * vector_length(
        b_nm
    )
)

cell_c_A = (
    10.0
    * vector_length(
        c_nm
    )
)


alpha = vector_angle(
    b_nm,
    c_nm,
)

beta = vector_angle(
    a_nm,
    c_nm,
)

gamma = vector_angle(
    a_nm,
    b_nm,
)


def cryst1_line():

    return (
        f"CRYST1"
        f"{cell_a_A:9.3f}"
        f"{cell_b_A:9.3f}"
        f"{cell_c_A:9.3f}"
        f"{alpha:7.2f}"
        f"{beta:7.2f}"
        f"{gamma:7.2f}"
        f" P 1           1\n"
    )


# ============================================================
# 26. PDB ATOM HELPER
# ============================================================

def pdb_atom_line(
    serial,
    name,
    resname,
    chain,
    resid,
    xyz_nm,
    element,
):

    xyz_A = (
        10.0
        * np.asarray(
            xyz_nm
        )
    )

    x, y, z = (
        xyz_A
    )

    return (
        f"ATOM  "
        f"{serial:5d} "
        f"{name[:4]:^4s}"
        f" "
        f"{resname:>3s} "
        f"{chain:1s}"
        f"{resid:4d}"
        f"    "
        f"{x:8.3f}"
        f"{y:8.3f}"
        f"{z:8.3f}"
        f"{1.00:6.2f}"
        f"{0.00:6.2f}"
        f"          "
        f"{element:>2s}"
        f"\n"
    )


# ============================================================
# 27. BUILD LIGAND CONNECTIVITY TABLE
# ============================================================

ligand_neighbors = {
    i:
        []

    for i in range(
        N_LIGAND
            )
}


for bond in ligand_bonds:

    ligand_neighbors[
        bond[
            "a"
        ]
    ].append(
        bond[
            "b"
        ]
    )

    ligand_neighbors[
        bond[
            "b"
        ]
    ].append(
        bond[
            "a"
        ]
    )


# ============================================================
# 28. WRITE DRY VMD FILE
#
# 1250 graphene C + 70 ligand atoms.
# ============================================================

DRY_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    DRY_OUTPUT,
    "w",
) as f:

    f.write(
        "REMARK  Visualization-only converged minimized system\n"
    )

    f.write(
        "REMARK  Pyrene-PEG5 reconstructed across PBC\n"
    )

    f.write(
        "REMARK  Graphene pi pseudo-sites hidden\n"
    )

    f.write(
        cryst1_line()
    )

    serial = 1

    # --------------------------------------------------------
    # GRAPHENE
    # --------------------------------------------------------

    for i in range(
        N_GRAPHENE_CARBONS
    ):

        f.write(
            pdb_atom_line(

                serial,

                "C",

                "GRP",

                "G",

                1,

                display_positions[
                    i
                ],

                "C",
            )
        )

        serial += 1

    # --------------------------------------------------------
    # LIGAND
    # --------------------------------------------------------

    ligand_serial_start = (
        serial
    )

    for local_index, atom in enumerate(
        ligand_atoms
    ):

        f.write(
            pdb_atom_line(

                serial,

                atom[
                    "name"
                ],

                "LIG",

                "L",

                1,

                ligand_vmd[
                    local_index
                ],

                atom[
                    "element"
                ],
            )
        )

        serial += 1

    # --------------------------------------------------------
    # LIGAND CONECT
    # --------------------------------------------------------

    for local_index in range(
        N_LIGAND
    ):

        center = (
            ligand_serial_start
            +
            local_index
        )

        neighbors = [
            ligand_serial_start
            +
            neighbor

            for neighbor in (
                ligand_neighbors[
                    local_index
                ]
            )
        ]

        if neighbors:

            f.write(
                f"CONECT"
                f"{center:5d}"
                +
                "".join(
                    f"{neighbor:5d}"
                    for neighbor
                    in neighbors
                )
                +
                "\n"
            )

    f.write(
        "END\n"
    )


# ============================================================
# 29. WRITE FULL VMD FILE
#
# Graphene + whole ligand + visible water O/H/H.
#
# M-sites remain hidden.
# ============================================================

with open(
    FULL_OUTPUT,
    "w",
) as f:

    f.write(
        "REMARK  Visualization-only converged minimized system\n"
    )

    f.write(
        "REMARK  Pyrene-PEG5 reconstructed across PBC\n"
    )

    f.write(
        "REMARK  Graphene pi sites and OPC M sites hidden\n"
    )

    f.write(
        cryst1_line()
    )

    serial = 1

    # Graphene
    for i in range(
        N_GRAPHENE_CARBONS
    ):

        f.write(
            pdb_atom_line(

                serial,

                "C",

                "GRP",

                "G",

                1,

                display_positions[
                    i
                ],

                "C",
            )
        )

        serial += 1

    # Ligand
    ligand_serial_start_full = (
        serial
    )

    for local_index, atom in enumerate(
        ligand_atoms
    ):

        f.write(
            pdb_atom_line(

                serial,

                atom[
                    "name"
                ],

                "LIG",

                "L",

                1,

                ligand_vmd[
                    local_index
                ],

                atom[
                    "element"
                ],
            )
        )

        serial += 1

    # Waters
    first_water_serial = (
        serial
    )

    for water_index in range(
        n_waters
    ):

        base = (
            N_SOLUTE
            +
            4
            * water_index
        )

        resid = (
            water_index
            + 1
        )

        f.write(
            pdb_atom_line(
                serial,
                "O",
                "HOH",
                "W",
                resid,
                positions_nm[
                    base
                ],
                "O",
            )
        )

        serial += 1

        f.write(
            pdb_atom_line(
                serial,
                "H1",
                "HOH",
                "W",
                resid,
                positions_nm[
                    base + 1
                ],
                "H",
            )
        )

        serial += 1

        f.write(
            pdb_atom_line(
                serial,
                "H2",
                "HOH",
                "W",
                resid,
                positions_nm[
                    base + 2
                ],
                "H",
            )
        )

        serial += 1

    # Ligand connectivity
    for local_index in range(
        N_LIGAND
    ):

        center = (
            ligand_serial_start_full
            +
            local_index
        )

        neighbors = [
            ligand_serial_start_full
            +
            neighbor

            for neighbor in (
                ligand_neighbors[
                    local_index
                ]
            )
        ]

        if neighbors:

            f.write(
                f"CONECT"
                f"{center:5d}"
                +
                "".join(
                    f"{neighbor:5d}"
                    for neighbor
                    in neighbors
                )
                +
                "\n"
            )

    # Water connectivity
    for water_index in range(
        n_waters
    ):

        oxygen_serial = (
            first_water_serial
            +
            3
            * water_index
        )

        h1_serial = (
            oxygen_serial
            + 1
        )

        h2_serial = (
            oxygen_serial
            + 2
        )

        f.write(
            f"CONECT"
            f"{oxygen_serial:5d}"
            f"{h1_serial:5d}"
            f"{h2_serial:5d}"
            f"\n"
        )

    f.write(
        "END\n"
    )


# ============================================================
# 30. METADATA
# ============================================================

METADATA_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "status":
        "PASS",

    "ligand_atoms":
        N_LIGAND,

    "ligand_bonds":
        len(
            ligand_bonds
        ),

    "suspicious_original_direct_bonds":
        suspicious_direct_bonds,

    "number_suspicious_original_direct_bonds":
        len(
            suspicious_direct_bonds
        ),

    "maximum_original_direct_bond_nm":
        float(
            max(
                original_direct_bond_lengths
            )
        ),

    "maximum_periodic_bond_nm":
        float(
            periodic_bond_lengths.max()
        ),

    "maximum_unwrapped_bond_nm":
        float(
            unwrapped_bond_lengths.max()
        ),

    "bond_reconstruction_error_nm":
        bond_reconstruction_error,

    "graphene_z_min_nm":
        graphene_z_min,

    "graphene_z_max_nm":
        graphene_z_max,

    "graphene_z_mean_nm":
        graphene_z_mean,

    "ligand_z_min_nm":
        ligand_z_min,

    "ligand_z_max_nm":
        ligand_z_max,

    "aromatic_atom_count":
        len(
            aromatic_atoms
        ),

    "aromatic_atoms_one_based":
        [
            int(
                i + 1
            )
            for i in (
                aromatic_atoms
            )
        ],

    "aromatic_z_min_nm":
        aromatic_z_min,

    "aromatic_z_max_nm":
        aromatic_z_max,

    "aromatic_z_mean_nm":
        aromatic_z_mean,

    "aromatic_mean_height_above_graphene_nm":
        aromatic_height_above_graphene,

    "periodic_xy_translation":
        {
            "a_multiple":
                int(
                    best_nm[
                        0
                    ]
                ),

            "b_multiple":
                int(
                    best_nm[
                        1
                    ]
                ),

            "translation_nm":
                best_shift.tolist(),
        },

    "simulation_coordinates_modified":
        False,

    "md_run":
        False,

    "heating_run":
        False,
}


with open(
    METADATA_OUTPUT,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# 31. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("VMD FILES")
print("=" * 72)

print()
print("DRY:")
print(
    DRY_OUTPUT
)

print()
print("FULL WITH WATER:")
print(
    FULL_OUTPUT
)

print()
print("METADATA:")
print(
    METADATA_OUTPUT
)


print()
print("=" * 72)
print("VMD MINIMIZED STRUCTURE PREPARATION: PASS")
print("=" * 72)

print()
print(
    "Simulation coordinates were NOT modified."
)

print(
    "Ligand was only reconstructed for visualization."
)

print(
    "No MD was run."
)

print(
    "No heating was run."
)

from pathlib import Path
import math

import numpy as np

from openmm import openmm, unit


# ============================================================
# EXPORT CONVERGED MINIMIZED SYSTEM FOR VISUALIZATION
#
# INPUT
# -----
# Fully converged minimized OpenMM coordinates.
#
# OUTPUT
# ------
# 1. Full visualization:
#       graphene + Pyrene-PEG5 + visible OPC atoms
#
# 2. Dry visualization:
#       graphene + Pyrene-PEG5 only
#
# HIDDEN FROM VISUALIZATION
# -------------------------
# - IFF graphene pi pseudo-particles
# - OPC M virtual sites
#
# NO SIMULATION FILES ARE MODIFIED.
# NO MD IS RUN.
# ============================================================


# ============================================================
# 1. INDEXING
# ============================================================

N_GRAPHENE_CARBONS = 1250
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70
N_SOLUTE = 3820

SITES_PER_WATER = 4

WATER_O = 0
WATER_H1 = 1
WATER_H2 = 2
WATER_M = 3


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

FULL_PDB = Path(
    "structures/"
    "pyrene_peg5_graphene_minimized_full.pdb"
)

DRY_PDB = Path(
    "structures/"
    "pyrene_peg5_graphene_minimized_dry.pdb"
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
# 5. LOAD SYSTEM + POSITIONS
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
        "Position/System mismatch: "
        f"{positions_nm.shape} vs {n_particles}"
    )


if not np.isfinite(
    positions_nm
).all():

    raise RuntimeError(
        "Coordinates contain NaN/Inf."
    )


n_water_sites = (
    n_particles
    - N_SOLUTE
)


if (
    n_water_sites
    % SITES_PER_WATER
) != 0:

    raise RuntimeError(
        "Water site count is not divisible by four."
    )


n_waters = (
    n_water_sites
    // SITES_PER_WATER
)


print()
print("=" * 72)
print("MINIMIZED STRUCTURE VISUALIZATION EXPORT")
print("=" * 72)

print()
print("SYSTEM")
print("------")

print(
    "Total simulation particles:",
    n_particles,
)

print(
    "Graphene carbons shown:",
    N_GRAPHENE_CARBONS,
)

print(
    "Graphene pi sites hidden:",
    N_GRAPHENE_TOTAL
    - N_GRAPHENE_CARBONS,
)

print(
    "Pyrene-PEG5 atoms shown:",
    N_LIGAND,
)

print(
    "OPC waters:",
    n_waters,
)

print(
    "OPC M sites hidden:",
    n_waters,
)


# ============================================================
# 6. PARSE LIGAND MOL2
#
# We use the MOL2 only for:
#
#   - ligand atom names
#   - chemical elements
#   - ligand connectivity
#
# Coordinates come ONLY from the converged OpenMM structure.
# ============================================================

lines = (
    LIGAND_MOL2
    .read_text()
    .splitlines()
)


atom_start = None
bond_start = None


for i, line in enumerate(
    lines
):

    stripped = line.strip()

    if stripped == "@<TRIPOS>ATOM":

        atom_start = (
            i + 1
        )

    elif stripped == "@<TRIPOS>BOND":

        bond_start = (
            i + 1
        )


if atom_start is None:

    raise RuntimeError(
        "Could not find MOL2 ATOM section."
    )


if bond_start is None:

    raise RuntimeError(
        "Could not find MOL2 BOND section."
    )


# ============================================================
# 7. MOL2 ATOMS
# ============================================================

ligand_atoms = []


for line in lines[
    atom_start:
]:

    if line.startswith(
        "@<TRIPOS>"
    ):

        break

    if not line.strip():

        continue

    fields = (
        line.split()
    )

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

    # Molecule contains only:
    #
    # C30 H33 N O6
    #
    # Determine display element robustly from atom name/type.
    name_upper = (
        atom_name.upper()
    )

    if name_upper.startswith("H"):

        element = "H"

    elif name_upper.startswith("O"):

        element = "O"

    elif name_upper.startswith("N"):

        element = "N"

    elif name_upper.startswith("C"):

        element = "C"

    else:

        type_upper = (
            atom_type.upper()
        )

        if type_upper.startswith("H"):

            element = "H"

        elif type_upper.startswith("O"):

            element = "O"

        elif type_upper.startswith("N"):

            element = "N"

        elif type_upper.startswith("C"):

            element = "C"

        else:

            raise RuntimeError(
                "Could not determine element for "
                f"MOL2 atom {atom_id}: "
                f"{atom_name} / {atom_type}"
            )

    ligand_atoms.append(
        {
            "id":
                atom_id,

            "name":
                atom_name[:4],

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


print()
print("LIGAND TOPOLOGY")
print("----------------")

print(
    "MOL2 ligand atoms:",
    len(
        ligand_atoms
    ),
)


# ============================================================
# 8. MOL2 BONDS
# ============================================================

ligand_bonds = []


for line in lines[
    bond_start:
]:

    if line.startswith(
        "@<TRIPOS>"
    ):

        break

    if not line.strip():

        continue

    fields = (
        line.split()
    )

    if len(fields) < 4:

        continue

    atom_1 = int(
        fields[1]
    )

    atom_2 = int(
        fields[2]
    )

    ligand_bonds.append(
        (
            atom_1,
            atom_2,
        )
    )


print(
    "MOL2 ligand bonds:",
    len(
        ligand_bonds
    ),
)


# ============================================================
# 9. PERIODIC CELL FOR PDB CRYST1
# ============================================================

a, b, c = (
    system.getDefaultPeriodicBoxVectors()
)


def vec_nm(
    vector
):

    return np.asarray(
        vector.value_in_unit(
            unit.nanometer
        ),
        dtype=float,
    )


a_nm = vec_nm(a)
b_nm = vec_nm(b)
c_nm = vec_nm(c)


def vector_length(
    v
):

    return float(
        np.linalg.norm(
            v
        )
    )


def angle_degrees(
    v1,
    v2,
):

    cosine = (
        np.dot(
            v1,
            v2,
        )
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


a_angstrom = (
    10.0
    * vector_length(
        a_nm
    )
)

b_angstrom = (
    10.0
    * vector_length(
        b_nm
    )
)

c_angstrom = (
    10.0
    * vector_length(
        c_nm
    )
)


alpha = angle_degrees(
    b_nm,
    c_nm,
)

beta = angle_degrees(
    a_nm,
    c_nm,
)

gamma = angle_degrees(
    a_nm,
    b_nm,
)


print()
print("PDB PERIODIC CELL")
print("-----------------")

print(
    f"a = {a_angstrom:.3f} A"
)

print(
    f"b = {b_angstrom:.3f} A"
)

print(
    f"c = {c_angstrom:.3f} A"
)

print(
    f"alpha = {alpha:.3f} deg"
)

print(
    f"beta  = {beta:.3f} deg"
)

print(
    f"gamma = {gamma:.3f} deg"
)


# ============================================================
# 10. PDB HELPERS
# ============================================================

def cryst1_line():

    return (
        f"CRYST1"
        f"{a_angstrom:9.3f}"
        f"{b_angstrom:9.3f}"
        f"{c_angstrom:9.3f}"
        f"{alpha:7.2f}"
        f"{beta:7.2f}"
        f"{gamma:7.2f}"
        f" P 1           1\n"
    )


def pdb_atom_line(
    serial,
    atom_name,
    residue_name,
    chain,
    residue_number,
    xyz_nm,
    element_name,
):

    x, y, z = (
        10.0
        * np.asarray(
            xyz_nm,
            dtype=float,
        )
    )

    atom_name = (
        atom_name[:4]
    )

    return (
        f"ATOM  "
        f"{serial:5d} "
        f"{atom_name:^4s}"
        f" "
        f"{residue_name:>3s} "
        f"{chain:1s}"
        f"{residue_number:4d}"
        f"    "
        f"{x:8.3f}"
        f"{y:8.3f}"
        f"{z:8.3f}"
        f"{1.00:6.2f}"
        f"{0.00:6.2f}"
        f"          "
        f"{element_name:>2s}"
        f"\n"
    )


# ============================================================
# 11. BUILD DISPLAY ATOM LIST
# ============================================================

display_atoms = []


# ------------------------------------------------------------
# GRAPHENE REAL CARBONS ONLY
# ------------------------------------------------------------

for i in range(
    N_GRAPHENE_CARBONS
):

    display_atoms.append(
        {
            "source_index":
                i,

            "name":
                "C",

            "resname":
                "GRP",

            "chain":
                "G",

            "resid":
                1,

            "element":
                "C",

            "kind":
                "graphene",
        }
    )


# ------------------------------------------------------------
# PYRENE-PEG5
# ------------------------------------------------------------

for local_index, atom in enumerate(
    ligand_atoms
):

    source_index = (
        N_GRAPHENE_TOTAL
        + local_index
    )

    display_atoms.append(
        {
            "source_index":
                source_index,

            "name":
                atom[
                    "name"
                ],

            "resname":
                "LIG",

            "chain":
                "L",

            "resid":
                1,

            "element":
                atom[
                    "element"
                ],

            "kind":
                "ligand",
        }
    )


# ------------------------------------------------------------
# OPC WATER: SHOW O/H/H; HIDE M
# ------------------------------------------------------------

for water_index in range(
    n_waters
):

    base_index = (
        N_SOLUTE
        +
        water_index
        * SITES_PER_WATER
    )

    residue_number = (
        water_index
        + 1
    )

    display_atoms.extend(
        [
            {
                "source_index":
                    base_index
                    + WATER_O,

                "name":
                    "O",

                "resname":
                    "HOH",

                "chain":
                    "W",

                "resid":
                    residue_number,

                "element":
                    "O",

                "kind":
                    "water",
            },

            {
                "source_index":
                    base_index
                    + WATER_H1,

                "name":
                    "H1",

                "resname":
                    "HOH",

                "chain":
                    "W",

                "resid":
                    residue_number,

                "element":
                    "H",

                "kind":
                    "water",
            },

            {
                "source_index":
                    base_index
                    + WATER_H2,

                "name":
                    "H2",

                "resname":
                    "HOH",

                "chain":
                    "W",

                "resid":
                    residue_number,

                "element":
                    "H",

                "kind":
                    "water",
            },
        ]
    )


# ============================================================
# 12. WRITE FULL PDB
# ============================================================

FULL_PDB.parent.mkdir(
    parents=True,
    exist_ok=True,
)


serial_for_source = {}


with open(
    FULL_PDB,
    "w",
) as f:

    f.write(
        "REMARK  Minimized supported graphene + "
        "Pyrene-PEG5 + OPC water\n"
    )

    f.write(
        "REMARK  IFF pi pseudo-sites are hidden\n"
    )

    f.write(
        "REMARK  OPC M virtual sites are hidden\n"
    )

    f.write(
        cryst1_line()
    )

    for serial, atom in enumerate(
        display_atoms,
        start=1,
    ):

        source_index = (
            atom[
                "source_index"
            ]
        )

        serial_for_source[
            source_index
        ] = serial

        f.write(
            pdb_atom_line(

                serial,

                atom[
                    "name"
                ],

                atom[
                    "resname"
                ],

                atom[
                    "chain"
                ],

                atom[
                    "resid"
                ],

                positions_nm[
                    source_index
                ],

                atom[
                    "element"
                ],
            )
        )

    # --------------------------------------------------------
    # Explicit ligand connectivity
    # --------------------------------------------------------

    ligand_serial_start = (
        N_GRAPHENE_CARBONS
        + 1
    )

    ligand_neighbors = {
        i:
            []

        for i in range(
            1,
            N_LIGAND
            + 1
                )
    }

    for atom_1, atom_2 in (
        ligand_bonds
    ):

        ligand_neighbors[
            atom_1
        ].append(
            atom_2
        )

        ligand_neighbors[
            atom_2
        ].append(
            atom_1
        )

    for local_atom in range(
        1,
        N_LIGAND
        + 1
    ):

        center = (
            ligand_serial_start
            + local_atom
            - 1
        )

        neighbors = [
            ligand_serial_start
            + neighbor
            - 1

            for neighbor
            in ligand_neighbors[
                local_atom
            ]
        ]

        if neighbors:

            f.write(
                f"CONECT{center:5d}"
                +
                "".join(
                    f"{n:5d}"
                    for n in neighbors
                )
                +
                "\n"
            )

    # --------------------------------------------------------
    # Explicit water O-H connectivity
    # --------------------------------------------------------

    first_water_serial = (
        N_GRAPHENE_CARBONS
        + N_LIGAND
        + 1
    )

    for water_index in range(
        n_waters
    ):

        o_serial = (
            first_water_serial
            +
            water_index
            * 3
        )

        h1_serial = (
            o_serial
            + 1
        )

        h2_serial = (
            o_serial
            + 2
        )

        f.write(
            f"CONECT"
            f"{o_serial:5d}"
            f"{h1_serial:5d}"
            f"{h2_serial:5d}"
            f"\n"
        )

    f.write(
        "END\n"
    )


# ============================================================
# 13. WRITE DRY PDB
#
# Graphene + Pyrene-PEG5 only.
# ============================================================

dry_atoms = [
    atom

    for atom in display_atoms

    if atom[
        "kind"
    ] in {
        "graphene",
        "ligand",
    }
]


with open(
    DRY_PDB,
    "w",
) as f:

    f.write(
        "REMARK  Minimized graphene + Pyrene-PEG5\n"
    )

    f.write(
        "REMARK  Waters and force-field pseudo-sites hidden\n"
    )

    f.write(
        cryst1_line()
    )

    for serial, atom in enumerate(
        dry_atoms,
        start=1,
    ):

        source_index = (
            atom[
                "source_index"
            ]
        )

        f.write(
            pdb_atom_line(

                serial,

                atom[
                    "name"
                ],

                atom[
                    "resname"
                ],

                atom[
                    "chain"
                ],

                atom[
                    "resid"
                ],

                positions_nm[
                    source_index
                ],

                atom[
                    "element"
                ],
            )
        )

    ligand_serial_start = (
        N_GRAPHENE_CARBONS
        + 1
    )

    for local_atom in range(
        1,
        N_LIGAND
        + 1
    ):

        center = (
            ligand_serial_start
            + local_atom
            - 1
        )

        neighbors = [
            ligand_serial_start
            + neighbor
            - 1

            for neighbor
            in ligand_neighbors[
                local_atom
            ]
        ]

        if neighbors:

            f.write(
                f"CONECT{center:5d}"
                +
                "".join(
                    f"{n:5d}"
                    for n in neighbors
                )
                +
                "\n"
            )

    f.write(
        "END\n"
    )


# ============================================================
# 14. FINAL COUNTS
# ============================================================

n_visible_full = (
    len(
        display_atoms
    )
)


n_visible_dry = (
    len(
        dry_atoms
    )
)


expected_full = (
    N_GRAPHENE_CARBONS
    +
    N_LIGAND
    +
    3
    * n_waters
)


if n_visible_full != expected_full:

    raise RuntimeError(
        "Unexpected full visualization atom count."
    )


expected_dry = (
    N_GRAPHENE_CARBONS
    +
    N_LIGAND
)


if n_visible_dry != expected_dry:

    raise RuntimeError(
        "Unexpected dry visualization atom count."
    )


# ============================================================
# 15. REPORT
# ============================================================

print()
print("=" * 72)
print("VISUALIZATION FILES")
print("=" * 72)

print()
print(
    "FULL:"
)

print(
    FULL_PDB
)

print(
    "Visible atoms:",
    n_visible_full,
)


print()
print(
    "DRY:"
)

print(
    DRY_PDB
)

print(
    "Visible atoms:",
    n_visible_dry,
)


print()
print("DISPLAY CONTENT")
print("---------------")

print(
    f"Graphene carbons: "
    f"{N_GRAPHENE_CARBONS}"
)

print(
    f"Pyrene-PEG5 atoms: "
    f"{N_LIGAND}"
)

print(
    f"Visible water atoms: "
    f"{3 * n_waters}"
)

print(
    f"Hidden graphene pi sites: "
    f"{N_GRAPHENE_TOTAL - N_GRAPHENE_CARBONS}"
)

print(
    f"Hidden OPC M sites: "
    f"{n_waters}"
)


print()
print("=" * 72)
print("MINIMIZED VISUALIZATION EXPORT: PASS")
print("=" * 72)

print()
print(
    "Simulation coordinates were not modified."
)

print(
    "No MD was run."
)

print(
    "These PDB files are visualization-only."
)

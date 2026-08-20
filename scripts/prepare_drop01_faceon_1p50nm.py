from pathlib import Path
import json
import math

import numpy as np

from openmm import XmlSerializer, unit, openmm


# ============================================================
# DROP-01 INITIAL CONFIGURATION
#
# Face-on pyrene, 1.50 nm above equilibrated graphene.
#
# NO MD
# NO MINIMIZATION
# NO RESTRAINTS
# NO PULLING
# NO VELOCITY GENERATION
#
# Only:
#   1. extract equilibrated graphene/water background
#   2. insert extended minimized ligand by rigid translation
#   3. remove whole OPC waters with LJ-core overlaps
#   4. save coordinates + VMD inspection PDB
# ============================================================


TARGET_PYRENE_HEIGHT_NM = 1.500

N_GRAPHENE_CARBONS = 1250
N_GRAPHENE_TOTAL = 3750
N_LIGAND = 70
N_SOLUTE = N_GRAPHENE_TOTAL + N_LIGAND

SITES_PER_WATER = 4

LIGAND_START = N_GRAPHENE_TOTAL
LIGAND_STOP = N_SOLUTE


SYSTEM_XML = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb.xml"
)

MINIMIZED_POSITIONS = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb_"
    "minimized_positions_nm.npy"
)

EQUILIBRATED_POSITIONS = Path(
    "parameters/combined/"
    "ff3_pyrene_peg5_graphene_wallfree_opc_yb_"
    "equilibrated_300K_600ps_total_positions_nm.npy"
)

MOL2 = Path(
    "structures/"
    "pyrene_peg5_propargyl.mol2"
)


OUTPUT_SOLUTE = Path(
    "parameters/combined/"
    "drop01_faceon_1p50nm_supported_solute_positions_nm.npy"
)

OUTPUT_WATER = Path(
    "parameters/combined/"
    "drop01_faceon_1p50nm_opc_water_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "drop01_faceon_1p50nm_initialization.json"
)

OUTPUT_PDB = Path(
    "visualization/"
    "drop01_faceon_1p50nm_initial.pdb"
)


# ============================================================
# 1. LOAD
# ============================================================

system = XmlSerializer.deserialize(
    SYSTEM_XML.read_text()
)

min_xyz = np.load(
    MINIMIZED_POSITIONS
)

eq_xyz = np.load(
    EQUILIBRATED_POSITIONS
)


if min_xyz.shape[0] < N_SOLUTE:
    raise RuntimeError(
        "Minimized position array is too short."
    )

if (
    eq_xyz.shape[0] - N_SOLUTE
) % SITES_PER_WATER != 0:
    raise RuntimeError(
        "Equilibrated water-site count is not divisible by 4."
    )


n_waters_original = (
    eq_xyz.shape[0] - N_SOLUTE
) // SITES_PER_WATER


print()
print("=" * 78)
print("DROP-01 INITIAL-CONDITION BUILD")
print("=" * 78)
print()
print(
    "Original equilibrated waters:",
    n_waters_original,
)


# ============================================================
# 2. PARSE PYRENE ATOMS FROM MOL2
# ============================================================

lines = MOL2.read_text().splitlines()

atom_start = None
bond_start = None

for i, line in enumerate(lines):

    text = line.strip()

    if text == "@<TRIPOS>ATOM":
        atom_start = i + 1

    elif text == "@<TRIPOS>BOND":
        bond_start = i + 1


if atom_start is None or bond_start is None:
    raise RuntimeError(
        "Could not find MOL2 ATOM/BOND sections."
    )


atoms = []

for line in lines[atom_start:]:

    if line.startswith("@<TRIPOS>"):
        break

    if not line.strip():
        continue

    f = line.split()

    atoms.append(
        {
            "name": f[1],
            "type": f[5].lower(),
        }
    )


if len(atoms) != N_LIGAND:
    raise RuntimeError(
        f"Expected {N_LIGAND} ligand atoms; "
        f"found {len(atoms)}."
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

    bond_type = f[3].lower()

    if bond_type in {
        "ar",
        "aro",
    }:
        aromatic.add(a)
        aromatic.add(b)


for i, atom in enumerate(atoms):

    if atom["type"] in {
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
        "Expected exactly 16 pyrene aromatic atoms, "
        f"found {len(aromatic)}."
    )


# ============================================================
# 3. FIT EQUILIBRATED GRAPHENE PLANE
# ============================================================

graphene_c = (
    eq_xyz[
        :N_GRAPHENE_CARBONS
    ].copy()
)

graphene_center = (
    graphene_c.mean(
        axis=0
    )
)

graphene_centered = (
    graphene_c
    -
    graphene_center
)

_, _, vh = np.linalg.svd(
    graphene_centered,
    full_matrices=False,
)

graphene_normal = vh[-1]

graphene_normal /= np.linalg.norm(
    graphene_normal
)

if graphene_normal[2] < 0:
    graphene_normal *= -1.0


# ============================================================
# 4. CREATE EXTENDED DROP LIGAND
#
# Start from minimized conformation.
#
# Preserve the lateral adsorption registry of the final
# equilibrated structure, but move the pyrene rigidly to
# exactly 1.50 nm above the fitted graphene plane.
# ============================================================

ligand_min = (
    min_xyz[
        LIGAND_START:
        LIGAND_STOP
    ].copy()
)

ligand_eq = (
    eq_xyz[
        LIGAND_START:
        LIGAND_STOP
    ]
)


pyrene_min_center = (
    ligand_min[
        aromatic
    ].mean(
        axis=0
    )
)

pyrene_eq_center = (
    ligand_eq[
        aromatic
    ].mean(
        axis=0
    )
)


delta_to_eq = (
    pyrene_eq_center
    -
    pyrene_min_center
)


# Keep only displacement parallel to graphene.
lateral_shift = (
    delta_to_eq
    -
    np.dot(
        delta_to_eq,
        graphene_normal,
    )
    *
    graphene_normal
)


ligand_drop = (
    ligand_min
    +
    lateral_shift
)


pyrene_center = (
    ligand_drop[
        aromatic
    ].mean(
        axis=0
    )
)


current_height = float(
    np.dot(
        pyrene_center
        -
        graphene_center,
        graphene_normal,
    )
)


vertical_shift = (
    TARGET_PYRENE_HEIGHT_NM
    -
    current_height
) * graphene_normal


ligand_drop += vertical_shift


# Verify.
pyrene_center = (
    ligand_drop[
        aromatic
    ].mean(
        axis=0
    )
)


drop_height = float(
    np.dot(
        pyrene_center
        -
        graphene_center,
        graphene_normal,
    )
)


# ============================================================
# 5. PYRENE TILT
# ============================================================

pyrene_centered = (
    ligand_drop[
        aromatic
    ]
    -
    ligand_drop[
        aromatic
    ].mean(
        axis=0
    )
)

_, _, pvh = np.linalg.svd(
    pyrene_centered,
    full_matrices=False,
)

pyrene_normal = pvh[-1]

pyrene_normal /= np.linalg.norm(
    pyrene_normal
)


cos_angle = float(
    np.clip(
        abs(
            np.dot(
                pyrene_normal,
                graphene_normal,
            )
        ),
        0.0,
        1.0,
    )
)

drop_tilt_deg = float(
    np.degrees(
        np.arccos(
            cos_angle
        )
    )
)


# ============================================================
# 6. BOX VECTORS
# ============================================================

a, b, c = (
    system.getDefaultPeriodicBoxVectors()
)


def vector_nm(v):

    try:
        q = v.value_in_unit(
            unit.nanometer
        )

        return np.asarray(
            [
                q[0],
                q[1],
                q[2],
            ],
            dtype=float,
        )

    except AttributeError:

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
        vector_nm(a),
        vector_nm(b),
        vector_nm(c),
    ]
)

inv_box = np.linalg.inv(
    box
)


def minimum_image(delta):

    fractional = (
        delta
        @
        inv_box
    )

    fractional -= np.round(
        fractional
    )

    return (
        fractional
        @
        box
    )


# ============================================================
# 7. EXTRACT EQUILIBRATED OPC WATERS
# ============================================================

waters = (
    eq_xyz[
        N_SOLUTE:
    ]
    .reshape(
        n_waters_original,
        SITES_PER_WATER,
        3,
    )
    .copy()
)

oxygen_xyz = (
    waters[
        :,
        0,
        :
    ]
)


# ============================================================
# 8. GET ACTUAL FF-3 LJ PARAMETERS
#
# Remove a water if its oxygen enters the LJ minimum
#
#   r_min = 2^(1/6) * sigma_ij
#
# of any ligand atom carrying nonzero LJ epsilon.
#
# This changes INITIAL CONDITIONS only.
# ============================================================

nb = None

for force in system.getForces():

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):
        nb = force
        break


if nb is None:
    raise RuntimeError(
        "Could not locate NonbondedForce."
    )


q_o, sigma_o, epsilon_o = (
    nb.getParticleParameters(
        N_SOLUTE
    )
)

sigma_o_nm = float(
    sigma_o.value_in_unit(
        unit.nanometer
    )
)

epsilon_o_value = float(
    epsilon_o.value_in_unit(
        unit.kilojoule_per_mole
    )
)


if epsilon_o_value <= 0:
    raise RuntimeError(
        "First OPC oxygen has no positive LJ epsilon."
    )


remove_water = np.zeros(
    n_waters_original,
    dtype=bool,
)

ligand_lj_atoms = []


for local_i in range(
    N_LIGAND
):

    global_i = (
        LIGAND_START
        +
        local_i
    )

    _, sigma_i, epsilon_i = (
        nb.getParticleParameters(
            global_i
        )
    )

    epsilon_value = float(
        epsilon_i.value_in_unit(
            unit.kilojoule_per_mole
        )
    )

    if epsilon_value <= 1.0e-12:
        continue

    sigma_i_nm = float(
        sigma_i.value_in_unit(
            unit.nanometer
        )
    )

    sigma_mix = (
        sigma_i_nm
        +
        sigma_o_nm
    ) / 2.0

    rmin = (
        2.0 ** (1.0 / 6.0)
        *
        sigma_mix
    )

    delta = (
        oxygen_xyz
        -
        ligand_drop[
            local_i
        ]
    )

    delta = minimum_image(
        delta
    )

    distances = np.linalg.norm(
        delta,
        axis=1,
    )

    overlap = (
        distances
        <
        rmin
    )

    remove_water |= overlap

    ligand_lj_atoms.append(
        {
            "local_index":
                local_i,

            "sigma_nm":
                sigma_i_nm,

            "rmin_with_OPC_O_nm":
                rmin,
        }
    )


n_removed = int(
    remove_water.sum()
)

waters_pruned = (
    waters[
        ~remove_water
    ]
)

n_waters_final = int(
    waters_pruned.shape[0]
)

prune_fraction = (
    n_removed
    /
    n_waters_original
)


# ============================================================
# 9. FINAL SOLUTE ARRAY
# ============================================================

solute_drop = (
    eq_xyz[
        :N_SOLUTE
    ].copy()
)

solute_drop[
    LIGAND_START:
    LIGAND_STOP
] = ligand_drop


# ============================================================
# 10. GEOMETRY CHECKS
# ============================================================

graphene_z_max = float(
    graphene_c[
        :,
        2
    ].max()
)

oxygen_after = (
    waters_pruned[
        :,
        0,
        :
    ]
)

waters_below_graphene = int(
    np.sum(
        oxygen_after[
            :,
            2
        ]
        <
        graphene_z_max
    )
)


ligand_heights = (
    (
        ligand_drop
        -
        graphene_center
    )
    @
    graphene_normal
)


terminal_heights = {
    str(i):
        float(
            ligand_heights[i]
        )

    for i in [
        35,
        36,
        69,
    ]
}


if abs(
    drop_height
    -
    TARGET_PYRENE_HEIGHT_NM
) > 1.0e-8:

    raise RuntimeError(
        "Target pyrene height was not reproduced."
    )


if waters_below_graphene != 0:

    raise RuntimeError(
        "Unexpected water below graphene."
    )


if prune_fraction > 0.025:

    raise RuntimeError(
        "More than 2.5% of waters required pruning: "
        f"{100.0 * prune_fraction:.3f}%."
    )


# ============================================================
# 11. SAVE ARRAYS
# ============================================================

OUTPUT_SOLUTE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_WATER.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PDB.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_SOLUTE,
    solute_drop,
)

np.save(
    OUTPUT_WATER,
    waters_pruned,
)


# ============================================================
# 12. SIMPLE GRAPHENE + LIGAND PDB FOR VMD
# ============================================================

def element_from_type(t):

    t = t.lower()

    if t.startswith("c"):
        return "C"

    if t.startswith("n"):
        return "N"

    if t.startswith("o"):
        return "O"

    if t.startswith("h"):
        return "H"

    if t.startswith("s"):
        return "S"

    return t[0].upper()


def pdb_atom(
    serial,
    name,
    resname,
    chain,
    resid,
    position_nm,
    element,
):

    x, y, z = (
        position_nm
        *
        10.0
    )

    return (
        f"ATOM  {serial:5d} "
        f"{name[:4]:^4s} "
        f"{resname:>3s} "
        f"{chain}{resid:4d}    "
        f"{x:8.3f}"
        f"{y:8.3f}"
        f"{z:8.3f}"
        f"{1.00:6.2f}"
        f"{0.00:6.2f}"
        f"          "
        f"{element:>2s}"
    )


def angle_deg(v1, v2):

    cval = float(
        np.dot(v1, v2)
        /
        (
            np.linalg.norm(v1)
            *
            np.linalg.norm(v2)
        )
    )

    cval = np.clip(
        cval,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cval
            )
        )
    )


la = np.linalg.norm(box[0]) * 10.0
lb = np.linalg.norm(box[1]) * 10.0
lc = np.linalg.norm(box[2]) * 10.0

alpha = angle_deg(
    box[1],
    box[2],
)

beta = angle_deg(
    box[0],
    box[2],
)

gamma = angle_deg(
    box[0],
    box[1],
)


pdb = [
    (
        f"CRYST1"
        f"{la:9.3f}"
        f"{lb:9.3f}"
        f"{lc:9.3f}"
        f"{alpha:7.2f}"
        f"{beta:7.2f}"
        f"{gamma:7.2f} "
        f"P 1           1"
    )
]


serial = 1

for pos in graphene_c:

    pdb.append(
        pdb_atom(
            serial,
            "C",
            "GRA",
            "G",
            1,
            pos,
            "C",
        )
    )

    serial += 1


for i, pos in enumerate(
    ligand_drop
):

    if i in {
        35,
        36,
        69,
    }:
        resname = "ALK"

    elif i in set(
        aromatic.tolist()
    ):
        resname = "PYR"

    else:
        resname = "LNK"

    pdb.append(
        pdb_atom(
            serial,
            atoms[i]["name"],
            resname,
            "L",
            2,
            pos,
            element_from_type(
                atoms[i]["type"]
            ),
        )
    )

    serial += 1


pdb.append("END")

OUTPUT_PDB.write_text(
    "\n".join(pdb)
    +
    "\n"
)


# ============================================================
# 13. METADATA
# ============================================================

metadata = {

    "status":
        "DROP01_INITIAL_CONFIGURATION_PASS",

    "md_steps":
        0,

    "minimization_steps":
        0,

    "target_pyrene_height_nm":
        TARGET_PYRENE_HEIGHT_NM,

    "actual_pyrene_height_nm":
        drop_height,

    "initial_pyrene_tilt_deg":
        drop_tilt_deg,

    "ligand_min_height_nm":
        float(
            ligand_heights.min()
        ),

    "ligand_max_height_nm":
        float(
            ligand_heights.max()
        ),

    "terminal_heights_nm":
        terminal_heights,

    "original_waters":
        n_waters_original,

    "removed_overlap_waters":
        n_removed,

    "final_waters":
        n_waters_final,

    "prune_fraction":
        prune_fraction,

    "waters_below_graphene":
        waters_below_graphene,

    "ligand_lj_atoms_checked":
        len(
            ligand_lj_atoms
        ),

    "solute_output":
        str(
            OUTPUT_SOLUTE
        ),

    "water_output":
        str(
            OUTPUT_WATER
        ),

    "vmd_pdb":
        str(
            OUTPUT_PDB
        ),
}


OUTPUT_METADATA.write_text(
    json.dumps(
        metadata,
        indent=2,
    )
    +
    "\n"
)


# ============================================================
# 14. REPORT
# ============================================================

print()
print("PLACEMENT")
print("---------")
print(
    f"Pyrene height: "
    f"{drop_height:.6f} nm"
)
print(
    f"Pyrene tilt:   "
    f"{drop_tilt_deg:.3f} deg"
)
print(
    f"Ligand height range: "
    f"{ligand_heights.min():.6f} "
    f"to "
    f"{ligand_heights.max():.6f} nm"
)

print()
print("TERMINAL ALKYNE")
print("----------------")
print(
    f"C35: {terminal_heights['35']:.6f} nm"
)
print(
    f"C36: {terminal_heights['36']:.6f} nm"
)
print(
    f"H69: {terminal_heights['69']:.6f} nm"
)

print()
print("WATER PRUNING")
print("-------------")
print(
    "Original waters:",
    n_waters_original,
)
print(
    "Removed:",
    n_removed,
)
print(
    "Final waters:",
    n_waters_final,
)
print(
    f"Pruned fraction: "
    f"{100.0 * prune_fraction:.4f}%"
)
print(
    "Waters below graphene:",
    waters_below_graphene,
)

print()
print("SAVED")
print("-----")
print(
    "Solute:",
    OUTPUT_SOLUTE,
)
print(
    "Water:",
    OUTPUT_WATER,
)
print(
    "Metadata:",
    OUTPUT_METADATA,
)
print(
    "VMD:",
    OUTPUT_PDB,
)

print()
print("=" * 78)
print("DROP01 INITIAL CONFIGURATION: PASS")
print("=" * 78)
print(
    "No MD or minimization was performed."
)

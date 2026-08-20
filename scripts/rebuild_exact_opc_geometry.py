from pathlib import Path
import json
import math

import numpy as np

from openmm import openmm, unit, app
from openmm.app import element


# ============================================================
# REBUILD EXACT OPC WATER GEOMETRY
#
# PURPOSE
# -------
# The existing one-sided water placement is good:
#
#   - 6196 waters
#   - oxygen locations are correct
#   - one-sided slab placement is correct
#
# But the stored internal 4-site geometry does not exactly
# match OpenMM's OPC rigid-water definition.
#
# This script:
#
#   1. keeps every oxygen EXACTLY fixed
#   2. preserves each water's approximate orientation
#   3. rebuilds H1/H2 using exact OpenMM OPC constraints
#   4. rebuilds the M site using OpenMM's exact virtual-site rule
#
# NO MD.
# NO minimization.
# NO force-field parameters are changed.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

N_WATERS = 6196
SITES_PER_WATER = 4

O = 0
H1 = 1
H2 = 2
M = 3


INPUT_WATER = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_nm.npy"
)

BOX_VECTORS_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_periodic_box_vectors_nm.npy"
)

OUTPUT_WATER = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_exact_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/"
    "exact_opc_geometry_rebuild.json"
)


LOWER_BOUNDARY_NM = 0.100
UPPER_BOUNDARY_NM = 6.800


# ============================================================
# 2. VERIFY INPUTS
# ============================================================

for path in [
    INPUT_WATER,
    BOX_VECTORS_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )


# ============================================================
# 3. LOAD ORIGINAL WATER GEOMETRY
# ============================================================

water = np.load(
    INPUT_WATER
)

box_vectors = np.load(
    BOX_VECTORS_FILE
)


print()
print("=" * 72)
print("EXACT OPC GEOMETRY REBUILD")
print("=" * 72)

print()
print("Input water shape:", water.shape)
print("Box-vector shape:", box_vectors.shape)


if water.shape != (
    N_WATERS,
    SITES_PER_WATER,
    3,
):

    raise RuntimeError(
        "Unexpected water array shape. "
        f"Expected ({N_WATERS}, 4, 3), "
        f"found {water.shape}."
    )


if box_vectors.shape != (
    3,
    3,
):

    raise RuntimeError(
        "Unexpected box-vector shape. "
        f"Found {box_vectors.shape}."
    )


if not np.isfinite(
    water
).all():

    raise RuntimeError(
        "Water coordinates contain NaN or Inf."
    )


# ============================================================
# 4. BUILD ONE AUTHORITATIVE OPC WATER WITH OPENMM
#
# We ask amber19/opc.xml directly for:
#
#   - rigid O-H distance
#   - rigid H-H distance
#   - M-site parent particles
#   - M-site weights
#
# Nothing is hard-coded.
# ============================================================

print()
print("BUILDING OPENMM OPC REFERENCE")
print("-----------------------------")


topology = app.Topology()

chain = topology.addChain()

residue = topology.addResidue(
    "HOH",
    chain,
)


atom_o = topology.addAtom(
    "O",
    element.oxygen,
    residue,
)

atom_h1 = topology.addAtom(
    "H1",
    element.hydrogen,
    residue,
)

atom_h2 = topology.addAtom(
    "H2",
    element.hydrogen,
    residue,
)


topology.addBond(
    atom_o,
    atom_h1,
)

topology.addBond(
    atom_o,
    atom_h2,
)


# Only O/H/H are required here.
# The M site will be added by the force field.
reference_positions = unit.Quantity(
    [
        openmm.Vec3(
            *water[0, O, :]
        ),

        openmm.Vec3(
            *water[0, H1, :]
        ),

        openmm.Vec3(
            *water[0, H2, :]
        ),
    ],

    unit.nanometer,
)


forcefield = app.ForceField(
    "amber19/opc.xml"
)


modeller = app.Modeller(
    topology,
    reference_positions,
)


modeller.addExtraParticles(
    forcefield
)


reference_names = [
    atom.name
    for atom
    in modeller.topology.atoms()
]


print(
    "Reference particle names:",
    reference_names,
)


if len(reference_names) != 4:

    raise RuntimeError(
        "Expected OPC reference water to contain "
        "four particles."
    )


if reference_names != [
    "O",
    "H1",
    "H2",
    "M",
]:

    raise RuntimeError(
        "Unexpected OPC site ordering: "
        f"{reference_names}"
    )


reference_system = (
    forcefield.createSystem(

        modeller.topology,

        nonbondedMethod=app.NoCutoff,

        constraints=None,

        rigidWater=True,

        removeCMMotion=False,
    )
)


if reference_system.getNumParticles() != 4:

    raise RuntimeError(
        "OPC reference System does not contain "
        "exactly four particles."
    )


# ============================================================
# 5. READ EXACT OPC CONSTRAINT DISTANCES
# ============================================================

print()
print("OPC RIGID CONSTRAINTS")
print("---------------------")


constraint_distances = {}


for i in range(
    reference_system.getNumConstraints()
):

    p1, p2, distance = (
        reference_system
        .getConstraintParameters(i)
    )

    pair = tuple(
        sorted(
            [
                int(p1),
                int(p2),
            ]
        )
    )

    distance_nm = (
        distance.value_in_unit(
            unit.nanometer
        )
    )

    constraint_distances[
        pair
    ] = float(
        distance_nm
    )

    print(
        f"{pair}: "
        f"{distance_nm:.10f} nm"
    )


expected_pairs = {
    (O, H1),
    (O, H2),
    (H1, H2),
}


if set(
    constraint_distances.keys()
) != expected_pairs:

    raise RuntimeError(
        "Unexpected OPC rigid-water constraint pattern. "
        f"Found {constraint_distances.keys()}."
    )


R_OH1 = (
    constraint_distances[
        (O, H1)
    ]
)

R_OH2 = (
    constraint_distances[
        (O, H2)
    ]
)

R_HH = (
    constraint_distances[
        (H1, H2)
    ]
)


if not np.isclose(
    R_OH1,
    R_OH2,
    rtol=0.0,
    atol=1e-10,
):

    raise RuntimeError(
        "OPC O-H constraint distances are not equal."
    )


R_OH = (
    0.5
    * (
        R_OH1
        + R_OH2
    )
)


# ============================================================
# 6. CALCULATE EXACT OPC H-O-H ANGLE
#
# From the triangle:
#
#       H1
#        \
#         O
#        /
#       H2
#
# with:
#
#   O-H = R_OH
#   H-H = R_HH
#
# ============================================================

cos_theta = (
    (
        2.0
        * R_OH**2
        - R_HH**2
    )
    /
    (
        2.0
        * R_OH**2
    )
)


cos_theta = np.clip(
    cos_theta,
    -1.0,
    1.0,
)


THETA_RAD = float(
    np.arccos(
        cos_theta
    )
)


THETA_DEG = float(
    np.degrees(
        THETA_RAD
    )
)


HALF_THETA = (
    0.5
    * THETA_RAD
)


print()
print("EXACT OPC GEOMETRY")
print("------------------")

print(
    f"O-H distance: "
    f"{R_OH:.10f} nm"
)

print(
    f"H-H distance: "
    f"{R_HH:.10f} nm"
)

print(
    f"H-O-H angle: "
    f"{THETA_DEG:.8f} degrees"
)


# ============================================================
# 7. READ EXACT M-SITE RULE
# ============================================================

if not reference_system.isVirtualSite(M):

    raise RuntimeError(
        "OPC site 3 is not a virtual site."
    )


virtual_site = (
    reference_system
    .getVirtualSite(M)
)


if not isinstance(
    virtual_site,
    openmm.ThreeParticleAverageSite,
):

    raise RuntimeError(
        "Expected OPC M site to be "
        "ThreeParticleAverageSite."
    )


parents = [
    int(
        virtual_site.getParticle(i)
    )

    for i in range(
        virtual_site.getNumParticles()
    )
]


weights = [
    float(
        virtual_site.getWeight(i)
    )

    for i in range(
        virtual_site.getNumParticles()
    )
]


print()
print("OPC M-SITE DEFINITION")
print("---------------------")

print(
    "Parent particles:",
    parents,
)

print(
    "Weights:",
    weights,
)


if parents != [
    O,
    H1,
    H2,
]:

    raise RuntimeError(
        "Unexpected OPC M-site parent ordering."
    )


if not np.isclose(
    sum(weights),
    1.0,
    atol=1e-12,
):

    raise RuntimeError(
        "OPC virtual-site weights do not sum to 1."
    )


W_O = weights[0]
W_H1 = weights[1]
W_H2 = weights[2]


# ============================================================
# 8. PERIODIC MINIMUM-IMAGE HELPER
#
# Some waters can sit near the x/y periodic edges.
#
# For orientation only, make sure each O-H vector represents
# the nearest periodic copy of that hydrogen.
#
# We DO NOT wrap z.
# ============================================================

box = np.asarray(
    box_vectors,
    dtype=float,
)


if abs(
    np.linalg.det(box)
) < 1e-12:

    raise RuntimeError(
        "Periodic box matrix is singular."
    )


inverse_box = np.linalg.inv(
    box
)


def minimum_image_xy(displacements):

    frac = (
        displacements
        @ inverse_box
    )

    frac[:, 0] -= np.round(
        frac[:, 0]
    )

    frac[:, 1] -= np.round(
        frac[:, 1]
    )

    # Do NOT periodic-wrap z.

    return (
        frac
        @ box
    )


# ============================================================
# 9. EXTRACT ORIGINAL LOCAL WATER VECTORS
# ============================================================

oxygen = (
    water[:, O, :].copy()
)


v1_raw = (
    water[:, H1, :]
    - oxygen
)

v2_raw = (
    water[:, H2, :]
    - oxygen
)


v1 = minimum_image_xy(
    v1_raw
)

v2 = minimum_image_xy(
    v2_raw
)


old_r1 = np.linalg.norm(
    v1,
    axis=1,
)

old_r2 = np.linalg.norm(
    v2,
    axis=1,
)


if np.any(
    old_r1 < 0.05
) or np.any(
    old_r1 > 0.15
):

    raise RuntimeError(
        "Original H1-O distances are unreasonable."
    )


if np.any(
    old_r2 < 0.05
) or np.any(
    old_r2 > 0.15
):

    raise RuntimeError(
        "Original H2-O distances are unreasonable."
    )


u1 = (
    v1
    / old_r1[:, None]
)

u2 = (
    v2
    / old_r2[:, None]
)


# ============================================================
# 10. MEASURE ORIGINAL WATER ANGLE
# ============================================================

old_cos_angle = np.sum(
    u1 * u2,
    axis=1,
)


old_cos_angle = np.clip(
    old_cos_angle,
    -1.0,
    1.0,
)


old_angles_deg = np.degrees(
    np.arccos(
        old_cos_angle
    )
)


print()
print("ORIGINAL SAVED GEOMETRY")
print("-----------------------")

print(
    f"Mean O-H1: "
    f"{np.mean(old_r1):.10f} nm"
)

print(
    f"Mean O-H2: "
    f"{np.mean(old_r2):.10f} nm"
)

print(
    f"Mean H-O-H angle: "
    f"{np.mean(old_angles_deg):.8f} degrees"
)


# ============================================================
# 11. PRESERVE EACH WATER'S ORIENTATION
#
# Define two directions:
#
#   bisector:
#       direction between H1 and H2
#
#   side direction:
#       direction from H2 side toward H1 side
#
# This preserves:
#
#   - oxygen position
#   - water molecular plane
#   - water pointing direction
#   - H1 versus H2 orientation
#
# while changing only the internal geometry.
# ============================================================

bisector = (
    u1
    + u2
)


side = (
    u1
    - u2
)


bisector_norm = np.linalg.norm(
    bisector,
    axis=1,
)

side_norm = np.linalg.norm(
    side,
    axis=1,
)


if np.any(
    bisector_norm < 1e-8
):

    raise RuntimeError(
        "A water has an undefined orientation bisector."
    )


if np.any(
    side_norm < 1e-8
):

    raise RuntimeError(
        "A water has an undefined H1/H2 direction."
    )


bisector /= (
    bisector_norm[:, None]
)

side /= (
    side_norm[:, None]
)


# They should be orthogonal.
orthogonality = np.sum(
    bisector * side,
    axis=1,
)


max_orthogonality_error = float(
    np.max(
        np.abs(
            orthogonality
        )
    )
)


print()
print(
    "Maximum orientation-basis "
    f"orthogonality error: "
    f"{max_orthogonality_error:.3e}"
)


if max_orthogonality_error > 1e-8:

    raise RuntimeError(
        "Water orientation basis is not orthogonal."
    )


# ============================================================
# 12. BUILD EXACT OPC H POSITIONS
# ============================================================

cos_half = math.cos(
    HALF_THETA
)

sin_half = math.sin(
    HALF_THETA
)


new_v1 = (
    R_OH
    * (
        cos_half
        * bisector
        +
        sin_half
        * side
    )
)


new_v2 = (
    R_OH
    * (
        cos_half
        * bisector
        -
        sin_half
        * side
    )
)


new_h1 = (
    oxygen
    + new_v1
)


new_h2 = (
    oxygen
    + new_v2
)


# ============================================================
# 13. BUILD EXACT OPC M SITE
#
# OpenMM definition:
#
# M =
#     wO  * O
#   + wH1 * H1
#   + wH2 * H2
# ============================================================

new_m = (
    W_O
    * oxygen
    +
    W_H1
    * new_h1
    +
    W_H2
    * new_h2
)


# ============================================================
# 14. CREATE CORRECTED WATER ARRAY
# ============================================================

corrected = np.empty_like(
    water
)


corrected[:, O, :] = (
    oxygen
)

corrected[:, H1, :] = (
    new_h1
)

corrected[:, H2, :] = (
    new_h2
)

corrected[:, M, :] = (
    new_m
)


if not np.isfinite(
    corrected
).all():

    raise RuntimeError(
        "Corrected coordinates contain NaN/Inf."
    )


# ============================================================
# 15. VERIFY OXYGENS DID NOT MOVE
# ============================================================

oxygen_shift = np.linalg.norm(
    corrected[:, O, :]
    - water[:, O, :],
    axis=1,
)


max_oxygen_shift = float(
    np.max(
        oxygen_shift
    )
)


print()
print("OXYGEN PRESERVATION")
print("-------------------")

print(
    f"Maximum oxygen displacement: "
    f"{max_oxygen_shift:.12f} nm"
)


if max_oxygen_shift > 1e-12:

    raise RuntimeError(
        "A water oxygen moved during OPC correction."
    )


# ============================================================
# 16. VALIDATE NEW OPC DISTANCES
# ============================================================

new_oh1 = np.linalg.norm(
    corrected[:, H1, :]
    - corrected[:, O, :],
    axis=1,
)


new_oh2 = np.linalg.norm(
    corrected[:, H2, :]
    - corrected[:, O, :],
    axis=1,
)


new_hh = np.linalg.norm(
    corrected[:, H1, :]
    - corrected[:, H2, :],
    axis=1,
)


max_oh1_error = float(
    np.max(
        np.abs(
            new_oh1
            - R_OH
        )
    )
)


max_oh2_error = float(
    np.max(
        np.abs(
            new_oh2
            - R_OH
        )
    )
)


max_hh_error = float(
    np.max(
        np.abs(
            new_hh
            - R_HH
        )
    )
)


print()
print("EXACT CONSTRAINT VALIDATION")
print("---------------------------")

print(
    f"Maximum O-H1 error: "
    f"{max_oh1_error:.3e} nm"
)

print(
    f"Maximum O-H2 error: "
    f"{max_oh2_error:.3e} nm"
)

print(
    f"Maximum H1-H2 error: "
    f"{max_hh_error:.3e} nm"
)


if max(
    max_oh1_error,
    max_oh2_error,
    max_hh_error,
) > 1e-10:

    raise RuntimeError(
        "Corrected waters do not satisfy "
        "the exact OPC rigid geometry."
    )


# ============================================================
# 17. VALIDATE M-SITE FORMULA
# ============================================================

expected_m = (
    W_O
    * corrected[:, O, :]
    +
    W_H1
    * corrected[:, H1, :]
    +
    W_H2
    * corrected[:, H2, :]
)


m_error = np.linalg.norm(
    corrected[:, M, :]
    - expected_m,
    axis=1,
)


max_m_error = float(
    np.max(
        m_error
    )
)


print()
print("M-SITE VALIDATION")
print("-----------------")

print(
    f"Maximum M-site formula error: "
    f"{max_m_error:.3e} nm"
)


if max_m_error > 1e-12:

    raise RuntimeError(
        "Corrected OPC M-site positions are inconsistent."
    )


# ============================================================
# 18. MEASURE EXACT O-M DISTANCE
# ============================================================

new_om = np.linalg.norm(
    corrected[:, M, :]
    - corrected[:, O, :],
    axis=1,
)


print(
    f"Mean exact OPC O-M distance: "
    f"{np.mean(new_om):.10f} nm"
)


# ============================================================
# 19. MEASURE SIZE OF CORRECTION
#
# Compare each corrected H against the corresponding old
# nearest-image local H position.
# ============================================================

old_local_h1 = (
    oxygen
    + v1
)

old_local_h2 = (
    oxygen
    + v2
)


h1_shift = np.linalg.norm(
    new_h1
    - old_local_h1,
    axis=1,
)


h2_shift = np.linalg.norm(
    new_h2
    - old_local_h2,
    axis=1,
)


max_hydrogen_shift = float(
    max(
        np.max(h1_shift),
        np.max(h2_shift),
    )
)


mean_hydrogen_shift = float(
    0.5
    * (
        np.mean(h1_shift)
        + np.mean(h2_shift)
    )
)


print()
print("GEOMETRY CORRECTION SIZE")
print("------------------------")

print(
    f"Mean hydrogen adjustment: "
    f"{mean_hydrogen_shift:.10f} nm"
)

print(
    f"Maximum hydrogen adjustment: "
    f"{max_hydrogen_shift:.10f} nm"
)


# ============================================================
# 20. VERIFY WATER OXYGEN SLAB IS UNCHANGED
# ============================================================

oxygen_z = (
    corrected[:, O, 2]
)


print()
print("WATER OXYGEN Z RANGE")
print("--------------------")

print(
    f"Minimum: "
    f"{oxygen_z.min():.6f} nm"
)

print(
    f"Maximum: "
    f"{oxygen_z.max():.6f} nm"
)


if oxygen_z.min() < (
    LOWER_BOUNDARY_NM
):

    raise RuntimeError(
        "A corrected water oxygen is below "
        "the lower boundary."
    )


if oxygen_z.max() > (
    UPPER_BOUNDARY_NM
):

    raise RuntimeError(
        "A corrected water oxygen is above "
        "the upper boundary."
    )


all_site_z = (
    corrected[:, :, 2]
)


print()
print("ALL EXACT OPC SITE Z RANGE")
print("--------------------------")

print(
    f"Minimum: "
    f"{all_site_z.min():.6f} nm"
)

print(
    f"Maximum: "
    f"{all_site_z.max():.6f} nm"
)


# ============================================================
# 21. SAVE CORRECTED WATER FILE
#
# IMPORTANT:
#
# We DO NOT overwrite the old file.
#
# This preserves provenance:
#
#   old file = original placement checkpoint
#   new file = exact OPC geometry checkpoint
# ============================================================

OUTPUT_WATER.parent.mkdir(
    parents=True,
    exist_ok=True,
)


np.save(
    OUTPUT_WATER,
    corrected,
)


# ============================================================
# 22. SAVE METADATA
# ============================================================

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "input_water_file":
        str(INPUT_WATER),

    "output_water_file":
        str(OUTPUT_WATER),

    "n_waters":
        N_WATERS,

    "sites_per_water":
        SITES_PER_WATER,

    "oxygen_positions_preserved":
        True,

    "maximum_oxygen_shift_nm":
        max_oxygen_shift,

    "original_mean_oh1_nm":
        float(
            np.mean(old_r1)
        ),

    "original_mean_oh2_nm":
        float(
            np.mean(old_r2)
        ),

    "original_mean_hoh_angle_deg":
        float(
            np.mean(
                old_angles_deg
            )
        ),

    "exact_opc_oh_nm":
        float(
            R_OH
        ),

    "exact_opc_hh_nm":
        float(
            R_HH
        ),

    "exact_opc_hoh_angle_deg":
        THETA_DEG,

    "exact_opc_mean_om_nm":
        float(
            np.mean(
                new_om
            )
        ),

    "m_site_parents":
        parents,

    "m_site_weights":
        weights,

    "max_oh1_error_nm":
        max_oh1_error,

    "max_oh2_error_nm":
        max_oh2_error,

    "max_hh_error_nm":
        max_hh_error,

    "max_m_site_error_nm":
        max_m_error,

    "mean_hydrogen_adjustment_nm":
        mean_hydrogen_shift,

    "max_hydrogen_adjustment_nm":
        max_hydrogen_shift,

    "oxygen_z_min_nm":
        float(
            oxygen_z.min()
        ),

    "oxygen_z_max_nm":
        float(
            oxygen_z.max()
        ),

    "md_run":
        False,

    "minimization_run":
        False,

    "status":
        "PASS",
}


with open(
    OUTPUT_METADATA,
    "w",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
    )


# ============================================================
# 23. FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("SAVED")
print("=" * 72)

print(
    "Exact OPC positions:",
    OUTPUT_WATER,
)

print(
    "Metadata:",
    OUTPUT_METADATA,
)


print()
print("=" * 72)
print("EXACT OPC GEOMETRY REBUILD: PASS")
print("=" * 72)

print()
print(
    "All 6196 oxygen positions were preserved exactly."
)

print(
    "H1/H2 were rebuilt to exact OpenMM OPC geometry."
)

print(
    "All M sites were rebuilt from the exact "
    "OpenMM virtual-site rule."
)

print(
    "No MD has been run."
)

print(
    "No minimization has been run."
)

print(
    "The original water file was NOT overwritten."
)

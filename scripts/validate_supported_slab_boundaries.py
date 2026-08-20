from pathlib import Path
import json

import numpy as np
from openmm import openmm, unit


# ============================================================
# CONFIGURATION
# ============================================================

# Solute particle layout:
#
#   0 ... 1249       graphene carbon cores
#   1250 ... 2499    upper IFF pi sites
#   2500 ... 3749    lower IFF pi sites
#   3750 ... 3819    Pyrene-PEG5
#
N_CARBON = 1250
N_GRAPHENE_TOTAL = 3750
N_SOLUTE = 3820

EXPECTED_WATERS = 6196
SITES_PER_WATER = 4

# The saved OPC water ordering is expected to be:
#
#   site 0 = O
#   site 1 = H1
#   site 2 = H2
#   site 3 = M
#
OXYGEN_SITE_INDEX = 0


GRAPHENE_SYSTEM_XML = Path(
    "parameters/iff/graphene_iff_bonded_oop.xml"
)

SUPPORTED_SOLUTE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_supported_positions_nm.npy"
)

WATER_POSITIONS = Path(
    "parameters/combined/"
    "one_sided_opc_water_positions_nm.npy"
)

OUTPUT_METADATA = Path(
    "analysis/supported_slab_boundary_validation.json"
)


# ============================================================
# SUPPORTED SLAB GEOMETRY
# ============================================================

GRAPHENE_PLANE_NM = 0.500

LOWER_BOUNDARY_NM = 0.100
UPPER_BOUNDARY_NM = 6.800

# Taller z-box leaves vacuum between periodic slab copies.
BOX_Z_NM = 12.0


# ============================================================
# RESTRAINT CONSTANTS
# ============================================================

# Graphene vertical support
K_GRAPHENE = 1000.0  # kJ/(mol nm^2)

# Water guard wall
K_WALL = 5000.0      # kJ/(mol nm^2)

# Artificial displacement used ONLY for validation.
TEST_DISPLACEMENT_NM = 0.10

SUPPORT_FORCE_GROUP = 5
WALL_FORCE_GROUP = 6


# ============================================================
# 1. VERIFY INPUT FILES
# ============================================================

required_files = [
    GRAPHENE_SYSTEM_XML,
    SUPPORTED_SOLUTE,
    WATER_POSITIONS,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {path}"
        )


# ============================================================
# 2. LOAD COORDINATES
# ============================================================

solute_positions = np.load(
    SUPPORTED_SOLUTE
)

water_positions = np.load(
    WATER_POSITIONS
)


print("SUPPORTED SLAB BOUNDARY VALIDATION")
print("----------------------------------")
print(
    "Solute coordinate shape:",
    solute_positions.shape,
)
print(
    "Water coordinate shape: ",
    water_positions.shape,
)


# ============================================================
# 3. VALIDATE SOLUTE ARRAY
# ============================================================

if solute_positions.shape != (
    N_SOLUTE,
    3,
):
    raise RuntimeError(
        "Supported solute positions have unexpected shape. "
        f"Expected ({N_SOLUTE}, 3), "
        f"found {solute_positions.shape}."
    )

if not np.isfinite(
    solute_positions
).all():
    raise RuntimeError(
        "Supported solute positions contain NaN or Inf."
    )


# ============================================================
# 4. VALIDATE WATER ARRAY
#
# Actual discovered format:
#
#       (6196, 4, 3)
#
# meaning:
#
#       6196 waters
#       4 OPC sites per water
#       xyz coordinate for each site
# ============================================================

expected_water_shape = (
    EXPECTED_WATERS,
    SITES_PER_WATER,
    3,
)

if water_positions.shape != expected_water_shape:
    raise RuntimeError(
        "Unexpected OPC water array shape. "
        f"Expected {expected_water_shape}, "
        f"found {water_positions.shape}."
    )

if not np.isfinite(
    water_positions
).all():
    raise RuntimeError(
        "Water positions contain NaN or Inf."
    )


n_waters = water_positions.shape[0]

print()
print("OPC WATER REPRESENTATION")
print("------------------------")
print("Waters:          ", n_waters)
print("Sites per water: ", SITES_PER_WATER)
print(
    "Total OPC sites: ",
    n_waters * SITES_PER_WATER,
)


# ============================================================
# 5. OPC SITE-ORDER SANITY CHECK
#
# Expected:
#
#   site 0 = oxygen
#   site 1 = H1
#   site 2 = H2
#   site 3 = M charge site
#
# We do NOT need exact force-field geometry here.
#
# We only make a broad sanity check:
#
#   O-H distances should be around ~0.09 nm
#   O-M should be much shorter.
# ============================================================

site0 = water_positions[:, 0, :]
site1 = water_positions[:, 1, :]
site2 = water_positions[:, 2, :]
site3 = water_positions[:, 3, :]


d01 = np.linalg.norm(
    site1 - site0,
    axis=1,
)

d02 = np.linalg.norm(
    site2 - site0,
    axis=1,
)

d03 = np.linalg.norm(
    site3 - site0,
    axis=1,
)


mean_d01 = float(
    np.mean(d01)
)

mean_d02 = float(
    np.mean(d02)
)

mean_d03 = float(
    np.mean(d03)
)


print()
print("OPC SITE SANITY CHECK")
print("---------------------")
print(
    f"Mean site0-site1 distance: "
    f"{mean_d01:.6f} nm"
)
print(
    f"Mean site0-site2 distance: "
    f"{mean_d02:.6f} nm"
)
print(
    f"Mean site0-site3 distance: "
    f"{mean_d03:.6f} nm"
)


# Broad ranges only.
#
# These checks are designed to detect a completely wrong
# interpretation of the 4 sites, not enforce an exact OPC
# geometry parameter.
#
if not (
    0.07
    < mean_d01
    < 0.11
):
    raise RuntimeError(
        "Site 0 -> site 1 distance does not look like "
        "an O-H bond. OPC site ordering may be wrong."
    )

if not (
    0.07
    < mean_d02
    < 0.11
):
    raise RuntimeError(
        "Site 0 -> site 2 distance does not look like "
        "an O-H bond. OPC site ordering may be wrong."
    )

if not (
    0.005
    < mean_d03
    < 0.04
):
    raise RuntimeError(
        "Site 0 -> site 3 distance does not look like "
        "an OPC O-M separation. Site ordering may be wrong."
    )


print(
    "OPC site ordering: PASS"
)


# ============================================================
# 6. EXTRACT WATER OXYGEN POSITIONS
#
# water_positions[:, 0, :] gives:
#
#       one oxygen coordinate per water
#
# shape:
#
#       (6196, 3)
# ============================================================

water_oxygen_positions = (
    water_positions[
        :,
        OXYGEN_SITE_INDEX,
        :
    ]
)

if water_oxygen_positions.shape != (
    EXPECTED_WATERS,
    3,
):
    raise RuntimeError(
        "Water oxygen extraction failed."
    )


oxygen_z = (
    water_oxygen_positions[:, 2]
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


# ============================================================
# 7. CHECK ALL WATER SITE Z VALUES
#
# This is informational.
#
# The wall will act on oxygen only, but it is useful to know
# the full geometric extent of every OPC site.
# ============================================================

all_water_z = (
    water_positions[:, :, 2]
)

print()
print("ALL OPC SITE Z RANGE")
print("--------------------")
print(
    f"Minimum: "
    f"{all_water_z.min():.6f} nm"
)
print(
    f"Maximum: "
    f"{all_water_z.max():.6f} nm"
)


# ============================================================
# 8. VERIFY OXYGENS START INSIDE ALLOWED SLAB
# ============================================================

if oxygen_z.min() < LOWER_BOUNDARY_NM:
    raise RuntimeError(
        "At least one starting water oxygen lies below "
        f"the lower boundary of {LOWER_BOUNDARY_NM:.3f} nm."
    )

if oxygen_z.max() > UPPER_BOUNDARY_NM:
    raise RuntimeError(
        "At least one starting water oxygen lies above "
        f"the upper boundary of {UPPER_BOUNDARY_NM:.3f} nm."
    )


# ============================================================
# 9. VERIFY GRAPHENE PLANE
# ============================================================

graphene_carbon_z = (
    solute_positions[
        :N_CARBON,
        2
    ]
)


print()
print("GRAPHENE CARBON Z RANGE")
print("-----------------------")
print(
    f"Minimum: "
    f"{graphene_carbon_z.min():.6f} nm"
)
print(
    f"Maximum: "
    f"{graphene_carbon_z.max():.6f} nm"
)
print(
    f"Mean:    "
    f"{graphene_carbon_z.mean():.6f} nm"
)


if not np.allclose(
    graphene_carbon_z,
    GRAPHENE_PLANE_NM,
    atol=1e-6,
):
    raise RuntimeError(
        "Graphene carbon cores are not all on the "
        f"expected z={GRAPHENE_PLANE_NM:.3f} nm plane."
    )


# ============================================================
# 10. FLATTEN OPC POSITIONS
#
# Input:
#
#       (6196, 4, 3)
#
# becomes:
#
#       (24784, 3)
#
# Ordering remains:
#
#       water 0: O H1 H2 M
#       water 1: O H1 H2 M
#       ...
#
# This means our dummy test-system indexing now matches the
# natural flattened OpenMM particle layout.
# ============================================================

water_positions_flat = (
    water_positions.reshape(
        -1,
        3,
    )
)

expected_water_sites = (
    EXPECTED_WATERS
    * SITES_PER_WATER
)

if water_positions_flat.shape != (
    expected_water_sites,
    3,
):
    raise RuntimeError(
        "Flattened OPC water array has unexpected shape."
    )


# ============================================================
# 11. COMBINE SOLUTE + ALL OPC SITES
# ============================================================

combined_positions = np.vstack(
    [
        solute_positions,
        water_positions_flat,
    ]
)


n_total = len(
    combined_positions
)

expected_total = (
    N_SOLUTE
    + expected_water_sites
)


if n_total != expected_total:
    raise RuntimeError(
        f"Expected {expected_total} combined particles, "
        f"found {n_total}."
    )


print()
print("BOUNDARY-TEST SYSTEM")
print("--------------------")
print(
    "Solute particles: ",
    N_SOLUTE,
)
print(
    "OPC water sites:  ",
    expected_water_sites,
)
print(
    "Total particles:  ",
    n_total,
)


# ============================================================
# 12. CONSTRUCT GLOBAL WATER OXYGEN INDICES
#
# Since every water occupies 4 consecutive flattened sites:
#
#   water 0 O = 3820
#   water 1 O = 3824
#   water 2 O = 3828
#   ...
# ============================================================

water_oxygen_global = (
    N_SOLUTE
    + (
        np.arange(
            EXPECTED_WATERS,
            dtype=int,
        )
        * SITES_PER_WATER
    )
    + OXYGEN_SITE_INDEX
)


if len(
    water_oxygen_global
) != EXPECTED_WATERS:
    raise RuntimeError(
        "Water oxygen global-index construction failed."
    )


# Confirm that indexed coordinates exactly reproduce the
# oxygen positions extracted from the original 3D array.
#
indexed_oxygen_positions = (
    combined_positions[
        water_oxygen_global
    ]
)


if not np.allclose(
    indexed_oxygen_positions,
    water_oxygen_positions,
    atol=1e-12,
):
    raise RuntimeError(
        "Flattened OPC indexing does not reproduce "
        "the original oxygen coordinates."
    )


print()
print("WATER OXYGEN INDEXING")
print("---------------------")
print(
    "First oxygen index:",
    int(
        water_oxygen_global[0]
    ),
)
print(
    "Second oxygen index:",
    int(
        water_oxygen_global[1]
    ),
)
print(
    "Last oxygen index: ",
    int(
        water_oxygen_global[-1]
    ),
)

print(
    "Flattened oxygen indexing: PASS"
)


# ============================================================
# 13. LOAD ORIGINAL GRAPHENE PERIODIC X/Y BOX
# ============================================================

with open(
    GRAPHENE_SYSTEM_XML,
    "r",
) as f:

    graphene_system = (
        openmm.XmlSerializer.deserialize(
            f.read()
        )
    )


if graphene_system.getNumParticles() != (
    N_GRAPHENE_TOTAL
):
    raise RuntimeError(
        "Unexpected graphene XML particle count. "
        f"Expected {N_GRAPHENE_TOTAL}, found "
        f"{graphene_system.getNumParticles()}."
    )


a, b, old_c = (
    graphene_system
    .getDefaultPeriodicBoxVectors()
)


# Replace only z.
#
c = (
    openmm.Vec3(
        0.0,
        0.0,
        BOX_Z_NM,
    )
    * unit.nanometer
)


# ============================================================
# 14. CREATE BOUNDARY-TEST OPENMM SYSTEM
#
# This is NOT the production force field.
#
# Every particle gets a dummy mass because:
#
#       NO MD WILL BE RUN
#
# We are only evaluating external-force energies.
# ============================================================

system = openmm.System()


for _ in range(
    n_total
):
    system.addParticle(
        1.0
        * unit.dalton
    )


system.setDefaultPeriodicBoxVectors(
    a,
    b,
    c,
)


# ============================================================
# 15. GRAPHENE SUPPORT RESTRAINT
#
# Same restraint already independently validated.
#
# Only graphene carbon cores:
#
#       particles 0 ... 1249
#
# Pyrene-PEG5 remains unrestrained.
# ============================================================

support_force = (
    openmm.CustomExternalForce(
        "0.5*kz*periodicdistance(0,0,z,0,0,z0)^2"
    )
)

support_force.setName(
    "GrapheneSupportZRestraint"
)

support_force.addGlobalParameter(
    "kz",
    K_GRAPHENE,
)

support_force.addPerParticleParameter(
    "z0"
)

support_force.setForceGroup(
    SUPPORT_FORCE_GROUP
)


for i in range(
    N_CARBON
):

    support_force.addParticle(
        i,
        [
            float(
                combined_positions[
                    i,
                    2,
                ]
            )
        ],
    )


system.addForce(
    support_force
)


# ============================================================
# 16. DEFINE WATER SLAB
#
# Allowed oxygen region:
#
#       0.100 <= z <= 6.800 nm
#
# Center:
#
#       3.450 nm
#
# Half-width:
#
#       3.350 nm
#
# Wall energy:
#
#       zero inside slab
#
#       1/2*k*(penetration)^2
#       outside slab
# ============================================================

Z_MID_NM = (
    LOWER_BOUNDARY_NM
    + UPPER_BOUNDARY_NM
) / 2.0


HALF_WIDTH_NM = (
    UPPER_BOUNDARY_NM
    - LOWER_BOUNDARY_NM
) / 2.0


wall_force = (
    openmm.CustomExternalForce(
        "0.5*kwall*step(d-halfwidth)*(d-halfwidth)^2;"
        "d=periodicdistance(0,0,z,0,0,zmid)"
    )
)


wall_force.setName(
    "OneSidedWaterSlabWall"
)


wall_force.addGlobalParameter(
    "kwall",
    K_WALL,
)

wall_force.addGlobalParameter(
    "zmid",
    Z_MID_NM,
)

wall_force.addGlobalParameter(
    "halfwidth",
    HALF_WIDTH_NM,
)

wall_force.setForceGroup(
    WALL_FORCE_GROUP
)


# IMPORTANT:
#
# Only oxygen is acted on by the wall.
#
# The hydrogens and M-site are part of the same rigid OPC
# molecule in the eventual production system.
#
for index in water_oxygen_global:

    wall_force.addParticle(
        int(index),
        [],
    )


system.addForce(
    wall_force
)


# ============================================================
# 17. VERIFY FORCE COUNTS
# ============================================================

if support_force.getNumParticles() != (
    N_CARBON
):
    raise RuntimeError(
        "Incorrect number of graphene support restraints."
    )


if wall_force.getNumParticles() != (
    EXPECTED_WATERS
):
    raise RuntimeError(
        "Incorrect number of water wall particles."
    )


# ============================================================
# 18. PRINT SLAB DESIGN
# ============================================================

aqueous_height_nm = (
    UPPER_BOUNDARY_NM
    - LOWER_BOUNDARY_NM
)


vacuum_gap_nm = (
    BOX_Z_NM
    - aqueous_height_nm
)


print()
print("SLAB DESIGN")
print("-----------")
print(
    f"Lower water boundary: "
    f"{LOWER_BOUNDARY_NM:.3f} nm"
)
print(
    f"Upper water boundary: "
    f"{UPPER_BOUNDARY_NM:.3f} nm"
)
print(
    f"Aqueous slab height:   "
    f"{aqueous_height_nm:.3f} nm"
)
print(
    f"Periodic box z:        "
    f"{BOX_Z_NM:.3f} nm"
)
print(
    f"Vacuum separation:     "
    f"{vacuum_gap_nm:.3f} nm"
)
print(
    f"Slab center:           "
    f"{Z_MID_NM:.3f} nm"
)
print(
    f"Slab half-width:       "
    f"{HALF_WIDTH_NM:.3f} nm"
)


print()
print("FORCES")
print("------")
print(
    "Graphene carbon restraints:",
    support_force.getNumParticles(),
)
print(
    "Water oxygen wall terms:   ",
    wall_force.getNumParticles(),
)
print(
    f"Graphene kz: "
    f"{K_GRAPHENE:.1f} kJ/(mol nm^2)"
)
print(
    f"Water wall k: "
    f"{K_WALL:.1f} kJ/(mol nm^2)"
)
print(
    "Pyrene-PEG5 restrained: NO"
)


# ============================================================
# 19. CONVERT POSITIONS TO OPENMM
# ============================================================

positions_openmm = (
    unit.Quantity(
        [
            openmm.Vec3(
                float(x),
                float(y),
                float(z),
            )
            for x, y, z
            in combined_positions
        ],
        unit.nanometer,
    )
)


# ============================================================
# 20. CREATE CONTEXT
#
# NO INTEGRATION STEPS WILL BE TAKEN.
# ============================================================

integrator = (
    openmm.VerletIntegrator(
        0.001
        * unit.picoseconds
    )
)


try:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CUDA"
        )
    )

except Exception:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CPU"
        )
    )


context = openmm.Context(
    system,
    integrator,
    platform,
)


context.setPositions(
    positions_openmm
)


print()
print(
    "OpenMM platform:",
    platform.getName(),
)


# ============================================================
# 21. STARTING GRAPHENE SUPPORT ENERGY
# ============================================================

support_state = (
    context.getState(
        getEnergy=True,
        groups=(
            1
            << SUPPORT_FORCE_GROUP
        ),
    )
)


support_initial = (
    support_state
    .getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


# ============================================================
# 22. STARTING WATER WALL ENERGY
# ============================================================

wall_state = (
    context.getState(
        getEnergy=True,
        groups=(
            1
            << WALL_FORCE_GROUP
        ),
    )
)


wall_initial = (
    wall_state
    .getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


print()
print("STARTING BOUNDARY ENERGIES")
print("--------------------------")
print(
    f"Graphene support: "
    f"{support_initial:.8f} kJ/mol"
)
print(
    f"Water wall:       "
    f"{wall_initial:.8f} kJ/mol"
)


# ============================================================
# 23. GRAPHENE SUPPORT TEST
#
# Artificially move one graphene carbon by +0.10 nm.
#
# Expected:
#
#   1/2 * 1000 * 0.1^2
#   = 5 kJ/mol
#
# NOT MD.
# ============================================================

graphene_test_positions = (
    combined_positions.copy()
)


graphene_test_positions[
    0,
    2,
] += TEST_DISPLACEMENT_NM


graphene_test_openmm = (
    unit.Quantity(
        [
            openmm.Vec3(
                float(x),
                float(y),
                float(z),
            )
            for x, y, z
            in graphene_test_positions
        ],
        unit.nanometer,
    )
)


context.setPositions(
    graphene_test_openmm
)


state = context.getState(
    getEnergy=True,
    groups=(
        1
        << SUPPORT_FORCE_GROUP
    ),
)


support_test_energy = (
    state
    .getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


support_expected = (
    0.5
    * K_GRAPHENE
    * TEST_DISPLACEMENT_NM**2
)


# ============================================================
# 24. SELECT FIRST WATER OXYGEN FOR WALL TEST
# ============================================================

test_water_oxygen = int(
    water_oxygen_global[0]
)


if test_water_oxygen != N_SOLUTE:
    raise RuntimeError(
        "First flattened OPC oxygen index "
        "is not where expected."
    )


# ============================================================
# 25. UPPER WALL TEST
#
# Move ONE oxygen 0.10 nm above ceiling.
#
# Expected:
#
#   1/2 * 5000 * 0.1^2
#   = 25 kJ/mol
#
# NOT MD.
# ============================================================

upper_test_positions = (
    combined_positions.copy()
)


upper_test_positions[
    test_water_oxygen,
    2,
] = (
    UPPER_BOUNDARY_NM
    + TEST_DISPLACEMENT_NM
)


upper_test_openmm = (
    unit.Quantity(
        [
            openmm.Vec3(
                float(x),
                float(y),
                float(z),
            )
            for x, y, z
            in upper_test_positions
        ],
        unit.nanometer,
    )
)


context.setPositions(
    upper_test_openmm
)


state = context.getState(
    getEnergy=True,
    groups=(
        1
        << WALL_FORCE_GROUP
    ),
)


upper_test_energy = (
    state
    .getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


# ============================================================
# 26. LOWER WALL TEST
#
# Move same oxygen 0.10 nm below floor.
# ============================================================

lower_test_positions = (
    combined_positions.copy()
)


lower_test_positions[
    test_water_oxygen,
    2,
] = (
    LOWER_BOUNDARY_NM
    - TEST_DISPLACEMENT_NM
)


lower_test_openmm = (
    unit.Quantity(
        [
            openmm.Vec3(
                float(x),
                float(y),
                float(z),
            )
            for x, y, z
            in lower_test_positions
        ],
        unit.nanometer,
    )
)


context.setPositions(
    lower_test_openmm
)


state = context.getState(
    getEnergy=True,
    groups=(
        1
        << WALL_FORCE_GROUP
    ),
)


lower_test_energy = (
    state
    .getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


wall_expected = (
    0.5
    * K_WALL
    * TEST_DISPLACEMENT_NM**2
)


# ============================================================
# 27. PRINT TEST RESULTS
# ============================================================

print()
print("BOUNDARY FORCE TESTS")
print("--------------------")


print(
    f"Graphene +0.10 nm: "
    f"{support_test_energy:.8f} kJ/mol"
)

print(
    f"Graphene expected: "
    f"{support_expected:.8f} kJ/mol"
)


print()


print(
    f"Water O +0.10 nm above ceiling: "
    f"{upper_test_energy:.8f} kJ/mol"
)

print(
    f"Water O -0.10 nm below floor:   "
    f"{lower_test_energy:.8f} kJ/mol"
)

print(
    f"Water wall expected:            "
    f"{wall_expected:.8f} kJ/mol"
)


# ============================================================
# 28. VALIDATE STARTING ENERGIES
# ============================================================

if abs(
    support_initial
) > 1e-5:

    raise RuntimeError(
        "Graphene support has nonzero starting energy. "
        f"Measured {support_initial:.8f} kJ/mol."
    )


if abs(
    wall_initial
) > 1e-5:

    raise RuntimeError(
        "Water wall has nonzero starting energy. "
        f"Measured {wall_initial:.8f} kJ/mol."
    )


# ============================================================
# 29. VALIDATE GRAPHENE ENERGY
# ============================================================

if not np.isclose(
    support_test_energy,
    support_expected,
    rtol=1e-5,
    atol=1e-5,
):

    raise RuntimeError(
        "Graphene support validation failed. "
        f"Expected {support_expected:.8f}, "
        f"measured {support_test_energy:.8f} kJ/mol."
    )


# ============================================================
# 30. VALIDATE UPPER WALL ENERGY
# ============================================================

if not np.isclose(
    upper_test_energy,
    wall_expected,
    rtol=1e-5,
    atol=1e-5,
):

    raise RuntimeError(
        "Upper water wall validation failed. "
        f"Expected {wall_expected:.8f}, "
        f"measured {upper_test_energy:.8f} kJ/mol."
    )


# ============================================================
# 31. VALIDATE LOWER WALL ENERGY
# ============================================================

if not np.isclose(
    lower_test_energy,
    wall_expected,
    rtol=1e-5,
    atol=1e-5,
):

    raise RuntimeError(
        "Lower water wall validation failed. "
        f"Expected {wall_expected:.8f}, "
        f"measured {lower_test_energy:.8f} kJ/mol."
    )


# ============================================================
# 32. RESTORE ORIGINAL POSITIONS
# ============================================================

context.setPositions(
    positions_openmm
)


# ============================================================
# 33. SAVE METADATA
# ============================================================

OUTPUT_METADATA.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metadata = {

    "solute_particles": int(
        N_SOLUTE
    ),

    "graphene_carbons": int(
        N_CARBON
    ),

    "graphene_particles_total": int(
        N_GRAPHENE_TOTAL
    ),

    "opc_waters": int(
        n_waters
    ),

    "opc_sites_per_water": int(
        SITES_PER_WATER
    ),

    "opc_total_sites": int(
        expected_water_sites
    ),

    "water_coordinate_shape": list(
        water_positions.shape
    ),

    "oxygen_site_index": int(
        OXYGEN_SITE_INDEX
    ),

    "mean_site0_site1_distance_nm": float(
        mean_d01
    ),

    "mean_site0_site2_distance_nm": float(
        mean_d02
    ),

    "mean_site0_site3_distance_nm": float(
        mean_d03
    ),

    "combined_test_particles": int(
        n_total
    ),

    "lower_boundary_nm": float(
        LOWER_BOUNDARY_NM
    ),

    "upper_boundary_nm": float(
        UPPER_BOUNDARY_NM
    ),

    "aqueous_height_nm": float(
        aqueous_height_nm
    ),

    "periodic_box_z_nm": float(
        BOX_Z_NM
    ),

    "vacuum_gap_nm": float(
        vacuum_gap_nm
    ),

    "slab_center_nm": float(
        Z_MID_NM
    ),

    "slab_halfwidth_nm": float(
        HALF_WIDTH_NM
    ),

    "graphene_k_kj_mol_nm2": float(
        K_GRAPHENE
    ),

    "water_wall_k_kj_mol_nm2": float(
        K_WALL
    ),

    "graphene_restrained_particles": int(
        support_force.getNumParticles()
    ),

    "water_oxygen_wall_particles": int(
        wall_force.getNumParticles()
    ),

    "pyrene_peg5_restrained": False,

    "starting_graphene_support_energy_kj_mol": float(
        support_initial
    ),

    "starting_water_wall_energy_kj_mol": float(
        wall_initial
    ),

    "graphene_test_expected_energy_kj_mol": float(
        support_expected
    ),

    "graphene_test_measured_energy_kj_mol": float(
        support_test_energy
    ),

    "water_test_expected_energy_kj_mol": float(
        wall_expected
    ),

    "upper_wall_test_measured_energy_kj_mol": float(
        upper_test_energy
    ),

    "lower_wall_test_measured_energy_kj_mol": float(
        lower_test_energy
    ),

    "force_field_parameters_modified": False,

    "md_run": False,

    "status": "PASS",
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
# 34. FINAL REPORT
# ============================================================

print()
print("SAVED")
print("-----")
print(
    "Metadata:",
    OUTPUT_METADATA,
)


print()
print(
    "SUPPORTED SLAB BOUNDARIES: PASS"
)

print()
print(
    "No MD has been run."
)
print(
    "No force-field parameters were modified."
)
print(
    "All 6196 OPC waters retain all four sites."
)
print(
    "Water wall acts only on oxygen sites."
)
print(
    "Pyrene-PEG5 remains completely unrestrained."
)

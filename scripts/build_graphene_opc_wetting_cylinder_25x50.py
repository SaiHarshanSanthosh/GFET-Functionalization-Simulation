from pathlib import Path
import json
import math

import numpy as np

import openmm
from openmm import unit

from ase.build import graphene


# ============================================================
# FF-1B WIDE-BOX WETTING CONTROL
#
# Hybrid build:
#
#   GRAPHENE DONOR:
#       validated 25x50 IFF graphene
#
#   WATER DONOR:
#       exact original 1642-water OPC cylinder
#
# Final system:
#       25x50 graphene
#       1642 OPC waters
#       PME
#       12 nm z box
#       graphene z support
#       NO water walls
#       NO ligand
#
# Yeh-Berkowitz is NOT added here.
# That is the next separately validated step.
# ============================================================


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)


GRAPHENE_XML = (
    ROOT
    / "parameters"
    / "iff"
    / "graphene_iff_full_terms_validation_nocutoff_25x50.xml"
)


WETTING_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder.xml"
)


WETTING_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_positions_nm.npy"
)


OUTPUT_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_25x50.xml"
)


OUTPUT_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_25x50_positions_nm.npy"
)


OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_cylinder_25x50_build.json"
)


# ============================================================
# INDEXING
# ============================================================

N_CARBON_NEW = 2500
N_GRAPHENE_NEW = 7500

N_GRAPHENE_OLD = 3750

N_WATERS = 1642
WATER_SITES = 4

OLD_WATER_START = N_GRAPHENE_OLD

N_WATER_PARTICLES = (
    N_WATERS
    *
    WATER_SITES
)

EXPECTED_NEW_PARTICLES = (
    N_GRAPHENE_NEW
    +
    N_WATER_PARTICLES
)


TARGET_CARBON_Z_NM = 0.500

TARGET_LZ_NM = 12.0

PI_DISTANCE_NM = 0.065


# ============================================================
# HELPERS
# ============================================================

def clone_force(force):

    return openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(
            force
        )
    )


def vec_nm(v):

    return np.asarray(
        [
            v[0].value_in_unit(
                unit.nanometer
            ),
            v[1].value_in_unit(
                unit.nanometer
            ),
            v[2].value_in_unit(
                unit.nanometer
            ),
        ],
        dtype=float,
    )


def circular_mean_y(
    y,
    ly,
):

    angles = (
        2.0
        *
        np.pi
        *
        np.mod(
            y,
            ly
        )
        /
        ly
    )

    angle = math.atan2(
        float(
            np.mean(
                np.sin(
                    angles
                )
            )
        ),
        float(
            np.mean(
                np.cos(
                    angles
                )
            )
        ),
    )

    if angle < 0:
        angle += (
            2.0
            *
            np.pi
        )

    return (
        angle
        *
        ly
        /
        (
            2.0
            *
            np.pi
        )
    )


# ============================================================
# LOAD DONORS
# ============================================================

print()
print("=" * 80)
print("BUILD 25x50 GRAPHENE / OPC WETTING CYLINDER")
print("=" * 80)


graphene_system = (
    openmm.XmlSerializer.deserialize(
        GRAPHENE_XML.read_text(
            encoding="utf-8"
        )
    )
)


wetting_system = (
    openmm.XmlSerializer.deserialize(
        WETTING_XML.read_text(
            encoding="utf-8"
        )
    )
)


wetting_positions_nm = np.load(
    WETTING_POSITIONS
)


if (
    graphene_system.getNumParticles()
    !=
    N_GRAPHENE_NEW
):
    raise RuntimeError(
        "Unexpected wide graphene particle count."
    )


if (
    wetting_system.getNumParticles()
    !=
    N_GRAPHENE_OLD
    +
    N_WATER_PARTICLES
):
    raise RuntimeError(
        "Unexpected old wetting donor size."
    )


if wetting_positions_nm.shape != (
    wetting_system.getNumParticles(),
    3,
):
    raise RuntimeError(
        "Old wetting position shape mismatch."
    )


print()
print(
    f"Wide graphene particles: "
    f"{graphene_system.getNumParticles()}"
)

print(
    f"Water donor particles:   "
    f"{wetting_system.getNumParticles()}"
)

print(
    f"OPC waters:              "
    f"{N_WATERS}"
)


# ============================================================
# FIND NONBONDED DONOR FORCES
# ============================================================

graphene_nb = None
wetting_nb = None


for force in graphene_system.getForces():

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):
        graphene_nb = force


for force in wetting_system.getForces():

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):
        wetting_nb = force


if graphene_nb is None:
    raise RuntimeError(
        "Graphene donor has no NonbondedForce."
    )


if wetting_nb is None:
    raise RuntimeError(
        "Wetting donor has no NonbondedForce."
    )


print()
print("NONBONDED DONORS")

print(
    f"  graphene particles: "
    f"{graphene_nb.getNumParticles()}"
)

print(
    f"  graphene exceptions: "
    f"{graphene_nb.getNumExceptions()}"
)

print(
    f"  wetting particles:   "
    f"{wetting_nb.getNumParticles()}"
)

print(
    f"  wetting exceptions:  "
    f"{wetting_nb.getNumExceptions()}"
)


# ============================================================
# START FROM GRAPHENE DONOR
#
# Keep its bonded forces.
# Remove its validation-only NoCutoff NonbondedForce.
# ============================================================

new_system = openmm.System()


for i in range(
    N_GRAPHENE_NEW
):

    new_system.addParticle(
        graphene_system.getParticleMass(
            i
        )
    )


# Copy only graphene bonded forces.
for force in graphene_system.getForces():

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):
        continue

    new_system.addForce(
        clone_force(
            force
        )
    )


# ============================================================
# ADD WATER PARTICLES
# ============================================================

old_to_new_water = {}


for old_index in range(
    OLD_WATER_START,
    wetting_system.getNumParticles(),
):

    new_index = (
        N_GRAPHENE_NEW
        +
        (
            old_index
            -
            OLD_WATER_START
        )
    )

    old_to_new_water[
        old_index
    ] = new_index

    new_system.addParticle(
        wetting_system.getParticleMass(
            old_index
        )
    )


if (
    new_system.getNumParticles()
    !=
    EXPECTED_NEW_PARTICLES
):
    raise RuntimeError(
        "Particle count mismatch after water copy."
    )


# ============================================================
# BOX
#
# Use wide graphene a/b.
# Use wetting benchmark's 12 nm z vacuum.
# ============================================================

g_a, g_b, g_c = (
    graphene_system
    .getDefaultPeriodicBoxVectors()
)


a_nm = vec_nm(
    g_a
)

b_nm = vec_nm(
    g_b
)


a = openmm.Vec3(
    *a_nm
) * unit.nanometer


b = openmm.Vec3(
    *b_nm
) * unit.nanometer


c = openmm.Vec3(
    0.0,
    0.0,
    TARGET_LZ_NM,
) * unit.nanometer


new_system.setDefaultPeriodicBoxVectors(
    a,
    b,
    c,
)


LX_NM = float(
    np.linalg.norm(
        a_nm
    )
)


LY_NM = float(
    np.linalg.norm(
        b_nm
    )
)


print()
print("FINAL BOX")

print(
    f"  Lx = {LX_NM:.9f} nm"
)

print(
    f"  Ly = {LY_NM:.9f} nm"
)

print(
    f"  Lz = {TARGET_LZ_NM:.9f} nm"
)


# ============================================================
# WATER CONSTRAINTS
#
# Old wetting donor contains only OPC constraints.
# ============================================================

copied_constraints = 0


for i in range(
    wetting_system.getNumConstraints()
):

    p1, p2, distance = (
        wetting_system
        .getConstraintParameters(
            i
        )
    )

    p1 = int(
        p1
    )

    p2 = int(
        p2
    )


    if (
        p1 < OLD_WATER_START
        or
        p2 < OLD_WATER_START
    ):
        raise RuntimeError(
            "Unexpected non-water constraint "
            "in wetting donor."
        )


    new_system.addConstraint(
        old_to_new_water[
            p1
        ],
        old_to_new_water[
            p2
        ],
        distance,
    )

    copied_constraints += 1


if copied_constraints != (
    3
    *
    N_WATERS
):
    raise RuntimeError(
        "Expected exactly 3 constraints "
        "per OPC water."
    )


# ============================================================
# WATER VIRTUAL SITES
# ============================================================

copied_virtual_sites = 0


for old_index in range(
    OLD_WATER_START,
    wetting_system.getNumParticles(),
):

    if not wetting_system.isVirtualSite(
        old_index
    ):
        continue


    site = wetting_system.getVirtualSite(
        old_index
    )


    if not isinstance(
        site,
        openmm.ThreeParticleAverageSite,
    ):
        raise RuntimeError(
            "Unexpected virtual site type."
        )


    p0 = int(
        site.getParticle(
            0
        )
    )

    p1 = int(
        site.getParticle(
            1
        )
    )

    p2 = int(
        site.getParticle(
            2
        )
    )


    new_site = (
        openmm.ThreeParticleAverageSite(
            old_to_new_water[
                p0
            ],
            old_to_new_water[
                p1
            ],
            old_to_new_water[
                p2
            ],
            site.getWeight(
                0
            ),
            site.getWeight(
                1
            ),
            site.getWeight(
                2
            ),
        )
    )


    new_system.setVirtualSite(
        old_to_new_water[
            old_index
        ],
        new_site,
    )


    copied_virtual_sites += 1


if copied_virtual_sites != N_WATERS:

    raise RuntimeError(
        "Unexpected OPC virtual-site count."
    )


# ============================================================
# UNIFIED PME NONBONDED FORCE
#
# Settings come from the successful wetting donor.
#
# Graphene particle parameters/exceptions come from
# the validated 25x50 graphene donor.
#
# Water parameters/exceptions come from exact 1642-water
# wetting donor.
# ============================================================

nb = openmm.NonbondedForce()


nb.setName(
    "IFF + OPC unified nonbonded 25x50 wetting"
)


nb.setForceGroup(
    wetting_nb.getForceGroup()
)


nb.setNonbondedMethod(
    wetting_nb.getNonbondedMethod()
)


nb.setCutoffDistance(
    wetting_nb.getCutoffDistance()
)


nb.setEwaldErrorTolerance(
    wetting_nb
    .getEwaldErrorTolerance()
)


nb.setUseDispersionCorrection(
    wetting_nb
    .getUseDispersionCorrection()
)


nb.setReactionFieldDielectric(
    wetting_nb
    .getReactionFieldDielectric()
)


nb.setUseSwitchingFunction(
    wetting_nb
    .getUseSwitchingFunction()
)


if wetting_nb.getUseSwitchingFunction():

    nb.setSwitchingDistance(
        wetting_nb
        .getSwitchingDistance()
    )


try:

    nb.setReciprocalSpaceForceGroup(
        wetting_nb
        .getReciprocalSpaceForceGroup()
    )

except Exception:

    pass


try:

    nb.setExceptionsUsePeriodicBoundaryConditions(
        wetting_nb
        .getExceptionsUsePeriodicBoundaryConditions()
    )

except Exception:

    pass


if (
    graphene_nb.getNumParticleParameterOffsets()
    !=
    0
    or
    graphene_nb.getNumExceptionParameterOffsets()
    !=
    0
    or
    wetting_nb.getNumParticleParameterOffsets()
    !=
    0
    or
    wetting_nb.getNumExceptionParameterOffsets()
    !=
    0
):
    raise RuntimeError(
        "Parameter offsets are not supported "
        "by this hybrid builder."
    )


# ------------------------------------------------------------
# Graphene particles
# ------------------------------------------------------------

for i in range(
    N_GRAPHENE_NEW
):

    q, sigma, epsilon = (
        graphene_nb
        .getParticleParameters(
            i
        )
    )

    nb.addParticle(
        q,
        sigma,
        epsilon,
    )


# ------------------------------------------------------------
# Water particles
# ------------------------------------------------------------

for old_index in range(
    OLD_WATER_START,
    wetting_system.getNumParticles(),
):

    q, sigma, epsilon = (
        wetting_nb
        .getParticleParameters(
            old_index
        )
    )

    nb.addParticle(
        q,
        sigma,
        epsilon,
    )


if (
    nb.getNumParticles()
    !=
    EXPECTED_NEW_PARTICLES
):
    raise RuntimeError(
        "Unified NonbondedForce particle mismatch."
    )


# ------------------------------------------------------------
# Graphene exceptions
# ------------------------------------------------------------

for i in range(
    graphene_nb.getNumExceptions()
):

    p1, p2, qq, sigma, epsilon = (
        graphene_nb
        .getExceptionParameters(
            i
        )
    )

    nb.addException(
        int(
            p1
        ),
        int(
            p2
        ),
        qq,
        sigma,
        epsilon,
    )


n_graphene_exceptions = (
    graphene_nb.getNumExceptions()
)


# ------------------------------------------------------------
# Water-only exceptions from old wetting donor
# ------------------------------------------------------------

water_exception_count = 0
cross_exception_count = 0


for i in range(
    wetting_nb.getNumExceptions()
):

    p1, p2, qq, sigma, epsilon = (
        wetting_nb
        .getExceptionParameters(
            i
        )
    )

    p1 = int(
        p1
    )

    p2 = int(
        p2
    )


    p1_water = (
        p1
        >=
        OLD_WATER_START
    )

    p2_water = (
        p2
        >=
        OLD_WATER_START
    )


    if (
        p1_water
        and
        p2_water
    ):

        nb.addException(
            old_to_new_water[
                p1
            ],
            old_to_new_water[
                p2
            ],
            qq,
            sigma,
            epsilon,
        )

        water_exception_count += 1


    elif (
        p1_water
        !=
        p2_water
    ):

        cross_exception_count += 1


if cross_exception_count != 0:

    raise RuntimeError(
        "Unexpected graphene-water exception "
        "in old wetting donor."
    )


expected_water_exceptions = (
    6
    *
    N_WATERS
)


if (
    water_exception_count
    !=
    expected_water_exceptions
):

    raise RuntimeError(
        "Unexpected OPC internal exception count: "
        f"{water_exception_count}"
    )


expected_total_exceptions = (
    n_graphene_exceptions
    +
    expected_water_exceptions
)


if (
    nb.getNumExceptions()
    !=
    expected_total_exceptions
):

    raise RuntimeError(
        "Unified exception count mismatch."
    )


new_system.addForce(
    nb
)


# ============================================================
# GRAPHENE SUPPORT
#
# Exact same z restraint used previously:
#
#   0.5*kz*periodicdistance(0,0,z,0,0,z0)^2
#
# Applied only to C cores.
# ============================================================

support = openmm.CustomExternalForce(
    "0.5*kz*periodicdistance(0,0,z,0,0,z0)^2"
)


support.setName(
    "GrapheneSupportZRestraint_global_z0"
)


support.setForceGroup(
    5
)


support.addGlobalParameter(
    "kz",
    1000.0,
)


support.addGlobalParameter(
    "z0",
    TARGET_CARBON_Z_NM,
)


for i in range(
    N_CARBON_NEW
):

    support.addParticle(
        i,
        [],
    )


new_system.addForce(
    support
)


# ============================================================
# GRAPHENE POSITIONS
#
# Rebuild the exact 25x50 ASE lattice and IFF pi sites.
# Then translate only in z so carbon plane = 0.500 nm.
# ============================================================

sheet = graphene(
    formula="C2",
    a=2.46,
    size=(
        25,
        50,
        1,
    ),
    vacuum=20.0,
)


sheet.pbc = (
    True,
    True,
    False,
)


carbon_A = np.asarray(
    sheet.get_positions(),
    dtype=float,
)


cell_A = np.asarray(
    sheet.cell.array,
    dtype=float,
)


if len(
    carbon_A
) != N_CARBON_NEW:

    raise RuntimeError(
        "ASE carbon count mismatch."
    )


normal = np.cross(
    cell_A[0],
    cell_A[1],
)


normal /= np.linalg.norm(
    normal
)


pi_above_A = (
    carbon_A
    +
    0.65
    *
    normal
)


pi_below_A = (
    carbon_A
    -
    0.65
    *
    normal
)


graphene_positions_nm = np.vstack(
    [
        carbon_A,
        pi_above_A,
        pi_below_A,
    ]
) / 10.0


carbon_plane_initial = float(
    np.mean(
        graphene_positions_nm[
            :N_CARBON_NEW,
            2,
        ]
    )
)


z_shift = (
    TARGET_CARBON_Z_NM
    -
    carbon_plane_initial
)


graphene_positions_nm[
    :,
    2,
] += z_shift


carbon_plane_final = float(
    np.mean(
        graphene_positions_nm[
            :N_CARBON_NEW,
            2,
        ]
    )
)


# ============================================================
# WATER POSITIONS
#
# Copy the exact old 1642-water cylinder.
#
# Move the entire cylinder from old Ly/2 to new Ly/2.
# Whole molecules are translated together.
#
# Also canonicalize each whole water molecule along x.
# ============================================================

water_positions_nm = np.array(
    wetting_positions_nm[
        OLD_WATER_START:
    ],
    dtype=float,
    copy=True,
)


if water_positions_nm.shape != (
    N_WATER_PARTICLES,
    3,
):

    raise RuntimeError(
        "Water coordinate count mismatch."
    )


old_a, old_b, old_c = (
    wetting_system
    .getDefaultPeriodicBoxVectors()
)


old_b_nm = vec_nm(
    old_b
)

# The old box is triclinic:
#
#     b = (-3.075, 5.326..., 0)
#
# For a cylinder periodic along x, the transverse periodic
# spacing is the y component of b, not |b|.
OLD_LY_NM = abs(
    float(
        old_b_nm[1]
    )
)


old_O_indices_local = np.arange(
    0,
    N_WATER_PARTICLES,
    WATER_SITES,
    dtype=int,
)


old_center_y = circular_mean_y(
    water_positions_nm[
        old_O_indices_local,
        1,
    ],
    OLD_LY_NM,
)


new_center_y = (
    0.5
    *
    LY_NM
)


delta_y = (
    new_center_y
    -
    old_center_y
)


# Fold each WHOLE water into the old cylinder's central
# triclinic image.
#
# The old cell has a skewed b vector.  Therefore crossing a
# periodic y boundary also changes x.  We must apply the full
# old lattice vector, not just wrap the Cartesian y value.
for water_number in range(
    N_WATERS
):

    start = (
        WATER_SITES
        *
        water_number
    )

    stop = (
        start
        +
        WATER_SITES
    )


    oxygen_y_old = float(
        water_positions_nm[
            start,
            1,
        ]
    )


    # Number of old b-images separating this oxygen from the
    # central cylinder representation.
    image_b = int(
        round(
            (
                oxygen_y_old
                -
                old_center_y
            )
            /
            OLD_LY_NM
        )
    )


    # Apply the FULL old lattice-vector shift so this operation
    # is exactly periodic-equivalent in the old triclinic cell.
    water_positions_nm[
        start:stop,
        :,
    ] -= (
        image_b
        *
        old_b_nm
    )


    # Now physically translate the centered old cylinder into
    # the middle of the new wide box.
    water_positions_nm[
        start:stop,
        1,
    ] += (
        new_center_y
        -
        old_center_y
    )


    # x is periodic with the same 6.15 nm period in both cells.
    # Canonicalize each whole molecule according to its oxygen.
    oxygen_x = float(
        water_positions_nm[
            start,
            0,
        ]
    )


    image_a = math.floor(
        oxygen_x
        /
        LX_NM
    )


    water_positions_nm[
        start:stop,
        0,
    ] -= (
        image_a
        *
        LX_NM
    )


    # Safety check: the recentered cylinder should now sit well
    # inside the new 10.65 nm transverse box.
    oxygen_y_new = float(
        water_positions_nm[
            start,
            1,
        ]
    )


    if not (
        0.0
        <=
        oxygen_y_new
        <
        LY_NM
    ):

        raise RuntimeError(
            "Recentered water lies outside new y box: "
            f"water={water_number}, "
            f"y={oxygen_y_new:.6f} nm"
        )

# ============================================================
# FINAL POSITIONS
# ============================================================

new_positions_nm = np.vstack(
    [
        graphene_positions_nm,
        water_positions_nm,
    ]
)


if new_positions_nm.shape != (
    EXPECTED_NEW_PARTICLES,
    3,
):

    raise RuntimeError(
        "Final position array mismatch."
    )


# ============================================================
# STRUCTURAL CHECKS
# ============================================================

oxygen_indices = (
    N_GRAPHENE_NEW
    +
    np.arange(
        0,
        N_WATER_PARTICLES,
        WATER_SITES,
        dtype=int,
    )
)


O_xyz = (
    new_positions_nm[
        oxygen_indices
    ]
)


O_heights = (
    O_xyz[
        :,
        2
    ]
    -
    carbon_plane_final
)


O_y = np.mod(
    O_xyz[
        :,
        1
    ],
    LY_NM,
)


new_cylinder_center_y = circular_mean_y(
    O_y,
    LY_NM,
)


angles = (
    2.0
    *
    np.pi
    *
    O_y
    /
    LY_NM
)


# Approximate initial periodic dry gap using oxygen centers.
y_sorted = np.sort(
    O_y
)


y_gaps = np.diff(
    np.concatenate(
        [
            y_sorted,
            [
                y_sorted[0]
                +
                LY_NM
            ],
        ]
    )
)


largest_initial_dry_gap = float(
    np.max(
        y_gaps
    )
)


occupied_width = (
    LY_NM
    -
    largest_initial_dry_gap
)


# ============================================================
# TOTAL CHARGE
# ============================================================

total_charge_e = 0.0


for i in range(
    nb.getNumParticles()
):

    q, sigma, epsilon = (
        nb.getParticleParameters(
            i
        )
    )

    total_charge_e += q.value_in_unit(
        unit.elementary_charge
    )


if abs(
    total_charge_e
) > 1.0e-5:

    raise RuntimeError(
        f"System is not neutral: "
        f"{total_charge_e} e"
    )


# ============================================================
# SYSTEM COUNTS
# ============================================================

n_virtual = sum(
    new_system.isVirtualSite(i)
    for i in range(
        new_system.getNumParticles()
    )
)


print()
print("=" * 80)
print("NEW 25x50 WETTING SYSTEM")
print("=" * 80)

print(
    f"Particles:            "
    f"{new_system.getNumParticles()}"
)

print(
    f"Graphene particles:   "
    f"{N_GRAPHENE_NEW}"
)

print(
    f"Graphene C cores:     "
    f"{N_CARBON_NEW}"
)

print(
    f"OPC waters:           "
    f"{N_WATERS}"
)

print(
    f"Virtual sites:        "
    f"{n_virtual}"
)

print(
    f"Constraints:          "
    f"{new_system.getNumConstraints()}"
)

print(
    f"Forces:               "
    f"{new_system.getNumForces()}"
)

print(
    f"NB exceptions:        "
    f"{nb.getNumExceptions()}"
)

print(
    f"Total charge:         "
    f"{total_charge_e:+.12e} e"
)


print()
print("GEOMETRY")

print(
    f"Carbon plane:         "
    f"{carbon_plane_final:.6f} nm"
)

print(
    f"Old cylinder center y:"
    f" {old_center_y:.6f} nm"
)

print(
    f"New cylinder center y:"
    f" {new_cylinder_center_y:.6f} nm"
)

print(
    f"Lowest O above C:     "
    f"{np.min(O_heights):.6f} nm"
)

print(
    f"Highest O above C:    "
    f"{np.max(O_heights):.6f} nm"
)

print(
    f"Initial occupied y:   "
    f"{occupied_width:.6f} nm"
)

print(
    f"Initial largest gap:  "
    f"{largest_initial_dry_gap:.6f} nm"
)


# ============================================================
# FINITE ENERGY SMOKE TEST
# ============================================================

integrator = openmm.VerletIntegrator(
    0.001
    *
    unit.picoseconds
)


platform = (
    openmm.Platform
    .getPlatformByName(
        "CUDA"
    )
)


context = openmm.Context(
    new_system,
    integrator,
    platform,
    {
        "Precision": "mixed"
    },
)


context.setPositions(
    new_positions_nm
    *
    unit.nanometer
)


context.computeVirtualSites()


state = context.getState(
    getEnergy=True,
)


energy_kj = float(
    state
    .getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


if not np.isfinite(
    energy_kj
):

    raise RuntimeError(
        "Initial energy is not finite."
    )


print()
print("CUDA SMOKE TEST")

print(
    f"Initial PE:           "
    f"{energy_kj:.6f} kJ/mol"
)

print(
    "Finite energy:        PASS"
)


del context
del integrator


# ============================================================
# SAVE
# ============================================================

OUTPUT_XML.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_JSON.parent.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_XML.write_text(
    openmm.XmlSerializer.serialize(
        new_system
    ),
    encoding="utf-8",
)


np.save(
    OUTPUT_POSITIONS,
    new_positions_nm,
)


metadata = {

    "purpose": (
        "25x50 wide-box graphene/OPC wetting "
        "finite-size control"
    ),

    "graphene_source": str(
        GRAPHENE_XML
    ),

    "water_source": str(
        WETTING_XML
    ),

    "water_position_source": str(
        WETTING_POSITIONS
    ),

    "output_system": str(
        OUTPUT_XML
    ),

    "output_positions": str(
        OUTPUT_POSITIONS
    ),

    "particle_count": int(
        new_system.getNumParticles()
    ),

    "graphene_particles": int(
        N_GRAPHENE_NEW
    ),

    "graphene_carbon_cores": int(
        N_CARBON_NEW
    ),

    "water_count": int(
        N_WATERS
    ),

    "virtual_sites": int(
        n_virtual
    ),

    "constraints": int(
        new_system.getNumConstraints()
    ),

    "nonbonded_exceptions": int(
        nb.getNumExceptions()
    ),

    "total_charge_e": float(
        total_charge_e
    ),

    "box_nm": {
        "a": a_nm.tolist(),
        "b": b_nm.tolist(),
        "c": [
            0.0,
            0.0,
            TARGET_LZ_NM,
        ],
    },

    "carbon_plane_nm": float(
        carbon_plane_final
    ),

    "water_O_height_range_nm": [
        float(
            np.min(
                O_heights
            )
        ),
        float(
            np.max(
                O_heights
            )
        ),
    ],

    "initial_periodic_y": {
        "occupied_width_nm": float(
            occupied_width
        ),
        "largest_dry_gap_nm": float(
            largest_initial_dry_gap
        ),
    },

    "support": {
        "particle_count": int(
            N_CARBON_NEW
        ),
        "kz_kJ_mol_nm2": 1000.0,
        "z0_nm": TARGET_CARBON_Z_NM,
    },

    "water_walls": False,

    "yeh_berkowitz": False,

    "initial_PE_kJ_mol": float(
        energy_kj
    ),
}


OUTPUT_JSON.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(
    f"System:"
    f"\n  {OUTPUT_XML}"
)

print(
    f"Positions:"
    f"\n  {OUTPUT_POSITIONS}"
)

print(
    f"Metadata:"
    f"\n  {OUTPUT_JSON}"
)

print()
print(
    "IMPORTANT: Yeh-Berkowitz correction "
    "has NOT been added yet."
)

print()
print(
    "GRAPHENE_OPC_WETTING_25X50_BUILD_PASS"
)

print("=" * 80)
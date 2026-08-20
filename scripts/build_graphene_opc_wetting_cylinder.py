from pathlib import Path
import json

import numpy as np
import openmm
from openmm import unit


# ============================================================
# FF-1B BUILD
#
# Dedicated graphene / OPC wetting benchmark.
#
# Starting source:
#   exact validated production System + 5.5 ns endpoint
#
# New system:
#   - graphene only
#   - NO Pyrene-PEG5
#   - cylindrical OPC water droplet
#   - periodic along x
#   - vacuum around droplet in y/z
#   - graphene z support retained
#   - water slab walls REMOVED
#
# NO minimization
# NO MD
# ============================================================


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

SOURCE_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "pyrene_peg5_graphene_one_sided_opc_supported_pme_production_safe.xml"
)

SOURCE_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "pyrene_peg5_graphene_one_sided_opc_production_300K_5500ps_positions_nm.npy"
)

OUTPUT_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder.xml"
)

OUTPUT_POSITIONS = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_positions_nm.npy"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_cylinder_build.json"
)


# ============================================================
# ORIGINAL SYSTEM INDEXING
# ============================================================

N_GRAPHENE = 3750

GRAPHENE_C_END = 1250

LIGAND_START = 3750
LIGAND_END = 3820

WATER_START = 3820

EXPECTED_SOURCE_PARTICLES = 28224
EXPECTED_SOURCE_WATERS = 6101


# ============================================================
# CYLINDER DESIGN
#
# Cylinder axis = periodic x direction.
#
# Cross-section is a circle in y/z.
#
# Radius 1.60 nm:
#
#   diameter = 3.20 nm
#
# y box span is ~5.326 nm, leaving ~2.13 nm between
# neighboring periodic cylinder surfaces.
#
# Bottom water-O target ~0.80 nm.
# Graphene carbon plane ~0.50 nm.
# ============================================================

CYLINDER_RADIUS_NM = 1.60

BOTTOM_O_TARGET_NM = 0.80

MIN_EXPECTED_WATERS = 1000
MAX_EXPECTED_WATERS = 2500


# ============================================================
# HELPERS
# ============================================================

def vec_nm(v):

    return np.array(
        [
            v[0].value_in_unit(unit.nanometer),
            v[1].value_in_unit(unit.nanometer),
            v[2].value_in_unit(unit.nanometer),
        ],
        dtype=float,
    )


def copy_force_metadata(
    source,
    target,
):

    target.setForceGroup(
        source.getForceGroup()
    )

    try:
        target.setName(
            source.getName()
        )
    except Exception:
        pass


def copy_custom_bond_force(
    source,
    old_to_new,
):

    target = openmm.CustomBondForce(
        source.getEnergyFunction()
    )

    for i in range(
        source.getNumPerBondParameters()
    ):
        target.addPerBondParameter(
            source.getPerBondParameterName(i)
        )

    for i in range(
        source.getNumGlobalParameters()
    ):
        target.addGlobalParameter(
            source.getGlobalParameterName(i),
            source.getGlobalParameterDefaultValue(i),
        )

    try:
        target.setUsesPeriodicBoundaryConditions(
            source.usesPeriodicBoundaryConditions()
        )
    except Exception:
        pass

    for i in range(
        source.getNumBonds()
    ):

        p1, p2, params = (
            source.getBondParameters(i)
        )

        p1 = int(p1)
        p2 = int(p2)

        if (
            p1 in old_to_new
            and
            p2 in old_to_new
        ):
            target.addBond(
                old_to_new[p1],
                old_to_new[p2],
                params,
            )

    copy_force_metadata(
        source,
        target,
    )

    return target


def copy_custom_angle_force(
    source,
    old_to_new,
):

    target = openmm.CustomAngleForce(
        source.getEnergyFunction()
    )

    for i in range(
        source.getNumPerAngleParameters()
    ):
        target.addPerAngleParameter(
            source.getPerAngleParameterName(i)
        )

    for i in range(
        source.getNumGlobalParameters()
    ):
        target.addGlobalParameter(
            source.getGlobalParameterName(i),
            source.getGlobalParameterDefaultValue(i),
        )

    try:
        target.setUsesPeriodicBoundaryConditions(
            source.usesPeriodicBoundaryConditions()
        )
    except Exception:
        pass

    for i in range(
        source.getNumAngles()
    ):

        p1, p2, p3, params = (
            source.getAngleParameters(i)
        )

        p1 = int(p1)
        p2 = int(p2)
        p3 = int(p3)

        if (
            p1 in old_to_new
            and
            p2 in old_to_new
            and
            p3 in old_to_new
        ):
            target.addAngle(
                old_to_new[p1],
                old_to_new[p2],
                old_to_new[p3],
                params,
            )

    copy_force_metadata(
        source,
        target,
    )

    return target


def copy_periodic_torsion_force(
    source,
    old_to_new,
):

    target = openmm.PeriodicTorsionForce()

    try:
        target.setUsesPeriodicBoundaryConditions(
            source.usesPeriodicBoundaryConditions()
        )
    except Exception:
        pass

    for i in range(
        source.getNumTorsions()
    ):

        (
            p1,
            p2,
            p3,
            p4,
            periodicity,
            phase,
            k,
        ) = source.getTorsionParameters(i)

        indices = [
            int(p1),
            int(p2),
            int(p3),
            int(p4),
        ]

        if all(
            p in old_to_new
            for p in indices
        ):

            target.addTorsion(
                old_to_new[indices[0]],
                old_to_new[indices[1]],
                old_to_new[indices[2]],
                old_to_new[indices[3]],
                periodicity,
                phase,
                k,
            )

    copy_force_metadata(
        source,
        target,
    )

    return target


def copy_harmonic_bond_force(
    source,
    old_to_new,
):

    target = openmm.HarmonicBondForce()

    try:
        target.setUsesPeriodicBoundaryConditions(
            source.usesPeriodicBoundaryConditions()
        )
    except Exception:
        pass

    for i in range(
        source.getNumBonds()
    ):

        p1, p2, length, k = (
            source.getBondParameters(i)
        )

        p1 = int(p1)
        p2 = int(p2)

        if (
            p1 in old_to_new
            and
            p2 in old_to_new
        ):

            target.addBond(
                old_to_new[p1],
                old_to_new[p2],
                length,
                k,
            )

    copy_force_metadata(
        source,
        target,
    )

    return target


def copy_harmonic_angle_force(
    source,
    old_to_new,
):

    target = openmm.HarmonicAngleForce()

    try:
        target.setUsesPeriodicBoundaryConditions(
            source.usesPeriodicBoundaryConditions()
        )
    except Exception:
        pass

    for i in range(
        source.getNumAngles()
    ):

        p1, p2, p3, angle, k = (
            source.getAngleParameters(i)
        )

        p1 = int(p1)
        p2 = int(p2)
        p3 = int(p3)

        if (
            p1 in old_to_new
            and
            p2 in old_to_new
            and
            p3 in old_to_new
        ):

            target.addAngle(
                old_to_new[p1],
                old_to_new[p2],
                old_to_new[p3],
                angle,
                k,
            )

    copy_force_metadata(
        source,
        target,
    )

    return target


def copy_nonbonded_force(
    source,
    kept_old_indices,
    old_to_new,
):

    if source.getNumParticleParameterOffsets() != 0:
        raise RuntimeError(
            "Particle parameter offsets are not supported "
            "by this subset builder."
        )

    if source.getNumExceptionParameterOffsets() != 0:
        raise RuntimeError(
            "Exception parameter offsets are not supported "
            "by this subset builder."
        )


    target = openmm.NonbondedForce()

    target.setNonbondedMethod(
        source.getNonbondedMethod()
    )

    target.setCutoffDistance(
        source.getCutoffDistance()
    )

    target.setReactionFieldDielectric(
        source.getReactionFieldDielectric()
    )

    target.setEwaldErrorTolerance(
        source.getEwaldErrorTolerance()
    )

    target.setUseDispersionCorrection(
        source.getUseDispersionCorrection()
    )

    try:
        target.setUseSwitchingFunction(
            source.getUseSwitchingFunction()
        )

        target.setSwitchingDistance(
            source.getSwitchingDistance()
        )
    except Exception:
        pass


    alpha, nx, ny, nz = (
        source.getPMEParameters()
    )

    # If explicit PME grid parameters were supplied in the
    # source, preserve them exactly.
    #
    # Otherwise nx=ny=nz=0 and OpenMM chooses PME parameters
    # from the copied Ewald error tolerance.
    if (
        int(nx) != 0
        or
        int(ny) != 0
        or
        int(nz) != 0
    ):
        target.setPMEParameters(
            alpha,
            int(nx),
            int(ny),
            int(nz),
        )


    for old_index in kept_old_indices:

        q, sigma, epsilon = (
            source.getParticleParameters(
                old_index
            )
        )

        target.addParticle(
            q,
            sigma,
            epsilon,
        )


    copied_exceptions = 0


    for i in range(
        source.getNumExceptions()
    ):

        (
            p1,
            p2,
            charge_prod,
            sigma,
            epsilon,
        ) = source.getExceptionParameters(i)

        p1 = int(p1)
        p2 = int(p2)

        if (
            p1 in old_to_new
            and
            p2 in old_to_new
        ):

            target.addException(
                old_to_new[p1],
                old_to_new[p2],
                charge_prod,
                sigma,
                epsilon,
            )

            copied_exceptions += 1


    copy_force_metadata(
        source,
        target,
    )

    return (
        target,
        copied_exceptions,
    )


def copy_support_force(
    source,
    old_to_new,
):

    target = openmm.CustomExternalForce(
        source.getEnergyFunction()
    )

    for i in range(
        source.getNumPerParticleParameters()
    ):
        target.addPerParticleParameter(
            source.getPerParticleParameterName(i)
        )

    for i in range(
        source.getNumGlobalParameters()
    ):
        target.addGlobalParameter(
            source.getGlobalParameterName(i),
            source.getGlobalParameterDefaultValue(i),
        )


    copied = 0


    for i in range(
        source.getNumParticles()
    ):

        old_particle, parameters = (
            source.getParticleParameters(i)
        )

        old_particle = int(
            old_particle
        )

        if old_particle in old_to_new:

            target.addParticle(
                old_to_new[
                    old_particle
                ],
                parameters,
            )

            copied += 1


    copy_force_metadata(
        source,
        target,
    )

    return (
        target,
        copied,
    )


# ============================================================
# LOAD SOURCE
# ============================================================

print()
print("=" * 78)
print("FF-1B GRAPHENE / OPC WETTING CYLINDER BUILD")
print("=" * 78)


if not SOURCE_XML.exists():
    raise FileNotFoundError(
        SOURCE_XML
    )

if not SOURCE_POSITIONS.exists():
    raise FileNotFoundError(
        SOURCE_POSITIONS
    )


source_system = openmm.XmlSerializer.deserialize(
    SOURCE_XML.read_text(
        encoding="utf-8"
    )
)

source_positions_nm = np.load(
    SOURCE_POSITIONS
)


if (
    source_system.getNumParticles()
    !=
    EXPECTED_SOURCE_PARTICLES
):
    raise RuntimeError(
        "Unexpected source particle count."
    )


if source_positions_nm.shape != (
    EXPECTED_SOURCE_PARTICLES,
    3,
):
    raise RuntimeError(
        "Unexpected source coordinate shape."
    )


print(
    f"Source particles: "
    f"{source_system.getNumParticles()}"
)


# ============================================================
# BOX
# ============================================================

a, b, c = (
    source_system
    .getDefaultPeriodicBoxVectors()
)

a_nm = vec_nm(a)
b_nm = vec_nm(b)
c_nm = vec_nm(c)


lx_nm = float(
    np.linalg.norm(a_nm)
)

ly_nm = float(
    abs(
        b_nm[1]
    )
)

lz_nm = float(
    np.linalg.norm(c_nm)
)


print()
print("Box:")

print(
    f"  Lx-like a = "
    f"{lx_nm:.6f} nm"
)

print(
    f"  Ly span   = "
    f"{ly_nm:.6f} nm"
)

print(
    f"  Lz        = "
    f"{lz_nm:.6f} nm"
)


# ============================================================
# GRAPHENE PLANE
# ============================================================

carbon_plane_nm = float(
    np.mean(
        source_positions_nm[
            :GRAPHENE_C_END,
            2,
        ]
    )
)


print()
print(
    f"Graphene carbon plane: "
    f"{carbon_plane_nm:.6f} nm"
)


# ============================================================
# CYLINDER CENTER
# ============================================================

center_y_nm = (
    0.5
    *
    ly_nm
)

center_z_nm = (
    BOTTOM_O_TARGET_NM
    +
    CYLINDER_RADIUS_NM
)


print()
print("Cylinder design:")

print(
    f"  radius:      "
    f"{CYLINDER_RADIUS_NM:.3f} nm"
)

print(
    f"  center y:    "
    f"{center_y_nm:.3f} nm"
)

print(
    f"  center z:    "
    f"{center_z_nm:.3f} nm"
)

print(
    f"  target O bottom: "
    f"{BOTTOM_O_TARGET_NM:.3f} nm"
)

print(
    f"  target O top:    "
    f"{center_z_nm + CYLINDER_RADIUS_NM:.3f} nm"
)

print(
    f"  periodic y surface gap: "
    f"{ly_nm - 2.0*CYLINDER_RADIUS_NM:.3f} nm"
)


# ============================================================
# SELECT WHOLE WATERS BY OXYGEN POSITION
# ============================================================

selected_waters = []

water_o_z = []


for water_number in range(
    EXPECTED_SOURCE_WATERS
):

    O_index = (
        WATER_START
        +
        4 * water_number
    )

    O_xyz = (
        source_positions_nm[
            O_index
        ]
    )


    # Wrap y into 0..Ly.
    y = float(
        O_xyz[1]
        %
        ly_nm
    )

    z = float(
        O_xyz[2]
    )


    dy = (
        y
        -
        center_y_nm
    )

    # minimum image in y
    dy -= (
        round(
            dy
            /
            ly_nm
        )
        *
        ly_nm
    )


    dz = (
        z
        -
        center_z_nm
    )


    r_cross = (
        dy*dy
        +
        dz*dz
    ) ** 0.5


    if (
        r_cross
        <=
        CYLINDER_RADIUS_NM
    ):

        selected_waters.append(
            water_number
        )

        water_o_z.append(
            z
        )


n_selected_waters = len(
    selected_waters
)


print()
print(
    f"Selected OPC waters: "
    f"{n_selected_waters}"
)


if not (
    MIN_EXPECTED_WATERS
    <=
    n_selected_waters
    <=
    MAX_EXPECTED_WATERS
):
    raise RuntimeError(
        "Cylinder water count is outside expected range: "
        f"{n_selected_waters}"
    )


# ============================================================
# BUILD RETAINED INDEX LIST
#
# Keep:
#
#   graphene 0..3749
#   selected whole OPC waters
#
# Remove:
#
#   ligand 3750..3819
#   all other waters
# ============================================================

kept_old_indices = list(
    range(
        N_GRAPHENE
    )
)


for water_number in selected_waters:

    base = (
        WATER_START
        +
        4 * water_number
    )

    kept_old_indices.extend(
        [
            base,
            base + 1,
            base + 2,
            base + 3,
        ]
    )


old_to_new = {
    old_index: new_index
    for new_index, old_index
    in enumerate(
        kept_old_indices
    )
}


expected_new_particles = (
    N_GRAPHENE
    +
    4 * n_selected_waters
)


if len(
    kept_old_indices
) != expected_new_particles:

    raise RuntimeError(
        "Retained-particle count mismatch."
    )


# ============================================================
# CREATE NEW SYSTEM
# ============================================================

new_system = openmm.System()


for old_index in kept_old_indices:

    new_system.addParticle(
        source_system.getParticleMass(
            old_index
        )
    )


new_system.setDefaultPeriodicBoxVectors(
    a,
    b,
    c,
)


# ============================================================
# CONSTRAINTS
# ============================================================

copied_constraints = 0


for i in range(
    source_system.getNumConstraints()
):

    p1, p2, distance = (
        source_system
        .getConstraintParameters(i)
    )

    p1 = int(p1)
    p2 = int(p2)

    if (
        p1 in old_to_new
        and
        p2 in old_to_new
    ):

        new_system.addConstraint(
            old_to_new[p1],
            old_to_new[p2],
            distance,
        )

        copied_constraints += 1


# ============================================================
# VIRTUAL SITES
# ============================================================

copied_virtual_sites = 0


for old_index in kept_old_indices:

    if not source_system.isVirtualSite(
        old_index
    ):
        continue


    site = source_system.getVirtualSite(
        old_index
    )


    if not isinstance(
        site,
        openmm.ThreeParticleAverageSite,
    ):
        raise RuntimeError(
            "Unsupported virtual-site type at "
            f"particle {old_index}"
        )


    parents = [
        int(
            site.getParticle(i)
        )
        for i in range(3)
    ]


    if not all(
        parent in old_to_new
        for parent in parents
    ):
        raise RuntimeError(
            "Retained virtual site has a removed parent."
        )


    new_site = openmm.ThreeParticleAverageSite(
        old_to_new[
            parents[0]
        ],
        old_to_new[
            parents[1]
        ],
        old_to_new[
            parents[2]
        ],
        site.getWeight(0),
        site.getWeight(1),
        site.getWeight(2),
    )


    new_system.setVirtualSite(
        old_to_new[
            old_index
        ],
        new_site,
    )

    copied_virtual_sites += 1


# ============================================================
# COPY FORCES
#
# Keep all bonded/nonbonded terms that survive subsetting.
#
# CustomExternalForce:
#   retain ONLY force group 5 = graphene support.
#
# This intentionally removes the artificial water walls.
# ============================================================

force_summary = []


for force_index in range(
    source_system.getNumForces()
):

    source_force = (
        source_system.getForce(
            force_index
        )
    )


    if isinstance(
        source_force,
        openmm.CustomBondForce,
    ):

        target = copy_custom_bond_force(
            source_force,
            old_to_new,
        )

        new_system.addForce(
            target
        )

        force_summary.append(
            {
                "old_index": force_index,
                "type": "CustomBondForce",
                "copied": True,
                "group": source_force.getForceGroup(),
                "terms": target.getNumBonds(),
            }
        )


    elif isinstance(
        source_force,
        openmm.CustomAngleForce,
    ):

        target = copy_custom_angle_force(
            source_force,
            old_to_new,
        )

        new_system.addForce(
            target
        )

        force_summary.append(
            {
                "old_index": force_index,
                "type": "CustomAngleForce",
                "copied": True,
                "group": source_force.getForceGroup(),
                "terms": target.getNumAngles(),
            }
        )


    elif isinstance(
        source_force,
        openmm.PeriodicTorsionForce,
    ):

        target = copy_periodic_torsion_force(
            source_force,
            old_to_new,
        )

        new_system.addForce(
            target
        )

        force_summary.append(
            {
                "old_index": force_index,
                "type": "PeriodicTorsionForce",
                "copied": True,
                "group": source_force.getForceGroup(),
                "terms": target.getNumTorsions(),
            }
        )


    elif isinstance(
        source_force,
        openmm.HarmonicBondForce,
    ):

        target = copy_harmonic_bond_force(
            source_force,
            old_to_new,
        )

        new_system.addForce(
            target
        )

        force_summary.append(
            {
                "old_index": force_index,
                "type": "HarmonicBondForce",
                "copied": True,
                "group": source_force.getForceGroup(),
                "terms": target.getNumBonds(),
            }
        )


    elif isinstance(
        source_force,
        openmm.HarmonicAngleForce,
    ):

        target = copy_harmonic_angle_force(
            source_force,
            old_to_new,
        )

        new_system.addForce(
            target
        )

        force_summary.append(
            {
                "old_index": force_index,
                "type": "HarmonicAngleForce",
                "copied": True,
                "group": source_force.getForceGroup(),
                "terms": target.getNumAngles(),
            }
        )


    elif isinstance(
        source_force,
        openmm.NonbondedForce,
    ):

        (
            target,
            n_exceptions,
        ) = copy_nonbonded_force(
            source_force,
            kept_old_indices,
            old_to_new,
        )

        new_system.addForce(
            target
        )

        force_summary.append(
            {
                "old_index": force_index,
                "type": "NonbondedForce",
                "copied": True,
                "group": source_force.getForceGroup(),
                "terms": n_exceptions,
            }
        )


    elif isinstance(
        source_force,
        openmm.CustomExternalForce,
    ):

        # Graphene support only.
        if (
            source_force.getForceGroup()
            ==
            5
        ):

            (
                target,
                n_particles_force,
            ) = copy_support_force(
                source_force,
                old_to_new,
            )

            new_system.addForce(
                target
            )

            force_summary.append(
                {
                    "old_index": force_index,
                    "type": "CustomExternalForce",
                    "copied": True,
                    "group": source_force.getForceGroup(),
                    "terms": n_particles_force,
                    "note": "graphene support retained",
                }
            )

        else:

            force_summary.append(
                {
                    "old_index": force_index,
                    "type": "CustomExternalForce",
                    "copied": False,
                    "group": source_force.getForceGroup(),
                    "terms": source_force.getNumParticles(),
                    "note": "non-support external force removed",
                }
            )


    elif isinstance(
        source_force,
        openmm.CMMotionRemover,
    ):

        # Not expected, but safe to reproduce if present.
        target = openmm.CMMotionRemover(
            source_force.getFrequency()
        )

        copy_force_metadata(
            source_force,
            target,
        )

        new_system.addForce(
            target
        )

        force_summary.append(
            {
                "old_index": force_index,
                "type": "CMMotionRemover",
                "copied": True,
                "group": source_force.getForceGroup(),
                "terms": 1,
            }
        )


    else:

        raise RuntimeError(
            "Unsupported force type encountered: "
            f"{type(source_force).__name__}"
        )


# ============================================================
# NEW POSITIONS
# ============================================================

new_positions_nm = (
    source_positions_nm[
        kept_old_indices,
        :
    ].copy()
)


# ============================================================
# VALIDATION
# ============================================================

new_particles = (
    new_system.getNumParticles()
)

new_virtual_sites = sum(
    1
    for i in range(
        new_particles
    )
    if new_system.isVirtualSite(i)
)


print()
print("=" * 78)
print("NEW WETTING SYSTEM")
print("=" * 78)

print(
    f"Particles:       "
    f"{new_particles}"
)

print(
    f"Graphene:        "
    f"{N_GRAPHENE}"
)

print(
    f"OPC waters:      "
    f"{n_selected_waters}"
)

print(
    f"Virtual sites:   "
    f"{new_virtual_sites}"
)

print(
    f"Constraints:     "
    f"{new_system.getNumConstraints()}"
)

print(
    f"Forces:          "
    f"{new_system.getNumForces()}"
)


if new_particles != (
    N_GRAPHENE
    +
    4*n_selected_waters
):
    raise RuntimeError(
        "New particle count mismatch."
    )


if new_virtual_sites != n_selected_waters:
    raise RuntimeError(
        "Virtual-site count does not equal water count."
    )


if new_system.getNumConstraints() != (
    3
    *
    n_selected_waters
):
    raise RuntimeError(
        "Expected exactly 3 constraints per OPC water."
    )


# ============================================================
# CHARGE
# ============================================================

new_nb = None


for i in range(
    new_system.getNumForces()
):

    force = new_system.getForce(i)

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):
        new_nb = force
        break


if new_nb is None:
    raise RuntimeError(
        "New NonbondedForce not found."
    )


total_charge_e = 0.0


for i in range(
    new_particles
):

    q, sigma, epsilon = (
        new_nb.getParticleParameters(i)
    )

    total_charge_e += float(
        q.value_in_unit(
            unit.elementary_charge
        )
    )


print(
    f"Total charge:    "
    f"{total_charge_e:+.10e} e"
)


if abs(
    total_charge_e
) > 1.0e-5:
    raise RuntimeError(
        "New wetting system is not neutral."
    )


# ============================================================
# SUPPORT / WALL CHECK
# ============================================================

support_particle_count = 0
other_external_forces = 0


for i in range(
    new_system.getNumForces()
):

    force = new_system.getForce(i)

    if isinstance(
        force,
        openmm.CustomExternalForce,
    ):

        if force.getForceGroup() == 5:

            support_particle_count += (
                force.getNumParticles()
            )

        else:

            other_external_forces += 1


print(
    f"Support particles: "
    f"{support_particle_count}"
)

print(
    f"Other external forces: "
    f"{other_external_forces}"
)


if support_particle_count != 1250:
    raise RuntimeError(
        "Expected graphene support on exactly 1250 carbon cores."
    )


if other_external_forces != 0:
    raise RuntimeError(
        "Water-wall force survived unexpectedly."
    )


# ============================================================
# OXYGEN EXTENTS AFTER SUBSETTING
# ============================================================

new_water_start = N_GRAPHENE

oxygen_indices = np.arange(
    new_water_start,
    new_particles,
    4,
    dtype=int,
)


O_positions = (
    new_positions_nm[
        oxygen_indices
    ]
)


O_z_min = float(
    O_positions[:, 2].min()
)

O_z_max = float(
    O_positions[:, 2].max()
)


print()
print(
    f"Water O z range: "
    f"{O_z_min:.6f} -> "
    f"{O_z_max:.6f} nm"
)


# ============================================================
# FORCE SUMMARY
# ============================================================

print()
print("=" * 78)
print("FORCE COPY SUMMARY")
print("=" * 78)


for entry in force_summary:

    print(
        f"old force {entry['old_index']:2d}  "
        f"{entry['type']:22s}  "
        f"group={entry['group']:2d}  "
        f"copied={entry['copied']}  "
        f"terms={entry['terms']}"
    )


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
        "Dedicated bare-graphene OPC cylindrical "
        "wetting benchmark build."
    ),

    "source_system_xml": str(
        SOURCE_XML
    ),

    "source_positions": str(
        SOURCE_POSITIONS
    ),

    "output_system_xml": str(
        OUTPUT_XML
    ),

    "output_positions": str(
        OUTPUT_POSITIONS
    ),

    "graphene_particles": (
        N_GRAPHENE
    ),

    "water_count": (
        n_selected_waters
    ),

    "total_particles": (
        new_particles
    ),

    "virtual_sites": (
        new_virtual_sites
    ),

    "constraints": (
        new_system.getNumConstraints()
    ),

    "total_charge_e": (
        total_charge_e
    ),

    "carbon_plane_nm": (
        carbon_plane_nm
    ),

    "cylinder": {
        "axis": "periodic x",
        "radius_nm": CYLINDER_RADIUS_NM,
        "center_y_nm": center_y_nm,
        "center_z_nm": center_z_nm,
        "target_bottom_O_nm": BOTTOM_O_TARGET_NM,
        "periodic_y_surface_gap_nm": (
            ly_nm
            -
            2.0*CYLINDER_RADIUS_NM
        ),
    },

    "water_O_z_range_nm": [
        O_z_min,
        O_z_max,
    ],

    "graphene_support_retained": True,

    "water_wall_retained": False,

    "ligand_retained": False,

    "force_summary": force_summary,

    "note": (
        "Build only. No minimization or dynamics performed."
    ),
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        metadata,
        handle,
        indent=2,
    )


print()
print("=" * 78)
print("OUTPUTS")
print("=" * 78)

print(
    f"System XML:"
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
    "NO MD HAS BEEN RUN."
)

print()
print(
    "GRAPHENE_OPC_WETTING_CYLINDER_BUILD_PASS"
)

print("=" * 78)
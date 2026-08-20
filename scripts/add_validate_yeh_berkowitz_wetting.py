from pathlib import Path
import json
import math
import copy

import numpy as np

import openmm
from openmm import unit


# ============================================================
# PURPOSE
#
# Add a dynamic Yeh-Berkowitz slab correction to the
# graphene/OPC wetting System and independently verify:
#
#   Ucorr = 2*pi*k_e*Mz^2 / V
#
#   Fz_i = -4*pi*k_e*q_i*Mz / V
#
# BEFORE any minimization or dynamics.
# ============================================================


ROOT = Path(
    r"C:\Users\saisa\GFET Simulation\graphene-functionalization-md"
)

INPUT_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder.xml"
)

POSITIONS_NPY = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_positions_nm.npy"
)

OUTPUT_XML = (
    ROOT
    / "parameters"
    / "combined"
    / "graphene_opc_wetting_cylinder_yb.xml"
)

OUTPUT_JSON = (
    ROOT
    / "analysis"
    / "graphene_opc_wetting_yb_validation.json"
)


K_E = 138.935456
YB_FORCE_GROUP = 29


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


def charge_e(q):

    return float(
        q.value_in_unit(
            unit.elementary_charge
        )
    )


# ============================================================
# LOAD
# ============================================================

print()
print("=" * 80)
print("DYNAMIC YEH-BERKOWITZ CORRECTION VALIDATION")
print("=" * 80)


system = openmm.XmlSerializer.deserialize(
    INPUT_XML.read_text(
        encoding="utf-8"
    )
)

positions_nm = np.load(
    POSITIONS_NPY
)


n_particles = system.getNumParticles()


if positions_nm.shape != (
    n_particles,
    3,
):
    raise RuntimeError(
        "Position shape does not match System."
    )


# ============================================================
# FIND NONBONDED FORCE / CHARGES
# ============================================================

nb = None


for i in range(
    system.getNumForces()
):

    force = system.getForce(i)

    if isinstance(
        force,
        openmm.NonbondedForce,
    ):

        if nb is not None:
            raise RuntimeError(
                "Multiple NonbondedForce objects found."
            )

        nb = force


if nb is None:
    raise RuntimeError(
        "NonbondedForce not found."
    )


charges = np.zeros(
    n_particles,
    dtype=float,
)


for i in range(
    n_particles
):

    q, sigma, epsilon = (
        nb.getParticleParameters(i)
    )

    charges[i] = charge_e(q)


total_charge = float(
    np.sum(charges)
)


print()
print(
    f"Particles:     {n_particles}"
)

print(
    f"Total charge:  {total_charge:+.12e} e"
)


if abs(total_charge) > 1.0e-6:

    raise RuntimeError(
        "System is not sufficiently neutral."
    )


# ============================================================
# BOX / VOLUME
# ============================================================

a_raw, b_raw, c_raw = (
    system.getDefaultPeriodicBoxVectors()
)

a = vec_nm(a_raw)
b = vec_nm(b_raw)
c = vec_nm(c_raw)


volume_nm3 = float(
    abs(
        np.dot(
            a,
            np.cross(
                b,
                c,
            )
        )
    )
)


print(
    f"Volume:        {volume_nm3:.9f} nm^3"
)


# ============================================================
# VIRTUAL-SITE-AWARE ANALYTICAL GEOMETRY
#
# The saved coordinate array contains positions for every
# particle, including OPC M sites.  For the analytical
# reference we reconstruct every virtual site exactly from
# its defining parent coordinates.
#
# This makes the analytical Mz consistent with the actual
# constrained OPC geometry evaluated by OpenMM.
# ============================================================

realized_positions_nm = positions_nm.copy()

virtual_site_records = []


for v in range(
    n_particles
):

    if not system.isVirtualSite(v):
        continue


    site = system.getVirtualSite(v)


    if not isinstance(
        site,
        openmm.ThreeParticleAverageSite,
    ):

        raise RuntimeError(
            "Unsupported virtual-site type at "
            f"particle {v}: "
            f"{type(site).__name__}"
        )


    parents = [
        int(
            site.getParticle(i)
        )
        for i in range(3)
    ]


    weights = [
        float(
            site.getWeight(i)
        )
        for i in range(3)
    ]


    realized_positions_nm[
        v,
        :
    ] = (
        weights[0]
        *
        realized_positions_nm[
            parents[0],
            :
        ]
        +
        weights[1]
        *
        realized_positions_nm[
            parents[1],
            :
        ]
        +
        weights[2]
        *
        realized_positions_nm[
            parents[2],
            :
        ]
    )


    virtual_site_records.append(
        (
            v,
            parents,
            weights,
        )
    )


print()
print(
    f"Virtual sites reconstructed analytically: "
    f"{len(virtual_site_records)}"
)


# ============================================================
# ANALYTICAL REFERENCE
# ============================================================

Mz = float(
    np.sum(
        charges
        *
        realized_positions_nm[:, 2]
    )
)


coefficient = (
    2.0
    *
    math.pi
    *
    K_E
    /
    volume_nm3
)


expected_energy = (
    coefficient
    *
    Mz
    *
    Mz
)


# ------------------------------------------------------------
# First compute the formal force associated with each charge
# coordinate before applying virtual-site geometry.
# ------------------------------------------------------------

expected_forces_z_direct = (
    -4.0
    *
    math.pi
    *
    K_E
    *
    charges
    *
    Mz
    /
    volume_nm3
)


# ------------------------------------------------------------
# Convert virtual-site forces into forces on their defining
# massive parent atoms using the chain rule.
#
# For
#
#   r_v = w1*r1 + w2*r2 + w3*r3
#
# a force F_v contributes
#
#   w1*F_v, w2*F_v, w3*F_v
#
# to the three parent particles.
# ------------------------------------------------------------

expected_forces_z = (
    expected_forces_z_direct.copy()
)


for (
    virtual_index,
    parents,
    weights,
) in virtual_site_records:

    virtual_force = float(
        expected_forces_z_direct[
            virtual_index
        ]
    )


    # Virtual site itself is not an independent dynamical
    # degree of freedom.
    expected_forces_z[
        virtual_index
    ] = 0.0


    for (
        parent,
        weight,
    ) in zip(
        parents,
        weights,
    ):

        expected_forces_z[
            parent
        ] += (
            weight
            *
            virtual_force
        )


print()
print("Analytical reference:")

print(
    f"  Mz               = "
    f"{Mz:+.9f} e nm"
)

print(
    f"  coefficient      = "
    f"{coefficient:.12f}"
)

print(
    f"  YB energy        = "
    f"{expected_energy:.9f} kJ/mol"
)

print(
    f"  max |YB Fz|      = "
    f"{np.max(np.abs(expected_forces_z)):.9f} "
    f"kJ/mol/nm"
)


# ============================================================
# BUILD Mz COLLECTIVE VARIABLE
#
# CustomExternalForce energies are summed over particles.
#
# We use the numerical value q*z as the CV:
#
#       Mz = sum_i q_i*z_i
#
# ============================================================

mz_force = openmm.CustomExternalForce(
    "q*z"
)

mz_force.addPerParticleParameter(
    "q"
)


charged_particle_count = 0


for i in range(
    n_particles
):

    q = float(
        charges[i]
    )

    if abs(q) > 1.0e-14:

        mz_force.addParticle(
            i,
            [
                q
            ],
        )

        charged_particle_count += 1


# ============================================================
# CUSTOM CV FORCE
#
# U = coefficient * Mz^2
# ============================================================

yb = openmm.CustomCVForce(
    "yb_coeff*Mz^2"
)

yb.addCollectiveVariable(
    "Mz",
    mz_force,
)

yb.addGlobalParameter(
    "yb_coeff",
    coefficient,
)

yb.setForceGroup(
    YB_FORCE_GROUP
)

try:
    yb.setName(
        "Yeh-Berkowitz slab correction"
    )
except Exception:
    pass


system.addForce(
    yb
)


print()
print(
    f"Charged particles in Mz CV: "
    f"{charged_particle_count}"
)

print(
    f"YB force group:              "
    f"{YB_FORCE_GROUP}"
)


# ============================================================
# CREATE CONTEXT
# ============================================================

try:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CUDA"
        )
    )

    properties = {
        "Precision": "mixed"
    }

except Exception:

    platform = (
        openmm.Platform
        .getPlatformByName(
            "CPU"
        )
    )

    properties = {}


integrator = openmm.VerletIntegrator(
    0.001
    *
    unit.picoseconds
)


context = openmm.Context(
    system,
    integrator,
    platform,
    properties,
)


context.setPositions(
    positions_nm
    *
    unit.nanometer
)

context.computeVirtualSites()


# ============================================================
# GET YB FORCE GROUP ONLY
# ============================================================

state_yb = context.getState(
    getEnergy=True,
    getForces=True,
    groups=(
        1 << YB_FORCE_GROUP
    ),
)


actual_energy = float(
    state_yb
    .getPotentialEnergy()
    .value_in_unit(
        unit.kilojoule_per_mole
    )
)


actual_forces = np.asarray(
    state_yb
    .getForces(
        asNumpy=True
    )
    .value_in_unit(
        unit.kilojoule_per_mole
        /
        unit.nanometer
    ),
    dtype=float,
)


# ============================================================
# COMPARISON
# ============================================================

energy_error = (
    actual_energy
    -
    expected_energy
)


# ------------------------------------------------------------
# OpenMM distributes virtual-site forces onto the massive
# parent atoms for dynamics.
#
# The returned force array can still contain the formal force
# value in the virtual-site slot itself.  Those massless slots
# are not independent dynamical degrees of freedom.
#
# Therefore validate:
#
#   1. massive-particle forces against the chain-rule result
#   2. virtual-site slots against their formal direct force
#
# separately.
# ------------------------------------------------------------

virtual_mask = np.array(
    [
        system.isVirtualSite(i)
        for i in range(n_particles)
    ],
    dtype=bool,
)

massive_mask = ~virtual_mask


massive_force_error_z = (
    actual_forces[
        massive_mask,
        2,
    ]
    -
    expected_forces_z[
        massive_mask
    ]
)


max_force_error = float(
    np.max(
        np.abs(
            massive_force_error_z
        )
    )
)


rms_force_error = float(
    np.sqrt(
        np.mean(
            massive_force_error_z**2
        )
    )
)


virtual_slot_error_z = (
    actual_forces[
        virtual_mask,
        2,
    ]
    -
    expected_forces_z_direct[
        virtual_mask
    ]
)


max_virtual_slot_error = float(
    np.max(
        np.abs(
            virtual_slot_error_z
        )
    )
)


rms_virtual_slot_error = float(
    np.sqrt(
        np.mean(
            virtual_slot_error_z**2
        )
    )
)


max_xy_force = float(
    np.max(
        np.abs(
            actual_forces[
                :,
                :2
            ]
        )
    )
)


print()
print("=" * 80)
print("CUSTOM FORCE VS ANALYTICAL FORMULA")
print("=" * 80)

print(
    f"Expected YB energy: "
    f"{expected_energy:.12f} kJ/mol"
)

print(
    f"Actual YB energy:   "
    f"{actual_energy:.12f} kJ/mol"
)

print(
    f"Energy error:       "
    f"{energy_error:+.12e} kJ/mol"
)

print()

print(
    f"Max massive Fz error: "
    f"{max_force_error:.12e} kJ/mol/nm"
)

print(
    f"RMS massive Fz error: "
    f"{rms_force_error:.12e} kJ/mol/nm"
)

print(
    f"Max virtual-slot error: "
    f"{max_virtual_slot_error:.12e} kJ/mol/nm"
)

print(
    f"RMS virtual-slot error: "
    f"{rms_virtual_slot_error:.12e} kJ/mol/nm"
)

print(
    f"Max |Fx/Fy|:          "
    f"{max_xy_force:.12e} kJ/mol/nm"
)


# ============================================================
# STRICT VALIDATION
# ============================================================

ENERGY_TOL = 1.0e-3
FORCE_TOL = 1.0e-3


if abs(
    energy_error
) > ENERGY_TOL:

    raise RuntimeError(
        "Dynamic YB energy does not match "
        "analytical expression."
    )


if max_force_error > FORCE_TOL:

    raise RuntimeError(
        "Dynamic YB force on massive particles does not "
        "match the analytical chain-rule expression."
    )


if max_virtual_slot_error > FORCE_TOL:

    raise RuntimeError(
        "Reported virtual-site force slots do not match "
        "the formal direct virtual-site force."
    )


if max_xy_force > FORCE_TOL:

    raise RuntimeError(
        "YB correction produced unexpected xy force."
    )


# ============================================================
# SAVE CORRECTED SYSTEM
# ============================================================

OUTPUT_XML.write_text(
    openmm.XmlSerializer.serialize(
        system
    ),
    encoding="utf-8",
)


results = {
    "method": (
        "Dynamic Yeh-Berkowitz correction implemented "
        "with CustomCVForce."
    ),

    "formula_energy": (
        "U = 2*pi*k_e*Mz^2/V"
    ),

    "formula_force": (
        "Fz_i = -4*pi*k_e*q_i*Mz/V"
    ),

    "particles": n_particles,

    "charged_particles_in_cv": (
        charged_particle_count
    ),

    "total_charge_e": total_charge,

    "volume_nm3": volume_nm3,

    "Mz_e_nm": Mz,

    "coefficient": coefficient,

    "expected_energy_kJ_mol": (
        expected_energy
    ),

    "actual_energy_kJ_mol": (
        actual_energy
    ),

    "energy_error_kJ_mol": (
        energy_error
    ),

    "max_force_error_kJ_mol_nm": (
        max_force_error
    ),

    "rms_force_error_kJ_mol_nm": (
        rms_force_error
    ),

    "max_xy_force_kJ_mol_nm": (
        max_xy_force
    ),

    "yb_force_group": (
        YB_FORCE_GROUP
    ),

    "platform": platform.getName(),

    "output_system_xml": str(
        OUTPUT_XML
    ),
}


OUTPUT_JSON.write_text(
    json.dumps(
        results,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 80)
print("OUTPUTS")
print("=" * 80)

print(
    f"Corrected System:"
    f"\n  {OUTPUT_XML}"
)

print(
    f"Validation JSON:"
    f"\n  {OUTPUT_JSON}"
)

print()
print(
    "No minimization or MD has been run."
)

print()
print(
    "DYNAMIC_YEH_BERKOWITZ_VALIDATION_PASS"
)

print("=" * 80)
from pathlib import Path
import numpy as np

import openmm
from openmm import unit


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


def vec_nm(v):
    return np.array(
        [
            v[0].value_in_unit(unit.nanometer),
            v[1].value_in_unit(unit.nanometer),
            v[2].value_in_unit(unit.nanometer),
        ],
        dtype=float,
    )


def inspect_system(label, path):

    print()
    print("=" * 80)
    print(label)
    print("=" * 80)

    system = openmm.XmlSerializer.deserialize(
        path.read_text(encoding="utf-8")
    )

    print(f"XML:           {path}")
    print(f"Particles:     {system.getNumParticles()}")
    print(f"Constraints:   {system.getNumConstraints()}")

    n_virtual = sum(
        system.isVirtualSite(i)
        for i in range(system.getNumParticles())
    )

    print(f"Virtual sites: {n_virtual}")

    a, b, c = system.getDefaultPeriodicBoxVectors()

    print()
    print("Box vectors (nm):")
    print("  a =", vec_nm(a))
    print("  b =", vec_nm(b))
    print("  c =", vec_nm(c))

    print()
    print("FORCES")
    print("------")

    for i in range(system.getNumForces()):

        force = system.getForce(i)

        print(
            f"[{i}] "
            f"{type(force).__name__}  "
            f"group={force.getForceGroup()}  "
            f"name='{force.getName()}'"
        )

        if isinstance(force, openmm.NonbondedForce):

            print(
                f"     particles  = "
                f"{force.getNumParticles()}"
            )

            print(
                f"     exceptions = "
                f"{force.getNumExceptions()}"
            )

            print(
                f"     method     = "
                f"{force.getNonbondedMethod()}"
            )

            print(
                f"     cutoff     = "
                f"{force.getCutoffDistance()}"
            )

            print(
                f"     ewald tol  = "
                f"{force.getEwaldErrorTolerance()}"
            )

            print(
                f"     dispersion = "
                f"{force.getUseDispersionCorrection()}"
            )

        if isinstance(force, openmm.CustomExternalForce):

            print(
                f"     terms      = "
                f"{force.getNumParticles()}"
            )

            print(
                f"     energy     = "
                f"{force.getEnergyFunction()}"
            )

            print(
                f"     globals    = "
                f"{force.getNumGlobalParameters()}"
            )

            for j in range(
                force.getNumGlobalParameters()
            ):

                print(
                    f"       "
                    f"{force.getGlobalParameterName(j)} = "
                    f"{force.getGlobalParameterDefaultValue(j)}"
                )

            print(
                f"     per-particle params = "
                f"{force.getNumPerParticleParameters()}"
            )

    return system


graphene_system = inspect_system(
    "25x50 GRAPHENE DONOR",
    GRAPHENE_XML,
)

wetting_system = inspect_system(
    "1642-WATER WETTING DONOR",
    WETTING_XML,
)


positions = np.load(
    WETTING_POSITIONS
)


print()
print("=" * 80)
print("DONOR CONSISTENCY")
print("=" * 80)

print(
    f"Wetting positions shape: "
    f"{positions.shape}"
)

print(
    f"Wetting XML particles:   "
    f"{wetting_system.getNumParticles()}"
)


if positions.shape != (
    wetting_system.getNumParticles(),
    3,
):
    raise RuntimeError(
        "Wetting position count mismatch."
    )


expected_old_wetting_particles = (
    3750
    +
    4 * 1642
)


print(
    f"Expected old wetting particles: "
    f"{expected_old_wetting_particles}"
)


if (
    wetting_system.getNumParticles()
    !=
    expected_old_wetting_particles
):
    raise RuntimeError(
        "Unexpected 1642-water donor size."
    )


if graphene_system.getNumParticles() != 7500:

    raise RuntimeError(
        "25x50 graphene donor is not 7500 particles."
    )


print()
print("HYBRID DONOR INSPECTION_PASS")
print("=" * 80)
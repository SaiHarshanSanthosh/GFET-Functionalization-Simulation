from pathlib import Path
import numpy as np

from openmm import app, openmm, unit


BOX_FILE = Path(
    "parameters/combined/"
    "pyrene_peg5_graphene_periodic_box_vectors_nm.npy"
)


def main():

    box = np.load(BOX_FILE)

    print("Loaded box vectors:")
    print(box)
    print()

    # --------------------------------------------------------
    # Empty topology: for this test we are generating ONLY
    # water.  Graphene and ligand are not being touched yet.
    # --------------------------------------------------------

    topology = app.Topology()

    a = openmm.Vec3(*box[0])
    b = openmm.Vec3(*box[1])
    c = openmm.Vec3(*box[2])

    box_vectors = (
        a * unit.nanometer,
        b * unit.nanometer,
        c * unit.nanometer
    )

    topology.setPeriodicBoxVectors(
        box_vectors
    )

    modeller = app.Modeller(
        topology,
        []
    )

    # OPC parameters.
    forcefield = app.ForceField(
        "amber19/opc.xml"
    )

    # OpenMM supplies a pre-equilibrated TIP4P-Ew coordinate
    # box.  OPC is also a four-site water model, so this is
    # used only to generate starting water geometry.
    modeller.addSolvent(
        forcefield,
        model="tip4pew",
        boxVectors=box_vectors,
        neutralize=False,
        ionicStrength=0.0 * unit.molar
    )

    residues = list(
        modeller.topology.residues()
    )

    atoms = list(
        modeller.topology.atoms()
    )

    n_waters = len(residues)
    n_atoms = len(atoms)

    print("WATER BOX")
    print("---------")
    print("Water molecules:", n_waters)
    print("Topology particles:", n_atoms)

    if n_waters == 0:
        raise RuntimeError(
            "No waters were generated."
        )

    atoms_per_water = (
        n_atoms / n_waters
    )

    print(
        "Particles per water:",
        f"{atoms_per_water:.3f}"
    )

    # OPC should be a four-site model.
    if abs(atoms_per_water - 4.0) > 1e-8:
        raise RuntimeError(
            "Expected four particles per OPC water."
        )

    # --------------------------------------------------------
    # Ask OPC to parameterize the water topology.
    # --------------------------------------------------------

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=False
    )

    print()
    print("OPC SYSTEM")
    print("----------")
    print(
        "System particles:",
        system.getNumParticles()
    )

    print(
        "Constraints:",
        system.getNumConstraints()
    )

    # --------------------------------------------------------
    # Locate nonbonded force and verify total charge.
    # --------------------------------------------------------

    nb_forces = [
        force
        for force in system.getForces()
        if isinstance(
            force,
            openmm.NonbondedForce
        )
    ]

    if len(nb_forces) != 1:
        raise RuntimeError(
            "Expected exactly one NonbondedForce."
        )

    nb = nb_forces[0]

    total_charge = 0.0

    for i in range(
        nb.getNumParticles()
    ):

        q, _, _ = (
            nb.getParticleParameters(i)
        )

        total_charge += (
            q.value_in_unit(
                unit.elementary_charge
            )
        )

    print(
        "Total water-box charge:",
        f"{total_charge:+.12f} e"
    )

    if abs(total_charge) > 1e-6:
        raise RuntimeError(
            "Pure water box is not neutral."
        )

    # --------------------------------------------------------
    # OPC has one massless charge site per water.
    # --------------------------------------------------------

    massless = 0

    for i in range(
        system.getNumParticles()
    ):

        mass = (
            system.getParticleMass(i)
            .value_in_unit(
                unit.dalton
            )
        )

        if abs(mass) < 1e-12:
            massless += 1

    print(
        "Massless OPC sites:",
        massless
    )

    if massless != n_waters:
        raise RuntimeError(
            "Expected one OPC virtual site "
            "per water molecule."
        )

    print()
    print(
        "OPC WATER GENERATION: PASS"
    )

    print()
    print(
        "Nothing from the graphene/ligand "
        "system was modified."
    )


if __name__ == "__main__":
    main()

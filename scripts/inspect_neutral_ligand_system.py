from openmm import app, openmm

PRMTOP = "parameters/gaff2/pyrene_peg5_neutral.prmtop"

prmtop = app.AmberPrmtopFile(PRMTOP)

system = prmtop.createSystem(
    nonbondedMethod=app.NoCutoff,
    constraints=None,
    removeCMMotion=False
)

print("Particles:", system.getNumParticles())
print("Constraints:", system.getNumConstraints())
print("Forces:", system.getNumForces())
print()

for i, force in enumerate(system.getForces()):

    name = force.__class__.__name__

    print(f"Force {i}: {name}")

    if isinstance(force, openmm.HarmonicBondForce):
        print("  bonds:", force.getNumBonds())

    elif isinstance(force, openmm.HarmonicAngleForce):
        print("  angles:", force.getNumAngles())

    elif isinstance(force, openmm.PeriodicTorsionForce):
        print("  torsions:", force.getNumTorsions())

    elif isinstance(force, openmm.NonbondedForce):
        print("  particles:", force.getNumParticles())
        print("  exceptions:", force.getNumExceptions())
        print("  method:", force.getNonbondedMethod())

    else:
        print("  *** unexpected force type ***")

print()
print("Inspection complete.")

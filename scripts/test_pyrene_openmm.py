from openmm import app, unit, openmm

# Load the AMBER files we just created
prmtop = app.AmberPrmtopFile(
    "parameters/gaff2/pyrene_peg5.prmtop"
)

inpcrd = app.AmberInpcrdFile(
    "parameters/gaff2/pyrene_peg5.inpcrd"
)

# Build the OpenMM system
system = prmtop.createSystem(
    nonbondedMethod=app.NoCutoff,
    constraints=None
)

# OpenMM requires an integrator even though we're only minimizing
integrator = openmm.LangevinMiddleIntegrator(
    300 * unit.kelvin,
    1.0 / unit.picosecond,
    0.001 * unit.picoseconds
)

# Explicitly use your NVIDIA GPU
platform = openmm.Platform.getPlatformByName("CUDA")

simulation = app.Simulation(
    prmtop.topology,
    system,
    integrator,
    platform
)

simulation.context.setPositions(inpcrd.positions)

# Energy before minimization
state = simulation.context.getState(getEnergy=True)
initial_energy = state.getPotentialEnergy()

print("Initial potential energy:", initial_energy)

# Relax the molecule
print("Minimizing...")
simulation.minimizeEnergy(maxIterations=500)

# Energy after minimization
state = simulation.context.getState(
    getEnergy=True,
    getPositions=True
)

final_energy = state.getPotentialEnergy()

print("Final potential energy:", final_energy)
print("CUDA platform:", platform.getName())

# Save the minimized structure so we can look at it
with open("structures/pyrene_peg5_minimized.pdb", "w") as f:
    app.PDBFile.writeFile(
        prmtop.topology,
        state.getPositions(),
        f
    )

print("Saved structures/pyrene_peg5_minimized.pdb")

from openmm import openmm

system = openmm.System()
system.addParticle(1.0)

integrator = openmm.VerletIntegrator(0.001)

platform = openmm.Platform.getPlatformByName("CUDA")

context = openmm.Context(
    system,
    integrator,
    platform,
    {"Precision": "mixed"},
)

print(
    "CUDA smoke test: PASS | "
    f"platform={platform.getName()}"
)

del context
del integrator

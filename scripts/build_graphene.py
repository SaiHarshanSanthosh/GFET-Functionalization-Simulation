from ase.build import graphene
from ase.io import write

# pristine graphene sheet, 25 x 25 roughly 6 nm across

sheet = graphene(
    formula="C2",
    a=2.46,
    size=(25, 25, 1),
    vacuum=20.0
)

write("structures/graphene.pdb", sheet)
write("structures/graphene.xyz", sheet)

print("Number of carbon atoms:", len(sheet))
print("\nSimulation cell:")
print(sheet.cell)

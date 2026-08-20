from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from pathlib import Path

# Exact Pyrene-PEG5-propargyl structure
smiles = "O=C(C1=C(C2=C34)C=CC4=CC=CC3=CC=C2C=C1)NCCOCCOCCOCCOCCOCC#C"

# Convert SMILES into a molecule
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    raise ValueError("RDKit could not understand the SMILES string.")

# Add the hydrogens explicitly
mol = Chem.AddHs(mol)

# Generate a reasonable 3D starting shape
params = AllChem.ETKDGv3()
params.randomSeed = 42

result = AllChem.EmbedMolecule(mol, params)

if result != 0:
    raise RuntimeError("3D structure generation failed.")

# Light geometry cleanup
if AllChem.MMFFHasAllMoleculeParams(mol):
    AllChem.MMFFOptimizeMolecule(mol)
else:
    AllChem.UFFOptimizeMolecule(mol)

# Make sure our output folder exists
Path("structures").mkdir(exist_ok=True)

# SDF preserves chemical bonding information better
Chem.MolToMolFile(mol, "structures/pyrene_peg5_propargyl.sdf")

# PDB is convenient for molecular viewers
Chem.MolToPDBFile(mol, "structures/pyrene_peg5_propargyl.pdb")

print("Formula:", rdMolDescriptors.CalcMolFormula(mol))
print("Molecular weight:", round(Descriptors.MolWt(mol), 2))
print("Number of atoms including H:", mol.GetNumAtoms())
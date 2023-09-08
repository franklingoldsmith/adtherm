from geometry_generation.adtherm import AdTherm
from ase.io import read
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import BFGS

atoms = read('POSCAR_10')
del atoms.constraints
c = FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Ni'])
atoms.set_constraint(c)
calc = EMT()
atoms.calc = calc
opt = BFGS(atoms)
opt.run(fmax=0.05)
indices = [36, 37, 38, 39]
dyn = AdTherm(atoms, indices, calc, n_gauss=5, n_sobol=5, n_random=5, relax_internals=False, bootstrap=False)
dyn.run()

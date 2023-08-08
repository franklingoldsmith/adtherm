###  Import some stuff ###




#########################



class Adtherm:

	def __init__(atoms,indeces,n_gauss=5000,n_sobol=0,n_random=0,relax_internals=False):

		self.N_atoms=len(atoms)
		self.N_gauss=n_gauss
		self.N_sobol=n_sobol
		self.N_random=n_random

	def run():
		geometry_generate(atoms,indices,'hessian')
		for img in images:
			img.get_potential_energy()

	







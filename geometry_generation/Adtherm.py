###  Import some stuff ###
import os
import sys
import system_train as syst
import numpy as np
import sobol_seq
from sobol_seq import i4_sobol, i4_sobol_generate
#########################


class Adtherm:

	def __init__(atoms,indices,n_gauss=5000,n_sobol=0,n_random=0,relax_internals=False):
        
        self.atoms=atoms
        self.adsorbate=self.atoms[indices].copy()
		self.N_atoms=len(atoms)
		self.N_gauss=n_gauss
		self.N_sobol=n_sobol
		self.N_random=n_random

    def geometry_generate(method,hessian=None)
        ads=self.adsorbate



    def run():
		H_list=geometry_generate('hessian')
		for img in H_list:
			img.get_potential_energy()
        H=calculate_H(H_list)
        np.savetxt(H,'Rigid_H.txt')

        gauss_list=geometry_generate('gauss',hessian=H)
	







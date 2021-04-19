#!/usr/bin/env python

import numpy as np
import generate_random_positions
import system as syst
import os, sys
import time

path = os.getcwd()
#sys.path.append('path' + "',")
#sys.path.insert(1, 'path')
print(sys.path)
import integration as integ

print("have defined the required routines, work starting")

f = open("output.txt", "w")
f.write("Run starts at " + time.strftime("%m/%d/%Y, %H:%M:%S") + "\n")
f.close()

filename = "input.inp"
adsorbate = integ.Adsorbate()
integ.parse_input_file(filename,adsorbate)

#Initialize system coordinates
#path = os.getcwd()
##================= PREPARE WORK ===================##
#open the two trajectories, where surf is the bare metal surface (periodic slab) and ads is the adsorbate.
surf = syst.Trajectory()
syst.get_system_coordinates(surf, path+'/' + adsorbate.surface + '.inp', translate=False)

ads = syst.Trajectory()
syst.get_system_coordinates(ads, path + '/' + adsorbate.name + '.inp')

ads.rotate=False #Rotation are not enabled at this time, but are to be added soon

METHOD = 'random'
#z direction range (relative to COM z coordinate at optimized position on surface)
ads.z_low = adsorbate.z_low+ads.COM[2]
ads.z_high = adsorbate.z_high+ads.COM[2]

#Boundaries for each d.o.f. here (currently assumes a 3x3 unit cell, 111 facet)
#ADD ROTATIONAL BOUNDARIES HERE 
ads.gamma_boundaries = [0,0]
ads.alpha_boundaries = [0,0]
ads.beta_boundaries = [0,0]
ads.x_boundaries = [0.0,adsorbate.uc_x/3.0]
ads.y_boundaries = [0.0,adsorbate.uc_y/3.0]
ads.z_boundaries = [ads.z_low,ads.z_high]

T_array = np.arange(adsorbate.T_low, adsorbate.T_high+adsorbate.dT-1, adsorbate.dT)

f = open("output.txt", "a")
f.write("convergence parameter: " + str(adsorbate.q_accuracy) + "\n")
f.write("Temperature array in K: \n")
f.write(str(T_array) + '\n')
f.close()

f = open("output.txt", "a")
f.write("Integration routine starts at " + time.strftime("%m/%d/%Y, %H:%M:%S") + "\n")
f.close()

integ.q_MCsampling(adsorbate,surf,ads,METHOD,T_array)     

f = open("output.txt", "a")
f.write("Integration routine finishes at " + time.strftime("%m/%d/%Y, %H:%M:%S") + "\n")
f.close()

MC_array=(adsorbate.q_MC) 

PG_3D_corr_array = []
Analytical_array_2Dgas = []
q_quantum = []
q_class = []
e_quantum = []
e_class = []
e_quantum_3vals = []
square_e_quantum = []
square_e_class = []
e_class_3vals = []
e_squared_quantum_array = []
e_squared_classical_array = []

for i, Temp in enumerate(T_array): 
    PG_3D_corr_array.append(integ.PG_corr_3D(adsorbate,Temp))
    Analytical_array_2Dgas.append(integ.q_2Dgas_zHO(adsorbate,Temp,ads)) 
    q_quantum.append(np.prod(integ.PG_transfer_direct(adsorbate, Temp)[0]))
    q_class.append(np.prod(integ.PG_transfer_direct(adsorbate,Temp)[1]))
    e_quantum.append(integ.PG_transfer_direct(adsorbate,Temp)[2])
    e_class.append(integ.PG_transfer_direct(adsorbate,Temp)[3])
    e_quantum_3vals.append(integ.PG_transfer_direct(adsorbate,Temp)[4])
    e_class_3vals.append(integ.PG_transfer_direct(adsorbate,Temp)[5])
    square_e_quantum.append(np.sum(e_quantum_3vals[-1]**2.0))
    square_e_class.append(np.sum(e_class_3vals[-1]**2.0))
    e_squared_quantum_array.append(integ.PG_squared_direct(adsorbate,Temp)[0])
    e_squared_classical_array.append(integ.PG_squared_direct(adsorbate,Temp)[1])

PG_3D_corr_array = np.array(PG_3D_corr_array)
Analytical_array_2Dgas = np.array(Analytical_array_2Dgas)
e_quantum = np.array(e_quantum)
q_quantum=np.array(q_quantum) 
e_class = np.array(e_class)
e_quantum_3vals = np.array(e_quantum_3vals)
e_class_3vals = np.array(e_class_3vals)
e_squared_quantum_array = np.array(e_squared_quantum_array)
e_squared_classical_array = np.array(e_squared_classical_array)

int_E_class = 3.0/2.0 + e_class
int_E_array = (adsorbate.I1 + 3.0/2.0 + e_quantum - int_E_class)

Cp_array_quantum = e_squared_quantum_array - square_e_quantum
Cp_array_classical = 3.0/2.0 + e_squared_classical_array - square_e_class
Cp_array = (3.0/2.0 + adsorbate.I2 - adsorbate.I1**2.0) + Cp_array_quantum - Cp_array_classical

S_array = int_E_array+np.array(np.log(MC_array*PG_3D_corr_array))

E_min = np.amin(adsorbate.potential_array)
E_max = np.amax(adsorbate.potential_array) 
min_E_index = np.where(adsorbate.potential_array == E_min)
z_eq = adsorbate.dz_array[min_E_index[0][0]]

kJ = adsorbate.kJ

f = open("output.txt", "a")
f.write("The minimum potential energy is " + str(E_min*kJ/1e3) + "eV at dz = " + str(z_eq) + "\n")
f.write("The maximum potential energy is " + str(E_max*kJ/1e3) + " eV. \n")

f.write("Pitzer-Gwinn 3D quantum correction array (including ZPE shift): \n")
f.write(str(PG_3D_corr_array))
f.write("\nMC-PSI Q (without quantum correction and ZPE shift): \n")
f.write(str(MC_array))
f.write("\n3D Harmonic oscillator Q: \n")
f.write(str(q_quantum))
f.write("\nFree translator Q: \n")
f.write(str(Analytical_array_2Dgas))
f.write("\ndH/RT: \n")
f.write(str(int_E_array))
f.write("\nCp/R: \n")
f.write(str(Cp_array))
f.write("\nS/R: \n")
f.write(str(S_array))

f.write('\n Run finished at ' + time.strftime('%m/%d/%Y, %H:%M:%S'))
f.close()

print("Job done. See results in output.txt")

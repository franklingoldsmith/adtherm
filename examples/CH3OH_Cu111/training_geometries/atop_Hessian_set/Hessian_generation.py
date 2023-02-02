#!/usr/bin/env python3

import os
import numpy as np
from ase.io import read

dh = 0.01
max_r_atop = 1.5280960929858642 
new_r = 1.000000
dh_angle=0.01/max_r_atop*new_r #displacement in each direction

origin_energy= -105261.6910587031 #eV - DFT energy at atop origin

ndim = 6
hessian=np.zeros((ndim,ndim))

#counter for displacements (84 in this case)
count=0

for i in range(ndim):
    for j in range(ndim):
        add=[]
        if i>j:
            continue
        elif i==j:
            for disp in range(4): #because there are 4 displacements per matrix element
                syst = read('atop_Hessian_set_84.traj', index=count)
                add.append(syst.get_potential_energy())
                count +=1
            if i<3:
                element = (-add[3]+16.0*add[2]-30.0*origin_energy+16.0*add[1]-add[0])/(12.0*dh**2.0)
            else:
                element = (-add[3]+16.0*add[2]-30.0*origin_energy+16.0*add[1]-add[0])/(12.0*dh_angle**2.0)
        elif i<j:
            for disp in range(4):
                syst = read('atop_Hessian_set_84.traj', index=count)
                add.append(syst.get_potential_energy())
                count +=1
            if i<3 and j<3:
                element = (add[3]-add[2]-add[1]+add[0])/(4.0*dh**2.0)
            elif i>2 and j>2:
                element = (add[3]-add[2]-add[1]+add[0])/(4.0*dh_angle**2.0)
            else:
                element = (add[3]-add[2]-add[1]+add[0])/(4.0*dh*dh_angle)
        hessian[i,j]=element
        hessian[j,i]=element

print(hessian)

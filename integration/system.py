#! /usr/bin/env python
import sys, os
import math
import numpy as np
import numpy.linalg as LA
#import scipy

class Trajectory:
    """Generate a data set for either training NN or integrating over it
    """

    def __init__(self):
        #unit converstions
        self.h = 6.626070040E-34 #J-sec or kg m^2/sec planck constant
        self.h_bar = 0.5*self.h/np.pi
        self.amu_to_kg =  1.660539040E-27

#========================================================================================
# Get Mass (include here or in input file?)
#========================================================================================
def get_mass(data):
    """ Get the mass of each atom in the system in unit of amu"""

    mass = np.zeros(data.N_atoms)
    # get the masses
    for (i,a) in enumerate(data.atoms):
        if a=='H':
            mass[i]= 1.0 #1.00794
        elif a=='C':
            mass[i] =12.0 #12.0107
        elif a=='N':
            mass[i] =14.0 #14.00674
        elif a=='O':
            mass[i]= 16.0 #15.9994

    data.mass = mass

#========================================================================================
# Center of Mass
#========================================================================================
def get_center_of_mass(data):
    """ Get coordinate of each atom in the system relative to the center of the mass of the system"""

    coordinates = data.cartesian.copy()
    mass = data.mass
    N_atoms = data.N_atoms

    COM_coordinates = np.zeros([N_atoms, 3])
    center_of_mass = np.zeros([3])

    # first determine the center of mass for run i
    # sum over the weighted coordinates.  
    for i in range(N_atoms):
        center_of_mass[0] += coordinates[i,0]*mass[i] #x-component
        center_of_mass[1] += coordinates[i,1]*mass[i] #y-component
        center_of_mass[2] += coordinates[i,2]*mass[i] #z-component
    #divide by the total mass
    center_of_mass = center_of_mass / np.sum(mass)
    data.COM = center_of_mass

    #shift coordinates relative to the center of mass
    for i in range(N_atoms):
        COM_coordinates[i,0] = coordinates[i,0] - center_of_mass[0]
        COM_coordinates[i,1] = coordinates[i,1] - center_of_mass[1]
        COM_coordinates[i,2] = coordinates[i,2] - center_of_mass[2]
        #print  "%s\t%.8F\t%.8F\t%.8F"%(data.atoms[i], COM_coordinates[i,0], COM_coordinates[i,1], COM_coordinates[i,2])
    data.COM_coordinates = COM_coordinates
    return

#========================================================================================
# maximum distance from center of mass within a fragment
#========================================================================================
def get_max_distance(data):
    """ Get the maximum distance from center of mass within a system"""

    COM_coordinates = data.COM_coordinates.copy()
    mass = data.mass
    N_atoms = data.N_atoms
    r_list = []
    for i in range(N_atoms):
        local_r = np.sqrt( COM_coordinates[i,0]**2.0 + COM_coordinates[i,1]**2.0 + COM_coordinates[i,2]**2.0 )
        r_list.append(local_r)
    max_r = max(r_list)
    data.max_r = max_r
    return

#========================================================================================
# get data from each fragment
#========================================================================================
def get_system_coordinates(data, filename, translate=True):

    text = open(filename, 'r')
    lines = text.readlines()
    text.close()

    N_atoms = int(lines[0])
    method = lines[1].strip('\n')

    atoms = []
    cartesian = np.zeros([N_atoms,3])

    for i in range(N_atoms):
        bits = lines[2+i].split()
        atoms.append(bits[0])
        cartesian[i,0] = float(bits[1])
        cartesian[i,1] = float(bits[2])
        cartesian[i,2] = float(bits[3])

    data.N_atoms = N_atoms
    data.method = method
    data.atoms = atoms
    data.cartesian = cartesian

    get_mass(data)
    if translate:
        get_center_of_mass(data)
        get_max_distance(data)
    return

#========================================================================================
# rotation about center of mass
#========================================================================================
#To be added soon

#========================================================================================
# translation
#========================================================================================
def translate(data,i,j):
    """ This function is used for getting the geometry after the relatively translating related to the 1st fragment
    T = [1 0 0 V_x;
         0 1 0 V_y;
         0 0 1 V_z;
         0 0 0 1]
    """
    if data.rotate==False:
        local_coordinates = data.COM_coordinates.copy()
        #local_coordinates = data.cartesian
    else:
        local_coordinates = data.rotated_COM_coordinates.copy()

    N_atoms = data.N_atoms
    augmented_coordinates = np.insert(local_coordinates, 3,1, axis=1)
    
    dx = data.dx[i]+data.dy[i]*np.tan(30*np.pi/180) 
    dy = data.dy[i]
    dz = data.dz[j]

    # translate
    T = np.zeros([4,4])
    T[0,0] = 1.0
    T[1,1] = 1.0
    T[2,2] = 1.0
    T[3,3] = 1.0
    T[0,3] = dx
    T[1,3] = dy
    T[2,3] = dz

    for i in range(N_atoms):
        augmented_coordinates[i,:] = T.dot(augmented_coordinates[i,:])

    data.translated_coordinates = augmented_coordinates[:,:3]
    return

#========================================================================================
# check minimum distance between the adsorbate and catalytic surface atoms
#========================================================================================
def get_min_max_distance(surf, ads):

    min_distance = 100.0
    max_distance = 0.0
    coordinates = np.zeros([surf.N_atoms + ads.N_atoms,3])
    coordinates[:surf.N_atoms,:] = surf.cartesian
    coordinates[surf.N_atoms:,:] = ads.translated_coordinates

    for i in range(surf.N_atoms):
        for j in range(ads.N_atoms):
            local_distance = LA.norm(surf.cartesian[i,:] - ads.translated_coordinates[j,:]  )
            if local_distance < min_distance:
                min_distance = local_distance
            if local_distance > max_distance:
                max_distance = local_distance
    return (min_distance, max_distance)


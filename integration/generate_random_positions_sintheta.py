#! /usr/bin/env python
import os
import sys

import system as syst
import numpy as np

##################################################
def manipulate_ads(data, option):
    N_values = data.N_values
    if option == 'random':
        matrix = np.zeros((6, N_values))
        for i in range(0, 6):
            if i==1:
                matrix[i,:] = np.random.uniform(-1, 1, size = N_values)
            else:
                matrix[i,:] = np.random.uniform(0, 1, size = N_values)

    data.matrix = matrix
    data.alpha = matrix[0,:]*(data.alpha_boundaries[1]-data.alpha_boundaries[0]) - (- data.alpha_boundaries[0])
    data.beta = np.arccos(matrix[1,:]) - (- data.beta_boundaries[0])
    data.gamma = matrix[2,:] * (data.gamma_boundaries[1] - data.gamma_boundaries[0]) - (- data.gamma_boundaries[0])
    data.dx = matrix[3,:]*(data.x_boundaries[1]-data.x_boundaries[0])-(-data.x_boundaries[0])
    data.dy = matrix[4,:]*(data.y_boundaries[1]-data.y_boundaries[0])-(-data.y_boundaries[0])
    data.dz = matrix[5,:]*(data.z_boundaries[1]-data.z_boundaries[0])-(-data.z_boundaries[0])
    return

##################################################
def fragments_get_coordmatrix(surf,ads):
    #unpack coordinates
    coordinates = np.zeros([surf.N_atoms + ads.N_atoms,3])
    coordinates[:surf.N_atoms,:] = surf.cartesian
    coordinates[surf.N_atoms:,:] = ads.translated_coordinates

    coordinates = coordinates.round(8)

    return coordinates

#! /usr/bin/env python
import os
import sys

import system_train as syst
import numpy as np
import sobol_seq
from sobol_seq import i4_sobol, i4_sobol_generate

#Declare a class for Adsorbates
class Adsorbate:

        def __init__(self):
                #define physical constants
                self.kB = 1.380649e-23 #Boltzmann constant in J/K
                self.h = 6.62607e-34 #Planck's constant in J*s
                self.c = 2.99792458e8 #speed of light in m/s
                #units we will need
                self.kJ = 6.241509125883258e+21 #eV per kJ

##################################################
def manipulate_ads(data, adsorbate, option):
    N_values = data.N_values
    if option == 'gauss':
        matrix=np.zeros((6,N_values))
        COM_x = data.COM[0]
        COM_y = data.COM[1]
        COM_z = data.COM[2]  
        if adsorbate.N_atoms > 2:
            gaussmean = np.array([COM_x, COM_y, COM_z, 0.0, 0.0, 0.0])
        elif adsorbate.N_atoms == 2:
            gaussmean = np.array([COM_x, COM_y, COM_z, 0.0, 0.0])
        else:
            gaussmean = np.array([COM_x, COM_y, COM_z])
        #The Hessian matrix at the minimum
        hess = adsorbate.hessian
        invhess = np.linalg.inv(hess)
        gausscov = invhess.copy()
        gaussmat = np.random.multivariate_normal(gaussmean, gausscov, size=N_values, check_valid='warn')

        for i in range(0, len(gaussmean)):
                matrix[i,:] = gaussmat[:,i]

    if option == 'sobol':
        print("N_atoms")
        print(adsorbate.N_atoms)
        matrix=np.zeros((6,N_values))
        if adsorbate.N_atoms > 2:
            sobolmatrix = i4_sobol_generate (6, N_values, 1) #for 6D systems (nonlinear adsorbates)
        elif adsorbate.N_atoms == 2:
            sobolmatrix = i4_sobol_generate (5, N_values, 1) #for 5D systems (linear adsorbates)
        elif adsorbate.N_atoms == 1:
            sobolmatrix = i4_sobol_generate (3, N_values, 1) #for 3D systems (atomic adsorbates)
        for i in range(0,6):
           if len(sobolmatrix[0,:]) == 6:
               matrix[i,:]=sobolmatrix[:,i]
           elif len(sobolmatrix[0,:]) == 5:
               if i<5:
                   matrix[i,:]=sobolmatrix[:,i]
           elif len(sobolmatrix[0,:]) == 3:
               if i<3:
                   matrix[i,:]=sobolmatrix[:,i]
    if option == 'random':
        matrix = np.zeros((6, N_values))
        for i in range(0, 6):
                matrix[i,:] = np.random.uniform(0, 1, size = N_values)

    data.matrix = matrix

    if option == 'random' or option == 'sobol':
        data.dy = matrix[1,:]*adsorbate.uc_y/3.0
        data.dx = matrix[0,:]*adsorbate.uc_x/3.0+data.dy*np.tan(30*np.pi/180)
        data.dz = matrix[2,:]*(data.z_high-data.z_low)+data.z_low
        data.alpha = matrix[3,:]*2.0*np.pi - np.pi #rotate pi in each direction around x-axis
        data.beta = np.arccos(matrix[4,:]*2.0-1.0) - np.arccos(0.0) #rotate 1/2 pi in each direction around y axis
        data.gamma = matrix[5,:]*2.0*np.pi - np.pi #rotate pi in each direction around x-axis
        if adsorbate.N_atoms < 3:
            data.gamma[:] = 0.0
        if adsorbate.N_atoms == 1:
            data.alpha[:] = 0.0
            data.beta[:] = 0.0    
    elif option == 'gauss':
        data.dx = matrix[0,:]
        data.dy = matrix[1,:]
        data.dz = matrix[2,:]
        data.alpha = matrix[3,:]
        data.beta = matrix[4,:]
        data.gamma = matrix[5,:]
    return

##################################################
def get_coordmatrix(surf,ads):
    #unpack coordinates
    coordinates = np.zeros([surf.N_atoms + ads.N_atoms,3])
    coordinates[:surf.N_atoms,:] = surf.cartesian
    coordinates[surf.N_atoms:,:] = ads.translated_coordinates

    coordinates = coordinates.round(8)

    return coordinates

##################################################
def create_DFT_input(POSCAR_lines, job_name, new_coord): #create VASP/QE input with xyz coord
    """create the DFT input,
    the geometry has already been checked the minimum distance"""
    new_geom = ''

    for atom_i in range(len(new_coord[:,0])):
        new_geom += str(new_coord[atom_i,:]).replace("[","").replace("]","") + "   F   F   F \n"

    # here got all the new geometry set up
    new_lines = []
    for line in POSCAR_lines:
            newline = line
            newline = newline.replace('!GEOM_HERE',new_geom)
            new_lines.append(newline)
    return new_lines

##################################################
def main():

    path = os.getcwd()
    ##================= PREPARE WORK ===================##
    blank = open(path + '/blank_POSCAR', 'r') #separate POSCAR files for each geometry
    POSCAR_template = blank.readlines()
    blank.close()

    filename = "input.inp"
    adsorbate = Adsorbate()
    parse_input_file(filename, adsorbate)

    #open the two trajectories. Ideally, they will be centered along their centers of mass
    surf = syst.Trajectory()
    syst.get_system_coordinates(surf, path+'/' + adsorbate.surface + '.inp', translate=False)

    ads = syst.Trajectory()
    syst.get_system_coordinates(ads, path + '/' + adsorbate.name + '.inp')

    min_cutoff = 0.5 #don't allow less than a 0.5 angstrom separation between atom centers
    max_cutoff = 100.0 
    if adsorbate.N_atoms == 1:
        ads.rotate=False 
    else:
        ads.rotate=True
    #print("Maximum Distance from Center of Mass: %.2F\t%.2F"%(0, ads.max_r))

    METHOD = adsorbate.method
    N_values = 0
    ads.z_low = adsorbate.z_low+adsorbate.z_globmin
    ads.z_high = adsorbate.z_high+adsorbate.z_globmin
    #the base-2 exponent to determine the number of geometries
    N_values = 2**adsorbate.exponent
    ads.N_values = N_values
    print(str(N_values) + " geometries will be generated using method "+ METHOD +" with sampling "+str(N_values))

    ref = open('coord_list.dat','w')

    ##================= manipulate the adsorbate ===================##
    manipulate_ads(ads, adsorbate, METHOD)

    # Now cycle through each of the Sobol sequences
    job_index = []
    too_close_index= []
    beta_outside_index = []
    number_valid = 0
    for i in range(0, N_values):
        #start generating the unique cartesian coordinates for the adsorbate

        job_name = 'POSCAR_'+str(i)
        if ads.rotate:
            syst.euler_rotate_COM(ads,i)
        syst.translate(ads,i,i)
        valid = True

        uc_x = adsorbate.uc_x/3.0
        uc_y = adsorbate.uc_y/3.0
        y_ub = uc_y
        y_lb = 0.0
        x_ub = uc_x+ads.dy[i]*(1./np.sqrt(3))
        x_lb = ads.dy[i]*(1./np.sqrt(3))
        z_ub = ads.z_high
        z_lb = ads.z_low
        
        #if minimum distance is too small
        min_dist,max_dist = syst.get_min_max_distance(surf, ads)
        new_coord = get_coordmatrix(surf, ads)
        if min_dist < min_cutoff or max_dist>max_cutoff:
            valid = False
            too_close_index.append(i)
        #or if coordinates are outside the rhombus boundaries (can happen in gaussian distribution case)
        elif ads.dz[i] > z_ub or ads.dz[i] < z_lb:
            valid = False
            print("coord dz = " + str(ads.dz[i]) + " outside rhombus")
            too_close_index.append(i)
        elif ads.beta[i] > 0.5*np.pi or ads.beta[i] < -0.5*np.pi:
            valid = False
            print("coord beta = " + str(ads.beta[i]) + " outside allowed range")
            beta_outside_index.append(i)

        if valid==True:
            while ads.dy[i] > y_ub or ads.dy[i] < y_lb:
                print("coord dy = " + str(ads.dy[i]) + " outside rhombus")
                if ads.dy[i] > y_ub:
                    ads.dy[i] -= uc_y
                    ads.dx[i] -= uc_y*(1./np.sqrt(3))
                    new_coord[-adsorbate.N_atoms:,0] -= uc_y*(1./np.sqrt(3))
                    new_coord[-adsorbate.N_atoms:,1] -= uc_y
                elif ads.dy[i] < y_lb:
                    ads.dy[i] += uc_y
                    ads.dx[i] += uc_y*(1./np.sqrt(3))
                    new_coord[-adsorbate.N_atoms:,0] += uc_y*(1./np.sqrt(3))
                    new_coord[-adsorbate.N_atoms:,1] += uc_y
                print("new dy = "+ str(ads.dy[i]) + " and dx = " + str(ads.dx[i]))
                x_ub = uc_x+ads.dy[i]*(1./np.sqrt(3))
                x_lb = ads.dy[i]*(1./np.sqrt(3))

            while (ads.dx[i] > x_ub or ads.dx[i] < x_lb):
                print("coord dx = " + str(ads.dx[i]) + " outside rhombus when dy = " + str(ads.dy[i]))
                if ads.dx[i] > x_ub:
                    ads.dx[i] -= uc_x
                    new_coord[-adsorbate.N_atoms:,0] -= uc_x
                elif ads.dx[i] < x_lb:
                    ads.dx[i] += uc_x
                    new_coord[-adsorbate.N_atoms:,0] += uc_x
                print("new dx = " + str(ads.dx[i]))

        paramline = "%d\t%.6F\t%.6F\t%.6F\t%.6F\t%.6F\t%.6F\n"%(i, ads.dx[i], ads.dy[i], ads.dz[i],  ads.alpha[i], ads.beta[i], ads.gamma[i])
        if valid == True: 
            ref.write(paramline)
            number_valid +=1 


        file_lines = create_DFT_input(POSCAR_template,job_name,new_coord)

        if valid == False:
            if not os.path.exists("eliminated/"):
                os.makedirs("eliminated/")
            job_name = "eliminated/" + job_name

        new_file = open(job_name, 'w')
        for line in file_lines:
            new_file.write(line)
        new_file.close()

    ref.close()
    print("Generating %d geometries done"%(N_values))
    print("Total geometries eliminated because of too close min_distance or z outside allowed range: %d"%(len(too_close_index)))
    print("Total geometries eliminated because beta was outside the allowed range: %d"%(len(beta_outside_index)))
    print("The number of valid points is: " + str(number_valid)) 

#######################################################################################################
def parse_input_file(filename, adsorbate):
    abs_file_path = str(filename)

    input_file = open(abs_file_path, 'r')
    lines = input_file.readlines()
    input_file.close()

    error_name = True
    error_surface = True
    error_composition = True
    error_adsorbate_mass = True
    error_zlow = True
    error_zhigh = True
    error_ucx = True
    error_ucy = True
    error_Hessian = True
    error_exponent = True
    error_method = True

    for line in lines:
        #Start by looking for the name
        if line.strip().startswith("name"):
            bits = line.split('=')
            name = bits[1].strip().replace("'","").replace('"','')
            adsorbate.name = name
            error_name = False
        #Now look for the surface
        if line.strip().startswith("surface"):
            bits = line.split('=')
            name = bits[1].strip().replace("'","").replace('"','')
            adsorbate.surface = name
            error_surface = False
        #Now look for the composition    
        elif line.strip().startswith("composition"):
            bits = line.split('=')
            composition = bits[1].strip().replace("{","").replace("}","").split(',')
            adsorbate.composition = {}
            for pair in composition:
                element, number = pair.split(":")
                element = element.strip().replace("'","").replace('"','')
                number = int(number)
                adsorbate.composition[element]=number
            N_adsorbate_atoms = 0
            for element in adsorbate.composition:
                if element!='Pt':
                    N_adsorbate_atoms += adsorbate.composition[element]
            adsorbate.N_atoms = N_adsorbate_atoms
            error_composition = False
        #Now look for the adsorbate mass
        elif line.strip().startswith("adsorbate_mass"):
            bits = line.split('=')
            adsorbate_mass_info = bits[1].strip().replace("[","").replace("]","").split(',')
            adsorbate_mass = float(adsorbate_mass_info[0])
            units = adsorbate_mass_info[1].strip().replace("'","").replace('"','')
            if units=='kg':
                adsorbate.mu = adsorbate_mass
                adsorbate.mu_units = units.strip()
                error_adsorbate_mass = False
            else:
                print("Adsorbate mass is missing proper units!\n Please use 'kg'")
                break
        #Now look for the z lower boundary relative to COM
        elif line.strip().startswith("z_low"):
            bits = line.split('=')
            zlow_info = bits[1].strip().replace("[","").replace("]","").split(',')
            zlow = float(zlow_info[0])
            units = zlow_info[1].strip().replace("'","").replace('"','')
            if units=='Angstrom' or units=='angstrom':
                adsorbate.z_low = zlow
                adsorbate.z_low_units = units.strip()
                error_zlow = False
            else:
                print("Adsorbate z lower boundary is missing proper units!\n Please use 'Angstrom'")
                break
        #Now look for the z higher boundary relative to COM
        elif line.strip().startswith("z_high"):
            bits = line.split('=')
            zhigh_info = bits[1].strip().replace("[","").replace("]","").split(',')
            zhigh = float(zhigh_info[0])
            units = zhigh_info[1].strip().replace("'","").replace('"','')
            if units=='Angstrom' or units=='angstrom':
                adsorbate.z_high = zhigh
                adsorbate.z_high_units = units.strip()
                error_zhigh = False
            else:
                print("Adsorbate z higher boundary is missing proper units!\n Please use 'Angstrom'")
                break
        #Now look for the uc_x parameter (width of 3x3 unit cell for a 111 surface)
        elif line.strip().startswith("uc_x"):
            bits = line.split('=')
            ucx_info = bits[1].strip().replace("[","").replace("]","").split(',')
            ucx = float(ucx_info[0])
            units = ucx_info[1].strip().replace("'","").replace('"','')
            if units=='Angstrom' or units=='angstrom':
                adsorbate.uc_x = ucx
                adsorbate.ucx_units = units.strip()
                error_ucx = False
            else:
                print("Unit cell width (x) is missing proper units!\n Please use 'Angstrom'")
                break
        #Now look for the uc_y parameter (right angle length of 3x3 unit cell for a 111 surface)
        elif line.strip().startswith("uc_y"):
            bits = line.split('=')
            ucy_info = bits[1].strip().replace("[","").replace("]","").split(',')
            ucy = float(ucy_info[0])
            units = ucy_info[1].strip().replace("'","").replace('"','')
            if units=='Angstrom' or units=='angstrom':
                adsorbate.uc_y = ucy
                adsorbate.ucy_units = units.strip()
                error_ucy = False
            else:
                print("Unit cell length (y) is missing proper units!\n Please use 'Angstrom'")
                break
        #Now look for the global minimum z position
        elif line.strip().startswith("z_globmin"):
            bits = line.split('=')
            z_globmin_info = bits[1].strip().replace("[","").replace("]","").split(',')
            z_globmin = float(z_globmin_info[0])
            units = z_globmin_info[1].strip().replace("'","").replace('"','')
            if units=='Angstrom' or units=='angstrom':
                adsorbate.z_globmin = z_globmin
                adsorbate.z_globmin_units = units.strip()
                error_zglobmin = False
            else:
                print("Global minimum z position is missing proper units!\n Please use 'Angstrom'")
                break
        #Now look for the Hessian
        elif line.strip().startswith("Hessian"):
            bits = line.split('=')
            mat = bits[1].split(';')
            hess_info = np.zeros((len(mat)-1, len(mat)-1))
            for i in range(len(mat)-1):
                hessinfo = mat[i].strip().replace("[","").replace("]","")
                hessinfo = hessinfo.split(',')
                for j in range(len(mat)-1):
                    hess_info[i,j] = float(hessinfo[j])
            units = mat[-1].strip().replace("]","").replace("'","")
            if units=='eV/Angstrom2':
                adsorbate.hessian_units = units.strip()
                adsorbate.hessian = hess_info
                error_Hessian = False
            else:
                print("Hessian matrix values are missing proper units!\n Please use 'eV/Angstrom2'")
                break
        #Now look for the exponent
        elif line.strip().startswith("exponent"):
            bits = line.split('=')
            exp_info = bits[1].strip().replace("[","").replace("]","").split(',')
            exp = int(exp_info[0])
            adsorbate.exponent = exp
            error_exponent = False
        #Now look for the distribution method
        if line.strip().startswith("method"):
            bits = line.split('=')
            method = bits[1].strip().replace("'","").replace('"','')
            adsorbate.method = method
            error_method = False

    if error_name or error_surface or error_composition or error_adsorbate_mass or error_zlow or error_zhigh or error_ucx or error_ucy or error_Hessian or error_exponent or error_method:
        print("Input file is missing information: %s"%(filename))
    else:
        print("successfully parsed file %s"%(filename))

    return

if __name__ == "__main__":
    main()
                                                                            

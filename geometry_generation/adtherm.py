###  Import some stuff ###
import os
import sys
import numpy as np
import sobol_seq
from sobol_seq import i4_sobol, i4_sobol_generate
import numpy.linalg as LA
#########################


class AdTherm:

    def __init__(self, atoms, indices, n_gauss=5000, n_sobol=200, n_random=200, relax_internals=False, z_below=1, z_above=5):
        self.atoms = atoms
        self.indices=indices
        self.adsorbate = self.atoms[indices].copy()
        self.N_atoms_in_adsorbate = len(self.adsorbate)
        self.N_gauss = n_gauss
        self.N_sobol = n_sobol
        self.N_random = n_random
        self.uc_x = atoms.get_cell_lengths_and_angles()[0]
        self.uc_y = atoms.get_cell_lengths_and_angles()[1]
        self.adsorbate_center_of_mass=self.adsorbate.get_center_of_mass()
        self.adsorbate_mass=np.sum(self.adsorbate.get_masses())
        self.z_low=-z_below+np.min(self.adsorbate.positions[:,2])
        self.z_high=z_above+np.min(self.adsorbate.positions[:,2])
        self.min_cutoff = 0.5
        self.max_cutoff = 100.0
        self.hessian_displacement_size = 1e-2
        if self.N_atoms_in_adsorbate == 1:
            self.rotate = False 
        else:
            self.rotate = True
        self.N_hessian = 12
        if self.N_atoms_in_adsorbate == 1:
            self.N_hessian = 10
        elif self.N_atoms_in_adsorbate == 2:
            self.N_hessian = 6

    def get_min_max_distance(self, new_coord):

        min_distance = 100.0
        max_distance = 0.0
        for i in range(len(new_coord)):
            if i not in self.indices:
                for j in self.indices:
                    local_distance = LA.norm(new_coord[i,:]-new_coord[j,:])
                    if local_distance < min_distance:
                        min_distance = local_distance
                    if local_distance > max_distance:
                        max_distance = local_distance
        return min_distance, max_distance


    def coord_generate(self, method, N_values):

        if method == 'gauss':
            matrix = np.zeros((6,N_values))
            COM_x, COM_y, COM_z  = self.adsorbate_center_of_mass
            if self.N_atoms_in_adsorbate > 2:
                gaussmean = np.array([COM_x, COM_y, COM_z, 0.0, 0.0, 0.0])
            elif self.N_atoms_in_adsorbate == 2:
                gaussmean = np.array([COM_x, COM_y, COM_z, 0.0, 0.0])
            else:
                gaussmean = np.array([COM_x, COM_y, COM_z])
            hess = self.rigid_body_hessian
            invhess = np.linalg.inv(hess)
            gausscov = invhess.copy()
            gaussmat = np.random.multivariate_normal(gaussmean, gausscov, size=N_values, check_valid='warn')
            for i in range(0, len(gaussmean)):
                    matrix[i,:] = gaussmat[:,i]
        
        if method == 'sobol':
            matrix=np.zeros((6,N_values))
            if self.N_atoms_in_adsorbate > 2:
                sobolmatrix = i4_sobol_generate (6, N_values, 1)
            elif self.N_atoms_in_adsorbate == 2:
                sobolmatrix = i4_sobol_generate (5, N_values, 1)
            elif self.N_atoms_in_adsorbate == 1:
                sobolmatrix = i4_sobol_generate (3, N_values, 1)
            for i in range(0,6):
                if len(sobolmatrix[0,:]) == 6:
                    matrix[i,:]=sobolmatrix[:,i]
                elif len(sobolmatrix[0,:]) == 5:
                    if i<5:
                        matrix[i,:]=sobolmatrix[:,i]
                elif len(sobolmatrix[0,:]) == 3:
                    if i<3:
                        matrix[i,:]=sobolmatrix[:,i]

        if method == 'random':
            matrix = np.zeros((6, N_values))
            for i in range(0, 6):
                matrix[i,:] = np.random.uniform(0, 1, size = N_values)

        if method == 'hessian':
            matrix=np.zeros((6,N_values))
            count=0
            dh = self.hessian_displacement_size
            if self.N_atoms_in_adsorbate == 1:
                ndim = 3
            elif self.N_atoms_in_adsorbate == 2:
                ndim = 5
            else:
                ndim = 6
            for i in range(ndim):
                for disp in [-1,1]:
                    displacement=[0,0,0,0,0,0]
                    displacement[i] = disp*dh 
                    matrix[:,count] = displacement
                    count += 1

        if method == 'random' or method == 'sobol':
            dy = matrix[1,:]*self.uc_y/3.0
            dx = matrix[0,:]*self.uc_x/3.0+dy*np.tan(30*np.pi/180)
            dz = matrix[2,:]*(self.z_high-self.z_low)+self.z_low
            alpha = matrix[3,:]*2.0*np.pi - np.pi
            beta = np.arccos(matrix[4,:]*2.0-1.0) - np.arccos(0.0)
            gamma = matrix[5,:]*2.0*np.pi - np.pi
            if self.N_atoms_in_adsorbate < 3:
                gamma[:] = 0.0
            if self.N_atoms_in_adsorbate == 1:
                alpha[:] = 0.0
                beta[:] = 0.0    
        elif method == 'gauss':
            dx = matrix[0,:]
            dy = matrix[1,:]
            dz = matrix[2,:]
            alpha = matrix[3,:]
            beta = matrix[4,:]
            gamma = matrix[5,:]
        elif method == 'hessian':
            COM_x, COM_y, COM_z  = self.adsorbate_center_of_mass
            dx = matrix[0,:] + COM_x
            dy = matrix[1,:] + COM_y
            dz = matrix[2,:] + COM_z
            alpha = matrix[3,:]
            beta = matrix[4,:]
            gamma = matrix[5,:]
            if self.N_atoms_in_adsorbate == 1:
                alpha[:] = 0.0
                beta[:] = 0.0
                gamma[:] = 0.0
            elif self.N_atoms_in_adsorbate == 2:
                gamma[:] = 0.
        coord=np.transpose(np.asarray([dx,dy,dz,alpha,beta,gamma]))
        print(np.shape(coord))
        return coord

    def check_coord(self,coord):
    
        ref = open('coord_list.dat','w')
        ref_eliminated = open('invalid_coord_list.dat','w')
        dft_jobs = []
        number_valid = 0
        number_invalid = 0
        for i in range(0, len(coord)):
            atoms=self.atoms.copy()
            adsorbate=self.adsorbate.copy()
            if self.rotate:
                adsorbate.euler_rotate(coord[i,3],coord[i,4],coord[i,5], center=self.adsorbate_center_of_mass)
            adsorbate.translate(coord[i,0:3])
            valid = True
            atoms[self.indices].positions=adsorbate.positions
            new_coord = atoms.positions

            uc_x = self.uc_x/3.0
            uc_y = self.uc_y/3.0
            y_ub = uc_y
            y_lb = 0.0
            x_ub = uc_x+coord[i,1]*(1./np.sqrt(3))
            x_lb = coord[i,1]*(1./np.sqrt(3))
            z_ub = self.z_high
            z_lb = self.z_low
        
            min_dist,max_dist = self.get_min_max_distance(new_coord) ## Gotta think about this one
            if min_dist < self.min_cutoff or max_dist>self.max_cutoff:
                valid = False
            elif coord[i,2] > z_ub or coord[i,2] < z_lb:
                valid = False
            elif coord[i,4] > 0.5*np.pi or coord[i,4] < -0.5*np.pi:
                valid = False

            if valid==True:
                while coord[i,1] > y_ub or coord[i,1] < y_lb:
                    if coord[i,1] > y_ub:
                        coord[i,1] -= uc_y
                        coord[i,0] -= uc_y*(1./np.sqrt(3))
                        new_coord[self.indices,0] -= uc_y*(1./np.sqrt(3))
                        new_coord[self.indices,1] -= uc_y
                    elif coord[i,1] < y_lb:
                        coord[i,1] += uc_y
                        coord[i,0] += uc_y*(1./np.sqrt(3))
                        new_coord[self.indices,0] += uc_y*(1./np.sqrt(3))
                        new_coord[self.indices,1] += uc_y
                    x_ub = uc_x+coord[i,1]*(1./np.sqrt(3))
                    x_lb = coord[i,1]*(1./np.sqrt(3))

                while (coord[i,0] > x_ub or coord[i,0] < x_lb):
                    if coord[i,0] > x_ub:
                        coord[i,0] -= uc_x
                        new_coord[self.indices,0] -= uc_x
                    elif coord[i,0] < x_lb:
                        coord[i,0] += uc_x
                        new_coord[self.indices,0] += uc_x

            paramline = "%d\t%.6F\t%.6F\t%.6F\t%.6F\t%.6F\t%.6F\n"%(i, coord[i,0], coord[i,1], coord[i,2],  coord[i,3], coord[i,4], coord[i,5])
            if valid == True: 
                ref.write(paramline)
                number_valid +=1 
                dft_jobs.append(atoms)


            if valid == False:
                ref_eliminated.write(paramline)
                number_invalid+=1

        ref.close()
        ref_eliminated.close()

        print("Generating %d geometries done"%(len(coord)))
        print("The number of invalid points is:" + str(number_invalid))
        print("The number of valid points is: " + str(number_valid))

    def calculate_rigid_body_hessian(self, displacement_forces):

        dh = self.hessian_displacement_size
        if len(self.indices) > 2:
            ndim = 6
        if len(self.indices) == 2:
            ndim = 5
        if len(self.indices) ==1:
            ndim = 3
        hessian=np.zeros((ndim,ndim))
        f=displacement_forces
        for i in range(ndim):
            hessian[:,i]=(1/(2*dh))*(f[2*i+1]-f[2*i])      

    def run(self):
        hessian_job_list=self.coord_generate('hessian', self.N_hessian)
        checked_hessian_job_list=self.check_coord(hessian_job_list)
        displacement_forces=np.zeros([self.indic  #stopped here
		for img in checked_hessian_job_list:
			img.get_forces()
#        H=calculate_H(H_list)
#        np.savetxt(H,'Rigid_H.txt')

#        gauss_list=geometry_generate('gauss',hessian=H)







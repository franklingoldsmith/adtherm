###  Import some stuff ###
import os
import sys
import numpy as np
import sobol_seq
from sobol_seq import i4_sobol, i4_sobol_generate
import numpy.linalg as LA
import copy
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.constraints import FixExternals
#########################


class AdTherm:

    def __init__(self, atoms, indices, calc, n_gauss=None, n_sobol=None, n_random=None, relax_internals=False, z_below=1, z_above=5):
        self.atoms = atoms
        self.indices = indices
        self.calc = calc
        self.adsorbate = self.atoms[indices].copy()
        self.N_atoms_in_adsorbate = len(self.adsorbate)
        self.N_gauss = n_gauss
        self.N_sobol = n_sobol
        self.N_random = n_random
        self.relax_internals = relax_internals
        self.uc_x = atoms.get_cell_lengths_and_angles()[0]
        self.uc_y = atoms.get_cell_lengths_and_angles()[1]
        self.adsorbate_center_of_mass=self.adsorbate.get_center_of_mass()
        self.z_low=-z_below+np.min(self.adsorbate.positions[:,2])
        self.z_high=z_above+np.min(self.adsorbate.positions[:,2])
        self.min_cutoff = 0.5
        self.max_cutoff = 100.0
        self.hessian_displacement_size = 1e-4
        self.rotate = True
        self.N_hessian = 12
        self.ndim = 6
        if self.N_atoms_in_adsorbate == 2:
            self.N_hessian = 10
            self.ndim = 5
        elif self.N_atoms_in_adsorbate == 1:
            self.N_hessian = 6
            self.ndim = 3
            self.rotate = False 

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

        traj = Trajectory(method+'_set.traj', 'w')
        name="{}_x_train.dat".format(str(method))
        ref = open(name,'w')
        dft_jobs = []
        Iter = 0
        while Iter < N_values:
            displacement = np.zeros(6)

            if method == 'gauss':
                gaussmean = np.zeros(self.ndim)
                gaussmean[0:3] = self.adsorbate_center_of_mass
                hess = self.rigid_body_hessian
                gausscov = np.copy(LA.inv(hess))
                displacement[0:self.ndim]  = np.random.multivariate_normal(gaussmean, gausscov, size=1, check_valid='warn')
                
            if method == 'random' or method == 'sobol': 
                if method == 'sobol':
                    if self.N_atoms_in_adsorbate > 2:
                        displacement = i4_sobol_generate(6, 1, Iter+1)[0,:]
                    elif self.N_atoms_in_adsorbate == 2:
                        displacement[0:5] = i4_sobol_generate(5, 1, Iter+1)[0,:]
                    elif self.N_atoms_in_adsorbate == 1:
                        displacement[0:3] = i4_sobol_generate(3, 1, Iter+1)[0,:]
                if method == 'random':
                    displacement[0:self.ndim] = np.random.uniform(0, 1, size = self.ndim)

                displacement[1] = np.copy(displacement[1]*self.uc_y/3.0)
                displacement[0] = np.copy(displacement[0]*self.uc_x/3.0+displacement[1]*np.tan(30*np.pi/180))
                displacement[2] = np.copy(displacement[2]*(self.z_high-self.z_low)+self.z_low)
                displacement[3] = np.copy(displacement[3]*2.0*np.pi - np.pi)
                displacement[4] = np.copy(np.arccos(displacement[4]*2.0-1.0) - np.arccos(0.0))
                displacement[5] = np.copy(displacement[5]*2.0*np.pi - np.pi)

            if method == 'hessian':
                displacement[int(Iter / 2)] = self.hessian_displacement_size
                displacement[0:3] += self.adsorbate_center_of_mass
                self.hessian_displacement_size*=-1

            displacement[self.ndim::] = 0
            d=displacement
            paramline = "%.6F\t%.6F\t%.6F\t%.6F\t%.6F\t%.6F\n"%(d[0], d[1], d[2], d[3], d[4], d[5])
            valid, atoms = self.check_coord(displacement)
            print(valid)
            if valid: 
                ref.write(paramline)
                Iter +=1
                atoms.calc=self.calc
                traj.write(atoms)
                dft_jobs.append(atoms)
        ref.close()
        return dft_jobs

    def check_coord(self,coord):
        conv = 180 / np.pi
#        pa=np.transpose(self.adsorbate.get_moments_of_inertia(vectors=True)[1])
        atoms=self.atoms.copy()
        adsorbate=self.adsorbate.copy()
#        adsorbate.positions=np.copy(np.transpose(np.matmul(pa,np.transpose(adsorbate.positions))))
        if self.rotate:
            adsorbate.rotate(conv * coord[3],'x','COM')
            adsorbate.rotate(conv * coord[4],'y','COM')
            adsorbate.rotate(conv * coord[5],'z','COM')
#        adsorbate.positions=np.copy(np.transpose(np.matmul(LA.inv(pa),np.transpose(adsorbate.positions))))
        adsorbate.translate(coord[0:3] - self.adsorbate_center_of_mass)
        valid = True
        atoms.positions[self.indices]=adsorbate.positions
        new_coord = atoms.positions

        uc_x = self.uc_x/3.0
        uc_y = self.uc_y/3.0
        y_ub = uc_y
        y_lb = 0.0
        x_ub = uc_x+coord[1]*(1./np.sqrt(3))
        x_lb = coord[1]*(1./np.sqrt(3))
        z_ub = self.z_high
        z_lb = self.z_low
        
        min_dist,max_dist = self.get_min_max_distance(new_coord)
        if min_dist < self.min_cutoff or max_dist>self.max_cutoff:
            valid = False
        elif coord[2] > z_ub or coord[2] < z_lb:
            valid = False
        elif coord[4] > 0.5*np.pi or coord[4] < -0.5*np.pi:
            valid = False

        if valid==True:
            while coord[1] > y_ub or coord[1] < y_lb:
                if coord[1] > y_ub:
                    coord[1] -= uc_y
                    coord[0] -= uc_y*(1./np.sqrt(3))
                    new_coord[self.indices,0] -= uc_y*(1./np.sqrt(3))
                    new_coord[self.indices,1] -= uc_y
                elif coord[1] < y_lb:
                    coord[1] += uc_y
                    coord[0] += uc_y*(1./np.sqrt(3))
                    new_coord[self.indices,0] += uc_y*(1./np.sqrt(3))
                    new_coord[self.indices,1] += uc_y
                x_ub = uc_x+coord[1]*(1./np.sqrt(3))
                x_lb = coord[1]*(1./np.sqrt(3))

            while (coord[0] > x_ub or coord[0] < x_lb):
                if coord[0] > x_ub:
                    coord[0] -= uc_x
                    new_coord[self.indices,0] -= uc_x
                elif coord[0] < x_lb:
                    coord[0] += uc_x
                    new_coord[self.indices,0] += uc_x

        return valid, atoms

    def calculate_rigid_hessian(self, displacement_list, force_list):
        dh = self.hessian_displacement_size
        f = force_list
        x = displacement_list
        B = self.get_external_basis(self.atoms)
        f_sub = np.matmul(np.transpose(B),f)
        df_sub = np.zeros([self.ndim,self.ndim])
        dx = np.zeros([3*len(self.indices),self.ndim])
        for i in range(self.ndim):
            dx[:,i]=x[:, 2*i+1]-x[:, 2*i]
            df_sub[:,i]=f_sub[:, 2*i]-f_sub[:, 2*i+1]
        dx_sub = np.matmul(np.transpose(B),dx)
        H=np.matmul(LA.inv(dx_sub),df_sub)
        print(np.matmul(np.transpose(B),B))

        return H

    def get_external_basis(self,atoms):
        
        B=np.zeros([3*len(self.indices),6])
        ads_pos=atoms.positions[self.indices]
        for i in range(len(self.indices)):
            B[3*i,0]=1
            B[3*i+1,1]=1
            B[3*i+2,2]=1
            B[3*i:3*i+3,3]=np.cross(ads_pos[i,:],np.asarray([1,0,0]))
            B[3*i:3*i+3,4]=np.cross(ads_pos[i,:],np.asarray([0,1,0]))
            B[3*i:3*i+3,5]=np.cross(ads_pos[i,:],np.asarray([0,0,1]))
        for i in range(6):
            B[:,i] *= 1 / LA.norm(np.copy(B[:,i]))
        return B


    def run(self):

        dft_list=self.coord_generate('hessian', self.N_hessian)
        force_list=np.zeros([3*len(self.indices),self.N_hessian])
        displacement_list=np.zeros([3*len(self.indices),self.N_hessian])
        E_list=np.zeros(self.N_hessian)
        y_train = open('hessian_y_train.dat','w')
        for i, img in enumerate(dft_list):
            f_atoms=img.get_forces()[self.indices]
            force_list[:,i]=f_atoms.reshape(-1)
            E_list[i] = img.get_potential_energy()
            y_train.write(str(E_list[i]) + "\n")
            disp_atoms=img.positions[self.indices]
            displacement_list[:,i]=disp_atoms.reshape(-1)
        y_train.close()
        self.rigid_body_hessian = self.calculate_rigid_hessian(displacement_list,force_list)
        np.savetxt('rigid_hessian.out', self.rigid_body_hessian)
        print(LA.eig(self.rigid_body_hessian))

        if self.N_gauss:
            dft_list = self.coord_generate('gauss', self.N_gauss)
            force_list=np.zeros([3*len(self.indices),self.N_gauss])
            E_list=np.zeros(self.N_gauss)
            y_train = open('gauss_y_train.dat','w')
            for i, img in enumerate(dft_list):
                if self.relax_internals:
                    c = FixExternals(img, self.indices)
                    img.set_constraint(c)
                    dyn = BFGS(img)
                    dyn.run(fmax=0.05)
                force_list[:,i] = img.get_forces()[self.indices].reshape(-1)
                E_list[i] = img.get_potential_energy()
                y_train.write(str(E_list[i]) + '\n')
            y_train.close()

        if self.N_sobol:
            dft_list = self.coord_generate('sobol', self.N_sobol)
            force_list=np.zeros([3*len(self.indices),self.N_sobol])
            E_list=np.zeros(self.N_sobol)
            y_train = open('sobol_y_train.dat','w')
            for i, img in enumerate(dft_list):
                if self.relax_internals:
                    c = FixExternals(img, self.indices)
                    img.set_constraint(c)
                    dyn = BFGS(img)
                    dyn.run(fmax=0.05)
                force_list[:,i] = img.get_forces()[self.indices].reshape(-1)
                E_list[i] = img.get_potential_energy()
                y_train.write(str(E_list[i]) + '\n')
            y_train.close()

        if self.N_random:
            dft_list = self.coord_generate('random', self.N_random)
            force_list=np.zeros([3*len(self.indices),self.N_random])
            E_list=np.zeros(self.N_random)
            y_train = open('random_y_train.dat','w')
            for i, img in enumerate(dft_list):
                if self.relax_internals:
                    c = FixExternals(img, self.indices)
                    img.set_constraint(c)
                    dyn = BFGS(img)
                    dyn.run(fmax=0.05)
                force_list[:,i] = img.get_forces()[self.indices].reshape(-1)
                E_list[i] = img.get_potential_energy()
                y_train.write(str(E_list[i]) + '\n')
            y_train.close()



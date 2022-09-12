<p align="center">
  <img width="255" height="103.4366" src="AdTherm_logo.png">
</p>

# Adsorbate Thermochemistry | AdTherm

AdTherm is a code that performs Monte Carlo phase space integration (MC-PSI) to calculate adsorbate partition functions that include anharmonicity, as well as anharmonic thermophysical quantities of adsorbates.

## How to Install

Before installing AdTherm, please make sure you have installed:
- PyTorch (if using MP-NN for the surrogate construction)
- sobol-seq (can install with `pip install sobol_seq`)

Install the latest version of AdTherm by cloning the source code using Git. 

- `git clone https://github.com/franklingoldsmith/adtherm.git`

Before using AdTherm, add the following directories to your path:
- `export PATH=$PATH:/.../AdTherm/geometry_generation/`
- `export PATH=$PATH:/.../AdTherm/integration/`

(replace `...` with your path to AdTherm)

## Workflow

Once you know which adsorbate-surface system you want to calculate anharmonic thermodynamic properties for, follow these steps:

1. Perform a geometry optimization with your preferred electronic structure code. Identify the lowest energy binding site of the adsorbate and calculate the single-point energy there. 
2. Obtain the Hessian matrix at the minimum. Since the internal bonds of molecular adsorbates are fixed during sampling and integration, the Hessian matrix that preserves the bond length needs to be obtained consistently. Use `option = 'hessian'` in the input file to obtain the required displacement structures, with the instructions in steps 4, 5, and 6. See examples in `examples/CH3OH_Cu111/training_geometries/atop_Hessian_set/` or `examples/CO_Pt111/training_geometries/fcc_Hessian_set/`. After calculating the corresponding DFT energies for the displacements and saving them in order in a trajectory, one can obtain the Hessian with the `Hessian_generation.py` script in the aforementioned example directories.
3. Repeat for other binding site types in the unit cell that you would want to include as important low-energy regions. 
4. Generate a file containing the adsorbate coordinates at the minima (one file for each minimum) and another containing the surface atom coordinates. These will be needed for the training data generation as well as to initialize the integration routine later. (See `Cu111.inp` and `H_ads.inp` in examples.)
5. Create an input file for geometry generation using the `input.inp` template as done in the example `training_geometries` directories. Make sure the names of the surface and adsorbate are consistent with the name given to the respective coordinate input files. Make sure that a template for the geometries is at hand in the run directories, `blank_POSCAR` with the correct lattice parameters etc..
6. Generate the training data by running `generate_coord.py` in the directory containing the input files. Run a single sobol run and a hessian training data generation for each minimum. Make sure to have separate directories for each training set.
7. Perform DFT calculations using the generated geometries (POSCAR files) to generate training data.
8. Use the training data to train a [Minima-Preserving Neural Network (MP-NN) surrogate](https://github.com/sandialabs/MPNN).
9. (Optional) Obtain the frequencies of the adsorbate on the surrogate in the lowest energy binding site (this is required for a consistent comparison with the HO (Harmonic Oscillator) and FT (Free Translator) models, but not necessary for the PSI results). 
10. Create an `input.inp` file for the integration as done in the example `integ` directories. Additionally, the geometry input files of the adsorbate and surface are to be present in the run directory (the same ones as for the geometry generation).
11. Use the resulting surrogate model to perform the phase space integration to obtain the partition function as well as the enthalpy increment (dH/RT), entropy (S/R) and heat capacity (Cp/R). Please be aware that currently, a unique `integration.py` script needs to be present in the run directories to account for possible inconsistencies in which potential energy surface (PES) is used and how the ML surrogate is read. Perform the integration by running `jobscript.py` from within the run directory that contains the input files for that specific system. 

How to read the output:

In `output.txt`, you get the `MC-PSI Q (without quantum correction and ZPE shift)` at each temperature value as defined by T_low, dT and T_high in `input.inp`. To get the quantum-corrected Q, please multiply the `MC-PSI Q by the Pitzer-Gwinn 3D quantum correction array (including ZPE shift)`. 

The `Harmonic Oscillator Q` refers to the quantum harmonic oscillator and is relative to the ZPE as it is (aforementioned PG correction neither to be applied to that nor `Free translator Q`).

The thermodynamic properties dH/RT, Cp/R and S/R correspond to the MC-PSI Q direct derivation and have been quantum corrected before printing.


## How to Give Feedback

Please post any issues to the [issues page](https://github.com/franklingoldsmith/adtherm/issues)

Please let us know if you intend to use AdTherm or have any questions about it (feel free to send an email to katrin_blondal@brown.edu or franklin_goldsmith@brown.edu).

## Credits
 - [ECC project](https://www.ecc-project.org/)
 - [Goldsmith group](https://www.brown.edu/Departments/Engineering/Labs/Goldsmith/index.html)
 - The approach taken in this project is inspired by a project of another student in the Goldsmith group, Dr. Xi Chen, who has now graduated: Generating rate constants for barrierless gas-phase reactions with VRC-TST using [rotd_py](https://bitbucket.org/xi_chen_1/rotd_python/src/master/)



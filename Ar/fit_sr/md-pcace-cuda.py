# *************************************************************************
# Import Statements
# *************************************************************************

# ==== global ====

import sys
import torch
torch.set_default_dtype(torch.float64)

# ==== pcace ====

from pcace.calculators import CACECalculator

# ==== ase ====

import ase
from ase import units
from ase.md import MDLogger
from ase.io import read

# ==== md ====
n_run   = 40000
n_log   = 100
n_print = 100
n_write = 100
n_steps = int(n_run/n_print)

print("md parameters:")
print(f"n_run   = {n_run}")
print(f"n_log   = {n_log}")
print(f"n_print = {n_print}")
print(f"n_write = {n_write}")
print(f"n_steps = {n_steps}")

# *************************************************************************
# Functions
# *************************************************************************

def print_energy(a):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.4feV  Ekin = %.4feV (T=%3.0fK)  '
        'Etot = %.4feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
    )

def write_frame():
    dyn.atoms.write('md.xyz', append=True)

# *************************************************************************
# Molecular Dynamics
# *************************************************************************

# ==== read initial configuration ====
print("reading intial configuration")
conf_file = 'Ar_fcc.xyz'
conf_read = read(conf_file, '0')
conf_init = conf_read.repeat((3,3,3))
nnp = torch.load(sys.argv[1], map_location=torch.device('cuda'),weights_only=False)
nnp.calc_stress = True
print("init = ",conf_init)

# create the calculator
print("creating the calculator")
atomic_energies={18: -574.498250374157}
calculator = CACECalculator(
        model_path=nnp, 
        device='cuda', 
        key_energy='energy_nnp',
        key_forces='forces_nnp',
        key_stress='stress_nnp',
        compute_stress=True,
        atomic_energies=atomic_energies
    )
conf_init.calc=calculator

# set initial velocities 
print("setting initial velocities")
temperature = float(sys.argv[2])# in K
# set temperature
ase.md.velocitydistribution.thermalize_momenta(
    atoms = conf_init,
    temperature_K = temperature,
)
# remove COM momentum
ase.md.velocitydistribution.Stationary(
    atoms = conf_init,
    preserve_temperature = True,
)
# remove net rotation
ase.md.velocitydistribution.ZeroRotation(
    atoms = conf_init,
    preserve_temperature = True,
)

# Define the NPT ensemble
print("defining npt ensemble")
ts = 1.0 * units.fs # timestep
dyn = ase.md.nose_hoover_chain.MTKNPT(
    atoms = conf_init,
    timestep = ts,
    temperature_K = temperature,
    pressure_au = 0.0 * units.GPa,
    tdamp =  500.0 * ts,
    pdamp = 5000.0 * ts,
    tchain = 3,
    pchain = 3,
    tloop = 1,
    ploop = 1,
)
dyn.attach(
        write_frame, 
        interval=n_write
)
dyn.attach(
        MDLogger(
            dyn, 
            conf_init, 
            'md.log', 
            header=True,
            stress=True,
            peratom=False, 
            mode="w"), 
        interval=n_log
    )

# Run the MD simulation
print("running md simulation")
for step in range(n_steps):
    print_energy(a=conf_init)
    dyn.run(n_print)


#!/usr/bin/env python
# coding: utf-8

#**************************************************************************
# Import Statements
#**************************************************************************

#==== standard libraries ====

import torch
import sys
import numpy as np

#==== pcace ====

import pcace
# torch_geometric
from pcace import torch_geometric
# data
from pcace.data import Molecule
from pcace.data.load_data import read_atoms_xyz

#**************************************************************************
# Global Variables
#**************************************************************************

# general
seed = 42
torch.set_default_dtype(torch.float64)

# device
device_str = 'cpu'
device = pcace.tools.init_device(device_str)
print(f"device: {device}")

# model
model = torch.load(sys.argv[1],weights_only=False)
model.to(device_str)
#rc = float(model.cutoff.rc.detach().to('cpu').numpy()) # cutoff radius
rc = 6.0

# elements, keys
z_list = [18]
atomic_energies={18: -574.498250374157} 
key_data = {
    'energy': 'energy_ref',
    'forces': 'forces_ref'
}

print("=========================================================")
print( "Global Variables")
print( "Elements:")
print(f"z_list        = {z_list}")
print(f"atom_energies = {atomic_energies}")
print(f"random_seed   = {seed}")
print(f"key_data      = {key_data}")
print("=========================================================")

#**************************************************************************
# Write Data
#**************************************************************************

print("computing energy")
configs, key_data = read_atoms_xyz(
    path_file = sys.argv[2],
    key_data = key_data
)
writer_energy=open("pcace_energy.dat","w")
writer_energy.write("#index \
    energy_ref \
    energy_nnp \
    \n")
writer_force=open("pcace_forces.dat","w")
writer_force.write("#index \
    forces_ref \
    forces_nnp \
    \n")
index=1
for atoms in configs:
    # compute natoms
    natoms=len(atoms.positions) 
    # read energy/force
    energy_ref = atoms.info.get(key_data["energy"], None)
    forces_ref = atoms.arrays.get(key_data["forces"], None)
    forces_total_ref =np.sqrt(np.sum(forces_ref**2))
    # compute shifted reference energy
    energy_zero = sum(atomic_energies.get(Z, 0) for Z in atoms.get_atomic_numbers())
    energy_ref_shift = energy_ref - energy_zero
    # compute cace energy
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
           Molecule.from_atoms(
                atoms,
                cutoff=rc,
                key_data=key_data
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch_base = next(iter(data_loader)).to(device)
    batch = batch_base.clone()
    # store output
    kwargs = dict(
        training=True,
        compute_forces=True,
        compute_virials=True,
        compute_stress=True,
    )
    output = model(batch.to_dict(), **kwargs)
    energy_nnp = output["energy_nnp"].detach().to('cpu').numpy()[0]
    forces_nnp = output["forces_nnp"].detach().to('cpu').numpy()
    forces_nnp_total = np.sqrt(np.sum(forces_nnp**2))
    # write
    writer_energy.write(f"{index} \
        {energy_ref_shift/natoms} \
        {energy_nnp/natoms} \
    \n")
    writer_force.write(f"{index} \
        {forces_total_ref} \
        {forces_nnp_total} \
    \n")
    # increment
    index=index+1
writer_energy.close()
writer_force.close()


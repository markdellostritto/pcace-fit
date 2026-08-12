#!/usr/bin/env python
# coding: utf-8

#**************************************************************************
# Import Statements
#**************************************************************************

#==== standard libraries ====

import torch
import logging
import sys
import numpy as np

#==== ase ===

from ase.io import read

#==== pcace ====

import pcace
# mlp
from pcace.mlp import CACE
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
pcace_model = torch.load(sys.argv[1],weights_only=False)
pcace_model.to(device_str)
#rc = float(pcace_model.cutoff.rc.detach().to('cpu').numpy()) # cutoff radius
rc = 6.0

# elements, keys
z_list = [1,8]
atomic_energies={18: -574.498250374157} 
key_data = {
        'energy': 'energy_ref',
        'forces': 'forces_ref'
}

print("=========================================================")
print("Global Variables")
print("Elements:")
print("z_list        = ",z_list)
print("atom_energies = ",atomic_energies)
print(f"random_seed  = {seed}")
print("key_data      = ",key_data)
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
writer_energy.write("#index energy_ref energy_pcace\n")
writer_force=open("pcace_force.dat","w")
writer_force.write("#index force_ref force_pcace\n")
index=1
for atoms in configs:
    # compute natoms
    natoms=len(atoms.positions) 
    # read energy/force
    energy_ref = atoms.info.get(key_data["energy"], None)
    force_ref = atoms.arrays.get(key_data["forces"], None)
    force_total_ref =np.sqrt(np.sum(force_ref**2))
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
    output      = pcace_model(batch.to_dict(), **kwargs)
    energy_out  = output[pcace_model.key_energy].detach().to('cpu').numpy()[0]
    force_out   = output[pcace_model.key_forces].detach().to('cpu').numpy()
    force_total = np.sqrt(np.sum(force_out**2))
    # write
    writer_energy.write(f"{index} {energy_ref_shift/natoms} {energy_out/natoms}\n")
    writer_force.write(f"{index} {force_total_ref} {force_total}\n")
    # increment
    index=index+1
writer_energy.close()
writer_force.close()


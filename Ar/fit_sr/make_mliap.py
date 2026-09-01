#!/usr/bin/env python
# coding: utf-8

"""
    This small script is intended to convert a PCACE model from a Pytorch binary
    to a model that conforms to the LAMMPS MLIAP format.
    This script thus reads in a model and then writes the MLIAP format 
    to the file "mliap.pt".
"""

#**************************************************************************
# Import Statements
#**************************************************************************

#==== standard libraries ====

import torch
import sys

#==== pcace ====

import pcace
# calculators
from pcace.calculators import MLIAP_PCACE

#**************************************************************************
# Global Variables
#**************************************************************************

# general
torch.set_default_dtype(torch.float64)

# device
device_str = 'cuda'
device = pcace.tools.init_device(device_str)
print(f"device: {device}")

#**************************************************************************
# Main
#**************************************************************************

# read model
pcace_model = torch.load(sys.argv[1],weights_only=False)
pcace_model.to(device_str)

# make lammps class
lammps_class = MLIAP_PCACE(pcace_model)

# write lammps class
torch.save(lammps_class, "mliap.pt")


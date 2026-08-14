#!/usr/bin/env python
# coding: utf-8

#**************************************************************************
# Import Statements
#**************************************************************************

#==== standard libraries ====

import sys
import torch
import logging

#==== ase ===

from ase.io import read

#==== pcace ====

import pcace
# basis
from pcace.basis import CutoffCos, CutoffPoly3
from pcace.basis import RadialBesselJ
from pcace.basis import AngularBasis
# ml
from pcace.ml import LossMSE, LossHuber, LossAsinh
from pcace.ml import NormT
from pcace.ml import Metrics
# optimization
from pcace.opt import TrainingTask
# mlp
from pcace.mlp import CACE
from pcace.mlp import ANN_SR, ANN_Pauli_Gauss, ANN_LDamp_Long
from pcace.mlp import NNP

#**************************************************************************
# Global Variables
#**************************************************************************

# == double type ==
torch.set_default_dtype(torch.float64)
# == logging ==
pcace.tools.setup_logger(level='INFO')
# == elements ==
# list of atomic numbers
z_list = [18] 
# dictionary pairing atomic numbers 
# with zero-point energies (i.e. isolated atom)
atomic_energies={18: -574.498250374157}
# == basis ==
rc = 6.0 # cutoff radius
nr = 6 # number of radial functions
dre = 6 # dimension - radial embedding
l_max = 3 # max angular momentum
order_max = 2 # max body order
d_node_embed = 1 # node embedding dimension
cutoff_fn = CutoffCos(rc=rc) # cutoff function
radial_basis = RadialBesselJ(rc=rc, nr=nr, train=True) # radial basis
# == nnh ==
n_hidden = [12,12] # number of hidden nodes
activation = torch.nn.SiLU() # activation function
# == device ==
use_device = 'cpu' 
device = pcace.tools.init_device(use_device)
print(f"device: {use_device}")
# == loss ==
lw = 1.0e-2
loss_fn_energy = LossAsinh(a=lw)
loss_fn_force  = LossHuber(a=lw)
# == training ==
valid_fraction = 0.1
batch_size = 4
seed = 42
key_data = {
        'energy': 'energy_ref',
        'forces': 'forces_ref'
}
nepoch = 200

print("=========================================================")
print( "Global Variables")
print( "Elements:")
print(f"z_list        = {z_list}")
print(f"atom_energies = {atomic_energies}")
print(f"d_node_embed  = {d_node_embed}")
print( "Basis:")
print(f"cutoff_radius = {rc}")
print(f"num_radial    = {nr}")
print(f"dim_radial_e  = {dre}")
print(f"l_max         = {l_max}")
print(f"order_max     = {order_max}")
print( "NNH:")
print(f"n_hidden      = {n_hidden}")
print(f"activation    = {activation}")
print( "Training:")
print(f"nepoch        = {nepoch}")
print(f"valid_frac    = {valid_fraction}")
print(f"batch_size    = {batch_size}")
print(f"random_seed   = {seed}")
print(f"key_data      = {key_data}")
print("=========================================================")

#**************************************************************************
# Data
#**************************************************************************

#==== load the data set ====
print("loading the data set")

# read data
path_data = sys.argv[1]
subset = pcace.data.read_dataset_xyz(
    # atom properties
    cutoff = rc,
    atomic_energies = atomic_energies,
    # paths
    path_train = path_data,
    path_valid = None,
    path_test  = None,
    # optimization
    valid_fraction = valid_fraction,
    seed = seed,
    key_data = key_data,
)
print(f"ntrain = {len(subset.train)}")
print(f"nval   = {len(subset.valid)}")

#==== create the data loaders ====

print("Creating the data loaders")

loader_train = pcace.data.load_data_loader(
    collection = subset,
    data_type = "train",
    batch_size = batch_size,
)
loader_valid = pcace.data.load_data_loader(
    collection = subset,
    data_type = "valid",
    batch_size = batch_size,
)

#**************************************************************************
# Representation
#**************************************************************************

print("Creating the CACE Model")

cace_rep = CACE(
    # atomic numbers
    z_list = z_list,
    # body order
    order = order_max,
    # basis
    cutoff = cutoff_fn,
    radial = radial_basis,
    angular = AngularBasis(l_max),
    # node/edge encoding/embedding
    dim_node_embed = d_node_embed,
    # radial embedding
    dim_radial_embed = dre,
    # message passing
    avg_num_neighbors=1,
    device=device,
)
for param in cace_rep.parameters(): param.requires_grad = True
cace_rep.to(device)
logging.info(f"Representation: {cace_rep}")

#**************************************************************************
# Atomic Neural Network - Short Range
#**************************************************************************

ann_sr = ANN_SR(
    # neural network
    n_in = cace_rep.n_input,
    n_out = 1,
    n_hidden = n_hidden,
    activation = activation,
    skip = True,
)

#**************************************************************************
# Atomic Neural Network - Pauli
#**************************************************************************

ann_pauli = ANN_Pauli_Gauss(
    # neural network
    n_in = cace_rep.n_input,
    n_out = 1,
    n_hidden = [12],
    activation = torch.nn.Softplus(),
    skip = False,
    linout = False,
    # radii - covalent
    radii = {18: 1.06}, 
)

#**************************************************************************
# Atomic Neural Network - London (Damped)
#**************************************************************************

ann_ldamp = ANN_LDamp_Long(
    # neural network
    n_in = cace_rep.n_input,
    n_out = 1,
    n_hidden = [12],
    activation = torch.nn.Softplus(),
    skip = False,
    linout = False,
    # radii - vdw
    radii = {18: 3.81},
    # kspace
    prec = 1.0e-6,
    rc = rc,
)

#**************************************************************************
# Neural Network Potential 
#**************************************************************************

print("Creating the NNP")

nnp = NNP(
    rep = cace_rep,
    annl = torch.nn.ModuleList([
        ann_pauli,
        ann_ldamp,
        ann_sr
    ])
)

#**************************************************************************
# Loss/Metrics
#**************************************************************************

print("Creating loss functions")

# loss - energy
loss_energy = pcace.ml.LossMap(
    name_target  = 'energy',
    name_predict = nnp.key_energy,
    loss_fn = loss_fn_energy,
    loss_weight = 1.0,
    normT = NormT.LINEAR,
)

# loss - force
loss_force = pcace.ml.LossMap(
    name_target = 'forces',
    name_predict = nnp.key_forces,
    loss_fn = loss_fn_force,
    loss_weight = 1.0,
    normT = NormT.NONE,
)

print(loss_energy)
print(loss_force)

print("Creating metric functions")

# metric - energy
metric_energy = Metrics(
    name_target  = 'energy',
    name_predict = nnp.key_energy,
    name_metric  = 'e/atom',
    per_atom     = True
)

# metric - force
metric_force = Metrics(
    name_target  = 'forces',
    name_predict = nnp.key_forces,
    name_metric  = 'f'
)

print(metric_energy)
print(metric_force)

#**************************************************************************
# Training
#**************************************************************************

print("Creating optimizer")

# ==== optimizer ====
#optimizer=torch.optim.Adam
optimizer=torch.optim.NAdam
#optimizer=Yogi
optimizer_args = {
    'lr': 1e-2,
}

# ==== scheduler ====
# -- reduce-on-plateau --
#scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau
#scheduler_args = {
#    'mode': 'min',
#    'factor': 0.1,
#    'patience': 10,
#    'threshold': 0.0001,
#    'threshold_mode': 'rel',
#    'cooldown': 0,
#    'min_lr': 1.0e-5,
#    'eps': 1.0e-8,
#}
# -- exponential --
#scheduler=torch.optim.lr_scheduler.ExponentialLR
#scheduler_args = {
#    'gamma': 0.95
#}
# -- step --
#scheduler=torch.optim.lr_scheduler.StepLR
#scheduler_args = {'step_size': 20, 'gamma': 0.5} # step
# -- one-cycle --
scheduler=torch.optim.lr_scheduler.OneCycleLR
scheduler_args = {
    'max_lr' : 1.0e-2,
    'total_steps' : None,
    'epochs' : nepoch,
    'steps_per_epoch' : len(loader_train),
    'pct_start' : 0.3,
    'anneal_strategy' : 'cos',
    'cycle_momentum' : True,
    'base_momentum' : 0.85,
    'max_momentum' : 0.95,
    'div_factor' : 25,
    'final_div_factor' : 40,
    'three_phase' : False,
    'last_epoch' : -1,
}

print(optimizer)
print(optimizer_args)
print(scheduler)
print(scheduler_args)

print("Creating training task - Pauli")

for param in nnp.annl[0].parameters(): # pauli
    param.requires_grad = True
for param in nnp.annl[1].parameters(): # ldamp
    param.requires_grad = False
for param in nnp.annl[2].parameters(): # sr
    param.requires_grad = False
nnp.annl[0].weight = 1.0 # pauli
nnp.annl[1].weight = 0.0 # ldamp
nnp.annl[2].weight = 0.0 # sr

task = TrainingTask(
    model=nnp,
    losses=[loss_energy, loss_force],
    metrics=[metric_energy, metric_force],
    device=device,
    optimizer_cls=optimizer,
    optimizer_args=optimizer_args,
    scheduler_cls=scheduler,
    scheduler_args=scheduler_args,
    ema=False,
    ema_decay=0.99,
    ema_start=10,
    max_grad_norm=None,
    warmup_steps=10,
)

print("Fitting the model - Pauli")

task.fit(
    loader_train,
    loader_valid,
    epochs=nepoch,
    val_stride=1,
    print_stride=1,
)

task.save_model('model1.pth')
cace_rep.to(device)

print("Creating training task - LDamp")

for param in nnp.annl[0].parameters(): # pauli
    param.requires_grad = False
for param in nnp.annl[1].parameters(): # ldamp
    param.requires_grad = True
for param in nnp.annl[2].parameters(): # sr
    param.requires_grad = False
nnp.annl[0].weight = 1.0 # pauli
nnp.annl[1].weight = 1.0 # ldamp
nnp.annl[2].weight = 0.0 # sr

task = TrainingTask(
    model=nnp,
    losses=[loss_energy, loss_force],
    metrics=[metric_energy, metric_force],
    device=device,
    optimizer_cls=optimizer,
    optimizer_args=optimizer_args,
    scheduler_cls=scheduler,
    scheduler_args=scheduler_args,
    ema=False,
    ema_decay=0.99,
    ema_start=10,
    max_grad_norm=None,
    warmup_steps=10,
)

print("Fitting the model - LDamp")

task.fit(
    loader_train,
    loader_valid,
    epochs=nepoch,
    val_stride=1,
    print_stride=1,
)

task.save_model('model2.pth')
cace_rep.to(device)

print("Creating training task - SR")

for param in nnp.annl[0].parameters(): # pauli
    param.requires_grad = False
for param in nnp.annl[1].parameters(): # ldamp
    param.requires_grad = False
for param in nnp.annl[2].parameters(): # sr
    param.requires_grad = True
nnp.annl[0].weight = 1.0 # pauli
nnp.annl[1].weight = 1.0 # ldamp
nnp.annl[2].weight = 1.0 # sr

task = TrainingTask(
    model=nnp,
    losses=[loss_energy, loss_force],
    metrics=[metric_energy, metric_force],
    device=device,
    optimizer_cls=optimizer,
    optimizer_args=optimizer_args,
    scheduler_cls=scheduler,
    scheduler_args=scheduler_args,
    ema=False,
    ema_decay=0.99,
    ema_start=10,
    max_grad_norm=None,
    warmup_steps=10,
)

print("Fitting the model - SR")

task.fit(
    loader_train,
    loader_valid,
    epochs=nepoch,
    val_stride=1,
    print_stride=1,
)

task.save_model('model3.pth')
cace_rep.to(device)

#**************************************************************************
# Finish
#**************************************************************************

logging.info(f"Finished")
trainable_params = sum(p.numel() for p in cace_rep.parameters() if p.requires_grad)
logging.info(f"Number of trainable parameters: {trainable_params}")


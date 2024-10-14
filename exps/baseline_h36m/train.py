import argparse
import os, sys
import json
import math
import numpy as np
import copy

from config import config
from model import siMLPe as Model
from datasets.h36m import H36MDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
from datasets.h36m_eval import H36MEval

from test import test

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Training params
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp-name', type=str, default=None, help='=exp name')
parser.add_argument('--seed', type=int, default=888, help='=seed')
parser.add_argument('--temporal-only', action='store_true', help='=temporal only')
parser.add_argument('--layer-norm-axis', type=str, default='spatial', help='=layernorm axis')
parser.add_argument('--with-normalization', action='store_true', help='=use layernorm')
parser.add_argument('--spatial-fc', action='store_true', help='=use only spatial fc')
parser.add_argument('--num', type=int, default=64, help='=num of blocks')
parser.add_argument('--weight', type=float, default=1., help='=loss weight')

args = parser.parse_args()

# Ensure reproducibility of experiments
torch.use_deterministic_algorithms(True)
torch.manual_seed(args.seed)

# Log metrics
acc_log = open(args.exp_name, 'a')
writer = SummaryWriter() # Tensorboard metrics

# Set config parameters
config.motion_fc_in.temporal_fc = args.temporal_only
config.motion_fc_out.temporal_fc = args.temporal_only
config.motion_mlp.norm_axis = args.layer_norm_axis
config.motion_mlp.spatial_fc_only = args.spatial_fc
config.motion_mlp.with_normalization = args.with_normalization
config.motion_mlp.num_layers = args.num

# Write log file
acc_log.write(''.join('Seed : ' + str(args.seed) + '\n'))

def get_dct_matrix(N):
    """
    Compute DCT and IDCT matrix with dim NxN to transform data
    """
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

# create DCT with dimensions of input lenght data (50)
dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)

# create tensor, load GPU and add 3rd dim (1,N,N) to dct matrices
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer) :
    """
    Reduce learning rate to min_lr after 30000 iterations
    """
    if nb_iter > 30000:
        current_lr = 1e-5
    else:
        current_lr = 3e-4

    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

def gen_velocity(m):
    """
    Generates the velocity between two frames Xt+1 - Xt
    for 1 joint: j = [t5,t6,t7,t8]
    v =  [t6 - t5, t7 - t6, t8 - t7]
    """
    dm = m[:, 1:] - m[:, :-1]
    return dm

def train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, total_iter, max_lr, min_lr) :
    """
    Do the prediction, compute loss and update params
    """
    # If deriv_input we apply DCT to input data
    if config.deriv_input:
        b,n,c = h36m_motion_input.shape
        h36m_motion_input_ = h36m_motion_input.clone()
        # Transforms input data into dct (load h36m_motion_input_ to cuda with dct_m which was prev loaded)
        h36m_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], h36m_motion_input_.cuda())
    else:
        # Remain equal since deriv_input == False
        h36m_motion_input_ = h36m_motion_input.clone()

    # Load DCT transformed data to GPU and model predicts
    motion_pred = model(h36m_motion_input_.cuda())
    # Do the inverse dct_m of the output (output and idct_m were already in GPU)
    motion_pred = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], motion_pred)

    # Check if we want an offset correction
    if config.deriv_output:
        # Load to the GPU the last frame xT (all joints time T) of each batch
        offset = h36m_motion_input[:, -1:].cuda()
        # We slice target_length frames and add the offset to the prediction (done to obtain absolute 3D points since we predicted difference with XT)
        motion_pred = motion_pred[:, :config.motion.h36m_target_length] + offset
    else:
        # Remains the same if not
        motion_pred = motion_pred[:, :config.motion.h36m_target_length]

    # --- Compute MPJPE loss --- #
    # Safe "goal" results
    b,n,c = h36m_motion_target.shape
    # First reshape is to undo flattened 22*3, 2nd reshape flattens the first 3 dims (b*n*22,3)
    motion_pred = motion_pred.reshape(b,n,22,3).reshape(-1,3)
    h36m_motion_target = h36m_motion_target.cuda().reshape(b,n,22,3).reshape(-1,3)
    # Compute L2 eucl. dist. between points and after that the mean.
    loss = torch.mean(torch.norm(motion_pred - h36m_motion_target, 2, 1))

    # If we want to use Lv (loss with velocities)
    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,22,3)
        # generate the velocity xt+1 - xt of the prediction
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = h36m_motion_target.reshape(b,n,22,3)
        # generate the velocity xt+1 - xt of the ground truth
        dmotion_gt = gen_velocity(motion_gt)
        # Compute L2 mean of the reshaped difference tensor
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        # Add losses
        loss = loss + dloss
    else:
        # again mean? unnecessary i think
        loss = loss.mean()

    # Save loss value to be able to visualize in tensorboard
    writer.add_scalar('Loss/angle', loss.detach().cpu().numpy(), nb_iter)

    # delete derivates saved in loss.backward of previous iter.
    optimizer.zero_grad()
    # accumulate derivatives of the current loss with respect to params
    loss.backward()
    # update paràmeters
    optimizer.step()
    # update optimizer and lr using update_lr_multistep
    optimizer, current_lr = update_lr_multistep(nb_iter, total_iter, max_lr, min_lr, optimizer)
    # Save current lr to tensorboard
    writer.add_scalar('LR/train', current_lr, nb_iter)

    # loss.item() returns the value of loss tensor as float.
    return loss.item(), optimizer, current_lr

# Create model instance
model = Model(config)
model.train()
model.cuda()

# define target lenght com target lenght train
config.motion.h36m_target_length = config.motion.h36m_target_length_train
# create instance of H36MDataset
dataset = H36MDataset(config, 'train', config.data_aug)

shuffle = True
# Be careful cause sampler overrides shuffle if != None
sampler = None
# create instance of dataloader with its parameters (drop last drops the last incomplete batch)
# pin_memory improves speed between CPU and GPU
dataloader = DataLoader(dataset, batch_size=config.batch_size,
                        num_workers=config.num_workers, drop_last=True,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)

# Define eval_config teh same as config with small changes
eval_config = copy.deepcopy(config)
# change h36m_target_length with h36m_target_length_eval
eval_config.motion.h36m_target_length = eval_config.motion.h36m_target_length_eval
# create instance of eval dataset (why test?)
eval_dataset = H36MEval(eval_config, 'test')

# repeat process to load eval data (no shuffle nor drop last)
shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                        num_workers=1, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)

# crete logger and stuff to add log files with config and info
ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file, 'train')
link_file(config.log_file, config.link_log_file)

print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

# load the model if a path is provided
if config.model_pth is not None :
    state_dict = torch.load(config.model_pth)
    model.load_state_dict(state_dict, strict=True)
    print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

##### ------ training ------- #####
nb_iter = 0
avg_loss = 0.
avg_lr = 0.

# until max num iters
while (nb_iter + 1) < config.cos_lr_total_iters:
    # iterates over batches of data from the dataloder
    for (h36m_motion_input, h36m_motion_target) in dataloader:

        # compute the train step and save loss, lr and opt
        loss, optimizer, current_lr = train_step(h36m_motion_input, h36m_motion_target, model, optimizer, nb_iter, config.cos_lr_total_iters, config.cos_lr_max, config.cos_lr_min)
        # save avg loss and lr to add it to the log file
        avg_loss += loss
        avg_lr += current_lr

        # every config.print_every we print and log avg_loss and avg_lr
        if (nb_iter + 1) % config.print_every ==  0 :
            print(nb_iter + 1)
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every
            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        # every config.save_every we save the model
        if (nb_iter + 1) % config.save_every ==  0 :
            torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            # eval model
            model.eval()
            # calc accuracy
            acc_tmp = test(eval_config, model, eval_dataloader)
            print(acc_tmp)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            # keep training the model
            model.train()

        # stop training when we reach max iter
        if (nb_iter + 1) == config.cos_lr_total_iters :
            break
        nb_iter += 1

writer.close()

import argparse
import os, sys
import json
import math
import numpy as np
import copy

from fontTools.misc.bezierTools import epsilon
from torch.nn import CrossEntropyLoss

from config import config
from config_classifier import config_classifier
if config.use_int_class:
    config = config_classifier
    print("Training Classifier...")
from model import siMLPe as Model
from datasets.handover import HandoverDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
from lib.datasets.handover_eval import HandoverEvalDataset
from lib.utils.handover_functions import find_intentions_mode, get_dct_matrix


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

# create DCT with dimensions of input lenght data (50)
dct_m,idct_m = get_dct_matrix(config.motion.handover_input_length_dct)

# create tensor, load GPU and add 3rd dim (1,N,N) to dct matrices
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def update_lr_multistep(nb_iter, optimizer) :
    """
    Reduce learning rate to min_lr after 30000 iterations
    """
    if nb_iter > 30000:
        current_lr = 5e-8
    else:
        current_lr = 1e-5

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

def gen_rh_distance_to_joints(motion):
    b,c,n,_ = motion.shape
    rh_motion = motion[:,:,5,:] # select rh joint
    rh_motion = rh_motion.unsqueeze(dim = 2)
    rh_motion_expanded = rh_motion.expand(-1, -1, n, -1)  # Shape: [256, 10, 9, 3]
    dist_tensor = torch.norm(motion - rh_motion_expanded, dim = 3)
    return dist_tensor

def filter_collaboration_samples(int_motion,batch_intentions):
    """
    Function created to select only samples where the subject collaborates.
    This will be used to calculate ree_loss and rh_joint_dist.

    Args:
    int_motion (torch.Tensor): A tensor of shape [batch_size, 50, 1] representing intention associated to motion data.

    Returns:
    List[int]: A list of batch indexes where all 50 values in dim 1 are equal to 0.
    """
    batch_collaborative_indexes = []
    # Check each batch in the tensor
    for batch_idx in range(len(batch_intentions)):
        if config.use_colab_loss:
            if batch_intentions[batch_idx] == 0:
                batch_collaborative_indexes.append(batch_idx)
        else:
            batch_collaborative_indexes.append(batch_idx)
    return batch_collaborative_indexes


def train_step(handover_motion_input, handover_motion_target, ree_motion_input, ree_motion_target, int_motion_input, int_motion_target, model, optimizer,nb_iter) :
    """
    Do the prediction, compute loss and update params
    """
    b,n,c = handover_motion_input.shape
    handover_motion_input_ = handover_motion_input.clone()
    # Transforms input data into dct (load handover_motion_input_ to cuda with dct_m which was prev loaded)
    handover_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.handover_input_length], handover_motion_input_.cuda())

    ree_motion_input_ = ree_motion_input.clone()
    ree_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.handover_input_length],
                                          ree_motion_input_.cuda())
    # keep only the position of the last frame
    ree_motion_input_ = ree_motion_input_[:,config.motion.handover_input_length-1,:]

    int_motion_target_ = int_motion_target.clone()
    #print(int_motion_target_.shape)
    # select mode of the intention detected in the next 10 future frames
    int_motion_target_ = find_intentions_mode(int_motion_target_)

    # Load DCT transformed data to GPU and model predicts
    motion_pred, int_class_logits,intention_pred = model(handover_motion_input_.cuda(),ree_motion_input_.cuda(),int_motion_target_.cuda())
    # Do the inverse dct_m of the output (output and idct_m were already in GPU)
    motion_pred = torch.matmul(idct_m[:, :config.motion.handover_input_length, :], motion_pred)

    # Check if we want an offset correction
    if config.deriv_output:
        # Load to the GPU the last frame xT (all joints time T) of each batch
        offset = handover_motion_input[:, -1:].cuda()
        # We slice target_length frames and add the offset to the prediction (done to obtain absolute 3D points since we predicted difference with XT)
        motion_pred = motion_pred[:, :config.motion.handover_target_length] + offset
    else:
        # Remains the same if not
        motion_pred = motion_pred[:, :config.motion.handover_target_length]

    # --- Compute MPJPE loss --- #
    # Safe "goal" results
    b,n,c = handover_motion_target.shape
    # First reshape is to undo flattened 9*3, 2nd reshape flattens the first 3 dims (b*n*9,3)
    motion_pred = motion_pred.reshape(b,n,9,3).reshape(-1,3)
    handover_motion_target = handover_motion_target.cuda().reshape(b,n,9,3).reshape(-1,3)
    # Compute L2 eucl. dist. between points and after that the mean.
    loss = torch.mean(torch.norm(motion_pred - handover_motion_target, 2, 1))
    # If we want to use Lv (loss with velocities)
    if config.use_relative_loss:
        motion_pred = motion_pred.reshape(b,n,9,3)
        # generate the velocity xt+1 - xt of the prediction
        dmotion_pred = gen_velocity(motion_pred)
        motion_gt = handover_motion_target.reshape(b,n,9,3)
        # generate the velocity xt+1 - xt of the ground truth
        dmotion_gt = gen_velocity(motion_gt)
        # Compute L2 mean of the reshaped difference tensor
        dloss = torch.mean(torch.norm((dmotion_pred - dmotion_gt).reshape(-1,3), 2, 1))
        loss += dloss
        total_loss = loss
        if config.use_relative_loss_rh:
            # focus to improve right hand velocity
            dloss_rh = torch.mean(torch.norm((dmotion_pred[:,:,[5],:] - dmotion_gt[:,:,[5],:]).reshape(-1,3), 2, 1))
            total_loss += dloss_rh

    if config.use_rh_loss:
        # Compute L2 between only Right Hand to see if adding more weight predictions improve
        motion_pred = motion_pred.reshape(b, n, 9, 3)
        motion_gt = handover_motion_target.reshape(b, n, 9, 3)
        right_hand_gt = motion_gt[:, :, 5, :]
        right_hand_pred = motion_pred[:, :, 5, :]
        rhloss = torch.mean(torch.mean(torch.norm(right_hand_pred - right_hand_gt, dim=2), dim=1), dim = 0)
        total_loss += rhloss


    if config.use_ree_loss:
        # filter out collaborative samples to compute ree_loss and use_rh_distance_joints_loss
        collaboration_samples = filter_collaboration_samples(int_motion_input,int_motion_target_)
        # Compute L2 between only Right Hand to see if adding more weight predictions improve
        motion_pred = motion_pred.reshape(b, n, 9, 3)
        motion_pred_collab = motion_pred[collaboration_samples,:,:,:]
        right_hand_pred_last_frame = motion_pred_collab[:, config.motion.handover_target_length_train-1, 5, :]
        ree_target = ree_motion_target[collaboration_samples, config.motion.handover_target_length_train-1, :]
        #print("MOTION PRED COLAB SHAPE", motion_pred_collab.shape)
        #print("right_hand_pred_last_frame:", right_hand_pred_last_frame.shape)
        #print("ree_target:",ree_target.shape)
        reeloss = torch.mean(torch.norm(right_hand_pred_last_frame - ree_target.cuda(), dim = 1), dim = 0)
        if config.use_rh_distance_joints_loss:
            # Compute L2 between the correct mean distance from RH to other joints and the predicted distance
            # Calcular distancia de RH als altres joints a pred i gt, fer una funció que ho faci.
            motion_gt = handover_motion_target.reshape(b, n, 9, 3)
            motion_gt_collab = motion_gt[collaboration_samples,:,:,:]
            #print("MOTION GT COLAB SHAPE", motion_gt_collab.shape)
            pred_dist = gen_rh_distance_to_joints(motion_pred_collab)
            gt_dist = gen_rh_distance_to_joints(motion_gt_collab)
            #print("PRED AND GT DIST", pred_dist.shape, gt_dist.shape)
            distance_joints_loss = torch.mean(abs(gt_dist-pred_dist),dim= [0,1,2])
            total_loss += 0.05*reeloss + 0.95*distance_joints_loss
        else:
            total_loss += 0.01 * reeloss


    if config.use_int_class:
        criterion = torch.nn.CrossEntropyLoss()
        ce_loss = criterion(int_class_logits,int_motion_target_.long().cuda())
        writer.add_scalar('CE loss', ce_loss.detach().cpu().numpy(), nb_iter)
        if config.only_classification:
            total_loss = ce_loss
        else:
            total_loss += ce_loss

    # Save loss value to be able to visualize in tensorboard
    writer.add_scalar('Loss/angle', total_loss.detach().cpu().numpy(), nb_iter)

    # delete derivates saved in loss.backward of previous iter.
    optimizer.zero_grad()
    # accumulate derivatives of the current loss with respect to params
    total_loss.backward()
    # update paràmeters
    optimizer.step()
    # we set lr to min when config.lr_cos_total_iter is exceeded
    if nb_iter >= config.cos_lr_total_iters:
        current_lr = config.cos_lr_min
    else: # save lr and update optimizer
        current_lr = optimizer.param_groups[0]["lr"]
        if config.cosine_lr:
            scheduler.step()
        else:
            optimizer, current_lr = update_lr_multistep(nb_iter, optimizer)
    # Save current lr to tensorboard
    writer.add_scalar('LR/train', current_lr, nb_iter)

    # loss.item() returns the value of loss tensor as float.
    return total_loss.item(), optimizer, current_lr

# Create model instance
model = Model(config)
model.train()
model.cuda()

# define target lenght com target lenght train
config.motion.handover_target_length = config.motion.handover_target_length_train
# create instance of handoverDataset
dataset = HandoverDataset(config, 'train', config.data_aug)

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
# change handover_target_length with handover_target_length_eval
eval_config.motion.handover_target_length = eval_config.motion.handover_target_length_eval
# create instance of eval dataset (why test?)
eval_dataset = HandoverEvalDataset(eval_config, 'test')

# repeat process to load eval data (no shuffle nor drop last)
shuffle = False
sampler = None
eval_dataloader = DataLoader(eval_dataset, batch_size=128,
                        num_workers=1, drop_last=False,
                        sampler=sampler, shuffle=shuffle, pin_memory=True)


# initialize optimizer model and loss layer params
optimizer = torch.optim.Adam(list(model.parameters()),
                             lr=config.cos_lr_max,
                             weight_decay=config.weight_decay)


if config.cosine_lr:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cos_lr_total_iters, eta_min=config.cos_lr_min)

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
while (nb_iter + 1) < config.total_iters:
    # iterates over batches of data from the dataloder
    for (handover_motion_input, handover_motion_target,ree_motion_input,ree_motion_target, int_motion_input, int_motion_target) in dataloader:

        # compute the train step and save loss, lr and opt
        loss, optimizer, current_lr = train_step(handover_motion_input, handover_motion_target, ree_motion_input,ree_motion_target, int_motion_input, int_motion_target ,model, optimizer, nb_iter)
        # save avg loss and lr to add it to the log file
        avg_loss += loss
        avg_lr += current_lr

#10 like training in mm
        # every config.print_every we print and log avg_loss and avg_lr
        if (nb_iter + 1) % config.print_every ==  0 :
            avg_loss = avg_loss / config.print_every
            avg_lr = avg_lr / config.print_every
            print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
            print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
            avg_loss = 0
            avg_lr = 0

        if (nb_iter + 1) % config.eval_every == 0 :
            model.eval()
            # calc loss in all timeframes
            acc_tmp, rh_loss, under_10, under_15, under_20, under_30, under_35, under_40, accuracy, f1, f1_binary = test(eval_config, model, eval_dataloader)
            avg_rh_loss = np.mean(np.array(rh_loss))
            avg_l2_body_loss = np.mean(np.array(acc_tmp))  # mean of all time frames
            print("Iteration:",nb_iter)
            print("L2_body test", round(avg_l2_body_loss,3))
            print("L2_right_hand test", round(avg_rh_loss,3))
            print("% Under 10", round(under_10, 3))
            print("% Under 15", round(under_15, 3))
            print("% Under 20", round(under_20, 3))
            print("% Under 30", round(under_30,3))
            print("% Accuracy", round(accuracy, 3))
            print("% F1 score", round(f1, 3))
            print("% Binary F1 score", round(f1_binary, 3))
            writer.add_scalar('Body Test Loss', avg_l2_body_loss, nb_iter)
            print_and_log_info(logger, f"\t Body Test loss: {avg_l2_body_loss}")
            writer.add_scalar('RH Test Loss', avg_rh_loss, nb_iter)
            print_and_log_info(logger, f"\t RH Test loss: {avg_rh_loss}")
            writer.add_scalar('F1 score', f1, nb_iter)
            print_and_log_info(logger, f"\t F1 score: {f1}")
            model.train()

        # every config.save_every we save the model
        if (nb_iter + 1) % config.save_every ==  0 :
            if config.motion_gcn_in.gcn_in:
                torch.save(model.state_dict(), config.snapshot_dir + '/model-GCN-iter-' + str(nb_iter + 1) + '.pth')
            else:
                torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
            # eval model
            model.eval()
            # calc loss
            acc_tmp, rh_loss,_,_,_,_,_,_,_,_,_ = test(eval_config, model, eval_dataloader)
            print("Body loss values", acc_tmp)
            print("RH loss values", rh_loss)
            acc_log.write(''.join(str(nb_iter + 1) + '\n'))
            line = ''
            for ii in acc_tmp:
                line += str(ii) + ' '
            line += '\n'
            acc_log.write(''.join(line))
            # keep training the model
            model.train()

        # stop training when we reach max iter
        if (nb_iter + 1) == config.total_iters :
            total_params = sum(p.numel() for p in model.parameters())
            print_and_log_info(logger, f"\t Total number of parameters: {total_params}")
            print(f"Total number of parameters: {total_params}")
            break
        nb_iter += 1

writer.close()

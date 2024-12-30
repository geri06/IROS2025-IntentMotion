import argparse
import os, sys
import json
import math
import numpy as np
import copy
import itertools

from config import config
from model import siMLPe as Model
from datasets.handover import HandoverDataset
from utils.logger import get_logger, print_and_log_info
from utils.pyt_utils import link_file, ensure_dir
from lib.datasets.handover_eval import HandoverEvalDataset
from train import train_step
from lib.utils.handover_functions import get_dct_matrix


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


def subject_splits():
    subjects = []

    subjects += np.loadtxt(
        os.path.join(config.handover_anno_dir.replace('handover', ''), "handover_train.txt"), dtype=str
    ).tolist()

    test_subj = open(
        os.path.join(config.handover_anno_dir.replace('handover', ''), "handover_test.txt"), 'r'
    ).readlines()[0].strip()

    subjects.append(test_subj)
    # Generate all train/test splits: Leave-One-Subject-Out or Leave-Two-Subjects-Out
    num_to_leave_out = 1  # You can change this to 2 for Leave-Two-Subjects-Out, etc.
    splits = list(itertools.combinations(subjects, num_to_leave_out))
    return splits, subjects


### ------------ Training with cross validation ------------- ###
metrics = {"L2_body": [], "L2_right_hand":[], "under_0.10m":[], "under_0.15m":[],"under_0.20m":[], "under_0.30m":[], "accuracy":[], "f1_score":[]}

# crete logger and stuff to add log files with config and info

ensure_dir(config.snapshot_dir)
logger = get_logger(config.log_file_leave_one_out, 'train')
link_file(config.log_file_leave_one_out, config.link_log_file)
print_and_log_info(logger, json.dumps(config, indent=4, sort_keys=True))

# Obtain subjects combinations
splits, all_subjects = subject_splits()

for split in splits:
    # Define train and test sets
    test_subjects = list(split)
    train_subjects = [s for s in all_subjects if s not in test_subjects]
    # Create text files for the current split
    with open("data/handover_train.txt", "w") as f:
        f.write("\n".join(train_subjects))
    with open("data/handover_test.txt", "w") as f:
        f.write("\n".join(test_subjects))

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

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.cos_lr_max,
                                 weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cos_lr_total_iters,
                                                           eta_min=config.cos_lr_min)

    # load the model if a path is provided
    if config.model_pth is not None:
        state_dict = torch.load(config.model_pth)
        model.load_state_dict(state_dict, strict=True)
        print_and_log_info(logger, "Loading model path from {} ".format(config.model_pth))

    print_and_log_info(logger, "Testing with subject {} ".format(test_subjects))
    print_and_log_info(logger, "Training with subjects {} ".format(train_subjects))

    ##### ------ training ------- #####
    nb_iter = 0
    avg_loss = 0.
    avg_lr = 0.

    # until max num iters
    while (nb_iter + 1) < config.total_iters:
        # iterates over batches of data from the dataloder
        for (handover_motion_input, handover_motion_target, ree_motion_input, ree_motion_target, int_motion_input,
             int_motion_target) in dataloader:

            # compute the train step and save loss, lr and opt
            loss, optimizer, current_lr = train_step(handover_motion_input, handover_motion_target, ree_motion_input,ree_motion_target, int_motion_input, int_motion_target ,model, optimizer, nb_iter)
            # save avg loss and lr to add it to the log file
            avg_loss += loss
            avg_lr += current_lr

            # 10 like training in mm
            # every config.print_every we print and log avg_loss and avg_lr
            if (nb_iter + 1) % config.print_every == 0:
                avg_loss = avg_loss / config.print_every
                avg_lr = avg_lr / config.print_every
                print_and_log_info(logger, "Iter {} Summary: ".format(nb_iter + 1))
                print_and_log_info(logger, f"\t lr: {avg_lr} \t Training loss: {avg_loss}")
                avg_loss = 0
                avg_lr = 0

            if (nb_iter + 1) % config.eval_every == 0:
                model.eval()
                # calc loss in all timeframes
                acc_tmp, rh_loss, under_10, under_15, under_20, under_30, accuracy, f1, f1_binary = test(eval_config, model,
                                                                                              eval_dataloader)
                avg_rh_loss = np.mean(np.array(rh_loss))
                avg_l2_body_loss = np.mean(np.array(acc_tmp))  # mean of all time frames
                print("Iteration:", nb_iter)
                print("L2_body test", round(avg_l2_body_loss, 3))
                print("L2_right_hand test", round(avg_rh_loss, 3))
                print("% Under 10", round(under_10, 3))
                print("% Under 15", round(under_15, 3))
                print("% Under 20", round(under_20, 3))
                print("% Under 30", round(under_30, 3))
                print("% Accuracy", round(accuracy, 3))
                print("% F1 score", round(f1, 3))
                writer.add_scalar('Body Test Loss', avg_l2_body_loss, nb_iter)
                print_and_log_info(logger, f"\t Body Test loss: {avg_l2_body_loss}")
                writer.add_scalar('RH Test Loss', avg_rh_loss, nb_iter)
                print_and_log_info(logger, f"\t RH Test loss: {avg_rh_loss}")
                writer.add_scalar('F1 score', f1, nb_iter)
                print_and_log_info(logger, f"\t F1 score: {f1}")
                model.train()

            # every config.save_every we save the model
            if (nb_iter + 1) % config.save_every == 0:
                if config.motion_gcn_in.gcn_in:
                    torch.save(model.state_dict(), config.snapshot_dir + '/model-GCN-iter-' + str(nb_iter + 1) + '.pth')
                else:
                    torch.save(model.state_dict(), config.snapshot_dir + '/model-iter-' + str(nb_iter + 1) + '.pth')
                # eval model
                model.eval()
                # calc loss
                acc_tmp, rh_loss, _, _, _, _, _, _,_ = test(eval_config, model, eval_dataloader)
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
            if (nb_iter + 1) == config.total_iters:
                total_params = sum(p.numel() for p in model.parameters())
                print_and_log_info(logger, f"\t Total number of parameters: {total_params}")
                print(f"Total number of parameters: {total_params}")

                # Save metrics
                metrics["L2_body"].append(np.mean(np.array(acc_tmp)))
                metrics["L2_right_hand"].append(np.mean(np.array(rh_loss)))
                metrics["under_0.10m"].append(under_10)
                metrics["under_0.15m"].append(under_15)
                metrics["under_0.20m"].append(under_20)
                metrics["under_0.30m"].append(under_30)
                metrics["accuracy"].append(accuracy)
                metrics["f1_score"].append(f1)
                break
            nb_iter += 1

    writer.close()

print_and_log_info(logger, "Mean L2_body is {}".format(sum(metrics["L2_body"])/len(metrics["L2_body"])))
print_and_log_info(logger, "Mean L2_right_hand is {}".format(sum(metrics["L2_right_hand"])/len(metrics["L2_right_hand"])))
print_and_log_info(logger, "Under 0.10 is {}".format(sum(metrics["under_0.10m"])/len(metrics["under_0.10m"])))
print_and_log_info(logger, "Under 0.15 is {}".format(sum(metrics["under_0.15m"])/len(metrics["under_0.15m"])))
print_and_log_info(logger, "Under 0.20 is {}".format(sum(metrics["under_0.20m"])/len(metrics["under_0.20m"])))
print_and_log_info(logger, "Under 0.30 is {}".format(sum(metrics["under_0.30m"])/len(metrics["under_0.30m"])))
print_and_log_info(logger, "Intention Accuracy is {}".format(sum(metrics["accuracy"])/len(metrics["accuracy"])))
print_and_log_info(logger, "F1 Score is {}".format(sum(metrics["f1_score"])/len(metrics["f1_score"])))

print_and_log_info(logger, "Metrics are {}".format(metrics))
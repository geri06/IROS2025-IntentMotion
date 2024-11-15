import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config  import config
from model import siMLPe as Model
from datasets.handover_eval import HandoverEvalDataset
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch
from lib.utils.loss import L2_right_hand, L2_body, quality_metrics

import torch
from torch.utils.data import DataLoader

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

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
dct_m,idct_m = get_dct_matrix(config.motion.handover_input_length_dct)
# create tensor, load GPU and add 3rd dim (1,N,N) to dct matrices
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

# pbar is == dataloader
def regress_pred(model, pbar, num_samples, m_p3d_handover, right_hand_loss):
    """
    Do the prediction of the data and compute mean loss per joint for each time frame
    """
    under_30 = []
    under_20 = []
    under_10 = []
    for (motion_input, motion_target, ree_motion_input, ree_motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,n,c = motion_input.shape
        # num samples updated adding batch size
        num_samples += b
        outputs = []
        # how many frames are predicted in one forward pass
        step = config.motion.handover_target_length_train
        # if 25 or more is 1
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            # without gradients, useful for inference
            with torch.no_grad():
                # if we want dct encoding
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.handover_input_length], motion_input_.cuda())
                    if config.motion_ree.ree_cond:
                        ree_motion_input_ = ree_motion_input.clone()
                        ree_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.handover_input_length],
                                                         ree_motion_input_.cuda())
                    else:
                        ree_motion_input_ = torch.empty(0)

                else:
                    motion_input_ = motion_input.clone()
                    ree_motion_input_ = None
                output = model(motion_input_,ree_motion_input_)
                # transform output using idct_m for the rows of, handover_input_length. Then we slice to extract the first step frames of the result.
                output = torch.matmul(idct_m[:, :config.motion.handover_input_length, :], output)[:, :step, :]
                # if deriv output
                if config.deriv_output:
                    # we add the input last frame tensor in the step frames predicted from the output (cause to displacement prediction)
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            # reshape output to be (b,step,27), for some reason is done in 2 lines
            output = output.reshape(-1, 9*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            # delete the first step frames in input and add the output step frames at the end.
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)

        # concatenate outputs and keep the first 25
        motion_pred = torch.cat(outputs, axis=1)[:,:25]
        # use detach to avoid this tensor being tracked with gradient computations
        motion_target = motion_target.detach()
        b,n,c = motion_target.shape

        motion_gt = motion_target.clone()
        motion_gt = motion_gt.reshape(b, n, -1, 3)
        motion_pred = motion_pred.reshape(b, n, -1, 3)
        motion_pred = motion_pred.detach().cpu()

        # compute L2 distance between joints pred and goal, compute mean of joints diff in each time frame, sum the values of each time frame in each batch.
        mpjpe_p3d_handover = L2_body(motion_gt, motion_pred)
        # Compute % of predicted joints under 0.2m
        metrics = quality_metrics(motion_gt, motion_pred)
        under_10.append(metrics[0])
        under_20.append(metrics[1])
        under_30.append(metrics[2])

        # accumulate loss for each batch of data
        m_p3d_handover += mpjpe_p3d_handover.cpu().numpy()
        right_hand_loss_batch = L2_right_hand(motion_gt, motion_pred)
        right_hand_loss += right_hand_loss_batch.cpu().numpy()
    # compute mean loss diving by the total number of batches giving the mean loss error per timestep
    m_p3d_handover = m_p3d_handover / num_samples
    u10 = np.array(under_10).mean() * 100
    u20 = np.array(under_20).mean() * 100
    u30 = np.array(under_30).mean()*100
    right_hand_loss = right_hand_loss / num_samples
    return m_p3d_handover, right_hand_loss, u10, u20, u30

def test(config, model, dataloader) :

    m_p3d_handover = np.zeros([config.motion.handover_target_length])
    right_hand_loss = np.zeros([config.motion.handover_target_length])
    titles = np.array(range(config.motion.handover_target_length)) + 1
    num_samples = 0

    pbar = dataloader
    m_p3d_handover, right_hand_loss, under_10, under_20, under_30  = regress_pred(model, pbar, num_samples, m_p3d_handover,right_hand_loss)

    # This returns a dictionary with the correspondant loss to each time frame in results time frames
    ret = {}
    for j in range(config.motion.handover_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_handover[j], m_p3d_handover[j]]
    return [round(ret[key][0], 2) for key in results_keys], right_hand_loss, under_10, under_20, under_30

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.handover_target_length = config.motion.handover_target_length_eval
    dataset = HandoverEvalDataset(config,'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    print(test(config, model, dataloader))


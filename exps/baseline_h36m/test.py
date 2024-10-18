import argparse
import os, sys
from scipy.spatial.transform import Rotation as R

import numpy as np
from config  import config
from model import siMLPe as Model
from datasets.h36m_eval import H36MEval
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

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
dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
# create tensor, load GPU and add 3rd dim (1,N,N) to dct matrices
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

# pbar is == dataloader
def regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36):
    """
    Do the prediction of the data and compute mean loss per joint for each time frame
    """
    # ignored or mapped to other joints
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

    for (motion_input, motion_target) in pbar:
        motion_input = motion_input.cuda()
        b,n,c,_ = motion_input.shape
        # num samples updated adding batch size
        num_samples += b

        motion_input = motion_input.reshape(b, n, 32, 3)
        # keep only the joints used and reshape to (b,n,22*3)
        motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)
        outputs = []
        # how many frames are predicted in one forward pass
        step = config.motion.h36m_target_length_train
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
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_)
                # transform output using idct_m for the rows of, h36m_input_length. Then we slice to extract the first step frames of the result.
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                # if deriv output
                if config.deriv_output:
                    # we add the input last frame tensor in the step frames predicted from the output (cause to displacement prediction)
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            # reshape output to be (b,step,66), for some reason is done in 2 lines
            output = output.reshape(-1, 22*3)
            output = output.reshape(b,step,-1)
            outputs.append(output)
            # delete the first step frames in input and add the output step frames at the end.
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        # concatenate outputs and keep the first 25
        motion_pred = torch.cat(outputs, axis=1)[:,:25]

        # use detach to avoid this tensor being tracked with gradient computations
        motion_target = motion_target.detach()
        b,n,c,_ = motion_target.shape

        motion_gt = motion_target.clone()

        # we make motion pred to have values of motion target in not used joints and predicted values on the used joints
        # I think we can delete this lines
        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b,n,22,3)
        motion_pred = motion_target.clone().reshape(b,n,32,3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        # we modify again motion pred with predicted values in used joints and gt values in the others
        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        # we set values of joints to ignore with values of joints_equal
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]
        # compute L2 distance between joints pred and goal, compute mean of joints diff in each time frame, sum the values of each time frame in each batch.
        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred*1000 - motion_gt*1000, dim=3), dim=2), dim=0)
        # accumulate loss for each batch of data
        m_p3d_h36 += mpjpe_p3d_h36.cpu().numpy()
    # compute mean loss diving by the total number of batches giving the mean loss error per timestep
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36

def test(config, model, dataloader) :

    m_p3d_h36 = np.zeros([config.motion.h36m_target_length])
    titles = np.array(range(config.motion.h36m_target_length)) + 1
    joint_used_xyz = np.array([2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,21,22,25,26,27,29,30]).astype(np.int64)
    num_samples = 0

    pbar = dataloader
    m_p3d_h36 = regress_pred(model, pbar, num_samples, joint_used_xyz, m_p3d_h36)

    # This returns a dictionary with the correspondant loss to each time frame in results time frames
    ret = {}
    for j in range(config.motion.h36m_target_length):
        ret["#{:d}".format(titles[j])] = [m_p3d_h36[j], m_p3d_h36[j]]
    return [round(ret[key][0], 1) for key in results_keys],

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    dataset = H36MEval(config, 'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)

    print(test(config, model, dataloader))


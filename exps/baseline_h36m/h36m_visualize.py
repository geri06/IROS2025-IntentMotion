#!/usr/bin/env python
# coding: utf-8
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from lib.datasets.h36m_eval import H36MEval
from config  import config
from model import siMLPe as Model

"""Code adapted from STSGCN Repo: https://github.com/FraLuca/STSGCN/blob/main/utils/h36_3d_viz.py"""

def create_pose(ax, plots, vals, pred=True, update=False):
    # h36m 32 joints(full)
    connect = [
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10),
        (0, 1), (0, 6),
        (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
        (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (24, 25), (24, 17),
        (24, 14), (14, 15)
    ]
    LR = [
        False, True, True, True, True,
        True, False, False, False, False,
        False, True, True, True, True,
        True, True, False, False, False,
        False, False, False, False, True,
        False, True, True, True, True,
        True, True
    ]

    # Start and endpoints of our representation
    I = np.array([touple[0] for touple in connect])
    J = np.array([touple[1] for touple in connect])
    # Left / right indicator
    LR = np.array([LR[a] or LR[b] for a, b in connect])
    if pred:
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#383838"

    for i in np.arange(len(I)):
        x = np.array([vals[I[i], 0], vals[J[i], 0]])
        z = np.array([vals[I[i], 1], vals[J[i], 1]])
        y = np.array([vals[I[i], 2], vals[J[i], 2]])

        # Debugging print to check the values
        # print(f"Updating line {i}: x={x}, y={y}, z={z}")

        if not update:

            if i == 0:
                plots.append(ax.plot(x, y, z, lw=2, linestyle='--', c=lcolor if LR[i] else rcolor,
                                     label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2, linestyle='--', c=lcolor if LR[i] else rcolor))

        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)

    return plots

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

def data_to_viz(model, pbar, num_samples, joint_used_xyz, n_viz):
    import random
    """
    regress_pred() from test.py modified to return pred_data and gt_data
    """
    # ignored or mapped to other joints
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)

    for cnt, (motion_input, motion_target) in enumerate(pbar):
        motion_input = motion_input.cuda()
        b, n, c, _ = motion_input.shape
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
                    output = output + motion_input[:, -1:, :].repeat(1, step, 1)

            # reshape output to be (b,step,66), for some reason is done in 2 lines
            output = output.reshape(-1, 22 * 3)
            output = output.reshape(b, step, -1)
            outputs.append(output)
            # delete the first step frames in input and add the output step frames at the end.
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        # concatenate outputs and keep the first 25
        motion_pred = torch.cat(outputs, axis=1)[:, :25]

        # use detach to avoid this tensor being tracked with gradient computations
        motion_target = motion_target.detach()
        b, n, c, _ = motion_target.shape

        motion_gt = motion_target.clone()

        # we make motion pred to have values of motion target in not used joints and predicted values on the used joints
        # I think we can delete this lines
        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b, n, 22, 3)
        motion_pred = motion_target.clone().reshape(b, n, 32, 3)
        motion_pred[:, :, joint_used_xyz] = pred_rot

        # we modify again motion pred with predicted values in used joints and gt values in the others
        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        # we set values of joints to ignore with values of joints_equal
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]
        # compute L2 distance between joints pred and goal, compute mean of joints diff in each time frame, sum the values of each time frame in each batch.
        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred * 1000 - motion_gt * 1000, dim=3), dim=2), dim=0)

        data_pred = torch.squeeze(motion_pred, 0).cpu().data.numpy() # in meters
        data_gt = torch.squeeze(motion_target, 0).cpu().data.numpy()

        i = random.randint(1, 128)

        data_pred = data_pred[i]
        data_gt = data_gt[i]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=-40)
        vals = np.zeros((32, 3))  # or joints_to_consider
        gt_plots = []
        pred_plots = []

        gt_plots = create_pose(ax, gt_plots, vals, pred=False, update=False)
        pred_plots = create_pose(ax, pred_plots, vals, pred=True, update=False)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc='lower left')

        ax.set_xlim3d([-1, 1.5])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1, 1.5])
        ax.set_ylabel('Y')

        ax.set_zlim3d([0.0, 1.5])
        ax.set_zlabel('Z')
        ax.set_title('mean loss in mm is: ' + str(round(mpjpe_p3d_h36[-1].item()/b, 4)) + ' for action : ' + str(config.actions_to_load) + ' for ' + str(
            25) + ' frames')

        line_anim = animation.FuncAnimation(fig, update, 25, fargs=(data_gt, data_pred, gt_plots, pred_plots,
                                                                          fig, ax), interval=70, blit=False)
        plt.show()

        line_anim.save('./visualizations/pred{}/human_viz{}.gif'.format(25, i), writer='pillow')

        if cnt == n_viz - 1:
            break

# ax.legend(loc='lower left')


# In[11]:


def update(num, data_gt, data_pred, plots_gt, plots_pred, fig, ax):
    gt_vals = data_gt[num]
    pred_vals = data_pred[num]
    plots_gt = create_pose(ax, plots_gt, gt_vals, pred=False, update=True)
    plots_pred = create_pose(ax, plots_pred, pred_vals, pred=True, update=True)

    r = 1
    xroot, zroot, yroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
    ax.set_xlim3d([-r + xroot, r + xroot])
    ax.set_ylim3d([-r + yroot, r + yroot])
    ax.set_zlim3d([-r + zroot, r + zroot])

    # ax.set_title('pose at time frame: '+str(num))
    # ax.set_aspect('equal')

    return plots_gt, plots_pred


# %%

def visualize(model_pth,n_viz):
    model = Model(config)
    config.motion.h36m_target_length = config.motion.h36m_target_length_eval
    state_dict = torch.load(model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()
    actions = config.actions_to_load
    dataset = H36MEval(config, 'test', actions)

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)
    joint_used_xyz = np.array(
            [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30]).astype(
            np.int64)
    num_samples = 0
    pbar = dataloader
    data_to_viz(model, pbar, num_samples, joint_used_xyz, n_viz)

    ### ----- Create visualization of data and prediction ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()
    visualize(args.model_pth,1)



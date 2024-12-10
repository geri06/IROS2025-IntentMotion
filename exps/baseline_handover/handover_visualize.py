#!/usr/bin/env python
# coding: utf-8
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from config  import config
from lib.datasets.handover_eval import HandoverEvalDataset
from model import siMLPe as Model

"""Code adapted from STSGCN Repo: https://github.com/FraLuca/STSGCN/blob/main/utils/h36_3d_viz.py"""

def create_pose(ax, plots, vals, pred=True, update=False):
    # handover 32 joints(full)
    connect = [
        (0, 1),  # nose to left shoulder
        (0, 2),  # nose to right shoulder
        (1, 2),  # left shoulder to right shoulder
        (1, 3),  # left shoulder to left elbow
        (2, 4),  # right shoulder to right elbow
        (3, 5),  # left elbow to left wrist
        (4, 6),  # right elbow to right wrist
        (1, 7),  # left shoulder to left hip
        (2, 8),  # right shoulder to right hip
        (7, 8)  # left hip to right hip
    ]

    LR = [
        True, False, True, False,
        True, False, True, False,
        True
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
        y = np.array([vals[I[i], 1], vals[J[i], 1]])
        z = np.array([vals[I[i], 2], vals[J[i], 2]])

        # Debugging print to check the values
        #print(f"Updating line {i}: x={x}, y={y}, z={z}")

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

def create_ree(ax, plots, vals):
    lcolor = "#8e8e8e"
    x = vals[0]
    y = vals[1]
    z = vals[2]

    if len(plots) == 0:
        plots.append((ax.scatter([x], [y], [z], color=lcolor, s=50)))
    else:
        plots[0]._offsets3d = ([x], [y], [z])

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
dct_m,idct_m = get_dct_matrix(config.motion.handover_input_length_dct)
# create tensor, load GPU and add 3rd dim (1,N,N) to dct matrices
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def data_to_viz(model, pbar, num_samples, n_viz):
    import random
    """
    regress_pred() from test.py modified to return pred_data and gt_data
    """
    idxs = [20, 22, 23, 24, 25]
    k=0
    for cnt, (motion_input, motion_target, ree_motion_input, ree_motion_target,int_motion_input, int_motion_target) in enumerate(pbar):
        motion_input = motion_input.cuda()
        b, n, c = motion_input.shape
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
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.handover_input_length],
                                                 motion_input_.cuda())
                    if config.motion_ree.ree_cond:
                        ree_motion_input_ = ree_motion_input.clone()
                        ree_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.handover_input_length],
                                                         ree_motion_input_.cuda())
                    else:
                        ree_motion_input_ = torch.empty(0)

                    if config.motion_int.int_cond:
                        int_motion_input_ = int_motion_input.clone()
                        int_motion_input_ = int_motion_input_.reshape(-1,config.motion.handover_input_length)
                    else:
                        int_motion_input_ = torch.empty(0)
                else:
                    motion_input_ = motion_input.clone()
                    ree_motion_input_ = torch.empty(0)
                output = model(motion_input_, ree_motion_input_, int_motion_input_.cuda())
                # transform output using idct_m for the rows of, handover_input_length. Then we slice to extract the first step frames of the result.
                output = torch.matmul(idct_m[:, :config.motion.handover_input_length, :], output)[:, :step, :]
                # if deriv output
                if config.deriv_output:
                    # we add the input last frame tensor in the step frames predicted from the output (cause to displacement prediction)
                    output = output + motion_input[:, -1:, :].repeat(1, step, 1)

            # reshape output to be (b,step,66), for some reason is done in 2 lines
            output = output.reshape(-1, 9 * 3)
            output = output.reshape(b, step, -1)
            outputs.append(output)
            # delete the first step frames in input and add the output step frames at the end.
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        # concatenate outputs and keep the first 25
        motion_pred = torch.cat(outputs, axis=1)[:, :25]

        # use detach to avoid this tensor being tracked with gradient computations
        motion_target = motion_target.detach()

        b, n, c = motion_target.shape
        motion_target = motion_target.reshape(b,n, -1, 3)
        motion_gt = motion_target.clone()
        motion_pred = motion_pred.reshape(b, n, -1, 3)
        motion_pred = motion_pred.detach().cpu()

        # compute L2 distance between joints pred and goal, compute mean of joints diff in each time frame, sum the values of each time frame in each batch.
        mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred - motion_gt, dim=3), dim=2), dim=0)
        mpjpe_p3d_h36 = mpjpe_p3d_h36/b

        data_pred = torch.squeeze(motion_pred, 0).cpu().data.numpy() # in meters
        data_gt = torch.squeeze(motion_target, 0).cpu().data.numpy()
        ree_data = torch.squeeze(ree_motion_target, 0).cpu().data.numpy()


        #i = random.randint(1, b-1)
        i = idxs[k]
        k+=1

        data_pred = data_pred[i]
        data_gt = data_gt[i]
        ree_data = ree_data[i]
        print(data_gt.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=70)
        vals = np.zeros((9, 3))  # or joints_to_consider
        ree_vals = np.zeros(3)
        gt_plots = []
        pred_plots = []
        ree_plots = []

        gt_plots = create_pose(ax, gt_plots, vals, pred=False, update=False)
        pred_plots = create_pose(ax, pred_plots, vals, pred=True, update=False)
        ree_plot = create_ree(ax, ree_plots, ree_vals)


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

        ax.set_title('mean loss in mm is: ' + str(round(mpjpe_p3d_h36[-1].item(), 4)) + ' with intention : ' + str(sum(int_motion_input[i,:]).item()) + ' for ' + str(
            25) + ' frames')

        line_anim = animation.FuncAnimation(fig, update, 25, fargs=(data_gt, data_pred, ree_data, gt_plots, pred_plots, ree_plot,
                                                                          fig, ax), interval=70, blit=False)
        plt.show()

        if config.viz_GCN_folder:
            line_anim.save('./visualizations/pred-GCN{}/human_viz{}.gif'.format(25, i), writer='pillow')
            print('./visualizations/pred-GCN{}/human_viz{}.gif'.format(25, i))
        else:
            line_anim.save('./visualizations/pred{}/human_viz{}.gif'.format(25, i), writer='pillow')
            print('./visualizations/pred{}/human_viz{}.gif'.format(25, i))
        if cnt == n_viz - 1:
            break

# ax.legend(loc='lower left')


# In[11]:

def set_root(data_gt):
    """
    Function to set fixed root when visualizing. Root set to first frame of sequence.
    """
    gt_vals = data_gt[0]
    xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
    return xroot, yroot, zroot

def update(num, data_gt, data_pred, ree_data ,plots_gt, plots_pred,ree_plot,fig, ax):
    gt_vals = data_gt[num]
    pred_vals = data_pred[num]
    ree_vals = ree_data[num]
    plots_gt = create_pose(ax, plots_gt, gt_vals, pred=False, update=True)
    plots_pred = create_pose(ax, plots_pred, pred_vals, pred=True, update=True)
    ree_plot = create_ree(ax,ree_plot,ree_vals)

    r = 1
    xroot, yroot, zroot = set_root(data_gt)
    ax.set_xlim3d([-r + xroot, r + xroot])
    ax.set_ylim3d([-r + yroot, r + yroot])
    ax.set_zlim3d([-r + zroot, r + zroot])

    # ax.set_title('pose at time frame: '+str(num))
    # ax.set_aspect('equal')

    return plots_gt, plots_pred, ree_plot


# %%

def visualize(model_pth,n_viz):
    model = Model(config)
    config.motion.handover_target_length = config.motion.handover_target_length_eval
    state_dict = torch.load(model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()
    actions = config.actions_to_load
    dataset = HandoverEvalDataset(config, 'test')

    shuffle = False
    sampler = None
    train_sampler = None
    dataloader = DataLoader(dataset, batch_size=128,
                            num_workers=1, drop_last=False,
                            sampler=sampler, shuffle=shuffle, pin_memory=True)
    num_samples = 0
    pbar = dataloader
    data_to_viz(model, pbar, num_samples, n_viz)

    ### ----- Create visualization of data and prediction ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()
    print(args.model_pth)
    visualize(args.model_pth,5)



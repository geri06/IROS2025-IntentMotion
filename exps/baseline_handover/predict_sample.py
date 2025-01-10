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
from lib.utils.handover_functions import find_intentions_mode, get_dct_matrix

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


# create DCT with dimensions of input lenght data (50)
dct_m,idct_m = get_dct_matrix(config.motion.handover_input_length_dct)
# create tensor, load GPU and add 3rd dim (1,N,N) to dct matrices
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

def get_data(path):
    """
    Receives an input file and returns 50 samples for input and 25 samples as ground truth
    """
    used_joint_indexes = np.array(
        [0, 1, 2,  # nose (0, 1, 2)
         # 4, 5, 6,       #left_eye_inner
         # 8, 9, 10,      #left_eye
         # 12, 13, 14,    #left_eye_outer
         # 16, 17, 18,    #right_eye_inner
         # 20, 21, 22,    #right_eye
         # 24, 25, 26,    #right_eye_outer
         # 28, 29, 30,    #left_ear
         # 32, 33, 34,    #right_ear
         # 36, 37, 38,    #mouth_left
         # 40, 41, 42,    #mouth_right
         44, 45, 46,  # left_shoulder (3, 4, 5)
         48, 49, 50,  # right_shoulder (6, 7, 8)
         52, 53, 54,  # left_elbow (9, 10, 11)
         56, 57, 58,  # right_elbow (12, 13, 14)
         60, 61, 62,  # left_wrist (15, 16, 17)
         64, 65, 66,  # right_wrist (18, 19, 20)
         # 68, 69, 70,    #left_pinky
         # 72, 73, 74,    #right_pinky
         # 76, 77, 78,    #left_index
         # 80, 81, 82,    #right_index
         # 84, 85, 86,    #left_thumb
         # 88, 89, 90,    #right_thumb
         92, 93, 94,  # left_hip (21, 22, 23)
         96, 97, 98]  # right_hip (24, 25, 26)
    ).astype(np.int64)
    _end_effector_dims = np.array([132, 133, 134]).astype(int)
    _intention_dim = [136]

    # Define used joints in case context is used

    _points_to_load = np.concatenate((used_joint_indexes, _end_effector_dims, _intention_dim))

    sample_rate = 1

    info = open(path, 'r').readlines()
    the_sequence = []
    # for each line split by space and define each value as a float
    for line in info:
        line = line.strip().split(',')
        # load only data with intention label
        if len(line) == 137:
            line = np.array(line)[_points_to_load.astype(int)]
            the_sequence.append(np.array([float(x) for x in line]))
    # pose_info is a list of np arrays
    the_sequence = np.array(the_sequence)

    n, _ = the_sequence.shape
    sampled_index = range(0, n, sample_rate)
    T = len(sampled_index)
    the_sequence = np.array(the_sequence[sampled_index, :])
    xyz_info = torch.from_numpy(the_sequence).float()

    N = len(xyz_info)
    assert N >= config.motion.handover_input_length + config.motion.handover_target_length_eval, "This input hasn't enough samples we need a minimum of 75"

    # we start from the last possible frame
    start_frame = T - config.motion.handover_input_length - config.motion.handover_target_length_eval
    input_frame_index = start_frame + config.motion.handover_input_length

    input_data = xyz_info[start_frame:input_frame_index,:]
    gt_data = xyz_info[input_frame_index:,:]
    # Separate data
    input_motion_data = input_data[:,:config.motion.dim]
    gt_motion_data = gt_data[:,:config.motion.dim]
    input_ree_data = input_data[:,[28,29,30]]
    gt_ree_data = gt_data[:,[27,28,29]]
    input_int_data = input_data[:,30]
    gt_int_data = gt_data[:,30]

    return input_motion_data, input_ree_data, input_int_data, gt_motion_data, gt_ree_data, gt_int_data


def predict(model,motion_input, motion_target, ree_motion_input, ree_motion_target, int_motion_target, forced_intention = None):
    motion_input = motion_input.unsqueeze(0)
    ree_motion_input = ree_motion_input.unsqueeze(0)
    int_motion_target = int_motion_target.unsqueeze(0)
    motion_target = motion_target.unsqueeze(0)

    motion_input = motion_input.cuda()
    b, n, c = motion_input.shape
    outputs = []
    # how many frames are predicted in one forward pass
    step = config.motion.handover_target_length_train
    # if 25 or more is 1
    num_step = 1 if step == 25 else 25 // step + 1
    for idx in range(num_step):
        # without gradients, useful for inference
        with torch.no_grad():
            # if we want dct encoding
            motion_input_ = motion_input.clone()
            motion_input_ = torch.matmul(dct_m[:, :, :config.motion.handover_input_length],
                                         motion_input_.cuda())

            ree_motion_input_ = ree_motion_input.clone()
            ree_motion_input_ = torch.matmul(dct_m[:, :, :config.motion.handover_input_length],
                                             ree_motion_input_.cuda())
            # keep only the position of the last frame
            ree_motion_input_ = ree_motion_input_[:, config.motion.handover_input_length - 1, :]

            int_motion_prediction_ = int_motion_target.clone()
            # select mode of the intention detected in the next 10 future frames

            if forced_intention is not None:
                int_motion_prediction_ = torch.tensor([forced_intention]) # hard coded intention
            else:
                int_motion_prediction_ = find_intentions_mode(int_motion_prediction_)

            output, int_class_logits,intention_pred = model(motion_input_, ree_motion_input_, int_motion_prediction_.cuda())
            # transform output using idct_m for the rows of, handover_input_length. Then we slice to extract the first step frames of the result.
            output = torch.matmul(idct_m[:, :config.motion.handover_input_length, :], output)[:, :step, :]

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
    motion_target = motion_target.reshape(b, n, -1, 3)
    motion_gt = motion_target.clone()
    motion_pred = motion_pred.reshape(b, n, -1, 3)
    motion_pred = motion_pred.detach().cpu()

    # compute L2 distance between joints pred and goal, compute mean of joints diff in each time frame, sum the values of each time frame in each batch.
    mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(motion_pred - motion_gt, dim=3), dim=2), dim=0)

    data_pred = torch.squeeze(motion_pred, 0).cpu().data.numpy()  # in meters
    data_gt = torch.squeeze(motion_target, 0).cpu().data.numpy()
    ree_data = torch.squeeze(ree_motion_target, 0).cpu().data.numpy()
    intention_data = int_motion_prediction_.item()

    print(data_pred.shape, data_gt.shape, ree_data.shape, intention_data)
    return data_pred, data_gt, ree_data, intention_data, mpjpe_p3d_h36

def viz_prediction(data_pred, data_gt, ree_data, intention_data, mpjpe_p3d_h36, subject_info):

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

    ax.set_title('mean loss in mm is: ' + str(round(torch.mean(mpjpe_p3d_h36).item(), 3)) + ' with intention : ' + str(
        intention_data) + ' for ' + str(
        25) + ' frames')

    line_anim = animation.FuncAnimation(fig, update, 25,
                                        fargs=(data_gt, data_pred, ree_data, gt_plots, pred_plots, ree_plot,
                                               fig, ax), interval=70, blit=False)
    plt.show()

    line_anim.save('/home/gerard/Documents/IRI/Repos/siMLPe/visualizations/S7/{}.gif'.format(subject_info), writer='pillow')
    print(subject_info)

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

    ### ----- Create visualization of data and prediction ----
if __name__ == '__main__':
    # model_path = "/home/gerard/Documents/IRI/Repos/siMLPe/checkpoints/keypoints_predictor_v1.pth"
    model_path = "/home/gerard/Documents/IRI/Repos/siMLPe/exps/baseline_handover/log/snapshot/model-iter-5000.pth"
    model = Model(config)
    config.motion.handover_target_length = config.motion.handover_target_length_eval
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()
    #visualize(args.model_pth,5)
    sample_path = "/home/gerard/Documents/IRI/Repos/siMLPe/data/handover/S7/straight/right_nd.txt"
    # define input data
    input_motion_data, input_ree_data, input_int_data, gt_motion_data, gt_ree_data, gt_int_data = get_data(sample_path)
    forced_intention = 1

    # do the prediction
    data_pred, data_gt, ree_data, intention_data, mpjpe_p3d_h36 = predict(model, input_motion_data, gt_motion_data, input_ree_data, gt_ree_data, gt_int_data, forced_intention)

    # Define name of the gif
    path_parts = sample_path.split("/S7/")
    remaining_path = path_parts[1]
    subject_info = "_".join(remaining_path.split("/")[:2]).rsplit(".", 1)[0] + f"_{int(intention_data)}"

    # call visualization function
    viz_prediction(data_pred, data_gt, ree_data, intention_data, mpjpe_p3d_h36, subject_info)




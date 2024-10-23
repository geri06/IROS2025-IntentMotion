#!/usr/bin/env python
# coding: utf-8
import argparse
import os

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
        False, True, False, True,
        False, True, False, True,
        False
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


# In[11]:


def update(num, data_gt, data_pred, plots_gt, plots_pred, fig, ax):
    gt_vals = data_gt[num]
    pred_vals = data_pred[num]
    plots_gt = create_pose(ax, plots_gt, gt_vals, pred=False, update=True)
    plots_pred = create_pose(ax, plots_pred, pred_vals, pred=True, update=True)

    r = 1
    xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
    ax.set_xlim3d([-r + xroot, r + xroot])
    ax.set_ylim3d([-r + yroot, r + yroot])
    ax.set_zlim3d([-r + zroot, r + zroot])

    # ax.set_title('pose at time frame: '+str(num))
    # ax.set_aspect('equal')

    return plots_gt, plots_pred


# %%

def visualize_data(folder_to_viz):
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

    files = os.listdir(folder_to_viz)
    for file in files:
        file_pth = os.path.join(folder_to_viz,file)
        print(file_pth)

        with open(file_pth, 'r') as f:
            lines = f.readlines()

        data_gt = []
        for line in lines:
            line = np.fromstring(line.strip(), sep=',')
            line = line[used_joint_indexes]
            data_gt.append(line)
        data_gt = np.array(data_gt)
        data_gt = data_gt.reshape(-1, 9, 3)
        print(data_gt.shape)
        data_pred = data_gt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=-40)
        vals = np.zeros((9, 3))  # or joints_to_consider
        gt_plots = []

        gt_plots = create_pose(ax, gt_plots, vals, pred=False, update=False)
        pred_plots = gt_plots


        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend(loc='lower left')

        ax.set_xlim3d([-1, 1.5])
        ax.set_xlabel('X')

        ax.set_ylim3d([0.0, 2.25])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-1, 1.5])
        ax.set_zlabel('Z')

        ax.set_title('Visualising:' + file_pth)

        line_anim = animation.FuncAnimation(fig, update, data_gt.shape[0], fargs=(data_gt, data_pred, gt_plots, pred_plots,
                                                                    fig, ax), interval=70, blit=False)
        plt.show()

        line_anim.save(f'./visualizations/handover_data_viz/human_viz-{subject}-{scenario}-{file}.gif', writer='pillow')

    ### ----- Create visualization of data and prediction ----
if __name__ == '__main__':

    subject = "S5"
    scenario = "straight"
    folder_to_viz = f"./data/handover/{subject}/{scenario}"
    visualize_data(folder_to_viz)



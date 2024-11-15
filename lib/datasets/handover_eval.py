import numpy as np
import os
import glob
import torch
import torch.utils.data as data

from exps.baseline_handover.config import config


class HandoverEvalDataset(data.Dataset):
    def __init__(self, config, split_name):
        super(HandoverEvalDataset, self).__init__()
        # data paertition (validation test or train)
        self._split_name = split_name
        self.sample_rate = 1

        # Dir h36
        self._handover_anno_dir = config.handover_anno_dir
        # indexes of used joints
        self.used_joint_indexes = np.array(
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
        self.end_effector_dims = np.array([132, 133, 134]).astype(int)
        self.ree_cond = config.motion_ree.ree_cond

        # Define used joints in case context is used

        self._points_to_load = np.concatenate((self.used_joint_indexes, self.end_effector_dims))

        self._scenarios = ["straight", "one_obstacle", "multiple_obstacles"]

        self._handover_files = self._get_handover_files()

        self.handover_motion_input_length = config.motion.handover_input_length
        self.handover_motion_target_length = config.motion.handover_target_length

        self.motion_dim = config.motion.dim
        self.shift_step = config.shift_step
        self._collect_all()
        self._file_length = len(self.data_idx)

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._handover_files)

    def _get_handover_files(self):
        # create list
        seq_names = []

        seq_names += open(
            os.path.join(self._handover_anno_dir.replace('handover', ''), "handover_test.txt"), 'r'
        ).readlines()

        # save paths of Subjects (SNum) and diverse actions into a list
        file_list = []
        for subject in seq_names:
            subject = subject.strip()
            print("Evaluation done in subject:", subject)
            for scenario in self._scenarios:
                subjects = glob.glob(self._handover_anno_dir + subject + '/' + scenario + '/*')
                for s in subjects:
                    file_list.append(s)

        handover_files = []

        for path in file_list:
            info = open(path, 'r').readlines()
            the_sequence = []
            # for each line split by space and define each value as a float
            for line in info:
                line = line.strip().split(',')
                if len(line) > 0:
                    line = np.array(line)[self._points_to_load.astype(int)]
                    the_sequence.append(np.array([float(x) for x in line]))
            # pose_info is a list of np arrays
            the_sequence = np.array(the_sequence)

            if len(the_sequence.shape) == 2:
                n, _ = the_sequence.shape
                sampled_index = range(0, n, self.sample_rate)
                T = len(sampled_index)
                the_sequence = np.array(the_sequence[sampled_index, :])
                xyz_info = torch.from_numpy(the_sequence).float()
                xyz_info = xyz_info.reshape(T, -1, 3)
                handover_files.append(xyz_info)
        print("Number of files for evaluation:", len(handover_files), handover_files[0].shape)
        return handover_files

    def _collect_all(self):
        """
        Create self.handover_seqs with blocks of data from each txt
        file flattened: (T,9*3).Generates valid frame indices for creating
        training samples.
        """
        # Keep align with HisRep dataloader
        self.handover_seqs = []
        self.data_idx = []
        idx = 0
        # for each processed txt file (T,9,3)
        for handover_motion_poses in self._handover_files:
            N = len(handover_motion_poses)
            if N < self.handover_motion_target_length + self.handover_motion_input_length:
                continue

            # define T, new number of frames
            T = handover_motion_poses.shape[0]
            # flatten numjoints*xyz --> (T,numjoints*xyz)
            handover_motion_poses = handover_motion_poses.reshape(T,-1)

            # add processed motion poses to handover_seqs
            self.handover_seqs.append(handover_motion_poses)
            # list of valid frame indices where motion sequence can start
            valid_frames = np.arange(0, T - self.handover_motion_input_length - self.handover_motion_target_length + 1,
                                     self.shift_step)
            # create a list of tuples (idx, valid_frame_index), idx is saved to know which seq corresponds tht valid frame
            self.data_idx.extend(zip([idx] * len(valid_frames), valid_frames.tolist()))
            idx += 1

    def __getitem__(self, index):
        """
        Called when index passed to data. Dataloader will call this function.
        Select data, apply data augmentation randomly and return input and target.
        """
        idx, start_frame = self.data_idx[index]
        frame_indexes = np.arange(start_frame,
                                  start_frame + self.handover_motion_input_length + self.handover_motion_target_length)

        # define motion and ree input separetly
        all_motion = self.handover_seqs[idx][frame_indexes]

        ree_idx = [27, 28, 29]
        ree_motion = all_motion[:, ree_idx]  # canviar per self.ree coordinates de config
        # set input to provide training data and target to compute loss
        ree_motion_input = ree_motion[:self.handover_motion_input_length]
        ree_motion_target = ree_motion[self.handover_motion_input_length:]
        # change to float
        ree_motion_input = ree_motion_input.float()
        ree_motion_target = ree_motion_target.float()

        motion = all_motion[:, :27]  # select only motion input data

        # define input and target of motion
        handover_motion_input = motion[:self.handover_motion_input_length]  # meter
        handover_motion_target = motion[self.handover_motion_input_length:]  # meter

        # change to float
        handover_motion_input = handover_motion_input.float()
        handover_motion_target = handover_motion_target.float()
        return handover_motion_input, handover_motion_target, ree_motion_input, ree_motion_target


if __name__ == "__main__":
    config.motion.handover_target_length = config.motion.handover_target_length_eval
    dataset = HandoverEvalDataset(config, 'test')
    motion_input, motion_target, ree_input, ree_target = dataset[5]

#    print(motion_input.shape, motion_target.shape, ree_input.shape, ree_target.shape)
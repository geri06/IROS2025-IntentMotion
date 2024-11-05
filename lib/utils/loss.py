import torch

def L2_body(motion_gt, motion_pred):
    mpjpe_p3d = torch.sum(torch.mean(torch.norm(motion_pred - motion_gt, dim=3), dim=2), dim=0)
    return mpjpe_p3d

def L2_right_hand(motion_gt, motion_pred):
    # 6 is right hand joint index
    right_hand_gt = motion_gt[:,:,6,:]
    right_hand_pred = motion_pred[:,:,6,:]
    right_hand_loss = torch.sum(torch.norm(right_hand_pred - right_hand_gt, dim=2),dim=0)
    return right_hand_loss
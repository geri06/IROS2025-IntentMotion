import torch

def L2_body(motion_gt, motion_pred):
    mpjpe_p3d = torch.sum(torch.mean(torch.norm(motion_pred - motion_gt, dim=3), dim=2), dim=0)
    return mpjpe_p3d

def L2_right_hand(motion_gt, motion_pred):
    # 6 is right hand joint index
    right_hand_gt = motion_gt[:,:,5,:]
    right_hand_pred = motion_pred[:,:,5,:]
    right_hand_loss = torch.sum(torch.norm(right_hand_pred - right_hand_gt, dim=2),dim=0)
    return right_hand_loss

def quality_metrics(motion_gt, motion_pred):
    dist_tensor = torch.norm(motion_pred - motion_gt, dim=3)
    under_30 = (dist_tensor < 0.3).float().mean()
    under_20 = (dist_tensor < 0.2).float().mean()
    under_10 = (dist_tensor < 0.1).float().mean()

    return under_10,under_20,under_30
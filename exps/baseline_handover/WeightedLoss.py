import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLossLayer(nn.Module):
    def __init__(self, num_losses):
        super(WeightedLossLayer, self).__init__()
        # Initialize learnable weights for each loss
        self.alpha = nn.Parameter(torch.tensor(1.0, requires_grad=True))  # Initial value of 1.0

    def forward(self, main_loss, extra_loss):
        """
        Compute total loss as the sum of the main loss and a weighted extra loss.
        Args:
        - main_loss (Tensor): The main loss (scalar).
        - extra_loss (Tensor): The extra loss (scalar).

        Returns:
        - total_loss (Tensor): Combined weighted loss.
        - alpha (Tensor): The learnable weight for the extra loss.
        """
        # Weighted combination of main loss and extra loss
        total_loss = main_loss + F.relu(self.alpha) * extra_loss
        return total_loss, self.alpha

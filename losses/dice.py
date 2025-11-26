# Load modules
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import flatten
from typing import (Optional, Union, List)



class DiceLoss(nn.Module):
    def __init__(self, 
                 n_classes: int, 
                 smooth: float = 1e-5,
                 weight: Optional[Union[List[float], torch.Tensor]] = None,
                 softmax: bool = False
        ) -> None:
        """
            `Dice loss` for multi-class segmentation.

            Args:
                n_classes (int): Number of classes.
                smooth (float): Smoothing factor to avoid division by zero.
                weight (Optional[List[float] or Tensor]): Class weights of shape (C, ).
                softmax (bool): Whether to apply softmax to inputs.
        """
        super().__init__()        
        self.n_classes = n_classes
        self.smooth = smooth        
        self.softmax = softmax

        if weight is None:
            self.register_buffer("weight", torch.ones(n_classes, dtype = torch.float))                  # shape: (C, )
        else:
            self.register_buffer("weight", torch.tensor(weight, dtype = torch.float))                   # shape: (C, )
        


    def _dice_loss(self, 
                   inputs: torch.Tensor, 
                   target: torch.Tensor
            ) -> torch.Tensor:
        """
            Compute `Dice loss` for one class.

            Args:
                inputs (Tensor): Flattened predicted probabilities for one class, shape (N, )
                target (Tensor): Flattened binary ground truth mask for that class, shape (N, )

            Returns:
                Tensor: Scalar Dice loss for the class
        """
        intersect = torch.sum(inputs * target)
        inputs_sum = torch.sum(inputs)
        target_sum = torch.sum(target)
        dice_score = (2 * intersect + self.smooth) / (inputs_sum + target_sum + self.smooth)
        dice_loss_value = (1 - dice_score)
        return dice_loss_value


    def forward(self,
                inputs: torch.Tensor,                   
                target: torch.Tensor                    
            ) -> torch.Tensor:
        """
            Compute multi-class `Dice loss`.

            Args:
                inputs (Tensor): Raw logits or probabilities, shape (B, C, H, W)
                target (Tensor): Ground truth class indices, shape (B, H, W)

            Returns:
                Tensor: Scalar loss (mean over classes)
        """

        inputs, target = flatten(inputs, target)                                                        # shape: (B * H * W, C)

        if self.softmax:
            inputs = F.softmax(inputs, dim = 1)                                                         # shape: (B * H * W, C)
        
        loss = 0.0
        
        for c in range(self.n_classes):
            inputs_c = inputs[:, c]                                                                     # shape: (B * H * W, )
            target_c = (target == c).float()                                                            # shape: (B * H * W, )

            loss_c = self._dice_loss(inputs_c, target_c)
            loss += self.weight[c] * loss_c
        
        return loss / self.n_classes

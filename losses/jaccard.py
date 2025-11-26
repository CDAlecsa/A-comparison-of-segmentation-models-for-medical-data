# Load modules
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import flatten
from typing import (Optional, Union, List)



class JaccardLoss(nn.Module):
    def __init__(self, 
                 n_classes: int, 
                 smooth: float = 1e-5,
                 weight: Optional[Union[List[float], torch.Tensor]] = None,
                 softmax: bool = False
        ) -> None:
        """
            `Jaccard loss` for multi-class segmentation.

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


    def _jaccard_loss(self, 
                      inputs: torch.Tensor, 
                      target: torch.Tensor
            ) -> torch.Tensor:
        """
            Compute `Jaccard loss` for one class.

            Args:
                inputs (Tensor): Flattened predicted probabilities for one class, shape (N, )
                target (Tensor): Flattened binary ground truth mask for that class, shape (N, )

            Returns:
                Tensor: Scalar Jaccard loss for the class
        """
        intersection = torch.sum(inputs * target)                                           # scalar
        total = torch.sum(inputs + target)                                                  # scalar
        union = total - intersection                                                        # scalar
        IOU_score = (intersection + self.smooth) / (union + self.smooth)                    # scalar
        IOU_loss_value = (1 - IOU_score)                                                    # scalar
        return IOU_loss_value


    def forward(self,
                inputs: torch.Tensor,                   
                target: torch.Tensor                    
            ) -> torch.Tensor:
        """
            Compute multi-class `Jaccard (IoU) loss`.

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

            loss_c = self._jaccard_loss(inputs_c, target_c)
            loss += self.weight[c] * loss_c

        return loss / self.n_classes

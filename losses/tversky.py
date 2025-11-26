import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import flatten
from typing import (Optional, Union, List)



class TverskyLoss(nn.Module):
    def __init__(self,
                 n_classes: int,
                 smooth: float = 1e-5,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 weight: Optional[Union[List[float], torch.Tensor]] = None,
                 softmax: bool = False
            ) -> None:
        """
            `Tversky loss` for multi-class segmentation.

            Args:
                n_classes (int): Number of classes.
                smooth (float): Smoothing factor to avoid division by zero.
                alpha (float): The `alpha` coefficient of the Tversky loss.
                beta (float): The `beta` coefficient of the Tversky loss.
                weight (Optional[List[float] or Tensor]): Class weights of shape (C, ).
                softmax (bool): Whether to apply softmax to inputs.
        """
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.softmax = softmax

        if weight is None:
            self.register_buffer("weight", torch.ones(n_classes, dtype = torch.float))                  # shape: (C, )
        else:
            self.register_buffer("weight", torch.tensor(weight, dtype = torch.float))                   # shape: (C, )


    def _tversky_loss(self, 
                      inputs: torch.Tensor, 
                      target: torch.Tensor
            ) -> torch.Tensor:
        """
            Compute `Tversky loss` for one class.

            Args:
                inputs (Tensor): Flattened predicted probabilities for one class, shape (N, )
                target (Tensor): Flattened binary ground truth mask for that class, shape (N, )

            Returns:
                Tensor: Scalar Tversky loss for the class
        """
        tp = torch.sum(inputs * target)                                                                 # scalar
        fp = torch.sum((1 - target) * inputs)                                                           # scalar
        fn = torch.sum(target * (1 - inputs))                                                           # scalar
        tversky_score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)      # scalar
        tversky_loss_value = (1 - tversky_score)                                                        # scalar
        return tversky_loss_value

    def forward(self,
                inputs: torch.Tensor,                   
                target: torch.Tensor                   
            ) -> torch.Tensor:
        """
            Compute multi-class `Tversky loss`.

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

            loss_c = self._tversky_loss(inputs_c, target_c)
            loss += self.weight[c] * loss_c

        return loss / self.n_classes               

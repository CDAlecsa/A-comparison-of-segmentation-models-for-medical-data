import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import flatten
from typing import (Optional, Union, List)



class FocalLoss(nn.Module):
    def __init__(self, 
                 n_classes: int, 
                 gamma: float = 2.0,
                 weight: Optional[Union[List[float], torch.Tensor]] = None,
                 softmax: bool = False
            ) -> None:
        """
            `Focal loss` for multi-class segmentation.

            Args:
                n_classes (int): Number of classes.
                gamma (float): Focal loss modulating factor.
                weight (Optional[List[float] or Tensor]): Class weights of shape (C, ).
                softmax (bool): Whether to apply softmax to inputs.
        """
        super().__init__()
        self.n_classes = n_classes
        self.gamma = gamma
        self.softmax = softmax
        
        if weight is None:
            self.register_buffer("weight", torch.ones(n_classes, dtype = torch.float))                  # shape: (C, )
        else:
            self.register_buffer("weight", torch.tensor(weight, dtype = torch.float))                   # shape: (C, )


    def forward(self,
                inputs: torch.Tensor,          
                target: torch.Tensor           
        ) -> torch.Tensor:
        """
            Compute multi-class `focal loss`.

            Args:
                inputs (Tensor): Raw logits of shape (B, C, H, W)
                target (Tensor): Ground truth class indices of shape (B, H, W)
                weight (Optional[List[float] or Tensor]): Class weights of shape (C, )

            Returns:
                Tensor: Scalar loss value (mean over all pixels)
        """

        inputs, target = flatten(inputs, target)                                                # shape: (B * H * W, C)

        if self.softmax:
            probs = F.softmax(inputs, dim = 1)                                                  # shape: (B * H * W, C)
        else:
            probs = inputs                                                                      # shape: (B * H * W, C)

        target = target.long()                                                                  # shape: (B * H * W, )
        
        true_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)                            # shape: (B * H * W, )

        ce_loss = F.cross_entropy(inputs, target, reduction = 'none')                           # shape: (B * H * W, )
        focal_weight = torch.pow(1 - true_probs, self.gamma)                                    # shape: (B * H * W, )
        loss = focal_weight * ce_loss                                                           # shape: (B * H * W, )

        class_weights = self.weight[target]                                                     # shape: (B * H * W, )
        loss = loss * class_weights                                                             # shape: (B * H * W, )

        return loss.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import flatten
from typing import (Optional, Union, List)



class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 n_classes: int,
                 weight: Optional[Union[List[float], torch.Tensor]] = None,
                 softmax: bool = False
                 ) -> None:
        """
            Wrapper for multi-class `Cross Entropy loss`, with optional class weights.

            Args:
                n_classes (int): Number of classes.
                weight (Optional[List[float] or Tensor]): Class weights of shape (C,)
                softmax (bool): If True, inputs are probabilities and Cross Entropy is computed manually.
        """
        super().__init__()
        self.n_classes = n_classes
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
            Compute `Cross Entropy loss`.

            Args:
                inputs (Tensor): Raw logits, shape (B, C, H, W)
                target (Tensor): Ground truth class indices, shape (B, H, W)

            Returns:
                Tensor: Scalar loss (mean over pixels)
        """

        inputs, target = flatten(inputs, target)                                                        # shape: (B * H * W, C)

        # Input are probabilities
        if self.softmax:
            log_probs = torch.log(inputs.clamp(min = 1e-9)) 
            loss = F.nll_loss(log_probs, target.long(), weight = self.weight, reduction = 'mean')       # scalar

        # Inputs are raw logits
        else:
            loss = F.cross_entropy(inputs, target.long(), weight = self.weight, reduction = 'mean')     # scalar

        return loss

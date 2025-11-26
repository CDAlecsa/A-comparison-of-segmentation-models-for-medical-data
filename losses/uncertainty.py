# Load modules
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (Any, Tuple, List, Dict)

from .dice import DiceLoss
from .jaccard import JaccardLoss
from .tversky import TverskyLoss
from .ce import CrossEntropyLoss
from .focal import FocalLoss



LOSS_NAME_to_CLASS = {
                        'dice': DiceLoss,
                        'jaccard': JaccardLoss,
                        'tversky': TverskyLoss,
                        'ce': CrossEntropyLoss,
                        'focal': FocalLoss
                    }


class UW_loss(nn.Module):
    r"""
        The multi-task `Uncertainty Weighting` loss function.
    """

    def __init__(self,
                 losses: Dict[str, Dict[str, Any]],
                 learnable_params: bool,
                 default_param_values: List[float]
            ) -> None:
        """
        Args:
            losses (Dict): Maps loss name to kwargs for that loss.
                For example: { 'dice': {'n_classes': 3, 'softmax': True}, ... }
            learnable_params (bool): Whether the log variance parameters are learnable.
            default_param_values (List[float]): Initial value for the noise params.
        """
        super().__init__()

        # Define the loss modules
        self.losses = nn.ModuleDict()
        self.learnable_params = learnable_params

        for (name, kwargs) in losses.items():
            if name not in LOSS_NAME_to_CLASS:
                raise ValueError(f"Loss '{name}' is not recognized. Available: {list(LOSS_NAME_to_CLASS.keys())}")

            loss_class = LOSS_NAME_to_CLASS[name]
            self.losses[name] = loss_class(**kwargs)

        # Noise parameters for the loss' components
        if self.learnable_params:
            self.tasks_log_noise_params = nn.Parameter( torch.tensor(default_param_values) )
        else:
            self.tasks_log_noise_params = torch.tensor(default_param_values)


    def forward(self, 
                inputs: torch.Tensor, 
                target: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict, Dict]:
        r"""
            Compute the final loss using the given loss components.
            
            Args:
                inputs (Tensor): Flattened predicted probabilities for one class, shape (N, )
                target (Tensor): Flattened binary ground truth mask for that class, shape (N, )

            Returns:
                (loss, stop_gradient_values, loss_contributions)(`Tuple[torch.Tensor, Dict, Dict]`): 
                The computed loss, the values of gradient-free loss components,
                along with the contribution of each individual loss.
        """

        # Compute the actual values of the loss' components
        loss_values = {}

        for (name, loss_fn) in self.losses.items():
            loss_values[name] = loss_fn(inputs, target)

        loss_names = list(loss_values.keys())

        # Compute the forward pass' outputs
        losses = torch.stack(list(loss_values.values()))

        # Retrieve the trainable noise parameters
        tasks_noise_params = torch.exp( self.tasks_log_noise_params ).to(device = losses.device)
        weights = 1.0 / ( 2.0 * tasks_noise_params + 1e-12 )

        loss = torch.sum(weights * losses)
        
        if self.learnable_params:
            loss = loss + torch.sum( self.tasks_log_noise_params )

        stop_gradient_values = {n: sg_v.item() for (n, sg_v) in zip(loss_names, loss_values.values())}
        loss_contributions = {n: w.item() for (n, w) in zip(loss_names, weights)}

        return loss, stop_gradient_values, loss_contributions



class UW_SO_loss(nn.Module):
    r"""
        The multi-task `Soft Optimal Uncertainty Weighting` loss function..
    """

    def __init__(self,
                 losses: Dict[str, Dict[str, Any]],
                 learnable_params: bool,
                 default_param_values: float
            ) -> None:
        """
        Args:
            losses (Dict): Maps loss name to kwargs for that loss.
                For example: { 'dice': {'n_classes': 3, 'softmax': True}, ... }
            learnable_params (bool): Whether the log variance parameters are learnable.
            default_param_values (float): Initial value for the temperature param.
        """
        super().__init__()

        # Define the loss modules
        self.losses = nn.ModuleDict()

        for (name, kwargs) in losses.items():
            if name not in LOSS_NAME_to_CLASS:
                raise ValueError(f"Loss '{name}' is not recognized. Available: {list(LOSS_NAME_to_CLASS.keys())}")

            loss_class = LOSS_NAME_to_CLASS[name]
            self.losses[name] = loss_class(**kwargs)

        # The temperature parameter which will be involved in the weights of the loss' components
        if learnable_params:
            self.tasks_log_noise_params = nn.Parameter( torch.tensor(default_param_values) )
        else:
            self.tasks_log_noise_params = torch.tensor(default_param_values)


    def forward(self, 
                inputs: torch.Tensor, 
                target: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict, Dict]:
        r"""
            Compute the final loss using the given loss components.
            
            Args:
                inputs (Tensor): Flattened predicted probabilities for one class, shape (N, )
                target (Tensor): Flattened binary ground truth mask for that class, shape (N, )

            Returns:
                (loss, stop_gradient_values, loss_contributions)(`Tuple[torch.Tensor, Dict, Dict]`): 
                The computed loss, the values of gradient-free loss components,
                along with the contribution of each individual loss.
        """

        # Compute the actual values of the loss' components
        loss_values = {}

        for (name, loss_fn) in self.losses.items():
            loss_values[name] = loss_fn(inputs, target)
                    
        loss_names = list(loss_values.keys())

        # Retrieve the trainable temperature parameter
        tasks_temperature = torch.exp(self.tasks_log_noise_params)

        # Compute the forward pass' outputs
        stop_gradient_values = [v.detach() for v in loss_values.values()]
        inv_losses = torch.stack([ 
                                    1.0 / ( sg_value * tasks_temperature + 1e-12 ) 
                                            for sg_value in stop_gradient_values
                                ])
        weights = F.softmax(inv_losses, dim = 0)
        loss_contributions = {n: w.item() for (n, w) in zip(loss_names, weights)}

        loss = torch.sum( weights * torch.stack(list(loss_values.values())) )
        stop_gradient_values = {n: sg_v.item() for (n, sg_v) in zip(loss_names, stop_gradient_values)}

        return loss, stop_gradient_values, loss_contributions
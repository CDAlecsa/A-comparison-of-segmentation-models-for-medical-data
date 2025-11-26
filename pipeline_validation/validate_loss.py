import torch
from losses.uncertainty import (UW_loss, UW_SO_loss)



# Run the script from the root folder "main_codes" using the -m flag: "python -m pipeline_validation.validate_loss"
if __name__ == "__main__":

    # Initialize seed & choose device
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy config
    n_classes, height, width, batch_size = 3, 64, 64, 4

    # Simulated model outputs (logits) & ground-truth labels
    inputs = torch.randn(batch_size, n_classes, height, width)                                  # raw logits
    targets = torch.randint(0, n_classes, (batch_size, height, width))                          # ground truth indices

    # Define the loss configuration
    loss_config_variants = {
                            'ce_dice': {
                                            'ce': {'n_classes': 3, 'softmax': False},
                                            'dice': {'n_classes': 3, 'softmax': True}
                                        },
                            'ce_focal': {
                                            'ce': {'n_classes': 3, 'softmax': False},
                                            'focal': {'n_classes': 3, 'softmax': True}
                                        },
                            'dice_jaccard': {
                                                'dice': {'n_classes': 3, 'softmax': True},
                                                'jaccard': {'n_classes': 3, 'softmax': True}
                                            },
                            'dice_tversky': {
                                                'dice': {'n_classes': 3, 'softmax': True},
                                                'tversky': {'n_classes': 3, 'softmax': True, 'alpha': 0.7, 'beta': 0.3}
                                            },
                            'ce_dice_focal': {
                                                'ce': {'n_classes': 3, 'softmax': False},
                                                'dice': {'n_classes': 3, 'softmax': True},
                                                'focal': {'n_classes': 3, 'softmax': True}
                                            }
    }

    # Choose loss type: 'UW_loss' or 'UW_SO_loss'
    loss_type = 'UW_loss'
    learnable_params = True

    for (config_name, loss_config) in loss_config_variants.items():
        print(f"\n{'='*80}")
        print(f"Testing config: {config_name} | Loss type: {loss_type} | Learnable: {learnable_params}")
        print(f"{'='*80}\n")

        default_params = [0.0] * len(loss_config)

        # Initialize the composite loss
        if loss_type == 'UW_loss':
            criterion = UW_loss(losses = loss_config, 
                                learnable_params = learnable_params, 
                                default_param_values = default_params
                            )
        elif loss_type == 'UW_SO_loss':
            criterion = UW_SO_loss(losses = loss_config, 
                                   learnable_params = learnable_params, 
                                   default_param_values = default_params
                            )
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

        # Move to device
        criterion = criterion.to(device)
        
        # Forward pass
        inputs, targets = inputs.to(device), targets.to(device)
        loss, stop_gradient_values, loss_contributions = criterion(inputs, targets)

        # Print results
        print(f"Total Loss: {loss.item():.4f}")

        print("\nPer-Loss Values:")
        for (name, value) in stop_gradient_values.items():
            print(f"  {name}: {value:.4f}")

        print("\nPer-Loss Contributions (weights):")
        for name, weight in loss_contributions.items():
            print(f"  {name}: {weight:.4f}")

        if learnable_params:
            print("\nLog-Variance Parameters:")
            print(criterion.tasks_log_noise_params.data.cpu().numpy())
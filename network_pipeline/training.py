# Load modules
import numpy as np
import os, json, yaml, shutil, torch

from copy import deepcopy
from rich.markup import escape

from torch.utils.tensorboard import SummaryWriter
from utils.utils import (logger, 
                         post_process_targets,
                         initialize_progress_bar, 
                         find_file_in_subfolders)

from .inference import (validate_model, 
                        model_inference)
from dataloader.dataset import (create_train_val_dataloaders, 
                                get_base_dataset)

from .initialize_model import (get_model, 
                               initialize_loss, 
                               OPTIMIZER_NAME_to_CLASS)



def model_training(config_filename: str) -> None:
    """
        Train a segmentation model with respect to the custom `MedicalSegmentationDataset` class.

        Args:
            config_filename(`str`): The path of the configuration file.
    """

    # Get model's name & configuration file
    config_path = os.path.abspath('configs')
    model_name = find_file_in_subfolders(config_path, config_filename)

    config_path = os.path.join( config_path, f"{model_name}" )
    config_file = os.path.join( config_path, f"{config_filename}.yaml" )

    # Read the configuration file
    with open(config_file, 'r') as f:
        args = yaml.safe_load(f)

    # Retrieve configuration parameters
    data_config = args['data']
    aug_config = args['aug']
    model_config = args['model']
    loss_config = args['loss']
    optimizer_config = args['optimizer']

    assert model_name == model_config['name']
    print('\n')
    logger.info(f"Model [{model_name}] from file [{config_filename}].")

    # Initialize output path from the model's name
    experiments_path = os.path.abspath('experiments')    

    # Create output folder structure for saving results
    output_path = os.path.join( experiments_path, f"{config_filename}" )
    os.makedirs(output_path, exist_ok = True)

    train_path = os.path.join(output_path, 'train')
    os.makedirs(train_path, exist_ok = True)

    log_path = os.path.join(train_path, 'log')
    os.makedirs(log_path, exist_ok = True)

    checkpoint_path = os.path.join(train_path, 'checkpoint')
    checkpoint_file = os.path.join(checkpoint_path, 'best_model.pth')
    os.makedirs(checkpoint_path, exist_ok = True)

    results_file = os.path.join(checkpoint_path, 'results_metrics.json')
    split_ids_file = os.path.join(checkpoint_path, 'train_val_split.json')
    
    # Create output subfolder structure for saving the configuration settings
    settings_path = os.path.join( train_path, f"settings" )
    os.makedirs(settings_path, exist_ok = True)

    # Retrieve image size
    folder_name = os.path.basename(data_config['data_path'])
    size_str = folder_name.split('_')[-2:]
    dim1, dim2 = map(int, size_str)
    img_size = dim1  
    assert dim1 == dim2, f"Dimensions mismatch: {dim1} != {dim2}"
    
    # Create train & val dataloaders
    train_dataloader, val_dataloader, split_ids = create_train_val_dataloaders(data_config = data_config,
                                                                               train_aug_config = aug_config)
    
    with open(split_ids_file, "w") as f:
        json.dump(split_ids, f, indent = 4)

    # Get the number of images belonging to the datasets
    n_train_images = len(train_dataloader.dataset)
    n_val_images = len(val_dataloader.dataset)
    
    print('\n')
    logger.info(f"Number of `train` images: {n_train_images}.")
    logger.info(f"Number of `val` images: {n_val_images}.")

    # Print number of train / validation iterations per epoch
    n_train_batches = len(train_dataloader)
    n_val_batches = len(val_dataloader)
    
    print('\n')
    logger.info(f"Number of `train` batches: {n_train_batches}.")
    logger.info(f"Number of `val` batches: {n_val_batches}.")

    # Check that the train / val label encoders are equal
    train_label_encoder = get_base_dataset(train_dataloader.dataset).label_encoder
    val_label_encoder = get_base_dataset(val_dataloader.dataset).label_encoder

    n_classes = len(train_label_encoder)
    
    if train_label_encoder != val_label_encoder:
        raise ValueError("Train and validation label encoders do not match!")

    # Initialize PyTorch device
    device = model_config['device']

    # Define model
    in_channels, *img_size = train_dataloader.dataset[0]['raw_image'].shape

    assert img_size[0] == img_size[1], f"Dimensions must be equal, but got `{img_size}`."
    img_size = img_size[0]

    model = get_model(model_name = model_name, 
                      model_config = model_config, 
                      img_size = img_size, 
                      in_channels = in_channels,
                      n_classes = n_classes, 
                      bs = data_config['batch_size'],
                      device = device,
                      eval_mode = False
                    )

    # Put model in training mode
    model.train()

    # Get training epochs / iterations / initial learning rate
    base_lr = optimizer_config['params']['lr']
    epochs = model_config['epochs']
    max_iterations = epochs * len(train_dataloader)

    # Initialize the loss function
    loss, loss_names = initialize_loss(loss_config, n_classes)
    loss = loss.to(device)

    # Get trainable parameters
    trainable_params = list(model.parameters())

    if loss_config['learnable_params']:
        trainable_params += list(loss.parameters())
    
    # Define optimizer
    optimizer_type = optimizer_config['type']
    optimizer_params = optimizer_config['params']
    optimizer = OPTIMIZER_NAME_to_CLASS[optimizer_type](trainable_params, **optimizer_params)

    # Initialize Tensorboard SummaryWriter
    writer = SummaryWriter(log_path)
    
    # Initialize number of epochs
    range_epochs = range(epochs)
    n_epochs = len(range_epochs)

    # Initialize variables & define tqdm iterator
    iter_num, best_mean_dice, best_mean_jaccard = 0, 0.0, 0.0

    # Saving model before epoch training
    logger.info(f"[INFO] Saving model before training ...")
    torch.save(model, checkpoint_file)
    with open(results_file, "w") as f:
        json.dump({"epoch": 0}, f, indent = 2)

    # Loop over epochs
    print('\n')
    for epoch_idx in range_epochs:

        # Initialize progress bar
        train_progress_bar = initialize_progress_bar()

        with train_progress_bar:
            
            task = train_progress_bar.add_task(description = ":arrows_counterclockwise: "
                                                   "[TRAIN] "
                                                  f"[Epoch: {epoch_idx}/{n_epochs - 1}] "
                                                  f"[Batch: {0}/{n_train_batches - 1}] "
                                                  f"[TotalLoss: {None}]\n"
                                                  f"[LossValues] - {None}\n"
                                                  f"[LossContrib] - {None}\n"
                                                  f"[LossParams] - {None}",
                                     total = n_train_batches
                            )
            
            for (batch_idx, train_sampled_batch) in enumerate(train_dataloader):
                
                # Retrieve training images & segmentation masks
                train_images = train_sampled_batch['raw_image'].to(device)                                          # shape: (B, 1, H, W)
                train_lumen = train_sampled_batch['lumen'].to(device)                                               # shape: (B, H, W)
                train_plaque = train_sampled_batch['plaque'].to(device)                                             # shape: (B, H, W)

                # Post-process `lumen`, by eliminating `plaque`
                post_processed_train_lumen = post_process_targets(lumen = train_lumen,                             # shape: (B, H, W)
                                                                  plaque = train_plaque)

                # Compute forward pass
                outputs = model(train_images)                                                                       # shape: (B, N_cls, H, W)

                # Compute loss
                loss_value, stop_gradient_values, loss_contributions = loss(outputs, post_processed_train_lumen)
                
                # Retrieve & append the tasks parameters
                loss_params = torch.exp(loss.tasks_log_noise_params).detach()
                
                if loss_params.dim() > 0:
                    loss_params = loss_params
                    formatted_loss_params = {loss_name: loss_params[i].item() 
                                                for (i, loss_name) in enumerate(loss_names) 
                                            }
                else:
                    formatted_loss_params = {'loss_tasks_temperature': loss_params.item()}

                # Update progress bar
                # Show the value of the average training batch loss
                formatted_stop_gradient_values = ' '.join([f"[{n}: {sg_v:.4f}]" for (n, sg_v) in stop_gradient_values.items()])
                formatted_stop_gradient_values = escape(formatted_stop_gradient_values)
                
                formatted_loss_contributions = ' '.join([f"[{n}: {sg_v:.4f}]" for (n, sg_v) in loss_contributions.items()])
                formatted_loss_contributions = escape(formatted_loss_contributions)
                
                formatted_loss_params = ' '.join([f"[{n}: {p:.4f}]" for (n, p) in formatted_loss_params.items()])
                formatted_loss_params = escape(formatted_loss_params)

                train_progress_bar.update(task, 
                                          advance = 1,
                                          description = ":arrows_counterclockwise: "
                                                    "[TRAIN] "
                                                    f"[Epoch: {epoch_idx}/{n_epochs - 1}] "
                                                    f"[Batch: {batch_idx}/{n_train_batches - 1}] "
                                                    f"[TotalLoss: {loss_value.item():.4f}]\n"
                                                    f"[LossValues] - {formatted_stop_gradient_values}\n"
                                                    f"[LossContrib] - {formatted_loss_contributions}\n"
                                                    f"[LossParams] - {formatted_loss_params}"
                            )

                # Compute backward pass
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

                # Update learning rate
                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                # Update current number of iterations
                iter_num += 1

                # Log current learning rate & the value of the total loss
                writer.add_scalar('info/lr', lr_, iter_num)
                writer.add_scalar('info/total_loss', loss_value, iter_num)

                # Log the values of the losses components
                for (loss_name, loss_component_value) in stop_gradient_values.items():
                    writer.add_scalar(f'info/{loss_name}_loss', loss_component_value, iter_num)

                # Log the values of the losses contributions
                for (loss_name, loss_weight) in loss_contributions.items():
                    writer.add_scalar(f'info/{loss_name}_weight', loss_weight, iter_num)

                # Log `training` images / masks & predictions
                if iter_num % 20 == 0:
                    
                    # In the old version we had `train_images[1, 0:1, :, :]` & `train_labels[1, ...]`,
                    # but since at the last batch item it happens that the actual batch size is 1,
                    # hence the shapes (1, 1, H, W) & (1, H, W),
                    # we will always consider from now on `train_images[0, 0:1, :, :]` & `train_labels[0, ...]`
                    inference_idx = 0

                    train_img = train_images[inference_idx, 0:1, :, :]                                              # shape: (B, H, W)                            
                    train_img = (train_img - train_img.min()) / (train_img.max() - train_img.min() + 1e-12)         # shape: (B, H, W)  

                    outputs_argmax = torch.argmax(torch.softmax(outputs, dim = 1), dim = 1, keepdim = True)         # shape: (B, 1, H, W)
                    
                    train_lb = train_lumen[inference_idx, ...].unsqueeze(0) * 50                                    # shape: (1, H, W)
                    post_processed_train_lb = post_processed_train_lumen[inference_idx, ...].unsqueeze(0) * 50      # shape: (1, H, W)

                    writer.add_image('train/Image', train_img, iter_num)
                    writer.add_image('train/Prediction', outputs_argmax[inference_idx, ...] * 50, iter_num)
                    writer.add_image('train/GroundTruth', train_lb, iter_num)
                    writer.add_image('train/PostProcessedGroundTruth', post_processed_train_lb, iter_num)

                # Test the model on the validation dataloader
                if iter_num > 0 and iter_num % 500 == 0:
                    
                    # Set the model to evaluation mode
                    model.eval()

                    # Initialize the metric list which will be needed later
                    metric_list = None

                    # Loop over validation batches
                    for val_sampled_batch in val_dataloader:
                        
                        val_case_name = val_sampled_batch['case_name'][0]

                        val_image = val_sampled_batch['raw_image'].to(device)                                        # shape: (1, 1, H, W)
                        val_lumen = val_sampled_batch['lumen'].to(device)                                            # shape: (1, H, W)
                        val_plaque = val_sampled_batch['plaque'].to(device)                                          # shape: (1, H, W)

                        # Post-process `lumen`, by eliminating `plaque`
                        post_processed_val_lumen = post_process_targets(lumen = val_lumen,                           # shape: (1, H, W)
                                                                        plaque = val_plaque)

                        val_image = val_image.squeeze(0)                                                             # shape: (1, H, W)
                        post_processed_val_lumen = post_processed_val_lumen.squeeze(0)                               # shape: (H, W)
                        val_plaque = val_plaque.squeeze(0)                                                           # shape: (H, W)

                        # Get the values of the segmentation metrics
                        metric_i = validate_model(image = val_image, 
                                                  label = post_processed_val_lumen, 
                                                  model = model, 
                                                  n_classes = n_classes,
                                                  label_encoder = val_label_encoder,
                                                  save_path = None,
                                                  case = val_case_name
                                            )
                        
                        # Initialize metric list for each type of metric
                        if metric_list is None:
                            metric_list = []
                            for class_metrics in metric_i:
                                zero_metrics = {k: 0.0 for k in class_metrics.keys()}
                                metric_list.append(zero_metrics)
                        
                        # Sum metrics per class
                        for (i, class_metrics) in enumerate(metric_i):
                            for (k, v) in class_metrics.items():
                                metric_list[i][k] += v


                    # Get values averaged over metrics
                    for class_metrics in metric_list:
                        for k in class_metrics.keys():
                            class_metrics[k] /= n_val_images

                    # Log the metrics values corresponding to the validation results per class
                    for class_i in range(n_classes - 1):
                        class_metrics = metric_list[class_i]

                        writer.add_scalar(f'info/val_class_{class_i + 1}_dice', class_metrics['dice'], iter_num)
                        writer.add_scalar(f'info/val_class_{class_i + 1}_hd95', class_metrics['hd_95'], iter_num)
                        writer.add_scalar(f'info/val_class_{class_i + 1}_jaccard', class_metrics['jaccard'], iter_num)
                        writer.add_scalar(f'info/val_class_{class_i + 1}_precision', class_metrics['precision'], iter_num)
                        writer.add_scalar(f'info/val_class_{class_i + 1}_recall', class_metrics['recall'], iter_num)


                    # Get values averaged over classes
                    mean_dice = np.mean([m['dice'] for m in metric_list])
                    mean_hd95 = np.mean([m['hd_95'] for m in metric_list])
                    mean_jaccard = np.mean([m['jaccard'] for m in metric_list])
                    mean_precision = np.mean([m['precision'] for m in metric_list])
                    mean_recall = np.mean([m['recall'] for m in metric_list])

                    # Log the mean values into Tensorboard
                    writer.add_scalar('info/val_mean_dice', mean_dice, iter_num)
                    writer.add_scalar('info/val_mean_hd_95', mean_hd95, iter_num)
                    writer.add_scalar('info/val_mean_jaccard', mean_jaccard, iter_num)
                    writer.add_scalar('info/val_mean_precision', mean_precision, iter_num)
                    writer.add_scalar('info/val_mean_recall', mean_recall, iter_num)

                    # Save model if the computed `Jaccard` or `Dice` values are higher
                    if mean_jaccard > best_mean_jaccard:
                        train_progress_bar.console.log(
                                                        f"[INFO] Saving model with `mean_jaccard: {mean_jaccard:.4f}` > `best_mean_jaccard: {best_mean_jaccard:.4f}` ..."
                                                    )
                        best_mean_jaccard = mean_jaccard
                        torch.save(model, checkpoint_file)

                        with open(results_file, "w") as f:
                            json.dump({
                                            "epoch": epoch_idx,
                                            "best_mean_jaccard": best_mean_jaccard,
                                            "best_mean_dice": best_mean_dice
                                      }, f, indent = 2
                                    )

                    elif mean_dice > best_mean_dice:
                        train_progress_bar.console.log(
                                                        f"[INFO] Saving model with `mean_dice: {mean_dice:.4f}` > `best_mean_dice: {best_mean_dice:.4f}` ..."
                                                    )
                        best_mean_dice = mean_dice
                        torch.save(model, checkpoint_file)

                        with open(results_file, "w") as f:
                            json.dump({
                                            "epoch": epoch_idx,
                                            "best_mean_jaccard": best_mean_jaccard,
                                            "best_mean_dice": best_mean_dice
                                      }, f, indent = 2
                                    )

                    # Print the mean validation values
                    train_progress_bar.console.log(
                                                    f"[VAL] iteration {iter_num} : mean_dice : {mean_dice:.4f} "
                                                    f"mean_hd_95 : {mean_hd95:.4f} mean_jaccard : {mean_jaccard:.4f} "
                                                    f"mean_precision : {mean_precision:.4f} mean_recall : {mean_recall:.4f}"
                                                )

                    # Set the model to the training mode
                    model.train()

    # Save the initial configuration settings
    shutil.copy( config_file, os.path.join(settings_path, "config.yaml") )

    # Make inference on `train` dataset
    train_progress_bar.console.log("\nINFERENCE on `train` dataset ...")

    del train_dataloader
    train_config = deepcopy(data_config)
    train_config['batch_size'] = 1
    train_dataloader, _, _ = create_train_val_dataloaders(data_config = train_config,
                                                          train_aug_config = {})
    
    model_inference(config_filename = config_filename, 
                    inference_dataloader = train_dataloader,
                    data_type = 'train'
                )
    
    # Make inference on `validation` dataset
    train_progress_bar.console.log("\nINFERENCE on `val` dataset ...")
    model_inference(config_filename = config_filename, 
                    inference_dataloader = val_dataloader,
                    data_type = 'val'
                )    

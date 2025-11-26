# Load modules
import os, yaml, cv2, torch

import numpy as np
import pandas as pd
import torch.nn as nn

from medpy import metric as medpy_metric
from typing import (Optional, Union, List, Dict)

from .initialize_model import get_model
from utils.utils import (logger,
                         post_process_targets, 
                         initialize_progress_bar, 
                         find_file_in_subfolders)

from torch.utils.data import DataLoader
from dataloader.dataset import (create_test_dataloader, get_base_dataset)



def calculate_metric_percase(pred: np.ndarray, 
                             gt: np.ndarray
                ) -> Dict[str, float]:
    """
        Compute multiple binary segmentation metrics for a single class prediction.
    """
    
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    metrics = dict()

    if pred.sum() > 0 and gt.sum() > 0:
        metrics['dice'] = medpy_metric.binary.dc(pred, gt)
        metrics['hd_95'] = medpy_metric.binary.hd95(pred, gt)
        metrics['jaccard'] = medpy_metric.binary.jc(pred, gt)
        metrics['precision'] = medpy_metric.binary.precision(pred, gt)
        metrics['recall'] = medpy_metric.binary.recall(pred, gt)

    elif pred.sum() > 0 and gt.sum() == 0:
        metrics['dice'] = 1.0
        metrics['hd_95'] = 0.0
        metrics['jaccard'] = 1.0
        metrics['precision'] = 1.0
        metrics['recall'] = 0.0
    
    else:
        metrics['dice'] = 0.0
        metrics['hd_95'] = 0.0
        metrics['jaccard'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0

    return metrics



def validate_model(image: Union[np.ndarray, torch.Tensor],
                   label: Union[np.ndarray, torch.Tensor],
                   model: nn.Module,
                   n_classes: int,
                   label_encoder: Dict,
                   save_path: Optional[str] = None,
                   case: Optional[str] = None
            ) -> List[Dict]:
    """
        Evaluate a single image and return per-class metrics.

        Args:
            image (np.ndarray or torch.Tensor): Input image with shape (1, H, W).
            label (np.ndarray or torch.Tensor): Ground truth label with shape (H, W).
            model (torch.nn.Module): Trained segmentation model.
            n_classes (int): Number of classes.
            label_encoder (Dict): The label encoder defined in the dataset class.
            save_path (str, optional): Directory to save prediction and ground-truth PNGs.
            case (str, optional): Case identifier used in saved filenames.

        Returns:
            metric_list (List[dict]): One dictionary per class (except background) with metric values.
    """

    # Set model in evaluation mode
    model.eval()
    device = next(model.parameters()).device

    # Convert to Numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()                                                                # shape: (1, H, W)

    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()                                                                # shape: (H, W)

    input_tensor = torch.from_numpy(image).unsqueeze(0).float()                                    # shape: (1, 1, H, W)
    input_tensor = input_tensor.to(device)                                                         # shape: (1, 1, H, W)

    with torch.no_grad():
        outputs = model(input_tensor)                                                              # shape: (1, N_cls, H, W)
        preds = torch.argmax(torch.softmax(outputs, dim = 1), dim = 1).squeeze(0)                  # shape: (H, W)
        prediction = preds.cpu().numpy()                                                           # shape: (H, W)

    # Compute metrics per class (excluding background 0)
    metric_list = []
    
    for i in range(1, n_classes):
        metric_list.append( calculate_metric_percase(prediction == i, label == i) )

    # Save predictions
    if ( (case is not None) and (save_path is not None) ):

        prediction_flatten = prediction.flatten()                                                  # shape: (H * W, )

        inverse_label_encoder = {v: k for (k, v) in label_encoder.items()}
        inverse_label_map_func = np.vectorize(inverse_label_encoder.get)
        
        prediction_decoded = inverse_label_map_func(prediction_flatten)                            # shape: (H * W, )
        prediction_decoded = prediction_decoded.reshape(prediction.shape).astype(np.uint8)         # shape: (H, W)

        cv2.imwrite(
                        os.path.join(save_path, f"{case}_pred.png"), 
                        prediction_decoded.astype(np.uint8)
                   )

    return metric_list


def model_inference(config_filename: str,
                    inference_dataloader: Optional[DataLoader],
                    data_type: str
            ) -> None:
    """
        Run inference on the test dataset.
        
        Args:
            config_filename (str): The name of the model which needs to be trained.
            dataloader (torch.utils.data.DataLoader): The input dataloader. If it is `None`, then
                the underlying inference dataloader will be created.
            data_type (str): The type of data associated with the (possibly) given dataloader.
                Choose between `"val"` or `"test"`. 
    """

    assert data_type in ["train", "val", "test"], f"The data type must be either `train`, `val` or `test`, but got `{data_type}`."
    
    # Retrieve paths associated to the saved model
    experiments_path = os.path.abspath('experiments')
    model_path = os.path.join( experiments_path, f"{config_filename}" )

    train_path = os.path.join(model_path, 'train')

    checkpoint_path = os.path.join(train_path, 'checkpoint')
    checkpoints_file = os.path.join(checkpoint_path, 'best_model.pth')

    config_path = os.path.join( train_path, "settings" )
    config_file = os.path.join( config_path, "config.yaml" )

    # Create the inference output directory
    inference_path = os.path.join(model_path, data_type)
    predictions_path = os.path.join(inference_path, f'{data_type}_preds')

    os.makedirs(inference_path, exist_ok = True)
    os.makedirs(predictions_path, exist_ok = True)

    # Read the configuration file
    with open(config_file, 'r') as f:
        args = yaml.safe_load(f)

    # Retrieve configuration parameters
    data_config = args['data']
    model_config = args['model']

    # Create inference dataloader    
    if inference_dataloader is None:
        inference_dataloader = create_test_dataloader(data_config = data_config)
    
    n_inference_images = len(inference_dataloader.dataset)

    print('\n')
    logger.info(f"Number of `{data_type}` images: {n_inference_images}.")

    inference_label_encoder = get_base_dataset(inference_dataloader.dataset).label_encoder
    n_classes = len(inference_label_encoder)

    # Print number of inference iterations per epoch
    n_inference_batches = len(inference_dataloader)

    print('\n')
    logger.info(f"Number of `{data_type}` batches: {n_inference_batches}.")

    # Initialize PyTorch device
    device = model_config['device']

    # Define model
    in_channels, *img_size = inference_dataloader.dataset[0]['raw_image'].shape

    assert img_size[0] == img_size[1], f"Dimensions must be equal, but got `{img_size}`."
    img_size = img_size[0]

    model_name = find_file_in_subfolders( os.path.abspath('configs'), config_filename )
    model = get_model(model_name = model_name, 
                      model_config = model_config, 
                      img_size = img_size, 
                      in_channels = in_channels,
                      n_classes = n_classes, 
                      bs = 1,
                      device = device,
                      eval_mode = True
            )

    # Load model weights
    logger.info(f'Loading model from [{checkpoints_file}].')
    print('\n')
    
    model = torch.load(checkpoints_file, map_location = device)
    model.to(device)

    # Put model in evaluation mode
    model.eval()
    
    # Initialize variables
    per_image_metrics_list = []
    accumulated_metrics_per_class = None

    # Initialize progress bar
    inference_progress_bar = initialize_progress_bar()

    with inference_progress_bar:
        
        task = inference_progress_bar.add_task(description = ":arrows_counterclockwise: "
                                                f"[{data_type.upper()}] "
                                                f"[Batch: {0}/{n_inference_batches - 1}] "
                                                 "[CaseName: ``]\n"
                                                f"[MeanDice] - {0.0}\n"
                                                f"[MeanHd95] - {1.0}\n"
                                                f"[MeanJaccard] - {0.0}\n"
                                                f"[MeanPrecision] - {0.0}\n"
                                                f"[MeanRecall] - {0.0}\n",
                                    total = n_inference_batches
                        )

        # Loop over the inference batches
        for (batch_idx, inference_sampled_batch) in enumerate(inference_dataloader):

            # Retrieve inference images & segmentation masks
            inference_case_name = inference_sampled_batch['case_name'][0]

            inference_image = inference_sampled_batch['raw_image'].to(device)                                     # shape: (B, 1, H, W)
            inference_lumen = inference_sampled_batch['lumen'].to(device)                                         # shape: (B, H, W)
            inference_plaque = inference_sampled_batch['plaque'].to(device)                                       # shape: (B, H, W)

            # Post-process `lumen`, by eliminating `plaque`
            post_processed_inference_lumen = post_process_targets(lumen = inference_lumen,                        # shape: (B, H, W)
                                                                  plaque = inference_plaque)

            inference_image = inference_image.squeeze(0)                                                          # shape: (1, H, W)
            post_processed_inference_lumen = post_processed_inference_lumen.squeeze(0)                            # shape: (H, W)
            inference_plaque = inference_plaque.squeeze(0)                                                        # shape: (H, W)
            
            # Get the values of the segmentation metrics
            metric_i = validate_model(image = inference_image, 
                                      label = post_processed_inference_lumen, 
                                      model = model, 
                                      n_classes = n_classes,
                                      label_encoder = inference_label_encoder,
                                      save_path = predictions_path,
                                      case = inference_case_name
                                )

            # Initialize metric list for each type of metric
            if accumulated_metrics_per_class is None:
                accumulated_metrics_per_class = []
                for class_metrics in metric_i:
                    zero_metrics = {k: 0.0 for k in class_metrics.keys()}
                    accumulated_metrics_per_class.append(zero_metrics)
            
            # Sum metrics per class
            for (i, class_metrics) in enumerate(metric_i):
                for (k, v) in class_metrics.items():
                    accumulated_metrics_per_class[i][k] += v

            # Print values corresponding to the current image
            mean_dice = np.mean([m['dice'] for m in metric_i])
            mean_hd95 = np.mean([m['hd_95'] for m in metric_i])
            mean_jaccard = np.mean([m['jaccard'] for m in metric_i])
            mean_precision = np.mean([m['precision'] for m in metric_i])
            mean_recall = np.mean([m['recall'] for m in metric_i])

            inference_progress_bar.update(task, 
                                     advance = 1,
                                     description = ":arrows_counterclockwise: "
                                            f"[{data_type.upper()}] "
                                            f"[Batch: {batch_idx}/{n_inference_batches - 1}] "
                                            f"[CaseName: `{inference_case_name}`]\n"
                                            f"[MeanDice] - {mean_dice:.4f}\n"
                                            f"[MeanHd95] - {mean_hd95:.4f}\n"
                                            f"[MeanJaccard] - {mean_jaccard:.4f}\n"
                                            f"[MeanPrecision] - {mean_precision:.4f}\n"
                                            f"[MeanRecall] - {mean_recall:.4f}\n"
                            )
            
            # Compute metrics per current image
            per_image_metrics = {'case_name': inference_case_name}

            for (i, class_metrics) in enumerate(metric_i, start = 1):
                for (k, v) in class_metrics.items():
                    per_image_metrics[f'class{i}_{k}'] = v
            
            per_image_metrics_list.append(per_image_metrics)


    # Delete variables
    del mean_dice, mean_hd95, mean_jaccard, mean_precision, mean_recall

    # Get values averaged over metrics
    for class_metrics in accumulated_metrics_per_class:
        for k in class_metrics.keys():
            class_metrics[k] /= n_inference_images

    # Print accumulated metrics per class
    for i in range(1, n_classes):
        cls_metrics = accumulated_metrics_per_class[i - 1]
        inference_progress_bar.console.log(f'Mean class {i} metrics: ' + ', '.join([f'{k} = {v:.4f}' for (k, v) in cls_metrics.items()]))

    # Get values averaged over classes
    mean_dice = np.mean([m['dice'] for m in accumulated_metrics_per_class])
    mean_hd95 = np.mean([m['hd_95'] for m in accumulated_metrics_per_class])
    mean_jaccard = np.mean([m['jaccard'] for m in accumulated_metrics_per_class])
    mean_precision = np.mean([m['precision'] for m in accumulated_metrics_per_class])
    mean_recall = np.mean([m['recall'] for m in accumulated_metrics_per_class])

    inference_progress_bar.console.log(f'Testing performance in `best {data_type} model`: mean_dice {mean_dice:.4f} mean_hd_95 {mean_hd95:.4f}'
                                       f'mean_jaccard {mean_jaccard:.4f} mean_precision {mean_precision:.4f} mean_recall {mean_recall:.4f}'
                                    )

    # Save the mean value metrics corresponding to all the inference images
    summary_path = os.path.join(experiments_path, f"{data_type.upper()}__models_summary_metrics.csv")
    model_summary = {'model_name': config_filename}

    for (metric_name, metric_value) in {'dice': mean_dice, 
                                        'hd_95': mean_hd95, 
                                        'jaccard': mean_jaccard, 
                                        'precision': mean_precision, 
                                        'recall': mean_recall
                                    }.items():
        model_summary[f'mean_{metric_name}'] = metric_value

    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        if config_filename not in df_summary['model_name'].values:
            df_summary = pd.concat([df_summary, pd.DataFrame([model_summary])], ignore_index = True)
    else:
        df_summary = pd.DataFrame([model_summary])

    df_summary.to_csv(summary_path, index = False)
    inference_progress_bar.console.log(f"Saved summary metrics dataframe at `{summary_path}`.")

    # Save the metrics results corresponding to the inference images
    per_image_csv_path = os.path.join(inference_path, f"{data_type.upper()}__image_results.csv")

    df_per_image = pd.DataFrame(per_image_metrics_list)
    df_per_image.to_csv(per_image_csv_path, index = False)

    inference_progress_bar.console.log(f"Saved per-image metrics dataframe at `{per_image_csv_path}`.")


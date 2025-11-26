# Load modules
import os, math, torch

import numpy as np
import torch.nn as nn

from utils.utils import logger

from typing import Dict
from losses.uncertainty import (UW_loss, UW_SO_loss)
from torch.optim import (SGD, Adam, AdamW)

from models.MT_UNet.MT_UNet import MTUNet as MTUnet_224
from models.MT_UNet.Dynamic_MT_UNet import MTUNet as MTUNet_dynamic

from models.Trans_UNet.vit_seg_modeling import VisionTransformer as TransUNet
from models.Trans_UNet.vit_seg_modeling import CONFIGS as TransUNet_VIT_configs

from models.Swin_UNet.config import get_config as get_config_SwinUnet
from models.Swin_UNet.config import get_valid_patches_by_window
from models.Swin_UNet.vision_transformer import SwinUnet

import models.SMP as smp
from models.CBIM.utils import get_model as CBIM_model



OPTIMIZER_NAME_to_CLASS = {
                            'SGD': SGD,
                            'Adam': Adam,
                            'AdamW': AdamW
                          }

MODEL_NAMES = ['MT_UNet', 'Trans_UNet', 'Swin_UNet', 'CBIM', 'SMP']

TRANSUNET_SUBMODEL_NAMES = ['ViT-B_16', 'R50+ViT-B_16']

CBIM_SUBMODEL_NAMES = ['unet', 'unet++', 'attention_unet', 'resunet', 'daunet', 'medformer']                                
CBIM_BASIC_BLOCKS = ['SingleConv', 'BasicBlock', 'Bottleneck', 'MBConv', 'FusedMBConv']

SMP_SUBMODEL_NAMES = ['Unet', 'FPN', 'DeepLabV3', 'UnetPlusPlus', 'PSPNet', 'MAnet', 'Linknet']
SMP_ENCODERS = [
                    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                    'mobilenet_v2'
                ]



def get_model(model_name: str,
              model_config: Dict,
              img_size: int,
              in_channels: int,
              n_classes: int,
              bs: int,
              device: str,
              eval_mode: bool
        ) -> nn.Module:
    """
        Initialize a `nn.Module` for the image segmentation tasks.

        Args:
            model_name(`str`): The name of the model which needs to be trained.
            model_config (dict): Configuration dictionary containing training parameters.
            img_size(`int`): The size of the dataset images.
            in_channels(`int`): The number of image channels.
            n_classes(`int`): The number of classes belonging to the dataset.
            bs(`int`): The batch-size used by the dataloader.
            device(`str`): The CPU/CUDA device.
            eval_mode(`bool`): The setting specifying if the model will be trained or evaluated
    """

    assert model_name in MODEL_NAMES

    pretrained = False if eval_mode else model_config['pretrained']
    logger.info(f"Load pretrained model: `{pretrained}`")

    # Case: `MT-Unet`
    if model_name == 'MT_UNet':

        mtunet_class = MTUnet_224 if img_size == 224 else MTUNet_dynamic
        model = mtunet_class(out_ch = n_classes).to(device)

        if pretrained:

            mtunet_weights = 'pretrained_networks/MT_Unet'
            mtunet_weights = os.path.join(mtunet_weights, "ACDC_epoch=87_avg_dcs=0.9161366117718356_avg_hd=1.2015421870433782.pth")
            weights = torch.load(mtunet_weights)
            
            filtered_weights = {k: v for k, v in weights.items() if not k.startswith('SegmentationHead.')}
            model.load_state_dict(filtered_weights, strict = False)

            out_channels = model.SegmentationHead.weight.shape[1]
            model.SegmentationHead = nn.Conv2d(out_channels, n_classes, kernel_size = 1).to(device)


    # Case: `Trans-UNet`
    elif model_name == 'Trans_UNet':

        vit_name = model_config['vit_name']
        assert vit_name in TRANSUNET_SUBMODEL_NAMES

        patch_size = model_config['patch_size']

        config_TrasUnet = TransUNet_VIT_configs[vit_name]

        config_TrasUnet.n_skip = model_config['n_skip']
        config_TrasUnet.n_classes = n_classes
        config_TrasUnet.patch_size = patch_size
        config_TrasUnet.patches.size = (patch_size, patch_size)

        if vit_name.find('R50') != -1:
            config_TrasUnet.patches.grid = (img_size // patch_size, img_size // patch_size)

        model = TransUNet(config = config_TrasUnet, img_size = img_size, num_classes = n_classes, verbose = False).to(device)

        if pretrained:
            model.load_from(weights = np.load(config_TrasUnet.pretrained_path))


    # Case: `Swin-UNet`
    elif model_name == 'Swin_UNet':

        valid_patches = get_valid_patches_by_window( img_size, get_config_SwinUnet(), window_sizes = [4, 7, 14, 16] )

        patch_size = model_config['patch_size']
        window_size = model_config['window_size']

        assert patch_size in valid_patches[window_size]

        config_SwinUnet = get_config_SwinUnet()
        config_SwinUnet.patch_size = patch_size
        config_SwinUnet.window_size = window_size

        model = SwinUnet(config = config_SwinUnet, img_size = img_size, num_classes = n_classes, verbose = False).to(device)

        if pretrained:
            model.load_from(config_SwinUnet)


    # Case: `CBIM`
    elif model_name == 'CBIM':

        assert model_config['network'] in CBIM_SUBMODEL_NAMES

        model_config['in_chan'] = in_channels
        model_config['classes'] = n_classes

        if model_config['network'] == 'medformer':
            assert 'conv_block' in model_config.keys()
            assert model_config['conv_block'] in CBIM_BASIC_BLOCKS

            model_config['conv_num'] = [2,0,0,0,  0,0,2,2]
            model_config['trans_num'] = [0,2,2,2,  2,2,0,0]
            model_config['num_heads'] = [1,4,8,16, 8,4,1,1] 
            model_config['map_size'] = 3
            model_config['expansion'] = 2
            model_config['fusion_depth'] = 2
            model_config['fusion_dim'] = 512
            model_config['fusion_heads'] = 16
            model_config['proj_type'] = 'depthwise'
            model_config['attn_drop'] = 0.0
            model_config['proj_drop'] = 0.0
            model_config['aux_loss'] = False

        else:
            assert 'block' in model_config.keys()
            assert model_config['block'] in CBIM_BASIC_BLOCKS

        if pretrained:
            raise ValueError("CBIM models do not support the option `pretrained = True`.")

        model = CBIM_model(model_config, pretrain = pretrained).to(device)


    # Case: `SMP`
    elif model_name == 'SMP':

        assert model_config['network'] in SMP_SUBMODEL_NAMES
        assert model_config['encoder'] in SMP_ENCODERS

        if ( model_config['network'] == 'DeepLabV3' and bs == 1 and (not eval_mode) ):
            raise ValueError(f"For `DeepLabV3` model from `SMP`, batch size must be strictly greater than 1, "
                             "since we are in training mode.")
                    
        SMP_Model = getattr(smp, model_config['network'])
        model = SMP_Model(
                            encoder_name = model_config['encoder'],
                            encoder_weights = "imagenet",
                            in_channels = in_channels,
                            classes = n_classes
                        ).to(device)


    # Return the initialized model
    return model


def initialize_loss(loss_config: Dict,
                    n_classes: int,
            ) -> nn.Module:
    """
        Initialize the loss function based on the configuration.

        Args:
            loss_config (dict): Dictionary containing the following relevant keys:
                - loss_type (str or None): 'UW' | 'UW_SO_loss'
                - loss_components (dict): Mapping from loss name to its kwargs, e.g.,
                    {
                        'dice': {},
                        'tversky': {'alpha': 0.7, 'beta': 0.3}
                    }
                - learnable_params (bool): Whether loss weighting parameters are learnable.
                - default_param_values (list or float): Initial values for learnable parameters.
            n_classes (int): The number of classes from the dataset.

        Returns:
            nn.Module: A loss function or composite loss module.
    """

    # Retrieve the underlying configuration settings
    losses = loss_config['loss_components']
    loss_type = loss_config['loss_type']
    learnable_params = loss_config['learnable_params']
    default_param_values = loss_config['default_param_values']

    if loss_type == 'UW_loss':
        assert len(default_param_values) == len(losses), "The number of losses must be equal to the number of default parameter values, " \
                                                        f"but got `{len(losses)}` and `{len(default_param_values)}`."
    elif loss_type == 'UW_SO_loss':
         assert isinstance(default_param_values, float), "The default parameter must be a `float`, " \
                                                        f"but got `{type(default_param_values)}`."

    if isinstance(default_param_values, list):
        default_param_values = [math.log(v) for v in default_param_values]
    else:
        default_param_values = math.log(default_param_values)

    # Inject n_classes into each loss component
    losses = {}
    for (name, kwargs) in loss_config['loss_components'].items():
        
        kwargs = {} if kwargs is None else kwargs
        kwargs['n_classes'] = n_classes

        if name in ['ce', 'focal']:
            kwargs['softmax'] = False
        elif name in ['dice', 'jaccard', 'tversky']:
            kwargs['softmax'] = True
        else:
            raise ValueError(f"The `{name}` loss is not implemented.")
        
        losses[name] = kwargs

    # Initialize the uncertainty-aware loss
    if loss_type == 'UW_loss':
        criterion = UW_loss(losses = losses, 
                            learnable_params = learnable_params, 
                            default_param_values = default_param_values
                        )
    elif loss_type == 'UW_SO_loss':
        criterion = UW_SO_loss(losses = losses, 
                               learnable_params = learnable_params, 
                               default_param_values = default_param_values
                        )
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    
    loss_names = list(losses.keys())
    return criterion, loss_names

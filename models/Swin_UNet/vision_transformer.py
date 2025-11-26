# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from utils.utils import logger
import torch.nn.functional as F

def pjoin(*args):
    return '/'.join(args)

import torch
import torch.nn as nn

from torch.nn.modules.utils import _pair
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys




class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, verbose = False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.verbose = verbose

        self.swin_unet = SwinTransformerSys(img_size = img_size,
                                            patch_size = config.patch_size,
                                            in_chans = config.in_chans,
                                            num_classes = self.num_classes,
                                            embed_dim = config.embed_dim,
                                            depths = config.depths,
                                            depths_decoder = config.depths_decoder,
                                            num_heads = config.num_heads,
                                            window_size = config.window_size,
                                            mlp_ratio = config.mlp_ratio,
                                            qkv_bias = config.qkv_bias,
                                            qk_scale = config.qk_scale,
                                            drop_rate = config.drop_rate,
                                            drop_path_rate = config.drop_path_rate,
                                            ape = config.ape,
                                            patch_norm = config.patch_norm,
                                            use_checkpoint = config.gradient_checkpointing,
                                            verbose = verbose
                                    )


    def forward(self, x):
        
        input_h, input_w = x.size(2), x.size(3)

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        
        logits = self.swin_unet(x)

        # Interpolate back to input size if needed
        if logits.size(2) != input_h or logits.size(3) != input_w:
            logits = F.interpolate(logits, size=(input_h, input_w), mode='bilinear', align_corners=False)
        
        return logits


    def load_from(self, config):
        pretrained_path = config.pretrained_path

        # The window size your current model uses
        new_window_size = config.window_size

        if pretrained_path is not None:
            
            if self.verbose:
                logger.info("pretrained_path:{}".format(pretrained_path))

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            
            if "model"  not in pretrained_dict:

                if self.verbose:
                    logger.info("---start load pretrained modle by splitting---")
                
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}

                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        if self.verbose:
                            logger.info(f"delete key: {k}")
                        del pretrained_dict[k]
                
                # Remove patch embedding weights to avoid shape mismatch
                for k in list(pretrained_dict.keys()):
                    if k.startswith("patch_embed."):
                        if self.verbose:
                            logger.info(f"Skipping patch_embed weights: {k}")
                        del pretrained_dict[k]
                
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                if self.verbose:
                    logger.info(msg)
                return
            
            pretrained_dict = pretrained_dict['model']
    
            if self.verbose:
                logger.info("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)

            # Map decoder layers if necessary
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            
            # Interpolate relative_position_bias_table if present
            for k in list(full_dict.keys()):
                if "relative_position_bias_table" in k and k in model_dict:
                    pretrained_bias = full_dict[k]
                    num_heads, old_len = pretrained_bias.shape
                    expected_len = (2 * new_window_size - 1)**2

                    if old_len == expected_len:
                        if pretrained_bias.shape != model_dict[k].shape:
                            if self.verbose:
                                logger.info(f"[LOAD] Interpolating {k} from {pretrained_bias.shape} to {model_dict[k].shape}")
                            full_dict[k] = interpolate_relative_position_bias(pretrained_bias, new_window_size)
                    else:
                        if self.verbose:
                            logger.info(f"[LOAD] Skipping {k} – size {old_len} doesn't match expected {expected_len}")
                        del full_dict[k]
                            
            # Remove weights that don’t match shapes
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        if self.verbose:
                            logger.info("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            # Remove patch embedding weights here too to avoid shape mismatch
            for k in list(full_dict.keys()):
                if k.startswith("patch_embed."):
                    if self.verbose:
                        logger.info(f"Skipping patch_embed weights: {k}")
                    del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            if self.verbose:
                logger.info(msg)

        else:
            print("none pretrain")
 




def interpolate_relative_position_bias(pretrained_bias, new_window_size):
    """
    Interpolates relative position bias from pretrained window size to new window size.
    pretrained_bias: Tensor of shape [num_heads, old_len]
    Returns: Tensor of shape [num_heads, new_len]
    """
    num_heads, old_len = pretrained_bias.shape
    old_size = int(old_len ** 0.5)
    new_len = new_window_size * new_window_size

    # Reshape to [num_heads, old_size, old_size, 1] for interpolation
    bias = pretrained_bias.view(num_heads, old_size, old_size).unsqueeze(1)

    # Interpolate to new window size using bicubic
    bias = F.interpolate(bias, size=(new_window_size, new_window_size), mode='bicubic', align_corners=False)

    # Reshape back to [num_heads, new_len]
    bias = bias.squeeze(1).view(num_heads, new_len)
    return bias
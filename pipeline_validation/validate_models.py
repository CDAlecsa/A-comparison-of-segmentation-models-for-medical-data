# Load modules
import os, torch
import numpy as np
import torch.nn as nn

from models.MT_UNet.MT_UNet import MTUNet as MTUnet_224
from models.MT_UNet.Dynamic_MT_UNet import MTUNet as MTUNet_dynamic

from models.Trans_UNet.vit_seg_modeling import VisionTransformer as TransUNet
from models.Trans_UNet.vit_seg_modeling import CONFIGS as TransUNet_VIT_configs

from models.Swin_UNet.config import get_config as get_config_SwinUnet
from models.Swin_UNet.config import get_valid_patches_by_window
from models.Swin_UNet.vision_transformer import SwinUnet

from models.CBIM.utils import get_model as CBIM_model
import models.SMP as smp



# Run the script from the root folder "main_codes" using the -m flag: "python -m pipeline_validation.validate_models"
if __name__ == "__main__":
    print('\n')

    bs = 2
    in_channels = 1
    n_classes = 2


    ####################################### MT-Unet #######################################
    for img_size in [224, 128]:
        for pretrained in [True, False]:
            
            mtunet_class = MTUnet_224 if img_size == 224 else MTUNet_dynamic
            model = mtunet_class(out_ch = n_classes).cuda()
            str_rep = str(img_size)

            if pretrained:
                mtunet_weights = 'pretrained_networks/MT_Unet'
                mtunet_weights = os.path.join(mtunet_weights, "ACDC_epoch=87_avg_dcs=0.9161366117718356_avg_hd=1.2015421870433782.pth")
                weights = torch.load(mtunet_weights)
                
                filtered_weights = {k: v for k, v in weights.items() if not k.startswith('SegmentationHead.')}
                model.load_state_dict(filtered_weights, strict = False)

                out_channels = model.SegmentationHead.weight.shape[1]
                model.SegmentationHead = nn.Conv2d(out_channels, n_classes, kernel_size = 1).cuda()

            dummy_input = torch.randn(bs, in_channels, img_size, img_size).cuda()
            output = model(dummy_input)            
            input_shape = tuple(dummy_input.shape)
            output_shape = tuple(output.shape)
            print(f"[MTUnet_{str_rep} - pretrained: {pretrained}]: input shape: {input_shape} -> output shape: {output_shape}\n\n")


    ####################################### Trans-UNet #######################################
    for (img_size, patch_size) in zip( [224, 128, 128, 128], [16, 16, 8, 4] ):

        for (vit_name, n_skip) in zip( ['ViT-B_16', 'R50+ViT-B_16'], [0, 3] ):

            for pretrained in [True, False]:

                config_TrasUnet = TransUNet_VIT_configs[vit_name]
                config_TrasUnet.n_skip = n_skip
                config_TrasUnet.n_classes = n_classes

                config_TrasUnet.patch_size = patch_size
                config_TrasUnet.patches.size = (patch_size, patch_size)

                str_rep = str(img_size)

                if vit_name.find('R50') != -1:
                    config_TrasUnet.patches.grid = (img_size // patch_size, img_size // patch_size)

                model = TransUNet(config = config_TrasUnet, img_size = img_size, num_classes = n_classes, verbose = False).cuda()
                if pretrained:
                    model.load_from(weights = np.load(config_TrasUnet.pretrained_path))

                dummy_input = torch.randn(bs, in_channels, img_size, img_size).cuda()
                output = model(dummy_input)
                input_shape = tuple(dummy_input.shape)
                output_shape = tuple(output.shape)

                print(f"[TransUNet_{str_rep} - {vit_name} - pretrained: {pretrained} - patch_size: {config_TrasUnet.patch_size}]: "
                      f"input shape: {input_shape} -> output shape: {output_shape}\n")

        print('\n')

    
    ####################################### Swin-UNet #######################################
    valid_patches = dict()

    for img_size in [224, 128]:
        valid_patches = get_valid_patches_by_window( img_size, get_config_SwinUnet(), window_sizes = [4, 7, 14, 16] )

        for (window, patch) in valid_patches.items():
            print(f"SwinUNet: [img_size = {img_size}] [window = {window}] [valid_patches = {valid_patches[window]}]")
    
    print('\n')

    for (img_size, patch_size, window_size) in zip( [224, 224, 224, 128, 128], [14, 14, 4, 16, 8], [4, 4, 7, 4, 16] ):

        for pretrained in [True, False]:

            config_SwinUnet = get_config_SwinUnet()
            config_SwinUnet.patch_size = patch_size
            config_SwinUnet.window_size = window_size

            str_rep = str(img_size)

            model = SwinUnet(config = config_SwinUnet, img_size = img_size, num_classes = n_classes, verbose = False).cuda()
            if pretrained:
                model.load_from(config_SwinUnet)

            dummy_input = torch.randn(bs, in_channels, img_size, img_size).cuda()
            output = model(dummy_input)
            input_shape = tuple(dummy_input.shape)
            output_shape = tuple(output.shape)

            print(f"[SwinUNet_{str_rep} - pretrained: {pretrained}]: "
                    f"input shape: {input_shape} -> output shape: {output_shape}\n")

    print('\n')

    
    ####################################### CBIM #######################################
    CBIM_allowed_models = ['unet', 'unet++', 'attention_unet', 'resunet', 'daunet', 'medformer']
    CBIM_allowed_basic_blocks = ['SingleConv', 'BasicBlock', 'Bottleneck', 'MBConv', 'FusedMBConv']

    for img_size in [224, 128]:
        for cbim_model_type in CBIM_allowed_models:
            for basic_block in CBIM_allowed_basic_blocks:
                for base_chan in [32, 64]:
                    
                    config = dict()
                    config['model'] = cbim_model_type
                    config['classes'] = n_classes
                    config['in_chan'] = in_channels
                    config['base_chan'] = base_chan

                    if cbim_model_type == 'medformer':
                        config['conv_block'] = basic_block
                        config['conv_num'] = [2,0,0,0,  0,0,2,2]
                        config['trans_num'] = [0,2,2,2,  2,2,0,0]
                        config['num_heads'] = [1,4,8,16, 8,4,1,1] 
                        config['map_size'] = 3
                        config['expansion'] = 2
                        config['fusion_depth'] = 2
                        config['fusion_dim'] = 512
                        config['fusion_heads'] = 16
                        config['proj_type'] = 'depthwise'
                        config['attn_drop'] = 0.0
                        config['proj_drop'] = 0.0
                        config['aux_loss'] = False
                    else:
                        config['block'] = basic_block

                    model = CBIM_model(config, pretrain = False).cuda()
                    str_rep = str(img_size)

                    dummy_input = torch.randn(bs, in_channels, img_size, img_size).cuda()
                    output = model(dummy_input)
                    input_shape = tuple(dummy_input.shape)
                    output_shape = tuple(output.shape)
                    print(f"[CBIM_{str_rep} - model: {cbim_model_type} - basic_block: {basic_block} - base_chan: {base_chan}]: "
                        f"input shape: {input_shape} -> output shape: {output_shape}\n\n")

        print('\n')


    ####################################### SMP #######################################
    SMP_allowed_models = ['Unet', 'FPN', 'DeepLabV3', 'UnetPlusPlus', 'PSPNet', 'MAnet', 'Linknet']

    SMP_allowed_encoders = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                            'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                            'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                            'mobilenet_v2'
                       ]

    for img_size in [224, 128]:
        for smp_model_type in SMP_allowed_models:
            for smp_encoder_type in SMP_allowed_encoders:
                                
                ModelClass = getattr(smp, smp_model_type)
                model = ModelClass(
                                    encoder_name = smp_encoder_type,
                                    encoder_weights = "imagenet",
                                    in_channels = in_channels,
                                    classes = n_classes
                                ).cuda()
                
                if smp_model_type == 'DeepLabV3' and bs == 1:
                    model = model.eval()

                str_rep = str(img_size)

                dummy_input = torch.randn(bs, in_channels, img_size, img_size).cuda()
                output = model(dummy_input)            
                input_shape = tuple(dummy_input.shape)
                output_shape = tuple(output.shape)
                print(f"[SMP_{str_rep} - model: {smp_model_type} - encoder: {smp_encoder_type}]: "
                      f"input shape: {input_shape} -> output shape: {output_shape}\n\n")

        print('\n')
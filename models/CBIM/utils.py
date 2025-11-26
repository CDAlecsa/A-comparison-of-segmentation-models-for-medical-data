import argparse
import torch.nn as nn
import torch.nn.functional as F

from .conv_layers import BasicBlock, Bottleneck, SingleConv, MBConv, FusedMBConv, ConvNeXtBlock



def get_block(name):
    block_map = { 
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
        'MBConv': MBConv,
        'FusedMBConv': FusedMBConv,
        'ConvNeXtBlock': ConvNeXtBlock
    }   
    return block_map[name]



def get_norm(name):
    norm_map = {'bn': nn.BatchNorm3d,
                'in': nn.InstanceNorm3d
                }

    return norm_map[name]



def get_model(args, pretrain = False):
    args = argparse.Namespace(**args)

    if args.network == 'unet':
        from .unet import UNet
        if pretrain:
            raise ValueError('No pretrain model available')
        return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
    if args.network == 'unet++':
        from .unetpp import UNetPlusPlus
        if pretrain:
            raise ValueError('No pretrain model available')
        return UNetPlusPlus(args.in_chan, args.classes, args.base_chan)
    if args.network == 'attention_unet':
        from .attention_unet import AttentionUNet
        if pretrain:
            raise ValueError('No pretrain model available')
        return AttentionUNet(args.in_chan, args.classes, args.base_chan)

    elif args.network == 'resunet':
        from .unet import UNet
        if pretrain:
            raise ValueError('No pretrain model available')
        return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
    elif args.network == 'daunet':
        from .dual_attention_unet import DAUNet
        if pretrain:
            raise ValueError('No pretrain model available')
        return DAUNet(args.in_chan, args.classes, args.base_chan, block=args.block)

    elif args.network in ['medformer']:
        from .medformer import MedFormer
        if pretrain:
            raise ValueError('No pretrain model available')
        return MedFormer(args.in_chan, 
                         args.classes, 
                         args.base_chan, 
                         conv_block=args.conv_block, 
                         conv_num=args.conv_num, 
                         trans_num=args.trans_num, 
                         num_heads=args.num_heads, 
                         fusion_depth=args.fusion_depth, 
                         fusion_dim=args.fusion_dim, 
                         fusion_heads=args.fusion_heads, 
                         map_size=args.map_size, 
                         proj_type=args.proj_type, 
                         act=nn.ReLU, 
                         expansion=args.expansion, 
                         attn_drop=args.attn_drop, 
                         proj_drop=args.proj_drop, 
                         aux_loss=args.aux_loss)


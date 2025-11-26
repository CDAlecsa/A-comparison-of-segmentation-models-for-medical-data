import ml_collections



def get_config():
    """Returns the Swin-UNet configuration."""
    config = ml_collections.ConfigDict()

    config.patch_size = 4
    config.in_chans = 3
    config.embed_dim = 96
    config.depths = [2, 2, 6, 2]
    config.depths_decoder = [2, 2, 6, 2]
    config.num_heads = [3, 6, 12, 24]
    config.window_size = 7
    config.mlp_ratio = 4.0
    config.qkv_bias = True
    config.qk_scale = None
    config.drop_rate = 0.0
    config.drop_path_rate = 0.1
    config.ape = False
    config.patch_norm = True
    config.gradient_checkpointing = False
    config.pretrained_path = 'pretrained_networks/Swin/swin_tiny_patch4_window7_224.pth'

    return config



def get_valid_patches_by_window(img_size, config, window_sizes):
    """
    Returns a dictionary mapping window sizes to lists of valid patch sizes for a given image size.
    
    Args:
        img_size (int): Size of the input image (assumed square).
        config: Configuration object/dict containing 'depths' for downsampling info.
        window_sizes (list of int): List of window sizes to check.
        
    Returns:
        dict: {window_size: [valid_patch_sizes]}
    """
    num_downsamples = len(config.depths) - 1
    results = {}

    for window_size in window_sizes:
        valid_patches = []
        for patch_size in range(1, img_size + 1):
            if img_size % patch_size == 0:
                num_patches_per_dim = img_size // patch_size
                if (num_patches_per_dim % (2 ** num_downsamples) == 0 and
                    num_patches_per_dim % window_size == 0):
                    valid_patches.append(patch_size)
        results[window_size] = valid_patches

    return results



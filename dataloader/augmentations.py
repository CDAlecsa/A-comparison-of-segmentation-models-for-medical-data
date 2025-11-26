# Load modules
import torch, math, warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "albumentations")

import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataAugmenter:
    def __init__(self, aug_config):

        transforms = []

        if 'gaussian_noise' in aug_config:
            p = aug_config['gaussian_noise'].get('prob', 0.3)
            std = aug_config['gaussian_noise'].get('std', 0.02)
            transforms.append(A.GaussNoise(std_range = (std, std), p = p))

        if 'gaussian_blur' in aug_config:
            p = aug_config['gaussian_blur'].get('prob', 0.3)
            sigma_range = aug_config['gaussian_blur'].get('sigma', [0.5, 1.0])
            sigma = torch.rand(1) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
            ksize = 2 * math.ceil(3 * sigma) + 1
            transforms.append(A.GaussianBlur(blur_limit = ksize, sigma_limit = sigma, p = p))

        if 'brightness' in aug_config:
            p = aug_config['brightness'].get('prob', 0.3)
            brightness_range = aug_config['brightness'].get('range', [0.8, 1.2])
            transforms.append(
                                A.RandomBrightnessContrast(
                                                            brightness_limit = (brightness_range[0] - 1, brightness_range[1] - 1),
                                                            contrast_limit = 0,
                                                            p = p
                                                        )
                            )
        
        if 'gamma' in aug_config:
            p = aug_config['gamma'].get('prob', 0.3)
            gamma_range = aug_config['gamma'].get('range', (0.7, 1.5))
            gamma_range = ( int(100 * gamma_range[0]), int(100 * gamma_range[1]) )
            transforms.append(A.RandomGamma(gamma_limit = gamma_range, p = p))

        if 'contrast' in aug_config:
            p = aug_config['contrast'].get('prob', 0.3)
            contrast_range = aug_config['contrast'].get('range', (0.8, 1.2))
            transforms.append(
                                A.RandomBrightnessContrast(
                                                            brightness_limit = 0,
                                                            contrast_limit = (contrast_range[0] - 1, contrast_range[1] - 1),
                                                            p = p
                                                        )
                            )
        if 'mirror' in aug_config:
            p = aug_config['mirror'].get('prob', 0.5)
            transforms.append(
                                A.OneOf(
                                    [
                                        A.HorizontalFlip(p = 1.0),
                                        A.VerticalFlip(p = 1.0),
                                    ], 
                                p = p)
                            )

        if 'affine' in aug_config:
            p = aug_config['affine'].get('prob', 0.3)
            scale = aug_config['affine'].get('scale', 0.1)
            rotate = aug_config['affine'].get('rotate', 10)
            translate = aug_config['affine'].get('translate', 0.1)
            transforms.append(A.Affine(
                                        scale = (1 - scale, 1 + scale),
                                        translate_percent = translate,
                                        rotate = (-rotate, rotate),
                                        p = p
                                    )
                            )

        if 'elastic' in aug_config:
            p = aug_config['elastic'].get('prob', 0.3)
            alpha = aug_config['elastic'].get('alpha', 20)
            sigma = aug_config['elastic'].get('sigma', 4)

            transforms.append(A.ElasticTransform(
                                                    alpha = alpha,
                                                    sigma = sigma,
                                                    p = p
                                                )
                            )
            
        transforms.append(ToTensorV2())
        self.transform = A.Compose(transforms)


    def __call__(self, sample):
        
        case_name = sample['case_name']

        raw_image = sample['raw_image']                                                             # shape: (1, H, W)
        lumen = sample['lumen']                                                                     # shape: (H, W)
        plaque = sample['plaque']                                                                   # shape: (H, W)

        raw_image = raw_image.numpy().transpose(1, 2, 0)                                            # shape: (H, W, 1)
        lumen = lumen.numpy()                                                                       # shape: (H, W)
        plaque = plaque.numpy()                                                                     # shape: (H, W)

        augmented = self.transform(image = raw_image, masks = [lumen, plaque])

        image_aug = augmented['image'].float()                                                      # shape: (1, H, W)
        lumen_aug, plaque_aug = augmented['masks']

        lumen_aug = torch.as_tensor(lumen_aug, dtype = torch.long)                                  # shape: (H, W)
        plaque_aug = torch.as_tensor(plaque_aug, dtype = torch.long)                                # shape: (H, W)

        aug_results =  {'raw_image': image_aug, 
                        'lumen': lumen_aug, 
                        'plaque': plaque_aug, 
                        'case_name': case_name}
        
        return aug_results

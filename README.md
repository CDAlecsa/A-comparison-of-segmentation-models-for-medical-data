# *A comparison of segmentation models for medical data*

[![GitHub Repositories](https://img.shields.io/badge/GitHub_Account-Cristian_Daniel_Alecsa-black?logo=github)](https://github.com/CDAlecsa)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Profile-blue?logo=google-scholar)](https://scholar.google.ro/citations?hl=ro&user=394mWmgAAAAJ&view_op=list_works&sortby=pubdate)

This repository contains the comparison of different segmentation models for medical datasets.

---
### ⚡Prerequisites
- Python 3.10.11
- CPU or NVIDIA GPU + CUDA 11.8
- the main environment packages can be found in [requirements.txt](requirements.txt)
---
### 📁 Dataset Structure
- The user must create an original data folder named `Vascular_pathologies_data`, with the structure mentioned below.  
- Use the script [utils/crop_dataset.py](utils/crop_dataset.py) to create the cropped dataset:  
  - `Vascular_pathologies_data_cropped_128_128` (where `128` is the model output size).  
  - This folder is used by the models and preserves the same structure as the original data.  
  - Includes a folder `png_lumen_crop` with the bounding box of the cropped lumen around the lumen center.

- Folder structure inside the folders `Vascular_pathologies_data` and `Vascular_pathologies_data_cropped_128_128` (except the `png_lumen_crop` subfolders):
```
Vascular_pathologies_data_cropped_128_128/
├─ train_val
│ ├─ patient_trainval_1
│ │ ├─ png_lumen
│ │ ├─ png_mask     (not used)
│ │ ├─ png_mdc      (not used)
│ │ ├─ png_plaque
│ │ └─ png_raw
│ └─ ...
└─ test
├─ patient_test_1
│ ├─ png_lumen
│ ├─ png_mask       (not used)
│ ├─ png_mdc        (not used)
│ ├─ png_plaque
│ └─ png_raw
└─ ...
```
- Subfolder description:
  - `png_raw` - images acquired without contrast medium
  - `png_lumen` - images representing the lumen of the aortic vessels
  - `png_mask` - images representing the entire aortic vessel 
  - `png_plaque` - images representing calcified plaques
  - `png_mdc` - images acquired with contrast medium
- The only used images are the ones belonging to `png_raw`, `png_lumen` and `png_plaque`.  
- Segmentation targets for models:  ```png_lumen AND (NOT png_plaque)```
> ⚠️ Make sure to create the `Vascular_pathologies_data_cropped_128_128` data folder by processing your original dataset exactly like it is described above.
---
### ⚙️ Segmentation models
All these models are pretrained, except the CBIM models. The MT-UNet models with identifiers MTU_1, MTU_2, and MTU_3 are pretrained on the ACDC dataset, so we modified the segmentation head to have only 2 target classes (corresponding to binary segmentation targets).

| Identifier       | Model        | GitHub repository | Architecture / Notes |
|-----------------|-------------|-----------------|-------------------|
| *CBIM_ATT_UNET_1* | Attention U-Net | CBIM | network=attention_unet <br> block=Bottleneck <br> base_chan=32 <br> pretrained=False |
| *CBIM_DA_UNET_1*  | DANet        | CBIM | network=daunet <br> block=BasicBlock <br> base_chan=32 <br> pretrained=False |
| *CBIM_MEDFORMER_1* | Medformer    | CBIM | network=medformer <br> conv_block=FusedMBConv <br> base_chan=32 <br> pretrained=False |
| *MTU_1*           | MT-UNet      | MT-Unet | pretrained=True |
| *MTU_2*           | MT-UNet      | MT-Unet | pretrained=True |
| *MTU_3*           | MT-UNet      | MT-Unet | pretrained=True |
| *SMP_FPN_1*       | FPN          | SMP | network=FPN <br> encoder=resnet152 <br> pretrained=True |
| *SMP_LINK_NET_1*  | Linknet      | SMP | network=Linknet <br> encoder=efficientnet-b2 <br> pretrained=True |
| *SMP_MA_NET_1*    | Ma-net       | SMP | network=MAnet <br> encoder=mobilenet_v2 <br> pretrained=True |
| *SMP_UNET_1*      | U-net        | SMP | network=Unet <br> encoder=efficientnet-b0 <br> pretrained=True |
| *SMP_UNET_PP_1*   | Unet++       | SMP | network=UnetPlusPlus <br> encoder=efficientnet-b5 <br> pretrained=True |
| *SU_1*            | Swin-Unet    | SwinUnet | patch_size=8 <br> window_size=16 <br> pretrained=True |
| *SU_2*            | Swin-Unet    | SwinUnet | patch_size=16 <br> window_size=4 <br> pretrained=True |
| *SU_3*            | Swin-Unet    | SwinUnet | patch_size=8 <br> window_size=16 <br> pretrained=True |
| *TU_1*            | TransUNet    | TransUnet | vit_name=R50+ViT-B_16 <br> patch_size=8 <br> n_skip=3 <br> pretrained=True |
| *TU_2*            | TransUNet    | TransUnet | vit_name=ViT-B_16 <br> patch_size=8 <br> n_skip=0 <br> pretrained=True |
| *TU_3*            | TransUNet    | TransUnet | vit_name=R50+ViT-B_16 <br> patch_size=8 <br> n_skip=3 <br> pretrained=True |
---
### 🚀 Quick Start

##### 
- After preparing the dataset, for training a single model (example `TU_1` corresponding to settings given in [configs/Trans_UNet/TU_1.yaml](configs/Trans_UNet/TU_1.yaml)) use the command `python train.py --config-file TU_1`
- For training all the models, run the [train_all_models.bat](train_all_models.bat) file (this is only for Windows users, while Linux/macOS users would need to implement an equivalent shell script). 
- For testing all the trained models, run `python test.py`
> ⚠️ It is recommended to make a `.env` file which contains only `PYTHONPATH=${workspaceFolder}`.
---
## 📌 License and Attribution
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This repository is intended for **academic and research purposes**. 
- The following folders contain code developed in this repository and are licensed under the [MIT License](./LICENSE):  
  - `configs` – configuration files  
  - `dataloader` – dataset loading and preprocessing scripts  
  - `losses` – loss functions, including uncertainty-aware losses  
  - `network_pipeline` – training and inference scripts (including [network_pipeline/training.py](network_pipeline/training.py) and [network_pipeline/inference.py](network_pipeline/inference.py), which are adapted and modified versions of the *TransUNet pipeline* for this project)
  - `pipeline_validation` – scripts for code validation
  - `utils` – utility scripts & helper functions

- The `models` folder contains the (original or slightly modified)  codes corresponding to the architectures belonging to the following repositories, integrated directly for **pipeline consistency and academic comparisons**:
  - MT-UNet
  - Swin-Unet
  - TransUNet
  - Segmentation Models PyTorch
  - CBIM - Medical Image Segmentation
- The `pretrained_networks` folder must be created and it needs to contain the pretrained weights (obtained from the original repositories `MT-UNet`, `Swin Transformer` and `TransUNet`):
```
pretrained_networks/
├─ MT_UNet/
│ └─ ACDC_epoch=87_avg_dcs=0.9161366117718356_avg_hd=1.2015421870433782.pth
├─ Swin/
│ └─ swin_tiny_patch4_window7_224.pth
├─ VIT/
│ ├─ R50+ViT-B_16.npz
│ └─ ViT-B_16.npz
└─ __init__.py        (empty file)
```

> ⚠️ Make sure to download and place the pretrained weights in the `pretrained_networks` folder before training.

> **Note**: *Once the folders and files are created, the code will automatically detect them. No further attribution is required for the pretrained weights in terms of license, since they are obtained directly from the original repositories.*

> **Note**: *While these models are included here for ease of use, the original repositories maintain their own licenses. Users should respect the terms of the original repositories when using these models or weights.*

---
### 🎖️ Original Repositories
The following *GitHub repositories* correspond to the model implementations & pretrained weights:
- [MT-UNet](https://github.com/Dootmaan/MT-UNet/)
- [Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- [Segmentation Models Pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [CBIM - Medical Image Segmentation](https://github.com/yhygao/CBIM-Medical-Image-Segmentation)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)

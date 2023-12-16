## Comparing U-Net, U-Net++, and DeepLabV3+ for Car Part Segmentation

## Abstract

This GitHub repository houses the implementation and evaluation of three image segmentation models, U-Net, U-Net++, and DeepLabV3+, for car part segmentation. Developed as part of the **02456 Deep Learning** course at **Danmarks Tekniske Universitet (DTU)**, this project addresses the growing demand for accurate multi-class image segmentation techniques in the automotive industry. The models were trained and evaluated on a dataset comprising real and synthetic car images, with performance metrics such as the Dice coefficient, mIoU, and Pixel Accuracy utilized for assessment. 

## Project Structure

```
. project 
├─ dataset
│  └─ (empty)
├─ models
│  ├─ DeepLabV3Plus_ResNet18.py
│  ├─ DeepLabV3Plus_ResNet50.py
│  ├─ DeepLabV3Plus_ResNet101.py
│  ├─ UNet.py
│  └─ UNetPlusPlus.py
├─ notebooks
│  ├─ deeplabv3plus_train.ipynb
│  ├─ unet_train.ipynb
│  └─ unetplusplus_train.ipynb
├─ scripts
│  ├─ predict.py
│  └─ train.py
├─ utils
│  ├─ dataloaders.py
│  ├─ metrics.py
│  └─ plot_utils.py
├─ weights
│  └─ (empty)
└─ Results.ipynb
```
## Usage

For those interested, you can explore the codebase to examine the pipeline followed.

## Results

The outcomes of this project are documented in the **Results.ipynb** notebook. Due to confidentiality reasons, the dataset is not available on **GitHub** as it contains proprietary information from **Deloitte**. Also, It has been decided not to upload the weights of all the models due to their substantial file sizes.

### Note: Do Not Run Results.ipynb

Please note that running the **Results.ipynb** notebook is unnecessary for typical usage, as it already contains the finalized results of the project.

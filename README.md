# SyNET
## Analyzing the Efficacy of Synthetic Data in Pre-training and Downstream Tasks

This research investigates the use of synthetic data during both the pre-training and downstream training stages.

The HOPE dataset (https://research.nvidia.com/publication/2021-09_multi-view-fusion-multi-level-robotic-scene-understanding) was used to get the object meshes to generate synthetic data. Frames from the HOPE-Video dataset were used for testing.

The dataset that I created can be downloaded here: [coco_train](https://drive.google.com/drive/folders/10CJ1DbVGPcFO8Rkr_xpKV5gMbXH9Jnd5?usp=share_link) [coco_val](https://drive.google.com/drive/folders/1025Fdw_pURCbx_TD5cuSasKFEeIt2Ciq?usp=share_link)

The main notebook is synet_train. In order to run it, make sure to update the paths to the coco_train and coco_val directories. 

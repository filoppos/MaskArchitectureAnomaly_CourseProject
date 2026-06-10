# Mask Architecture for Road Scenes
This is the starting repository for two projects:
- Mask Architecture Anomaly Segmentation for Road Scenes  [[Project Description](https://drive.google.com/file/d/1Vz08DHsP_mojpCTAQTR6NHVq-2rEqAZM/view?usp=sharing)]
- Comprehensive Road Scene Understanding for Autonomous Driving  [[Project Description](https://drive.google.com/file/d/1tq5F_j_8O2vlGWbkU1ayPjYvCml1VEwr/view?usp=sharing)]

This repository consists of the code base for training/testing ERFNet on the Cityscapes dataset and perform anomaly segmentation. It also contains some code referring to EoMT. Some of this code may be unnecessary for your project.

## Folders
For instructions, please refer to the README in each folder:

* [eval](eval) contains tools for evaluating/visualizing an ERFNet model's output and performing anomaly segmentation.
* [trained_models](trained_models) Contains the ERFNet trained models for the baseline eval. 
* [eomt](eomt) It is almost the original folder of the EoMT project. Inside it you will find code to train and pretrained checkpoints for EoMT.

* step4_visualization.ipynb: Qualitative analysis of EoMT predictions on Cityscapes. Includes visual inspection of segmentation outputs and label mapping between dataset annotation formats.

* step5_finetuning.ipynb: Fine-tuning of EoMT on Cityscapes with four freezing strategies- Analyzes trainable parameter distribution.

* step7_erfnet_anomaly.ipynb: ERFNet anomaly scoring pipeline. Implements MSP, MaxLogit, and MaxEntropy scoring methods evaluated on five anomaly datasets. Reports AuPRC and FPR95 metrics.
  
* step8_eomt_anomaly.ipynb: EoMT anomaly scoring pipeline. Adapts the mask transformer architecture (per-query logits via einsum) to MSP (with temperature scaling), MaxLogit, MaxEntropy, and RbA scoring.

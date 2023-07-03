# Increasing-Margin-Adversarial-Training
The official source code for the paper "Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks" https://arxiv.org/abs/2005.09147

This paper has been published by Computer Methods and Programs in Biomedicine, and the latest version is available at https://doi.org/10.1016/j.cmpb.2023.107687.

## Classification

### CIFAR10
1. run CIFAR10_CNNM_IMA.py to get IMA model
2. run CIFAR10_CNNM_ce.py to get the defenseless model

### PathMNIST
1. run MedMNIST_CNN_IMA.py to get IMA model
2. run MedMNIST_CNN_ce.py to get the defenseless model

### COVID-19
0. The data is downloaded from https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset
1. run COVID19a_CNN_IMA.py to get IMA model
2. run COVID19a_CNN_ce.py to get the defenseless model


## Segmentation
1. Please go to this link(https://github.com/MIC-DKFZ/nnUNet) for instructions on installing and configuring nnUnet and dependent libraries.
2. All the experimental data can be downloaded from http://medicaldecathlon.com/ or as shown in paper(Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). nnU-Net: a self-configuring method 
for deep learning-based biomedical image segmentation. Nature Methods, 1-9.)
3. Go to "nnunet/run/"
4. use "python run_training --task id" to get the baseline model
5. use "python run_IMA_training --task id" to get the IMA model
6. use "python run_PGD_training --task id" to get the PGD model
7. use "python run_TE_training --task id" to get the TE model
8. use "python run_TRADES_training --task id" to get the TRADES model
9. use "run_test.py" to get the evaluation result from autoattack and white attack.
10. use "run_extra_test.py" to get the evaluation result from FGSM, IFGSM and PGD attack.

The experiment was run on Tesla V100 GPUs, CentOS operating system.

Based on the original nnUnet, we made modifications on:
1.nnunet/training/network_training/network_trainer.py
2.nnunet/training/network_training/nnUNetTrainer.py
3.nnunet/training/network_training/nnUNetTrainerV2.py
4.nnunet/training/loss_functions/dice_loss.py
5.nnunet/training/loss_functions/crossentropy.py
6.nnunet/training/loss_functions/deep_supervision.py
7.nnunet/training/dataloading/dataset_loading.py
8.nnunet/training/data_augmentation/data_augmentation_moreDA.py
9.nnunet/utilities/to_torch.py
10.nnunet/TE/
11.nnunet/TRADES/
12.nnunet/autoattackfornnunet/

## Trained models
Models are available at https://drive.google.com/drive/folders/1G51RnA4HxHivSy8RsKzvhFzkqob8ahLY?usp=sharing


## Contact
Should you have any question, please contact liang.liang@miami.edu or l.ma@miami.edu

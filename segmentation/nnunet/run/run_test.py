#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import inspect
import os
import sys
curDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
curDir = os.path.dirname(curDir)
curDir = os.path.dirname(curDir)
sys.path.insert(0, curDir)

import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.run.load_pretrained_weights import load_pretrained_weights
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes
from nnunet.training.network_training.nnUNetTrainerV2_CascadeFullRes import nnUNetTrainerV2CascadeFullRes
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import matplotlib.pyplot as plt
import numpy as np
import csv

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

def main(noise, filename, taskid, white, norm_type):
    parser = argparse.ArgumentParser()
    """
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task", help="can be task name or task id")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    """
    #parser.add_argument("-norm", "--norm_type",type=str, default = "Linf", action="store_true")
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic",
                        help="Makes training deterministic, but reduces training speed substantially. I (Fabian) think "
                             "this is not necessary. Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true",
                        help="hands off. This is not intended to be used")
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set nnU-Net will not save any parameter files (except a temporary checkpoint that "
                             "will be removed at the end of the training). Useful for development when you are "
                             "only interested in the results and want to save some disk space")
    parser.add_argument("--disable_postprocessing_on_folds", required=False, action='store_true',
                        help="Running postprocessing on each fold only makes sense when developing with nnU-Net and "
                             "closely observing the model performance on specific configurations. You do not need it "
                             "when applying nnU-Net because the postprocessing for this will be determined only once "
                             "all five folds have been trained and nnUNet_find_best_configuration is called. Usually "
                             "running postprocessing on each fold is computationally cheap, but some users have "
                             "reported issues with very large images. If your images are large (>600x600x600 voxels) "
                             "you should consider setting this flag.")
    # parser.add_argument("--interp_order", required=False, default=3, type=int,
    #                     help="order of interpolation for segmentations. Testing purpose only. Hands off")
    # parser.add_argument("--interp_order_z", required=False, default=0, type=int,
    #                     help="order of interpolation along z if z is resampled separately. Testing purpose only. "
    #                          "Hands off")
    # parser.add_argument("--force_separate_z", required=False, default="None", type=str,
    #                     help="force_separate_z resampling. Can be None, True or False. Testing purpose only. Hands off")
    parser.add_argument('--val_disable_overwrite', action='store_false', default=True,
                        help='Validation does not overwrite existing segmentations')
    parser.add_argument('--disable_next_stage_pred', action='store_true', default=False,
                        help='do not predict next stage')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='path to nnU-Net checkpoint file to be used as pretrained model (use .model '
                             'file, for example model_final_checkpoint.model). Will only be used when actually training. '
                             'Optional. Beta. Use with caution.')

    args = parser.parse_args()
    """
    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    """
    task = taskid
    fold = "0"
    network = "2d"
    network_trainer = "nnUNetTrainerV2"
    #validation_only = args.validation_only
    validation_only = False
    plans_identifier = args.p
    find_lr = args.find_lr
    disable_postprocessing_on_folds = args.disable_postprocessing_on_folds

    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest

    fp32 = args.fp32
    run_mixed_precision = not fp32

    val_folder = args.val_folder
    # interp_order = args.interp_order
    # interp_order_z = args.interp_order_z
    # force_separate_z = args.force_separate_z

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    # if force_separate_z == "None":
    #     force_separate_z = None
    # elif force_separate_z == "False":
    #     force_separate_z = False
    # elif force_separate_z == "True":
    #     force_separate_z = True
    # else:
    #     raise ValueError("force_separate_z must be None, True or False. Given: %s" % force_separate_z)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, (nnUNetTrainerCascadeFullRes, nnUNetTrainerV2CascadeFullRes)), \
            "If running 3d_cascade_fullres then your " \
            "trainer class must be derived from " \
            "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class,
                          nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"
    # e.g. nnUNetTrainerV2
    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)
    if args.disable_saving:
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to
        # self.best_val_eval_criterion_MA
        trainer.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest. We need that in case
        # the training chashes
        trainer.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each

    trainer.initialize(training = not validation_only)#only validate it


    trainer.my_load_final_checkpoint(filename, train=False)
    if white:# white noise
        return trainer.run_validate_white(noise, norm_type)
    else: #pgd
        return trainer.run_validate_adv(noise, norm_type)
    
      




if __name__ == "__main__":
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    import random
    random.seed(10)
    ##########################need to be configured############################
    base = "YOUR_PATH_TO_MODELS"
    choice = 0
    
    ###########################################################################
    #dataset name
    dataset = ["Task002_Heart","Task004_Hippocampus","Task005_Prostate","Task009_Spleen"]
    selected = dataset[choice]
    # noise name
    #methods names
    #,"IMA", "TRADES","TE","PGD"
    #["IMA15","PGD15","PGD5","PGD1","nnUnet"],#D4
    netDict = [["IMA", "TRADES","TE","PGD","STD"],#D2
                ["STD","IMA", "TRADES","TE"],#D4
                ["IMA", "TRADES","TE","PGD","STD"],#D5
                ]
        
    nets = netDict[choice]
    folderDict = [] 
    #D2
    folders2 = ["result_CBM/002/model_IMA002_N_20_D_5_final_checkpoint.model",               
               "result_CBM/002/TRADES002.model",
               "result_CBM/002/model_TE_final_checkpoint.model",
               "result_CBM/002/model_PGD20_final_checkpoint.model",
               "result_CBM/002/model_final_checkpoint.model"]
    
    #D2
    folders4 = ["result_CBM/004/model_final_checkpoint.model",
                "result_CBM/004/model_IMA004_N_15_D_2.0_final_checkpoint.model",               
               "result_CBM/004/TRADES004.model",
               "result_CBM/004/model_TE_final_checkpoint.model"]
    
    #D2
    folders5 = ["result_CBM/005/model_IMA005_N_40_D_10_final_checkpoint.model",               
               "result_CBM/005/TRADES005.model",
               "result_CBM/005/model_TE_final_checkpoint.model",
               "result_CBM/005/model_PGD40_final_checkpoint.model",
               "result_CBM/005/model_final_checkpoint.model"]

    folderDict.append(folders2)
    folderDict.append(folders4)
    folderDict.append(folders5)
    folders = folderDict[choice]
    import os
    for model in folders:
        assert os.path.exists(model)
    #basePath = base+"/nnUnet/nnUNet/resultFolder/nnUNet/2d/"+selected+"/nnUNetTrainerV2__nnUNetPlansv2.1"
    for norm_type in ['L2', 'Linf']:        
        if norm_type == 'Linf':
            noiseDict =[[0, 2/255, 4/255, 8/255],#D2
                        [0, 2/255, 4/255, 8/255],#D4 
                        [0, 2/255, 4/255, 8/255]]#D5 
        else:
            noiseDict =[[0, 10, 20, 30],#D2
                        [0 ,5, 15, 25],#D4 
                        [0, 10, 20, 30]]#D5             
        noises = noiseDict[choice]
        print ("PGD validation")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111)    
        cols = ['b','g','r','y','k','m','c']
        yAxises = []
        yAxises2 = []
        fields = ["noise"]+[str(i) for i in noises]
        rows1 = []
        rows2 = []
        for i, net in enumerate(nets):
            print ("++++++++++++++++++the net is ", net,"++++++++++++++++++++++++++++++")
            for noise in noises:
                vixelDice,avgDice = main(noise, folders[i], selected[4:7], False, norm_type) 
                yAxises.append(vixelDice)
                yAxises2.append(avgDice)
            ax.plot(noises, yAxises, color=cols[i], label=nets[i])
            rows1.append([net]+[str(round(i,4)) for i in yAxises])
            ax2.plot(noises, yAxises2, color=cols[i], label=nets[i])
            rows2.append([net]+[str(round(i,4)) for i in yAxises2])
            yAxises = []
            yAxises2=[]
    
        
        """
        ax.set_title(selected)
        ax.set_xlabel("noise(Linf)")
        ax.set_ylabel("AVG Dice Index")
        ax.set_ylim(0,1)
        ax.set_yticks(np.arange(0, 1.05, step=0.05))
        ax.legend()
        ax.grid(True)
        fig.savefig("V_Dice_result_pgd_"+selected+".pdf",bbox_inches='tight')
        """
        with open("V_Dice_result_auto_L"+norm_type+"_"+selected+".csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows1)
        """
        ax2.set_title(selected)
        ax2.set_xlabel("noise(Linf)")
        ax2.set_ylabel("AVG Dice Index")
        ax2.set_ylim(0,1)
        ax2.set_yticks(np.arange(0, 1.05, step=0.05))
        ax2.legend()
        ax2.grid(True)
        fig2.savefig("AVG_Dice_result_pgd_"+selected+".pdf",bbox_inches='tight')    
        """
        with open("AVG_Dice_result_auto_L"+norm_type+"_"+selected+".csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows2)    
        
    #%%
    
    
        print("white validation")
        if norm_type == "Linf":
            noiseDict =[[0, 8/255, 16/255, 32/255],#D2
                        [0, 8/255, 16/255, 32/255],#D4 
                        [0, 8/255, 16/255, 32/255]]#D5 
        else:
            noiseDict =[[0, 10, 20, 30],#D2
                        [0, 5, 15, 25],#D4 
                        [0, 10, 20, 30]]#D5             
        noises = noiseDict[choice]  
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111)    
        cols = ['b','g','r','y','k','m','c']
        yAxises = []
        yAxises2 = []
        fields = ["noise"]+[str(i) for i in noises]
        rows1 = []
        rows2 = []
        for i, net in enumerate(nets):
            print ("++++++++++++++++++the net is ", net,"++++++++++++++++++++++++++++++")
            for noise in noises:
                vixelDice,avgDice = main(noise, folders[i], selected[4:7], True, norm_type) 
                yAxises.append(vixelDice)
                yAxises2.append(avgDice)
            ax.plot(noises, yAxises, color=cols[i], label=nets[i])
            rows1.append([net]+[str(round(i,4)) for i in yAxises])
            ax2.plot(noises, yAxises2, color=cols[i], label=nets[i])
            rows2.append([net]+[str(round(i,4)) for i in yAxises2])
            yAxises = []
            yAxises2=[]
    
        
        """
        ax.set_title(selected)
        ax.set_xlabel("noise(Linf)")
        ax.set_ylabel("AVG Dice Index")
        ax.set_ylim(0,1)
        ax.set_yticks(np.arange(0, 1.05, step=0.05))
        ax.legend()
        ax.grid(True)
        fig.savefig("V_Dice_result_white_"+selected+".pdf",bbox_inches='tight')
        """
        with open("V_Dice_result_white_L"+norm_type+"_"+selected+".csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows1)
        """
        ax2.set_title(selected)
        ax2.set_xlabel("noise(Linf)")
        ax2.set_ylabel("AVG Dice Index")
        ax2.set_ylim(0,1)
        ax2.set_yticks(np.arange(0, 1.05, step=0.05))
        ax2.legend()
        ax2.grid(True)
        fig2.savefig("AVG_Dice_result_white_"+selected+".pdf",bbox_inches='tight')    
        """
        with open("AVG_Dice_result_white_L"+norm_type+"_"+selected+".csv",'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerows(rows2)       
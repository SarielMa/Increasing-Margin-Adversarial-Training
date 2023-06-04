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


from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2, MultipleOutputLossKL
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.IMA.RobustDNN_IMA_claregseg import IMA_loss
from nnunet.TRADES.RobustDNN_TRADES import TRADES_loss
from nnunet.training.loss_functions.dice_loss import My_DC_and_CE_loss
from nnunet.training.loss_functions.dice_loss import MyDiceIndex
from nnunet.IMA.PGD import pgd_attack


def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    

class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs =50
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            self.loss2 = MultipleOutputLoss2(My_DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, {}), self.ds_loss_weights)
            self.loss_kl = MultipleOutputLossKL(nn.KLDivLoss(), self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val, self.dl_ts = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen, self.ts_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val, self.dl_ts,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]# Nx1x...
        output = output[0]# Nx3x...
        return super().run_online_evaluation(output, target)
    
    def my_run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]# Nx1x...
        output = output[0]# Nx3x...
        return super().my_run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def mrse_shape(self,Sp, S, reduction):
        S=S.view(S.shape[0], -1, 2)
        Sp=Sp.view(Sp.shape[0], -1, 2)
        mrse = ((Sp-S)**2).sum(dim=2).sqrt().mean(dim=1)
        if reduction =='mean':
            mrse=mrse.mean()
        elif reduction =='sum':
            mrse=mrse.sum()
        return mrse
    """
    def classify_model_std_output_reg(self,Yp, Y):
        mrse=self.mrse_shape(Yp, Y, reduction='none')
        Yp_e_Y=(mrse<=10)
        return Yp_e_Y
    
    def classify_model_adv_output_reg(self,Ypn, Y):
        #Y could be Ytrue or Ypred
        mrse=self.mrse_shape(Ypn, Y, reduction='none')
        Ypn_e_Y=(mrse<=5)
        return Ypn_e_Y
    """
    """
    def classify_model_std_output_seg_old(self,Yp, Y):
        dice=dice_seg(Yp, Y, reduction='none')
        Yp_e_Y=(dice>0.5)
        return Yp_e_Y
    #
    def classify_model_adv_output_seg_old(self,Ypn, Y):
        #Y could be Ytrue or Ypred
        dice=dice_seg(Ypn, Y, reduction='none')
        Ypn_e_Y=(dice>0.85)
        return Ypn_e_Y
    """
    def classify_model_std_output_seg(self,Yp, Y):
        #valDice = MyDiceIndex(batch_dice=False)
        Yp = Yp[0]
        Y = Y[0]
        #dice=valDice(Yp, Y)
        dice = super().getOnlineDiceMax(Yp, Y)
        Yp_e_Y=(dice>=0)
        return Yp_e_Y
    #
    def classify_model_adv_output_seg(self,Ypn, Y, task):
        #Y could be Ytrue or Ypred
        #valDice = MyDiceIndex(batch_dice=False)
        Yp = Ypn[0]
        Y = Y[0]
        #dice=valDice(Yp, Y)
        dice = super().getOnlineDiceMax(Yp, Y)
        
        # D2: avg mean: 0.9114448;
        # D4: avg mean: 0.8065869;
        # D5: avg mean: 0.86081946;
        threshold = 0
        if task.task == "002":
            threshold = 0.9114448
        if task.task == "005":
            threshold = 0.86081946
        if task.task == "004":
            threshold = 0.8065869
        if task.stop != 0:
            threshold = 0.60
        #print ("the task for threshould is ",threshold,"+++++++++++++++++++++++++++++++++++++")
        Ypn_e_Y=(dice>=threshold)
        return Ypn_e_Y
    
    def run_model_std_seg(self, model, X, Y=None, return_loss=False, reduction='none'):
        Z=model(X)    
        if return_loss == True:
            loss=self.loss2(Z, Y)
            if reduction =='sum':
                return Z, loss.sum()
            elif reduction == 'mean':
                return Z, loss.mean()
            else:
                return Z, loss
        else:
            return Z
    #
    def run_model_adv_seg(self, model, X, Y=None, return_loss=False, reduction='sum'):
        #valDice = MyDiceIndex(batch_dice=False)
        Z=model(X)
        if return_loss == True:  
            loss_dice=self.loss2(Z, Y)
            if reduction =='sum':
                return Z, loss_dice.sum()
            elif reduction == 'mean':
                return Z, loss_dice.mean()
            else:
                return Z, loss_dice
        else:
            return Z

    def run_PGD_iteration(self, data_generator, args, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']


        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()
        Z = self.network(data)
        loss_P = self.loss(Z, target)
        Xn = pgd_attack(self.network, data, target, args.noise, args.norm_type, args.max_iter, args.step, loss_fn=self.loss)
        Zn = self.network(Xn)
        loss_N = self.loss(Zn, target)
        loss = loss_N 
        loss.backward()
        # do the back propagation
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        
        del data
        if run_online_evaluation:
            self.run_online_evaluation(Z, target)

        del target

        return loss.detach().cpu().numpy()
    
    def run_IMA_iteration(self, data_generator, args,flag1, flag2, E_new, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        idx = torch.tensor(data_dict['id'])

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        # parameters that need to be set for IMA
        ###################################################
        stop=args.stop
        stop_near_boundary=False
        stop_if_label_change=False
        stop_if_label_change_next_step=False
        if stop==1:
            stop_near_boundary=True
        elif stop==2:
            stop_if_label_change=True
        elif stop==3:
            stop_if_label_change_next_step=True
        else:
            print ("AMAT is running")
        #E_new=args.E.detach().clone()
        ###################################################
        
        rand_init_norm=torch.clamp(args.E[idx]-args.delta, min=args.delta)
        margin = to_cuda(args.E[idx])
        step=args.alpha*margin/args.max_iter
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            rand_init_norm = to_cuda(rand_init_norm)

        self.optimizer.zero_grad()
        #output = self.network(data)
        #del data
        #l = self.loss(output, target)
        # loss should compare output[0] and target[0]
        loss, loss1, loss2, loss3, Yp, advc, Xn, Ypn, idx_n = IMA_loss(args, self.network, data, Y=target,#Y is a tuple
                                                                       norm_type=args.norm_type,
                                                                       rand_init_norm=rand_init_norm,
                                                                       margin=margin,
                                                                       max_iter=args.max_iter,
                                                                       step=step,
                                                                       refine_Xn_max_iter=args.refine_Xn_max_iter,
                                                                       Xn1_equal_X=args.Xn1_equal_X,
                                                                       Xn2_equal_Xn=args.Xn2_equal_Xn,
                                                                       stop_near_boundary=stop_near_boundary,
                                                                       stop_if_label_change=stop_if_label_change,
                                                                       stop_if_label_change_next_step=stop_if_label_change_next_step,
                                                                       beta=args.beta, beta_position=args.beta_position,
                                                                       use_optimizer=False,
                                                                       run_model_std=self.run_model_std_seg,
                                                                       classify_model_std_output=self.classify_model_std_output_seg,
                                                                       run_model_adv=self.run_model_adv_seg,
                                                                       classify_model_adv_output=self.classify_model_adv_output_seg,
                                                                       pgd_replace_Y_with_Yp=args.pgd_replace_Y_with_Yp,
                                                                       model_eval_attack=args.model_eval_attack,
                                                                       model_eval_Xn=args.model_eval_Xn,
                                                                       model_Xn_advc_p=args.model_Xn_advc_p)        

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        #--------------------

        self.optimizer.step()
        #--------------------update the margins
        Yp_e_Y=self.classify_model_std_output_seg(Yp, target)
        flag1[idx[advc==0]]=1
        flag2[idx[Yp_e_Y]]=1
        
        if idx_n.shape[0]>0:
            temp=torch.norm((Xn-data[idx_n]).view(Xn.shape[0], -1), p=args.norm_type, dim=1).cpu()
            #E_new[idx[idx_n]]=torch.min(E_new[idx[idx_n]], temp)     
            #bottom = args.delta*torch.ones(E_new.size(0), dtype=E_new.dtype, device=E_new.device)
            E_new[idx[idx_n]] = (E_new[idx[idx_n]]+temp)/2# use mean to refine the margin to reduce the effect of augmentation on margins
        #--------------------

        if run_online_evaluation:
            self.run_online_evaluation(Yp, target)

        del target

        return loss.detach().cpu().numpy(), flag1, flag2, E_new


    def run_TRADES_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        task = self.dataset_directory.split("\\")[-1]
        
        epsilon = None
        if "002" in task:
            epsilon = 20#cannot converge
            #epsilon = 5
        elif "004" in task:
            epsilon = 15#cannot converge
            #epsilon = 0.5
        elif "005" in task:
            epsilon = 40#cannot converge
            #epsilon = 10
        
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.network.zero_grad()
        #print ("epsilon is ", epsilon)
        loss, _, _ = TRADES_loss(self.network,
                            self.loss,
                            self.loss_kl,
                            data,
                            target,
                            self.optimizer,
                            step_size = epsilon/5,
                            epsilon= epsilon,
                            perturb_steps= 10,
                            beta = 6,
                            distance='l_2')
        loss.backward()      
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        #--------------------

        self.optimizer.step()
        #--------------------

        if run_online_evaluation:
            self.run_online_evaluation(Yp, target)

        del target

        return loss.detach().cpu().numpy()

    def run_TE_iteration(self, pgd_te, data_generator, epoch, weight, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        task = self.dataset_directory.split("\\")[-1]
        
        
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        idx = torch.tensor(data_dict['id'])

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.network.zero_grad()
        #print ("epsilon is ", epsilon)
        loss = pgd_te(data, target, idx, epoch, self.network, self.optimizer, weight)
        loss.backward()      
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        #--------------------

        self.optimizer.step()
        #--------------------

        if run_online_evaluation:
            self.run_online_evaluation(Yp, target)

        del target

        return loss.detach().cpu().numpy()


    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        valLen = len(val_keys)
        pivot = int(valLen/3*1)
        
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys[:pivot]:
            self.dataset_val[i] = self.dataset[i]
        self.dataset_ts = OrderedDict()
        """
        for i in tr_keys:
            self.dataset_ts[i] = self.dataset[i]# just for get the distribution
        """
        for i in val_keys[pivot:]:
            self.dataset_ts[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
    def run_validate_adv_IFGSM(self, noise):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret, ret2 = super().run_validate_adv_IFGSM(noise)
        self.network.do_ds = ds
        return ret,ret2
    
    def run_validate_adv(self, noise, norm_type):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret, ret2 = super().run_validate_adv(noise, norm_type)
        self.network.do_ds = ds
        return ret,ret2

    def run_validate_adv_cmpb(self, noise, norm_type, attack):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_validate_adv_cmpb(noise, norm_type, attack)
        self.network.do_ds = ds
        return ret

    
    def run_validate_white(self, noise, norm_type):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret, ret2 = super().run_validate_white(noise, norm_type)
        self.network.do_ds = ds
        return ret,ret2
    
    def run_validate_adv_showcase(self, noise):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        imgn, gtn = super().run_validate_adv_showcase(noise)
        self.network.do_ds = ds
        return imgn, gtn
    def run_IMA_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_IMA_training(self.dl_tr.counter)#pass the number of samples in training set
        self.network.do_ds = ds
        return ret
 
    
    def run_TE_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_TE_training(self.dl_tr.counter)#pass the number of samples in training set
        self.network.do_ds = ds
        return ret 
    
    def run_TRADES_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_TRADES_training(self.dl_tr.counter)#pass the number of samples in training set
        self.network.do_ds = ds
        return ret
  
    def run_IMA_training_grid(self, params):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_IMA_training_grid(self.dl_tr.counter, params)#pass the number of samples in training set
        self.network.do_ds = ds
        return ret
    
    def run_PGD_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_PGD_training()#pass the number of samples in training set
        self.network.do_ds = ds
        return ret
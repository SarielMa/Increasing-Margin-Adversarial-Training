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


from _warnings import warn
from typing import Tuple

import matplotlib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.network_architecture.neural_network import SegmentationNetwork
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler

#matplotlib.use("agg")
from time import time, sleep
import torch
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import torch.backends.cudnn as cudnn
from abc import abstractmethod
from datetime import datetime
from tqdm import trange
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.loss_functions.dice_loss import DiceIndex, SoftDiceLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.IMA.RobustDNN_IMA_claregseg import IMA_update_margin
import os.path
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.autoattackfornnunet.autoattack import AutoAttack
#from nnunet.autoattacklib.autoattack.autoattack import AutoAttack
#from Evaluate_advertorch import test_adv

def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)
    
    
    
def clip_norm_(noise, norm_type, norm_max):
    if not isinstance(norm_max, torch.Tensor):
        return clip_normA_(noise, norm_type, norm_max)
    else:
        return clip_normB_(noise, norm_type, norm_max)
#%%
def clip_normA_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max is a scalar noise level
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            noise.clamp_(-norm_max, norm_max)
        elif norm_type == 2 or norm_type == 'L2':
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("other norm clip is not implemented.")
    #-----------
    return noise
#%%
def clip_normB_(noise, norm_type, norm_max):
    # noise is a tensor modified in place, noise.size(0) is batch_size
    # norm_type can be np.inf, 1 or 2, or p
    # norm_max is 1D tensor, norm_max[k] is the maximum noise level for noise[k]
    if noise.size(0) == 0:
        return noise
    with torch.no_grad():
        if norm_type == np.inf or norm_type == 'Linf':
            #for k in range(noise.size(0)):
            #    noise[k].clamp_(-norm_max[k], norm_max[k])
            N=noise.view(noise.size(0), -1)
            norm_max=norm_max.view(norm_max.size(0), -1)
            N=torch.max(torch.min(N, norm_max), -norm_max)
            N=N.view(noise.size())
            noise-=noise-N
        elif norm_type == 2 or norm_type == 'L2':
            N=noise.view(noise.size(0), -1)
            l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
            norm_max=norm_max.view(norm_max.size(0), 1)
            #print(l2_norm.shape, norm_max.shape)
            temp = (l2_norm > norm_max).squeeze()
            if temp.sum() > 0:
                norm_max=norm_max[temp]
                norm_max=norm_max.view(norm_max.size(0), -1)
                N[temp]*=norm_max/l2_norm[temp]
        else:
            raise NotImplementedError("not implemented.")
        #-----------
    return noise

class NetworkTrainer(object):
    def __init__(self, deterministic=True, fp16=False):
        """
        A generic class that can train almost any neural network (RNNs excluded). It provides basic functionality such
        as the training loop, tracking of training and validation losses (and the target metric if you implement it)
        Training can be terminated early if the validation loss (or the target metric if implemented) do not improve
        anymore. This is based on a moving average (MA) of the loss/metric instead of the raw values to get more smooth
        results.

        What you need to override:
        - __init__
        - initialize
        - run_online_evaluation (optional)
        - finish_online_evaluation (optional)
        - validate
        - predict_test_case
        """
        self.fp16 = fp16
        self.amp_grad_scaler = None

        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

        ################# SET THESE IN self.initialize() ###################################
        self.network: Tuple[SegmentationNetwork, nn.DataParallel] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.tr_gen = self.val_gen = None
        self.was_initialized = False

        ################# SET THESE IN INIT ################################################
        self.output_folder = None
        self.fold = None
        self.loss = None
        self.dataset_directory = None

        ################# SET THESE IN LOAD_DATASET OR DO_SPLIT ############################
        self.dataset = None  # these can be None for inference mode
        self.dataset_tr = self.dataset_val = None  # do not need to be used, they just appear if you are using the suggested load_dataset_and_do_split

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.max_num_epochs = 1000
        self.num_batches_per_epoch = 50
        self.num_val_batches_per_epoch = 3
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

        ################# LEAVE THESE ALONE ################################################
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used
        self.epoch = 0
        self.log_file = None
        self.deterministic = deterministic

        self.use_progress_bar = False
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))

        ################# Settings for saving checkpoints ##################################
        self.save_every = 50
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

    @abstractmethod
    def initialize(self, training=True):
        """
        create self.output_folder

        modify self.output_folder if you are doing cross-validation (one folder per fold)

        set self.tr_gen and self.val_gen

        call self.initialize_network and self.initialize_optimizer_and_scheduler (important!)

        finally set self.was_initialized to True
        :param training:
        :return:
        """

    @abstractmethod
    def load_dataset(self):
        pass

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        splits_file = join(self.dataset_directory, "splits_final.pkl")
        if not isfile(splits_file):
            self.print_to_log_file("Creating new split...")
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

        splits = load_pickle(splits_file)

        if self.fold == "all":
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            tr_keys = splits[self.fold]['train']
            val_keys = splits[self.fold]['val']

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())
            
            
    def my_plot_progress(self,args):
        """
        Should probably by improved
        :return:
        """
        title = args.title+str(args.noise)+"_"
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)

            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(join(self.output_folder, title+"progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(join(self.output_folder, "model_best.model")):
            self.load_checkpoint(join(self.output_folder, "model_best.model"), train=train)
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, train=True):
        if isfile(join(self.output_folder, "model_final_checkpoint.model")):
            return self.load_checkpoint(join(self.output_folder, "model_final_checkpoint.model"), train=train)
        if isfile(join(self.output_folder, "model_latest.model")):
            return self.load_checkpoint(join(self.output_folder, "model_latest.model"), train=train)
        if isfile(join(self.output_folder, "model_best.model")):
            return self.load_best_checkpoint(train)
        raise RuntimeError("No checkpoint found")

    def load_final_checkpoint(self, train=False):
        filename = join(self.output_folder, "model_final_checkpoint.model")
        if not isfile(filename):
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)
    
    def my_load_final_checkpoint(self, filename, train=False):
        if not isfile(filename):
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    @abstractmethod
    def initialize_network(self):
        """
        initialize self.network here
        :return:
        """
        pass

    @abstractmethod
    def initialize_optimizer_and_scheduler(self):
        """
        initialize self.optimizer and self.lr_scheduler (if applicable) here
        :return:
        """
        pass

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()

    def plot_network_architecture(self):
        """
        can be implemented (see nnUNetTrainer) but does not have to. Not implemented here because it imposes stronger
        assumptions on the presence of class variables
        :return:
        """
        pass

    def run_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        #self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []

            # train one epoch
            self.network.train()

            if self.use_progress_bar:
                with trange(self.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))

                        l = self.run_iteration(self.tr_gen, True)

                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(self.tr_gen, True)
                    train_losses_epoch.append(l)

            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))



            
    class PGD_params_D2:
        def __init__(self):           
            #used to pass parameters to ima iteration
            self.noise = 20
            self.norm_type = 2
            self.max_iter = 20
            self.step = 4*self.noise/self.max_iter
            self.title = "PGD"
            
    class PGD_params_D4:
        def __init__(self):           
            #used to pass parameters to ima iteration
            self.noise = 15
            self.norm_type = 2
            self.max_iter = 20
            self.step = 4*self.noise/self.max_iter
            self.title = "PGD"
            
    class PGD_params_D5:
        def __init__(self):           
            #used to pass parameters to ima iteration
            self.noise = 40
            self.norm_type = 2
            self.max_iter = 20
            self.step = 4*self.noise/self.max_iter
            self.title = "PGD"

            
    def run_PGD_training(self):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        #self.plot_network_architecture()

        # config IMA parameters
        #######################################################################################
        task = self.dataset_directory.split("\\")[-1]
        
        args = None
        if "002" in task:
            args = self.PGD_params_D2()
        elif "004" in task:
            args = self.PGD_params_D4()
        elif "005" in task:
            args = self.PGD_params_D5()
        else:
            raise Exception("Not supported task id")
        
        #args.E=args.delta*torch.ones(counter, dtype=torch.float32)
        #E_new=args.E.detach().clone()
        #######################################################################################

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            #----------------------------
            #flag1=torch.zeros(len(args.E), dtype=torch.float32)
            #flag2=torch.zeros(len(args.E), dtype=torch.float32)
            #E_new=args.E.detach().clone()
            #---------------------------
            # train one epoch-------------------------------------------------------------------------------------------------
            self.network.train()
            for c in range(self.num_batches_per_epoch):
                l = self.run_PGD_iteration(self.tr_gen, args, True)
                #print("batch ",c,"finished, PGD train")
                train_losses_epoch.append(l)
            # one epoch finished
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])
            #-----------------------------------------------------------------------------------------------------------------
            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                # run one epoch.....
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
   
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.my_on_epoch_end(args)

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_PGD"+str(args.noise)+"_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_PGD_latest.model")):
            os.remove(join(self.output_folder, "model_PGD_latest.model"))
        if isfile(join(self.output_folder, "model_PGD_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_PGD_latest.model.pkl"))
            
    def plot_E(self, E, noise, filename=None):
        fig, ax = plt.subplots()
        ax.hist(E.cpu().numpy(), bins=20, range =[0, 2*E.max().item()] )
        #display.display(fig)
        if filename is not None:
            fig.savefig(filename+'AMAT.png')
        plt.close(fig)         
        
        
        
            
    def run_IMA_training_grid(self, counter, params):    
        if not torch.cuda.is_available():
            print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        #self.plot_network_architecture()

        # config IMA parameters
        #######################################################################################
        args = params
        args.E=args.delta*torch.ones(counter, dtype=torch.float32)
        title = args.title+"_N_"+str(args.noise)+"_D_"+str(args.delta)
        #E_new=args.E.detach().clone()
        #######################################################################################

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            #----------------------------
            flag1=torch.zeros(len(args.E), dtype=torch.float32)
            flag2=torch.zeros(len(args.E), dtype=torch.float32)
            E_new=args.E.detach().clone()
            #---------------------------
            # train one epoch
            self.network.train()
            for c in range(self.num_batches_per_epoch):
            #for c in range(1):
                l, flag1, flag2, E_new = self.run_IMA_iteration(self.tr_gen, args,flag1, flag2, E_new, True)
                #print("batch ",c,"finished")
                train_losses_epoch.append(l)
            # one epoch finished
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])
            #------update the margin
            IMA_update_margin(args, args.delta, args.noise, flag1, flag2, E_new)
            print('IMA_update_margin: done, margin updated')    
            self.plot_E(args.E, max(args.E),join(self.output_folder, title+'histE'))           
            
            #------
            """
            
            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                # run one epoch.....
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
   
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
            """

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training
            
            #continue_training = self.my_on_epoch_end_grid(args)

            epoch_end_time = time()

            #if not continue_training:
                # allows for early stopping
            #    break

            self.epoch += 1
            print ("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_"+title+"_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_IMA_latest.model")):
            os.remove(join(self.output_folder, "model_IMA_latest.model"))
        if isfile(join(self.output_folder, "model_IMA_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_IMA_latest.model.pkl"))
        
    def run_IMA_training(self, counter):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        #self.plot_network_architecture()

        # config IMA parameters
        #######################################################################################
        args = self.IMA_params()
        args.E=args.delta*torch.ones(counter, dtype=torch.float32)
        title = args.title+str(args.noise)
        #E_new=args.E.detach().clone()
        #######################################################################################

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            #----------------------------
            flag1=torch.zeros(len(args.E), dtype=torch.float32)
            flag2=torch.zeros(len(args.E), dtype=torch.float32)
            E_new=args.E.detach().clone()
            #---------------------------
            # train one epoch
            self.network.train()
            for c in range(self.num_batches_per_epoch):
                l, flag1, flag2, E_new = self.run_IMA_iteration(self.tr_gen, args,flag1, flag2, E_new, True)
                #print("batch ",c,"finished")
                train_losses_epoch.append(l)
            # one epoch finished
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])
            #------update the margin
            IMA_update_margin(args, args.delta, args.noise, flag1, flag2, E_new)
            print('IMA_update_margin: done, margin updated')    
            self.plot_E(args.E, args.noise,join(self.output_folder, title+'histE'))           
            
            #------
            
            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                # run one epoch.....
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
   
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
            

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.my_on_epoch_end(args)

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_"+title+"_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_IMA_latest.model")):
            os.remove(join(self.output_folder, "model_IMA_latest.model"))
        if isfile(join(self.output_folder, "model_IMA_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_IMA_latest.model.pkl"))

    def run_TRADES_training(self, counter):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        #self.plot_network_architecture()

        # config IMA parameters
        #######################################################################################
        title = "TRADES_beta6_epsilon8"
        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            #----------------------------
            # train one epoch
            self.network.train()
            for c in range(self.num_batches_per_epoch):
                l = self.run_TRADES_iteration(self.tr_gen, True)
                #print("batch ",c,"finished")
                train_losses_epoch.append(l)
            # one epoch finished
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])
            #------          
            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                # run one epoch.....
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
   
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
            

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_"+title+"_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_TRADES_latest.model")):
            os.remove(join(self.output_folder, "model_TRADES_latest.model"))
        if isfile(join(self.output_folder, "model_TRADES_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_TRADES_latest.model.pkl"))        

    def run_TE_training(self, counter):
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        _ = self.tr_gen.next()
        _ = self.val_gen.next()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)        
        #ramp up function
        def sigmoid_rampup(current, start_es, end_es):
            """Exponential rampup from https://arxiv.org/abs/1610.02242"""
            if current < start_es:
                return 0.0
            if current > end_es:
                return 1.0
            else:
                import math
                phase = 1.0 - (current - start_es) / (end_es - start_es)
                return math.exp(-5.0 * phase * phase)
        title = "TE"
        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")
        #config TE parameters
        from nnunet.TE.loss import PGD_TE
        task = self.dataset_directory.split("\\")[-1]
        
        epsilon = None
        if "002" in task:
            epsilon = 20#cannot converge
        elif "004" in task:
            epsilon = 15#cannot converge
        elif "005" in task:
            epsilon = 40#cannot converge
        else:
            raise Exception("Not supported task id")
        
        es_start = 25
        es_end = 35
        pgd_te = PGD_TE(  loss_fn = self.loss,
                           multioutput_weights = self.ds_loss_weights,
                           num_samples = counter,
                           momentum=0.9,
                           step_size=epsilon/4,
                           epsilon= epsilon,
                           perturb_steps=10,
                           norm ='l2',
                           es= es_start)

        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time()
            train_losses_epoch = []
            #----------------------------
            # compute the weight
            rampup_rate = sigmoid_rampup(self.epoch, es_start, es_end)
            weight = rampup_rate * 300
            self.network.train()
            # train one iteration
            for c in range(self.num_batches_per_epoch):
                l = self.run_TE_iteration( pgd_te, self.tr_gen,  self.epoch, weight, True)
                #print("batch ",c,"finished")
                train_losses_epoch.append(l)
            # one epoch finished
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])
            #------          
            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                # run one epoch.....
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_gen, False, True)
                    val_losses.append(l)
   
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
            

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()

            epoch_end_time = time()

            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1  # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_"+title+"_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_TRADES_latest.model")):
            os.remove(join(self.output_folder, "model_TRADES_latest.model"))
        if isfile(join(self.output_folder, "model_TRADES_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_TRADES_latest.model.pkl"))


    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.model"))
            self.print_to_log_file("done")
            
    def my_maybe_save_checkpoint(self, args):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        title = args.title+str(args.noise)+"_"
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file("my saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, title+"model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, title+"model_latest.model"))
            self.print_to_log_file("done")

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                is better, so we need to negate it.
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                #self.print_to_log_file("saving best epoch checkpoint...")
                if self.save_best_checkpoint: self.save_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                #self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                #self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    #self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    #self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                pass
                #self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def on_epoch_end(self):
        self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training
    
    def my_on_epoch_end(self, args):
        self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        self.my_plot_progress(args)

        self.maybe_update_lr()

        self.my_maybe_save_checkpoint(args)

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training
    def my_on_epoch_end_grid(self, args):
        #self.finish_online_evaluation()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        #self.my_plot_progress(args)

        self.maybe_update_lr()

        self.my_maybe_save_checkpoint(args)

        #self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training
    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]
                                 
                                 
                                 


    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
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
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()
    
    


    def run_online_evaluation(self, *args, **kwargs):
        """
        Can be implemented, does not have to
        :param output_torch:
        :param target_npy:
        :return:
        """
        pass

    def finish_online_evaluation(self):
        """
        Can be implemented, does not have to
        :return:
        """
        pass

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    def find_lr(self, num_iters=1000, init_value=1e-6, final_value=10., beta=0.98):
        """
        stolen and adapted from here: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        :param num_iters:
        :param init_value:
        :param final_value:
        :param beta:
        :return:
        """
        import math
        self._maybe_init_amp()
        mult = (final_value / init_value) ** (1 / num_iters)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []

        for batch_num in range(1, num_iters + 1):
            # +1 because this one here is not designed to have negative loss...
            loss = self.run_iteration(self.tr_gen, do_backprop=True, run_online_evaluation=False).data.item() + 1

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        import matplotlib.pyplot as plt
        lrs = [10 ** i for i in log_lrs]
        fig = plt.figure()
        plt.xscale('log')
        plt.plot(lrs[10:-5], losses[10:-5])
        plt.savefig(join(self.output_folder, "lr_finder.png"))
        plt.close()
        return log_lrs, losses

#%% adversarial part

    def clip_norm_(self,noise, norm_type, norm_max):
        if not isinstance(norm_max, torch.Tensor):
            self.clip_normA_(noise, norm_type, norm_max)
        else:
            self.clip_normB_(noise, norm_type, norm_max)
            
    def clip_normA_(self,noise, norm_type, norm_max):
        # noise is a tensor modified in place, noise.size(0) is batch_size
        # norm_type can be np.inf, 1 or 2, or p
        # norm_max is noise level
        if noise.size(0) == 0:
            return noise
        with torch.no_grad():
            if norm_type == np.inf or norm_type == 'Linf':
                noise.clamp_(-norm_max, norm_max)
            elif norm_type == 2 or norm_type == 'L2':
                N=noise.view(noise.size(0), -1)
                l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
                temp = (l2_norm > norm_max).squeeze()
                if temp.sum() > 0:
                    N[temp]*=norm_max/l2_norm[temp]
            else:
                raise NotImplementedError("other norm clip is not implemented.")
        #-----------
        return noise
    
    def clip_normB_(self,noise, norm_type, norm_max):
        # noise is a tensor modified in place, noise.size(0) is batch_size
        # norm_type can be np.inf, 1 or 2, or p
        # norm_max[k] is noise level for every noise[k]
        if noise.size(0) == 0:
            return noise
        with torch.no_grad():
            if norm_type == np.inf or norm_type == 'Linf':
                #for k in range(noise.size(0)):
                #    noise[k].clamp_(-norm_max[k], norm_max[k])
                N=noise.view(noise.size(0), -1)
                norm_max=norm_max.view(norm_max.size(0), -1)
                N=torch.max(torch.min(N, norm_max), -norm_max)
                N=N.view(noise.size())
                noise-=noise-N
            elif norm_type == 2 or norm_type == 'L2':
                N=noise.view(noise.size(0), -1)
                l2_norm= torch.sqrt(torch.sum(N**2, dim=1, keepdim=True))
                norm_max=norm_max.view(norm_max.size(0), 1)
                #print(l2_norm.shape, norm_max.shape)
                temp = (l2_norm > norm_max).squeeze()
                if temp.sum() > 0:
                    norm_max=norm_max[temp]
                    norm_max=norm_max.view(norm_max.size(0), -1)
                    N[temp]*=norm_max/l2_norm[temp]
            else:
                raise NotImplementedError("not implemented.")
            #-----------
        return noise       
    
    def get_noise_init(self,norm_type, noise_norm, init_norm, X):
        noise_init=2*torch.rand_like(X)-1
        noise_init=noise_init.view(X.size(0),-1)
        if isinstance(init_norm, torch.Tensor):
            init_norm=init_norm.view(X.size(0), -1)
        noise_init=init_norm*noise_init
        noise_init=noise_init.view(X.size())
        self.clip_norm_(noise_init, norm_type, init_norm)
        self.clip_norm_(noise_init, norm_type, noise_norm)
        return noise_init
    
    def normalize_grad_(self,x_grad, norm_type, eps=1e-8):
        #x_grad is modified in place
        #x_grad.size(0) is batch_size
        with torch.no_grad():
            if norm_type == np.inf or norm_type == 'Linf':
                x_grad-=x_grad-x_grad.sign()
            elif norm_type == 2 or norm_type == 'L2':
                g=x_grad.view(x_grad.size(0), -1)
                l2_norm=torch.sqrt(torch.sum(g**2, dim=1, keepdim=True))
                l2_norm = torch.max(l2_norm, torch.tensor(eps, dtype=l2_norm.dtype, device=l2_norm.device))
                g *= 1/l2_norm
            else:
                raise NotImplementedError("not implemented.")
        return x_grad
    
    def uniform_white_attack(self, ):
        pass
    

#%% adversarial part

    def maskIt(self, x):
        ma = x.max()
        mi = x.min()
        x = (x-mi)/(ma-mi)
        return x    
    
    def run_one_adv_pgd100(self, data_dict, noise):
        self.network.eval()
        #data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']# target is a mask, but should have two...
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)# only the first target among the six is useful
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.optimizer.zero_grad()
        Xn = 0
        if noise == 0:
            Xn = data
        else:
            Xn = self.pgd_attack(self.network, data, target, noise, np.inf, 100, 0.25*noise, use_optimizer=False, loss_fn=self.loss)
        
        ret = 0
        #valDice = DiceIndex()
        with torch.no_grad():
            output = self.network(Xn)
            ret = self.getOnlineDiceMean(output[0], target[0])
            self.my_run_online_evaluation(output, target)         
        del target   
        return ret.cpu().numpy()
    
  
    
    def run_one_adv(self, data_dict, noise, adv):
        self.network.eval()
        #data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']# target is a mask, but should have two...
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)# only the first target among the six is useful
        if target[0].shape[1] <=2:
            adv.attacks_to_run = ['apgd-ce','square'] # dlr loss only work for 2+ classes
        else:
            adv.attacks_to_run = ['apgd-ce','apgd-dlr','square']
        
        #adv.attacks_to_run = ['apgd-dlr']
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.optimizer.zero_grad()
        Xn = 0
        if noise == 0:
            Xn = data
        else:
            #Xn = self.pgd_attack(self.network, data, target, noise, np.inf, 100, 0.25*noise, use_optimizer=False, loss_fn=self.loss)
            Xn = adv.run_standard_evaluation(data, target, bs = target[0].shape[0], return_labels = False)
        
        ret = 0
        #valDice = DiceIndex()
        with torch.no_grad():
            output = self.network(Xn)
            ret = self.getOnlineDiceMean(output[0], target[0])
            self.my_run_online_evaluation(output, target)         
        del target   
        return ret.cpu().numpy()

    def rand_uniform_attack(self, model, X, Y, noise_norm, max_iter, clip_X_min=0, clip_X_max=1, norm_type = np.inf):
        with torch.no_grad():
            Xout=X.detach().clone()
            dice_pre = None
            for n in range(0, max_iter):
                if norm_type == 2:
                    deltaX = torch.rand_like(X)
                    deltaX = clip_norm_(deltaX, norm_type, noise_norm)
                    Xn = X + deltaX
                else:
                    Xn = X + noise_norm*(torch.rand_like(X)) # rand_like returns uniform noise in [0,1]
                    
                Xn.clamp_(clip_X_min, clip_X_max)
                Zn = model(Xn)
                loss_dice=self.getOnlineDiceMean(Zn[0], Y[0])
                if dice_pre is not None:
                    Ypn_ne_Y = (loss_dice < dice_pre)
                else: 
                    Ypn_ne_Y = (loss_dice < 1)
                Xout[Ypn_ne_Y]=Xn[Ypn_ne_Y]
                dice_pre = loss_dice
        return Xout    
    
    def run_one_white(self, data_dict, noise, norm_type):
        #norm type is np.inf or 2#
        self.network.eval()
        #data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']# target is a mask, but should have two...
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)# only the first target among the six is useful
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.optimizer.zero_grad()
        Xn = 0
        if noise == 0:
            Xn = data
        else:
            Xn = self.rand_uniform_attack(self.network, data, target, noise, 100, norm_type = norm_type)
        
        ret = 0
        #valDice = DiceIndex()
        with torch.no_grad():
            output = self.network(Xn)
            ret = self.getOnlineDiceMean(output[0], target[0])
            self.my_run_online_evaluation(output, target)         
        del target   
        return ret.cpu().numpy()



    def run_one_adv_to_show(self, data_dict, noise):
        self.network.eval()
        #data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']# target is a mask, but should have two...
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)# only the first target among the six is useful
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.optimizer.zero_grad()
        Xn = 0
        if noise == 0:
            Xn = data
        else:
            Xn = self.pgd_attack(self.network, data, target, noise, 2, 100, 0.05*noise, use_optimizer=False, loss_fn=self.loss)
        
        ret = 0
        #valDice = DiceIndex()
        with torch.no_grad():
            output = self.network(Xn)
            #ret = self.getOnlineDiceMeanOnlyDoubleClass(output[0], target[0])
            #self.my_run_online_evaluation(output, target) 
            output_softmax = softmax_helper(output[0])
            output_seg = output_softmax.argmax(1)
        del target   
        
        
        
        return 0, Xn, output_seg
    
    def pgd_attack_2(self,model, X, Y, noise_norm, norm_type, max_iter, step,
                   rand_init=True, rand_init_norm=None, targeted=False,
                   clip_X_min=0, clip_X_max=1, use_optimizer=False, loss_fn=None):
        #-----------------------------------------------------
        # this attack focus on only the highest solution of Y and Yp
        if loss_fn is None :
            raise ValueError('loss_fn is unkown')
        #-----------------
        X = X.detach()
        #-----------------
        if rand_init == True:
            init_norm=rand_init_norm
            if rand_init_norm is None:
                init_norm=noise_norm
            noise_init=self.get_noise_init(norm_type, noise_norm, init_norm, X)
            Xn = X + noise_init
        else:
            Xn = X.clone().detach() # must clone
        #-----------------
        noise_new=(Xn-X).detach()
        if use_optimizer == True:
            optimizer = optim.Adamax([noise_new], lr=step)
        #-----------------
        for n in range(0, max_iter):
            Xn = Xn.detach()
            Xn.requires_grad = True
            Zn = model(Xn)
            loss = loss_fn(Zn[0], Y[0])
            #---------------------------
            if targeted == True:
                loss=-loss
            #---------------------------
            #loss.backward() will update W.grad
            grad_n=torch.autograd.grad(loss, Xn)[0]
            grad_n=self.normalize_grad_(grad_n, norm_type)
            if use_optimizer == True:
                noise_new.grad=-grad_n.detach() #grad ascent to maximize loss
                optimizer.step()
            else:
                Xnew = Xn.detach() + step*grad_n.detach()
                noise_new = Xnew-X
            #---------------------
            self.clip_norm_(noise_new, norm_type, noise_norm)
            #Xn = torch.clamp(X+noise_new, clip_X_min, clip_X_max)
            Xn = X + noise_new
            noise_new.data -= noise_new.data-(Xn-X).data
            Xn=Xn.detach()
        #---------------------------
        return Xn
    def run_one_adv2(self, data_dict, noise):
        # this use Dice as loss to attack
        self.network.eval()
        #data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']# target is a mask, but should have two...
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)# only the first target among the six is useful
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.optimizer.zero_grad()
        Xn = 0
        if noise == 0:
            Xn = data
        else:
            valDice = SoftDiceLoss(apply_nonlin=softmax_helper)
            Xn = self.pgd_attack_2(self.network, data, target, noise, 2, 100, 0.05*noise, use_optimizer=False, loss_fn=valDice)
        
        #ret = 0
        #valDice = DiceIndex()
        with torch.no_grad():
            output = self.network(Xn)
            self.run_online_evaluation(output, target)         
        del target
#%% adversarial part   
    def run_validate_adv_pgd100(self, noise):
        print ("+++++++++++++++++noise ",str(noise)," is running+++++++++++++++++++++++++++++++")
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        #_ = self.tr_gen.next()
        #_ = self.val_gen.next()


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

      
        #self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        counter = 1
        epoch_start_time = time()
        # validation with train=False
        self.network.eval()
        #val_losses = []
        #counter = 0
        #print ("num val batches per epoch is ", self.num_val_batches_per_epoch)
        avg = []
        for data_dict in self.ts_gen:
            
            # autoattack is used here
            #https://github.com/BCV-Uniandes/ROG/tree/2a503eaed104e3748e1f6454372b349527253fd
            #check if this target has no foregroud classes, if yes, ignore it
            target = data_dict['target']
            temp = target[0]
            if temp.max()==0:
                continue           
            #finishe check
            avg.append(self.run_one_adv(data_dict, noise))
            print ("one batch is done")
            if data_dict['last']:
                break
            #if counter ==20:
            #    break
            counter +=1

        ret = self.my_finish_online_evaluation()
        
        avg = np.concatenate(avg)
        ret2 = avg.mean()
        validationDice = np.mean(ret)
        self.print_to_log_file("av global foreground dice: ", ret)
        self.print_to_log_file("av paired dice: (only with complete target)", ret2)
        self.print_to_log_file("validation dice: %.4f" % validationDice)
        epoch_end_time = time()
        self.print_to_log_file("This validate took %f s\n" % (epoch_end_time - epoch_start_time))
        return validationDice, ret2


    def run_validate_adv_auto(self, noise, norm_type):
        print ("+++++++++++++++++noise ",str(noise)," is running+++++++++++++++++++++++++++++++")
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._maybe_init_amp()
        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")
        if not self.was_initialized:
            self.initialize(True)
        counter = 1
        epoch_start_time = time()
        self.network.eval()
        avg = []
        adversary = AutoAttack(self.network, norm_type= norm_type, eps = noise, version='standard', loss_fn = self.loss2)
        #norm = "L2"
        for data_dict in self.ts_gen:
            #check if this target has no foregroud classes, if yes, ignore it
            target = data_dict['target']
            temp = target[0]
            if temp.max()==0:
                continue         
            avg.append(self.run_one_adv(data_dict, noise, adversary)) 
            print ("one batch is done")
            if data_dict['last']:
                break
            #if counter ==20:
            #    break
            counter +=1
        ret = self.my_finish_online_evaluation()       
        avg = np.concatenate(avg)
        ret2 = avg.mean()
        validationDice = np.mean(ret)
        self.print_to_log_file("av global foreground dice: ", ret)
        self.print_to_log_file("av paired dice: (only with complete target)", ret2)
        self.print_to_log_file("validation dice: %.4f" % validationDice)
        epoch_end_time = time()
        self.print_to_log_file("This validate took %f s\n" % (epoch_end_time - epoch_start_time))
        return validationDice, ret2
    
#%% adversarial part   
    def run_validate_white(self, noise, norm_type):
        print ("+++++++++++++++++noise ",str(noise)," is running+++++++++++++++++++++++++++++++")
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        #_ = self.tr_gen.next()
        #_ = self.val_gen.next()


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

      
        #self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        counter = 1
        epoch_start_time = time()
        # validation with train=False
        self.network.eval()
        #val_losses = []
        #counter = 0
        #print ("num val batches per epoch is ", self.num_val_batches_per_epoch)
        avg = []
        if norm_type == "Linf":
            norm_type = np.inf
        elif norm_type == "L2":
            norm_type = 2
        else:
            raise ValueError('norm not supported')
            
        for data_dict in self.ts_gen:
            #check if this target has no foregroud classes, if yes, ignore it
            target = data_dict['target']
            temp = target[0]
            if temp.max()==0:
                continue
            
            #finishe check
            avg.append(self.run_one_white(data_dict, noise, norm_type))
            print ("one batch is done")
            if data_dict['last']:
                break
            #if counter ==20:
            #    break
            counter +=1

        ret = self.my_finish_online_evaluation()
        
        avg = np.concatenate(avg)
        ret2 = avg.mean()
        validationDice = np.mean(ret)
        self.print_to_log_file("av global foreground dice: ", ret)
        self.print_to_log_file("av paired dice: (only with complete target)", ret2)
        self.print_to_log_file("validation dice: %.4f" % validationDice)
        epoch_end_time = time()
        self.print_to_log_file("This validate took %f s\n" % (epoch_end_time - epoch_start_time))
        return validationDice, ret2   
    
    #def run_validate_white
#%% adversarial part   
    def run_validate_adv_IFGSM_old(self, noise):
        print ("+++++++++++++++++noise ",str(noise)," is running, (IFGSM)+++++++++++++++++++++++++++++++")
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        #_ = self.tr_gen.next()
        #_ = self.val_gen.next()


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

      
        #self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        counter = 1
        epoch_start_time = time()
        # validation with train=False
        self.network.eval()
        #val_losses = []
        #counter = 0
        #print ("num val batches per epoch is ", self.num_val_batches_per_epoch)
        avg = []
        for data_dict in self.ts_gen:
            #check if this target has no foregroud classes, if yes, ignore it
            target = data_dict['target']
            temp = target[0]
            if temp.max()==0:
                continue
            
            #finishe check
            avg.append(self.run_one_adv_ifgsm(data_dict, noise))
            print ("one batch is done")
            if data_dict['last']:
                break
            #if counter ==20:
            #    break
            counter +=1

        ret = self.my_finish_online_evaluation()
        
        avg = np.concatenate(avg)
        ret2 = avg.mean()
        validationDice = np.mean(ret)
        self.print_to_log_file("av global foreground dice: ", ret)
        self.print_to_log_file("av paired dice: (only with complete target)", ret2)
        self.print_to_log_file("validation dice: %.4f" % validationDice)
        epoch_end_time = time()
        self.print_to_log_file("This validate took %f s\n" % (epoch_end_time - epoch_start_time))
        return validationDice, ret2
    
#%%  
    def run_validate_adv_IFGSM(self, noise):
        print ("+++++++++++++++++noise ",str(noise)," is running, (IFGSM and PGD)+++++++++++++++++++++++++++++++")
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        #_ = self.tr_gen.next()
        #_ = self.val_gen.next()


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

      
        #self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        counter = 1
        epoch_start_time = time()
        # validation with train=False
        self.network.eval()
        #val_losses = []
        #counter = 0
        #print ("num val batches per epoch is ", self.num_val_batches_per_epoch)
        avgIFGSM = []
        avgPGD = []
        for data_dict in self.ts_gen:
            #check if this target has no foregroud classes, if yes, ignore it
            target = data_dict['target']
            temp = target[0]
            if temp.max()==0:
                continue
            
            #finishe check
            print ("IFGSM evaluating is running")
            avgIFGSM.append(self.run_one_adv_ifgsm(data_dict, noise))
            print ("PGD evaluatiing is running")
            avgPGD.append(self.run_one_adv(data_dict, noise))
            
            if data_dict['last']:
                break
            #if counter ==20:
            #    break
            counter +=1

        ret = self.my_finish_online_evaluation()
        
        avgPGD = np.concatenate(avgPGD)
        resPGD = avgPGD.mean()
        avgIFGSM = np.concatenate(avgIFGSM)
        resIFGSM = avgIFGSM.mean()
        #validationDice = np.mean(ret)
        #self.print_to_log_file("av global foreground dice: ", ret)
        self.print_to_log_file("AVG ifgsm", resIFGSM)
        self.print_to_log_file("AVG pgd", resPGD)
        #self.print_to_log_file("validation dice: %.4f" % validationDice)
        epoch_end_time = time()
        self.print_to_log_file("This validate took %f s\n" % (epoch_end_time - epoch_start_time))
        return  resIFGSM, resPGD,
    
#%% run test for cmpb revision

    def pgd_attack(self,model, X, Y, noise_norm, norm_type, max_iter, step,
                   rand_init=True, rand_init_norm=None, targeted=False,
                   clip_X_min=0, clip_X_max=1, use_optimizer=False, loss_fn=None):
        #-----------------------------------------------------
        if loss_fn is None :
            raise ValueError('loss_fn is unkown')
        #-----------------
        X = X.detach()
        #-----------------
        if rand_init == True:
            init_norm=rand_init_norm
            if rand_init_norm is None:
                init_norm=noise_norm
            noise_init=self.get_noise_init(norm_type, noise_norm, init_norm, X)
            Xn = X + noise_init
        else:
            Xn = X.clone().detach() # must clone
        #-----------------
        noise_new=(Xn-X).detach()
        if use_optimizer == True:
            optimizer = optim.Adamax([noise_new], lr=step)
        #-----------------
        for n in range(0, max_iter):
            Xn = Xn.detach()
            Xn.requires_grad = True
            Zn = model(Xn)
            loss = loss_fn(Zn, Y)
            #---------------------------
            if targeted == True:
                loss=-loss
            #---------------------------
            #loss.backward() will update W.grad
            grad_n=torch.autograd.grad(loss, Xn)[0]
            grad_n=self.normalize_grad_(grad_n, norm_type)
            if use_optimizer == True:
                noise_new.grad=-grad_n.detach() #grad ascent to maximize loss
                optimizer.step()
            else:
                Xnew = Xn.detach() + step*grad_n.detach()
                noise_new = Xnew-X
            #---------------------
            self.clip_norm_(noise_new, norm_type, noise_norm)
            #Xn = torch.clamp(X+noise_new, clip_X_min, clip_X_max)
            Xn = X+noise_new
            noise_new.data -= noise_new.data-(Xn-X).data
            Xn=Xn.detach()
        #---------------------------
        return Xn
    
    def ifgsm_attack(self,model, X, Y, noise_norm, norm_type, max_iter, step,
                   rand_init=True, rand_init_norm=None, targeted=False,
                   clip_X_min=0, clip_X_max=1, use_optimizer=False, loss_fn=None):
        #-----------------------------------------------------
        if loss_fn is None :
            raise ValueError('loss_fn is unkown')
        #-----------------
        X = X.detach()
        #-----------------
        Xn = X.clone().detach() # must clone
        #-----------------
        noise_new=(Xn-X).detach()
        if use_optimizer == True:
            optimizer = optim.Adamax([noise_new], lr=step)
        #-----------------
        for n in range(0, max_iter):
            Xn = Xn.detach()
            Xn.requires_grad = True
            Zn = model(Xn)
            loss = loss_fn(Zn, Y)
            #---------------------------
            if targeted == True:
                loss=-loss
            #---------------------------
            #loss.backward() will update W.grad
            grad_n=torch.autograd.grad(loss, Xn)[0]
            grad_n=self.normalize_grad_(grad_n, norm_type)
            if use_optimizer == True:
                noise_new.grad=-grad_n.detach() #grad ascent to maximize loss
                optimizer.step()
            else:
                Xnew = Xn.detach() + step*grad_n.detach()
                noise_new = Xnew-X
            #---------------------
            self.clip_norm_(noise_new, norm_type, noise_norm)
            #Xn = torch.clamp(X+noise_new, clip_X_min, clip_X_max)
            Xn = X+noise_new
            noise_new.data -= noise_new.data-(Xn-X).data
            Xn=Xn.detach()
        #---------------------------
        return Xn
    def run_one_adv_ifgsm(self, data_dict, noise, norm_type, max_iter):
        self.network.eval()
        #data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']# target is a mask, but should have two...
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)# only the first target among the six is useful
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)        
        self.optimizer.zero_grad()
        Xn = 0
        stepsize = noise
        if max_iter > 1:
            stepsize = 4*noise/max_iter
            
        if noise == 0:
            Xn = data
        else:
            Xn = self.ifgsm_attack(self.network, data, target, noise, norm_type, max_iter, stepsize, use_optimizer=False, loss_fn=self.loss)
        #ret = 0
        #valDice = DiceIndex()
        with torch.no_grad():
            output = self.network(Xn)
            #ret = self.getOnlineDiceMeanOnlyDoubleClass(output[0], target[0])
            self.my_run_online_evaluation(output, target)         
        del target   
        #return ret.cpu().numpy()

    def run_one_adv_pgd(self, data_dict, noise, norm_type, max_iter):
        self.network.eval()
        #data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']# target is a mask, but should have two...
        data = maybe_to_torch(data)
        target = maybe_to_torch(target)# only the first target among the six is useful
        
        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.optimizer.zero_grad()
        Xn = 0
        stepsize = noise
        if max_iter > 1:
            stepsize = 4*noise/max_iter
        
        if noise == 0:
            Xn = data
        else:
            Xn = self.pgd_attack(self.network, data, target, noise, norm_type, max_iter, stepsize, use_optimizer=False, loss_fn=self.loss)
        
        #ret = 0
        #valDice = DiceIndex()
        with torch.no_grad():
            output = self.network(Xn)
            #ret = self.getOnlineDiceMean(output[0], target[0])
            self.my_run_online_evaluation(output, target)         
        del target   
        #return ret.cpu().numpy()

    def run_validate_adv_cmpb(self, noise, norm_type, attack):
        print ("+++++++++++++++++noise ",str(noise)," is running, (cmpb)+++++++++++++++++++++++++++++++")
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._maybe_init_amp()    
        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)
        counter = 1
        epoch_start_time = time()
        self.network.eval()
        
        print (attack + " evaluating is running")
        for data_dict in self.ts_gen:
            #check if this target has no foregroud classes, if yes, ignore it
            target = data_dict['target']
            temp = target[0]
            if temp.max()==0:
                continue
            if attack == "fgsm":
                self.run_one_adv_ifgsm(data_dict, noise, norm_type, max_iter = 1)
            elif attack == "ifgsm":
                self.run_one_adv_ifgsm(data_dict, noise, norm_type, max_iter = 10)
            elif attack == "pgd":
                self.run_one_adv_pgd(data_dict, noise, norm_type, max_iter = 10)
            else:
                raise Exception("Sorry, attack not supported")
                
            if data_dict['last']:
                break
            counter +=1
        ret3 = self.my_finish_online_evaluation()
        val3 = np.mean(ret3)        
        epoch_end_time = time()
        self.print_to_log_file("This validate took %f s\n" % (epoch_end_time - epoch_start_time))
        return  val3
    
#%% adversarial part   
    def run_validate_sample_wise(self, noise):
        print ("+++++++++++++++++noise ",str(noise)," is running+++++++++++++++++++++++++++++++")
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")
        if not self.was_initialized:
            self.initialize(True)
        counter = 1
        epoch_start_time = time()
        self.network.eval()
        avg = []
        for data_dict in self.ts_gen:
            #check if this target has no foregroud classes, if yes, ignore it
            target = data_dict['target']
            temp = target[0]
            if temp.max()==0:
                continue
            
            #finishe check
            avg.append(self.run_one_adv(data_dict, noise))
            print ("one batch is done")
            if data_dict['last']:
                break
            #if counter ==20:
            #    break
            counter +=1

        ret = self.my_finish_online_evaluation()
        
        avg = np.concatenate(avg)
        ret2 = avg
        validationDice = ret       
        self.print_to_log_file("av global foreground dice: ", ret)
        self.print_to_log_file("av paired dice: (only with complete target)", ret2.shape)
        epoch_end_time = time()
        self.print_to_log_file("This validate took %f s\n" % (epoch_end_time - epoch_start_time))
        return validationDice, ret2

#%% adversarial part   
    def run_validate_adv_showcase(self, noise):
        print ("+++++++++++++++++noise ",str(noise)," is running+++++++++++++++++++++++++++++++")
        if not torch.cuda.is_available():
            self.print_to_log_file("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

        #_ = self.tr_gen.next()
        #_ = self.val_gen.next()


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

      
        #self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. "
                 "But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! "
                 "If you want deterministic then set benchmark=False")

        if not self.was_initialized:
            self.initialize(True)

        counter = 1
        epoch_start_time = time()
        # validation with train=False
        self.network.eval()
        #val_losses = []
        #counter = 0
        #print ("num val batches per epoch is ", self.num_val_batches_per_epoch)
        avg = []
        for data_dict in self.ts_gen:
            #check if this target has no foregroud classes, if yes, ignore it
            target = data_dict['target']
            temp = target[0]
            if temp.max()==0:
                continue
            
            #finishe check
            _, Xn, Yn = self.run_one_adv_to_show(data_dict, noise)
            print ("one case is done")
            break
            if data_dict['last']:
                break
            #if counter ==20:
            #    break
        return Xn,Yn
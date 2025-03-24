"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""

from pathlib import Path
from threading import Thread
import streamlit as st
import pandas as pd
import torch.nn as nn
import torch
import os
import numpy as np
from torch.utils.data import Dataset, SubsetRandomSampler
from torch import optim
from utils.config_utils import revise_cfg_file
# OpenLabCluster import
from training_utils.ssl.SeqModel import seq2seq, SemiSeq2Seq
from training_utils.ssl.seq_train import training, clustering_knn_acc



class train_unsup_network(Thread):
    def __init__(self, config,
                 displayiters=None,
                 saveiters=None,
                 maxiters=None,
                 continue_training=False,
                 reducer_name='PCA',
                 dimension=2):
        """Initializes a thread to train the sequence-to-sequence model for behavior clustering (Unsupervised)
        Inputs:
            config: the directory of a config file
            canvas: the plot handle
            displayiters: frequency to update the plots
            saveiters: frequency to save the trained model
            maxiters: maximum number of training epochs
            continue_training: if true, uses pretrained model otherwise train from scratch
            reducer_name: method name for dimension reduction, options: "PCA", "tSNE", "UMAP"
            dimension: the number of dimensions to display
        """
        Thread.__init__(self)
        self.displayiters = displayiters
        self.maxiters = maxiters
        self.saveiters = saveiters
        self.cfg = config
        self.stop_work_thread = 0
        self.continue_training = continue_training
        self.reducer_name = reducer_name
        self.dimension = dimension
        self.init_train_variable()

    def update_parameters(self, reducer_name, dimension):
        self.reducer_name = reducer_name
        self.dimension = dimension
    def init_train_variable(self):
        """
        Starts to train the unsupervised clustering model
        """

        print(self.cfg)

        cfg = self.cfg
        self.num_class = cfg['Model']['num_class']
        self.root_path = cfg['Project_folders']["project_path"]
        self.batch_size = cfg['Training']['batch_size']
        self.model_name = cfg['Model']['tr_modelName']
        self.model_type = cfg['Model']['tr_modelType']
        self.cla_dim = self.cfg['Model']['cla_dim']
        import sys
        if len(cfg['Dataset']['train']) != 0:
            st.write('Start data loading')
            from training_utils.ssl.data_loader import UnsupData, pad_collate_iter, get_data_paths
            self.dataset_train = UnsupData(get_data_paths(self.root_path, cfg['Project_folders']['data_path'], cfg['Dataset']['train']))

            dataset_size_train = len(self.dataset_train)

            indices_train = list(range(dataset_size_train))

            random_seed = 11111

            np.random.seed(random_seed)
            np.random.shuffle(indices_train)

            print("training data length: %d" % (len(indices_train)))
            # Seperates train and validation
            train_sampler = SubsetRandomSampler(indices_train)
            self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size,
                                                       sampler=train_sampler, collate_fn=pad_collate_iter)

        fix_weight = cfg['Model']['fix_weight']
        fix_state = cfg['Model']['fix_state']
        teacher_force = cfg['Model']['teacher_force']
        phase = 'PC'
        if fix_weight:
            self.network = 'FW' + phase

        if fix_state:
            self.network = 'FS' + phase

        if not fix_state and not fix_weight:
            self.network = 'O' + phase

        # Hyperparameters
        # global variables
        self.feature_length = cfg['Model']['feature_length']
        self.hidden_size = cfg['Model']['hidden_size']
        self.batch_size = cfg['Training']['batch_size']
        self.en_num_layers = cfg['Model']['en_num_layers']
        self.de_num_layers = cfg['Model']['de_num_layers']
        self.learning_rate = cfg['Training']['learning_rate']
        self.cla_num_layers = self.cfg['Model']['cla_num_layers']
        self.num_class = self.cfg['Model']['num_class']

        self.device = cfg['Training']['device']
        if not torch.cuda.is_available():
            self.device = 'cpu'
        self.percentage = 1
        self.few_knn = False

        # Initializes the model
        self.model = seq2seq(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                        self.en_num_layers, self.de_num_layers, fix_state, fix_weight, teacher_force, self.device).to(self.device)

        with torch.no_grad():
            for child in list(self.model.children()):
                print(child)
                for param in list(child.parameters()):
                    if param.dim() == 2:
                        nn.init.uniform_(param, a=-0.05, b=0.05)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        if self.model_type == 'seq2seq':
            if isinstance(self.model_name, str) and os.path.exists(self.model_name) and self.continue_training:
                from training_utils.ssl.utilities import load_model
                model, self.optimizer = load_model(self.model_name, self.model, self.optimizer, self.device)
        elif self.model_type == 'semi_seq2seq':
            from training_utils.ssl.utilities import load_model
            if isinstance(self.model_name, str) and os.path.exists(self.model_name) and self.continue_training:
                model = SemiSeq2Seq(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                                     self.cla_dim, self.en_num_layers, self.de_num_layers, self.cla_num_layers,
                                     fix_state, fix_weight, teacher_force, device=self.device)
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                            lr=self.learning_rate)
                model, _ = load_model(self.model_name, model, optimizer, self.device)

                self.model = model.seq
                self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                            lr=self.learning_rate)
                del model, optimizer
        loss_type = 'L1'

        if loss_type == 'MSE':
            self.criterion_seq = nn.MSELoss(reduction='none')

        if loss_type == 'L1':
            self.criterion_seq = nn.L1Loss(reduction='none')

        self.criterion_cla = nn.CrossEntropyLoss(reduction='sum')

        self.alpha = 0

        self.file_output = open(
            os.path.join(self.root_path, self.cfg['Project_folders']['output_path'], '%sA%.2f_P%d_en%d_hid%d.txt' % (
                self.network, self.alpha, self.percentage * 100, self.en_num_layers, self.hidden_size)), 'a')
        self.model_prefix = os.path.join(self.root_path, cfg['Project_folders']['model_path'], '%sA%.2f_P%d_en%d_hid%d' % (
            self.network, self.alpha, self.percentage * 100, self.en_num_layers, self.hidden_size))
        self.model_path = Path(self.model_prefix).parent
        self.pre = Path(self.model_prefix).name
        lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 5)
        self.model_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.past_loss = sys.float_info.max

    def train_step(self, ith_epoch):
        print('one training step')
        past_loss, self.path_model = training(ith_epoch, self.maxiters, self.train_loader,
                                              self.model, self.optimizer, self.criterion_seq, self.criterion_cla, self.alpha, self.past_loss,
                                              self.model_path, self.pre,
                                              self.model_prefix,
                                              self.device)
        # self.model_scheduler.step()



        if ith_epoch % 50 == 0:

            filename = self.file_output.name
            self.file_output.close()
            self.file_output = open(filename)

        if self.path_model:
            kwargs = {'tr_modelName': self.path_model, 'tr_modelType': 'seq2seq'}
            revise_cfg_file(self.cfg, **kwargs)
        else:
            kwargs =  {'tr_modelType': 'seq2seq'}
            revise_cfg_file(self.cfg, **kwargs)
        if ith_epoch % self.displayiters == 0:
            # update the plot based on the displayiters
            return self.plot_data(ith_epoch)
        else:
            return None


    def plot_data(self, ith_epoch):
        """
        Plots sequences in the reduced dimension space
        """
        transformed,_ = clustering_knn_acc(
            self.model,
            self.train_loader,
            self.hidden_size,
            self.alpha,
            ith_epoch, self.device, self.reducer_name, self.dimension)
        index = np.arange(start=0, stop=len(transformed))[:, None]
        if self.dimension == '2d':
            data = pd.DataFrame(np.concatenate([transformed, index], axis=-1), columns=['x', 'y', 'index'])
        elif self.dimension == '3d':
            data = pd.DataFrame(np.concatenate([transformed, index], axis=-1), columns=['x','y','z', 'index'])
        return data


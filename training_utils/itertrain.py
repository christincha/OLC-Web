"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from threading import Thread

import torch
import torch.nn as nn
from torch.utils.data import Dataset, SubsetRandomSampler
from torch import optim

# OpenLabCluster import
from training_utils.ssl.SeqModel import SemiSeq2Seq
from training_utils.ssl.SeqModel import seq2seq
from training_utils.ssl.seq_train import training
from training_utils.ssl.utilities import load_model
from training_utils.ssl.seq_train import clustering_knn_acc
from utils.config_utils import revise_cfg_file
from training_utils.ssl.clustering_classification import iter_kmeans_cluster, \
    remove_labeled_cluster
from training_utils.ssl.labelSampling import SampleFromCluster
from training_utils.ssl.data_loader import SupDataset, pad_collate_iter
from training_utils.ssl.core_set import CoreSet
class train_iter_network(Thread):
    def __init__(self, config, sample_method, num_sample, reducer_name,
                 epochs: int = None,
                  acc_text=None):
        """Initializes the thread for training behavior classification
        Inputs:
            config: the directory of the config file
            canvas: the plot handle
            epochs: maximum number of training epochs
            model_name: the pretrained model directory
            model_type: str, 'seq2seq' for unsupervised learning 'semi_seq2seq' for semi-supervised learning
            acc_test: the training accuracy
        """
        Thread.__init__(self)


        self.cfg = config

        self.model_name = self.cfg['Model']['tr_modelName']
        self.model_type = self.cfg['Model']['tr_modelType']
        self.sample_method = sample_method
        self.num_sample = num_sample
        self.reducer_name = reducer_name
        self.acc_text = acc_text
        self.epochs = epochs
        self.init_train_variable()
        # Read file path for pose_config file. >> pass it on
    def update_parameters(self, sample_method, num_sample,reducer_name):
        self.sample_method = sample_method
        self.num_sample = num_sample
        self.reducer_name = reducer_name

    def init_train_variable(self):
        """
        Trains the semi-supervised behavior classification model (Initiated by Start)
        """

        self.num_class = self.cfg['Model']['num_class']
        self.root_path = self.cfg['Project_folders']["project_path"]
        self.batch_size = self.cfg['Training']['batch_size']
        self.displayiters = 3

        if len(self.cfg['Dataset']['train']) != 0:

            label_path = os.path.join(self.cfg['Project_folders']['label_path'], 'label.npy')
            if not os.path.exists(label_path):
                label_path = None
            dataset_train = SupDataset(self.root_path, self.cfg['Project_folders']['data_path'], self.cfg['Dataset']['train'], label_path)

            dataset_size_train = len(dataset_train)
            indices_train = list(range(dataset_size_train))

            random_seed = 11111

            np.random.seed(random_seed)
            np.random.shuffle(indices_train)

            print("training data length: %d" % (len(indices_train)))
            # Prepares training dataloader
            train_sampler = SubsetRandomSampler(indices_train)
            self.train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size,
                                                       sampler=train_sampler, collate_fn=pad_collate_iter)

        fix_weight = self.cfg['Model']['fix_weight']
        fix_state = self.cfg['Model']['fix_state']
        teacher_force = self.cfg['Model']['teacher_force']
        phase = 'PC'
        if fix_weight:
            self.network = 'FW' + phase

        if fix_state:
            self.network = 'FS' + phase

        if not fix_state and not fix_weight:
            self.network = 'O' + phase

        # Hyperparameters
        self.feature_length = self.cfg['Model']['feature_length']
        self.hidden_size = self.cfg['Model']['hidden_size']
        self.batch_size = self.cfg['Training']['batch_size']
        self.en_num_layers = self.cfg['Model']['en_num_layers']
        self.de_num_layers = self.cfg['Model']['de_num_layers']
        self.cla_num_layers = self.cfg['Model']['cla_num_layers']
        self.learning_rate = self.cfg['Training']['learning_rate']
        self.epoch = self.epochs if self.epochs is not None else self.cfg['Training']["su_epoch"]

        self.device = self.cfg['Training']['device']
        self.percentage = 1
        self.few_knn = False
        # Global variable
        self.cla_dim = self.cfg['Model']['cla_dim']  # 0 non labeled class

        self.print_every = 1

        self.model = SemiSeq2Seq(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                            self.cla_dim, self.en_num_layers, self.de_num_layers, self.cla_num_layers,
                            fix_state, fix_weight, teacher_force, device=self.device)
        print('network fix state=', fix_state)

        with torch.no_grad():
            for child in list(self.model.children()):
                print(child)
                for param in list(child.parameters()):
                    if param.dim() == 2:
                        # nn.init.xavier_uniform_(param)
                        nn.init.uniform_(param, a=-0.05, b=0.05)

        if self.model_type == 'seq2seq':
            model_tmp = seq2seq(self.feature_length, self.hidden_size, self.feature_length, self.batch_size,
                                self.en_num_layers, self.de_num_layers,
                                fix_state, fix_weight, teacher_force, device=self.device)
            optimizer_tmp = optim.Adam(filter(lambda p: p.requires_grad, model_tmp.parameters()), lr=self.learning_rate)
            model_tmp, _ = load_model(self.model_name, model_tmp, optimizer_tmp, self.device)
            self.model.seq = model_tmp
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)

        elif self.model_type == 'semi_seq2seq':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
            self.model, self.optimizer = load_model(self.model_name, self.model, self.optimizer, self.device)
        loss_type = 'L1'

        if loss_type == 'MSE':
            self.criterion_seq = nn.MSELoss(reduction='none')

        if loss_type == 'L1':
            self.criterion_seq = nn.L1Loss(reduction='none')

        self.criterion_cla = nn.CrossEntropyLoss(reduction='sum')
        alpha=0.5
        self.file_output = open(os.path.join(self.root_path, self.cfg['Project_folders']['output_path'], '%sA%.2f_P%d_en%d_hid%d.txt' % (
            self.network, alpha, self.percentage * 100, self.en_num_layers, self.hidden_size)), 'w')
        self.model_prefix = os.path.join(self.root_path, self.cfg['Project_folders']['model_path'], '%sA%.2f_P%d_en%d_hid%d' % (
            self.network, alpha, self.percentage * 100, self.en_num_layers, self.hidden_size))
        self.model_path = Path(self.model_prefix).parent
        self.pre = Path(self.model_prefix).name
        lambda1 = lambda ith_epoch: 0.95 ** (ith_epoch // 5)
        self.model_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.past_loss = sys.float_info.max
        self.alpha = alpha

    def update_dataloader(self):
        # Update dataloader to include labeled samples
        label_path = os.path.join(self.cfg['Project_folders']['label_path'], 'label.npy')
        if not os.path.exists(label_path):
            label_path = None
        dataset_train = SupDataset(self.root_path, self.cfg['Project_folders']['data_path'],
                                   self.cfg['Dataset']['train'], label_path)

        dataset_size_train = len(dataset_train)
        indices_train = list(range(dataset_size_train))

        random_seed = 11111

        np.random.seed(random_seed)
        np.random.shuffle(indices_train)

        print("training data length: %d" % (len(indices_train)))
        # Prepares training dataloader
        train_sampler = SubsetRandomSampler(indices_train)
        self.train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size,
                                                        sampler=train_sampler, collate_fn=pad_collate_iter)

    def train_step(self, ith_epoch):
        past_loss, model_name, self.acc = training(ith_epoch, self.epoch, self.train_loader,
                                                   self.model, self.optimizer, self.criterion_seq, self.criterion_cla,
                                                   self.alpha, self.past_loss,
                                                   self.model_path, self.pre,
                                                   self.model_prefix,
                                                   self.device)
        if ith_epoch % self.print_every == 0:
            self.ith_epoch = ith_epoch
        if model_name:
            kwargs =  {'tr_modelType': 'semi_seq2seq', 'tr_modelName': model_name}
            revise_cfg_file(self.cfg, **kwargs)
        else:
            kwargs = {'tr_modelType': 'semi_seq2seq'}
            revise_cfg_file(self.cfg, **kwargs)
        self.model_scheduler.step()

        if ith_epoch % 50 == 0:
            filename = self.file_output.name
            self.file_output.close()
            self.file_output = open(filename, 'a')
        if ith_epoch % self.displayiters == 0:
            # update the plot based on the displayiters
            return self.plot_data(ith_epoch)
        else:
            return None, None

    def plot_data(self, ith_epoch):
        """
        Plots learned hidden states in an embedded space
        """
        out = clustering_knn_acc(
            self.model,
            self.train_loader,
            self.hidden_size,
            self.alpha,
            ith_epoch, self.device, reducer_name=self.reducer_name, dimension='2d')
        if isinstance(out, tuple) and len(out) ==3:
            transformed, semilabel, mi = out[0], out[1], out[2]
        else:
            transformed, semilabel = out[0], out[1]
        index = np.arange(start=0, stop=len(transformed))[:, None]
        data = pd.DataFrame(np.concatenate([transformed, index], axis=-1), columns=['x', 'y', 'index'])
        return data, semilabel

    def active_label_selection(self, ith_epoch):

        out = clustering_knn_acc(
            self.model,
            self.train_loader,
            self.hidden_size,
            self.alpha,
            ith_epoch, self.device, reducer_name='PCA', dimension='2d', return_full=True)
        if isinstance(out, tuple) and len(out) ==3:
            hidarray, semilabel, mi = out[0], out[1], out[2]
        else:
            hidarray, semilabel = out[0], out[1]


        toLabel = np.where(semilabel != 0)[0].tolist()
        index_train_complete = np.arange(0, hidarray.shape[0])
        if len(toLabel) == 0:

            if self.sample_method == 'core_set':
                self.cor_set = CoreSet(hidarray, metric='euclidean')
                self.suggest = self.cor_set.select_samples(hidarray,self.num_sample, toLabel)
            else:
                hi_train, index_train = remove_labeled_cluster(hidarray,
                                                               index_train_complete, toLabel)
                train_id_list, dis_list, dis_list_prob, label_list = iter_kmeans_cluster(hi_train, index_train,
                                                                                         self.num_sample)
                self.suggest = SampleFromCluster(train_id_list, dis_list, dis_list_prob, "Cluster Center (Top)",
                                                 self.num_sample)
        else:
            if self.sample_method == 'core_set':
                self.cor_set = CoreSet(hidarray, metric='euclidean')
                self.suggest = self.cor_set.select_samples(hidarray,self.num_sample, toLabel)
            else:
                hi_train, index_train, mi, = remove_labeled_cluster(hidarray,
                                                                    index_train_complete, toLabel, mi)

                train_id_list, dis_list, dis_list_prob, label_list = iter_kmeans_cluster(hi_train, index_train,
                                                                                         self.num_sample, mi)
                print('sample from cluster', self.sample_method)
                self.suggest = SampleFromCluster(train_id_list, dis_list, dis_list_prob, self.sample_method,
                                                 self.num_sample)
        return self.suggest
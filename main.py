# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler
from dataset.TimeDataset import timeDataset
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep
from sklearn import preprocessing
#from preprocess import trainprepro
from test_preprocess import test_prepro
from scipy.fftpack import fft,fftshift
from thop import profile




from model.GG_Nets import GDN,Discrimanitor

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
        """
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)


       
        train1, test1 = train_orig, test_orig
        #print(train)

        if 'attack' in train1.columns:
            train = train1.drop(columns=['attack'])
            """
        """
        
        train = np.load('I:\\Project\\CNN-1DMEAE\\train_co.npy')[:,:,:,:]
        valid = train
        test = np.load('I:\\Project\\CNN-1DMEAE\\test_co.npy')
        np.random.shuffle(train)
        train_Y = []
        valid_Y = []
        test_Y = []
        for i in range(train.shape[0]):
            train_Y.append(0)
        for i in range(valid.shape[0]):
            valid_Y.append(0)
        for i in range(100060):
            test_Y.append(0)
        for i in range(5785):
            test_Y.append(1)
        train_Y = np.array(train_Y)
        valid_Y = np.array(valid_Y)
        test_Y = np.array(test_Y)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
        for i in range(test.shape[0]):
            test[i][0] = min_max_scaler.fit_transform(test[i][0])
        for i in range(train.shape[0]):
            train[i][0] = min_max_scaler.fit_transform(train[i][0])
        for i in range(valid.shape[0]):
            valid[i][0] = min_max_scaler.fit_transform(valid[i][0])
        """


        data = np.load('G:\毕业设计\CNN-1DMEAE\\trainnewnew.npz')
        valid = data['valid'][:,:,:,:]
        train = data['train'][:10000,:,:,:]
        np.random.shuffle(train)
        train = train[:, :, :, :]
        train_Y =[]
        valid_Y = []
        for i in range(train.shape[0]):
            train_Y.append(0)
        for i in range(valid.shape[0]):
            valid_Y.append(0)
        train_Y = np.array(train_Y)
        valid_Y = np.array(valid_Y)
        #print(train_Y)
        test_X, test_Y = test_prepro(d_path='G:\毕业设计\数据集\速变数据\\testdata', test_length=1000, test_number=1000,
                                     normal=False)
        test_X = np.expand_dims(test_X, axis=1)
        #print("test",test_X.shape)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
        for i in range(2000):
            test_X[i][0] = min_max_scaler.fit_transform(test_X[i][0])
        for i in range(train.shape[0]):
            train[i][0] = min_max_scaler.fit_transform(train[i][0])
        for i in range(valid.shape[0]):
            valid[i][0] = min_max_scaler.fit_transform(valid[i][0])


        feature_map = get_feature_map(dataset)
        #print(feature_map)


        fc_struc = get_fc_graph_struc(dataset)
        #print(fc_struc)
        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, feature_map, feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        #train_dataset_indata = construct_data(train, feature_map, labels=0)
        #test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())



        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
            'predict_num': env_config['predict_num'],
            'missing_rate': env_config['missing_rate']
        }

        train_dataset = timeDataset(train[:,:,:,:1000],train_Y[:],fc_edge_index, mode='train', config=cfg)
        valid_dataset = timeDataset(valid[:,:,:,:1000],valid_Y,fc_edge_index, mode='trian',config=cfg)
        test_dataset = timeDataset(test_X[:,:,:,:1000], test_Y, fc_edge_index, mode='test', config=cfg)


        #train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch'],
                            shuffle=True, num_workers=0)
        self.val_dataloader = DataLoader(valid_dataset, batch_size=1,
                            shuffle=False, num_workers=0)
        self.test_dataloader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, num_workers=0)


        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)
        #print(edge_index_sets)
        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=(1000-env_config['predict_num']),
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk'],
                predict_num = env_config['predict_num']
            ).to(self.device)
        self.model_D = Discrimanitor().to(self.device)



    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(self.model,self.model_D, model_save_path,
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                #dataset_name=self.env_config['dataset']
            )
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)
        #input = torch.randn(16,18, 850).to(self.device)
        #flopparams = profile(best_model, inputs=(input,))
        #print(flopparams)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)
        #print(self.test_result)
        #print(self.val_result)
        self.get_score(self.test_result, self.val_result)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        #print(test_result)
        np_test_result = np.array(test_result)
        #print()
        np_val_result = np.array(val_result)
        #print("npshape",np_test_result.shape)
        test_labels=test_result[2]
        #test_labels = np_test_result[2, :].tolist()
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)

        for i in range(18):

            top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=i)
            info = top1_val_info
            print(f'F1 score: {info[0]}')
            print(f'precision: {info[1]}')
            print(f'recall: {info[2]}')
            print(f'auc: {info[3]}')

        print('=========================** Result **============================\n')


        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=4)
        info = top1_val_info

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}')
        print(f'auc: {info[3]}')


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m.%d.%H.%M.%S')
        datestr = self.datestr          

        paths = [
            f'.\\\pretrained\\{dir_path}\\best_{datestr}.pt',
            f'.\\results\\{dir_path}\\{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=32)
    parser.add_argument('-epoch', help='train epoch', type = int, default=2)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-predict_num',type= int , default=150)
    parser.add_argument('-dim', help='dimension', type = int, default=250)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='LRE')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=3)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=10)
    parser.add_argument('-report', help='best / val', type = str, default='val')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-missing_rate', help='missing rate of data', type=float, default=0)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'predict_num': args.predict_num
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path,
        'predict_num' : args.predict_num,
        'missing_rate': args.missing_rate
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()






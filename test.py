import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from tqdm import tqdm

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F

import seaborn as sns
from util.data import *
from util.preprocess import *

def dis_loss(M,d_out):
    M = torch.squeeze(M, 1)
    M = M[:, :, 0]
    return -torch.mean(M * torch.log(d_out+ 1e-8) + (1-M) * torch.log(1. - d_out + 1e-8))

def gen_loss(M,d_out,New_X,X,G_sample):
    M = torch.squeeze(M, 1)
    M_S = M[:,:,0]
    G_loss1 = -torch.mean((1 - M_S) * torch.log(d_out + 1e-8))

    G_loss2_train = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)
    G_loss2_test = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    alpha = 10
    G_loss = G_loss1+alpha*G_loss2_train
    return G_loss,G_loss2_train,G_loss2_test

def test(model, dataloader):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []
    t_test_mask_list = []
    t_test_x_gen_list = []
    generatorlist0 = []
    generatorlist1 = []
    generatorlist2 = []
    generatorlist3 = []
    generatorlist4 = []
    generatorlist5 = []
    generatorlist6 = []
    generatorlist7 = []
    generatorlist8 = []
    generatorlist9 = []
    generatorlist10 = []
    generatorlist11 = []
    generatorlist12 = []
    generatorlist13 = []
    generatorlist14 = []
    generatorlist15 = []
    generatorlist16 = []
    generatorlist17 = []
    #generatorlist = np.zeros((1,18,1000))
    # generatorlist0 = []

    #generatorlist={"generatorlist0"=[],}
    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, raw,labels, edge_index,mask in tqdm(dataloader):
        x, y, raw,labels, edge_index,mask = [item.to(device).float() for item in [x, y,raw, labels, edge_index,mask]]
        #print(
        #    "x",x.shape
        #)
        with torch.no_grad():
            predicted,y_gen,x_gen = model(x,raw, edge_index,mask)
            predicted = predicted.float().to(device)
            predicted = predicted.float().to(device)
            y_gen = y_gen.float().to(device)
            x_gen = x_gen.float().to(device)
            #g_samples = g_samples.float().to(device)
            #print("mask",mask.shape)
            #print("x_gen",x_gen.shape)
            """for i in range(mask.shape[0]):
                for j in range(mask.shape[2]):

                    if mask[i,0,j,0] == 0:
                        (generatorlist+str(j))
                        """



            
            loss = loss_func(predicted, y_gen) #+ loss_func(y_gen,y)
            

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])
            #mask = mask.cpu()

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y_gen
                t_test_labels_list = labels
                t_test_mask_list = mask.cpu()
                t_test_x_gen_list = x_gen.cpu()
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y_gen), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
                t_test_x_gen_list = torch.cat((t_test_x_gen_list, x_gen.cpu()), dim=0)
                t_test_mask_list = torch.cat((t_test_mask_list, mask.cpu()), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))
        #if i==1005:
        #    sns.heatmap(attentionou.cpu().detach().numpy()[0])
        #    plt.show()

    #test_predicted_list = np.array(test_predicted_list)
    #print(test_predicted_list)
    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()
    #test_mask_list = t_test_mask_list.tolist()
    #test_x_gen_list = t_test_x_gen_list.tolist()
    #test_x_gen = np.array(test_x_gen_list)
    #test_x_mask = np.array(test_mask_list)
    #test_x_mask = test_x_mask
    #test_x_gen = test_x_gen
    #print("mask",test_x_mask.shape)
    #print("xgen",test_x_gen.shape)
    #np.savez(".//xgen//xgen0.4.npz",xgen=test_x_gen,mask=test_x_mask)
    #np.savez(".//xgen//mask0.4em.npz",)
    test_predicted_list1=[]
    test_ground_list1=[]
    test_labels_list1=[]
    test_mask_list1 = []
    test_x_gen_list1 = []
    #printtest_predicted_list[1])

    for i in range(test_len):
        """
        N = 425
        sample_freq = 12800
        signal = test_X[1, 0, 3, b]

        # fft_data = np.fft.fft(signal)
        fft_data = fft(signal)
        # 这里幅值要进行一定的处理，才能得到与真实的信号幅值相对应
        fft_amp0 = np.array(np.abs(fft_data) / N * 2)  # 用于计算双边谱
        direct = fft_amp0[0]
        fft_amp0[0] = 0.5 * direct
        N_2 = int(N / 2)

        fft_amp1 = fft_amp0[0:N_2]  # 单边谱
        fft_amp0_shift = fftshift(fft_amp0)  # 使用fftshift将信号的零频移动到中间

        # 计算频谱的频率轴
        list0 = np.array(range(0, N))
        list1 = np.array(range(0, int(N / 2)))
        list0_shift = np.array(range(0, N))
        freq0 = sample_freq * list0 / N  # 双边谱的频率轴
        freq1 = sample_freq * list1 / N  # 单边谱的频率轴
        freq0_shift = sample_freq * list0_shift / N - sample_freq / 2
        """

        test_predicted_list[i]=np.array(test_predicted_list[i]).sum(axis= 1)
        test_ground_list[i] = np.array(test_ground_list[i]).sum(axis=1)
        #test_x_gen_list[i] = np.array(test_x_gen_list[i]).sum(a)
    #for i in range(2)
    #print(test_predicted_list[1])



    for i in range(test_len):
        test_labels_list[i] = np.array(test_labels_list[i])
        test_predicted_list[i] = np.array(test_predicted_list[i])
        test_ground_list[i] = np.array(test_ground_list[i])
        sum_predicted = sum(test_predicted_list[i])
        sum_ground = sum(test_ground_list[i])
        sum_labels = sum(test_labels_list[i])
        test_labels_list1.append(sum_labels)
        test_predicted_list1.append(sum_predicted)
        test_ground_list1.append(sum_ground)

    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]





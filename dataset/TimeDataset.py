import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class timeDataset(Dataset):
    def __init__(self, raw_data,label, edge_index, mode='train', config = None):
        self.raw_data = raw_data
        #print(raw_data.shape)

        self.config = config
        self.missing_rate = self.config['missing_rate']

        self.edge_index = edge_index
        self.mode = mode
        self.predict_num = config['predict_num']
        ones = np.ones(raw_data.shape[0]*18)
        missing_num = int(self.missing_rate * 18 * raw_data.shape[0])
        ones[:missing_num] = 0
        np.random.shuffle(ones)
        #print(raw_data.shape[0])
        self.Mask = ones.reshape(raw_data.shape[0],1,18,1).repeat(1000,3)
        #print("M_shape",self.Mask.shape)
        #if(self.Mask[:,:,:,0].all()==self.Mask[:,:,:,1].all()):
       #     print(True)





        #Mask = np.ones([18,1000],dype = int)


        x_data = raw_data * self.Mask
        #print(x_data.shape)

        labels = label
        #print(labels)


        data = x_data

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()
        self.Mask = torch.tensor(self.Mask).double()

        self.x, self.y, self.raw,self.mask,self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)


    def process(self, data, labels):
        x_arr, y_arr,raw_arr,mask_arr= [], [], [],[]
        labels_arr = []
        """
        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train'
        print("datashape",data.shape)
        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        for i in rang:

            ft = data[:, i-slide_win:i]
            tar = data[:, i]

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])
        """
        #print("mask_shape", self.Mask[1, :, :, :].shape)
        for i in range(data.shape[0]):
            x_arr.append(data[i,0,:,:(1000-self.predict_num)])
            raw_arr.append(data[i,0,:,:])
            #print(data[i,0,:,:999].shape)
            y_arr.append(data[i,0,:,(1000-self.predict_num):])

            mask_arr.append(self.Mask[i,:,:,:1000])
            labels_arr.append(labels[i])
        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        raw = torch.stack(raw_arr).contiguous()
        mask = torch.stack(mask_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, raw,mask,labels

    def __getitem__(self, idx):

        feature = self.x[idx].double()
        y = self.y[idx].double()
        raw = self.raw[idx].double()
        mask = self.mask[idx].double()
        #mask = mask.squeeze(1)
        #print("ms..",mask.shape)
        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y,raw, label, edge_index, mask






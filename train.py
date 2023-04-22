import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
from torchviz import make_dot
from visdom import Visdom
import time
from tqdm import tqdm


def get_score(test_result, val_result):
    #feature_num = len(test_result[0][0])
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)
    # print("npshape",np_test_result.shape)
    test_labels = np_test_result[2, :].tolist()

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
    return info




def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')
    #print(loss)

    return loss
def dis_loss(M,d_out):
    M = torch.squeeze(M, 1)
    d_out = torch.squeeze(d_out,2)
    M = M[:, :, 0]
    #print("mshape",M.shape)
    #print("doutshape",d_out.shape)
    return -torch.mean(M * torch.log(d_out + 1e-8) + (1-M) * torch.log(1. - d_out + 1e-8))

def gen_loss(M,d_out,New_X,X,G_sample):
    M = torch.squeeze(M, 1)
    d_out = torch.squeeze(d_out, 2)
    M_S = M[:,:,0]
    G_loss1 = -torch.mean((1 - M_S) * torch.log(d_out + 1e-8))
    #print(New_X.shape)
    #print(G_sample.shape)
    G_loss2_train = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)
    G_loss2_test = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    alpha = 10
    #print("G_LOSS1",G_loss1)
    #print("G_LOSS2",G_loss2_train)
    G_loss = G_loss1+alpha*G_loss2_train
    #G_loss = alpha * G_loss2_train
    return G_loss#,G_loss2_train,G_loss2_test




def train(model = None,model_D=None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']
    """viz = Visdom()
    viz.line([0.], [0], win='lossD', opts=dict(title='lossD', legend=['loss_D']))
    viz.line([0.], [0], win='lossG', opts=dict(title='lossg', legend=['loss_G']))
    """
    #viz.line([[0.,0.]],[0],win='origal',opts=dict(title='origal&generated',legend = ['ori','generated']))
    optimizerG = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    optimizerD = torch.optim.Adam(model_D.parameters(), lr=0.01,  weight_decay=config['decay'])
    #print(model.parameters())
    #print(model_D.parameters)
    """for name, parameters in model.named_parameters():

        print(name, ':', parameters)
        """
    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 30
    #model_D.to(device)
    model.train()
    model_D.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader
    f1_list=[]
    auc_list=[]
    recall_list=[]
    pre_list=[]
    for i_epoch in tqdm(range(epoch)):


        acu_loss = 0
        model.train()
        model_D.train()

        for x, labels, raw, attack_labels, edge_index,mask in dataloader:
            _start = time.time()
            #print("ms=", mask.shape)
            x, labels,raw, edge_index,mask = [item.to(device).float() for item in [x, labels,raw, edge_index,mask]]
            #
            #print(x.shape)
            #print("ms",mask.shape)

            out,y_gen,x_gen = model(x,raw, edge_index,mask)

            raw = raw.float().to(device)
            x_gen = x_gen.float().to(device)
            #print(x_gen.device)
            #print(model_D.device)

            #d_out = model_D(raw,x_gen)
            #print("rawshape",raw.shape)
            #print("maskshape",mask.shape)
            #print("g_sampleshape",g_sample.shape)



            #d_out = d_out.float().to(device)
            #d_out_G = d_out.detach().clone()
            #mask_G = mask.detach().clone()
            x_gen_G = x_gen.detach().clone()
            #raw_G = raw.detach().clone()
            #g_sample_G = g_sample.detach().clone()

            #print(d_out.shape)
            #print(mask.shape)
            #out = out.float().to(device)
            y_gen = y_gen.float().to(device)
            #print("out", out.shape)
            #print("ygen", y_gen.shape)
            #print("X_gen", x_gen.shape)
            #print("gsample", g_sample.shape)
            #print("d_out", d_out.shape)
            #print("mask.shape", mask.shape)
            #optimizerD.zero_grad()
            #loss_D = dis_loss(mask, d_out)
            #print("lossd",loss_D)
            loss1 = loss_func(out, y_gen)
            #print("loss1",loss1)
            #loss2 = gen_loss(mask_G, d_out_G, x_gen, raw_G, g_sample)
            #print("loss2", loss2)
            loss_G = loss1# + loss2
            #loss_D.backward(retain_graph=True)

            #optimizerD.step()
            #i += 1
            """viz.line([ np.array(loss_D.cpu().detach().numpy())], [i],
                     win='lossD', update='append')
            viz.line([np.array(loss_G.cpu().detach().numpy())], [i],
                     win='lossG', update='append')
                     """
            optimizerG.zero_grad()
            loss1 = loss_func(out, y_gen)

            #loss2 = gen_loss(mask_G,d_out_G,x_gen,raw_G,g_sample)
            loss_G = loss1#+loss2
            #loss_D = dis_loss(mask, d_out)
            #print("lossg",loss_G)
            """"
            graph_d = make_dot(loss_D)
            graph_d.view('model_structured.pdf', '.\\figure\\')
            graph_g = make_dot(loss_G)
            graph_g.view('model_structureg.pdf', '.\\figure\\')
            """""

            #loss = loss1

            loss_G.backward()
            optimizerG.step()
            index_i = 10000
            index_j = 10000
            """
            for i in range(mask.shape[0]):

                for j in range(mask.shape[2]):
                    if mask[i,0,j,0] == 1:
                        index_i = i
                        index_j = j
                        break
                if index_j !=10000:
                    break
            """


            """
            viz.line([np.array(loss_G.cpu().detach().numpy())], [i], win='lossG', update='append')
            viz.line([np.array(loss_D.cpu().detach().numpy())], [i], win='lossD', update='append')
            """
            #viz.line([[np.array(raw[index_i,index_j,:].cpu().detach().numpy()), np.array(g_sample[index_i,index_j,:].cpu().detach().numpy())]], [epoch],
            #         win='origal', update='replace')
            #x_linespace =  np.linspace(0,raw.shape[2],num=raw.shape[2])
            #plt.plot(x_linespace,raw[index_i,index_j,:].cpu().detach().numpy(),label='raw')
            #plt.plot(x_linespace,g_sample[index_i,index_j,:].cpu().detach().numpy(),label='generated')
            #viz.matplot(plt,win='origal')
            #plt.close()
            time.sleep(0.5)

            
            train_loss_list.append(loss1.item())
            acu_loss += loss1.item()
                
            i += 1


        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader)
            #if i_epoch != 0:

            #val_result1 = test(model, val_dataloader)
                #_,test_result = test(model, test_dataloader)
            #print(test_result)
                #info = get_score(test_result, val_result)
                #f1_list.append(info[0])

                #auc_list.append(info[3])
                #recall_list.append(info[2])
                #pre_list.append(info[1])

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss
    f1_list = np.array(f1_list)
    auc_list = np.array(auc_list)
    recall_list = np.array(recall_list)
    pre_list  = np.array(pre_list)
    #print(f1_list.shape)
    #x = np.linspace(0,epoch,num=epoch)
   # plt.plot(x,f1_list)
    #plt.plot(x, auc_list)
    #plt.plot(x, recall_list)
    #plt.plot(x, pre_list)
    #plt.show()


        # print(self.test_result)


    return train_loss_list

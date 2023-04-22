import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F
from pyinform.transferentropy import transfer_entropy

import seaborn as sns
from .graph_layer import GraphLayer



def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index

class Discrimanitor(nn.Module):
    def __init__(self):
        super(Discrimanitor, self).__init__()
        """
        self.lin1 = nn.Linear(1000,500)
        self.bn1 = nn.BatchNorm1d(500)
        self.lin2 = nn.Linear(500,100)
        self.bn2 = nn.BatchNorm1d(100)
        self.lin3 = nn.Linear(100,10)
        self.bn3 = nn.BatchNorm1d(10)
        self.lin4 = nn.Linear(10,1)
        self.sig = nn.Sigmoid()
        """
        self.conv1 = nn.Conv1d(in_channels=1,  # batch , 4,402
                               out_channels=4,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn1 = nn.BatchNorm1d(num_features=4, )  # batch , 4 ,402

        self.conv2 = nn.Conv1d(in_channels=4,  # batch , 8 ,101
                               out_channels=8,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn2 = nn.BatchNorm1d(num_features=8, )

        self.conv3 = nn.Conv1d(in_channels=8,  # batch, 16, 50
                               out_channels=4,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn3 = nn.BatchNorm1d(num_features=4, )
        self.conv4 = nn.Conv1d(in_channels=4,  # batch, 16, 50
                               out_channels=1,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               )
        self.fc = nn.Linear(123,1)

        self.bn4 = nn.BatchNorm1d(num_features=1, )
        self.sig = nn.Sigmoid()


    def forward(self,x,x_gen):
        batch_num, node_num, all_feature = x.shape
        #print("xshape",x.shape)
        #print("xgenshape",x_gen.shape)
        x_gen=torch.reshape(x_gen,(batch_num*node_num,1,-1))
        #print("x",x_gen.shape)
        """
        D_out1 = self.lin1(x_gen)
        D_out1 = self.bn1(D_out1)
        D_out2 = self.lin2(D_out1)
        D_out2 = self.bn2(D_out2)
        D_out3 = self.lin3(D_out2)
        D_out3 = self.bn3(D_out3)
        D_out4 = self.lin4(D_out3)
        D_out5 = self.sig(D_out4)
        """
        D_out = self.conv1(x_gen)
        D_out = self.bn1(D_out)
        D_out = self.conv2(D_out)
        D_out = self.bn2(D_out)
        D_out = self.conv3(D_out)
        D_out = self.bn3(D_out)
        D_out = self.conv4(D_out)
        D_out = self.fc(D_out)
        D_out = self.sig(D_out)
        D_out = torch.reshape(D_out,(batch_num,node_num,-1))

        return D_out



class dcnn(nn.Module):
    def __init__(self):
        super(dcnn, self).__init__()
        self.con1 = nn.Conv1d(in_channels=18,
                               out_channels=18,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )
        self.bn1 = nn.BatchNorm1d(num_features=18, )
        self.conv2 = nn.Conv1d(in_channels=18,  # batch , 8 ,101
                               out_channels=18,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               )

        self.bn2 = nn.BatchNorm1d(num_features=18, )
        self.line = nn.Linear(213,250)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward (self,x):
        x = self.con1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #print(x.shape)
        x = self.line(x)
        x = self.relu(x)
        return x

class attention_layer(nn.Module):
    def __init__(self,hidden_dim):
        super(attention_layer, self).__init__()
        self.q = nn.Linear(hidden_dim,hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 125)
    def forward(self,x):
        Q = self.q(x)
        K = self.k(x).permute(0, 2, 1)
        V = self.v(x)
        alpha = torch.matmul(Q,K)
        alpha = F.softmax(alpha, dim=2)
        out = torch.matmul(alpha, V)
        out = out.reshape(-1,18,425)
        out = self.lin2(out)
        #print("out",out.shape)

        return out,alpha




class Generator(nn.Module):
    def __init__(self, singal_num,  node_num):
        super(Generator, self).__init__()  # input 18,250
        self.input_channel = singal_num
        #self.mask = mask


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.embeddings = embedding(torch.arange(node_num))
        self.deconv1 = nn.ConvTranspose1d(in_channels=self.input_channel,
                                          out_channels=self.input_channel * 4,  # 3
                                          kernel_size=2,
                                          stride=1,
                                          padding=0,
                                          # output_padding=1,
                                          )

        self.bn1 = nn.BatchNorm1d(num_features=self.input_channel * 4, )

        self.deconv2 = nn.ConvTranspose1d(in_channels=self.input_channel * 4,
                                          out_channels=self.input_channel * 8,  # 5
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          output_padding=0,
                                          )

        self.bn2 = nn.BatchNorm1d(num_features=self.input_channel * 8, )

        self.deconv3 = nn.ConvTranspose1d(in_channels=self.input_channel * 8,
                                          out_channels=self.input_channel,  # 9
                                          kernel_size=2,
                                          stride=2,
                                          padding=1,
                                          output_padding=0,
                                          )
        # self.bn3 = nn.BatchNorm1d(num_features=self.conv_channel_size, )
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.l1 = nn.Linear(1000,1000)
    def forward(self, x,mask,embeddings):
        batch_num, node_num, all_feature = x.shape
        all_embeddings = embeddings
        all_embeddings = all_embeddings.reshape(1,all_embeddings.shape[0],all_embeddings.shape[1])

        #print(all_embeddings.shape)
        #print(batch_num)
        all_embeddings = all_embeddings.repeat(batch_num, 1,1).to(self.device)
        noise = torch.randn_like(all_embeddings)
        all_embeddings = all_embeddings + noise
        #print(all_embeddings.shape)
        h = self.deconv1(all_embeddings)
        h = self.bn1(h)
        h = self.relu(h)
        h= self.deconv2(h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.deconv3(h)
        #h = self.relu(h)
        h= self.l1(h)
        h = self.tanh(h)

        #print("hshape", h.shape)
        mask1=torch.squeeze(mask,1)
        #print("maskshape",mask.shape)
        mask_b = torch.ones_like(mask1.cpu())
        #mask_b = torch.from_numpy(mask_b)
        #print(torch.mul(mask,h).shape)
        #print(torch.mul(mask_b.sub(mask.cpu()),h.cpu()).shape)
        x_gen = torch.mul(mask1,x).add(torch.mul(mask_b.sub(mask1.cpu()).cuda(),h))
        x_ = x_gen[:,:,:850]
        y_ = x_gen[:,:,850:]
        """
        print((mask*h).shape)
        print(((torch.from_numpy(np.ones_like(mask) - mask))*h).shape)
        x_gen = mask*h + (torch.from_numpy(np.ones_like(mask) - mask))*h
        """
        return x_,x_gen,y_,h


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512, out_num=100):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append((nn.Linear(in_num if layer_num == 1 else inter_num, out_num)))
                modules.append(nn.Tanh())
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        # print(out.shape)

        for mod in self.mlp:
            #print(mod)
            if isinstance(mod, nn.BatchNorm1d):
                #print(out.shape)
                out1 = out.permute(0, 2, 1)
                #print(out1.shape)
                out2 = mod(out1)
                #print(out2.shape)
                out = out2.permute(0, 2, 1)
                #print(out.shape)
            else:
                #print(out.shape)
                out = mod(out)
        # print(out.shape)

        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        #print()
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        #print('att',self.att_weight_1)
        #print('edge',self.edge_index_1)
        out = self.bn(out)

        return self.relu(out) ,self.att_weight_1 ,self.edge_index_1
class TemporalAttentionLayer(nn.Module):


    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout

        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2


        lin_input_dim = n_features
        a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        #print("a",self.a.data)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu =nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.lin2 = nn.Linear(window_size,125)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        #print(x.shape)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu

        # Original GAT attention
        #print("x",x.shape)
        Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
        #print(Wx)
        a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
        #print(a_input)
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        #print(e.shape)
        #attention = e
        #print(e)
        attention = torch.softmax(e, dim=2)
        #print("att",attention)
        #print(attention.shape)
        """for i in range(attention.shape[0]):
            for j in range(attention.shape[1]):
                for k in range(attention.shape[2]):
                    if attention[i,j,k]<0.01:
                        attention[i,j,k]=0
        """
        #print(attention.shape)
        #attention = (self.relu(attention - 0.001) * attention) \
        #             / (torch.abs(attention - 0.001) + 0.0000001)
        #print(attention.shape)
        #attention = torch.softmax(attention, dim=2)
        #attention = attention / attention.norm(p=1, dim=0)
        #print(attention.shape)


        #sns.heatmap(attention.cpu().detach().numpy()[0])
        #plt.show()
        #attention = torch.dropout(attention, self.dropout, train=self.training)
        #print(attention.shape)
        #sns.heatmap(attention.cpu()[0])

        h = torch.matmul(attention, x) + x  # (b, n, k)
        #print(h.shape)
        h = h.permute(0, 2, 1)
        #print(h.shape)

        h = self.lin2(h)
        h = self.relu(h)
        return h,attention

    def _make_attention_input(self, v):

        K = self.num_nodes
        # print(K)
        # print("v",v.shape)
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        # print(blocks_repeating.shape)
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix

        # print(blocks_alternating.shape)
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        return combined.view(v.size(0), K, K, 2 * self.embed_dim)

class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1,
                 topk=20, predict_num=100):

        super(GDN, self).__init__()
        self.predict_num = predict_num

        self.edge_index_sets = edge_index_sets

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)

        self.gnn_layers = nn.ModuleList([
            GNNLayer(250, dim, inter_dim=dim + embed_dim, heads=1) for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim * edge_set_num, node_num * 10, out_layer_num, inter_num=out_layer_inter_dim,
                                  out_num=predict_num)
        self.t_gats = TemporalAttentionLayer(18, 425, 0.2, 0.1, embed_dim=None, use_bias=False)
        self.att = attention_layer(425)
        #self.line = nn.Linear(850,250)
        #self.cnn = dcnn()
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()

        self.generator = Generator(node_num, node_num).to(device)
        #self.discriminator = Discrimanitor()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, raw_data, org_edge_index,mask):
        x = data.clone().detach()
        batch_num, node_num, all_feature = x.shape
        device = data.device
        all_embedding = self.embedding(torch.arange(node_num).to(device))
        generator,x_,y_gen,g_sample = self.generator(raw_data,mask,all_embedding)
        #print(x_.shape)
        #d_out = self.discriminator(raw_data,x)
        #x_ = raw_data[:,:,:850]
        #y_gen = raw_data[:,:,850:]

        edge_index_sets = self.edge_index_sets
        # print("edge_index_set",edge_index_sets)




        #x = generator.view(-1, all_feature).contiguous()
        x__  = generator.reshape(batch_num,all_feature,node_num)
        b = torch.linspace(0, 848, 425).long()
        c = torch.linspace(1, 849, 425).long()
        x_ji =x__[:,b,:].reshape(batch_num,18,-1)
        x_ou = x__[:,c,:].reshape(batch_num,18,-1)

        #x=self.line(x_)
        #x = x.reshape(-1,250)
        #print("x__",x__)
        #print(x__.shape)
        """
        for i in range(x__.shape[2]):
            x = x__[:,:,i]
            t_a = self.t_gats[i](x.unsqueeze(2))
            if i==0:
                t_ini = t_a;
            else:
                t_ini=torch.cat([t_ini,t_a],1)
        """
        """
        t_a,attentionji = self.t_gats(x_ji)
        t_a2,attentionou = self.t_gats(x_ou)
        """

        t_a, attentionji = self.att(x_ji)
        t_a2, attentionou = self.att(x_ou)





        #print(t_a.shape)
        #print("newx",t_ini.shape)

        tout = torch.cat((t_a,t_a2),dim=2)
        #x = tout.reshape(-1, 250)
        x = tout.reshape(-1, 250)
        #x = self.cnn(x)
        #x = x.reshape(-1, 250)
        #print("x",x)

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

            batch_edge_index = self.cache_edge_index_sets[i]

            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)
            #print(weights[0])
            #print(weights[1])

            cos_ji_mat = torch.matmul(weights, weights.T)
            """print(cos_ji_mat)
            for i in range(node_num):
                for j in range(node_num):
                    print(transfer_entropy(weights[i].cpu(), weights[j].cpu(), k=2))
                    cos_ji_mat[i][j] = transfer_entropy(weights[i].cpu(), weights[j].cpu(), k=2)
                    #print(cos_ji_mat)
            """
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
            cos_ji_mat = cos_ji_mat / normed_mat
            #print(cos_ji_mat)

            dim = weights.shape[-1]
            topk_num = self.topk


            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
            #print(topk_indices_ji)
            """topk_indices_ji = torch.tensor(
			[[0,1,2,3,4,5,6,7,8,17],
             [2,0,1,3,4,5,6,7,8,17],
             [0,1,2,3,4,5,6,7,8,17],
             [4,3,5,16,13,0,1,2,12,6],
             [4,3,5,16,13,0,1,2,12,6],
             [4,3,5,16,13,0,1,2,12,6],
             [6,7,8,12,11,0,1,2,10,9],
             [6,7,8,12,11,0,1,2,10,9],
             [6,7,8,12,11,0,1,2,10,9],
             [9,15,17,6,7,8,3,4,5,12],
             [10,11,6,7,8,12,13,0,1,2],
             [10,11,6,7,8,12,13,0,1,2],
             [10,11,6,7,8,12,13,0,1,2],
             [13,16,3,4,5,12,10,11,14,15],
             [14,15,9,17,0,1,2,10,11,13],
             [15,14,9,17,0,1,2,12,13,16],
             [13,16,3,4,5,12,10,11,14,15],
             [17,0,1,2,15,9,14,10,11,12]],
			dtype=None,
			device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
			requires_grad=False,
			pin_memory=False)
			"""

            self.learned_graph = topk_indices_ji

            #print('leanrned_graph',self.learned_graph)

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)

            #print(batch_gated_edge_index)
            gcn_out,att,edg=self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num,embedding=all_embeddings)
            """for i in range(18):
                print("the",i+1,"node")
                print(att[i*9:(i+1)*9,:,:])
                print(edg[0,i*9:(i+1)*9])
                """


            #print(edg[1,:])
            #print(alpha.shape)
            gcn_outs.append(gcn_out)


        x = torch.cat(gcn_outs, dim=1)
        #print(x.shape)
        x = x.view(batch_num, node_num, -1)
        #print(t_a.shape)
        #x = torch.cat([x,t_a],dim=2)
        x = x+tout
        #print(x.shape)


        indexes = torch.arange(0, node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))

        out1 = out.permute(0, 2, 1)
        out2 = F.relu(self.bn_outlayer_in(out1))
        out3 = out2.permute(0, 2, 1)

        out4 = self.dp(out3)
        #print("out4",out4.shape)
        out5 = self.out_layer(out4)
        # print(out.shape)
        out6 = out5.view(-1, node_num, self.predict_num)
        #print(out6)



        return out6,y_gen,x_#,attentionji,attentionou

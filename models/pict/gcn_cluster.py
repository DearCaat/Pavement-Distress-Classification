###################################################################
# File Name: gcn.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri 07 Sep 2018 01:16:31 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from os import X_OK

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import networkx as nx
import torch.multiprocessing as mp
from .similarity import similarity_matrix

def gcn_cluster(edge_col_indices,scores,feat,thr=0.5,is_cuda=True):
    B,N,K1 = edge_col_indices.shape
    crow_indices = torch.tensor([K1*i for i in range(N+1)]).contiguous()
    if is_cuda:
        crow_indices = crow_indices.cuda(non_blocking=True)
    
    # _mask = scores[:,:,:,1]>thr
    # _mask_invert = _mask==False
    # scores[_mask] = 1
    # scores[_mask_invert] = 1
    # B,N,_,_ = scores.shape
    cluster_feat = [[] for i in range (B)]
    cluster_idcs = [[] for i in range (B)]
    for b in range(B):
        csr = torch.sparse_csr_tensor(crow_indices,edge_col_indices[b,:,:].flatten().contiguous(),scores[b,:,:,1].flatten().contiguous(),size=(N,N),requires_grad=False)
        A = csr.to_dense()
        A[A>thr] = 1
        A[A<=thr] = 0

        #if scores shape == [B,N,N,2]
        #A_nx = nx.from_numpy_matrix(scores[b].detach().cpu().numpy())
        #print(A)
        A_nx = nx.from_numpy_matrix(A.cpu().numpy())
        for c in nx.connected_components(A_nx):
            c = list(c)
            cluster_feat[b].append(feat[b][c])
            cluster_idcs[b].append(c)
    #del crow_indices
    return cluster_feat,cluster_idcs

def normalize_adj(A, type="AD"):
    if type == "DAD":
        # d is  Degree of nodes A=A+I
        # L = D^-1/2 A D^-1/2
        A = A + np.eye(A.shape[0])  # A=A+I
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)
        G = torch.from_numpy(G)
    elif type == "AD":
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)
    else:
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        D = np.diag(D)
        G = D - A
    return G

def reshape_A(edges,scores,N):
    score_dict = torch.ones(N,N)
    for i,e in enumerate(edges):
        score_dict[e[0], e[1]] = scores[i]
    # a = torch.from_numpy(score_dict)
    # a = a.reshape(N,-1)
    # b = torch.ones(N, N)
    # b[:-1,1:] += torch.triu(a[:-1])
    # b[1:,:-1] += torch.tril(a[1:])
    return score_dict

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()
    def forward(self, features, A ):
        #x = torch.bmm(A, features)
        x = torch.einsum('bnii,bnid->bnid',(A, features))
        return x 

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(in_dim *2, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, i, d = features.shape
        assert(d==self.in_dim)
        out = self.agg(features,A)
        out = torch.cat([features, out], dim=3)
        out = torch.einsum('bnid,df->bnif', (out, self.weight))
        out = F.relu(out + self.bias)
        return out 
        

class GCN(nn.Module):
    def __init__(self,in_dim=512,out_dim=256,k1=20):
        super(GCN, self).__init__()
        self.k1 = k1
        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)
        self.conv1 = GraphConv(in_dim, in_dim, MeanAggregator)
        self.conv2 = GraphConv(in_dim, in_dim, MeanAggregator)
        self.conv3 = GraphConv(in_dim, out_dim, MeanAggregator)
        self.conv4 = GraphConv(out_dim, out_dim,MeanAggregator)
        
        self.classifier = nn.Sequential(
                            nn.Linear(out_dim, out_dim),
                            nn.PReLU(out_dim),
                            nn.Linear(out_dim, 2))

        initialize_weights(self.classifier)

    def forward(self, x, A, one_hop_mask, train=True):
        # data normalization l2 -> bn
        B,N,I,D = x.shape    # batch_size instance_size nodes_IPS dim
        # 原论文自己注释掉了L2 norm
        #xnorm = x.norm(2,2,keepdim=True) + 1e-8
        #xnorm = xnorm.expand_as(x)
        #x = x.div(xnorm)
        
        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B,N,I,D)
        # GCN是通过邻接矩阵表示的特征之间的关系来增强特征，这里最后通过增强后特征做了边预测
        # 这里是以每个IPS为单位进行增强，输入是 B N k1,k2 D，输出是 B N k1 2，原论文这里只保留了中心点与k1个顶点之间边的预测信息 
        x = self.conv1(x,A)
        x = self.conv2(x,A)
        x = self.conv3(x,A)
        x = self.conv4(x,A)

        dim_out = x.size(-1)
        x = x[one_hop_mask].view(B,N,self.k1,dim_out)
        x = x.view(-1,dim_out)
        x = self.classifier(x).view(B,N,self.k1,2)
            
        # shape: B N k1 2
        return x

class KnnGraph(object):
    def __init__(self,active_connection=4,k_at_hop=[20,5],distance='cosine'):
        self.active_connection = active_connection
        self.k_at_hop = k_at_hop
        self.distance = distance
    def get_KNN(self,feats,distance='cosine'):
        if distance == 'cosine':
            _similarity_matrix=similarity_matrix(feats,feats,'cosine',True)  # B*N*N
            knn_graph = torch.argsort(_similarity_matrix, axis=2,descending=True)  # B*N*N
        elif distance == 'euclidean':
            _similarity_matrix=similarity_matrix(feats,feats,'euclidean')
            knn_graph = torch.argsort(_similarity_matrix, axis=2,descending=False)  # B*N*N
        return knn_graph
    def get_KNN_adj(self,knn=None,feat=None):
        if knn==None and feat is not None:
            knn = self.get_KNN(feat)
        adj = knn.clone()         #B N N
        adj[:,:,:] = 0
        adj = adj.scatter_(2,knn[:,:,1:self.k_at_hop[0]+1],1)
        return adj

    
    #for multi-process
    def change_hops(self,B,N,hops_1,hops_2,knn_graph,mask_one_hop_idcs,A_,feat,feats):
        for i in B:
            for m in range(N):
                hops_2[i,m,:,:] = hops_1[i,hops_1[i,m,:],:self.k_at_hop[1]+1]
                #展平hops矩阵后两维，这里就不考虑自身的那一维
                uni_tmp = hops_2[i,m,1:,:].flatten(start_dim=-2, end_dim=-1)
                uni_tmp = torch.unique(uni_tmp)
                if m not in uni_tmp:
                    uni_tmp = torch.cat((torch.tensor([m]).cuda(non_blocking=True),uni_tmp))
                num_nodes = len(uni_tmp)
                a_tmp = torch.zeros(num_nodes,num_nodes) # 小维度的矩阵cpu上进行索引和改值更快一点
                neighbors = knn_graph[i,uni_tmp,1:self.active_connection+1]
                # 每个结点能够连接的顶点数不同，需要使用for循环
                for node in range(num_nodes):
                    nei_index = torch.isin(uni_tmp,neighbors[node,torch.isin(neighbors[node],uni_tmp)])
                    a_tmp[node,nei_index] = 1
                    a_tmp[nei_index,node] = 1
                # one-hop indices
                mask_one_hop_idcs[i,m,:num_nodes] = torch.isin(uni_tmp,hops_2[i,m,1:,0])
                A_[i,m,:num_nodes,:num_nodes] = a_tmp      
                feat[i,m,:num_nodes] = feats[i,uni_tmp]
    def change_A(self,B,N,uni,knn_graph,mask_one_hop_idcs,hops_2,A_,feat,feats):
        for i in B:
            for m in range(N):
                uni_tmp = torch.unique(uni[i,m])
                if m not in uni_tmp:
                    uni_tmp = torch.cat((torch.tensor([m]).cuda(non_blocking=True),uni_tmp))
                num_nodes = len(uni_tmp)
                a_tmp = torch.zeros(num_nodes,num_nodes) # 小维度的矩阵cpu上进行索引和改值更快一点
                neighbors = knn_graph[i,uni_tmp,1:self.active_connection+1]
                # 每个结点能够连接的顶点数不同，需要使用for循环
                for node in range(num_nodes):
                    nei_index = torch.isin(uni_tmp,neighbors[node,torch.isin(neighbors[node],uni_tmp)])
                    a_tmp[node,nei_index] = 1
                    a_tmp[nei_index,node] = 1
                # one-hop indices
                mask_one_hop_idcs[i,m,:num_nodes] = torch.isin(uni_tmp,hops_2[i,m,1:,0])
                A_[i,m,:num_nodes,:num_nodes] = a_tmp      
                feat[i,m,:num_nodes] = feats[i,uni_tmp]
    def __call__(self, feats,is_cuda=True):
        B,N,D=feats.shape
        # N和最大节点数谁小用谁，原论文这里N远大于最大节点数, max_nodes这个东西在原论文中为了避免N过大，导致矩阵过于稀疏采取的一种策略，这里Batch的实现没办法使用这个参数，在N小的时候也没必要，N过大可以都使用稀疏矩阵
        #max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1

        # ## 1. get the KNN graph
        knn_graph = self.get_KNN(feats,self.distance)
        #knn_graph = knn_graph[:,:, :self.k_at_hop[0] + 1]

        # 添加第一跳
        hops_1 = knn_graph[:,:,:self.k_at_hop[0] + 1]      # B N [1 K1] 这里的1是center point 的索引
        # 添加第二跳
        if len(self.k_at_hop) > 1:
            hops_2 = hops_1[0,hops_1[:,:]][:,:,:,:self.k_at_hop[1]+1].clone()  #B N [1 K1] [1 K2]
            hops = hops_2
            hops = hops.flatten(-2,-1)
        else:
            hops = hops_1
        

        # 构建唯一顶点矩阵 构建邻接矩阵 因为每一个lps的顶点数目都不相同，因此需要使用for 循环
        #uni_array = np.empty((B,N),dtype=object)
        # max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        # feat = feats.unsqueeze(-2).repeat(1,1,max_num_nodes,1)
        # feat[:,:,:,:] = 0
        # A_ = feat.clone()[:,:,:,:max_num_nodes]
        # mask_one_hop_idcs = torch.zeros(B,N,max_num_nodes) == 1

        # 构建邻接矩阵和特征矩阵，纯矩阵实现 202220228修改，之前版本可以看 notebook
        # 根据跳数构建候选顶点的mask
        mask_candidate = torch.zeros(B,N,N)
        if is_cuda:
            mask_candidate = mask_candidate.cuda(non_blocking=True)
        mask_candidate = mask_candidate.scatter_(2,hops,1) == True

        # 根据mask构建邻接矩阵
        A_ = mask_candidate.clone()
        A_[:,:,:] = 0
        A_ = A_.scatter_(2,knn_graph[:,:,1:self.active_connection+1],1) == True #每个顶点的激活集合
        A_ = A_.unsqueeze_(1).repeat(1,N,1,1)  #升维，对每个顶点来说，都要构建IPS图
        # 该mask用于判断在激活点集合中的顶点是否在所属IPS中
        _mask = mask_candidate.unsqueeze(1).repeat(1,N,1,1)
        # 该mask用于将不是IPS内的顶点全部屏蔽掉，如果是IPS内的顶点，& True后，值不变，但非IPS内的顶点， & False后，值永远为False
        mask_2 = _mask.clone()
        mask_2[:,:,:,:] = 0 ==1
        mask_2[mask_candidate] = True
        # 这里是每个IPS应该和主顶点建边的顶点信息，这样的实现不会自身与自身建边，因为activa_group那里，排除了自己
        A_ = (A_ & _mask & mask_2).float()
        # 因为是无向图，解决对称问题
        i, j = torch.triu_indices(N,N)
        A_.transpose(-2,-1)[:,:,i,j] += A_[:,:,i,j]
        A_[:,:,i,j] += A_.transpose(-2,-1)[:,:,i,j]
        A_[A_!=0]=1
        
        # 根据mask构建特征矩阵
        # 先用cpu处理，再用稀疏矩阵保存在gpu中，这种高维张量在cpu上非常耗时间，对比gpu上，会提升一倍的运行时间，而且如果不用稀疏张量的话，最大显存占用是相同的
        # feat_ = feats.cpu().clone().unsqueeze_(1).repeat(1,N,1,1)
        # feat = feat_.clone()
        # feat[:,:,:,:] = 0
        # feat[mask_candidate.cpu()] = feat_[mask_candidate.cpu()]
        # feat = feat.to_sparse()   # B max_nodes max_nodes D
        # if is_cuda:
        #     feat = feat.cuda(non_blocking=True)

        feat_ = feats.clone().unsqueeze_(1).repeat(1,N,1,1)
        feat = feat_.clone()
        feat[:,:,:,:] = 0
        feat[mask_candidate] = feat_[mask_candidate] # B max_nodes max_nodes D

        mask_one_hop_idcs = mask_candidate.clone()
        mask_one_hop_idcs[:] = 0
        mask_one_hop_idcs = mask_one_hop_idcs.scatter_(2,knn_graph[:,:,1:self.k_at_hop[0] + 1],1) == True

        # 效率问题不考虑第二跳
        # 添加第二跳
        # for i in range(B):
        #     for m in range(N):
        #         hops_2[i,m,:,:] = hops_1[i,hops_1[i,m,:],:self.k_at_hop[1]+1] 

        # hops_1 = hops_1.share_memory_()
        # hops_2 = hops_2.share_memory_()
        # knn_graph = knn_graph.share_memory_()
        # A_ = A_.share_memory_()
        # feat = feat.share_memory_()
        # feats = feats.share_memory_()

        # num_worker = 2
        # processes = []
        # for rank in range(num_worker):
        #     p = mp.Process(target=self.change_hops,args=(range(rank*int(B/num_worker),(rank+1)*int(B/num_worker)),N,hops_1,hops_2,knn_graph,mask_one_hop_idcs,A_,feat,feats))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
        #del hops_1
        # hops_2矩阵中，dim -1 第一个元素存储的是K1跳的顶点，后几个元素为K2跳个顶点，dim -2 的第一个元素是中心点的索引
        # 展平hops矩阵后两维，这里就不考虑自身的那一维
        #uni = hops_2[:,:,1:,:].flatten(start_dim=-2, end_dim=-1)
        #uni = hops[:,:,1:]



        # for i in range(B):
        #     for m in range(N):
        #         uni_tmp = torch.unique(uni[i,m])
        #         if m not in uni_tmp:
        #             uni_tmp = torch.cat((torch.tensor([m]).cuda(non_blocking=True),uni_tmp))
        #         num_nodes = len(uni_tmp)
        #         a_tmp = torch.zeros(num_nodes,num_nodes) # 小维度的矩阵cpu上进行索引和改值更快一点
        #         neighbors = knn_graph[i,uni_tmp,1:self.active_connection+1]
        #         # 每个结点能够连接的顶点数不同，需要使用for循环
        #         for node in range(num_nodes):
        #             nei_index = torch.isin(uni_tmp,neighbors[node,torch.isin(neighbors[node],uni_tmp)])
        #             a_tmp[node,nei_index] = 1
        #             a_tmp[nei_index,node] = 1
        #         # one-hop indices
        #         mask_one_hop_idcs[i,m,:num_nodes] = torch.isin(uni_tmp,hops_2[i,m,1:,0])
        #         A_[i,m,:num_nodes,:num_nodes] = a_tmp      
        #         feat[i,m,:num_nodes] = feats[i,uni_tmp]
        # 正则化邻接矩阵？
        D = A_.sum(-1,keepdim=True)
        A_.div_(D)
        A_[torch.isnan(A_)] = 0
        del D

        # 正则化特征，IPS特征减去中心点特征
        #feat = (-feats.unsqueeze(1).repeat(1,N,1,1)).add(feat)
        feat.sub_(feat_)
        feat[mask_candidate==False].fill_(0)
        #feat.sub_(feats.unsqueeze(-2))

        return feat,A_,mask_one_hop_idcs,hops_1[:,:,1:]
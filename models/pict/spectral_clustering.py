import torch
# from kmeans_pytorch import kmeans
from .similarity import *
import numpy as np
from sklearn.manifold import spectral_embedding

'''
pytorch 的谱聚类简单实现，旨在解决batch-based数据的并行处理问题，主流的实现或者API很少支持batch-based
没有支持多种特征值eigsovler，使用了torch.linear.eigh
在计算lapacian matrix时，基本和Sklearn一致，因为得到的RBF核得到的数据可能是稀疏矩阵的问题，所以常规矩阵的求解方法无效
'''
def spectral_embedding_torch(
    affinity='rbf',
    feats=None,
    n_components=3,
    norm_laplacian=True,
    drop_first=True,
    gamma=1,
    rbf_distance='euclidean'  # default euclidean, but i wanto try others.
):
    # 采用rbf核算相似度存在绝大多数距离值为0的情况无法进行下面的计算。具体原因未知（(N,768)维度情况下，考虑是否维度过大？），暂定采用# cosine距离
    if affinity == 'rbf':
        d = rbf_similarity(similarity_matrix(feats.double().detach(),feats.double().detach(),rbf_distance)**2,gamma)
        #d = d.float()
    elif affinity == 'cosine':
        d = similarity_matrix(feats,feats,'cosine',False)
    
    # 因为torch.sum的位数问题，才有将主对角线取0，再求度矩阵，再求L矩阵的形式
    # 前提是输入的矩阵为欧式距离正则化的邻接矩阵，主对角线全为1
    # 稀疏拉普拉斯
    # Get laplacian
    B_SIZE,D_SIZE,_ = d.size()
    E = torch.eye(D_SIZE).unsqueeze(0).repeat(B_SIZE,1,1).cuda(non_blocking=True)
    d = d - torch.diag_embed(torch.diagonal(d,dim1=-2,dim2=-1))
    #d = d - E
    D_ = d.sum(-1)
    D = torch.diag_embed(D_)
    L = D-d

    if norm_laplacian:
        dd = torch.pow(D_,0.5)  # sklearn实现出现该值，为了计算出特征矩阵后进行进一步normalized
        D_ = torch.sign(D_) * torch.pow(torch.abs(D_), -0.5)  #直接pow会得到nan
        D = torch.diag_embed(D_)
        L=D@L@D
        L = L - torch.diag_embed(torch.diagonal(L,dim1=-2,dim2=-1)) + E  #因为精度问题，有些对角线元素不为1
    
    # cal eig 
    #L = L.float()
    L *= -1
    _, V = torch.linalg.eigh(L) #已经排好序，从小到大
    #V = V[:,:,-n_components:]
    #V = V.transpose(1,2).flip(1)  #skleran [n_compoents,::-1]
    V = V.transpose(1,2)[:,:n_components]
    if norm_laplacian:
        V = V.div(dd.unsqueeze(1))

    # except RuntimeError:
    #     # When submatrices are exactly singular, an LU decomposition
    #     # in arpack fails. We fallback to lobpcg
    #     L *= -1
    #     L = L.double()
    #     V = 0
    # sklearn _deterministic_vector_sign_flip
    max_abs_rows = torch.argmax(torch.abs(V), dim=2)
    mask = V.clone()
    mask[:] = 0
    mask = mask.scatter_(2,max_abs_rows.unsqueeze(-1),1) == 1
    sign = torch.sign(V[mask].view(B_SIZE,n_components)).unsqueeze(-1)
    V *= sign
    return V.transpose(1,2)

def spectral_clustering(
    affinity='rbf',
    feats=None,
    n_clusters=3,
    n_components=None,
    n_init=10,
    assign_labels="kmeans",
    kmeans_distance='euclidean',
    rbf_distance='euclidean',
    gamma=1,
    cluster_centers=None,
    is_training=True
):
    n_components = n_clusters if n_components is None else n_components

    # We now obtain the real valued solution matrix to the
    # relaxed Ncut problem, solving the eigenvalue problem
    # L_sym x = lambda x  and recovering u = D^-1/2 x.
    # The first eigenvector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    # maps = spectral_embedding_torch(
    #     affinity,
    #     feats,
    #     n_components=n_components,
    #     norm_laplacian=True,
    #     drop_first=False,
    #     gamma=gamma,
    #     rbf_distance=rbf_distance
    # )
    if affinity == 'rbf':
        d = rbf_similarity(similarity_matrix(feats.double(),feats.double(),rbf_distance)**2,gamma)
        #d = d.float()
    elif affinity == 'cosine':
        d = similarity_matrix(feats,feats,'cosine',False)

    d = d.cpu().detach().numpy()
    for i in range(len(d)):
        map_ = torch.from_numpy(spectral_embedding(d[i],drop_first=False,n_components=n_components))
        map_.unsqueeze_(0)
        if i == 0:
            maps = map_.clone()
        else:
            maps = torch.cat((maps,map_))
            
    maps = maps.cuda(non_blocking=True)
    # Only support kmeans
    if assign_labels == "kmeans":
        if is_training:
            for b in range(maps.size(0)):
                labels, cluster_centers = kmeans(
                    maps[b], n_clusters, cluster_centers=cluster_centers, n_init=n_init, tqdm_flag=False,device=torch.cuda.current_device(),distance=kmeans_distance,iter_limit=2000 
                )
                labels = torch.unsqueeze(labels,dim=0)
                if b == 0:
                        clusters_idcs = labels.clone()
                else:
                    clusters_idcs = torch.cat((clusters_idcs,labels))
            return clusters_idcs,cluster_centers
        else:
            labels = kmeans_predict_bs(X=maps,device=torch.cuda.current_device(),cluster_centers = cluster_centers,distance= kmeans_distance)

            return labels

def kmeans_predict_bs(
    X,
    cluster_centers,
    distance='euclidean',
    device=torch.device('cpu'),
):
    """
    predict using cluster centers for batch data
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) (batch_size,cluster ids)
    """
    assert len(cluster_centers.size()) == 2
    if len(X.size()) == 2:
        X.unsqueeze_(0)
    if distance == 'euclidean':
        pairwise_distance_function = SimilarityMatrix('euclidean')
    elif distance == 'cosine':
        pairwise_distance_function = SimilarityMatrix('cosine',True)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)
    
    dis = pairwise_distance_function(X, cluster_centers.unsqueeze(0).repeat(X.size(0),1,1))
    choice_cluster = torch.argmin(dis, dim=2)

    return choice_cluster
import torch
from torch import Tensor

def similarity_matrix(data1,data2,distance,invert=False):
    '''
    invert  bool  Whether invert the result   1-result for cosine
    '''
    # B N D
    if len(data1.size()) == 2:
        data1.unsqueeze_(0)
    if len(data2.size()) == 2:
        data2.unsqueeze_(0) 
        
    if distance.lower() == 'euclidean':
        similarity = torch.cdist(data1,data2)
    elif distance.lower() == 'cosine':
        dis_func = torch.nn.CosineSimilarity(dim=-1)
        # 为了返回(B,N1,N2)的结果，这里torch的api没有统一不是很明白，只有自己替换了位置
        similarity = dis_func(data2.unsqueeze(1), data1.unsqueeze(2))
        
        if invert:
            similarity = 1 - similarity
    
    return similarity

# euclidean_distances
def rbf_similarity(d,gamma=1):
    return torch.exp(-gamma*(d))

class SimilarityMatrix(torch.nn.Module):
    r"""
    Computes the pairwise distance between vectors :math:`v_1`, :math:`v_2` using the p-norm:

    Args:
        p (real): the norm degree. Default: 2
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-6
        keepdim (bool, optional): Determines whether or not to keep the vector dimension.
            Default: False
    Shape:
        - Input1: :math:`(B, N1, D)` where `B = batch dimension` and `D = vector dimension`
        - Input2: :math:`(B, N2, D)` same shape as the Input1
        - Output: :math:`(B, N1, N2)` 
    """
    __constants__ = ['distance', 'invert']
    distance: str
    invert: bool

    def __init__(self, distance: str = '', invert: bool = False) -> None:
        super(SimilarityMatrix, self).__init__()
        self.distance = distance
        self.invert = invert

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return similarity_matrix(x1, x2, self.distance, self.invert)
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from itertools import chain
import numpy as np
from timm.models.layers import  trunc_normal_
from torch.nn import Parameter
import random

def patchify(imgs,patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size[0] if type(patch_size) in (list,tuple) else patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

def unpatchify(x,patch_size):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """

    p = patch_size[0] if type(patch_size) in (list,tuple) else patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

# concat the impl of simmim and mae
class MaskGenerator:
    def __init__(self,input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6,use_mae=False,mask_some_tokens=True):

        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.use_mae = use_mae
        self.mask_some_tokens = mask_some_tokens

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = mask_patch_size // model_patch_size
        self.mask_token_count = int(self.rand_size ** 2)
        self.mask_count = int(np.ceil(self.mask_token_count * mask_ratio))
        self.keep_count = self.mask_token_count - self.mask_count

    def __call__(self,x,mask_token=None,unmask=None):
        N, L, D = x.shape

        noise = torch.rand(N,self.mask_token_count,device=x.device)

        ids_shuffle = torch.argsort(noise,dim=1)
        ids_restore = torch.argsort(ids_shuffle,dim=1)
        # ids_keep = ids_shuffle[:,:self.keep_count]

        mask = torch.ones([N, self.mask_token_count],device=x.device)
        mask[:, :self.keep_count] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)

        mask = mask.view((-1,self.rand_size,self.rand_size))
        mask = mask.repeat_interleave(self.scale,1).repeat_interleave(self.scale,2).contiguous()
        # 'unmask' to keep the certain token always unmask
        # mask: 1 - unmask:1 = unmask:0 
        # mask: 1 - mask: 0 = mask:1
        # unmask: 0 - unmask:1 = unmask:-1
        # unmask: 0 - mask：0 = unmask:0
        if unmask is not None:
            # mask some tokens
            if self.mask_some_tokens:
                mask = mask + unmask
                mask[mask>1] = 1
            # unmask some tokens
            else:
                mask = mask - unmask
                mask[mask!=1] = 0
            # each sample maybe have the different masked token
            try:
                x[(mask == 0).flatten(1)].view(N,-1,D)
            except:
                mean_mask_num = int(mask.flatten(-2,-1).sum(-1).mean())
                rand_size = mask.size(1)
                mask = mask.flatten(-2,-1)
                unmask = unmask.flatten(-2,-1)

                for i in range(len(mask)):
                    _mask_num = mask[i].sum()
                    _gap = int(mean_mask_num-_mask_num)
                    if _gap == 0:
                        continue
                    elif _gap > 0:
                        if not self.mask_some_tokens and len(idx)>=_gap:
                            idx = torch.nonzero((mask[i]+unmask[i]) - 1).squeeze()
                        # 如果将unmask排除在外后，没办法mask这么多token，就只能将unmask里的token纳入其中
                        else:
                            idx = torch.nonzero((mask[i]) - 1).squeeze()
                        idx = idx[random.sample(range(len(idx)),_gap)]
                        idx_mask = torch.zeros(rand_size*rand_size,device=mask.device)
                        idx_mask = idx_mask.scatter_(0,idx,1) == 1
                        
                        mask[i,idx] = 1

                    elif _gap < 0:
                        if self.mask_some_tokens:
                            idx = torch.nonzero(mask[i] - unmask[i]).squeeze()
                        else:
                            idx = torch.nonzero(mask[i]).squeeze()
                        idx = idx[random.sample(range(len(idx)),-_gap)]
                        
                        idx_mask = torch.zeros(rand_size*rand_size,device=mask.device)
                        idx_mask = idx_mask.scatter_(0,idx,1) == 1
                        
                        mask[i,idx] = 0
                mask = mask.view((mask.size(0),rand_size,rand_size))

        if mask_token is None:
            x_masked = x[(mask == 0).flatten(1)].view(N,-1,D)
        else:
            mask_token = mask_token.expand(N, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)

            x_masked = x * (1 - w) + mask_token * w
        return x_masked,mask

def random_masking(x,mask_token,mask_patch_size,model_patch_size,mask_ratio,use_mae):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence

    original copyright/license mae, thanks to the authors
    """
    N, L, D = x.shape  # batch, length, dim
    print(x.size())
    # for block-random mask, impl from simmim
    rand_size = L ** .5
    scale = mask_patch_size // model_patch_size
    token_count = L
    mask_count = int(np.ceil(token_count * mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :mask_count]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :mask_count] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # the mae only fed the unmasked patch to encoder, but simmim use the all.
    if use_mae:
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    else:
        mask_token = mask_token.expand(N, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x_masked = x * (1 - w) + mask_token * w

    return x_masked, mask, ids_restore

# copy from timm https://github.com/rwightman/pytorch-image-models/blob/9e12530433f38c536e9a5fdfe5c9f455638a3e8a/timm/models/helpers.py#L690
# the timm==0.5.4 doesn't have this function
def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.
    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.
    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x

# copyright dino@facebook, See:
# https://github.com/facebookresearch/dino/blob/main/utils.py
class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone.forward_features(torch.cat(x[start_idx: end_idx]))
            _out = self._avg_pooling(_out)
            _out = _out.squeeze()
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class LinearProbWrapper(nn.Module):
    def __init__(self,backbone,config):
        super(LinearProbWrapper, self).__init__()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.linear_head = LinearClassifier(config.TEST.LINEAR_PROB.DIM,config.MODEL.NUM_CLASSES)

        self.avg = nn.AdaptiveAvgPool2d(1)
        self._init_weights(self.linear_head)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        bs = x.size(0)

        with torch.no_grad():
            feat = self.backbone.forward_features(x)
            feat = self.avg(feat)

        feat = feat.view(bs,-1)
        output = self.linear_head(feat)
        
        return output

# empty wrapper, only the backbone
class BackboneWrapper(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.backbone, 'no_weight_decay'):
            return {'backbone.' + i for i in self.backbone.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.backbone, 'no_weight_decay_keywords'):
            return {'backbone.' + i for i in self.backbone.no_weight_decay_keywords()}
        return {}

# compatible different model api
def get_num_features(model):
    num_features = 0
    # timm backbone
    
    if hasattr(model,'num_features'):
        num_features = model.num_features
    # elif hasattr(model,'feature_info'):
    #     num_features = model.feature_info.channels()[-1]
    else:
        raise NotImplementedError

    return num_features

# cope with the deepcopy issue in pytorch
# https://gist.github.com/rtqichen/b22a9c6bfc4f36e605a7b3ac1ab4122f
class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights=['weight']):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = Parameter(g.data)
            v = Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = WeightNorm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.module.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.module.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.size()) == 3:
            x = x.view((-1,x.size(-1)))
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
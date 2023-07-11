import torch
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler,Dataset
import torch.distributed as dist
import math
from .dataset import build_dataset

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        #return (self.indices[i] for i in torch.randperm(len(self.indices)))
        return (self.indices[i] for i in range(len(self.indices)))
    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def set_indices(self,indices):
        self.indices = indices

#为ioplin 多gpu训练准备，有别于Torch普通的DDP Sampler，该Sampler通过指定indicies进行取数据
T_co = TypeVar('T_co', covariant=True)
class SpecifiedIndicesDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,indices: Optional[list] = None) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        if indices is None:
            indices = list(range(len(dataset)))
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.indices = indices
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        '''
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        '''
        indices = self.indices
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def set_indicese(self, indices: list) -> None:
        self.indices = indices

def ioplin_dataloader(config,is_train):
    if is_train:
        dataset_train,dataset_val = build_dataset(config,'train_val')

        if config.DISTRIBUTED:
            sampler_train = SpecifiedIndicesDistributedSampler(dataset_train)
            sampler_test = SpecifiedIndicesDistributedSampler(dataset_val)
            loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config.DATA.BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,sampler=sampler_test,persistent_workers=True)
        else:
            sampler_train = SubsetRandomSampler(list(range(len(dataset_train))))
            loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config.DATA.VAL_BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,persistent_workers=True)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,sampler=sampler_train,drop_last=config.DATA.DROP_LAST,persistent_workers=True)

        return dataset_train, dataset_val, loader_train, loader_val
        
    else:
        dataset_test = build_dataset(config,'test')

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.DATA.VAL_BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,persistent_workers=True)
        return dataset_test,loader_test
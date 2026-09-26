import os
import random

import numpy as np
import torch

from data_provider.pv_multires_loader import Dataset_PV_MultiRes
from torch.utils.data import DataLoader

data_dict = {'pv_multires': Dataset_PV_MultiRes}


def _is_primary_process():
    return os.environ.get('RANK', '0') == '0'


def _seed_worker(worker_id):
    """Propagate the DataLoader-assigned seed to Python and NumPy."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _loader_rng(args, flag):
    """Use a stable, split-specific generator for shuffled loaders."""
    seed = int(getattr(args, 'seed', 0) or 0)
    if seed <= 0:
        return None
    offsets = {'train': 0, 'val': 100003, 'test': 200003,
               'VAL': 100003, 'TEST': 200003}
    generator = torch.Generator()
    generator.manual_seed(seed + offsets.get(flag, 300007))
    return generator


def _loader_kwargs(args, flag):
    kwargs = {
        'worker_init_fn': _seed_worker,
        'generator': _loader_rng(args, flag),
    }
    return kwargs


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    eval_batch_size = int(getattr(args, 'eval_batch_size', 0) or 0)
    batch_size = args.batch_size if flag == 'train' or eval_batch_size <= 0 else eval_batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        if _is_primary_process():
            print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            **_loader_kwargs(args, flag))
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len),
            **_loader_kwargs(args, flag)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        if _is_primary_process():
            print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            **_loader_kwargs(args, flag))
        return data_set, data_loader

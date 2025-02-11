from mpi4py import MPI
from torch.utils.data import DataLoader
from custom import CustomDataset
from torch.utils.data.distributed import DistributedSampler


def create_datasets(args):
    train_data = CustomDataset(
        root=args.data_dir,
        train=True
    )
    val_data = CustomDataset(
        root=args.data_dir,
        train=False
    )
    return train_data, val_data


def create_data_loaders_train(args):
    train_data, _ = create_datasets(args)

    train_sampler = DistributedSampler(train_data)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        # shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler,
        
    )
    return train_loader


def load_data(
    *,
    data_dir,
    args=None,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    loader = create_data_loaders_train(args)
    
    while True:
        yield from loader




import torch
import torch.nn.functional as F
from torchvision import transforms
from utils.multi_mnist_loader import MNIST
from utils.celeba_loader import CELEBA
import utils.multi_lenet as multi_lenet
from utils.multi_faces_resnet import (
    ResNet,
    FaceAttributeDecoder,
)
import utils.multi_faces_resnet as celeba_net


class ZippedDatasets(torch.utils.data.Dataset):
    """Easily zip multiple datasets"""

    def __init__(self, datasets=None):
        self.datasets = datasets

    def __len__(self):
        min_len = len(self.datasets[0])
        for dataset in self.datasets[1:]:
            min_len = min(min_len, len(dataset))

        return min_len

    def __getitem__(self, index):
        return tuple((dataset[index]) for dataset in self.datasets)

    def __repr__(self):
        fmt_str = "Zipped Dataset of:\n\n"
        for dataset in self.datasets:
            fmt_str += dataset.__repr__()
            fmt_str += "\n\n"
        return fmt_str


def global_transformer():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


def get_networks(cfg):
    if "mnist" in cfg.dataset.name:
        if cfg.model.tasks_split_layer == "dense":
            basic_net = multi_lenet.MultiLeNetR
            task_nets = []
            for k, t in enumerate(cfg.dataset.tasks):
                task_net = multi_lenet.MultiLeNetO
                task_nets.append(task_net)
        elif cfg.model.tasks_split_layer == "conv":
            basic_net = multi_lenet.MultiLeNetConvEnc
            task_nets = []
            for k, t in enumerate(cfg.dataset.tasks):
                task_nets.append(multi_lenet.MultiLeNetDenseHead)
        else:
            raise ValueError(f"Unknown split layer {cfg.model.tasks_split_layer}")

    elif "celeba" in cfg.dataset.name:
        # basic_net = ResNet(BasicBlock, [2, 2, 2, 2])
        basic_net = celeba_net.ResNet
        task_nets = []
        for k, t in enumerate(cfg.dataset.tasks):
            task_net = celeba_net.FaceAttributeDecoder
            task_nets.append(task_net)
    nets = [basic_net, task_nets]
    return nets


def get_dataset(cfg):
    if "mnist" in cfg.dataset.name:
        train_dst = MNIST(
            root=cfg.dataset.root_dir,
            train=True,
            download=True,
            transform=global_transformer(),
            multi=True,
            right_noise_p=cfg.dataset.right_noise_p,
            both_noise_min=cfg.dataset.both_noise_min,
        )

        if cfg.dataset.eval_noise_ratios:
            datasets = [
                MNIST(
                    root=cfg.dataset.root_dir,
                    train=False,
                    download=True,
                    transform=global_transformer(),
                    multi=True,
                    right_noise_p=noise_ratio,
                    both_noise_min=cfg.dataset.both_noise_min,
                )
                for noise_ratio in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            ]
            val_dst = ZippedDatasets(datasets)
        else:
            val_dst = MNIST(
                root=cfg.dataset.root_dir,
                train=False,
                download=True,
                transform=global_transformer(),
                multi=True,
                right_noise_p=cfg.dataset.right_noise_p,
                both_noise_min=cfg.dataset.both_noise_min,
            )

    if "celeba" in cfg.dataset.name:
        train_dst = CELEBA(
            root=cfg.dataset.root_dir,
            is_transform=True,
            split="train",
            img_size=(cfg.dataset.img_rows, cfg.dataset.img_cols),
            augmentations=None,
        )
        val_dst = CELEBA(
            root=cfg.dataset.root_dir,
            is_transform=True,
            split="val",
            img_size=(cfg.dataset.img_rows, cfg.dataset.img_cols),
            augmentations=None,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dst,
        batch_size=100,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        drop_last=True,
    )
    return train_loader, train_dst, val_loader, val_dst


def nll(pred, gt, val=False):
    if val:
        return F.nll_loss(pred, gt, size_average=False)
    else:
        return F.nll_loss(pred, gt)


def get_loss(cfg):
    loss_fn = {}
    if "mnist" in cfg.dataset.name:
        for t in cfg.dataset.tasks:
            loss_fn[t] = nll
    if "celeba" in cfg.dataset.name:
        for t in cfg.dataset.tasks:
            loss_fn[t] = nll
    return loss_fn

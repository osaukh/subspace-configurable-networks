import argparse
import pickle
import numpy as np

from torchvision import datasets, transforms
from models import *


def load_data(split, dataset_name, datadir, translation=False):
    ## https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])

    if dataset_name == 'FashionMNIST':
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    elif dataset_name == 'SVHN':
        normalize = transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    elif dataset_name == 'CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        tr_transform = transforms.Compose([transforms.Resize(32),
                                           transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    elif dataset_name == 'CIFAR100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
        tr_transform = transforms.Compose([transforms.Resize(32),
                                           transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    elif dataset_name == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tr_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])


    get_dataset = getattr(datasets, dataset_name)
    if dataset_name == 'SVHN':
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    else:
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)

    return dataset


def prepare_model(args, nchannels, nclasses, hin=1):
    save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}'

    if "mlp" == args.arch:
        model = MLP(n_layers=args.nlayers, n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    elif "mlpb" == args.arch:
        model = MLPB(n_layers=args.nlayers, n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    elif "hhnmlp" == args.arch:
        save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}_{args.dimensions}'
        model = HHN_MLP(hin=hin, dimensions=args.dimensions, n_layers=args.nlayers,
                        n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    elif "hhnmlpb" == args.arch:
        save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}_{args.dimensions}'
        model = HHN_MLPB(hin=hin, dimensions=args.dimensions, n_layers=args.nlayers,
                         n_units=args.width, n_channels=nchannels, n_classes=nclasses)

    elif "sconv" == args.arch:
        model = SConv(n_layers=args.nlayers, n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    elif "sconvb" == args.arch:
        model = SConvB(n_layers=args.nlayers, n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    elif "hhnsconv" == args.arch:
        save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}_{args.dimensions}'
        model = HHN_SConv(hin=hin, dimensions=args.dimensions, n_layers=args.nlayers,
                        n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    elif "hhnsconvb" == args.arch:
        save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}_{args.dimensions}'
        model = HHN_SConvB(hin=hin, dimensions=args.dimensions, n_layers=args.nlayers,
                         n_units=args.width, n_channels=nchannels, n_classes=nclasses)

    elif "hhnconvdense" == args.arch:
        save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}_{args.dimensions}'
        model = HHNConvDense(dimensions=args.dimensions, n_layers=args.nlayers,
                        n_units=args.width, n_channels=nchannels, n_classes=nclasses)
    elif "hhnshiftinv" == args.arch:
        save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}_{args.dimensions}'
        model = HHNShiftInv(dimensions=args.dimensions, n_layers=args.nlayers,
                        n_units=args.width, n_channels=nchannels, n_classes=nclasses)

    elif "resnet" == args.arch:
        model = ResNet18()

    elif "hhnresnet" == args.arch:
        save_dir = f'{args.arch}_{args.dataset}_{args.nlayers}_{args.width}_{args.dimensions}'
        model = HHN_ResNet18(D=args.dimensions, hin=hin)

    return model, save_dir


def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

import os
import timeit
import argparse
import numpy as np
import utils
import pickle
import random
from tqdm import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import wandb

os.environ["WANDB_API_KEY"] = ""

parser = argparse.ArgumentParser(description='HHN Project')
parser.add_argument('--dataset', default='CIFAR10', type=str, help='MNIST | FashionMNIST | CIFAR10 | CIFAR100 | SVHN')
parser.add_argument('--datadir', default='datasets', type=str)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='mlp')
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--dimensions', default=3, type=int)    # has no effect
parser.add_argument('--output', default='.', type=str)
args = parser.parse_args()


def main():
    start = timeit.default_timer()

    ######## shape parameters
    nchannels, nclasses = 3, 10
    if args.dataset == 'MNIST' or args.dataset == 'FashionMNIST': nchannels = 1
    if args.dataset == 'CIFAR100': nclasses = 100
    if args.dataset == 'imagenet': nclasses = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ######## download datasets
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_dataset = utils.load_data("train", args.dataset, args.datadir)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, **kwargs)
    test_dataset = utils.load_data("test", args.dataset, args.datadir)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, **kwargs)

    ######## prepare model structure
    model, save_dir = utils.prepare_model(args, nchannels, nclasses)
    wandb.init(project="HHN_scaling", entity="ahinea", name="One4All_%s" % save_dir)
    model.to(device)
    print(model)
    print(utils.count_model_parameters(model))

    ######## train model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    def train(dataloader, model, loss_fn, optimizer):
        for batch, (X, y) in enumerate(tqdm(dataloader, desc='Training')):
            scale = random.uniform(0.2, 2)
            X, y = X.to(device), y.to(device)
            X = TF.affine(X, scale=scale, angle=0, translate=(0, 0), shear=0.0)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    def validate(dataloader, model, loss_fn):
        scale = random.uniform(0.2, 2)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                X = TF.affine(X, scale=scale, angle=0, translate=(0, 0), shear=0.0)

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with scale={scale}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct, test_loss

    for t in range(args.epochs):
        print(f"=================\n Epoch: {t + 1} \n=================")
        train(train_loader, model, loss_fn, optimizer)
        test_acc, test_loss = validate(test_loader, model, loss_fn)
        wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    print("Done!")

    ######## test model
    def test(dataloader, model, loss_fn, scale):
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                X = TF.affine(X, scale=scale, angle=0, translate=(0, 0), shear=0.0)

                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with scale={scale}: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc = []
    for scale in tqdm(np.arange(0.2, 2, 0.05), desc='Testing'):
        acc.append(test(test_loader, model, loss_fn, scale))

    ######## write to the bucket
    destination_name = f'{args.output}/scaling/One4All/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)
    np.save(f'{destination_name}/acc.npy', pickle.dumps(acc))

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()

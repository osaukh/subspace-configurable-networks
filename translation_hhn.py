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

parser = argparse.ArgumentParser(description='SCN Project')
parser.add_argument('--dataset', default='FashionMNIST', type=str, help='MNIST | FashionMNIST | CIFAR10 | CIFAR100 | SVHN')
parser.add_argument('--datadir', default='datasets', type=str)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='hhnmlpb')
parser.add_argument('--nlayers', default=1, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--dimensions', default=3, type=int)
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
    model, save_dir = utils.prepare_model(args, nchannels, nclasses, hin=2)
    wandb.init(project="SCN_translation", entity="ahinea", name="SCN_%s" % save_dir)
    model.to(device)
    print(model)
    print(utils.count_model_parameters(model))

    ######## train model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def train(dataloader, model, loss_fn, optimizer):
        for batch, (X, y) in enumerate(tqdm(dataloader, desc='Training')):
            X, y = X.to(device), y.to(device)
            a, b = random.uniform(-8, 8), random.uniform(-8, 8)
            X = TF.affine(X, scale=1.0, angle=0, translate=(a, b), shear=0.0)

            pred = model(X, Tensor([a, b]).to(device))
            loss = loss_fn(pred, y)

            beta1 = model.hyper_stack(Tensor([a, b]).to(device))
            a2, b2 = random.uniform(-8, 8), random.uniform(-8, 8)
            beta2 = model.hyper_stack(Tensor([a2, b2]).to(device))
            loss += 0.1*pow(cos(beta1, beta2),2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    def validate(dataloader, model, loss_fn):
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                a, b = random.uniform(-8, 8), random.uniform(-8, 8)
                X = TF.affine(X, scale=1.0, angle=0, translate=(a, b), shear=0.0)

                pred = model(X, Tensor([a, b]).to(device))
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with translation={a,b}: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct, test_loss

    for t in range(args.epochs):
        print(f"=================\n Epoch: {t + 1} \n=================")
        train(train_loader, model, loss_fn, optimizer)
        test_acc, test_loss = validate(test_loader, model, loss_fn)
        wandb.log({"test/loss": test_loss, "test/acc": test_acc})
    print("Done!")

    ######## test model
    def test(dataloader, model, loss_fn, a, b):
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                X = TF.affine(X, scale=1.0, angle=0, translate=(a, b), shear=0.0)

                pred = model(X, Tensor([a, b]).to(device))
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with translation={a,b}: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc = []
    for a in tqdm(np.linspace(-8, 8, 17), desc='Testing'):
        for b in np.linspace(-8, 8, 17):
            acc.append(test(test_loader, model, loss_fn, a, b))

    ######## test model fixed degree
    def test_fixed(dataloader, model, loss_fn, a, b):
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                X = TF.affine(X, scale=1.0, angle=0, translate=(a, b), shear=0.0)

                pred = model(X, Tensor([0, 0]).to(device))  # fix the model
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /=  len(dataloader)
        correct /= len(dataloader.dataset)
        print(f"Test with translation={a,b}: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
        return correct

    acc_fixed = []
    for a in tqdm(np.linspace(-8, 8, 17), desc='Testing'):
        for b in np.linspace(-8, 8, 17):
            acc_fixed.append(test_fixed(test_loader, model, loss_fn, a, b))

    ######## compute beta space
    beta_space = []
    for a in tqdm(np.linspace(-8, 8, 17), desc='Testing'):
        for b in np.linspace(-8, 8, 17):
            Hyper_X = Tensor([a, b]).to(device)
            beta_space.append(model.hyper_stack(Hyper_X).cpu().detach().numpy())

    beta_space = np.stack(beta_space)
    print(beta_space.shape)

    hhn_dict = {'acc': acc, 'acc_fixed': acc_fixed, 'beta_space': np.array(beta_space)}

    ######## write to the bucket
    destination_name = f'{args.output}/translation/SCN/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)
    np.save(f'{destination_name}/acc.npy', pickle.dumps(hhn_dict))

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()

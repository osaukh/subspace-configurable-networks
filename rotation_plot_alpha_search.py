import os
import math
import matplotlib.patches as mpatches
import numpy as np
import pickle
from matplotlib.ticker import StrMethodFormatter

import random
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy import optimize

import models

output = "SCN_alpha"
os.makedirs(f'{output}/figs/rotation/', exist_ok=True)

def dacc(arch, dataset, widths, layers):
    dimensions = [1, 2, 3, 5, 8, 16, 32, 64]
    fixed_angles = [0, 30, 45, 60, 90]

    plt.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

    for l in layers:
        for w in widths:
            # One4All
            file_name = f'{output}/output/rotation/One4All/{arch}_{dataset}_{l}_{w}/acc.npy'
            acc_one4all = pickle.loads(np.load(file_name))

            # Inverse
            file_name = f'{output}/output/rotation/Inverse/{arch}_{dataset}_{l}_{w}/acc.npy'
            acc_inverse = pickle.loads(np.load(file_name))

            # One4one
            file_name = f'{output}/output/rotation/One4One/{arch}_{dataset}_{l}_{w}/acc.npy'
            acc_one4one = pickle.loads(np.load(file_name))

            # HHN
            acc_hhn = []
            for d in dimensions:
                file_name = f'{output}/output/rotation/HHN/hhn{arch}_{dataset}_{l}_{w}_{d}/acc.npy'
                a_hhn = pickle.loads(np.load(file_name))
                acc_hhn.append(a_hhn['acc'])

            # HHN alpha search
            acc_hhn_alpha = {}
            for b in [512, 64, 16, 4, 1]:
                acc_hhn_alpha[b] = []
                for d in dimensions[1:]:
                    acc_hhn_alpha[b].append(alpha_search(arch, dataset, w, l, d, b))
                print(f'batch={b}: ', acc_hhn_alpha[b])
            print(acc_hhn_alpha)

            # helper function
            labels = []
            def add_label(violin, label):
                color = violin["bodies"][0].get_facecolor().flatten()
                labels.append((mpatches.Patch(color=color), label))

            # Plot D vs acc, with One4All, Inverse, One4One
            fig = plt.figure()
            fig.tight_layout()
            fig, ax = plt.subplots()
            add_label(ax.violinplot(acc_hhn, dimensions, widths=5), 'SCNs')

            add_label(ax.violinplot(acc_one4all, [-7], widths=5), 'One4All')
            add_label(ax.violinplot(acc_inverse, [dimensions[-1] + 9], widths=5), 'Inverse')

            x = []
            for fixed_angle in fixed_angles:
                x.append(acc_one4one[str(fixed_angle)])
            add_label(ax.violinplot(x, [dimensions[-1] + 16], widths=5), "One4One")

            ax.legend(*zip(*labels), loc='lower right', prop={'size': 16})

            plt.xlabel('Number of dimensions D', fontsize=18)
            plt.ylabel('Test accuracy', fontsize=18)
            plt.title('Rotation - MLP - FMNIST', fontsize=20)

            ax2 = ax.twinx()
            ax2.get_shared_y_axes().join(ax, ax2)
            ax2.set_yticks([])
            ax2.plot(dimensions[1:], acc_hhn_alpha[512], 'ks', markersize=8, markerfacecolor='white', label='bs=512')
            ax2.plot(dimensions[1:], acc_hhn_alpha[64], 'ko', markersize=6, markerfacecolor='white', label='bs=64')
            ax2.plot(dimensions[1:], acc_hhn_alpha[16], 'kv', markersize=5, markerfacecolor='white', label='bs=16')
            ax2.plot(dimensions[1:], acc_hhn_alpha[4], 'k*', markersize=4, markerfacecolor='white', label='bs=4')
            ax2.plot(dimensions[1:], acc_hhn_alpha[1], 'ko', markersize=3, markerfacecolor='white', label='bs=1')
            ax2.legend(loc=(0.37, 0.04), title='I-SCNs:', prop={'size': 10})

            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='y', which='major', labelsize=16)
            # ax.set_xticks(dimensions)
            # ax.set_xticks([1, 5, 8, 16, 32, 64])

            ax.set_xticks([1, 2, 3, 5, 8, 16, 32, 64])
            ax.set_xticklabels(["1", "", "", "5", "8", "16", "32", "64"])

            ax.axvline(x=-2, color='k', ls='--')
            ax.axvline(x=67, color='k', ls='--')

            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            # ax.set_xscale('log')
            ax.grid(True, linestyle=':')
            plt.savefig(f"{output}/figs/rotation_d_{arch}_{dataset}_{l}_{w}.png", bbox_inches='tight', dpi=300)

def transform_angle(angle):
    cos = math.cos(angle / 180 * math.pi)
    sin = math.sin(angle / 180 * math.pi)
    return Tensor([cos, sin])

# function to minimize by the basin hopping algorithm
def f(z, *args):
    alpha = transform_angle(((1+z)*180)%360-180)
    X = args[0]['X']
    model = args[0]['model']
    logits = model(Tensor(X), hyper_x=Tensor(alpha))
    b = (F.softmax(logits, dim=1)) * (-1 * F.log_softmax(logits, dim=1))  # entropy
    return b.sum().numpy()

# given a batch of images find the rotation angle alpha
def findalpha(model, X):
    # Basin hopping algorithm
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
    minimizer_kwargs = {"method": "BFGS", "args":{'X':X, 'model':model}}
    res = optimize.basinhopping(f, 0.0, minimizer_kwargs=minimizer_kwargs, niter=100, T=0.1)

    alpha = ((1+res.x[0])*180)%360-180
    # print("alpha estimate = ", alpha)     # obtained minimum
    # print("fun = ", res.fun)              # function value at minimum
    return alpha

def alpha_search(arch, dataset, w, l, d, batch_size):
    # Download test data from open datasets
    test_data = datasets.FashionMNIST(root="datasets", train=False, download=True,
                                  transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.5], std=[0.5])]),)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model = models.HHN_MLPB(hin=2, dimensions=d, n_layers=1, n_units=32, n_channels=1)
    file_name = f'{output}/output/rotation/HHN/hhn{arch}_{dataset}_{l}_{w}_{d}/model_alpha.pt'
    model.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))

    # model to eval mode and move to cpu
    model.eval()
    model.cpu()
    # freeze Ws
    for param in model.parameters():
        param.requires_grad = False

    result = 0.0
    for (X, y) in test_loader:
        angle = random.uniform(-180, 180)
        # print("=============")
        # print("alpha true = ", angle)
        X = TF.rotate(X, angle)

        alpha = findalpha(model, X)

        # compute model prediction with the estimated alpha
        logits = model(X, hyper_x=transform_angle(alpha))
        # y is the true label --> calculate accuracy
        correct = (logits.argmax(1) == y).type(torch.float).sum().item() / batch_size
        # print(f"accuracy = {(100*correct):>0.1f}")
        result += correct

    result /= len(test_loader.dataset) / batch_size
    print(f"Test accuracy: {(100 * result):>0.1f}%")
    return result

if __name__ == '__main__':
    dacc('mlpb', 'FashionMNIST', widths=[32], layers=[1])

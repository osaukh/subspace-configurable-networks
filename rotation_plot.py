import os
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import numpy as np
import statistics
import timeit
import pickle
import utils
from matplotlib.ticker import StrMethodFormatter

output = "/User/anonymous/subspace-configurable-networks"
os.makedirs(f'{output}/figs/rotation/', exist_ok=True)


def viz(arch, dataset, widths, layers, dimensions):
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, 8))

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

            theta = np.arange(0, 360, 1) / 180 * math.pi
            # ref = np.ones_like(hypernet_acc) * 0.85

            # Plot acc
            fig = plt.figure()
            fig.tight_layout()
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.plot(theta, acc_one4all, label='One4All', color='grey', lw=3)
            ax.plot(theta, acc_inverse, label='Inverse', color='k', lw=3)
            for i, d in zip(range(len(dimensions)), dimensions):
                file_name = f'{output}/output/rotation/HHN/hhn{arch}_{dataset}_{l}_{w}_{d}/acc.npy'
                acc_hhn = pickle.loads(np.load(file_name))
                ax.plot(theta, acc_hhn['acc'], label=f'SCN D={d}', color=colors[i])
            for angle in fixed_angles:
                ax.plot(angle / 180 * math.pi, acc_one4one[str(angle)], '*', color='grey', lw=3, markersize=10, markerfacecolor='white')
            ax.plot(angle / 180 * math.pi, acc_one4one[str(angle)], '*', color='grey', label="One4One", lw=3, markersize=10, markerfacecolor='white')

            ax.set_theta_zero_location("N")
            ax.set_rticks([0.7, 0.8, 0.90])  # Less radial ticks
            ax.set_rmin(0.64)
            ax.set_rmax(0.90)
            ax.set_rlabel_position(-77.5)  # Move radial labels away from plotted line
            ax.grid(True)
            plt.title('Rotation - MLP - FMNIST', fontsize=16, pad=20)
#            plt.title('Rotation - ShallowCNN - SVHN', fontsize=16, pad=20)
#            plt.title('Rotation - ResNet18 - CIFAR10', fontsize=16, pad=20)
            plt.legend(bbox_to_anchor=(0.85, -0.1), ncol=2, prop={'size': 10})
            ax.tick_params(axis='x', which='major', labelsize=12)
            ax.tick_params(axis='y', which='major', labelsize=12)
            plt.savefig(f"./figs/rotation/viz_acc_{arch}_{dataset}_{l}_{w}.png", bbox_inches='tight', dpi=300)

            # Plot acc fixed
            fig = plt.figure()
            fig.tight_layout()
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.plot(theta, acc_one4all, label='One4All', color='grey', lw=3)
            ax.plot(theta, acc_inverse, label='Inverse', color='k', lw=3)
            for i, d in zip(range(len(dimensions)), dimensions):
                file_name = f'{output}/output/rotation/HHN/hhn{arch}_{dataset}_{l}_{w}_{d}/acc.npy'
                acc_hhn = pickle.loads(np.load(file_name))
                # ax.plot(theta, acc_hhn['acc'], label=f'HHN D={d}', color=colors[i])
                ax.plot(theta, acc_hhn['acc_fixed'], label=rf'SCN D={d}, $\alpha=0^\circ$', color=colors[i])
            ax.plot(0 / 180 * math.pi, acc_one4one[str(angle)], '*', color='grey', label="One4One", lw=3, markersize=10, markerfacecolor='white')
            ax.set_theta_zero_location("N")
            ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])  # Less radial ticks
            ax.set_rmin(0.0)
            ax.set_rmax(1.0)
            ax.set_rlabel_position(-77.5)  # Move radial labels away from plotted line
            ax.grid(True)
            plt.title('Rotation - MLP - FMNIST', fontsize=16, pad=20)
#            plt.title('Rotation - ShallowCNN - SVHN', fontsize=16, pad=20)
#            plt.title('Rotation - ResNet18 - CIFAR10', fontsize=16, pad=20)
            plt.legend(bbox_to_anchor=(0.99, -0.1), ncol=2, prop={'size': 10})
            ax.tick_params(axis='x', which='major', labelsize=12)
            ax.tick_params(axis='y', which='major', labelsize=12)
            plt.savefig(f"./figs/rotation/viz_acc_fixed_{arch}_{dataset}_{l}_{w}.png", bbox_inches='tight', dpi=300)

            # Beta space
            fig = plt.figure()
            fig.tight_layout()
            fig, ax = plt.subplots(1, len(dimensions[:5]), subplot_kw={"projection": "3d"})
            for d_id, d in zip(range(len(dimensions[:5])), dimensions[:5]):
                file_name = f'{output}/output/rotation/HHN/hhn{arch}_{dataset}_{l}_{w}_{d}/acc.npy'
                acc_hhn = pickle.loads(np.load(file_name))
                plt.rc('font', **{'size': 4})
                cols = cm.gist_rainbow(np.linspace(0, 1, d))
                xline = np.sin(theta)
                yline = np.cos(theta)
                for i in range(len(acc_hhn['beta_space'][0])):
                    zline = acc_hhn['beta_space'][:, i]
                    ax[d_id].plot(xline, yline, zline, alpha=0.5, color=cols[i], label=r'$\beta_{%d}$' % (i + 1))
                ax[d_id].set_aspect('equal', 'box')
                ax[d_id].view_init(elev=35, azim=30)
                ax[d_id].tick_params(axis='x', which='major', labelsize=4, pad=-5)
                ax[d_id].tick_params(axis='y', which='major', labelsize=4, pad=-5)
                ax[d_id].tick_params(axis='z', which='major', labelsize=4, pad=-2)
                ax[d_id].legend(loc='upper left', prop={'size': 3})
                ax[d_id].axes.xaxis.labelpad = -12
                ax[d_id].axes.yaxis.labelpad = -10
                ax[d_id].set_xlabel(r'$\alpha_1$', fontsize=4)
                ax[d_id].set_ylabel(r'$\alpha_2$', fontsize=4)
                ax[d_id].grid(True, linestyle=':')
                ax[d_id].set_title(f'D={d}')
            plt.savefig(f"{output}/figs/rotation/viz_beta_{arch}_{dataset}_{l}_{w}.png", bbox_inches='tight', dpi=600)

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

            add_label(ax.violinplot(acc_one4all, [0], widths=5), 'One4All')
            add_label(ax.violinplot(acc_inverse, [dimensions[-1] + 1], widths=5), 'Inverse')

            x = []
            for fixed_angle in fixed_angles:
                x.append(acc_one4one[str(fixed_angle)])
            add_label(ax.violinplot(x, [dimensions[-1] + 2], widths=5), "One4One")

            plt.legend(*zip(*labels), loc='lower right', prop={'size': 16})
            plt.xlabel('Number of dimensions D', fontsize=18)
            plt.ylabel('Test accuracy', fontsize=18)
            plt.title('Rotation - MLP - FMNIST', fontsize=20)
#            plt.title('Rotation - ShallowCNN - SVHN', fontsize=20)
#            plt.title('Rotation - ResNet18 - CIFAR10', fontsize=20)
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='y', which='major', labelsize=16)
            # ax.set_xticks(dimensions)
            ax.set_xticks([1, 5, 8, 16, 32, 64])
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            ax.grid(True, linestyle=':')
            plt.savefig(f"{output}/figs/rotation/d_{arch}_{dataset}_{l}_{w}.png", bbox_inches='tight', dpi=300)


def wacc(arch, dataset, layers, widths):
    dimensions = [1, 2, 3, 5, 8]
    fixed_angles = [0, 30, 45, 60, 90]
    angles = [0, 30, 45, 60, 90]
    dimension_colors = plt.cm.gist_rainbow(np.linspace(0, 1, 5))

    plt.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

    for l in layers:
        for i in range(len(angles)):
            # Plot W vs acc, with One4All and Inverse
            fig = plt.figure()
            fig.tight_layout()
            fig, ax = plt.subplots()

            # One4All
            acc_one4all = []
            for w in widths:
                file_name = f'{output}/output/rotation/One4All/{arch}_{dataset}_{l}_{w}/acc.npy'
                acc_one4all.append(pickle.loads(np.load(file_name))[angles[i]])
            ax.plot(widths, acc_one4all, 'v', color='grey', label="One4All", lw=3, markersize=10, markerfacecolor='white')

            # Inverse
            acc_inverse = []
            for w in widths:
                file_name = f'{output}/output/rotation/Inverse/{arch}_{dataset}_{l}_{w}/acc.npy'
                acc_inverse.append(pickle.loads(np.load(file_name))[angles[i]])
            ax.plot(widths, acc_inverse, 'H', color='grey', label="Inverse", lw=3, markersize=10, markerfacecolor='white')

            # One4one
            acc_one4one = []
            for w in widths:
                file_name = f'{output}/output/rotation/One4One/{arch}_{dataset}_{l}_{w}/acc.npy'
                acc_one4one.append(pickle.loads(np.load(file_name))[str(fixed_angles[i])])
            ax.plot(widths, acc_one4one, '*', color='grey', label="One4One", lw=3, markersize=10, markerfacecolor='white')

            # HHN
            for x, d in zip(range(len(dimensions)), dimensions):
                acc_hhn = []
                for w in widths:
                    file_name = f'{output}/output/rotation/HHN/hhn{arch}_{dataset}_{l}_{w}_{d}/acc.npy'
                    acc_hhn.append(pickle.loads(np.load(file_name))['acc'][angles[i]])
                ax.plot(widths, acc_hhn, 'o', color=dimension_colors[x], label=f"SCN D={d}", lw=3)

            plt.legend(loc='lower right', prop={'size': 14}, ncol=2)
            plt.xlabel('Width', fontsize=16)
            plt.ylabel('Test accuracy', fontsize=16)
            plt.title('Rotation - MLP - FMNIST', fontsize=20)
#            plt.title('Rotation - ShallowCNN - SVHN', fontsize=20)
#            plt.title('Rotation - ResNet18 - CIFAR10', fontsize=20)
            ax.set_xscale('log', basex=2)
            ax.tick_params(axis='x', which='major', labelsize=14)
            ax.tick_params(axis='y', which='major', labelsize=14)
            ax.grid(True, linestyle=':')
            # plt.title(f'{dataset} / {arch}')
            plt.savefig(f"{output}/figs/rotation/w_{arch}_{dataset}_{l}_{fixed_angles[i]}.png", bbox_inches='tight', dpi=300)

def lacc(arch, dataset, layers, widths):
    dimensions = [1, 2, 3, 5, 8]
    fixed_angles = [0, 30, 45, 60, 90]
    angles = [0, 30, 45, 60, 90]
    dimension_colors = plt.cm.gist_rainbow(np.linspace(0, 1, 5))

    plt.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

    for w in widths:
        for i in range(len(angles)):
            # Plot W vs acc, with One4All and Inverse
            fig = plt.figure()
            fig.tight_layout()
            fig, ax = plt.subplots()

            # One4All
            acc_one4all = []
            for l in layers:
                file_name = f'{output}/output/rotation/One4All/{arch}_{dataset}_{l}_{w}/acc.npy'
                acc_one4all.append(pickle.loads(np.load(file_name))[angles[i]])
            ax.plot(layers, acc_one4all, 'v', color='grey', label="One4All", lw=3, markersize=10, markerfacecolor='white')

            # Inverse
            acc_inverse = []
            for l in layers:
                file_name = f'{output}/output/rotation/Inverse/{arch}_{dataset}_{l}_{w}/acc.npy'
                acc_inverse.append(pickle.loads(np.load(file_name))[angles[i]])
            ax.plot(layers, acc_inverse, 'H', color='grey', label="Inverse", lw=3, markersize=10, markerfacecolor='white')

            # One4one
            acc_one4one = []
            for l in layers:
                file_name = f'{output}/output/rotation/One4One/{arch}_{dataset}_{l}_{w}/acc.npy'
                acc_one4one.append(pickle.loads(np.load(file_name))[str(fixed_angles[i])])
            ax.plot(layers, acc_one4one, '*', color='grey', label="One4One", lw=3, markersize=10, markerfacecolor='white')

            # HHN
            for x, d in zip(range(len(dimensions)), dimensions):
                acc_hhn = []
                for l in layers:
                    file_name = f'{output}/output/rotation/HHN/hhn{arch}_{dataset}_{l}_{w}_{d}/acc.npy'
                    acc_hhn.append(pickle.loads(np.load(file_name))['acc'][angles[i]])
                ax.plot(layers, acc_hhn, 'o', color=dimension_colors[x], label=f"SCN D={d}", lw=3)

            plt.legend(loc='lower right', prop={'size': 14}, ncol=2)
            plt.xlabel('Depth', fontsize=16)
            plt.ylabel('Test accuracy', fontsize=16)
            plt.title('Rotation - MLP - FMNIST', fontsize=20)
#            plt.title('Rotation - ShallowCNN - SVHN', fontsize=20)
#            plt.title('Rotation - ResNet18 - CIFAR10', fontsize=20)
            plt.xticks(layers)
            ax.tick_params(axis='x', which='major', labelsize=14)
            ax.tick_params(axis='y', which='major', labelsize=14)
            ax.grid(True, linestyle=':')
            # plt.title(f'{dataset} / {arch}')
            plt.savefig(f"{output}/figs/rotation/l_{arch}_{dataset}_{w}_{fixed_angles[i]}.png", bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    viz('mlpb', 'FashionMNIST', widths=[32], layers=[1], dimensions=[1, 2, 3, 5, 8,16,32,64])
    dacc('mlpb', 'FashionMNIST', widths=[32], layers=[1])

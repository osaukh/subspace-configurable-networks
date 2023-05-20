import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle
from matplotlib.ticker import StrMethodFormatter

transform = "sharpness"

output = "SCN"
os.makedirs(f'{output}/figs/{transform}/', exist_ok=True)

def dacc(arch, dataset, widths, layers):
    dimensions = [1, 2, 3, 5, 8, 16]
    fixed_setting = [0.2, 0.5, 1.0, 1.5]

    plt.rcParams.update({'font.size': 14, 'legend.fontsize': 14})

    for l in layers:
        for w in widths:
            print(f"arch={arch}, dataset={dataset}")

            # One4All
            file_name = f'{output}/output/{transform}/One4All/{arch}_{dataset}_{l}_{w}/acc.npy'
            acc_one4all = pickle.loads(np.load(file_name))
            print(f"One4All acc={np.mean(acc_one4all)}")

            # Inverse
            file_name = f'{output}/output/{transform}/Inverse/{arch}_{dataset}_{l}_{w}/acc.npy'
            acc_inverse = pickle.loads(np.load(file_name))
            print(f"Inverse acc={np.mean(acc_inverse)}")

            # One4one
            file_name = f'{output}/output/{transform}/One4One/{arch}_{dataset}_{l}_{w}/acc.npy'
            acc_one4one = pickle.loads(np.load(file_name))
            print(f"One4Onne acc={sum(acc_one4one.values()) / len(acc_one4one)}")

            # HHN
            acc_hhn = []
            for d in dimensions:
                file_name = f'{output}/output/{transform}/HHN/hhn{arch}_{dataset}_{l}_{w}_{d}/acc.npy'
                a_hhn = pickle.loads(np.load(file_name))
                acc_hhn.append(a_hhn['acc'])
                print(f"D={d}, acc={np.mean(a_hhn['acc'])}")

            # helper function
            labels = []
            def add_label(violin, label):
                color = violin["bodies"][0].get_facecolor().flatten()
                labels.append((mpatches.Patch(color=color), label))

            # Plot D vs acc, with One4All, Inverse, One4One
            fig = plt.figure()
            fig.tight_layout()
            fig, ax = plt.subplots()
            add_label(ax.violinplot(acc_hhn, dimensions, widths=1), 'SCNs')

            add_label(ax.violinplot(acc_one4all, [-1], widths=1), 'One4All')
            add_label(ax.violinplot(acc_inverse, [dimensions[-1] + 2], widths=1), 'Inverse')

            x = []
            for fixed_s in fixed_setting:
                x.append(acc_one4one[str(fixed_s)])
            add_label(ax.violinplot(x, [dimensions[-1] + 3], widths=1), "One4One")

            plt.legend(*zip(*labels), loc='lower center', prop={'size': 16})
            plt.xlabel('Number of dimensions D', fontsize=18)
            plt.ylabel('Test accuracy', fontsize=18)
            # plt.title(f'{transform} - {arch} - {dataset}', fontsize=20)
            plt.title(f'Sharpness - MLP - FMNIST', fontsize=20)
            # plt.title(f'Sharpness - ShallowCNN - SVHN', fontsize=20)
            ax.tick_params(axis='x', which='major', labelsize=16)
            ax.tick_params(axis='y', which='major', labelsize=16)
            # ax.set_xticks(dimensions)
            # ax.set_ylim([0.87,0.89])
            ax.set_xticks([1, 2, 3, 5, 8, 16])
            ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
            # ax.set_xscale('log')

            ax.axvline(x=0, color='k', ls='--')
            ax.axvline(x=17, color='k', ls='--')

            ax.grid(True, linestyle=':')
            plt.savefig(f"{output}/figs/{transform}/d_{arch}_{dataset}_{l}_{w}.png", bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    dacc('mlpb', 'FashionMNIST', widths=[32], layers=[1])
    # dacc('sconvb', 'SVHN', widths=[32], layers=[2])
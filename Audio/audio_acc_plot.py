import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
import pickle
from matplotlib.ticker import StrMethodFormatter

transform = "speed"

output = "./"
os.makedirs(f'{output}/figs/{transform}/', exist_ok=True)

colors = {
    'SCNs':[0.12156863, 0.46666667, 0.70588235, 0.3  ] ,
    'One4All':  [1.,         0.49803922, 0.05490196, 0.3       ],
    'One4One': [0.83921569, 0.15294118, 0.15686275, 0.3       ],

}

def dacc(arch, dataset):
    dimensions = [1, 2, 3, 5, 8, 16]
    fixed_setting = [-8., -4., 0., 4., 8.]

    plt.rcParams.update({'font.size': 14, 'legend.fontsize': 14})



    # One4All
    file_name = f'{output}/output/{transform}/One4All/audio/acc.npy'
    acc_one4all = pickle.loads(np.load(file_name))
    acc_one4all = [ r[0] for r in acc_one4all]

    print(f"One4All acc={np.mean(acc_one4all)}")

    # # Inverse
    # file_name = f'{output}/output/{transform}/Inverse/acc.npy'
    # acc_inverse = pickle.loads(np.load(file_name))
    # print(f"Inverse acc={np.mean(acc_inverse)}")

    # One4one
    file_name = f'{output}/output/{transform}/One4One/audio/acc.npy'
    acc_one4one = list(pickle.loads(np.load(file_name)).values())
    print(f"One4One acc={sum(acc_one4one) / len(acc_one4one)}")

    # HHN
    acc_hhn = []
    for d in dimensions:
        file_name = f'{output}/output/{transform}/HHN/audio_{d}/acc.npy'
        a_hhn = pickle.loads(np.load(file_name))
        acc = [ r[0] for r in a_hhn]
        acc_hhn.append(acc)
        print(f"D={d}, acc={np.mean(acc)}")

    # helper function
    labels = []
    def add_label(violin, label):
        color = violin["bodies"][0].get_facecolor().flatten()
        # color = colors[label]
        labels.append((mpatches.Patch(color=color), label))

    # Plot D vs acc, with One4All, Inverse, One4One
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors.values())

    fig = plt.figure()
    fig.tight_layout()
    fig, ax = plt.subplots()
    add_label(ax.violinplot(acc_hhn, dimensions, widths=1), 'SCNs')

    add_label(ax.violinplot(acc_one4all, [-1], widths=1), 'One4All')
    # add_label(ax.violinplot(acc_one4all, [dimensions[-1] + 1], widths=1), 'Inverse')

    x = []
    for i, fixed_s in enumerate(fixed_setting):
        x.append(acc_one4one[i])
    add_label(ax.violinplot(x, [dimensions[-1] + 2], widths=1), "One4One")

    plt.legend(*zip(*labels), loc='lower center', prop={'size': 16})
    plt.xlabel('Number of dimensions D', fontsize=18)
    plt.ylabel('Test accuracy', fontsize=18)
    # plt.title(f'{transform} - {arch} - {dataset}', fontsize=20)
    # plt.title(f'Sharpness - MLP - FMNIST', fontsize=20)
    # plt.title(f'Sharpness - ShallowCNN - SVHN', fontsize=20)
    plt.title(f'{transform.title()} - {arch} - {dataset}', fontsize=20)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='y', which='major', labelsize=16)
    ax.axvline(x=0, color = 'k', ls='--')
    ax.axvline(x=17, color='k', ls='--')
    # ax.set_xticks(dimensions)
    # ax.set_ylim([0.87,0.89])
    ax.set_xticks([1, 2, 3, 5, 8, 16])
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    # ax.set_xscale('log')
    ax.grid(True, linestyle=':')
    filename = f"{output}/figs/{transform}/{transform.title()}_{arch}_{dataset}.png"
    print(filename)
    plt.savefig(filename, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    # dacc('mlpb', 'FashionMNIST', widths=[32], layers=[1])
    dacc('M5', 'SpeechCommands')
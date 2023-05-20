# subspace-configurable-networks

This repository contains the code that was used to train subspace-configurable networks (SCNs) and all baselines for the 2D transformations used in the paper 'Representing Input Transformations by Low-Dimensional Parameter Subspaces'.

Also see the [interactive SCN's beta-subspace visualization page](https://subspace-configurable-networks.pages.dev) for all experiments in the paper.

The easiest way to train an SCN model on rotation is by running the ipython notebook `SCN_MLP_FMNIST_rotation.ipynb` in Google Colab. The file includes SCN training, testing, performance plot and visualization of the beta-subspace. 

Training SCNs takes from several minutes to several hours depending on the model size, the number of dimensions $D$ and the available resources.

Follow the instructions below to run the code from a command line. We used Python 3.7.7.

Install dependencies:
```
pip install -r requirements.txt
```

Run from the command line (also see `run.sh`) to train a SCN, D=3 for rotation for MLP inference network architecture on FMNIST with parameters as specified in the example.
```
CUDA_VISIBLE_DEVICES=0 \
python rotation_hhn.py \
	--arch=hhnmlpb \
	--dataset=FashionMNIST \
	--batchsize=64 \
	--epochs=500 \
	--learning_rate=0.001 \
	--output=output \
	--nlayers=1 \
	--width=32 \
	--dimensions=3
```

Replace `output` with the path to the output folder.

To train the baselines (One4All, Inverse and One4One), the specified network architecture should be `mlpb`. The other parameter values are kept the same.

```
CUDA_VISIBLE_DEVICES=0 \
python rotation_inverse.py \
	--arch=mlpb \
	--dataset=FashionMNIST \
	--batchsize=64 \
	--epochs=500 \
	--learning_rate=0.001 \
	--output=output \
	--nlayers=1 \
	--width=32
```

Sample trained models and computed statistics are stored in the `output` folder part of this repository. The folder structure is straightforwards with folder names including the architecture, the dataset, number of layers, network widths and the number of dimensions $D$.

You can use a sample plotting script to generate visualizations also presented in the paper:
```
python rotation_plot.py
```
Sample output is stores in the `figs` folder and is sorted by the transformation applied. We provide sample generated images visualizing the accuracy of SCNs for 2D rotation.

SCN accuracy for different D as a function of the input parameter alpha (=rotation angle):

<img src="./figs/rotation/viz_acc_mlpb_FashionMNIST_1_32.png" alt="SCN accuracy, D=1..8, 1-layer MLP with 32 hidden units" width="300"/>

SCN accuracy for different D, aggregated view for different D. Each violin comprises accuracies for all alphas sweeped with a discretization step of 1 degree:

<img src="./figs/rotation/d_mlpb_FashionMNIST_1_32.png" alt="SCN accuracy, D=1..8, 1-layer MLP with 32 hidden units, aggregated view" width="300"/>

SCN beta-space as a function of the input parameter alpha given as (cos(alpha), sin(alpha)):

<img src="./figs/rotation/viz_beta_mlpb_FashionMNIST_1_32.png" alt="SCN beta-space, D=1..8, 1-layer MLP with 32 hidden units" width="800"/>

We also provide [videos navigating the beta-subspace](./videos/) for different $D$.

The files `rotation_hhn_alpha_search.py` and `rotation_plot_alpha_search.py` provide implementation and testing of the invariant SCN (I-SCN) implementation via search in the alpha space. A simple example of the search for rotation and translation transformations is shown in the corresponding Google Colab files. Note that search in the alpha space is resource-intensive and takes hours for `bs=1`.

Visit subfolders "Audio" and "3D" for our experiments with audio transformations and 3D rotation-and-projection.

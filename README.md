# subspace-configurable-networks

The provided code was used to train SCN models and all baselines for the 2D transformations used in the paper 'Subspace-Configurable Networks'.

The easiest way to train an SCN model on rotation is by running the ipython notebook `SCN_Example_MLP_FMNIST_rotation.ipynb` in Google Colab.  

Follow the instructions below to run the code from the command line. We used Python 3.7.7.

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

To run the code to train the baselines (One4All, Inverse and One4One), the specified network architecture should be `mlpb`. The other parameter values are kept the same.

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

Sample trained models and computed statistics are provided in the `output` folder part of this repository. The folder structure is straightforwards with folder names including the architecture, the dataset, number of layers, network widths and the number of dimensions $D$.

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

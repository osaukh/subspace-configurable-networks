# SCN on 3D rotation



> Note: "HHN" is an alias name  of "SCN" in this folder.

**This document includes the following**

- How to train and evaluate SCN, One4One, and One4All models on Modelent10 dataset with 3D rotation-and-projection.
- How to visualize the hyper_output/beta-space of SCN.


[accuracy_result_npy](./accuracy_result_npy/) contains the dumped numpy narrays. For each file in this folder, you can take `np.load` to load the accuracy list of a model evaluated on some degrees.

[checkpoint](./checkpoint/) contains all checkpoints of all trained models. Normally, this folder would be accessed by `train_one4all.ipynb`, `train_one4one.ipynb`, `train_HHN.ipynb`, and `train_hhn_script.py`

[plots](./plots/) contains all generated images.

[weight](./weight/) contains the well-trained models for  evaluating the accuracy and hyper_ouput. After completeing training, I manully copy each `Epoch-6000-model-name.ptn` from [./checkpoint](./checkpoint) to [weight](./weight/)

[model](./model/) contains the definition of all models:
- SCNs
- One4One
- One4All

## Hardware

If you want to run all experiments in your local env, please make sure your **GPU at least has 13GB memory**.

## Download dataset

```
bash download_dataset.sh
```

## Setup environment

```
conda create -n scn3d python=3.9
conda activate scn3d
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
coonda install numpy matplotlib == 3.3.2 tensorboard pytz tqdm torchmetrics
conda install -c plotly plotly=5.11.0
conda install -c plotly python-kaleido
```


## How to train SCNs and baseline models

### One4One

One4One model learns 3D rotation transformation (fixed rotation parameters) from one fixed degree. 

Please run the notebook [./train_one4one.ipynb](./train_one4one.ipynb). 

### One4All

One4All model learns 3D rotation transformation (different rotation parameters) from different degrees. 

Please run the notebook [./train_one4all.ipynb](./train_one4all.ipynb).

### SCNs

Please run the notebook [./train_HHN.ipynb](./train_HHN.ipynb). For different `D`/number of dimensions, please change the `dimension` in the beginning of this notebook.


## Evaluating models


### One4All

Please see [./evaluate_accuracy_one4all.ipynb](./evaluate_accuracy_one4all.ipynb)

### SCNs

Please see [./evaluate_accuracy_HHN.ipynb](./evaluate_accuracy_HHN.ipynb)

### One4One

Please see [./train_one4one.ipynb](./train_one4one.ipynb) 

## Beta-space


Please check [./evaluate_beta_space_HHN.ipynb](./evaluate_beta_space_HHN.ipynb) for details. 

It shows:
1. How to compute the beta while fixing some angles;
2. How to draw beta space and beta-alpha space;
3. How to make the beta-space pictures and visualization webpages.

import time
import os
import time
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset
from pytorch3d.io import IO

from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.structures import join_meshes_as_batch
from tqdm import trange, tqdm
import warnings

random.seed = 231
import torch
from pytorch3d.io import IO


def test_load_3d_object(device):
    start = time.time()
    for i in range(10, 20, 1):
        mesh = IO().load_mesh(os.path.join(f'data/ModelNet10', f'chair/train/chair_00{i}.off'), device=device)
    end = time.time()
    print(f'[Average Load mesh from file] run time:{(end - start) / 10:>0.5f}s')


class ModelnetDataset(Dataset):
    def __init__(self, root_dir, folder="train", device=torch.device("cpu"), in_mem=True, infor="Default infor"):
        print(f'Preocessing {folder} data in_mem = {in_mem} and infor: {infor}')
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.files = []
        self.device = device
        self.in_mem = in_mem
        self.io = IO()
        if not self.in_mem:
            warnings.warn(
                "[ModelnetDataset.__init__] Mesh file will not be loaded into RAM during pre-processing.\n This will take longer time during training to load files.")
        for category in tqdm(self.classes.keys(), desc=f'Prepare {folder} dataset categories'):
            # for category in self.classes.keys():
            new_dir = root_dir / Path(category) / folder
            # for file in tqdm(os.listdir(new_dir), desc='Prepare dataset files'):
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {'pcd_path': new_dir / file, 'category': category}
                    if self.in_mem:
                        sample['mesh'] = self.__preproc__(sample['pcd_path'])
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):

        mesh = self.io.load_mesh(file, device=self.device, load_textures=True)

        # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
        # (scale, center) will be used to bring the predicted mesh to its original center and scale
        # Note that normalizing the target mesh, speeds up the optimization but is not necessary!
        verts = mesh.verts_packed()
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        trg_mesh = mesh.offset_verts_(-center)
        trg_mesh = trg_mesh.scale_verts_((1.0 / float(scale)))
        return trg_mesh

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']

        if not self.in_mem:
            mesh = self.__preproc__(pcd_path)
            self.files[idx]['mesh'] = mesh
        return {'mesh_idx': idx,
                'category': self.classes[category]}


def test_dataset(device):
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    path = Path("./data/ModelNet10")
    test_ds = ModelnetDataset(path, folder="test", device=device)
    print('Test dataset size: ', len(test_ds))

if __name__ == '__main__':
    test_device = "cuda:8" if torch.cuda.is_available() else "cpu"
    test_dataset(test_device)

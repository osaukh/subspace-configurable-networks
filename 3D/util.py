import torch
from torchvision.transforms import Resize, Grayscale
from tqdm.auto import tqdm
from pytorch3d.structures import join_meshes_as_batch
import numpy as np
import torchmetrics
from pytorch3d.transforms import euler_angles_to_matrix
import random
import shutil

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed=seed
    #  torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = True
     
class ResizeBatchRGBImages(object):
    def __init__(self, size):
        self.size = size
        assert (isinstance(size, tuple) and len(size) == 2), "'size' should be a tuple (h,w)"
        self.resize = Resize(size)
        self.grayscale = Grayscale()

    def __call__(self, images):
        assert len(images.shape) == 4
        images = images.permute(0, 3, 1, 2)
        t = []
        for i in range(len(images)):
            t.append(self.grayscale(self.resize(images[i])).unsqueeze(0))
        return torch.cat(t, 0)

# Train One4One and One4All
def train(dataloader, model, loss_fn, optimizer, epoch, pr, device, verbose=True):
    resize = ResizeBatchRGBImages((32, 32))
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    train_loss = 0.0
    train_accuracy = 0.0
    correct = 0.0
    display_freq = 10
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
        
    for i, data in enumerate(tqdm(dataloader, desc='Training')):
#     for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        y = data['category'].to(device)
        mesh_idx = data['mesh_idx'].cpu().detach().numpy().tolist()
        mesh_list = [dataloader.dataset.files[j]['mesh'] for j in mesh_idx]
        batch_meshes = join_meshes_as_batch(mesh_list, True)
        num_meshes = len(batch_meshes)


        

        angles = torch.randint(-180, 180, (3,))/180*torch.pi

        images = pr.render_silhouette(batch_meshes, device,
                                      dist=3.0,
                                      angles=angles)[..., :3]

        X = resize(images)
        X = X.to(device).float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss

        # Backpropagation

        loss.backward()
        optimizer.step()

        acc = accuracy_metric(pred, y)
        # if i % display_freq == 0 and i != 0:
        #     print(
        #         f'[Batch: {i:>4d}/{num_batches:>4d}] Accuracy: {100 * acc :>0.4f}% , Loss:{loss.item():>8f}')
    train_loss /= num_batches
    acc = accuracy_metric.compute()
    if verbose:
        print(f'[Train] Avg accuracy:{(100 * acc):>0.4f}%, Avg loss:{train_loss:>8f}')
    return acc.item(), train_loss

# Test One4One and One4All
def test(dataloader, model, loss_fn, pr, device, angles, verbose=True):
    if verbose:
        print(f'Testing angles: {angles}')
    resize = ResizeBatchRGBImages((32, 32))
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            y = data['category'].to(device)
            mesh_idx = data['mesh_idx'].cpu().detach().numpy().tolist()
            mesh_list = [dataloader.dataset.files[j]['mesh'] for j in mesh_idx]
            batch_meshes = join_meshes_as_batch(mesh_list, True)
            num_meshes = len(batch_meshes)

            images = pr.render_silhouette(batch_meshes, device,
                                dist=3.0,
                                angles=angles)[..., :3]

            X = resize(images)
            X = X.float()
            X = X.to(device).float()

            pred = model(X)

            loss = loss_fn(pred, y)
            test_loss += loss

            acc = accuracy_metric(pred, y)


    test_loss /= num_batches
    acc= accuracy_metric.compute()
    if verbose:
        print(f"[Test] Accuracy: {(100 * acc):>0.4f}%, Avg loss: {test_loss:>8f}")
    return acc.item(), test_loss

def transform_angles(angles):
    output = []
    for angle in angles:
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        output.append(cos)
        output.append(sin)
    return torch.Tensor(output)

# Train HHN
def train_HHN(dataloader, model, loss_fn, optimizer, epoch, pr, device, verbose=True):
    resize = ResizeBatchRGBImages((32, 32))
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    train_loss = 0.0
    train_accuracy = 0.0
    correct = 0.0
    display_freq = 10
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    
    for i, data in enumerate(tqdm(dataloader, desc='Training')):
        optimizer.zero_grad()
        y = data['category'].to(device)
        mesh_idx = data['mesh_idx'].cpu().detach().numpy().tolist()
        mesh_list = [dataloader.dataset.files[j]['mesh'] for j in mesh_idx]
        batch_meshes = join_meshes_as_batch(mesh_list, True)
        num_meshes = len(batch_meshes)

        angles = torch.randint(-180, 180, (3,))/180*torch.pi
        
        Hyper_x = transform_angles(angles=angles).to(device=device)
        images = pr.render_silhouette(batch_meshes, device,
                                      dist=3.0,
                                      angles=angles)[..., :3]

        X = resize(images)
        X = X.to(device).float()

        # Compute prediction error
        pred = model(X, Hyper_x)

        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()


        acc = accuracy_metric(pred, y)

    train_loss /= num_batches
    acc = accuracy_metric.compute()
    if verbose:
        print(f'[Train] Avg accuracy:{(100 * acc):>0.4f}%, Avg loss:{train_loss:>8f}')
    return acc.item(), train_loss

# Test HHN
def test_HHN(dataloader, model, loss_fn, pr, device, angles, verbose=True):
    # todo support test given angles
    if verbose:
        print(f'Testing angles: {angles}')
    resize = ResizeBatchRGBImages((32, 32))
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss, correct = 0, 0
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            y = data['category'].to(device)
            mesh_idx = data['mesh_idx'].cpu().detach().numpy().tolist()
            mesh_list = [dataloader.dataset.files[j]['mesh'] for j in mesh_idx]
            batch_meshes = join_meshes_as_batch(mesh_list, True)
            num_meshes = len(batch_meshes)

            Hyper_x = transform_angles(angles=angles).to(device=device)
            images = pr.render_silhouette(batch_meshes, device,
                                dist=3.0,
                                angles=angles)[..., :3]
            
        
            X = resize(images)
            X = X.float()
            X = X.to(device).float()

            pred = model(X, Hyper_x)

            loss = loss_fn(pred, y)
            test_loss += loss.item()

            acc = accuracy_metric(pred, y)


    test_loss /= num_batches
    acc= accuracy_metric.compute()
    if verbose:
        print(f"[Test] Accuracy: {(100 * acc):>0.4f}%, Avg loss: {test_loss:>8f}")
    return acc.item(), test_loss

def train_one4one(dataloader, model, loss_fn, optimizer, epoch, pr, device, angles, verbose=True):
    resize = ResizeBatchRGBImages((32, 32))
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    train_loss = 0.0
    train_accuracy = 0.0
    correct = 0.0
    display_freq = 10
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    # for i, data in enumerate(tqdm(dataloader, desc='Training')):
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        y = data['category'].to(device)
        mesh_idx = data['mesh_idx'].cpu().detach().numpy().tolist()
        mesh_list = [dataloader.dataset.files[j]['mesh'] for j in mesh_idx]
        batch_meshes = join_meshes_as_batch(mesh_list, True)
        num_meshes = len(batch_meshes)



        images = pr.render_silhouette(batch_meshes, device,
                                      dist=3.0,
                                      angles=angles)[..., :3]

        X = resize(images)
        X = X.to(device).float()

        # Compute prediction error
        pred = model(X)
        
        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()


        acc = accuracy_metric(pred, y)
        # if i % display_freq == 0 and i != 0:
        #     print(
        #         f'[Batch: {i:>4d}/{num_batches:>4d}] Accuracy: {100 * acc :>0.4f}% , Loss:{loss.item():>8f}')
    train_loss /= num_batches
    acc = accuracy_metric.compute()
    if verbose:
        print(f'[Train] Avg accuracy:{(100 * acc):>0.4f}%, Avg loss:{train_loss:>8f}')
    return acc.item(), train_loss
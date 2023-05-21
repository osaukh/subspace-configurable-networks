import os
import timeit
import argparse
import numpy as np
from torch import nn
import models
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from dataset import SpeechCommands, transform_audio
from models import *



parser = argparse.ArgumentParser(description='HHN Project')
parser.add_argument('--dataset', default='SpeedCommands', type=str, help='MNIST | FashionMNIST | CIFAR10 | CIFAR100 | SVHN')
parser.add_argument('--datadir', default='datasets', type=str)
parser.add_argument('--batchsize', default=64, type=int)
parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
parser.add_argument('--arch', '-a', metavar='ARCH', default='m5')
parser.add_argument('--nlayers', default=5, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--weightdecay', default=0.0001, type=float)
parser.add_argument('--dimensions', default=3, type=int)    # has no effect
parser.add_argument('--transform', default='pitchshift', type=str)
parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--output', default='.', type=str)
args = parser.parse_args()


def main():
    start = timeit.default_timer()

    ######## shape parameters
    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')
    if not torch.cuda.is_available():
        print(f'cuda is not available. using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.gpus)

    train_ds = SpeechCommands(root="data", subset='training', device=device, in_mem=True)
    test_ds =  SpeechCommands(root="data", subset='testing', device=device, in_mem=True)

    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.labels))

    train_loader = DataLoader(dataset=train_ds, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=args.batchsize, shuffle=True)


    result_dict = {}
    
    fixed_params = torch.FloatTensor([-8., -4., 0., 4., 8.]).to(device)

    if args.transform == "pitchshift":
        fixed_params = torch.FloatTensor([-8., -4., 0., 4., 8.]).to(device)
    if args.transform == "speed":
        fixed_params = torch.FloatTensor([0.1, 0.2, 0.4, 0.6, 1.]).to(device)


    for fixed_param in fixed_params:
        ######## prepare model structure
        model = models.M5()
        save_dir = "audio"


        model.to(device)
        print(model)
        # print(utils.count_model_parameters(model))

        ######## train model
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

        def train(model, fixed_param):
            model.train()
            for X in tqdm(train_loader, "Training"):
                data = X['waveform'].to(device)
                target = X['label'].to(device)

                # apply transform and model on whole batch directly on device
                # data = transform(data)

                # apply transform with random parameter from a range
                data = transform_audio(transform_type= args.transform, waveform=data, sample_rate=16000, factor=fixed_param, device=device)


                output = model(data)

                # negative log-likelihood for a tensor of size (batch x 1 x n_output)
                # loss = F.cross_entropy(output.squeeze(), target)
                loss = F.nll_loss(output.squeeze(), target)

                # print(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

        def number_of_correct(pred, target):
            # count number of correct predictions
            return pred.squeeze().eq(target).sum().item()

        def get_likely_index(tensor):
            # find most likely label index for each element in the batch
            return tensor.argmax(dim=-1)

        def validate(model, fixed_param):
            model.eval()
            tloss, correct = 0, 0
            for X in tqdm(test_loader, "Validating"):
                data = X['waveform'].to(device)
                target = X['label'].to(device)

                # apply transform and model on whole batch directly on device
                data = transform_audio(transform_type= args.transform, waveform=data, sample_rate=16000, factor=fixed_param, device=device)


                output = model(data)

                # loss = F.cross_entropy(output.squeeze(), target)
                loss = F.nll_loss(output.squeeze(), target)


                pred = get_likely_index(output)
                correct += number_of_correct(pred, target)
                tloss += loss.item()

            print(f"\nTest accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
            return correct / len(test_loader.dataset), tloss / len(test_loader.dataset)

        def deactivate_ema(model):
            print("deactivate_ema")
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    print(m)
                    m.track_running_stats = False
                    m._saved_running_mean, m.running_mean = m.running_mean, None
                    m._saved_running_var, m.running_var = m.running_var, None

        def activate_ema(model):
            print("activate_ema")
            for m in model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    print(m)
                    m.track_running_stats = True
                    m.running_mean = m._saved_running_mean
                    m.running_var = m._saved_running_var

        def calculate_estimates(model, data_loader):
            with torch.no_grad():
                for X in tqdm(data_loader):
                    data = X['waveform'].to(device)

                    # apply transform with random parameter from a range
                    if args.transform == "pitchshift":
                        factors = torch.FloatTensor(2,1).uniform_(-10, 10) #for pitchshift
                    if args.transform == "speed":
                        factors = torch.FloatTensor(2,1).uniform_(0.1, 1.0) #for speed
                    if args.transform == None:
                        factors = torch.Tensor([1., 1.]).to(device) #for None
                    data = transform_audio(transform_type= args.transform, waveform=data, sample_rate=16000, factor=factors[0], device=device)

                    model(data)


        # The transform needs to live on the same device as the model and the data.
        # transform = transform.to(device)
        for epoch in range(args.epochs):
            print(f"=================\n Epoch: {epoch + 1} \n=================")
            deactivate_ema(model=model)
            train(model,fixed_param)
            activate_ema(model=model)
            calculate_estimates(model=model, data_loader=train_loader)
            test_acc, test_loss = validate(model, fixed_param)
        print("Done!")

        result_dict[str(fixed_param)] = test_acc

    ######## write to the bucket
    destination_name = f'{args.output}/{args.transform}/One4One/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)
    np.save(f'{destination_name}/acc.npy', pickle.dumps(result_dict))

    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()

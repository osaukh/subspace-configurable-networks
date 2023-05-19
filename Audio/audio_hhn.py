import os
import timeit
import argparse
import numpy as np
import models
import pickle
from tqdm import tqdm
import torch
from torch import nn
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
parser.add_argument('--arch', '-a', metavar='ARCH', default='scn_m5')
parser.add_argument('--nlayers', default=5, type=int)
parser.add_argument('--width', default=32, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--weightdecay', default=0.0001, type=float)
parser.add_argument('--dimensions', default=3, type=int)    # has no effect
parser.add_argument('--gpus', default=0, type=int)
parser.add_argument('--transform', default='pitchshift', type=str, help="pitchshift | speed | None")
parser.add_argument('--output', default='.', type=str)
args = parser.parse_args()


def main():
    start = timeit.default_timer()

    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    print(f'using device {device}')
    if not torch.cuda.is_available():
        print(f'cuda is not available. using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.gpus)

    train_ds = SpeechCommands(root="data", subset='training', device='cpu', in_mem=True)
    test_ds =  SpeechCommands(root="data", subset='testing', device='cpu', in_mem=True)

    print('Train dataset size: ', len(train_ds))
    print('Test dataset size: ', len(test_ds))
    print('Number of classes: ', len(train_ds.labels))

    train_loader = DataLoader(dataset=train_ds, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=args.batchsize, shuffle=True)

 
    ######## prepare model structure
    model = models.SCN_M5(num_alpha=1, dimensions=args.dimensions, device=device)
    save_dir = f"audio_{args.dimensions}"


    model.to(device)
    print(model)
    # print(utils.count_model_parameters(model))

    ######## train model
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weightdecay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def train(model):
        model.train()
        train_loss, correct = 0, 0
        for X in tqdm(train_loader, "Training"):
            data = X['waveform'].to(device)
            target = X['label'].to(device)

            # apply transform with random parameter from a range
            if args.transform == "pitchshift":
                factors = torch.FloatTensor(2,1).uniform_(-10, 10) #for pitchshift
            if args.transform == "speed":
                factors = torch.FloatTensor(2,1).uniform_(0.1, 1.0) #for speed
            if args.transform == None:
                factors = torch.Tensor([1., 1.]).to(device) #for None
            data = transform_audio(transform_type= args.transform, waveform=data, sample_rate=16000, factor=factors[0], device=device)

            output = model(data, hyper_x=factors[0].to(device))

            # loss = F.cross_entropy(output.squeeze(), target)
            loss = F.nll_loss(output.squeeze(), target)


            beta0 = model.hyper_stack(factors[0].to(device))
            beta1 = model.hyper_stack(factors[1].to(device))
            loss += pow(cos(beta0, beta1),2)

            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = get_likely_index(output)
            correct += number_of_correct(pred, target)
            train_loss += loss.item()     
        return correct / len(train_loader.dataset), train_loss / len(train_loader.dataset)

    def number_of_correct(pred, target):
        # count number of correct predictions
        return pred.squeeze().eq(target).sum().item()

    def get_likely_index(tensor):
        # find most likely label index for each element in the batch
        return tensor.argmax(dim=-1)

    def validate(model):
        model.eval()
        tloss, correct = 0, 0

        with torch.no_grad():

            for X in tqdm(test_loader, "Validating"):
                data = X['waveform'].to(device)
                target = X['label'].to(device)

                # apply transform with random parameter from a range
                if args.transform == "pitchshift":
                    factors = torch.FloatTensor(2,1).uniform_(-10, 10).to(device) #for pitchshift
                if args.transform == "speed":
                    factors = torch.FloatTensor(2,1).uniform_(0.1, 1).to(device) #for speed
                if args.transform == None:
                    factors = torch.Tensor([1., 1.]).to(device) #for None

                data = transform_audio(transform_type= args.transform, waveform=data, sample_rate=16000, factor=factors[0], device=device)

                output = model(data, hyper_x=factors[0])

                # loss = F.cross_entropy(output.squeeze(), target)
                loss = F.nll_loss(output.squeeze(), target)

                pred = get_likely_index(output)
                correct += number_of_correct(pred, target)
                tloss += loss.item()
        
            print(f"\nTest accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
            return correct / len(test_loader.dataset), tloss / len(test_loader.dataset)

    for epoch in range(args.epochs):
        print(f"=================\n Epoch: {epoch + 1} \n=================")
        train_acc, train_loss  = train(model)
        test_acc, test_loss = validate(model)
        scheduler.step()
    print("Done!")


    ######## test model
    def test(model, param):
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X in tqdm(test_loader, "Testing"):
                data = X['waveform'].to(device)
                target = X['label'].to(device)

                # apply transform with random parameter from a range
                data = transform_audio(transform_type= args.transform, waveform=data, sample_rate=16000, factor=param, device=device)

                output = model(data, hyper_x=param)

                pred = get_likely_index(output)
                # loss = F.cross_entropy(output.squeeze(), target)
                loss = F.nll_loss(output.squeeze(), target)


                correct += number_of_correct(pred, target)
                test_loss += loss.item()

            print(f"\nTest accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")
            return correct / len(test_loader.dataset), test_loss / len(test_loader.dataset)

    acc = []
    if args.transform == "pitchshift":
        for param in tqdm(torch.unsqueeze(torch.arange(-10, 10, 1), 1).float().to(device), desc='Testing'):
            acc.append(test(model, param))
    if args.transform == "speed":
        for param in tqdm(torch.unsqueeze(torch.arange(0.1, 1.0, 0.1), 1).float().to(device), desc='Testing'):
            acc.append(test(model, param))
    if args.transform == None:
        for param in tqdm(torch.tensor([1.,]).to(device), desc='Testing'):
            acc.append(test(model, param))

    ######## write to the bucket
    destination_name = f'{args.output}/{args.transform}/HHN/{save_dir}'
    os.makedirs(destination_name, exist_ok=True)
    np.save(f'{destination_name}/acc.npy', pickle.dumps(acc))

    filename = f'{destination_name}/checkpoint.pth.tar'           
    torch.save(
        {
            'epoch': args.epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler':scheduler.state_dict()},           
            filename 
    )
    print(f'Models are saved in {filename}')
    stop = timeit.default_timer()
    print('Time: ', stop - start)


if __name__ == '__main__':
    main()

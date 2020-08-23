from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from model.colornet import Colornet
from data.colordataset import Colordataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch_sample in enumerate(train_loader):
        data = batch_sample['data']
        target = batch_sample['label']
 
        data, target = data.to(device).float(), target.to(device).float()

        optimizer.zero_grad()
        output = model(data)

        MSEloss = nn.MSELoss()
        loss = MSEloss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, save_output=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(test_loader):
            data = batch_sample['data']
            target = batch_sample['label']
 
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data)

            if save_output == True:
                data = data.cpu()
                target = target.cpu()
                output = output.cpu()

                out_file = open("output.txt", "w")
                np.savetxt(out_file, target, fmt="%4f", delimiter=" ", newline="\n")
                out_file.close()

                fig = plt.figure()
                ax = plt.axes(projection='3d')

                x = target[:,0]
                y = target[:,1]
                z = target[:,2]
                ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5);
                print(output)
                x = output[:,0]
                y = output[:,1]
                z = output[:,2]
                ax.scatter(x, y, z, c=z, cmap='turbo_r', linewidth=0.5);
                plt.show()
            

            MSEloss = nn.MSELoss()
            test_loss += MSEloss(output, target).item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.8f}\n'.format(test_loss))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--load_model', type=bool, default=False,
                        help='For Loading the current Model')
    parser.add_argument('--mode', type=str, default='train',
                        help='train / eval')
    parser.add_argument('--data_ratio', type=float, default=0.9,
                        help='For Loading the current Model')


    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")


    total_set = Colordataset(file_data = './data/xyY.txt', file_label= './data/Lab.txt')
    total_len = total_set.__len__()
    train_len = int(total_len * args.data_ratio)
    test_len = total_len - train_len
    train_set, test_set = torch.utils.data.random_split(total_set, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=test_len, shuffle=False)
    eval_loader = DataLoader(total_set, batch_size=total_len, shuffle=False)


    model = Colornet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=args.gamma)

    if args.load_model == True:
        model.load_state_dict(torch.load("color1.pt"))

    if args.mode == 'train':
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "color1.pt")

        test(model, device, eval_loader, save_output=True)

    elif args.mode == 'eval':
            test(model, device, eval_loader, save_output=True)


if __name__ == '__main__':
    main()

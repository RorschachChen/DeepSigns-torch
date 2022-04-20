import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import torchvision
from torchvision import transforms
from utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
from models.resnet import ResNet18


def run(args):
    device = torch.device('cuda')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    # ---- WM configs ------ #
    # binary prior info to be embedded, shape (T, 10)
    b = np.random.randint(2, size=(args.embed_bits, args.n_classes))
    # ---- Embed WM ------ #
    model = ResNet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for ep in range(args.epochs):
        for d, t in tqdm(trainloader):
            d = d.to(device)
            t = t.to(device)
            optimizer.zero_grad()
            pred = model(d)
            loss = criterion(pred, t)
            loss.backward()
            optimizer.step()

    model.eval()
    loss_meter = 0
    acc_meter = 0
    with torch.no_grad():
        for d, t in tqdm(testloader):
            data = d.to(device)
            target = t.to(device)
            pred = model(data)
            loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
            pred = pred.max(1, keepdim=True)[1]
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
    print('Test loss:', loss_meter)
    print('Test accuracy:', acc_meter / len(testloader.dataset))
    torch.save(model.state_dict(), 'logs/whitebox/marked/resnet18.pth')

    # ---- Validate WM ---- #
    marked_model = ResNet18()
    marked_model.load_state_dict(torch.load('logs/whitebox/marked/resnet18.pth'))
    x_train_subset, y_train_subset = subsample_training_data(trainset, args.target_class)
    marked_activations = get_activations(marked_model, x_train_subset)
    print("Get activations of marked FC layer")
    # choose the activations from first wmarked dense layer
    marked_FC_activations = marked_activations[args.target_dense_idx + 1]
    A = np.load('logs/whitebox/projection_matrix.npy')
    print('A = ', A)
    decoded_WM = extract_WM_from_activations(marked_FC_activations, A)
    BER = compute_BER(decoded_WM, b[:, args.target_class])
    print("BER in class {} is {}: ".format(args.target_class, BER))


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in data')
    parser.add_argument('--key_len', type=int, default=20,
                        help='Length of key')
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--th', default=0.1, type=float, help='p_threshold')
    parser.add_argument('--epochs', default=2, type=int, help='embed_epoch')
    parser.add_argument('--scale', default=0.01, type=float, help='for loss1')
    parser.add_argument('--gamma', default=0.01, type=float, help='for loss2')
    parser.add_argument('--target_dense_idx', default=2, type=int, help='target layer to carry WM')
    parser.add_argument('--embed_bits', default=16, type=int)
    parser.add_argument('--target_class', default=0, type=int)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

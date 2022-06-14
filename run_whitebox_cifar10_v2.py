import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torchvision
from torchvision import transforms
from torchsummary import summary

from models.resnet import ResNet18
from utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))


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
    np.save('logs/whitebox/resnet18/marked/b.npy', b)
    # ---- Embed WM ------ #
    model = ResNet18().to(device)
    centers = torch.nn.Parameter(torch.rand(args.n_classes, 512).to(device), requires_grad=True)
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
        {'params': centers}
    ], lr=args.lr,
        momentum=0.9, weight_decay=5e-4)

    train_whitebox(model, optimizer, trainloader, b, centers, args, save_path='./logs/whitebox/resnet18/marked/projection_matrix.npy')

    model.eval()
    loss_meter = 0
    acc_meter = 0
    with torch.no_grad():
        for d, t in testloader:
            data = d.to(device)
            target = t.to(device)
            pred, _ = model(data)
            loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
            pred = pred.max(1, keepdim=True)[1]
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
    print('Test loss:', loss_meter / len(testloader.dataset))
    print('Test accuracy:', acc_meter / len(testloader.dataset))
    sd_path = 'logs/whitebox/resnet18/marked/resnet18.pth'
    torch.save(model.state_dict(), sd_path)

    # ---- Validate WM ---- #
    marked_model = ResNet18().to(device)
    # summary(marked_model, input_size=(1, 28, 28))
    marked_model.load_state_dict(torch.load(sd_path))
    x_train_subset_loader = subsample_training_data(trainset, args.target_class)
    marked_activations = get_activations(marked_model, x_train_subset_loader)
    print("Get activations of marked FC layer")
    # choose the activations from first wmarked dense layer
    marked_FC_activations = marked_activations
    A = np.load('logs/whitebox/resnet18/marked/projection_matrix.npy')
    print('A = ', A)
    decoded_WM = extract_WM_from_activations(marked_FC_activations, A)
    BER = compute_BER(decoded_WM, b[:, args.target_class])
    print("BER in class {} is {}: ".format(args.target_class, BER))


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in data')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=50, type=int, help='embed_epoch')
    parser.add_argument('--scale', default=0.01, type=float, help='for loss1')
    parser.add_argument('--gamma', default=0.01, type=float, help='for loss2')
    parser.add_argument('--target_dense_idx', default=2, type=int, help='target layer to carry WM')
    parser.add_argument('--embed_bits', default=16, type=int)
    parser.add_argument('--target_class', default=0, type=int)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

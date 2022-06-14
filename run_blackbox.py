import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
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
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    # ---- Embed WM ------ #
    model = ResNet18().to(device)
    model.load_state_dict(torch.load('logs/blackbox/ummarked/resnet18.pth'))
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0, nesterov=True)
    x_key, y_key = key_generation(model, optimizer, trainset, args.key_len, 32, args.n_classes, args.epochs)
    key_data = torch.utils.data.TensorDataset(x_key, y_key)
    key_loader = DataLoader(key_data,
                            batch_size=128,
                            shuffle=False,
                            num_workers=2)

    # ----- Detect WM ------ #
    marked_model = ResNet18().to(device)
    marked_model.load_state_dict(torch.load('logs/blackbox/ummarked/resnet18.pth'))
    acc_meter = 0
    with torch.no_grad():
        for load in key_loader:
            data, target = load[:2]
            data = data.to(device)
            target = target.to(device)
            pred, _ = marked_model(data)
            pred = pred.max(1, keepdim=True)[1]
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
    theta = compute_mismatch_threshold(c=args.n_classes, kp=args.key_len, p=args.th)  # pk = 1/C, |K|: # trials
    print('probability threshold p is ', args.th)
    print('Mismatch threshold is : ', theta)
    print('Mismatch count of marked model on WM key set = ', acc_meter)
    print("If the marked model is correctly authenticated by owner: ", acc_meter < theta)


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in data')
    parser.add_argument('--key_len', type=int, default=20,
                        help='Length of key')
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--th', default=0.1, type=float, help='p_threshold')
    parser.add_argument('--epochs', default=2, type=int, help='embed_epoch')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

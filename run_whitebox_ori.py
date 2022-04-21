import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torchvision
from torchvision import transforms
from torchsummary import summary
from models.mlp import MLP
from utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))


def run(args):
    device = torch.device('cuda')
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./data/",
                                          transform=transform,
                                          train=True,
                                          download=True)

    data_test = torchvision.datasets.MNIST(root="./data/",
                                           transform=transform,
                                           train=False)

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=32,
                                              shuffle=False)

    testloader = torch.utils.data.DataLoader(dataset=data_test,
                                             batch_size=512,
                                             shuffle=False)

    # ---- WM configs ------ #
    # binary prior info to be embedded, shape (T, 10)
    b = np.random.randint(2, size=(args.embed_bits, args.n_classes))
    # ---- Embed WM ------ #
    model = MLP().to(device)
    centers = torch.nn.Parameter(torch.randn(args.n_classes, 512).to(device), requires_grad=True)
    optimizer = torch.optim.RMSprop([
        {'params': model.parameters()},
        {'params': centers}
    ], alpha=0.9, lr=args.lr, eps=1e-8,
        weight_decay=0.001)
    train_whitebox(model, optimizer, trainloader, b, centers, args)

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
    torch.save(model.state_dict(), 'logs/whitebox/marked/mlp.pth')

    # ---- Validate WM ---- #
    marked_model = MLP().to(device)
    summary(marked_model, input_size=(1, 28, 28))
    marked_model.load_state_dict(torch.load('logs/whitebox/marked/mlp.pth'))
    x_train_subset_loader = subsample_training_data(trainset, args.target_class)
    marked_activations = get_activations(marked_model, x_train_subset_loader)
    print("Get activations of marked FC layer")
    # choose the activations from first wmarked dense layer
    marked_FC_activations = marked_activations[0]
    A = np.load('logs/whitebox/projection_matrix.npy')
    print('A = ', A)
    decoded_WM = extract_WM_from_activations(marked_FC_activations, A)
    BER = compute_BER(decoded_WM, b[:, args.target_class])
    print("BER in class {} is {}: ".format(args.target_class, BER))


def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in data')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=1, type=int, help='embed_epoch')
    parser.add_argument('--scale', default=0.01, type=float, help='for loss1')
    parser.add_argument('--gamma', default=0.01, type=float, help='for loss2')
    parser.add_argument('--target_dense_idx', default=2, type=int, help='target layer to carry WM')
    parser.add_argument('--embed_bits', default=16, type=int)
    parser.add_argument('--target_class', default=0, type=int)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

import os
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torchvision
from torchvision import transforms, datasets
from utils import *
from torchvision.datasets import ImageFolder
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
from models.resnet import ResNet18
from torch.utils.data import DataLoader,Dataset

class ImageDataset(Dataset):
    def __init__(self, csv, img_folder, transform):
        self.labels = label_csv
        self.transform = transform
        self.img_folder = img_folder

        self.image_names = self.labels
        self.labels = np.array(self.csv.drop(['Id', 'Genre'], axis=1))

    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(self.img_folder + self.image_names.iloc[index] + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)
        targets = self.labels[index]

        sample = {'image': image, 'labels': targets}

        return sample
def run(args):
    device = torch.device('cpu')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # trainset = datasets.MNIST(
    #     root='./data', train=True, download=True, transform=transform_train)
    train_y = pd.read_csv('~/Downloads/trainLabels.csv')
    train_x = ImageFolder('~/Downloads/', transform=transform_train)

    trainset = torch.utils.data.TensorDataset(train_x[:1000], train_y[:1000])
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
    marked_model.load_state_dict(torch.load('logs/blackbox/resnet18.pth'))
    acc_meter = 0
    total_samples = 0
    with torch.no_grad():
        for load in key_loader:
            data, target = load[:2]
            data = data.to(device)
            target = target.to(device)
            pred, _ = marked_model(data)
            pred = pred.max(1, keepdim=True)[1]
            total_samples +=  len(pred)
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
    # theta = total_samples * args.th
    # acc_meter = total_samples - acc_meter
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

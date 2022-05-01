import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import comb
from torch.utils.data import ConcatDataset, DataLoader, Subset


def key_generation(marked_model, optimizer, original_data, desired_key_len, img_size=32, num_classes=10,
                   embed_epoch=20):
    key_len = 40 * desired_key_len
    batch_size = 1024
    key_gen_flag = 1
    while key_gen_flag:
        x_retrain_rand = torch.randn(key_len, 3, img_size, img_size)
        y_retrain_rand_vec = torch.randint(num_classes, size=[key_len])
        retrain_rand_data = torch.utils.data.TensorDataset(x_retrain_rand, y_retrain_rand_vec)
        retrain_rand_loader = DataLoader(retrain_rand_data,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=2)
        _, err_idx, _ = test(marked_model, retrain_rand_loader)
        retrain_data = ConcatDataset([original_data, retrain_rand_data])
        retrain_loader = DataLoader(retrain_data,
                                    batch_size=batch_size,
                                    shuffle=False)
        fine_tune(marked_model, optimizer, retrain_loader, embed_epoch)
        _, _, correct_idx = test(marked_model, retrain_rand_loader)
        selected_key_idx = np.intersect1d(err_idx, correct_idx)
        selected_keys = x_retrain_rand[np.array(selected_key_idx).astype(int), :]
        selected_keys_labels = y_retrain_rand_vec[np.array(selected_key_idx).astype(int)]
        usable_key_len = selected_keys.shape[0]
        print('usable key len is: ', usable_key_len)
        if usable_key_len < desired_key_len:
            key_gen_flag = 1
            print(' Desire key length is {}, Need longer key, skip this test. '.format(desired_key_len))
        else:
            key_gen_flag = 0
            selected_keys = selected_keys[0:desired_key_len, :]
            selected_keys_labels = selected_keys_labels[0:desired_key_len]
            np.save('logs/blackbox/keyRandomImage' + '_keyLength' + str(desired_key_len) + '.npy', selected_keys)
            np.savetxt('logs/blackbox/keyRandomLabel' + '_keyLength' + str() + '.txt', selected_keys_labels, fmt='%i',
                       delimiter=',')
            torch.save(marked_model.state_dict(), f'logs/blackbox/marked/resnet18.pth')
            print('WM key generation finished. Save watermarked model. ')
    return selected_keys, selected_keys_labels


def fine_tune(model, optimizer, dataloader, epochs):
    model.train()
    device = next(model.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        for d, t in dataloader:
            d = d.to(device)
            t = t.to(device)
            optimizer.zero_grad()
            pred = model(d)
            loss = criterion(pred, t)
            loss.backward()
            optimizer.step()


def test(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    loss_meter = 0
    err_idx = []
    correct_idx = []
    runcount = 0
    with torch.no_grad():
        for load in dataloader:
            data, target = load[:2]
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
            pred = pred.max(1, keepdim=True)[1]
            correct_idx += (pred.view_as(target) == target).nonzero(as_tuple=True)[0].cpu() + runcount
            err_idx += (pred.view_as(target) != target).nonzero(as_tuple=True)[0].cpu() + runcount
            runcount += data.size(0)
    return loss_meter, err_idx, correct_idx


def compute_mismatch_threshold(c=10, kp=50, p=0.05):
    prob_sum = 0
    p_err = 1 - 1.0 / c
    theta = 0
    for i in range(kp):
        cur_prob = comb(kp, i, exact=False) * np.power(p_err, i) * np.power(1 - p_err, kp - i)
        prob_sum = prob_sum + cur_prob
        if prob_sum > p:
            theta = i
            break
    return theta


def extract_WM_from_activations(activs, A):
    activ_classK = activs
    activ_centerK = np.mean(activ_classK, axis=0)
    activ_centerK = np.reshape(activ_centerK, (-1, 1))
    X_Ck = np.dot(A, activ_centerK)
    X_Ck_sigmoid = 1 / (1 + np.exp(-X_Ck))
    decode_wmark = (X_Ck_sigmoid > 0.5) * 1
    return decode_wmark


def get_activations(model, input_loader):
    activations = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for d, t in input_loader:
            d = d.to(device)
            _, feat = model(d)
            activations.extend(feat.detach().cpu().numpy())
    return np.stack(activations, 0)


def compute_BER(decode_wmark, b_classK):
    b_classK = np.reshape(b_classK, (-1, 1))
    diff = np.abs(decode_wmark - b_classK)
    BER = np.sum(diff) / b_classK.size
    return BER


def subsample_training_data(dataset, target_class):
    train_indices = (torch.tensor(dataset.targets) == target_class).nonzero().reshape(-1)
    subsample_len = int(np.floor(0.5 * len(train_indices)))
    subset_idx = np.random.randint(train_indices.shape[0], size=subsample_len)
    train_subset = Subset(dataset, train_indices[subset_idx])
    dataloader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=False)
    return dataloader


def train_whitebox(model, optimizer, dataloader, b, centers, args, pm_path, feat_length=512):
    model.train()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    device = next(model.parameters()).device
    x_value = np.random.randn(args.embed_bits, feat_length)
    np.save(pm_path, x_value)
    x_value = torch.tensor(x_value, dtype=torch.float32).to(device)
    b = torch.tensor(b).to(device)
    for ep in range(args.epochs):
        print(f'epochs: {ep}')
        for d, t in dataloader:
            d = d.to(device)
            t = t.to(device)
            optimizer.zero_grad()
            pred, feat = model(d)
            loss = criterion(pred, t)
            centers_batch = torch.gather(centers, 0, t.unsqueeze(1).repeat(1, feat.shape[1]))
            loss1 = F.mse_loss(feat, centers_batch, reduction='sum') / 2
            centers_batch_reshape = torch.unsqueeze(centers_batch, 1)
            centers_reshape = torch.unsqueeze(centers, 0)
            pairwise_dists = (centers_batch_reshape - centers_reshape) ** 2
            pairwise_dists = torch.sum(pairwise_dists, dim=-1)
            arg = torch.topk(-pairwise_dists, k=2)[1]
            arg = arg[:, -1]
            closest_cents = torch.gather(centers, 0, arg.unsqueeze(1).repeat(1, feat.shape[1]))
            dists = torch.sum((centers_batch - closest_cents) ** 2, dim=-1)
            cosines = torch.mul(closest_cents, centers_batch)
            cosines = torch.sum(cosines, dim=-1)
            loss2 = (cosines * dists - dists).mean()
            loss3 = (1 - torch.sum(centers ** 2, dim=1)).abs().sum()
            loss4 = 0
            embed_center_idx = args.target_class
            idx_classK = (t == embed_center_idx).nonzero(as_tuple=True)
            if len(idx_classK[0]) >= 1:
                idx_classK = idx_classK[0]
                activ_classK = torch.gather(centers_batch, 0,
                                            idx_classK.unsqueeze(1).repeat(1, feat.shape[1]))
                center_classK = torch.mean(activ_classK, dim=0)
                Xc = torch.matmul(x_value, center_classK)
                bk = b[:, embed_center_idx]
                bk_float = bk.float()
                probs = torch.sigmoid(Xc)
                entropy_tensor = F.binary_cross_entropy(target=bk_float, input=probs, reduce=False)
                loss4 += entropy_tensor.sum()
            (loss + args.scale * (loss1 + loss2 + loss3) + args.gamma * loss4).backward()
            # loss.backward()
            optimizer.step()

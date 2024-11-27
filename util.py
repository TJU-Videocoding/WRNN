
import datetime
import glob
import os
import re
from fileinput import filename
from turtle import forward

import cv2
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset


import os
import openpyxl
import cv2
import numpy as np
import torch.utils.data.dataset

from torch.utils.data import DataLoader


def parse_images(data_root, mode):
    excel_dir = os.path.join(data_root, 'excel', mode)
    excel_list = os.listdir(excel_dir)
    image_info = []
    new_k = []
    for file in excel_list:
        excel = openpyxl.load_workbook(os.path.join(data_root, 'excel', mode, file))
        sheet = excel['data']
        for row in sheet.values:
            poc, ctu_addr, C, K, R1, R2, R3, R4, R5, D1, D2, D3, D4, D5 = row
            if C <= 0.8 or C >= 3200:
                continue
            if K <= -3 or K >= 0:
                continue
            # para = (np.log(C), K)
            para = K
            new_k.append(para)
            sample = (R1, R2, R3, R4, R5, D1, D2, D3, D4, D5)
            box = ((ctu_addr % 2) * 128, (ctu_addr // 2) * 128, 128, 128)
            image_dir = os.path.join(data_root, mode, file.split('.')[0], '%04d.png' % (poc+1))
            item = (image_dir, box, para, sample)

            image_info.append(item)

    arr_mean = np.mean(new_k)
    arr_var = np.var(new_k)
    print('Image Parse Completed')
    return image_info, arr_mean, arr_var


class CIFAR100Train(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, data_root='/mnt/yf/pred/data/dataset', mode='train', transform=None): 
        self.data_root = data_root
        self.mode = mode
        self.image_info, self.mean, self.std = parse_images(self.data_root, self.mode)

        self.transform = transform

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        (
            image_dir,
            box,
            para,
            sample
        ) = self.image_info[index]
        try:
            img = cv2.imread(image_dir)
            img_ = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]
            img_crop = cv2.cvtColor(img_, cv2.COLOR_BGR2YUV)
            # img_crop = T.to_mytensor(img_crop)
            img_crop = img_crop / 255.

        except Exception as e:
            print("problem in, ", image_dir)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_info)))

        if self.transform:
            image = self.transform(img_crop)

        # para = (para - self.mean) / self.std
        return para, image.float()


class CIFAR100Test(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, data_root='/mnt/yf/pred/data/dataset', mode='val', transform=None): 
        self.data_root = data_root
        self.mode = mode
        self.image_info, self.mean, self.std = parse_images(self.data_root, self.mode)

        self.transform = transform

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        (
            image_dir,
            box,
            para,
            sample
        ) = self.image_info[index]
        try:
            img = cv2.imread(image_dir)
            img_ = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2], :]
            img_crop = cv2.cvtColor(img_, cv2.COLOR_BGR2YUV)
            # img_crop = T.to_mytensor(img_crop)
            img_crop = img_crop / 255.

        except Exception as e:
            print("problem in, ", image_dir)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_info)))

        if self.transform:
            image = self.transform(img_crop)
        # para = (para - self.mean) / self.std
        return para, image.float()


class PredictModel(nn.Module):
    def __init__(self) -> None:
        super(PredictModel, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=True)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, img):
        result = self.net(img)
        result = self.fc2(result)
        result = self.fc3(result)
        return result


def get_training_dataloader(args,  batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.height, args.weight)),
        transforms.RandomRotation(30),
        transforms.RandomCrop((224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomErasing(p=1, scale=(
            0.0002, 0.0005), ratio=(0.3, 3.3)),
        transforms.RandomAdjustSharpness(7, p=0.5),
    ])
    cifar100_training = CIFAR100Train(transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader


def get_test_dataloader(args, batch_size=16, num_workers=2):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.height, args.weight)),
        # transforms.RandomCrop((min(args.height, args.weight))),
    ])
    cifar100_test = CIFAR100Test(transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=False, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(
        os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(
        re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(
        regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(
        re.search(regex_str, w).groups()[1]))
    return best_files[-1]

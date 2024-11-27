import os

import numpy
import openpyxl
import numpy as np
import torch.utils.data.dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from functools import partial


def parse_classification(excel_root, data_root):
    image_info = []
    excel = openpyxl.load_workbook(os.path.join(excel_root))
    sheet = excel['data']
    for row in sheet.values:
        seq_name, poc, ctu_addr, bpp = row
        if bpp is None:
            continue
        yuv_dir = seq_name + '_' + str(poc) + '_' + str(ctu_addr) + '.yuv'
        image_dir = os.path.join(data_root, yuv_dir)
        item = (image_dir, bpp)
        image_info.append(item)
    print('Image Parse Completed, total %d samples' % len(image_info))
    return image_info


class HDRDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            excel_root,
            data_root
    ):
        self.data_root = data_root
        self.excel_root = excel_root
        self.image_info = parse_classification(self.excel_root, self.data_root)

    def __getitem__(self, index):
        (
            image_dir,
            bpp
        ) = self.image_info[index]
        try:
            img = readYUV(image_dir)
            img = img / 1023.0
            outputs = [
                img,
                bpp
            ]
        except Exception as e:
            print("problem in, ", image_dir)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_info)))
        return outputs

    def __len__(self):
        return len(self.image_info)


def readYUV(image_dir):
    height, width = 128, 128
    fp = open(image_dir, 'rb')
    h_h = height // 2
    h_w = width // 2
    bytes2num = partial(int.from_bytes, byteorder='little', signed=False)
    Yt = np.zeros(shape=(height, width), dtype='uint16', order='C')
    Ut = np.zeros(shape=(h_h, h_w), dtype='uint16', order='C')
    Vt = np.zeros(shape=(h_h, h_w), dtype='uint16', order='C')
    YY = np.zeros_like(Yt, dtype='uint16')
    UU = np.zeros_like(Yt, dtype='uint16')
    VV = np.zeros_like(Yt, dtype='uint16')
    YUV = np.zeros(shape=(height, width, 3), dtype='uint16')

    for m in range(height):
        for n in range(width):
            Yt[m, n] = bytes2num(fp.read(2))
    for m in range(h_h):
        for n in range(h_w):
            Ut[m, n] = bytes2num(fp.read(2))
    for m in range(h_h):
        for n in range(h_w):
            Vt[m, n] = bytes2num(fp.read(2))

    UU[0: height - 1: 2, 0: width - 1: 2] = Ut[:, :]
    UU[0: height - 1: 2, 1: width: 2] = Ut[:, :]
    UU[1: height: 2, 0: width - 1: 2] = Ut[:, :]
    UU[1: height: 2, 1: width: 2] = Ut[:, :]
    VV[0: height - 1: 2, 0: width - 1: 2] = Vt[:, :]
    VV[0: height - 1: 2, 1: width: 2] = Vt[:, :]
    VV[1: height: 2, 0: width - 1: 2] = Vt[:, :]
    VV[1: height: 2, 1: width: 2] = Vt[:, :]
    YUV[:, :, 0] = Yt
    YUV[:, :, 1] = UU
    YUV[:, :, 2] = VV
    return YUV


# val_data = HDRDataset('H:/Yuan/HDR/data/label/train_label.xlsx', 'H:/Yuan/HDR/data/train_set')
# img, para = val_data[0]
# print(img[:, :, 0])

# from tqdm import tqdm
#
# train_data = HDRDataset('/data1/YF/Classification/Label/train_label.xlsx',
#                         '/data1/YF/Classification/Train/train_set/')
# data_loader = DataLoader(train_data, batch_size=1, shuffle=False)
# iterator = tqdm(data_loader, maxinterval=100,
#                 mininterval=1, ncols=100,
#                 bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]',
#                 smoothing=0.01)
# for i, data in enumerate(iterator):
#     pass


def parse_hdr_test(data_root):
    image_info = []
    tmp_info = []
    for filename in os.listdir(data_root):
        seq_info = filename.split('_')
        patch_info = int(seq_info[-1].split('.')[0]), seq_info[-1].split('.')[1]
        seq_name = [seq_info[0], int(seq_info[1]), patch_info[0], patch_info[1]]
        tmp_info.append(seq_name)
    sort_list = sorted(tmp_info, key=(lambda x: [x[0], x[1], x[2]]))
    for i in sort_list:
        item = os.path.join(data_root, i[0] + '_' + str(i[1]) + '_' + str(i[2]) + '.' + str(i[3]))
        image_info.append(item)
    print('Image Parse Completed, total %d samples' % len(image_info))
    return image_info


# parse_hdr_test('/data0/Yuan/HDR_data/test/test_set/')


class HDRTestDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root
    ):
        self.data_root = data_root
        self.image_info = parse_hdr_test(self.data_root)

    def __getitem__(self, index):
        image_dir = self.image_info[index]
        try:
            img = readYUV(image_dir)
            img = img / 1023.0
            outputs = [
                image_dir,
                img
            ]
        except Exception as e:
            print("problem in, ", image_dir)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_info)))
        return outputs

    def __len__(self):
        return len(self.image_info)


# test_data = HDRTestDataset('/data0/Yuan/HDR_data/test/test_set/')
# for i in test_data:
#     print(i[0])


import time
import torch
from torch import nn
from hdr_data import HDRDataset
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import loss
import torch.optim as optim
from util import WarmUpLR
from network import MFCNN_C


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

writer = SummaryWriter(log_dir='tensorboard/TIP/HDR_YUV_K')

# data
train_data = HDRDataset('/data/YF/RC/data/label/train_label.xlsx',
                        '/data/YF/RC/data/Train/train_set/')
data_loader = DataLoader(train_data, batch_size=128, num_workers=8, shuffle=True)
test_data = HDRDataset('/data/YF/RC/data/label/val_label.xlsx',
                       '/data/YF/RC/data/Validation/val_set/')
test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)
print(len(test_data_loader))

# network
net = MFCNN_C(32, 2).cuda()

# loss
loss_l1 = nn.SmoothL1Loss().cuda()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
train_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 200], gamma=0.2)  # learning rate decay
iter_per_epoch = len(test_data_loader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)

best_acc = 10000

# train
for epoch in range(300):

    RD_loss = []
    iterator = tqdm(data_loader, maxinterval=100,
                    mininterval=1, ncols=100,
                    bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]',
                    smoothing=0.01)

    for i, data in enumerate(iterator):
        net.train()
        (
            img,
            para
        ) = data
        optimizer.zero_grad()

        img = img.cuda()
        C, K = para[0].cuda(), para[1].cuda()

        pred_K = net.forward(img.float().permute(0, 3, 1, 2))
        loss = loss_l1(pred_K.float(), K.unsqueeze(1).float())
        RD_loss.append(loss.item())

        loss.backward()
        optimizer.step()
        if epoch <= 1:
            warmup_scheduler.step()
        train_scheduler.step(epoch)

        n_iter = (epoch - 1) * len(data_loader) + i + 1

        iterator.set_description('epoch %d' % epoch)
        iterator.set_postfix_str('loss={:^7.3f}'.format(loss))

        writer.add_scalar('train_loss', loss.item(), n_iter)

    print('epoch:', epoch, ' label:', K.unsqueeze(1).float(), ' pred:', pred_K.float())

    with open('logs/TIP/log_HDR_YUV_K.txt', 'a') as f:
        f.write("train | epoch: {}, train_loss:{}\n".format(epoch, np.mean(RD_loss)))

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    # validation
    test_loss = 0.0
    max_error = []
    mean_error = []
    MAPE = []
    pred = []
    label = []

    if (epoch+1) % 1 == 0:
        start = time.time()
        net.eval()
        for i, (img, para) in enumerate(test_data_loader):
            with torch.no_grad():
                img = img.cuda()
                C, K = para[0].cuda(), para[1].cuda()

                pred_K = net.forward(img.float().permute(0, 3, 1, 2))
                loss = loss_l1(pred_K.float(), K.unsqueeze(1).float())
                test_loss += loss.item()

                print('index:', i, ' label:', K.unsqueeze(1).float(), ' pred:', pred_K.float())
                diff = abs(pred_K.float() - K.unsqueeze(1).float())

        finish = time.time()

        writer.add_scalar('Test/Average loss', test_loss / len(test_data_loader), epoch)
        acc = test_loss / len(test_data_loader)

        print('Evaluating Network.....')
        print('Test set: Epoch: {}, Average loss: {:.4f}, Mean Error: {:.4f}, Time consumed:{:.2f}s'.format(
                epoch,
                test_loss / len(test_data_loader),
                acc,
                finish - start
            ))

        print('Saving Model.....')
        torch.save(net, "./TIP/model_HDR_YUV_K_"+str(epoch)+".pkl")

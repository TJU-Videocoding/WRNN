import torch
from hdr_data import HDRTestDataset
from torch.utils.data import DataLoader
import time
import os

start = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


yuv_list = ['EBU_04_Hurdles', 'EBU_06_Starting', 'ShowGirl2', 'SunRise']
for yuv in yuv_list:
    yuv_dir = os.path.join('/data/YF/RC/data/test_set_total_seqs/test_set', yuv)
    frame_list = os.listdir(yuv_dir)
    frame_list = sorted(frame_list)
    for frame_idx in frame_list:
        frame_dir = os.path.join(yuv_dir, frame_idx)
        # test_data = HDRTestSeqs('/data1/YF/Test_Seqs')
        test_data = HDRTestDataset(frame_dir)
        data_loader = DataLoader(test_data, batch_size=1, shuffle=False)
        # net_bpp = torch.load('./model/Classification/model_HDR_YUV_Classification2_59.pkl')
        net_K = torch.load('/data/YF/RC/Code/TIP/model_HDR_YUV_K_14.pkl')
        net_C = torch.load('/data/YF/RC/Code/TIP/model_HDR_YUV_C_14.pkl')
        save_root = os.path.join('./results/total_seqs', yuv)

        if not os.path.exists(save_root):
            os.mkdir(save_root)
        save_dir = os.path.join(save_root, 'para%s.txt' % frame_idx)
        file = open(save_dir, 'w')

        # net_bpp.eval()
        net_K.eval()
        net_C.eval()

        acc_K = 0.0
        acc_C = 0.0

        for i, data in enumerate(data_loader):
            with torch.no_grad():
                (
                    img_dir,
                    img
                ) = data

                img = img.cuda()
                pred_K = net_K.forward(img.float().permute(0, 3, 1, 2))
                pred_C = net_C.forward(img.float().permute(0, 3, 1, 2))
                # pred_bpp = net_bpp.forward(img.float().permute(0, 3, 1, 2))
                file.write(str(abs(pred_C).cpu().numpy()[0][0]) + ' ' + str(pred_K.cpu().numpy()[0][0]) + '\n')
                # print(pred_bpp.cpu().numpy()[0])
                # file.write(img_dir[0] + ' ' + str(pred_bpp.cpu().numpy()[0]) + '\n')
        file.close()

        print(time.time()-start)


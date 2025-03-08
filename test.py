import torch
import torch.nn.functional as F
import sys


sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from data import test_dataset
from pamr import BinaryPamr
# from models.encoders.dual_vmamba import vssm_base as backbone
from SAM_Model.model import Model

print("GPU available:", torch.cuda.is_available())


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--model_type', type=str, default='vit_l', help='weight for edge loss')
parser.add_argument('--checkpoint', type=str, default='/home/xskwlz/pre_parameter/sam_vit_l_0b3195.pth',
                    help='test from checkpoints')
parser.add_argument('--test_path',type=str,default='/home/xskwlz/datasets/LFSOD/focal_stack/testset/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')

#load the model
model = Model(opt)
model.load_state_dict(torch.load(r'best/lfsod_epoch_best.pth'))


# test
model.cuda()
model.eval()

def run_pamr(img, sal):
    lbl_self = BinaryPamr(img, sal.clone().detach(), binary=0.4)
    return lbl_self


# test_datasets = ['DUTLF-FS','HFUT','LFSD']
# test_datasets = ['DUTLF-FS']
# test_datasets = ['HFUT']
test_datasets = ['LFSD']
for dataset in test_datasets:
    save_path = './test_maps/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/test_images/'
    gt_root = dataset_path + dataset + '/test_masks/'
    fs_root = dataset_path + dataset + '/test_focals/'
    test_loader = test_dataset(image_root, gt_root, fs_root, opt.testsize)
    for i in range(test_loader.size):
        #todo 位置
        image, focal, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        dim, height, width = focal.size()
        basize = 1
        focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
        focal = focal.view(-1, *focal.shape[2:])  # [basize*12, 6, 256, 256)
        focal = focal.cuda()
        image = image.cuda()
        res = model(focal, image, name)
        # res = run_pamr(image, res)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # pred = run_pamr(image, res)
        # print(pred.shape)
        # input()
        print('save img to: ', save_path+name)
        cv2.imwrite(save_path + name, res * 255)

    print('Test Done!')

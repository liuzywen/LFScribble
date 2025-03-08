import os
import random

import numpy
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import cv2
from PIL import ImageEnhance
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torch.nn as nn
import matplotlib.pyplot as plt


# rgbdirpath = r"D:\heqian\dataset\focal_stack\test_in_train\0000.jpg"
# focaldirpath = "D:\heqian\dataset\\focal_stack\\test_in_train\\0003.jpg"
# gtdirpath = r"D:\heqian\dataset\focal_stack\test_in_train\0002.jpg"


def cv_random_flip(img, label, scribble_gt, scribble_mask, grays, focal):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        scribble_gt = scribble_gt.transpose(Image.FLIP_LEFT_RIGHT)
        scribble_mask = scribble_mask.transpose(Image.FLIP_LEFT_RIGHT)
        grays = grays.transpose(Image.FLIP_LEFT_RIGHT)
        for i in range(focal.shape[2]):
            focal[:,:,i] = numpy.flip(focal[:,:,i],1)
        # img.save(rgbdirpath)
        # label.save(gtdirpath)
        # focal_view = focal[:, :, 0:3]
        # # print(focal_view.shape)
        # cv2.imwrite(focaldirpath, focal_view)
    return img, label,scribble_gt, scribble_mask, grays, focal


def randomCrop(image, label, scribble_gt, scribble_mask, grays, focal):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    W1 = (image_width - crop_win_width) >> 1
    H1 = (image_height - crop_win_height) >> 1
    W2 = (image_width + crop_win_width) >> 1
    H2 = (image_height + crop_win_height) >> 1
    random_region = (W1, H1, W2, H2)   #(2, 11, 253, 244) 与左边界的距离，与上边界的距离，与左边界的距离，与上边界的距离
    # print(random_region)
    # print(focal.shape)
    focal_crop = focal[H1:H2, W1:W2, :]
    image = image.crop(random_region)
    label = label.crop(random_region)
    scribble_gt = scribble_gt.crop(random_region)
    scribble_mask = scribble_mask.crop(random_region)
    grays = grays.crop(random_region)
    # print(focal_crop.shape)
    # print(image.size)
    # image.save(rgbdirpath)
    # # label.save(gtdirpath)
    # focal_view = focal_crop[:, :, 0:3]
    # # print(focal_view.shape)
    # cv2.imwrite(focaldirpath, focal_view)
    return image, label,scribble_gt, scribble_mask,grays, focal


def randomRotation(image, label, scribble_gt, scribble_mask,grays, focal):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        # random_angle = np.random.randint(-15, 15)
        angle = [90,180,270]
        random_angle = random.choice(angle)
        if random_angle == 90:
            m = 1
        elif random_angle == 180:
            m = 2
        else:
            m = 3
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        scribble_gt = scribble_gt.rotate(random_angle, mode)
        scribble_mask = scribble_mask.rotate(random_angle, mode)
        grays = grays.rotate(random_angle, mode)
        for i in range(focal.shape[2]):
            focal[:, :, i] = np.rot90(focal[:, :, i],m)
        # image.save(rgbdirpath)
        # label.save(gtdirpath)
        # focal_view = focal[:, :, 0:3]
        # cv2.imwrite(focaldirpath, focal_view)
    return image, label,scribble_gt, scribble_mask, grays, focal


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image



def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


def generate_unique_numbers(start, end, num):
    if end - start + 1 < num:
        raise ValueError("Range is not large enough to generate unique numbers.")

    result = []
    while len(result) < num:
        number = random.randint(start, end)
        if number not in result:
            result.append(number)

    return result

def generate_mask(img, ratio, mask_num):
    channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2]
    loss_mask = torch.ones(img_x, img_y)
    mask = torch.ones(img_x, img_y)
    # patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    patch_x, patch_y = int(img_x*ratio), int(img_y*ratio)
    w1 = np.random.randint(0, img_x - patch_x)
    h1 = np.random.randint(0, img_y - patch_y)
    # 获取选定的64x64矩阵
    block = mask[w1:w1 + patch_x, h1:h1 + patch_y]

    # 将块等分成16个16x16大小的小块
    blocks = np.split(block, 4, axis=0)
    blocks = [np.split(block, 4, axis=1) for block in blocks]

    if ratio == 0.25:
        # size = random.choice([4, 8, 12])
        size = 8
    else:
        # size = random.choice([16, 24])
        size = 16
    list = generate_unique_numbers(1, 16, mask_num)

    zero_blocks = []  # 用于存储中间值为0的小块位置
    block_x = patch_x//4
    block_y = patch_y//4
    # 遍历每个小块，并将中间的8x8部分的值置为0
    for i in range(4):
        for j in range(4):
            sub_block = blocks[i][j]
            sub_block[((block_x - size)//2):((block_y - size)//2) + size, ((block_x - size)//2):((block_y - size)//2) + size] = 0


    # 将修改后的块放回原矩阵中
    mask[w1:w1 + patch_x, h1:h1 + patch_y] = block

    # mask_show = mask.cpu().data.numpy()
    # plt.imshow(mask_show)
    # plt.show()

    return mask.long(), loss_mask.long(), zero_blocks



class SalObjDataset(data.Dataset):
    mean_rgb = np.array([0.485, 0.456, 0.406])
    std_rgb = np.array([0.229, 0.224, 0.225])
    # 将mean_rgb沿X轴复制12倍
    mean_focal = np.tile(mean_rgb, 12)
    # 将std_rgb沿X轴复制12倍
    std_focal = np.tile(std_rgb, 12)
    def __init__(self, image_root, gt_root, focal_root,scribble_gt_root, scribble_mask_root, gray_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.scribble_gt = [scribble_gt_root + f for f in os.listdir(scribble_gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.scribble_mask = [scribble_mask_root + f for f in os.listdir(scribble_mask_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.grays = [gray_root + f for f in os.listdir(gray_root) if f.endswith('.jpg')
                      or f.endswith('.png')]
        self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.scribble_gt = sorted(self.scribble_gt)
        self.scribble_mask = sorted(self.scribble_mask)
        self.grays = sorted(self.grays)
        self.gts = sorted(self.gts)
        self.focals = sorted(self.focals) # 排序
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
    #   return image, mask
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        focal = self.focal_loader(self.focals[index])
        scribble_gt = self.binary_loader(self.scribble_gt[index])
        scribble_mask = self.binary_loader(self.scribble_mask[index])
        grays = self.binary_loader(self.grays[index])


        image, gt, scribble_gt, scribble_mask, grays, focal = cv_random_flip(image, gt, scribble_gt, scribble_mask, grays, focal)
        image, gt, scribble_gt, scribble_mask, grays, focal = randomRotation(image, gt, scribble_gt, scribble_mask, grays, focal)
        image, gt, scribble_gt, scribble_mask, grays, focal = randomCrop(image, gt, scribble_gt, scribble_mask, grays, focal)

        # random augment
        # random_number = random.randint(1, 11)
        # image = strong_rgb_aug(image, random_number)

        image = colorEnhance(image)
        gt = randomPeper(gt)

        image = self.img_transform(image) #torch.Size([3, 256, 256])
        gt = self.gt_transform(gt) #torch.Size([1, 256, 256])
        scribble_gt = self.gt_transform(scribble_gt) #torch.Size([1, 256, 256])
        scribble_mask = self.gt_transform(scribble_mask) #torch.Size([1, 256, 256])
        grays = self.gt_transform(grays) #torch.Size([1, 256, 256])

        focal = np.array(focal, dtype=np.int32)
        if focal.shape[0] != 256:
            new_focal = []
            focal_num = focal.shape[2] // 3
            for i in range(focal_num):
                a = focal[:, :, i * 3:i * 3 + 3].astype(np.uint8)
                a = cv2.resize(a, (256, 256))
                new_focal.append(a)
            focal = np.concatenate(new_focal, axis=2)  # (256, 256, 36)

        # 将图片转化为numpy数组
        focal = focal.astype(np.float64)/255.0
        focal -= self.mean_focal
        focal /= self.std_focal
        focal = focal.transpose(2, 0, 1)
        # 把数组转换成张量，且二者共享内存
        focal = torch.from_numpy(focal).float() #torch.Size([36, 256, 256])

        return image, gt, focal, scribble_gt, scribble_mask, grays
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def focal_loader(self, path):
        with open(path, 'rb') as f:
            focal = sio.loadmat(f)
            focal = focal['img'] #上面函数返回的一键值对，选取键为’img‘的值给focal
            return focal

    def resize(self, img, gt, focal):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), focal
        else:
            return img, gt, focal

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, focal_root, scribble_gt_root, scribble_mask_root, gray_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True): #pin_memory设置为True可以将Tensor放入到内存的锁页区，加快训练速度

    dataset = SalObjDataset(image_root, gt_root, focal_root,scribble_gt_root, scribble_mask_root, gray_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    mean_rgb = np.array([0.485, 0.456, 0.406])
    std_rgb = np.array([0.229, 0.224, 0.225])
    mean_focal = np.tile(mean_rgb, 12)
    std_focal = np.tile(std_rgb, 12)
    def __init__(self, image_root, gt_root, focal_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.focals = [focal_root + f for f in os.listdir(focal_root) if f.endswith('.mat')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.focals = sorted(self.focals)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0
    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        focal = self.focal_loader(self.focals[self.index]) #focal.shape = (256, 256, 256)
        focal = np.array(focal, dtype=np.int32)
        if focal.shape[0] != 256:
            new_focal = []
            focal_num = focal.shape[2] // 3
            for i in range(focal_num):
                a = focal[:, :, i * 3:i * 3 + 3].astype(np.uint8)
                a = cv2.resize(a, (256, 256))
                new_focal.append(a)
            focal = np.concatenate(new_focal, axis=2)


        focal = focal.astype(np.float64)/255.0
        focal -= self.mean_focal
        focal /= self.std_focal
        focal = focal.transpose(2, 0, 1)
        focal = torch.from_numpy(focal).float()


        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, focal, gt, name



    def focal_loader(self, path):
        with open(path, 'rb') as f:
            focal = sio.loadmat(f)
            focal = focal['img']
            return focal

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')



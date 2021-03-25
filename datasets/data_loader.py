from __future__ import print_function, absolute_import

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import jpeg4py 
# from datasets.class_exchange_channel import exchange_channel
# from datasets.class_exchange_channel import start_exchange_channel

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            # img = jpeg4py.JPEG(img_path).decode()
            # img = Image.fromarray(img)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img # 需要查看此处的格式！！！

def channel_exchange1(img):
    img = np.array(img)
    # 0.RGB_img
    RGB_img = img
    # h,w,c->c,h,w
    img = img.transpose(2,0,1)
    # print(img.shape)

    # 取出每个channel
    R_channel = img[0]
    G_channel = img[1]
    B_channel = img[2]

    # 可以以某种概率随机打乱channel list# return 为None这是比较坑
    channel_list = [R_channel, G_channel, B_channel]
    # np.random.shuffle(channel_list) # return 为None这是比较坑的，如何解决
    channel_list1 = np.random.permutation(channel_list)
    final_img = np.transpose(channel_list1, (1,2,0))
    # matplotlib.image.imsave('./final_img.jpg', final_img)

    # 将final_img转换成PIL image
    final_img = Image.fromarray(final_img)

    return final_img # numpy格式的图像，shape：[128, 64, 3]

class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img, pid, camid = self.dataset[item]
        img = read_image(img) # 得到RGB图像
        if self.transform is not None:
            img = self.transform(img)

        return img, img, pid, camid

    def __len__(self):
        return len(self.dataset)


class ImageData1(Dataset):
    '''
    only for training set
    '''
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img, pid, camid = self.dataset[item]
        img = read_image(img) # 得到RGB图像
        # print('img', img)
        # return 
        # 在此处添加channel exchange函数 问题就出在这里
        ex_img = channel_exchange1(img)
        # ex_img需要转换成PIL image
        # ex_img = img
        if self.transform is not None:
            img = self.transform(img)
            ex_img = self.transform(ex_img)
        return img, ex_img, pid, camid # 返回两种图像（原始图像和随机通道交换图像） 注意：此时为四元组

    def __len__(self):
        return len(self.dataset)


class ImageData2(Dataset):
    '''
    only for training set
    '''
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img, pid, camid = self.dataset[item]
        img = read_image(img) # 得到RGB图像
        # print('img', img)
        # return 
        # 在此处添加channel exchange函数 问题就出在这里
        ex_img = channel_exchange1(img)
        # ex_img需要转换成PIL image
        # ex_img = img
        if self.transform is not None:
            # img = self.transform(img)
            ex_img = self.transform(ex_img)
        return ex_img, ex_img, pid, camid # 返回两种图像（原始图像和随机通道交换图像） 注意：此时为四元组

    def __len__(self):
        return len(self.dataset)
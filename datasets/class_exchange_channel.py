# 封装成类
from PIL import Image
import numpy as np

class exchange_channel(object):
    '''
    '''
    def __init__(self,img_path): #此处是一个img_path还是path列表暂定
        self.img_path = img_path

    def __call__(self, img):# img 格式待定(首先需要确定image具体情况)

        return start_exchange_channel(img)

    def start_exchange_channel(img):
        # img = cv2.imread(img_path)
        # BRG转RBG
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # h,w,c
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
        final_img = Image(final_img.astype('uint8')).convert('RGB')

        return final_img # numpy格式的图像，shape：[128, 64, 3]


def start_exchange_channel(img):
    # img = cv2.imread(img_path)
    # BRG转RBG
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # h,w,c
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
    final_img = Image(final_img.astype('uint8')).convert('RGB')

    return final_img # numpy格式的图像，shape：[128, 64, 3]